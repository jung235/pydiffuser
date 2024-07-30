from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array, vmap
from jax.random import PRNGKey
from numpy.random import randint
from scipy.stats import expon

from pydiffuser.logger import logger
from pydiffuser.models.core import (
    ContinuousTimeRandomWalk,
    ContinuousTimeRandomWalkConfig,
)
from pydiffuser.tracer import Ensemble
from pydiffuser.typing import ConstType
from pydiffuser.utils import jitted, random


class RunAndTumbleParticleConfig(ContinuousTimeRandomWalkConfig):
    name: str = "rtp"

    def __init__(self, speed: float = 1.0, rate: float = 0.1, **kwargs):
        super(RunAndTumbleParticleConfig, self).__init__(**kwargs)
        self.speed = speed
        self.rate = rate


class RunAndTumbleParticle(ContinuousTimeRandomWalk):
    name: str = "rtp"

    def __init__(self, speed: float, rate: float):
        super(RunAndTumbleParticle, self).__init__()
        self.speed = speed
        self.rate = rate

    @property
    def one_step(self) -> ConstType:
        return self.speed * self.generate_info["dt"]

    @property
    def tc(self) -> float:
        """Characteristic timescale, the mean of exponential distribution"""

        return 1 / self.rate

    def pre_generate(self, *generate_args) -> None:
        super().pre_generate(*generate_args)
        dt = self.generate_info["dt"]
        if dt >= self.tc:
            logger.warning("The precision is significantly low.", stacklevel=2)
        return

    def generate(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        ens = super().generate(realization, length, dimension, dt, **generate_kwargs)
        realization, length, dimension, _ = list(self.generate_info.values())[:4]
        interarrival: Dict[int, Array] = ens.meta["interarrival"]

        u = self.get_orientation_steps(realization, capacity=length)
        u = self.repeat(
            u, interarrival, cutoff=length - 1  # realization x (length - 1) x -1
        )
        x = self._get_microstate_from_orientation(u)
        ens.update_microstate(microstate=x)
        return ens

    def get_interarrival_times(self) -> Dict[int, Array]:
        realization, length, _, _ = self.generate_info.values()

        generator = partial(random.exponential, scale=self.tc)
        keys = jax.random.split(PRNGKey(randint(0, 1e9)), num=realization)
        vmap_fn = vmap(self._get_interarrival_times_per_tracer, (None, 0))
        taus, cutoffs = vmap_fn(generator, keys)

        interarrival = {}
        for id, (tau, cutoff) in enumerate(zip(taus, cutoffs, strict=True)):
            if cutoff > 0:
                arr = tau[:cutoff]
                interarrival[id] = jnp.append(arr, length - jnp.sum(arr))
            else:
                interarrival[id] = jnp.array([length], dtype=int)
        return interarrival

    def get_time_steps(self) -> Array:
        realization, length, _, dt = self.generate_info.values()

        num_runs = int(length / self.tc * 2) if self.rate < 1 else length
        tau = jitted.get_noise(
            generator=partial(expon.rvs, scale=self.tc),
            size=realization * num_runs,
            shape=(-1, realization),
        )
        tau = jnp.array(jnp.round(tau, -int(jnp.log10(dt))) / dt)  # countable
        tau = self.slice(arr=tau.astype(int), threshold=length - 1).T
        return tau
