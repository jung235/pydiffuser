from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array, vmap
from jax.random import PRNGKey
from numpy.random import randint
from scipy.stats import pareto

from pydiffuser.models.core import (
    ContinuousTimeRandomWalk,
    ContinuousTimeRandomWalkConfig,
)
from pydiffuser.tracer import Ensemble
from pydiffuser.typing import ConstType
from pydiffuser.utils import jitted


class LevyWalkConfig(ContinuousTimeRandomWalkConfig):
    name: str = "levy"

    def __init__(self, speed: float = 1.0, exponent: float = 1.5, **kwargs):
        """_summary_

        Args:
            speed (float): A constant speed.
            exponent (float): The positive scaling exponent `b` of pareto distribution
                in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html
                or https://jax.readthedocs.io/en/latest/_autosummary/jax.random.pareto.html.
        """

        super(LevyWalkConfig, self).__init__(**kwargs)
        self.speed = speed
        self.exponent = exponent


class LevyWalk(ContinuousTimeRandomWalk):
    name: str = "levy"

    def __init__(self, speed: float, exponent: float):
        super(LevyWalk, self).__init__()
        self.speed = speed
        self.exponent = exponent

    @property
    def one_step(self) -> ConstType:
        return self.speed * self.generate_info["dt"]

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

        generator = partial(jax.random.pareto, b=self.exponent)
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

        num_runs = (
            100 if self.exponent < 1 else 1000 if 1 < self.exponent < 2 else length
        )  # TODO memory allocation error (fatal) by heavy-tailed dist
        tau = jitted.get_noise(
            generator=partial(pareto.rvs, b=self.exponent),
            size=realization * num_runs,
            shape=(-1, realization),
        )
        tau = jnp.array(jnp.round(tau, -int(jnp.log10(dt))) / dt)  # countable
        if jnp.any(jnp.isinf(tau)):
            raise ValueError(
                f"`tau` contains infinite values for exponent={self.exponent}"
            )

        tau = self.slice(arr=tau.astype(int), threshold=length - 1).T
        return tau
