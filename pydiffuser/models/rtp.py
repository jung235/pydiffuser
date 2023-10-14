from functools import partial

import jax.numpy as jnp
from jax import Array
from scipy.stats import expon

from pydiffuser.logger import logger
from pydiffuser.models.core import (
    ContinuousTimeRandomWalk,
    ContinuousTimeRandomWalkConfig,
)
from pydiffuser.typing import ConstType
from pydiffuser.utils import jitted


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

    def get_time_steps(self) -> Array:
        realization, length, _, dt = self.generate_info.values()

        num_runs = int(length / self.tc * 2) if self.rate < 1 else length  # TODO
        tau = jitted.get_noise(
            generator=partial(expon.rvs, scale=self.tc),
            size=realization * num_runs,
            shape=(-1, realization),
        )
        tau = jnp.array(jnp.round(tau, -int(jnp.log10(dt))) / dt)  # countable
        tau = self.slice(arr=tau.astype(int), threshold=length - 1).T
        return tau
