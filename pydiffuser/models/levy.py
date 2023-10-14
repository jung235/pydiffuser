from functools import partial

import jax.numpy as jnp
from jax import Array
from scipy.stats import pareto

from pydiffuser.models.core import (
    ContinuousTimeRandomWalk,
    ContinuousTimeRandomWalkConfig,
)
from pydiffuser.typing import ConstType
from pydiffuser.utils import jitted


class LevyWalkConfig(ContinuousTimeRandomWalkConfig):
    name: str = "levy"

    def __init__(self, speed: float = 1.0, exponent: float = 1.5, **kwargs):
        """_summary_

        Args:
            speed (float): A constant speed.
            exponent (float): The positive scaling exponent `b` of pareto distribution
                in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html.
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
        tau = self.slice(arr=tau.astype(int), threshold=length - 1).T
        return tau
