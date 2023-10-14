from typing import List

import jax.numpy as jnp
from jax import Array

from pydiffuser.models.core import OverdampedLangevin, OverdampedLangevinConfig

_GENERATE_HOOKS = ["ornstein_uhlenbeck_process"]


class ActiveOUParticleConfig(OverdampedLangevinConfig):
    name: str = "aoup"

    def __init__(
        self,
        drift_coefficient: float = 0.1,
        diffusion_coefficient: float = 1.0,
        generate_hooks: List[str] = _GENERATE_HOOKS,
        **kwargs,
    ):
        potential = kwargs.pop("potential", None)
        potential_params = kwargs.pop("potential_params", None)
        external_force = kwargs.pop("external_force", 0.0)
        friction_coefficient = kwargs.pop("friction_coefficient", 1.0)
        diffusivity = kwargs.pop("diffusivity", 0.0)

        super(ActiveOUParticleConfig, self).__init__(
            potential=potential,
            potential_params=potential_params,
            external_force=external_force,
            friction_coefficient=friction_coefficient,
            diffusivity=diffusivity,
            generate_hooks=generate_hooks,
            **kwargs,
        )
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient


class ActiveOUParticle(OverdampedLangevin):
    name: str = "aoup"

    def __init__(
        self,
        drift_coefficient: float,
        diffusion_coefficient: float,
        diffusivity: float = 0.0,
        generate_hooks: List[str] = _GENERATE_HOOKS,
        **model_kwargs,
    ):
        """
        Consider an Ornstein-Uhlenbeck process for a self-propulsion velocity p:
            dp              ____
            ── = - μ x p + √2Dou Γ(t),
            dt
        where Γ(t) is a Gaussian white noise with zero mean and unit variance.
        Note that p is coupled with the overdamped Langevin equation
            written in `pydiffuser.models.core.sde.OverdampedLangevin`.

        Args:
            drift_coefficient (float): μ.
            diffusion_coefficient (float): Dou.
        """

        super(ActiveOUParticle, self).__init__(
            diffusivity=diffusivity,
            generate_hooks=generate_hooks,
            **model_kwargs,
        )
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient

    def ornstein_uhlenbeck_process(self) -> Array:
        realization, length, dimension, _ = self.generate_info.values()
        p = jnp.ones((realization, 1, dimension))  # init
        dp = self.get_diff_from_white_noise(
            diffusivity=self.diffusion_coefficient,
            shape=(realization, (length - 2), dimension),
        )
        p = jnp.concatenate((p, dp), axis=1)
        fn = jnp.frompyfunc(self.get_diff_from_drift, 2, 1)
        p = fn.accumulate(p, axis=1)
        return p

    def get_diff_from_drift(self, p: Array, remaining_terms: Array) -> Array:
        return (
            p
            - self.drift_coefficient * p * self.generate_info["dt"]
            + remaining_terms  # constant with respect to `p`
        )
