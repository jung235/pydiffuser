from typing import List

import jax.numpy as jnp
from jax import Array

from pydiffuser.exceptions import InvalidDimensionError
from pydiffuser.models.core import OverdampedLangevin, OverdampedLangevinConfig
from pydiffuser.utils import jitted

_GENERATE_HOOKS = ["rotational_diffusion"]


class ActiveBrownianParticleConfig(OverdampedLangevinConfig):
    name: str = "abp"

    def __init__(
        self,
        speed: float = 1.0,
        rotational_diffusivity: float = 0.1,
        angular_velocity: float = 0.0,
        diffusivity: float = 0.0,
        generate_hooks: List[str] = _GENERATE_HOOKS,
        **kwargs,
    ):
        potential = kwargs.pop("potential", None)
        potential_params = kwargs.pop("potential_params", None)
        external_force = kwargs.pop("external_force", 0.0)
        friction_coefficient = kwargs.pop("friction_coefficient", 1.0)

        super(ActiveBrownianParticleConfig, self).__init__(
            potential=potential,
            potential_params=potential_params,
            external_force=external_force,
            friction_coefficient=friction_coefficient,
            diffusivity=diffusivity,
            generate_hooks=generate_hooks,
            **kwargs,
        )
        self.speed = speed
        self.rotational_diffusivity = rotational_diffusivity
        self.angular_velocity = angular_velocity


class ActiveBrownianParticle(OverdampedLangevin):
    name: str = "abp"

    def __init__(
        self,
        speed: float,
        rotational_diffusivity: float,
        angular_velocity: float = 0.0,
        diffusivity: float = 0.0,
        generate_hooks: List[str] = _GENERATE_HOOKS,
        **model_kwargs,
    ):
        """
        Consider an overdamped Langevin equation for orientation φ:
            dφ        ___
            ── = ω + √2Dr η(t),
            dt
        where ω is a constant angular velocity,
        Dr is a rotational diffusion coefficient,
        and η(t) denotes a Gaussian white noise with zero mean and unit variance.
        Following the equation in `pydiffuser.models.core.sde.OverdampedLangevin`,
            we define a self-propulsion velocity p := v0 x (cosφ, sinφ) on a plane.

        Args:
            speed (float): A constant speed v0.
            rotational_diffusivity (float): Dr.
            angular_velocity (float, optional): A constant angular velocity ω.
                When ω > 0, chiral active particles (CAPs) are generated.
        """

        super(ActiveBrownianParticle, self).__init__(
            diffusivity=diffusivity,
            generate_hooks=generate_hooks,
            **model_kwargs,
        )
        self.speed = speed
        self.rotational_diffusivity = rotational_diffusivity
        self.angular_velocity = angular_velocity

    def pre_generate(self, *generate_args) -> None:
        super().pre_generate(*generate_args)
        dimension = self.generate_info["dimension"]
        if dimension != 2:
            raise InvalidDimensionError(
                f"Unsupported dimension {dimension} is encountered"
            )
        return

    def rotational_diffusion(self) -> Array:
        realization, length, _, dt = self.generate_info.values()

        phi = self._get_initial_orientation()
        dphi = self.get_diff_from_white_noise(
            diffusivity=self.rotational_diffusivity,
            shape=(realization, (length - 2), 1),
        )
        dphi += self.angular_velocity * dt
        phi = jnp.concatenate((phi, dphi), axis=1)
        phi = jnp.cumsum(phi, axis=1)  # realization x (length - 1) x 1
        dx = jnp.concatenate(
            arrays=jitted.polar_to_cartesian(r=self.speed * dt, phi=phi), axis=2
        )
        return dx  # realization x (length - 1) x 2
