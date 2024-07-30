import math
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import lax
from numpy.random import rand

from pydiffuser.exceptions import InvalidDimensionError
from pydiffuser.models.core import OverdampedLangevin, OverdampedLangevinConfig
from pydiffuser.tracer import Ensemble
from pydiffuser.typing import LongLongPosType, LongPosType
from pydiffuser.utils import jitted


class VicsekModelConfig(OverdampedLangevinConfig):
    name: str = "vicsek"

    def __init__(
        self,
        boxsize: float = 40.0,
        interaction_strength: float = 1.0,
        interaction_radius: float = 2.0,
        diffusivity: float = 0.0,
        speed: float = 1.0,
        rotational_diffusivity: float = 0.1,
        angular_velocity: float = 0.0,
        **kwargs,
    ):
        kwargs = {
            param: kwargs[param] if param in kwargs else default
            for param, default in OverdampedLangevinConfig.show_fields().items()
        }
        kwargs["diffusivity"] = diffusivity
        kwargs["model_alias"] = VicsekModelConfig.name
        super(VicsekModelConfig, self).__init__(**kwargs)

        self.boxsize = boxsize
        self.interaction_strength = interaction_strength
        self.interaction_radius = interaction_radius
        self.speed = speed
        self.rotational_diffusivity = rotational_diffusivity
        self.angular_velocity = angular_velocity


class VicsekModel(OverdampedLangevin):
    name: str = "vicsek"

    def __init__(
        self,
        boxsize: float,
        interaction_strength: float,
        interaction_radius: float,
        speed: float,
        rotational_diffusivity: float,
        angular_velocity: float,
        **model_kwargs,
    ):
        """
        We consider the Vicsek model utilizing active Brownian particles (ABPs)
        in a square box of size L x L, subjected to a periodic boundary condition (PBC).
        The following equation governs the velocity direction φᵢ of the ith particle:
        ```
            dφᵢ        K                    ___
            ─── = ω + ─── Σ sin(φⱼ - φᵢ) + √2Dr ηᵢ(t).
            dt        πR² j
        ```
        Here, we ignore the excluded volume effect and external force term, which means
            U = 0 and F = 0 in `pydiffuser.models.core.sde.OverdampedLangevin`.
        For a detailed description of ABPs, see `pydiffuser.models.abp.ActiveBrownianParticle`.

        Args:
            boxsize (float): L.
            interaction_strength (float): K.
            interaction_radius (float): R.
        """

        super(VicsekModel, self).__init__(**model_kwargs)
        self.interacting = True

        self.boxsize = boxsize
        self.interaction_strength = interaction_strength
        self.interaction_radius = interaction_radius
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

    def generate(
        self,
        realization: int = 1000,
        length: int = 100,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        self.pre_generate(realization, length, dimension, dt, *generate_kwargs.values())
        realization, length, dimension, dt = list(self.generate_info.values())[:4]
        ens = Ensemble(dt=dt)

        phi = self._get_initial_orientation()  # realization x 1 x 1
        x = self._get_initial_position()  # realization x 1 x 2
        state = jnp.concatenate((phi, x), axis=-1)  # realization x 1 x 3

        dphi = self.get_diff_from_white_noise(
            self.rotational_diffusivity, shape=(realization, (length - 1), 1)
        )
        dphi += self.angular_velocity * dt
        dx = self.get_diff_from_white_noise(
            self.diffusivity, shape=(realization, (length - 1), dimension)
        )
        dstate = jnp.concatenate((dphi, dx), axis=-1)  # realization x (length - 1) x 3

        _, stx = lax.scan(
            f=self.get_next_state_from_vicsek_interaction,
            init=jnp.squeeze(state, axis=1),  # realization x 3
            xs=jnp.transpose(dstate, (1, 0, 2)),  # (length - 1) x realization x 3
        )
        stx_phi, stx_x = jnp.split(stx, indices_or_sections=[1], axis=-1)
        phi = jnp.concatenate((phi, jnp.transpose(stx_phi, (1, 0, 2))), axis=1)
        x = jnp.concatenate((x, jnp.transpose(stx_x, (1, 0, 2))), axis=1)

        ens.update_microstate(microstate=x)
        ens.update_meta_dict(item={"direction": phi})
        return ens

    def get_next_state_from_vicsek_interaction(
        self,
        state: LongPosType,
        dstate: LongPosType,
    ) -> Tuple[LongPosType, LongPosType]:
        dt = self.generate_info["dt"]
        phi, x = jnp.split(state, indices_or_sections=[1], axis=-1)
        dphi, dx = jnp.split(dstate, indices_or_sections=[1], axis=-1)
        coeff = self.interaction_strength / (jnp.pi * self.interaction_radius**2)

        dx += jnp.concatenate(
            arrays=jitted.polar_to_cartesian(r=self.speed * dt, phi=phi),
            axis=1,
        )
        next_x = (x + dx) % self.boxsize  # PBC

        dx_vec = x[:, jnp.newaxis] - x[jnp.newaxis, :]
        dx_vec = dx_vec - self.boxsize * jnp.round(dx_vec / self.boxsize)  # PBC
        dr = jnp.linalg.norm(dx_vec, axis=-1)

        mask = jnp.where(dr <= self.interaction_radius, 1, 0)
        sine = jnp.sin(phi.T - phi)
        dphi_vicsek = coeff * jnp.sum(sine * mask, axis=-1) * dt
        next_phi = phi + dphi + dphi_vicsek[:, jnp.newaxis]

        next_state = jnp.concatenate((next_phi, next_x), axis=-1)  # realization x 3
        return next_state, next_state

    def _get_initial_position(
        self, realization: Optional[int] = None, dimension: Optional[int] = None
    ) -> LongLongPosType:
        if realization is None or dimension is None:
            realization, _, dimension = list(self.generate_info.values())[:3]
        shape = (realization, 1, dimension)
        x = jitted.get_noise(generator=rand, size=math.prod(shape), shape=shape)  # type: ignore[arg-type]
        return x * self.boxsize  # PBC
