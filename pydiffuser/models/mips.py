import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array, grad, lax, vmap
from numpy.random import rand

from pydiffuser.mech.fields import FIELD_REGISTRY, get_static_argsigs, wca_potential
from pydiffuser.models.aoup import ActiveOUParticle, ActiveOUParticleConfig
from pydiffuser.tracer import Ensemble
from pydiffuser.typing import LongLongPosType, LongPosType, PosType
from pydiffuser.utils import jitted

DEFAULT_POTENTIAL = wca_potential.__name__
DEFAULT_POTENTIAL_PARAMS = get_static_argsigs(wca_potential, 3, 4)


class PhaseSeparationConfig(ActiveOUParticleConfig):
    name: str = "mips"

    def __init__(
        self,
        boxsize: float = 100.0,
        mean_diameter: float = 2.0,
        potential: str = DEFAULT_POTENTIAL,
        potential_params: Dict[str, Any] = DEFAULT_POTENTIAL_PARAMS,
        diffusivity: float = 1.0,
        drift_coefficient: float = 1.0,
        diffusion_coefficient: float = 0.1,
        **kwargs,
    ):
        kwargs = {
            param: kwargs[param] if param in kwargs else default
            for param, default in ActiveOUParticleConfig.show_fields().items()
        }
        kwargs["potential"] = potential
        kwargs["potential_params"] = potential_params
        kwargs["diffusivity"] = diffusivity
        kwargs["drift_coefficient"] = drift_coefficient
        kwargs["diffusion_coefficient"] = diffusion_coefficient
        kwargs["model_alias"] = PhaseSeparationConfig.name
        super(PhaseSeparationConfig, self).__init__(**kwargs)

        self.boxsize = boxsize
        self.mean_diameter = mean_diameter


class PhaseSeparation(ActiveOUParticle):
    name: str = "mips"

    def __init__(
        self,
        boxsize: float,
        mean_diameter: float,
        potential: str,
        potential_params: Dict[str, Any],
        diffusivity: float,
        drift_coefficient: float,
        diffusion_coefficient: float,
        **model_kwargs,
    ):
        """
        We consider AOUPs interacting via the Weeks-Chandler-Andersen (WCA) potential
        in a square box of unit cell size L, subjected to a periodic boundary condition (PBC).
        Note that the particles can exhibit motility-induced phase separation (MIPS).
        The following equation governs the ith particle:
        ```
            drᵢ     1              1      __
            ─── = - ─ Σ ∇ᵢU(rᵢⱼ) + ─ F + √2D ξᵢ(t) + pᵢ,
            dt      γ j            γ
        ```
        where rᵢⱼ = |rⱼ - rᵢ| is the magnitude of the relative position vector.
        Here, U is the purely repulsive WCA potential defined in `pydiffuser.mech.fields`.
        For a detailed description of pᵢ, see `pydiffuser.models.aoup.ActiveOUParticle`.

        Args:
            boxsize (float): L.
            mean_diameter (float): The mean diameter of all particles.
        """

        super(PhaseSeparation, self).__init__(
            potential=potential,
            potential_params=potential_params,
            diffusivity=diffusivity,
            drift_coefficient=drift_coefficient,
            diffusion_coefficient=diffusion_coefficient,
            **model_kwargs,
        )
        self.interacting = True
        self.precision_x64 = True

        self.boxsize = boxsize
        self.mean_diameter = mean_diameter

    def generate(
        self,
        realization: int = 50,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 0.01,
        **generate_kwargs,
    ) -> Ensemble:
        self.pre_generate(realization, length, dimension, dt, *generate_kwargs.values())
        realization, length, dimension, dt = list(self.generate_info.values())[:4]
        ens = Ensemble(dt=dt)

        hook = self.generate_hooks[-1]  # type: ignore[index]
        assert (
            hook == self.ornstein_uhlenbeck_process.__name__
            and self.potential == DEFAULT_POTENTIAL
        ), "Only AOUPs imposed on the WCA potential are supported"

        sigma = jitted.get_noise(rand, size=realization) * self.mean_diameter * 2
        x = self._get_initial_position()  # realization x 1 x dimension
        dx: Array = self._load_hook(hook)()  # realization x (length - 1) x dimension
        dx_shape = dx.shape

        # add terms given in the Langevin eq.
        if self.diffusivity:
            dx += self.get_diff_from_white_noise(self.diffusivity, dx_shape)

        if self.external_force:
            dx += self.get_diff_from_const_force(self.external_force, dx_shape)

        _, stx = lax.scan(
            f=partial(self.get_next_position_from_pairwise_potential, sigma=sigma),
            init=jnp.squeeze(x, axis=1),  # realization x dimension
            xs=jnp.transpose(dx, (1, 0, 2)),  # (length - 1) x realization x dimension
        )
        x = jnp.concatenate((x, jnp.transpose(stx, (1, 0, 2))), axis=1)

        ens.update_microstate(microstate=x)
        ens.update_meta_dict(item={"diameter": sigma})
        return ens

    def get_next_position_from_pairwise_potential(
        self, x: LongPosType, dx: LongPosType, sigma: PosType
    ) -> Tuple[LongPosType, LongPosType]:
        dt = self.generate_info["dt"]
        potential_fn = partial(
            FIELD_REGISTRY[self.potential],
            boxsize=self.boxsize,  # PBC
            **self.potential_params,
        )
        sij = (sigma[:, jnp.newaxis] + sigma[jnp.newaxis, :]) / 2

        vmap_fn = vmap(
            vmap(grad(potential_fn, argnums=0), in_axes=(None, 0, 0)),
            in_axes=(0, None, 0),
        )
        del_fn = vmap_fn(x, x, sij)  # realization x realization x dimension
        del_fn = jnp.nan_to_num(del_fn, nan=0.0)

        # sum over all j to calculate Σⱼ∇ᵢU(rᵢⱼ)
        sum_del_fn = jnp.sum(del_fn, axis=1)  # realization x dimension
        dx_wca = -1 / self.friction_coefficient * sum_del_fn * dt

        next_x = (x + dx + dx_wca) % self.boxsize  # PBC
        return next_x, next_x

    def _get_initial_position(
        self, realization: Optional[int] = None, dimension: Optional[int] = None
    ) -> LongLongPosType:
        if realization is None or dimension is None:
            realization, _, dimension = list(self.generate_info.values())[:3]
        shape = (realization, 1, dimension)
        x = jitted.get_noise(generator=rand, size=math.prod(shape), shape=shape)  # type: ignore[arg-type]
        return x * self.boxsize  # PBC
