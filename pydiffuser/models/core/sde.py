import math
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, grad
from scipy.stats import norm

from pydiffuser.exceptions import ShapeMismatchError
from pydiffuser.mech.fields import FIELD_REGISTRY
from pydiffuser.models.core import BaseDiffusion
from pydiffuser.tracer import Ensemble
from pydiffuser.utils import BaseDiffusionConfig, jitted

ShapeType = Tuple[int, int, int]


class OverdampedLangevinConfig(BaseDiffusionConfig):
    name: str = "overdamped"

    def __init__(
        self,
        potential: Optional[str] = None,
        potential_params: Optional[Dict[str, Any]] = None,
        external_force: float = 0.0,
        friction_coefficient: float = 1.0,
        diffusivity: float = 1.0,
        generate_hooks: Optional[List[str]] = None,
        **kwargs,
    ):
        super(OverdampedLangevinConfig, self).__init__(**kwargs)
        self.potential = potential
        self.potential_params = potential_params
        self.external_force = external_force
        self.friction_coefficient = friction_coefficient
        self.diffusivity = diffusivity
        self.generate_hooks = generate_hooks


class OverdampedLangevin(BaseDiffusion):
    name: str = "overdamped"

    def __init__(
        self,
        potential: Optional[str] = None,
        potential_params: Optional[Dict[str, Any]] = None,
        external_force: float = 0.0,
        friction_coefficient: float = 1.0,
        diffusivity: float = 1.0,
        generate_hooks: Optional[List[str]] = None,
    ):
        """
        Consider an overdamped Langevin equation in d dimensions:
            dr     1         1      __
            ── = - ─ ∇U(r) + ─ F + √2D ξ(t) + p,
            dt     γ         γ
        where γ is a friction coefficient,
        D is a translational diffusion coefficient,
        p is a user-defined stochastic variable (e.g., self-propulsion velocity),
        and ξ(t) is a Gaussian white noise with zero mean and unit variance,
        which satisfies <ξ(t)ξ(t')> = d x δ(t-t').

        Args:
            potential (Optional[str], optional): Name of U(r).
            potential_params (Optional[Dict[str, Any]], optional): Parameters of U(r).
            external_force (float, optional): F.
            friction_coefficient (float, optional): γ.
            diffusivity (float, optional): D.
            generate_hooks (Optional[List[str]], optional):
                Additional overdamped Langevin description for p.
        """

        if potential is not None and potential not in FIELD_REGISTRY:
            raise KeyError(f"Unsupported potential {potential} is encountered")
        super(OverdampedLangevin, self).__init__()

        self.potential = potential
        self.potential_params = potential_params
        self.external_force = external_force
        self.friction_coefficient = friction_coefficient
        self.diffusivity = diffusivity
        self.generate_hooks = generate_hooks

    def generate(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        ens = super().generate(realization, length, dimension, dt, **generate_kwargs)
        realization, length, dimension, dt = list(self.generate_info.values())[:4]
        if self.interacting:
            raise RuntimeError("Interacting particles are not supported.")

        x = self._get_initial_position()  # realization x 1 x dimension
        dx = jnp.zeros((realization, (length - 1), dimension))  # init
        dx_shape = dx.shape

        if self.generate_hooks is not None:
            for hook in self.generate_hooks:
                out: Array = self._load_hook(hook)()
                if dx.shape == dx_shape:
                    dx += out
                else:
                    raise ShapeMismatchError((f"{dx_shape} is required"))

        # add terms given in the Langevin eq.
        if self.diffusivity:
            dx += self.get_diff_from_white_noise(self.diffusivity, dx_shape)

        if self.external_force:
            dx += self.get_diff_from_const_force(self.external_force, dx_shape)

        x = jnp.concatenate((x, dx), axis=1)  # realization x length x dimension

        if self.potential is not None:
            fn = jnp.frompyfunc(self.get_diff_from_potential, 2, 1)
            x = fn.accumulate(x, axis=1)
        else:
            x = jnp.cumsum(x, axis=1)
        ens.update_microstate(microstate=x)
        return ens

    def get_diff_from_white_noise(self, diffusivity: float, shape: ShapeType) -> Array:
        assert all(i > 0 for i in shape), (
            "We will calculate `size` from `shape`. "
            "Please use explicit `shape` syntax consisting of positive integers."
        )
        noise = jitted.get_noise(generator=norm.rvs, size=math.prod(shape), shape=shape)
        return noise * jnp.sqrt(2 * diffusivity * self.generate_info["dt"])

    def get_diff_from_const_force(self, const_force: float, shape: ShapeType) -> Array:
        assert all(i > 0 for i in shape), (
            "We will calculate `size` from `shape`. "
            "Please use explicit `shape` syntax consisting of positive integers."
        )
        realization, repeats, dimension = shape
        u = self._get_initial_orientation(realization, dimension)
        u = jnp.repeat(u, repeats=repeats, axis=1)  # realization x repeats x -1

        dr = 1 / self.friction_coefficient * const_force * self.generate_info["dt"]
        if dimension == 1:
            return dr * jnp.cos(u)
        elif dimension == 2:
            return jnp.concatenate(jitted.polar_to_cartesian(r=dr, phi=u), axis=2)
        return jnp.concatenate(
            jitted.spherical_to_cartesian(r=dr, theta=u[:, :, 0], phi=u[:, :, 1]),
            axis=2,
        )

    def get_diff_from_potential(self, r: Array, remaining_terms: Array) -> Array:
        coeff = 1 / self.friction_coefficient
        return (
            r
            - coeff
            * grad(FIELD_REGISTRY[self.potential])(r, **self.potential_params)  # type: ignore[index]
            * self.generate_info["dt"]
            + remaining_terms  # constant with respect to `r`
        )
