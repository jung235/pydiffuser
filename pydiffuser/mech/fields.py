from functools import partial
from inspect import signature
from typing import Any, Dict

import jax.numpy as jnp
from jax import Array, jit
from jaxlib.xla_extension import PjitFunction

from pydiffuser.typing import ArrayType, ConstType, PosType

FIELD_REGISTRY = {}


def register(potential: Any) -> Any:
    def decorator() -> Any:
        if not isinstance(potential, PjitFunction):
            raise RuntimeError(
                "Potential must be transformed via `jax.jit` to register"
            )
        FIELD_REGISTRY[f"{potential.__name__}"] = potential
        return potential

    return decorator()


def get_static_argsigs(
    potential: Any, static_args_start: int = 1, static_args_end: int | None = None
) -> Dict[str, ConstType]:
    sigs = signature(potential).parameters
    i = static_args_start if static_args_start else 0
    j = static_args_end if static_args_end else len(sigs)

    keys = list(sigs.keys())[i:j]
    values = list(sigs.values())[i:j]
    return {k: v.default for k, v in zip(keys, values, strict=True)}


@register
@partial(jit, static_argnames=("spring_const"))
def harmonic_potential(
    x: ConstType | ArrayType, spring_const: ConstType = 1.0
) -> Array:
    return 1 / 2 * spring_const * jnp.square(x)


@register
@partial(jit, static_argnames=("magnitude", "period"))
def periodic_potential(
    x: ConstType | ArrayType, magnitude: ConstType = 1.0, period: ConstType = 10.0
) -> Array:
    return magnitude * jnp.cos(2 * jnp.pi / period * x)


def lennard_jones_potential():
    pass


@register
@partial(jit, static_argnames=("epsilon", "boxsize"))
def wca_potential(
    ri: PosType,
    rj: PosType,
    sigma: ConstType,
    epsilon: ConstType = 10.0,
    boxsize: ConstType | None = None,
) -> Array:
    """The Weeks-Chandler-Andersen potential is given as
    ```
                     σᵢⱼ       σᵢⱼ     1
        U(rᵢⱼ) = 4ε[(───)¹² - (───)⁶ + ─]
                     rᵢⱼ       rᵢⱼ     4
    ```
    for rᵢⱼ < 2¹ᐟ⁶σᵢⱼ and U(rᵢⱼ) = 0 otherwise.
    Here, we have rᵢⱼ = |rᵢ - rⱼ| and σᵢⱼ = (σᵢ + σⱼ) / 2,
    where σᵢ represents the diameter of the ith particle.

    Args:
        ri (PosType): rᵢ.
        rj (PosType): rⱼ.
        sigma (ConstType): σᵢⱼ.
        epsilon (ConstType, optional): ε.
        boxsize (ConstType | None, optional): Size of the unit cell.
            If not None, periodic boundary conditions (PBCs) are enforced.
    """

    rij_vec = ri - rj
    if boxsize:
        rij_vec = rij_vec - boxsize * jnp.round(rij_vec / boxsize)
    rij = jnp.linalg.norm(rij_vec)

    return jnp.where(
        rij < 2 ** (1 / 6) * sigma,
        4 * epsilon * ((sigma / rij) ** 12 - (sigma / rij) ** 6 + 1 / 4),
        0,
    )
