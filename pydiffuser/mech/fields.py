from functools import partial
from inspect import signature
from typing import Any, Dict

import jax.numpy as jnp
from jax import Array, jit
from jaxlib.xla_extension import PjitFunction

from pydiffuser.typing import ArrayType, ConstType

FIELD_REGISTRY = {}


def register(potential: Any) -> Any:
    def decorator() -> Any:
        if not isinstance(potential, PjitFunction):
            raise RuntimeError(
                "Potential must be transformed via 'jax.jit' to register"
            )
        FIELD_REGISTRY[f"{potential.__name__}"] = potential
        return potential

    return decorator()


def get_static_argsigs(
    potential: Any, static_args_start: int = 1
) -> Dict[str, ConstType]:
    i = static_args_start
    keys = list(signature(potential).parameters.keys())[i:]
    values = list(signature(potential).parameters.values())[i:]
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


def weeks_chandler_andersen_potential():
    pass
