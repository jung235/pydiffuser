from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.numpy import linalg as jLA
from jaxlib.xla_extension import PjitFunction

from pydiffuser.typing import ArrayType, ConstType


@partial(jit, static_argnames=("axis"))
def normalize(arr: ArrayType, axis: int = -1) -> Array:
    arr = jnp.asarray(arr)
    denominator = jnp.expand_dims(jLA.norm(arr, axis=axis), axis=axis)
    return arr / denominator


@partial(jit)
def polar_to_cartesian(
    r: ConstType | ArrayType, phi: ConstType | ArrayType
) -> Tuple[Array, Array]:
    x1 = r * jnp.cos(phi)
    x2 = r * jnp.sin(phi)
    return x1, x2


@partial(jit)
def spherical_to_cartesian(
    r: ConstType | ArrayType, theta: ConstType | ArrayType, phi: ConstType | ArrayType
) -> Tuple[Array, Array, Array]:
    """radial, polar, azimuthal (in order)"""

    x1 = r * jnp.sin(theta) * jnp.cos(phi)
    x2 = r * jnp.sin(theta) * jnp.sin(phi)
    x3 = r * jnp.cos(theta)
    return x1, x2, x3


@partial(jit, static_argnames=("generator", "size", "shape"))
def get_noise(
    generator: Callable[[int], Any],
    size: int,
    shape: Tuple[int, ...] | None = None,
) -> Array:
    try:
        noise = jnp.array(generator(size=size))  # type: ignore[call-arg]

    except Exception as exc:
        if (hasattr(generator, "__name__") and "rand" in generator.__name__) or (
            isinstance(generator, partial) and "rand" in generator.func.__name__
        ):
            noise = jnp.array(generator(size))

        elif (
            (hasattr(generator, "__module__") and "jax" in generator.__module__)
            or (isinstance(generator, partial) and "jax" in generator.func.__module__)
            or (
                isinstance(generator, partial)
                and isinstance(generator.func, PjitFunction)
            )
        ):
            noise = generator(shape=(size,))  # type: ignore[call-arg]

        else:
            raise RuntimeError(f"{exc}") from exc
    if shape is not None:
        noise = noise.reshape(shape)
    return noise
