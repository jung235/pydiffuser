import inspect
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.numpy import linalg as jLA

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
    shape: Optional[Tuple[int, ...]] = None,
) -> Array:
    try:
        noise = generator(size=size)  # type: ignore[call-arg]
    except Exception as exc:
        if isinstance(exc, TypeError):
            module = inspect.getmodule(generator)
            if module is None:
                noise = generator(size)
            elif "jax" in module.__name__:
                raise NotImplementedError(
                    "Random number generator via JAX is unsupported"
                ) from exc
        else:
            raise RuntimeError(f"{exc}") from exc
    noise = jnp.array(noise)
    if shape is not None:
        noise = noise.reshape(shape)
    return noise
