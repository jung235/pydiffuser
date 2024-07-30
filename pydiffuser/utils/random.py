from collections.abc import Sequence
from functools import partial

from jax import Array, jit, lax, random
from jax._src.typing import ArrayLike, DTypeLike


@partial(jit, static_argnums=(2, 3))
def exponential(
    key: ArrayLike,
    scale: ArrayLike,
    shape: Sequence[int] = (),
    dtype: DTypeLike = float,
) -> Array:
    scale = lax.convert_element_type(scale, dtype)
    return lax.mul(random.exponential(key, shape, dtype), scale)
