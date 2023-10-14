from pathlib import Path
from typing import ParamSpec, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

P = ParamSpec("P")
T = TypeVar("T")

ArrayType = NDArray | Array

# TODO PEP 646
PosType = ArrayType | Sequence  # 1darray
LongPosType = ArrayType  # 2darray
LongLongPosType = ArrayType  # 3darray

ConstType = (
    int | float | np.int32 | np.float32 | np.int64 | np.float64
    | jnp.int32 | jnp.float32 | jnp.int64 | jnp.float64  # fmt: skip
)

PathType = str | Path
