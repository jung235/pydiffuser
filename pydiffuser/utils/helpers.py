import functools
from dataclasses import dataclass
from typing import Callable, Optional

from pydiffuser.exceptions import (
    InvalidDimensionError,
    InvalidTimeError,
    ShapeMismatchError,
)
from pydiffuser.typing import LongLongPosType, LongPosType, P, T


@dataclass
class NDAxis:
    N: int | None = 0
    L: int = 1
    D: int = 2


def checkstate(
    ndarray: LongPosType | LongLongPosType, ndaxis: Optional[NDAxis] = None
) -> None:
    if ndarray.ndim == 2:
        default_ndaxis = NDAxis(None, 0, 1)
    elif ndarray.ndim == 3:
        default_ndaxis = NDAxis()
    else:
        raise ShapeMismatchError(f"Invalid shape {ndarray.shape} is encountered")
    if ndaxis is not None:
        if ndaxis != default_ndaxis:
            raise ShapeMismatchError(f"Invalid shape {ndarray.shape} is encountered")
    else:
        ndaxis = default_ndaxis
    if not 1 <= ndarray.shape[ndaxis.D] <= 3:
        raise InvalidDimensionError(
            f"Invalid spatial dimension {ndarray.shape[ndaxis.D]} is encountered"
        )
    return


def checktime(nonzero: bool = False) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            lagtime: int = args[1] if len(args) > 1 else kwargs["lagtime"]  # type: ignore[assignment]
            if nonzero and lagtime <= 0:
                raise InvalidTimeError(
                    f"Only positive integers are allowed for lagtime in {fn.__name__}"
                )
            if lagtime < 0:
                raise InvalidTimeError(f"Negative time is encountered in {fn.__name__}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator
