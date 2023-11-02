from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.numpy import linalg as jLA

from pydiffuser.exceptions import (
    InvalidDimensionError,
    InvalidTimeError,
    ShapeMismatchError,
)
from pydiffuser.logger import logger
from pydiffuser.mech.constants import Axis
from pydiffuser.tracer.component import Component
from pydiffuser.tracer.trajectory import Trajectory
from pydiffuser.typing import ConstType, LongLongPosType, LongPosType, PathType
from pydiffuser.utils import helpers, jitted

NDAXIS = helpers.NDAxis()


class Ensemble(Component, OrderedDict):  # type: ignore[type-arg]
    def __init__(self, dt: ConstType) -> None:
        """_summary_

        Args:
            dt (ConstType): Real time interval between consecutive data points.
        """

        super().__init__(dt=dt)
        self.microstate = jnp.zeros((0, 0, 0))

    @property
    def N(self) -> int:
        """Number of particles"""

        return int(self.microstate.shape[NDAXIS.N])

    @property
    def length(self) -> int:
        """Number of footprints"""

        return int(self.microstate.shape[NDAXIS.L])

    @property
    def dimension(self) -> int:
        """Spatial dimension"""

        return int(self.microstate.shape[NDAXIS.D])

    def is_composite(self) -> bool:
        return True

    def update_microstate(
        self, microstate: LongLongPosType, lazy_dict: bool = True
    ) -> None:
        helpers.checkstate(ndarray=microstate, ndaxis=NDAXIS)
        if not self.length:
            self.microstate = microstate
        else:
            self.microstate = jnp.concatenate(
                (self.microstate, microstate), axis=NDAXIS.N
            )
        if not lazy_dict:
            for id in range(self.N - len(microstate), self.N):
                self[id] = Trajectory(self.dt, *microstate[id].T)
        return

    def update(self, *args, **kwargs):
        raise NotImplementedError("")

    def setdefault(self, *args, **kwargs):
        raise NotImplementedError("")

    def pop(self, *args, **kwargs):
        raise NotImplementedError("")

    def popitem(self, *args, **kwargs):
        raise NotImplementedError("")

    def __delitem__(self, *args, **kwargs):
        raise NotImplementedError("")

    def __setitem__(self, id: int, tracer: Trajectory) -> None:
        if not isinstance(tracer, Trajectory):
            raise ValueError("Only `Trajectory` can be an element of `Ensemble`")
        assert tracer.dt == self.dt
        if id >= self.N:
            if id > self.N:
                logger.warning(
                    f"Tight key is forced. Given {id} will be replaced with {self.N}. "
                    "Use `update_meta_dict` instead to assign extra id.",
                    stacklevel=2,
                )
                id = self.N
            self.update_microstate(
                microstate=jnp.expand_dims(tracer.position, axis=NDAXIS.N)
            )
        else:
            is_same_shape = jnp.all(tracer.position.shape == self.microstate[id].shape)
            if not is_same_shape:
                raise ShapeMismatchError(f"{self[id].__repr__()} is required")
            is_same_position = jnp.all(tracer.position == self.microstate[id])
            if is_same_shape and not is_same_position:
                raise RuntimeError("Overriding is not supported")  # TODO microstate
        return super().__setitem__(id, tracer)  # opt: sorting for items()

    def __getitem__(self, id: int) -> Trajectory:
        if not isinstance(id, int):
            raise KeyError("Only integers are allowed for key")
        if id in self.keys():
            tracer: Trajectory = super().__getitem__(id)
            return tracer
        try:
            position: LongPosType = self.microstate[id]
            self.__setitem__(id, Trajectory(self.dt, *position.T))
            return super().__getitem__(id)  # type: ignore[no-any-return]
        except Exception as exc:
            if isinstance(exc, IndexError):
                raise KeyError(f"{exc}") from None
            else:
                raise RuntimeError(f"{exc}") from exc

    def __str__(self) -> str:
        info = f"N={self.N}, L={self.length}, D={self.dimension}, dt={self.dt}"
        return f"{self.__class__.__name__}({info})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.microstate.shape}"

    def add(self, tracer: Trajectory) -> None:
        self[self.N] = tracer

    @classmethod
    def from_npy(
        cls, npy_path: PathType, dt: float, lazy_dict: bool = True
    ) -> Ensemble:
        ens = cls(dt)
        microstate: LongLongPosType = jnp.load(npy_path)
        ens.update_microstate(microstate, lazy_dict)
        return ens

    @helpers.checktime(nonzero=True)
    def get_increments(
        self, lagtime: int, rolling: bool = True, axis: Optional[str] = None
    ) -> List[float]:
        dx = _get_increments(self.microstate, lagtime, rolling, axis)
        return dx.tolist()  # type: ignore[no-any-return]

    @helpers.checktime(nonzero=True)
    def get_displacement_moment(
        self,
        lagtime: int,
        order: int = 2,
        rolling: bool = True,
        axis: Optional[str] = None,
    ) -> float:
        m = _get_displacement_moment(self.microstate, lagtime, order, rolling, axis)
        return float(m)

    @helpers.checktime()
    def get_cosine_moment(
        self, lagtime: int, order: int = 1, rolling: bool = True, epsilon: int = 1
    ) -> float:
        if self.dimension != 2:
            raise InvalidDimensionError(
                f"Unsupported dimension {self.dimension} is encountered"
            )
        if self.length - lagtime - epsilon <= 0 or epsilon <= 0:
            raise InvalidTimeError("Only positive integers are allowed for lagtime")
        m = _get_cosine_moment(self.microstate, lagtime, order, rolling, epsilon)
        return float(m)

    @helpers.checktime()
    def get_velocity_autocorrelation(
        self, lagtime: int, rolling: bool = True, epsilon: int = 1
    ) -> float:
        if self.dimension != 2:
            raise InvalidDimensionError(
                f"Unsupported dimension {self.dimension} is encountered"
            )
        if self.length - lagtime - epsilon <= 0 or epsilon <= 0:
            raise InvalidTimeError("Only positive integers are allowed for lagtime")
        m = _get_velocity_autocorrelation(
            self.microstate, lagtime, self.dt, rolling, epsilon
        )
        return float(m)

    def __reduce__(self) -> Tuple[Any, ...]:
        info = super().__reduce__()
        args = self.dt
        new_info = (self.__class__, args, *info[2:])
        return new_info

    def to_npy(self, npy_path: PathType) -> None:
        jnp.save(file=npy_path, arr=self.microstate)


@partial(jit, static_argnums=(1, 2, 3))
def _get_increments(
    x: LongLongPosType, lagtime: int, rolling: bool = True, axis: Optional[str] = None
) -> Array:
    x = jnp.asarray(x)
    i = getattr(Axis, axis) if axis else None
    dx = jnp.roll(x, shift=-lagtime, axis=NDAXIS.L) - x
    if not rolling:
        dx = dx[:, 0, i] if axis else jLA.norm(dx[:, 0], axis=-1)
    else:
        dx = dx[:, :-lagtime, i] if axis else jLA.norm(dx[:, :-lagtime], axis=NDAXIS.D)
    return dx.flatten()


@partial(jit, static_argnums=(1, 2, 3, 4))
def _get_displacement_moment(
    x: LongLongPosType,
    lagtime: int,
    order: int = 2,
    rolling: bool = True,
    axis: Optional[str] = None,
) -> Array:
    x = jnp.asarray(x)
    i = getattr(Axis, axis) if axis else None
    m = (jnp.roll(x, shift=-lagtime, axis=NDAXIS.L) - x) ** order
    if not rolling:
        m = m[:, 0, i] if axis else jnp.sum(m[:, 0], axis=-1)
        return jnp.mean(m)
    m = m[:, :-lagtime, i] if axis else jnp.sum(m[:, :-lagtime], axis=NDAXIS.D)
    return jnp.mean(jnp.mean(m, axis=-1))  # TA -> EA


@partial(jit, static_argnums=(1, 2, 3, 4))
def _get_cosine_moment(
    x: LongLongPosType,
    lagtime: int,
    order: int = 1,
    rolling: bool = True,
    epsilon: int = 1,
) -> Array:
    def _fix(_cos: Array) -> Array:
        _cos = jnp.where(_cos > 1, 1, _cos)  # fix cos > 1
        _cos = jnp.where(_cos < -1, -1, _cos)  # fix cos < -1
        return _cos

    x = jnp.asarray(x)
    vel = jnp.roll(x, shift=-epsilon, axis=NDAXIS.L) - x
    vel = jitted.normalize(vel[:, :-epsilon], axis=NDAXIS.D)
    if not rolling:
        cos = jnp.sum(vel[:, lagtime] * vel[:, 0], axis=-1)
        return jnp.mean(jnp.cos(order * jnp.arccos(_fix(cos))))
    mul = jnp.roll(vel, shift=-lagtime, axis=NDAXIS.L) * vel
    cos = (
        jnp.sum(mul, axis=NDAXIS.D)
        if not lagtime
        else jnp.sum(mul[:, :-lagtime], axis=NDAXIS.D)
    )
    return jnp.mean(
        jnp.mean(jnp.cos(order * jnp.arccos(_fix(cos))), axis=-1)
    )  # TA -> EA


@partial(jit, static_argnums=(1, 2, 3, 4))
def _get_velocity_autocorrelation(
    x: LongLongPosType,
    lagtime: int,
    dt: ConstType,
    rolling: bool = True,
    epsilon: int = 1,
) -> Array:
    x = jnp.asarray(x)
    vel = jnp.roll(x, shift=-epsilon, axis=NDAXIS.L) - x
    vel = vel[:, :-epsilon] / (epsilon * dt)
    if not rolling:
        dot = jnp.sum(vel[:, lagtime] * vel[:, 0], axis=-1)
        return jnp.mean(dot)
    mul = jnp.roll(vel, shift=-lagtime, axis=NDAXIS.L) * vel
    dot = (
        jnp.sum(mul, axis=NDAXIS.D)
        if not lagtime
        else jnp.sum(mul[:, :-lagtime], axis=NDAXIS.D)
    )
    return jnp.mean(jnp.mean(dot, axis=-1))  # TA -> EA
