from __future__ import annotations

from functools import partial
from typing import List, Optional

import jax.numpy as jnp
from jax import Array, jit
from jax.numpy import linalg as jLA

from pydiffuser.exceptions import InvalidDimensionError, InvalidTimeError
from pydiffuser.mech.constants import Axis
from pydiffuser.tracer.component import Component
from pydiffuser.typing import ConstType, LongPosType, PathType, PosType
from pydiffuser.utils import helpers, jitted

NDAXIS = helpers.NDAxis(N=None, L=0, D=1)


class Trajectory(Component):
    def __init__(
        self,
        dt: ConstType,
        position_x1: PosType,
        position_x2: Optional[PosType] = None,
        position_x3: Optional[PosType] = None,
    ) -> None:
        """_summary_

        Args:
            dt (ConstType): Real time interval between consecutive data points.
            position_x1 (PosType): _description_.
            position_x2 (Optional[PosType], optional): _description_.
            position_x3 (Optional[PosType], optional): _description_.
        """

        super(Trajectory, self).__init__(dt=dt)
        self.position_x1 = position_x1
        self.position_x2 = position_x2
        self.position_x3 = position_x3

        poses = (position_x1, position_x2, position_x3)
        r = jnp.vstack(jnp.array([list(pos) for pos in poses if pos is not None]))
        self.position: LongPosType = jnp.transpose(r)
        helpers.checkstate(ndarray=self.position, ndaxis=NDAXIS)

    @property
    def length(self) -> int:
        """Number of footprints"""

        return int(self.position.shape[NDAXIS.L])

    @property
    def dimension(self) -> int:
        """Spatial dimension"""

        return int(self.position.shape[NDAXIS.D])

    def __str__(self) -> str:
        info = f"L={self.length}, D={self.dimension}, dt={self.dt}"
        return f"{self.__class__.__name__}({info})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.position.shape}"

    @classmethod
    def from_npy(cls, npy_path: PathType, dt: ConstType) -> Trajectory:
        position: LongPosType = jnp.load(npy_path)
        return cls(dt, *position.T)

    @helpers.checktime(nonzero=True)
    def get_increments(
        self, lagtime: int, rolling: bool = True, axis: Optional[str] = None
    ) -> List[float]:
        dx = _get_increments(self.position, lagtime, rolling, axis)
        return dx.tolist()  # type: ignore[no-any-return]

    @helpers.checktime(nonzero=True)
    def get_displacement_moment(
        self,
        lagtime: int,
        order: int = 2,
        rolling: bool = True,
        axis: Optional[str] = None,
    ) -> float:
        m = _get_displacement_moment(self.position, lagtime, order, rolling, axis)
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
        m = _get_cosine_moment(self.position, lagtime, order, rolling, epsilon)
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
            self.position, lagtime, self.dt, rolling, epsilon
        )
        return float(m)

    def to_npy(self, npy_path: PathType) -> None:
        jnp.save(file=npy_path, arr=self.position)


@partial(jit, static_argnums=(1, 2, 3))
def _get_increments(
    x: LongPosType, lagtime: int, rolling: bool = True, axis: Optional[str] = None
) -> Array:
    x = jnp.asarray(x)
    i = getattr(Axis, axis) if axis else None
    dx = jnp.roll(x, shift=-lagtime, axis=NDAXIS.L) - x
    if not rolling:
        dx = dx[0, i] if axis else jLA.norm(dx[0], axis=-1)
    else:
        dx = dx[:-lagtime, i] if axis else jLA.norm(dx[:-lagtime], axis=NDAXIS.D)
    return dx.flatten()


@partial(jit, static_argnums=(1, 2, 3, 4))
def _get_displacement_moment(
    x: LongPosType,
    lagtime: int,
    order: int = 2,
    rolling: bool = True,
    axis: Optional[str] = None,
) -> Array:
    x = jnp.asarray(x)
    i = getattr(Axis, axis) if axis else None
    m = (jnp.roll(x, shift=-lagtime, axis=NDAXIS.L) - x) ** order
    if not rolling:
        return m[0, i] if axis else jnp.sum(m[0], axis=-1)
    m = m[:-lagtime, i] if axis else jnp.sum(m[:-lagtime], axis=NDAXIS.D)
    return jnp.mean(m)


@partial(jit, static_argnums=(1, 2, 3, 4))
def _get_cosine_moment(
    x: LongPosType,
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
    vel = jitted.normalize(vel[:-epsilon], axis=NDAXIS.D)
    if not rolling:
        cos = jnp.sum(vel[lagtime] * vel[0], axis=-1)
        return jnp.cos(order * jnp.arccos(_fix(cos)))
    mul = jnp.roll(vel, shift=-lagtime, axis=NDAXIS.L) * vel
    cos = (
        jnp.sum(mul, axis=NDAXIS.D)
        if not lagtime
        else jnp.sum(mul[:-lagtime], axis=NDAXIS.D)
    )
    return jnp.mean(jnp.cos(order * jnp.arccos(_fix(cos))))


@partial(jit, static_argnums=(1, 2, 3, 4))
def _get_velocity_autocorrelation(
    x: LongPosType,
    lagtime: int,
    dt: ConstType,
    rolling: bool = True,
    epsilon: int = 1,
) -> Array:
    x = jnp.asarray(x)
    vel = jnp.roll(x, shift=-epsilon, axis=NDAXIS.L) - x
    vel = vel[:-epsilon] / (epsilon * dt)
    if not rolling:
        dot = jnp.sum(vel[lagtime] * vel[0], axis=-1)
        return dot
    mul = jnp.roll(vel, shift=-lagtime, axis=NDAXIS.L) * vel
    dot = (
        jnp.sum(mul, axis=NDAXIS.D)
        if not lagtime
        else jnp.sum(mul[:-lagtime], axis=NDAXIS.D)
    )
    return jnp.mean(dot, axis=-1)
