from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.random import rand, randint

from pydiffuser.exceptions import MemoryAllocationError, ShapeMismatchError
from pydiffuser.models.core import BaseDiffusion
from pydiffuser.tracer import Ensemble
from pydiffuser.typing import ConstType, LongLongPosType
from pydiffuser.utils import BaseDiffusionConfig, helpers, jitted


class ContinuousTimeRandomWalkConfig(BaseDiffusionConfig):
    name: str = "ctrw"

    def __init__(self, **kwargs) -> None:
        super(ContinuousTimeRandomWalkConfig, self).__init__(**kwargs)


class ContinuousTimeRandomWalk(BaseDiffusion):
    name: str = "ctrw"

    def __init__(self) -> None:
        super(ContinuousTimeRandomWalk, self).__init__()
        self.precision_x64 = True

    @property
    def one_step(self) -> ConstType:
        """Length of the displacement per step"""

        raise NotImplementedError

    def generate(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        ens = super().generate(realization, length, dimension, dt, **generate_kwargs)
        realization, length, dimension, _ = list(self.generate_info.values())[:4]

        interarrival: Dict[int, Array] = self.get_interarrival_times()
        ens.update_meta_dict(item={"interarrival": interarrival})
        return ens

    def get_interarrival_times(self) -> Dict[int, Array]:
        raise NotImplementedError

    def _get_interarrival_times_per_tracer(
        self, generator: Callable[[int], Any], key: Array
    ) -> Tuple[Array, Array]:
        _, length, _, dt = list(self.generate_info.values())[:4]

        tau = jitted.get_noise(generator=partial(generator, key=key), size=length)
        # make it countable, and use `int64` to prevent overflow
        tau = jnp.array(jnp.round(tau, -int(np.log10(dt))) / dt, dtype=jnp.int64)
        cutoff = jnp.argmax(jnp.cumsum(tau) >= length)
        return tau, cutoff

    def _get_microstate_from_orientation(self, u: Array) -> LongLongPosType:
        x = self._get_initial_position()
        dim = x.shape[-1]
        if dim == 1:
            dx = self.one_step * jnp.cos(u)
        elif dim == 2:
            dx = jnp.concatenate(jitted.polar_to_cartesian(self.one_step, u), axis=2)
        else:
            dx = jnp.concatenate(
                jitted.spherical_to_cartesian(self.one_step, u[:, :, 0], u[:, :, 1]),
                axis=2,
            )
        x = jnp.concatenate((x, dx), axis=1)
        x = jnp.cumsum(x, axis=1)
        return x

    @helpers.deprecated
    def create(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        ens = super().generate(realization, length, dimension, dt, **generate_kwargs)
        realization, length, dimension, _ = list(self.generate_info.values())[:4]

        t = self.get_time_steps()
        if t.shape[0] != realization or t.ndim != 2:
            raise ShapeMismatchError("(`realization`, -1) is required")

        u = self.get_orientation_steps(realization, capacity=t.shape[1])
        u = jnp.stack(
            arrays=[jnp.repeat(ui, ti, axis=0) for ui, ti in zip(u, t, strict=True)],
            axis=0,
        )
        u = u[:, : length - 1]  # realization x (length - 1) x -1
        x = self._get_microstate_from_orientation(u)
        ens.update_microstate(microstate=x)
        return ens

    def get_time_steps(self) -> Array:
        raise NotImplementedError

    def get_orientation_steps(self, realization: int, capacity: int) -> Array:
        dim = self.generate_info["dimension"]
        u = jitted.get_noise(
            generator=rand if dim >= 2 else partial(randint, 0, 2),
            size=realization * capacity if dim <= 2 else realization * capacity * 2,
            shape=(realization, capacity, -1),
        )
        u *= 2 * jnp.pi if dim == 2 else jnp.pi
        u = u.at[:, :, -1].multiply(2) if dim == 3 else u
        return u

    @staticmethod
    def repeat(
        arr: Array, repeats_info: Dict[int, Array], cutoff: ConstType | None = None
    ) -> Array:
        arrs = []
        for id, ti in repeats_info.items():
            arrs.append(jnp.repeat(arr[id, : len(ti)], repeats=ti, axis=0))
        arr = jnp.stack(arrs, axis=0)
        if cutoff:
            arr = arr[:, :cutoff]  # realization x cutoff x -1
        return arr

    @staticmethod
    def slice(arr: Array, threshold: ConstType, axis: int = 0) -> Array:
        """For a 2darray with shape `realization` x -1 (here, `arr.T`),
        ```
        □□□□□...□□│□□□□□□□            □□□□□...□□ppppppp│
        □□□□□...□□│□□□□□□□            □□□□□...□□ppppppp│
        □□□□□...  └──┐                □□□□□...□□□□□pppp│
        □□□□□...     └──┐      =>     □□□□□...□□□□□□□□p│
        □□□□□...     ┌──┘             □□□□□...□□□□□□□□p│
        □□□□□...□□□□□│□□□□            □□□□□...□□□□□pppp│
        □□□□□...□□□□□│□□□□            □□□□□...□□□□□pppp│
                     └───> threshold                   └> padding
        ```
        _summary_

        Args:
            arr (Array): 2darray with shape -1 x `realization`.
            threshold (ConstType): _description_.
            axis (int, optional): Axis to apply cumulative summation.
        """

        cutoff = jnp.argmax(jnp.all(jnp.cumsum(arr, axis=axis) >= threshold, axis=1))
        if not cutoff:
            raise MemoryAllocationError(f"Reduce input `arr` size {arr.shape}.")
        arr = arr[:cutoff]
        accumed = jnp.cumsum(arr, axis=axis)[-1]
        padding = jnp.max(accumed) - accumed
        arr = jnp.vstack((arr, padding))
        return arr
