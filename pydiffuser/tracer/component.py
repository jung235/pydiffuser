from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from pydiffuser.typing import ConstType
from pydiffuser.utils import helpers


class Component(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dt: ConstType) -> None:
        self.dt = dt
        self.meta: Dict[str, Any] = {}

    @abc.abstractproperty
    def length(self) -> int:
        raise NotImplementedError

    @abc.abstractproperty
    def dimension(self) -> int:
        raise NotImplementedError

    def is_composite(self) -> bool:
        return False

    @abc.abstractmethod
    def get_increments(
        self, lagtime: int, rolling: bool = True, axis: Optional[str] = None
    ) -> List[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_displacement_moment(
        self,
        lagtime: int,
        order: int = 2,
        rolling: bool = True,
        axis: Optional[str] = None,
    ) -> float:
        raise NotImplementedError

    @helpers.checktime(nonzero=True)
    def get_mean_squared_displacement(
        self, lagtime: int, rolling: bool = True, axis: Optional[str] = None
    ) -> float:
        return self.get_displacement_moment(lagtime, 2, rolling, axis)

    @abc.abstractmethod
    def get_cosine_moment(
        self, lagtime: int, order: int = 1, rolling: bool = True, epsilon: int = 1
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_velocity_autocorrelation(
        self, lagtime: int, rolling: bool = True, epsilon: int = 1
    ) -> float:
        """_summary_"""

        raise NotImplementedError

    @helpers.checktime()
    def get_real_time(self, lagtime: int) -> float:
        return float(self.dt * lagtime)

    def update_meta_dict(self, item: Dict[str, Any]) -> None:
        self.meta.update(item)
