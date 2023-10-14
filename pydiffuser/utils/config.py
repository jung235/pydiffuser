from __future__ import annotations

import abc
import copy
import json
from dataclasses import make_dataclass
from typing import Any, Dict, Optional

from pydiffuser import __version__
from pydiffuser.typing import PathType

JAX_DEVICES = ("cpu", "gpu", "tpu")
JAX_PRECISIONS = ("x32", "x64")


class BaseDiffusionConfig(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def __init__(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        model_alias: Optional[str] = None,
        pydiffuser_version: str = __version__,
    ) -> None:
        self.model_alias = model_alias if model_alias else self.name
        self.pydiffuser_version = pydiffuser_version
        self.realization = realization
        self.length = length
        self.dimension = dimension
        self.dt = dt

    @classmethod
    def show_fields(cls) -> Dict[str, Any]:
        return copy.deepcopy(vars(cls()))

    @classmethod
    def from_json(cls, json_path: PathType) -> BaseDiffusionConfig:
        with open(json_path, "r") as f:
            info = json.load(f)
        return cls(**info)

    def to_json(self, json_path: PathType) -> None:
        info = vars(self)
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

    def to_dataclass(self) -> object:
        info = copy.deepcopy(vars(self))
        obj = make_dataclass(self.__class__.__name__, info.keys())
        return obj(**info)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
