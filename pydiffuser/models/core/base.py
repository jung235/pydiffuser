from __future__ import annotations

import abc
import inspect
from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from numpy.random import rand, randint

from pydiffuser.exceptions import ConfigException, InvalidDimensionError
from pydiffuser.logger import logger
from pydiffuser.tracer import Ensemble
from pydiffuser.typing import LongLongPosType, P, T
from pydiffuser.utils import BaseDiffusionConfig, jitted

MODEL_KWARGS = "model_kwargs"
GENERATE_KWARGS = "generate_kwargs"


class BaseDiffusion(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def __init__(self) -> None:
        self.config: Optional[BaseDiffusionConfig] = None
        self._state_dict: Dict[str, Any] = {}
        self._interacting: bool = False
        self._precision_x64: bool = False

    @classmethod
    def from_config(cls, config: BaseDiffusionConfig) -> BaseDiffusion:
        if config.name != cls.name:
            raise ConfigException(
                f"Incompatible configuration {config.name} is encountered"
            )
        params = list(signature(cls.__init__).parameters.keys())
        if MODEL_KWARGS in params:
            params.pop()
        model = cls(*[getattr(config, param) for param in params[1:]])
        model._inject_init_args(config)
        return model

    @property
    def interacting(self):
        return self._interacting

    @interacting.setter
    def interacting(self, interacting: bool):
        self._interacting = interacting

    @property
    def precision_x64(self):
        return self._precision_x64

    @precision_x64.setter
    def precision_x64(self, precision_x64: bool):
        self._precision_x64 = precision_x64

    @abc.abstractmethod
    def generate(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        self.pre_generate(realization, length, dimension, dt, *generate_kwargs.values())
        ens = Ensemble(dt=self.generate_info["dt"])
        return ens

    def pre_generate(self, *generate_args) -> None:
        self._stash_generate_args(*generate_args)
        dimension = self.generate_info["dimension"]
        if dimension >= 4:
            raise InvalidDimensionError(
                f"Unsupported dimension {dimension} is encountered"
            )
        if self.interacting:
            logger.debug(
                f"Generating interacting particles from `{self.__class__.__name__}`. "
                "The calculation will be significantly slower than with non-interacting particles."
            )

        jax.config.update("jax_platform_name", "cpu")  # TODO
        jax.config.update("jax_enable_x64", self.precision_x64)
        if self.precision_x64:
            logger.debug(
                f"The simulation launched from `{self.__class__.__name__}` requires x64 precision."
            )
        return

    @property
    def generate_info(self) -> Dict[str, Any]:
        """The last arguments passed to `generate` method"""

        info = {}
        try:
            for param in signature(self.generate).parameters:
                info[param] = self._state_dict[param]
        except KeyError:
            pass
        return info

    def _inject_init_args(self, config: BaseDiffusionConfig) -> None:
        for param in vars(self):
            if param in config:
                setattr(self, param, getattr(config, param))
        self.config = config
        return

    def _stash_generate_args(self, *user_args) -> None:
        params = list(signature(self.generate).parameters.keys())
        defaults = list(signature(self.generate).parameters.values())
        if GENERATE_KWARGS in params:
            params.pop()
            defaults.pop()

        for param, user_arg, default in zip(params, user_args, defaults, strict=True):
            if self.config is not None:
                config_arg = getattr(self.config, param)
                if user_arg != default.default:
                    logger.warning(
                        f"{param} {user_arg} will be ignored to use "
                        f"`{self.config.__class__.__name__}.{param}` {config_arg}.",
                        stacklevel=1,
                    )
                self._state_dict[param] = config_arg
            else:
                self._state_dict[param] = user_arg
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def _load_hook(self, hook: str) -> Callable[P, T]:  # type: ignore[return]
        for name, obj in inspect.getmembers(self):
            if name == hook and inspect.ismethod(obj):
                logger.debug(f"Loading hook: `{obj.__name__}` for {self.__class__}.")
                return obj

    def _get_initial_position(
        self, realization: Optional[int] = None, dimension: Optional[int] = None
    ) -> LongLongPosType:
        if realization is None or dimension is None:
            realization, _, dimension = list(self.generate_info.values())[:3]
        x = jnp.zeros((realization, 1, dimension))
        return x

    def _get_initial_orientation(
        self, realization: Optional[int] = None, dimension: Optional[int] = None
    ) -> LongLongPosType:
        if realization is None or dimension is None:
            realization, _, dimension = list(self.generate_info.values())[:3]
        u = jitted.get_noise(
            generator=rand if dimension >= 2 else partial(randint, 0, 2),  # type: ignore
            size=realization * 1 if dimension <= 2 else realization * 2,  # type: ignore
            shape=(realization, 1, -1),
        )
        u *= 2 * jnp.pi if dimension == 2 else jnp.pi
        u = u.at[:, :, -1].multiply(2) if dimension == 3 else u
        return u
