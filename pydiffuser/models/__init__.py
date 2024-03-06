import inspect
from typing import Dict, Type

from ..utils import BaseDiffusionConfig
from .abp import ActiveBrownianParticle, ActiveBrownianParticleConfig
from .aoup import ActiveOUParticle, ActiveOUParticleConfig
from .bm import BrownianMotion, BrownianMotionConfig
from .core.base import BaseDiffusion
from .levy import LevyWalk, LevyWalkConfig
from .rtp import RunAndTumbleParticle, RunAndTumbleParticleConfig
from .smoluchowski import SmoluchowskiEquation, SmoluchowskiEquationConfig
from .vicsek import VicsekModel, VicsekModelConfig

__all__ = [
    "ActiveBrownianParticle",
    "ActiveBrownianParticleConfig",
    "ActiveOUParticle",
    "ActiveOUParticleConfig",
    "BrownianMotion",
    "BrownianMotionConfig",
    "LevyWalk",
    "LevyWalkConfig",
    "RunAndTumbleParticle",
    "RunAndTumbleParticleConfig",
    "SmoluchowskiEquation",
    "SmoluchowskiEquationConfig",
    "VicsekModel",
    "VicsekModelConfig",
]


CONFIG_REGISTRY: Dict[str, Type[BaseDiffusionConfig]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BaseDiffusionConfig)
}

MODEL_REGISTRY: Dict[str, Type[BaseDiffusion]] = {
    obj.name: obj
    for _, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, BaseDiffusion)
}
