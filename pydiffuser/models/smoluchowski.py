from typing import Any, Dict, List, Optional

from pydiffuser.exceptions import InvalidDimensionError
from pydiffuser.mech.fields import get_static_argsigs, periodic_potential
from pydiffuser.models.core import OverdampedLangevin, OverdampedLangevinConfig

DEFAULT_POTENTIAL = periodic_potential.__name__
DEFAULT_POTENTIAL_PARAMS = get_static_argsigs(periodic_potential)


class SmoluchowskiEquationConfig(OverdampedLangevinConfig):
    name: str = "smoluchowski"

    def __init__(
        self,
        potential: str = DEFAULT_POTENTIAL,
        potential_params: Dict[str, Any] = DEFAULT_POTENTIAL_PARAMS,
        external_force: float = 0.0,
        friction_coefficient: float = 1.0,
        diffusivity: float = 1.0,
        generate_hooks: Optional[List[str]] = None,
        **kwargs,
    ):
        super(SmoluchowskiEquationConfig, self).__init__(
            potential=potential,
            potential_params=potential_params,
            external_force=external_force,
            friction_coefficient=friction_coefficient,
            diffusivity=diffusivity,
            generate_hooks=generate_hooks,
            **kwargs,
        )


class SmoluchowskiEquation(OverdampedLangevin):
    name: str = "smoluchowski"

    def __init__(
        self,
        potential: str = DEFAULT_POTENTIAL,
        potential_params: Dict[str, Any] = DEFAULT_POTENTIAL_PARAMS,
        external_force: float = 0.0,
        friction_coefficient: float = 1.0,
        diffusivity: float = 1.0,
        generate_hooks: Optional[List[str]] = None,
    ):
        super(SmoluchowskiEquation, self).__init__(
            potential=potential,
            potential_params=potential_params,
            external_force=external_force,
            friction_coefficient=friction_coefficient,
            diffusivity=diffusivity,
            generate_hooks=generate_hooks,
        )

    def pre_generate(self, *generate_args) -> None:
        super().pre_generate(*generate_args)
        dimension = self.generate_info["dimension"]
        if dimension == 3:
            raise InvalidDimensionError(
                f"Unsupported dimension {dimension} is encountered"
            )
        return
