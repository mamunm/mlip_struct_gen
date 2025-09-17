"""LAMMPS input parameters for water box simulations."""

from dataclasses import dataclass

from ..input_parameters import LAMMPSInputParameters


@dataclass
class WaterBoxLAMMPSParameters(LAMMPSInputParameters):
    """Simplified parameters for water box LAMMPS simulations focused on MLIP training."""

    # SHAKE is always enabled for water (better sampling)
    use_shake: bool = True

    def __post_init__(self) -> None:
        """Validate water-specific parameters."""
        super().__post_init__()

        # Validate water model
        valid_models = ["SPC/E", "SPCE", "TIP3P", "TIP4P"]
        if self.water_model.upper() not in [m.upper() for m in valid_models]:
            raise ValueError(
                f"Invalid water model: {self.water_model}. Must be one of {valid_models}"
            )
