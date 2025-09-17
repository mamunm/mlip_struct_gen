"""Input parameters for salt-water LAMMPS simulations."""

from dataclasses import dataclass

from ..water_box.input_parameters import WaterBoxLAMMPSParameters


@dataclass
class SaltWaterBoxLAMMPSParameters(WaterBoxLAMMPSParameters):
    """Parameters for salt-water box LAMMPS simulations.

    Inherits all water box parameters and adds ion-specific settings.
    """

    # Ion force field selection
    ion_ff: str = "joung-cheatham"  # Force field for ions

    # Additional salt-water specific parameters can be added here if needed