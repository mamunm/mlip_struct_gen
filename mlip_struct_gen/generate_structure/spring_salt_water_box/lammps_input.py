"""LAMMPS input file generation for DeepMD simulation with spring-restrained salt water."""

# Re-export from spring_water_box - same functionality
from ..spring_water_box.lammps_input import (
    ELEMENT_MASSES,
    LAMMPS_TEMPLATE,
    generate_ensemble_fix,
    generate_lammps_input,
    generate_mass_lines,
    generate_spring_restraints,
)

__all__ = [
    "LAMMPS_TEMPLATE",
    "ELEMENT_MASSES",
    "generate_spring_restraints",
    "generate_mass_lines",
    "generate_ensemble_fix",
    "generate_lammps_input",
]
