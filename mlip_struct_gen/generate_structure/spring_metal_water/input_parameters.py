"""Input parameters for spring-restrained metal-water interface generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger

# Re-export SpringConstraint from spring_water_box
from ..spring_water_box.input_parameters import SpringConstraint

__all__ = [
    "SpringConstraint",
    "MetalWaterSpringConstraint",
    "SpringMetalWaterParameters",
]


@dataclass
class MetalWaterSpringConstraint:
    """
    Spring restraint between metal surface atoms and water atoms.

    Args:
        water_element: Water atom element to restrain ("O" or "H")
        count: Number of pairs to restrain (int or "all")
        distance: Target equilibrium distance in Angstroms
        k_spring: Spring constant (default: 50.0)
    """

    water_element: str
    count: int | str
    distance: float
    k_spring: float = 50.0


@dataclass
class SpringMetalWaterParameters:
    """
    Parameters for spring-restrained metal-water interface generation.

    Generates a metal-water interface with spring restraints for MLIP training.
    Unlike frozen constraints, spring restraints allow atoms to move
    while being pulled toward target distances.

    Supports spring restraints on:
    - Metal-water distances (Metal-O, Metal-H)
    - O-H bonds (intramolecular water)
    - O-O distances (intermolecular water)
    """

    # Output
    output_file: str
    model_files: list[str] = field(
        default_factory=lambda: [
            "graph.000.pb",
            "graph.001.pb",
            "graph.002.pb",
            "graph.003.pb",
            "graph.004.pb",
        ]
    )

    # Metal surface
    metal: str = "Pt"
    size: tuple[int, int, int] = (4, 4, 4)
    lattice_constant: float | None = None
    fix_bottom_layers: int = 0

    # Water
    n_water: int = 50
    density: float = 1.0
    gap_above_metal: float = 3.0
    vacuum_above_water: float = 0.0
    water_model: str = "SPC/E"

    # Metal-water spring constraints
    metal_water_spring_constraints: list[MetalWaterSpringConstraint] = field(default_factory=list)

    # Water-only spring constraints
    spring_constraints: list[SpringConstraint] = field(default_factory=list)

    # Constraint settings
    constraint_seed: int = 42

    # LAMMPS MD parameters
    minimize: bool = False
    ensemble: str = "npt"
    nsteps: int = 1000
    temp: float = 300.0
    pres: float = 1.0
    timestep: float = 0.0005
    dump_freq: int = 10
    thermo_freq: int = 10
    tau_t: float = 0.1
    tau_p: float = 0.5

    # Packmol parameters
    packmol_tolerance: float = 2.0
    seed: int = 12345
    packmol_executable: str = "packmol"

    # Elements for LAMMPS
    elements: list[str] = field(default_factory=lambda: ["Pt", "O", "H"])

    # Logging
    log: bool = False
    logger: Optional["MLIPLogger"] = None
