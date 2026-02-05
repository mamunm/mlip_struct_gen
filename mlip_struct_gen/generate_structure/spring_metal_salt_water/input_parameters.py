"""Input parameters for spring-restrained metal-salt-water interface generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger

# Re-export from other modules
from ..spring_metal_water.input_parameters import MetalWaterSpringConstraint
from ..spring_water_box.input_parameters import SpringConstraint

__all__ = [
    "SpringConstraint",
    "MetalWaterSpringConstraint",
    "MetalIonSpringConstraint",
    "SpringMetalSaltWaterParameters",
]


@dataclass
class MetalIonSpringConstraint:
    """
    Spring restraint between metal surface atoms and salt ions.

    Args:
        ion_element: Ion element to restrain (e.g., "Na", "Cl", "K", "Li")
        count: Number of pairs to restrain (int or "all")
        distance: Target equilibrium distance in Angstroms
        k_spring: Spring constant (default: 50.0)
    """

    ion_element: str
    count: int | str
    distance: float
    k_spring: float = 50.0


@dataclass
class SpringMetalSaltWaterParameters:
    """
    Parameters for spring-restrained metal-salt-water interface generation.

    Generates a metal-salt-water interface with spring restraints for MLIP training.
    Unlike frozen constraints, spring restraints allow atoms to move
    while being pulled toward target distances.

    Supports spring restraints on:
    - Metal-water distances (Metal-O, Metal-H)
    - Metal-ion distances (Metal-Na, Metal-Cl, etc.)
    - Ion-water distances (Na-O, Cl-O)
    - Ion-ion distances (Na-Cl)
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

    # Salt
    salt_type: str = "NaCl"
    n_salt: int = 5
    include_salt_volume: bool = False
    no_salt_zone: float = 0.2

    # Water
    n_water: int = 50
    density: float = 1.0
    gap: float = 0.0
    vacuum_above_water: float = 0.0
    water_model: str = "SPC/E"

    # Metal-water spring constraints
    metal_water_spring_constraints: list[MetalWaterSpringConstraint] = field(default_factory=list)

    # Metal-ion spring constraints
    metal_ion_spring_constraints: list[MetalIonSpringConstraint] = field(default_factory=list)

    # General spring constraints (for ion-water, ion-ion, water-water)
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
    elements: list[str] = field(default_factory=lambda: ["Pt", "O", "H", "Na", "Cl"])

    # Logging
    log: bool = False
    logger: Optional["MLIPLogger"] = None
