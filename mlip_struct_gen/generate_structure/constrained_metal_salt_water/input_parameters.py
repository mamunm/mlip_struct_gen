"""Input parameters for constrained metal-salt-water interface generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger

# Re-export constraint classes from other modules
from ..constrained_metal_water.input_parameters import (
    MetalWaterAngleConstraint,
    MetalWaterDistanceConstraint,
)
from ..constrained_water_box.input_parameters import AngleConstraint, DistanceConstraint

__all__ = [
    "DistanceConstraint",
    "AngleConstraint",
    "MetalWaterDistanceConstraint",
    "MetalWaterAngleConstraint",
    "MetalIonDistanceConstraint",
    "ConstrainedMetalSaltWaterParameters",
]


@dataclass
class MetalIonDistanceConstraint:
    """
    Distance constraint between metal surface atoms and salt ions.

    Always uses top-layer metal atoms. Ion is moved to achieve target distance
    (metal stays fixed).

    Args:
        ion_element: Ion element to constrain (e.g., "Na", "Cl", "K", "Li")
        count: Number of pairs to constrain (int or "all")
        distance: Target distance in Angstroms

    Examples:
        >>> MetalIonDistanceConstraint("Na", 3, 2.8)   # 3 Metal-Na pairs at 2.8 A
        >>> MetalIonDistanceConstraint("Cl", "all", 3.0)  # All Metal-Cl pairs at 3.0 A
    """

    ion_element: str
    count: int | str
    distance: float


@dataclass
class ConstrainedMetalSaltWaterParameters:
    """
    Parameters for constrained metal-salt-water interface generation.

    Generates a metal-salt-water interface with specified geometric constraints for MLIP training.
    Outputs LAMMPS data file and input script for DeepMD simulation.

    Supports constraints on:
    - Metal-water distances (Metal-O, Metal-H)
    - Metal-O-H angles (water orientation relative to surface)
    - Metal-ion distances (Metal-Na, Metal-Cl, etc.)
    - Ion-water distances (Na-O, Cl-O via DistanceConstraint)
    - Ion-ion distances (Na-Cl via DistanceConstraint)
    - O-H bonds (intramolecular water)
    - H-O-H angles (intramolecular water)
    - O-O distances (intermolecular water)

    Args:
        output_file: Output LAMMPS data file path.
        model_files: DeepMD model files (e.g., ["graph.000.pb"]).
        metal: Metal element symbol (e.g., "Pt", "Cu").
        size: Surface size as (nx, ny, nz) unit cells.
        n_water: Number of water molecules.
        salt_type: Type of salt (e.g., "NaCl", "KCl").
        n_salt: Number of salt formula units.
        lattice_constant: Optional custom lattice constant in Angstroms.
        fix_bottom_layers: Number of bottom metal layers to fix. Default: 0
        include_salt_volume: Account for ion volume when computing box size. Default: False
        density: Water/solution density in g/cm3. Default: 1.0
        gap: Gap between metal surface and solution in Angstroms. Default: 0.0
        vacuum_above_water: Vacuum space above water in Angstroms. Default: 0.0
        no_salt_zone: Fraction of box height where ions are excluded. Default: 0.2
        water_model: Water model for initial geometry. Default: "SPC/E"
        metal_water_distance_constraints: List of MetalWaterDistanceConstraint objects.
        metal_water_angle_constraints: List of MetalWaterAngleConstraint objects.
        metal_ion_distance_constraints: List of MetalIonDistanceConstraint objects.
        distance_constraints: List of DistanceConstraint objects (O-H, O-O, Na-O, Cl-O, Na-Cl).
        angle_constraints: List of AngleConstraint objects (H-O-H).
        constraint_seed: Seed for random selection of constrained pairs. Default: 42
        constraint_type: "rigid" (K=10000) or "harmonic" (user K). Default: "rigid"
        harmonic_k: Spring constant for harmonic constraints. Default: 50.0
        minimize: Add energy minimization before MD. Default: False
        nsteps: MD steps for LAMMPS. Default: 1000
        temp: Temperature in K. Default: 300.0
        pres: Pressure in bar. Default: 1.0
        timestep: Timestep in ps. Default: 0.0005
        dump_freq: Trajectory dump frequency. Default: 10
        thermo_freq: Thermo output frequency. Default: 10
        tau_t: Thermostat time constant. Default: 0.1
        tau_p: Barostat time constant. Default: 0.5
        packmol_tolerance: Packmol tolerance in Angstroms. Default: 2.0
        seed: Packmol random seed. Default: 12345
        packmol_executable: Path to Packmol. Default: "packmol"
        elements: Element order for LAMMPS. Default: ["Pt", "O", "H", "Na", "Cl"]
        log: Enable logging. Default: False
        logger: Custom MLIPLogger instance. Default: None

    Examples:
        >>> params = ConstrainedMetalSaltWaterParameters(
        ...     output_file="pt_nacl_water_constrained.data",
        ...     model_files=["graph.000.pb"],
        ...     metal="Pt",
        ...     size=(4, 4, 4),
        ...     n_water=50,
        ...     salt_type="NaCl",
        ...     n_salt=5,
        ...     metal_water_distance_constraints=[
        ...         MetalWaterDistanceConstraint("O", 5, 2.5),
        ...     ],
        ...     metal_ion_distance_constraints=[
        ...         MetalIonDistanceConstraint("Na", 3, 2.8),
        ...     ],
        ...     distance_constraints=[
        ...         DistanceConstraint("Na", "O", 2, 2.3),  # Na-O solvation
        ...     ],
        ...     elements=["Pt", "O", "H", "Na", "Cl"],
        ... )
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

    # Metal-water constraints
    metal_water_distance_constraints: list[MetalWaterDistanceConstraint] = field(
        default_factory=list
    )
    metal_water_angle_constraints: list[MetalWaterAngleConstraint] = field(default_factory=list)

    # Metal-ion constraints
    metal_ion_distance_constraints: list[MetalIonDistanceConstraint] = field(default_factory=list)

    # General distance/angle constraints (for ion-water, ion-ion, water-water)
    distance_constraints: list[DistanceConstraint] = field(default_factory=list)
    angle_constraints: list[AngleConstraint] = field(default_factory=list)

    # Constraint settings
    constraint_seed: int = 42
    constraint_type: str = "rigid"  # "rigid" (K=10000) or "harmonic"
    harmonic_k: float = 50.0  # Spring constant for harmonic constraints

    # LAMMPS MD parameters
    minimize: bool = False  # Add energy minimization before MD
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
