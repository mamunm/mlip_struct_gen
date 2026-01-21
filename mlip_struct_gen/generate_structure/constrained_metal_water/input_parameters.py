"""Input parameters for constrained metal-water interface generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger

# Re-export constraint classes from constrained_water_box for consistency
from ..constrained_water_box.input_parameters import AngleConstraint, DistanceConstraint

__all__ = [
    "DistanceConstraint",
    "AngleConstraint",
    "MetalWaterDistanceConstraint",
    "MetalWaterAngleConstraint",
    "ConstrainedMetalWaterParameters",
]


@dataclass
class MetalWaterDistanceConstraint:
    """
    Distance constraint between metal surface atoms and water atoms.

    Always uses top-layer metal atoms. Water molecules are moved as rigid bodies
    to achieve the target distance (metal stays fixed).

    Args:
        water_element: Water atom element to constrain ("O" or "H")
        count: Number of pairs to constrain (int or "all")
        distance: Target distance in Angstroms

    Examples:
        >>> MetalWaterDistanceConstraint("O", 5, 2.5)   # 5 Metal-O pairs at 2.5 A
        >>> MetalWaterDistanceConstraint("H", "all", 2.0)  # All Metal-H pairs at 2.0 A
    """

    water_element: str
    count: int | str
    distance: float


@dataclass
class MetalWaterAngleConstraint:
    """
    Metal-O-H angle constraint for water molecules near the metal surface.

    Controls water molecule orientation relative to the metal surface.
    Rotates water molecule around the Metal-O axis to achieve target angle.

    Args:
        count: Number of angles to constrain (int or "all")
        angle: Target Metal-O-H angle in degrees

    Examples:
        >>> MetalWaterAngleConstraint(3, 120.0)   # 3 Metal-O-H angles at 120 deg
        >>> MetalWaterAngleConstraint("all", 110.0)  # All Metal-O-H angles at 110 deg
    """

    count: int | str
    angle: float


@dataclass
class ConstrainedMetalWaterParameters:
    """
    Parameters for constrained metal-water interface generation.

    Generates a metal-water interface with specified geometric constraints for MLIP training.
    Outputs LAMMPS data file and input script for DeepMD simulation.

    Supports constraints on:
    - Metal-water distances (Metal-O, Metal-H)
    - Metal-O-H angles (water orientation relative to surface)
    - O-H bonds (intramolecular water)
    - H-O-H angles (intramolecular water)
    - O-O distances (intermolecular water)

    Args:
        output_file: Output LAMMPS data file path.
        model_files: DeepMD model files (e.g., ["graph.000.pb"]).
        metal: Metal element symbol (e.g., "Pt", "Au", "Cu").
        size: Surface size as (nx, ny, nz) unit cells.
        n_water: Number of water molecules.
        lattice_constant: Optional custom lattice constant in Angstroms.
        fix_bottom_layers: Number of bottom metal layers to fix. Default: 0
        density: Water density in g/cm3. Default: 1.0
        gap_above_metal: Gap between metal surface and water in Angstroms. Default: 3.0
        vacuum_above_water: Vacuum space above water in Angstroms. Default: 0.0
        water_model: Water model for initial geometry. Default: "SPC/E"
        metal_water_distance_constraints: List of MetalWaterDistanceConstraint objects.
        metal_water_angle_constraints: List of MetalWaterAngleConstraint objects.
        distance_constraints: List of DistanceConstraint objects (O-H, O-O).
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
        elements: Element order for LAMMPS. Default: ["Pt", "O", "H"]
        log: Enable logging. Default: False
        logger: Custom MLIPLogger instance. Default: None

    Examples:
        >>> params = ConstrainedMetalWaterParameters(
        ...     output_file="pt_water_constrained.data",
        ...     model_files=["graph.000.pb"],
        ...     metal="Pt",
        ...     size=(4, 4, 4),
        ...     n_water=50,
        ...     metal_water_distance_constraints=[
        ...         MetalWaterDistanceConstraint("O", 5, 2.5),
        ...     ],
        ...     elements=["Pt", "O", "H"],
        ... )

        >>> params = ConstrainedMetalWaterParameters(
        ...     output_file="cu_water_constrained.data",
        ...     model_files=["graph.000.pb"],
        ...     metal="Cu",
        ...     size=(4, 4, 4),
        ...     n_water=50,
        ...     metal_water_distance_constraints=[
        ...         MetalWaterDistanceConstraint("O", 3, 2.3),
        ...     ],
        ...     metal_water_angle_constraints=[
        ...         MetalWaterAngleConstraint(3, 120.0),
        ...     ],
        ...     distance_constraints=[
        ...         DistanceConstraint("O", "H", 2, 0.85),
        ...     ],
        ...     elements=["Cu", "O", "H"],
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

    # Water
    n_water: int = 50
    density: float = 1.0
    gap_above_metal: float = 3.0
    vacuum_above_water: float = 0.0
    water_model: str = "SPC/E"

    # Metal-water constraints
    metal_water_distance_constraints: list[MetalWaterDistanceConstraint] = field(
        default_factory=list
    )
    metal_water_angle_constraints: list[MetalWaterAngleConstraint] = field(default_factory=list)

    # Water-only constraints (reused from constrained_water_box)
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
    elements: list[str] = field(default_factory=lambda: ["Pt", "O", "H"])

    # Logging
    log: bool = False
    logger: Optional["MLIPLogger"] = None
