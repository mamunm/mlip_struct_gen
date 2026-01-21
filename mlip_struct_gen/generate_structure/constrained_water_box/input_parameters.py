"""Input parameters for constrained water box generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class DistanceConstraint:
    """
    Distance constraint for atoms in water box.

    Automatically detects intramolecular vs intermolecular:
    - O-H: Intramolecular, moves H atom along bond vector
    - O-O: Intermolecular, moves entire molecule as rigid body

    Args:
        element1: First element type (e.g., "O")
        element2: Second element type (e.g., "H" or "O")
        count: Number of pairs to constrain (int or "all")
        distance: Target distance in Angstroms

    Examples:
        >>> DistanceConstraint("O", "H", 1, 0.7)   # 1 O-H bond to 0.7 A
        >>> DistanceConstraint("O", "O", 1, 2.5)   # 1 O-O pair to 2.5 A
        >>> DistanceConstraint("O", "H", "all", 0.8)  # All O-H bonds
    """

    element1: str
    element2: str
    count: int | str
    distance: float


@dataclass
class AngleConstraint:
    """
    H-O-H angle constraint for water molecules.

    Args:
        count: Number of angles to constrain (int or "all")
        angle: Target angle in degrees

    Examples:
        >>> AngleConstraint(1, 100.0)    # 1 H-O-H angle to 100 degrees
        >>> AngleConstraint("all", 90.0) # All H-O-H angles to 90 degrees
    """

    count: int | str
    angle: float


@dataclass
class ConstrainedWaterBoxParameters:
    """
    Parameters for constrained water box generation.

    Generates a water box with specified geometric constraints for MLIP training.
    Outputs LAMMPS data file and input script for DeepMD simulation.

    Args:
        output_file: Output LAMMPS data file path.
        model_files: DeepMD model files (e.g., ["graph.000.pb"]).
        box_size: Box dimensions in Angstroms (float for cubic, or 3-tuple).
        n_water: Number of water molecules.
        density: Water density in g/cm3.
        water_model: Water model for initial geometry. Default: "SPC/E"
        distance_constraints: List of DistanceConstraint objects.
        angle_constraints: List of AngleConstraint objects.
        constraint_seed: Seed for random selection of constrained pairs. Default: 42
        nsteps: MD steps for LAMMPS. Default: 1000
        temp: Temperature in K. Default: 300.0
        pres: Pressure in bar. Default: 1.0
        timestep: Timestep in ps. Default: 0.0005
        dump_freq: Trajectory dump frequency. Default: 10
        thermo_freq: Thermo output frequency. Default: 10
        tau_t: Thermostat time constant. Default: 0.1
        tau_p: Barostat time constant. Default: 0.5
        tolerance: Packmol tolerance in Angstroms. Default: 2.0
        seed: Packmol random seed. Default: 12345
        packmol_executable: Path to Packmol. Default: "packmol"
        elements: Element order for LAMMPS. Default: ["O", "H"]
        log: Enable logging. Default: False
        logger: Custom MLIPLogger instance. Default: None

    Examples:
        >>> params = ConstrainedWaterBoxParameters(
        ...     output_file="constrained.data",
        ...     model_files=["graph.000.pb"],
        ...     n_water=32,
        ...     distance_constraints=[DistanceConstraint("O", "H", 1, 0.7)]
        ... )

        >>> params = ConstrainedWaterBoxParameters(
        ...     output_file="constrained.data",
        ...     model_files=["graph.000.pb"],
        ...     n_water=32,
        ...     angle_constraints=[AngleConstraint(1, 100.0)]
        ... )
    """

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
    box_size: float | tuple[float, float, float] | None = None
    n_water: int | None = None
    density: float | None = None
    water_model: str = "SPC/E"
    distance_constraints: list[DistanceConstraint] = field(default_factory=list)
    angle_constraints: list[AngleConstraint] = field(default_factory=list)
    constraint_seed: int = 42
    constraint_type: str = "rigid"  # "rigid" (K=10000) or "harmonic"
    harmonic_k: float = 50.0  # Spring constant for harmonic constraints
    minimize: bool = False  # Add energy minimization before MD
    nsteps: int = 1000
    temp: float = 300.0
    pres: float = 1.0
    timestep: float = 0.0005
    dump_freq: int = 10
    thermo_freq: int = 10
    tau_t: float = 0.1
    tau_p: float = 0.5
    tolerance: float = 2.0
    seed: int = 12345
    packmol_executable: str = "packmol"
    elements: list[str] = field(default_factory=lambda: ["O", "H"])
    log: bool = False
    logger: Optional["MLIPLogger"] = None
