"""Input parameters for spring-restrained water box generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class SpringConstraint:
    """
    Spring bond restraint for atoms in water box.

    Uses LAMMPS fix restrain to apply harmonic spring between atom pairs.
    The spring allows atoms to move but penalizes deviation from target distance.

    Args:
        element1: First element type (e.g., "O")
        element2: Second element type (e.g., "H")
        count: Number of pairs to restrain (int or "all")
        distance: Target equilibrium distance in Angstroms
        k_spring: Spring constant (default: 50.0)

    Examples:
        >>> SpringConstraint("O", "H", 1, 1.02)  # 1 O-H bond with spring
        >>> SpringConstraint("O", "H", 1, 1.02, k_spring=100)  # stronger spring
    """

    element1: str
    element2: str
    count: int | str
    distance: float
    k_spring: float = 50.0


@dataclass
class SpringWaterBoxParameters:
    """
    Parameters for spring-restrained water box generation.

    Generates a water box with spring restraints for MLIP training.
    Unlike frozen constraints, spring restraints allow atoms to move
    while being pulled toward target distances.

    Args:
        output_file: Output LAMMPS data file path.
        model_files: DeepMD model files (e.g., ["graph.000.pb"]).
        box_size: Box dimensions in Angstroms (float for cubic, or 3-tuple).
        n_water: Number of water molecules.
        density: Water density in g/cm3.
        water_model: Water model for initial geometry. Default: "SPC/E"
        spring_constraints: List of SpringConstraint objects.
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
        >>> params = SpringWaterBoxParameters(
        ...     output_file="spring.data",
        ...     model_files=["graph.000.pb"],
        ...     n_water=32,
        ...     spring_constraints=[SpringConstraint("O", "H", 1, 1.02)]
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
    spring_constraints: list[SpringConstraint] = field(default_factory=list)
    constraint_seed: int = 42
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
    tolerance: float = 2.0
    seed: int = 12345
    packmol_executable: str = "packmol"
    elements: list[str] = field(default_factory=lambda: ["O", "H"])
    log: bool = False
    logger: Optional["MLIPLogger"] = None
