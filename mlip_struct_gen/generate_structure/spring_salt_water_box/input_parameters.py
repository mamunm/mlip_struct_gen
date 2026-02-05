"""Input parameters for spring-restrained salt water box generation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger

# Re-export SpringConstraint from spring_water_box
from ..spring_water_box.input_parameters import SpringConstraint

__all__ = ["SpringConstraint", "SpringSaltWaterBoxParameters"]


@dataclass
class SpringSaltWaterBoxParameters:
    """
    Parameters for spring-restrained salt water box generation.

    Generates a salt water box with spring restraints for MLIP training.
    Unlike frozen constraints, spring restraints allow atoms to move
    while being pulled toward target distances.

    Supports spring restraints on:
    - O-H bonds (intramolecular water)
    - O-O distances (intermolecular water)
    - Ion-water distances (Na-O, Cl-O, etc.)
    - Ion-ion distances (Na-Cl, etc.)

    Args:
        output_file: Output LAMMPS data file path.
        model_files: DeepMD model files (e.g., ["graph.000.pb"]).
        box_size: Box dimensions in Angstroms (float for cubic, or 3-tuple).
        n_water: Number of water molecules.
        density: Water density in g/cm3.
        salt_type: Type of salt (NaCl, KCl, LiCl, CaCl2, MgCl2, etc.)
        n_salt: Number of salt formula units.
        include_salt_volume: Account for ion volume when computing box/water count.
        water_model: Water model for initial geometry. Default: "SPC/E"
        spring_constraints: List of SpringConstraint objects.
        constraint_seed: Seed for random selection of constrained pairs. Default: 42
        minimize: Add energy minimization before MD.
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
        elements: Element order for LAMMPS. Default: ["O", "H", "Na", "Cl"]
        log: Enable logging. Default: False
        logger: Custom MLIPLogger instance. Default: None
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

    # Salt parameters
    salt_type: str = "NaCl"
    n_salt: int = 0
    include_salt_volume: bool = False

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
    elements: list[str] = field(default_factory=lambda: ["O", "H", "Na", "Cl"])
    log: bool = False
    logger: Optional["MLIPLogger"] = None
