"""Input parameters for water box generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class WaterBoxGeneratorParameters:
    """
    Parameters for water box generation using Packmol.

    This dataclass defines all the parameters needed to generate a water box
    using the Packmol molecular packing software. Parameters are validated
    when the WaterBoxGenerator is created.

    Args:
        box_size: Box dimensions in Angstroms. Can be:
            - Single number (float): Creates cubic box with size x size x size
            - Tuple/list of 3 numbers: Creates rectangular box (x, y, z)
            - None: Automatically computed from n_water and density
            Valid range: 5.0 - 1000.0 Å per dimension
            Note: Either box_size OR n_water must be provided

        output_file: Path to output file. Must include file extension (e.g., '.xyz').
            The output directory will be created if it doesn't exist.

        water_model: Water model to use for molecular geometry and parameters.
            Supported models:
            - "SPC/E": Simple Point Charge/Extended model (default)
            - "TIP3P": Transferable Intermolecular Potential 3-Point model
            - "TIP4P": Transferable Intermolecular Potential 4-Point model

        n_water: Number of water molecules to pack. If None, calculated
            automatically from box_size and density. Can be combined with
            density to compute box_size, or with box_size to pack exact
            number of molecules.
            Valid range: 1 - 1,000,000 molecules

        density: Water density in g/cm³. If None, uses the default density
            for the chosen water model. Can be combined with:
            - box_size: to calculate n_water at specified density
            - n_water: to compute box_size for exact n_water at density
            Valid range: 0.1 - 5.0 g/cm³
            Model defaults: SPC/E=0.997, TIP3P=0.997, TIP4P=0.997 g/cm³

        tolerance: Packmol tolerance parameter in Angstroms. Controls the
            minimum allowed distance between atoms during packing.
            Valid range: 0.1 - 10.0 Å (typical: 1.0 - 3.0 Å)

        seed: Random seed for Packmol to ensure reproducible results.
            Must be non-negative integer.

        packmol_executable: Path or command for Packmol executable.
            Default assumes 'packmol' is in system PATH.
            Install with: conda install -c conda-forge packmol

        output_format: Output file format for the generated structure.
            Default: "lammps" for LAMMPS data file with full atom style.
            Supported formats:
            - "xyz": Standard XYZ format
            - "lammps": LAMMPS data file with bonds and angles
            - "poscar": VASP POSCAR format with descending element ordering
            File extension will be added automatically: .xyz, .data, or no extension for POSCAR

        log: Enable logging output during water box generation.
            If True and logger is None, creates a new MLIPLogger instance.

        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically. If log=False, this
            parameter is ignored.

    Examples:
        Create a 20x20x20 Å cubic box with SPC/E water:
        >>> params = WaterBoxGeneratorParameters(
        ...     box_size=20.0,
        ...     output_file="water_box.xyz"
        ... )

        Create a rectangular box with TIP3P (uses TIP3P default density):
        >>> params = WaterBoxGeneratorParameters(
        ...     box_size=(30.0, 25.0, 20.0),
        ...     output_file="water_box.xyz",
        ...     water_model="TIP3P"
        ... )

        Override density for custom packing:
        >>> params = WaterBoxGeneratorParameters(
        ...     box_size=(30.0, 25.0, 20.0),
        ...     output_file="water_box.xyz",
        ...     water_model="TIP3P",
        ...     density=1.0  # Override TIP3P default
        ... )

        Create box with specific number of molecules (uses default density):
        >>> params = WaterBoxGeneratorParameters(
        ...     output_file="water_box.xyz",
        ...     n_water=500,
        ...     tolerance=1.5,
        ...     seed=42
        ... )

        Create box with specific n_molecules at custom density (box computed):
        >>> params = WaterBoxGeneratorParameters(
        ...     output_file="water_box.xyz",
        ...     n_water=500,
        ...     density=1.1,  # Custom density, box size computed
        ...     tolerance=1.5,
        ...     seed=42
        ... )

        Enable logging with default logger:
        >>> params = WaterBoxGeneratorParameters(
        ...     box_size=20.0,
        ...     output_file="water_box.xyz",
        ...     log=True
        ... )

        Use custom logger instance:
        >>> from mlip_struct_gen.utils import MLIPLogger
        >>> custom_logger = MLIPLogger()
        >>> params = WaterBoxGeneratorParameters(
        ...     box_size=20.0,
        ...     output_file="water_box.xyz",
        ...     log=True,
        ...     logger=custom_logger
        ... )

    Note:
        All validation is performed when creating a WaterBoxGenerator instance,
        not when creating this parameters object. The box_size will be
        automatically normalized to a 3-element tuple during validation.
    """

    output_file: str
    box_size: float | tuple[float, float, float] | None = None
    water_model: str = "SPC/E"
    n_water: int | None = None
    density: float | None = None
    tolerance: float = 2.0
    seed: int = 12345
    packmol_executable: str = "packmol"
    output_format: str = "lammps"
    log: bool = False
    logger: Optional["MLIPLogger"] = None
