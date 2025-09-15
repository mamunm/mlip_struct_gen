"""Input parameters for metal-water interface generation."""

from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class MetalWaterParameters:
    """
    Parameters for FCC(111) metal surface with water layer generation.

    This dataclass defines all the parameters needed to generate FCC(111) metal surfaces
    with water molecules above them, creating metal-water interfaces for MD simulations.

    Args:
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu", "Ag", "Pd", "Ni", "Al").
            Must be a valid element symbol for an FCC metal.

        size: Surface size as (nx, ny, nz) unit cells.
            - nx, ny: lateral dimensions (repetitions in x and y)
            - nz: number of atomic layers in z-direction
            Example: (4, 4, 4) creates a 4x4 surface with 4 layers

        n_water: Number of water molecules to add above the metal surface.
            The water box will be sized to achieve the target density.

        density: Target density of water in g/cm³.
            Default: 1.0 g/cm³ (standard water density at room temperature)

        gap_above_metal: Gap between the top of the metal surface and the bottom
            of the water layer in Angstroms. Default: 3.0 Å

        vacuum_above_water: Vacuum space above the water layer in Angstroms.
            Creates empty space for simulations. Default: 0.0 Å

        output_file: Path to output structure file. Supported formats:
            - ".xyz": XYZ coordinate file
            - ".vasp"/"POSCAR": VASP POSCAR format
            - ".lammps"/".data": LAMMPS data file
            File extension determines the output format.

        water_model: Water model geometry to use.
            Options: "SPC/E", "TIP3P", "TIP4P"
            Default: "SPC/E"

        lattice_constant: Optional custom lattice constant in Angstroms.
            If None, uses experimental lattice constant from database.
            Override for theoretical calculations or specific studies.

        fix_bottom_layers: Number of bottom metal layers to mark as fixed.
            Useful for constraining bulk-like behavior in MD/optimization.
            Valid range: 0 to nz-1 (must leave at least 1 free layer).
            Default: 0 (no fixed layers).

        packmol_executable: Path to packmol executable.
            Default: "packmol" (assumes it's in PATH)

        packmol_tolerance: Packmol tolerance for packing in Angstroms.
            Default: 2.0 Å

        seed: Random seed for reproducible water configurations.
            Default: 12345

        output_format: Output file format override.
            If specified, overrides format detection from file extension.
            Supported: "xyz", "vasp", "poscar", "lammps"

        log: Enable logging output during generation.
            If True and logger is None, creates a new MLIPLogger instance.

        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically.

    Examples:
        Basic Pt-water interface:
        >>> params = MetalWaterParameters(
        ...     metal="Pt",
        ...     size=(4, 4, 4),
        ...     n_water=100,
        ...     output_file="pt_water.data"
        ... )

        Gold-water interface with custom parameters:
        >>> params = MetalWaterParameters(
        ...     metal="Au",
        ...     size=(5, 5, 6),
        ...     n_water=200,
        ...     density=0.997,
        ...     gap_above_metal=3.5,
        ...     vacuum_above_water=10.0,
        ...     output_file="au_water.vasp",
        ...     fix_bottom_layers=2
        ... )

        Copper-water interface for large-scale MD:
        >>> params = MetalWaterParameters(
        ...     metal="Cu",
        ...     size=(10, 10, 8),
        ...     n_water=500,
        ...     water_model="TIP3P",
        ...     output_file="cu_water_large.lammps",
        ...     lattice_constant=3.615,
        ...     fix_bottom_layers=3
        ... )

    Note:
        - Always generates FCC(111) surfaces (most stable and commonly studied)
        - Water molecules are packed using PACKMOL for optimal distribution
        - The metal surface is always orthogonal for LAMMPS compatibility
        - All validation is performed when creating a MetalWaterGenerator instance
    """

    metal: str
    size: Tuple[int, int, int]  # (nx, ny, nz)
    n_water: int
    output_file: str
    density: float = 1.0  # g/cm³
    gap_above_metal: float = 0.0  # Angstroms
    vacuum_above_water: float = 0.0  # Angstroms
    water_model: str = "SPC/E"
    lattice_constant: Optional[float] = None
    fix_bottom_layers: int = 0
    packmol_executable: str = "packmol"
    packmol_tolerance: float = 2.0
    seed: int = 12345
    output_format: Optional[str] = None
    log: bool = False
    logger: Optional["MLIPLogger"] = None