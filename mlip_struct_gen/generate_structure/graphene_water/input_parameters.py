"""Input parameters for graphene-water interface generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class GrapheneWaterParameters:
    """
    Parameters for graphene monolayer with water layer generation.

    This dataclass defines all the parameters needed to generate graphene sheets
    with water molecules above them, creating graphene-water interfaces for MD simulations.

    Args:
        size: Surface size as (nx, ny) unit cells.
            - nx: repetitions in x-direction
            - ny: repetitions in y-direction
            Example: (20, 20) creates a 20x20 unit cell graphene sheet

        n_water: Number of water molecules to add above the graphene surface.
            The water box will be sized to achieve the target density.

        output_file: Path to output structure file. Supported formats:
            - ".xyz": XYZ coordinate file
            - ".vasp"/"POSCAR": VASP POSCAR format
            - ".lammps"/".data": LAMMPS data file
            - ".lammpstrj": LAMMPS trajectory format
            File extension determines the output format.

        a: Lattice constant of graphene in Angstroms.
            Default: 2.46 Å (experimental value for graphene)

        thickness: Thickness parameter for graphene sheet in Angstroms.
            Default: 0.0 Å (creates a true 2D sheet)
            Note: Some simulations may require a small non-zero value

        graphene_vacuum: In-plane vacuum spacing around graphene nanoribbon in Angstroms.
            Default: 0.0 Å (no in-plane vacuum)
            Note: This adds vacuum in the x-y plane around the graphene edges

        water_density: Target density of water in g/cm³.
            Default: 1.0 g/cm³ (standard water density at room temperature)

        gap_above_graphene: Gap between the graphene sheet and the bottom
            of the water layer in Angstroms. Default: 0.0 Å

        vacuum_above_water: Vacuum space above the water layer in Angstroms.
            Creates empty space for simulations. Default: 0.0 Å

        water_model: Water model geometry to use.
            Options: "SPC/E", "TIP3P", "TIP4P", "SPC/Fw"
            Default: "SPC/E"

        packmol_executable: Path to packmol executable.
            Default: "packmol" (assumes it's in PATH)

        packmol_tolerance: Packmol tolerance for packing in Angstroms.
            Default: 2.0 Å

        seed: Random seed for reproducible water configurations.
            Default: 12345

        output_format: Output file format override.
            If specified, overrides format detection from file extension.
            Supported: "xyz", "vasp", "poscar", "lammps", "lammps/dpmd", "lammpstrj"

        elements: List of elements defining atom type order for LAMMPS format.
            When specified, atom types are assigned based on the order in this list.
            For example: ["C", "O", "H"] will assign:
            - C = type 1, O = type 2, H = type 3
            Elements not in the structure will still have their masses defined.
            If None (default), uses sequential numbering based on occurrence.
            Only applies when output_format is "lammps" or "lammps/dpmd".

        log: Enable logging output during generation.
            If True and logger is None, creates a new MLIPLogger instance.

        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically.

    Examples:
        Basic graphene-water interface:
        >>> params = GrapheneWaterParameters(
        ...     size=(20, 20),
        ...     n_water=500,
        ...     output_file="graphene_water.data"
        ... )

        Graphene-water with custom gap and density:
        >>> params = GrapheneWaterParameters(
        ...     size=(30, 30),
        ...     n_water=1000,
        ...     water_density=0.997,
        ...     gap_above_graphene=3.3,
        ...     vacuum_above_water=10.0,
        ...     output_file="graphene_water.vasp"
        ... )

        Large-scale graphene-water system:
        >>> params = GrapheneWaterParameters(
        ...     size=(50, 50),
        ...     n_water=2000,
        ...     water_model="TIP3P",
        ...     output_file="graphene_water_large.lammps",
        ...     a=2.45,  # Custom lattice constant
        ...     elements=["C", "O", "H"]
        ... )

    Note:
        - Always generates periodic graphene sheets (PBC in x-y directions)
        - Water molecules are packed using PACKMOL for optimal distribution
        - The graphene sheet is always in the xy-plane at z=0
        - All validation is performed when creating a GrapheneWaterGenerator instance
    """

    size: tuple[int, int]  # (nx, ny) unit cells
    n_water: int
    output_file: str
    a: float = 2.46  # Graphene lattice constant in Angstroms
    thickness: float = 0.0  # Graphene thickness in Angstroms
    graphene_vacuum: float = 0.0  # In-plane vacuum for graphene in Angstroms
    water_density: float = 1.0  # g/cm³
    gap_above_graphene: float = 0.0  # Angstroms
    vacuum_above_water: float = 0.0  # Angstroms
    water_model: str = "SPC/E"
    packmol_executable: str = "packmol"
    packmol_tolerance: float = 2.0
    seed: int = 12345
    output_format: str | None = None
    elements: list[str] | None = None
    log: bool = False
    logger: Optional["MLIPLogger"] = None
