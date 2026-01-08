"""Input parameters for metal-salt-water interface generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class MetalSaltWaterParameters:
    """
    Parameters for FCC(111) metal surface with salt water layer generation.

    This dataclass defines all the parameters needed to generate FCC(111) metal surfaces
    with salt water (electrolyte) above them, creating metal-electrolyte interfaces for MD simulations.

    Args:
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu", "Ag", "Pd", "Ni", "Al").
            Must be a valid element symbol for an FCC metal.

        size: Surface size as (nx, ny, nz) unit cells.
            - nx, ny: lateral dimensions (repetitions in x and y)
            - nz: number of atomic layers in z-direction
            Example: (4, 4, 4) creates a 4x4 surface with 4 layers

        n_water: Number of water molecules to add above the metal surface.
            The water box will be sized to achieve the target density.

        salt_type: Type of salt to add. Supported salts:
            - "NaCl": Sodium Chloride
            - "KCl": Potassium Chloride
            - "LiCl": Lithium Chloride
            - "CaCl2": Calcium Chloride (2:1 stoichiometry)
            - "MgCl2": Magnesium Chloride (2:1 stoichiometry)
            - "NaBr": Sodium Bromide
            - "KBr": Potassium Bromide
            - "CsCl": Cesium Chloride

        n_salt: Number of salt formula units to add.
            For example, n_salt=5 for NaCl adds 5 Na+ and 5 Cl- ions.
            For CaCl2, it adds 5 Ca2+ and 10 Cl- ions.

        include_salt_volume: Whether to account for ion volume when calculating
            water box dimensions. If True, reduces water molecules to maintain
            target density considering ion volume. Default: False

        density: Target density of the solution in g/cm^3.
            Default: 1.0 g/cm^3 (standard water density at room temperature)
            Note: This is the total solution density if include_salt_volume=True,
            otherwise it's just the water density.

        gap: Gap between the top of the metal surface and the bottom
            of the solution layer in Angstroms. Default: 0.0 Angstroms

        vacuum_above_water: Vacuum space above the solution layer in Angstroms.
            Creates empty space for simulations. Default: 0.0 Angstroms

        output_file: Path to output structure file. Supported formats:
            - ".xyz": XYZ coordinate file
            - ".vasp"/"POSCAR": VASP POSCAR format
            - ".lammps"/".data": LAMMPS data file
            File extension determines the output format.

        water_model: Water model geometry to use.
            Options: "SPC/E", "TIP3P", "TIP4P", "SPC/Fw"
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
            Default: 2.0 Angstroms

        seed: Random seed for reproducible salt-water configurations.
            Default: 12345

        no_salt_zone: Fraction of the salt-water box height (0.0 to <0.5) where
            salt ions are excluded at both top and bottom. Creates symmetric
            exclusion zones. For example, no_salt_zone=0.2 means ions can only
            be placed in the middle 60% of the box (excluding bottom 20% and top 20%).
            Must be less than 0.5 to leave room for ions.
            Default: 0.2

        save_artifacts: If True, save intermediate files (PACKMOL input, molecule
            files, solution box, etc.) to a directory named "<output_file>_artifacts"
            alongside the output file. Useful for debugging or reproducing results.
            Default: False

        output_format: Output file format override.
            If specified, overrides format detection from file extension.
            Supported: "xyz", "vasp", "poscar", "lammps"

        elements: List of elements defining atom type order for LAMMPS format.
            When specified, atom types are assigned based on the order in this list.
            For example: ["Pt", "O", "H", "Na", "Cl"] will assign:
            - Pt = type 1, O = type 2, H = type 3, Na = type 4, Cl = type 5
            Elements not in the structure will still have their masses defined.
            If None (default), uses sequential numbering based on occurrence.
            Only applies when output_format is "lammps" or "lammps/dpmd".

        log: Enable logging output during generation.
            If True and logger is None, creates a new MLIPLogger instance.

        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically.

    Examples:
        Basic Pt-NaCl-water interface:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Pt",
        ...     size=(4, 4, 4),
        ...     n_water=100,
        ...     salt_type="NaCl",
        ...     n_salt=10,
        ...     output_file="pt_nacl_water.data"
        ... )

        Gold-CaCl2-water with custom parameters:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Au",
        ...     size=(5, 5, 6),
        ...     n_water=200,
        ...     salt_type="CaCl2",
        ...     n_salt=8,
        ...     include_salt_volume=True,
        ...     density=1.05,
        ...     gap=3.5,
        ...     vacuum_above_water=10.0,
        ...     output_file="au_cacl2_water.vasp",
        ...     fix_bottom_layers=2
        ... )

        Copper-KCl-water interface for large-scale MD:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Cu",
        ...     size=(10, 10, 8),
        ...     n_water=500,
        ...     salt_type="KCl",
        ...     n_salt=30,
        ...     water_model="TIP3P",
        ...     output_file="cu_kcl_water_large.lammps",
        ...     lattice_constant=3.615,
        ...     fix_bottom_layers=3
        ... )

    Note:
        - Always generates FCC(111) surfaces (most stable and commonly studied)
        - Salt ions and water molecules are packed using PACKMOL for optimal distribution
        - The metal surface is always orthogonal for LAMMPS compatibility
        - All validation is performed when creating a MetalSaltWaterGenerator instance
    """

    metal: str
    size: tuple[int, int, int]  # (nx, ny, nz)
    n_water: int
    salt_type: str
    n_salt: int
    output_file: str
    include_salt_volume: bool = False
    density: float = 1.0  # g/cm^3
    gap: float = 0.0  # Angstroms
    vacuum_above_water: float = 0.0  # Angstroms
    water_model: str = "SPC/E"
    lattice_constant: float | None = None
    fix_bottom_layers: int = 0
    packmol_executable: str = "packmol"
    packmol_tolerance: float = 2.0
    seed: int = 12345
    no_salt_zone: float = 0.2
    save_artifacts: bool = False
    output_format: str | None = None
    elements: list[str] | None = None
    log: bool = False
    logger: Optional["MLIPLogger"] = None
