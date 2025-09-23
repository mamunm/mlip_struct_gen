"""Input parameters for metal surface generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class MetalSurfaceParameters:
    """
    Parameters for FCC(111) metal surface generation using ASE.

    This dataclass defines all the parameters needed to generate FCC(111) metal surfaces
    with specified size and vacuum regions using the Atomic Simulation Environment (ASE).

    Args:
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu", "Ag", "Pd", "Ni", "Al").
            Must be a valid element symbol for an FCC metal.

        size: Surface size as (nx, ny, nz) unit cells.
            - nx, ny: lateral dimensions (repetitions in x and y)
            - nz: number of atomic layers in z-direction
            Example: (4, 4, 4) creates a 4x4 surface with 4 layers

        vacuum: Vacuum space above the surface in Angstroms.
            Creates empty space for surface calculations or adsorbate placement.
            Typical range: 10.0-20.0 Å.

        output_file: Path to output structure file. Supported formats:
            - ".xyz": XYZ coordinate file
            - ".vasp"/"POSCAR": VASP POSCAR format
            - ".lammps"/".data": LAMMPS data file
            File extension determines the output format.

        lattice_constant: Optional custom lattice constant in Angstroms.
            If None, uses experimental lattice constant from database.
            Override for theoretical calculations or specific studies.

        fix_bottom_layers: Number of bottom layers to mark as fixed.
            Useful for constraining bulk-like behavior in MD/optimization.
            Valid range: 0 to nz-1 (must leave at least 1 free layer).
            Default: 0 (no fixed layers).

        orthogonalize: Whether to create an orthogonal unit cell.
            If True, transforms the cell to have orthogonal axes (90° angles).
            Required for LAMMPS and some other simulation packages.
            Default: True.

        output_format: Output file format override.
            If specified, overrides format detection from file extension.
            Supported: "xyz", "vasp", "poscar", "lammps"

        elements: List of elements defining atom type order for LAMMPS format.
            When specified, atom types are assigned based on the order in this list.
            For example: ["Pt", "O", "H", "Na", "Cl"] will assign:
            - Pt = type 1, O = type 2, H = type 3, Na = type 4, Cl = type 5
            Elements not in the structure will still have their masses defined.
            If None (default), uses sequential numbering based on occurrence.
            Only applies when output_format is "lammps".

        log: Enable logging output during surface generation.
            If True and logger is None, creates a new MLIPLogger instance.

        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically.

    Examples:
        Gold (111) surface with 4 layers:
        >>> params = MetalSurfaceParameters(
        ...     metal="Au",
        ...     size=(3, 3, 4),
        ...     vacuum=12.0,
        ...     output_file="au_111.xyz"
        ... )

        Platinum (111) surface with fixed bottom layers:
        >>> params = MetalSurfaceParameters(
        ...     metal="Pt",
        ...     size=(4, 4, 5),
        ...     vacuum=15.0,
        ...     output_file="pt_111.vasp",
        ...     lattice_constant=3.92,
        ...     fix_bottom_layers=2
        ... )

        Copper surface for LAMMPS:
        >>> params = MetalSurfaceParameters(
        ...     metal="Cu",
        ...     size=(5, 5, 6),
        ...     vacuum=10.0,
        ...     output_file="cu_111.data",
        ...     orthogonalize=True
        ... )

    Note:
        - Always generates FCC(111) surfaces (most stable and commonly studied)
        - All validation is performed when creating a MetalSurfaceGenerator instance
        - The ASE library is used for all structure generation and manipulation
    """

    metal: str
    size: tuple[int, int, int]  # (nx, ny, nz)
    vacuum: float
    output_file: str
    lattice_constant: float | None = None
    fix_bottom_layers: int = 0
    orthogonalize: bool = True
    output_format: str | None = None
    elements: list[str] | None = None
    log: bool = False
    logger: Optional["MLIPLogger"] = None
