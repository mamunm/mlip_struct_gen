"""Input parameters for metal surface generation."""

from dataclasses import dataclass
from typing import Tuple, Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class MetalSurfaceParameters:
    """
    Parameters for metal surface generation using ASE.
    
    This dataclass defines all the parameters needed to generate metal surfaces
    with specified Miller indices, size, and vacuum regions using the Atomic
    Simulation Environment (ASE).
    
    Args:
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu", "Ag", "Pd", "Ni", "Al").
            Must be a valid element symbol for an FCC metal.
            
        miller_index: Miller indices for the surface orientation as (h, k, l).
            Common orientations:
            - (1, 1, 1): Close-packed surface
            - (1, 0, 0): Square surface  
            - (1, 1, 0): Rectangular surface
            - (2, 1, 1): Stepped surface
            
        size: Surface size as (nx, ny) unit cells in x and y directions.
            Determines the lateral dimensions of the surface.
            Valid range: (1, 1) to (20, 20) for computational efficiency.
            
        n_layers: Number of atomic layers in the z-direction.
            Determines the thickness of the metal slab.
            Valid range: 3-20 layers (minimum 3 for proper surface representation).
            
        vacuum: Vacuum space above the surface in Angstroms.
            Creates empty space for surface calculations or adsorbate placement.
            Valid range: 5.0-50.0 Å (typical: 10-15 Å).
            
        output_file: Path to output structure file. Supported formats:
            - ".xyz": XYZ coordinate file
            - ".cif": Crystallographic Information File
            - ".vasp"/"POSCAR": VASP POSCAR format
            - ".lammps": LAMMPS data file
            File extension determines the output format.
            
        lattice_constant: Optional custom lattice constant in Angstroms.
            If None, uses experimental lattice constant from ASE database.
            Override for theoretical calculations or specific studies.
            Valid range: 2.0-6.0 Å (typical FCC metals: 3.5-4.2 Å).
            
        center_slab: Whether to center the slab in the unit cell.
            If True, places vacuum equally above and below the slab.
            If False, places all vacuum above the slab.
            
        fix_bottom_layers: Number of bottom layers to mark as fixed.
            Useful for constraining bulk-like behavior in calculations.
            Valid range: 0 to n_layers-2 (must leave at least 2 free layers).
            
        add_adsorbate: Optional adsorbate specification.
            If provided, should be a dictionary with:
            - "element": Adsorbate element (e.g., "O", "H", "CO")
            - "position": Adsorption site ("top", "bridge", "hollow", "fcc", "hcp")  
            - "height": Height above surface in Angstroms (default: 2.0)
            - "coverage": Coverage fraction (0.0-1.0, default: 0.25)
            
        supercell: Optional supercell expansion as (nx, ny, nz).
            Creates larger surfaces by replicating the unit cell.
            Applied after initial surface generation.
            
        output_format: Output file format override.
            If specified, overrides format detection from file extension.
            Supported: "xyz", "cif", "vasp", "lammps"
            
        orthogonalize: Whether to create an orthogonal unit cell.
            If True, transforms the cell to have orthogonal axes (90° angles).
            Useful for LAMMPS and some other simulation packages that prefer
            orthogonal cells. Default is False.
            
        log: Enable logging output during surface generation.
            If True and logger is None, creates a new MLIPLogger instance.
            
        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically.
    
    Examples:
        Gold (111) surface with 4 layers:
        >>> params = MetalSurfaceParameters(
        ...     metal="Au",
        ...     miller_index=(1, 1, 1),
        ...     size=(3, 3),
        ...     n_layers=4,
        ...     vacuum=12.0,
        ...     output_file="au_111.xyz"
        ... )
        
        Platinum (100) surface with custom lattice constant:
        >>> params = MetalSurfaceParameters(
        ...     metal="Pt",
        ...     miller_index=(1, 0, 0),
        ...     size=(4, 4),
        ...     n_layers=5,
        ...     vacuum=15.0,
        ...     output_file="pt_100.vasp",
        ...     lattice_constant=3.92,
        ...     fix_bottom_layers=2
        ... )
        
        Copper surface with oxygen adsorbate:
        >>> params = MetalSurfaceParameters(
        ...     metal="Cu",
        ...     miller_index=(1, 1, 1),
        ...     size=(2, 2),
        ...     n_layers=4,
        ...     vacuum=12.0,
        ...     output_file="cu_111_o.cif",
        ...     add_adsorbate={
        ...         "element": "O",
        ...         "position": "fcc",
        ...         "height": 2.1,
        ...         "coverage": 0.25
        ...     }
        ... )
        
        Large aluminum surface with supercell:
        >>> params = MetalSurfaceParameters(
        ...     metal="Al",
        ...     miller_index=(1, 1, 0),
        ...     size=(2, 2),
        ...     n_layers=3,
        ...     vacuum=10.0,
        ...     output_file="al_110_large.xyz",
        ...     supercell=(2, 2, 1)
        ... )
    
    Note:
        All validation is performed when creating a MetalSurfaceGenerator instance,
        not when creating this parameters object. The ASE library is used for all
        structure generation and manipulation.
    """
    
    metal: str
    miller_index: Tuple[int, int, int]
    size: Tuple[int, int]
    n_layers: int
    vacuum: float
    output_file: str
    lattice_constant: Optional[float] = None
    center_slab: bool = True
    fix_bottom_layers: int = 0
    add_adsorbate: Optional[dict] = None
    supercell: Optional[Tuple[int, int, int]] = None
    output_format: Optional[str] = None
    orthogonalize: bool = False
    log: bool = False
    logger: Optional["MLIPLogger"] = None