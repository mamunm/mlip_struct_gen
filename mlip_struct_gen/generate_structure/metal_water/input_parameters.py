"""Input parameters for metal-water interface generation."""

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class MetalWaterParameters:
    """
    Parameters for metal-water interface generation using ASE and Packmol.
    
    This dataclass defines all the parameters needed to generate metal surfaces
    with water layers on top, creating realistic metal-water interfaces for
    molecular dynamics simulations and surface chemistry studies.
    
    Args:
        # Metal surface parameters
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu", "Ag", "Pd", "Ni", "Al").
            Must be a valid element symbol for an FCC metal.
            
        miller_index: Miller indices for the surface orientation as (h, k, l).
            Common orientations:
            - (1, 1, 1): Close-packed surface
            - (1, 0, 0): Square surface  
            - (1, 1, 0): Rectangular surface
            
        metal_size: Metal surface size as (nx, ny) unit cells in x and y directions.
            Determines the lateral dimensions of the metal surface.
            Valid range: (2, 2) to (10, 10) for computational efficiency.
            
        n_metal_layers: Number of atomic layers in the metal slab.
            Determines the thickness of the metal substrate.
            Valid range: 3-10 layers (minimum 3 for proper surface representation).
            
        lattice_constant: Optional custom lattice constant in Angstroms.
            If None, uses experimental lattice constant from ASE database.
            
        fix_bottom_layers: Number of bottom metal layers to mark as fixed.
            Useful for constraining bulk-like behavior in calculations.
            Valid range: 0 to n_metal_layers-2.
            
        # Water layer parameters
        water_model: Water model to use for molecular geometry and parameters.
            Supported models:
            - "SPC/E": Simple Point Charge/Extended model (default)
            - "TIP3P": Transferable Intermolecular Potential 3-Point model  
            - "TIP4P": Transferable Intermolecular Potential 4-Point model
            
        water_thickness: Thickness of the water layer in Angstroms.
            Determines how much water to add above the metal surface.
            Valid range: 10.0-100.0 Å (typical: 20-40 Å).
            
        n_water_molecules: Optional specific number of water molecules.
            If None, calculated from water_thickness and density.
            Valid range: 10-10000 molecules.
            
        water_density: Water density in g/cm³ for packing calculations.
            Default uses water model's experimental density (~0.997 g/cm³).
            Valid range: 0.5-1.5 g/cm³.
            
        # Interface parameters
        metal_water_gap: Gap between the metal surface and water layer in Angstroms.
            Controls the separation at the metal-water interface.
            Valid range: 1.5-10.0 Å (typical: 2.0-4.0 Å).
            - Small gaps (1.5-2.5 Å): Direct contact, chemisorption
            - Medium gaps (2.5-4.0 Å): Physisorption, hydrogen bonding
            - Large gaps (4.0+ Å): Separated phases
            
        vacuum_above_water: Vacuum space above the water layer in Angstroms.
            Creates empty space above the water for surface calculations.
            Valid range: 5.0-50.0 Å (typical: 10-20 Å).
            
        # Packmol parameters
        packmol_tolerance: Packmol tolerance parameter in Angstroms.
            Controls minimum allowed distance between molecules during packing.
            Valid range: 1.0-3.0 Å (default: 2.0 Å).
            
        packmol_seed: Random seed for Packmol to ensure reproducible results.
            Must be non-negative integer.
            
        packmol_executable: Path or command for Packmol executable.
            Default assumes 'packmol' is in system PATH.
            
        # Output parameters
        output_file: Path to output structure file. Supported formats:
            - ".xyz": XYZ coordinate file
            - ".vasp"/"POSCAR": VASP POSCAR format
            - ".lammps"/".data": LAMMPS data file with topology
            File extension determines the output format.
            
        output_format: Output file format override.
            If specified, overrides format detection from file extension.
            Supported: "xyz", "vasp", "lammps"
            
        # Advanced options
        water_orientation: Strategy for initial water orientation.
            - "random": Random orientations (default)
            - "ordered": Attempt to orient molecules toward surface
            - "bulk": Bulk-like orientations
            
        surface_coverage: Fraction of surface covered by water (0.0-1.0).
            1.0 means water covers entire surface area.
            <1.0 creates partial coverage or droplet-like structures.
            
        add_surface_hydroxyl: Add hydroxyl groups to metal surface atoms.
            Creates more realistic oxidized surface conditions.
            If True, adds OH groups to a fraction of surface metal atoms.
            
        hydroxyl_coverage: Fraction of surface atoms with OH groups (0.0-1.0).
            Only used if add_surface_hydroxyl=True.
            
        # System parameters
        center_system: Whether to center the entire system in the unit cell.
            Recommended for periodic boundary conditions.
            
        log: Enable logging output during generation.
            If True and logger is None, creates a new MLIPLogger instance.
            
        logger: Custom MLIPLogger instance for logging.
    
    Examples:
        Basic gold-water interface:
        >>> params = MetalWaterParameters(
        ...     metal="Au",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(4, 4),
        ...     n_metal_layers=4,
        ...     water_model="SPC/E",
        ...     water_thickness=20.0,
        ...     metal_water_gap=2.5,
        ...     vacuum_above_water=15.0,
        ...     output_file="au_water_interface.xyz"
        ... )
        
        Platinum-water with specific water count:
        >>> params = MetalWaterParameters(
        ...     metal="Pt",
        ...     miller_index=(1, 0, 0),
        ...     metal_size=(3, 3),
        ...     n_metal_layers=5,
        ...     water_model="TIP3P",
        ...     n_water_molecules=200,
        ...     metal_water_gap=3.0,
        ...     vacuum_above_water=12.0,
        ...     fix_bottom_layers=2,
        ...     output_file="pt_water.vasp"
        ... )
        
        Copper-water with surface hydroxylation:
        >>> params = MetalWaterParameters(
        ...     metal="Cu",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(3, 3),
        ...     n_metal_layers=4,
        ...     water_thickness=25.0,
        ...     metal_water_gap=2.0,
        ...     add_surface_hydroxyl=True,
        ...     hydroxyl_coverage=0.25,
        ...     output_file="cu_water_oh.lammps"
        ... )
        
        Large system with partial water coverage:
        >>> params = MetalWaterParameters(
        ...     metal="Ag",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(6, 6),
        ...     n_metal_layers=3,
        ...     water_thickness=30.0,
        ...     surface_coverage=0.7,  # 70% water coverage
        ...     metal_water_gap=3.5,
        ...     water_orientation="ordered",
        ...     output_file="ag_water_droplet.xyz"
        ... )
    
    Note:
        The metal surface is generated first using ASE, then water molecules
        are added using Packmol. The interface is constructed by placing water
        at the specified gap distance above the metal surface.
    """
    
    # Metal surface parameters
    metal: str
    miller_index: Tuple[int, int, int]
    metal_size: Tuple[int, int]
    n_metal_layers: int
    output_file: str
    
    # Optional metal parameters
    lattice_constant: Optional[float] = None
    fix_bottom_layers: int = 0
    
    # Water parameters
    water_model: str = "SPC/E"
    water_thickness: Optional[float] = 20.0
    n_water_molecules: Optional[int] = None
    water_density: Optional[float] = None
    
    # Interface parameters
    metal_water_gap: float = 2.5
    vacuum_above_water: float = 15.0
    
    # Packmol parameters
    packmol_tolerance: float = 2.0
    packmol_seed: int = 12345
    packmol_executable: str = "packmol"
    
    # Output parameters
    output_format: Optional[str] = None
    
    # Advanced options
    water_orientation: str = "random"
    surface_coverage: float = 1.0
    add_surface_hydroxyl: bool = False
    hydroxyl_coverage: float = 0.25
    center_system: bool = True
    
    # Logging
    log: bool = False
    logger: Optional["MLIPLogger"] = None