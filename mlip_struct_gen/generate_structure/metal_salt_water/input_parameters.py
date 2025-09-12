"""Input parameters for metal-salt-water interface generation."""

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class MetalSaltWaterParameters:
    """
    Parameters for metal-salt-water interface generation using ASE and Packmol.
    
    This dataclass combines the functionality of both metal-water and salt-water
    generation to create realistic metal-electrolyte interfaces for molecular
    dynamics simulations and electrochemistry studies.
    
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
            
        # Salt and water parameters
        salt_type: Type of salt to add. Supported salts:
            - "NaCl": Sodium Chloride
            - "NaBr": Sodium Bromide
            - "KCl": Potassium Chloride
            - "KBr": Potassium Bromide
            - "LiCl": Lithium Chloride
            - "CaCl2": Calcium Chloride (2:1 stoichiometry)
            - "MgCl2": Magnesium Chloride (2:1 stoichiometry)
            
        n_salt_molecules: Number of salt formula units to add.
            For example, n_salt_molecules=5 for NaCl adds 5 Na+ and 5 Cl- ions.
            For CaCl2, it adds 5 Ca2+ and 10 Cl- ions.
            Valid range: 1 - 1000 molecules
            
        water_model: Water model to use for molecular geometry and parameters.
            Supported models:
            - "SPC/E": Simple Point Charge/Extended model (default)
            - "TIP3P": Transferable Intermolecular Potential 3-Point model  
            - "TIP4P": Transferable Intermolecular Potential 4-Point model
            
        solution_thickness: Thickness of the solution layer in Angstroms.
            Determines how much solution to add above the metal surface.
            Valid range: 15.0-100.0 Å (typical: 25-50 Å).
            
        n_water_molecules: Optional specific number of water molecules.
            If None, calculated from solution_thickness, density, and ion volume.
            Valid range: 10-10000 molecules.
            
        water_density: Water density in g/cm³ for packing calculations.
            Default uses water model's experimental density (~0.997 g/cm³).
            Valid range: 0.5-1.5 g/cm³.
            
        # Interface parameters
        metal_solution_gap: Gap between the metal surface and solution layer in Angstroms.
            Controls the separation at the metal-solution interface.
            Valid range: 1.5-10.0 Å (typical: 2.0-4.0 Å).
            - Small gaps (1.5-2.5 Å): Direct contact, strong interactions
            - Medium gaps (2.5-4.0 Å): Physisorption, hydrogen bonding
            - Large gaps (4.0+ Å): Separated phases
            
        vacuum_above_solution: Vacuum space above the solution layer in Angstroms.
            Creates empty space above the solution for surface calculations.
            Valid range: 5.0-50.0 Å (typical: 10-20 Å).
            
        # Advanced salt parameters
        neutralize: Whether to ensure the system is electrically neutral by
            adjusting ion counts. Default is True.
            
        custom_salt_params: Optional custom parameters for salt not in the
            built-in library. Should contain:
            - "name": Salt name
            - "cation": Dict with element, charge, mass, vdw_radius
            - "anion": Dict with element, charge, mass, vdw_radius
            - "stoichiometry": Optional dict with cation/anion ratio
            
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
            Supported: "xyz", "vasp", "lammps-data"
            
        # Advanced surface options
        surface_coverage: Fraction of surface covered by solution (0.0-1.0).
            1.0 means solution covers entire surface area.
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
        Basic platinum-NaCl-water interface:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Pt",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(4, 4),
        ...     n_metal_layers=4,
        ...     salt_type="NaCl",
        ...     n_salt_molecules=10,
        ...     water_model="SPC/E",
        ...     solution_thickness=25.0,
        ...     metal_solution_gap=2.5,
        ...     vacuum_above_solution=15.0,
        ...     output_file="pt_nacl_water_interface.xyz"
        ... )
        
        Gold-CaCl2-water with specific water count:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Au",
        ...     miller_index=(1, 0, 0),
        ...     metal_size=(3, 3),
        ...     n_metal_layers=5,
        ...     salt_type="CaCl2",
        ...     n_salt_molecules=5,  # 5 Ca2+ + 10 Cl-
        ...     water_model="TIP3P",
        ...     n_water_molecules=300,
        ...     metal_solution_gap=3.0,
        ...     vacuum_above_solution=12.0,
        ...     fix_bottom_layers=2,
        ...     output_file="au_cacl2_water.vasp"
        ... )
        
        Copper-KCl-water with surface hydroxylation:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Cu",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(3, 3),
        ...     n_metal_layers=4,
        ...     salt_type="KCl",
        ...     n_salt_molecules=8,
        ...     solution_thickness=30.0,
        ...     metal_solution_gap=2.0,
        ...     add_surface_hydroxyl=True,
        ...     hydroxyl_coverage=0.25,
        ...     output_file="cu_kcl_water_oh.lammps"
        ... )
        
        Large system with partial surface coverage:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Ag",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(6, 6),
        ...     n_metal_layers=3,
        ...     salt_type="NaBr",
        ...     n_salt_molecules=15,
        ...     solution_thickness=35.0,
        ...     surface_coverage=0.7,  # 70% solution coverage
        ...     metal_solution_gap=3.5,
        ...     output_file="ag_nabr_water_droplet.xyz"
        ... )
        
        Custom salt parameters:
        >>> params = MetalSaltWaterParameters(
        ...     metal="Pt",
        ...     miller_index=(1, 1, 1),
        ...     metal_size=(4, 4),
        ...     n_metal_layers=4,
        ...     custom_salt_params={
        ...         "name": "Cesium Iodide",
        ...         "cation": {"element": "Cs", "charge": 1.0, "mass": 132.905, "vdw_radius": 3.4},
        ...         "anion": {"element": "I", "charge": -1.0, "mass": 126.904, "vdw_radius": 2.2}
        ...     },
        ...     n_salt_molecules=8,
        ...     solution_thickness=25.0,
        ...     output_file="pt_csi_water.xyz"
        ... )
    
    Note:
        The metal surface is generated first using ASE, then the salt-water solution
        is added using Packmol. The interface is constructed by placing the solution
        at the specified gap distance above the metal surface, creating realistic
        electrode-electrolyte interfaces for electrochemical simulations.
    """
    
    # Metal surface parameters
    metal: str
    miller_index: Tuple[int, int, int]
    metal_size: Tuple[int, int]
    n_metal_layers: int
    output_file: str
    
    # Salt and water parameters
    salt_type: Optional[str] = "NaCl"
    n_salt_molecules: int = 5
    water_model: str = "SPC/E"
    
    # Optional metal parameters
    lattice_constant: Optional[float] = None
    fix_bottom_layers: int = 0
    
    # Solution parameters
    solution_thickness: Optional[float] = 25.0
    n_water_molecules: Optional[int] = None
    water_density: Optional[float] = None
    
    # Interface parameters
    metal_solution_gap: float = 2.5
    vacuum_above_solution: float = 15.0
    
    # Advanced salt parameters
    neutralize: bool = True
    custom_salt_params: Optional[Dict] = None
    
    # Packmol parameters
    packmol_tolerance: float = 2.0
    packmol_seed: int = 12345
    packmol_executable: str = "packmol"
    
    # Output parameters
    output_format: Optional[str] = None
    
    # Advanced surface options
    surface_coverage: float = 1.0
    add_surface_hydroxyl: bool = False
    hydroxyl_coverage: float = 0.25
    center_system: bool = True
    
    # Logging
    log: bool = False
    logger: Optional["MLIPLogger"] = None