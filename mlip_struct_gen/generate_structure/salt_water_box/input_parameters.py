"""Input parameters for salt water box generation."""

from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class SaltWaterBoxGeneratorParameters:
    """
    Parameters for salt water box generation using Packmol.
    
    This dataclass defines all the parameters needed to generate a salt water box
    using the Packmol molecular packing software. Parameters are validated
    when the SaltWaterBoxGenerator is created.
    
    Args:
        box_size: Box dimensions in Angstroms. Can be:
            - Single number (float): Creates cubic box with size x size x size
            - Tuple/list of 3 numbers: Creates rectangular box (x, y, z)
            - None: Automatically computed from n_water_molecules and density
            Valid range: 5.0 - 1000.0 Å per dimension
            
        output_file: Path to output file. Must include file extension (e.g., '.xyz').
            The output directory will be created if it doesn't exist.
            
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
            Valid range: 1 - 10000 molecules
            
        water_model: Water model to use for molecular geometry and parameters.
            Supported models:
            - "SPC/E": Simple Point Charge/Extended model (default)
            - "TIP3P": Transferable Intermolecular Potential 3-Point model  
            - "TIP4P": Transferable Intermolecular Potential 4-Point model
            
        n_water_molecules: Number of water molecules to pack. If None, calculated
            automatically from box_size using the chosen water model's density
            adjusted for salt volume. Cannot be specified together with 
            water_density parameter.
            Valid range: 1 - 1,000,000 molecules
            
        water_density: Water density in g/cm³. If None, uses the default density
            for the chosen water model. If specified, overrides the model's
            default density for molecule count calculation. Cannot be 
            specified together with n_water_molecules parameter.
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
            Supported formats: "xyz", "lammps"
            File extension will be added automatically: .xyz, .data
            
        neutralize: Whether to ensure the system is electrically neutral by
            adjusting ion counts. Default is True.
            
        add_salt_volume: Whether to account for salt ion volume when calculating
            water molecules. If False, water molecules are calculated based on
            the entire box volume without subtracting ion volumes. Default is False.
            
        custom_salt_params: Optional custom parameters for salt not in the
            built-in library. Should contain:
            - "name": Salt name
            - "cation": Dict with element, charge, mass
            - "anion": Dict with element, charge, mass
            - "stoichiometry": Optional dict with cation/anion ratio
            
        log: Enable logging output during salt water box generation.
            If True and logger is None, creates a new MLIPLogger instance.
            
        logger: Custom MLIPLogger instance for logging. If None and log=True,
            a new logger will be created automatically. If log=False, this
            parameter is ignored.
    
    Examples:
        Create a 30x30x30 Å box with 10 NaCl molecules:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     box_size=30.0,
        ...     output_file="nacl_water.xyz",
        ...     salt_type="NaCl",
        ...     n_salt_molecules=10
        ... )
        
        Create a rectangular box with 20 KCl molecules and TIP3P water:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     box_size=(40.0, 30.0, 30.0),
        ...     output_file="kcl_water.xyz",
        ...     salt_type="KCl",
        ...     n_salt_molecules=20,
        ...     water_model="TIP3P"
        ... )
        
        Create box with specific number of water molecules and 5 CaCl2:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     output_file="cacl2_water.xyz",
        ...     salt_type="CaCl2",
        ...     n_salt_molecules=5,
        ...     n_water_molecules=1000,
        ...     tolerance=2.0,
        ...     seed=42
        ... )
        
        Use custom salt parameters:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     box_size=30.0,
        ...     output_file="custom_salt.xyz",
        ...     custom_salt_params={
        ...         "name": "Custom Salt",
        ...         "cation": {"element": "Cs", "charge": 1.0, "mass": 132.905},
        ...         "anion": {"element": "I", "charge": -1.0, "mass": 126.904}
        ...     },
        ...     n_salt_molecules=8
        ... )
    
    Note:
        All validation is performed when creating a SaltWaterBoxGenerator instance,
        not when creating this parameters object. The box_size will be
        automatically normalized to a 3-element tuple during validation.
    """
    
    output_file: str
    box_size: Optional[Union[float, Tuple[float, float, float]]] = None
    salt_type: Optional[str] = "NaCl"
    n_salt_molecules: int = 5
    water_model: str = "SPC/E"
    n_water_molecules: Optional[int] = None
    water_density: Optional[float] = None
    tolerance: float = 2.0
    seed: int = 12345
    packmol_executable: str = "packmol"
    output_format: str = "lammps"
    neutralize: bool = True
    add_salt_volume: bool = False
    custom_salt_params: Optional[Dict] = None
    log: bool = False
    logger: Optional["MLIPLogger"] = None