"""Parameter validation for metal surface generation."""

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_parameters import MetalSurfaceParameters

try:
    from ase.data import atomic_numbers
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


def validate_parameters(parameters: "MetalSurfaceParameters") -> None:
    """
    Comprehensive parameter validation and normalization.
    
    Args:
        parameters: Parameters to validate and normalize
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
        ImportError: If ASE is not available
    """
    if not ASE_AVAILABLE:
        raise ImportError(
            "ASE (Atomic Simulation Environment) is required for metal surface generation. "
            "Install with: pip install ase"
        )
    
    # Metal validation
    if not isinstance(parameters.metal, str):
        raise TypeError("metal must be a string")
    
    if not parameters.metal.strip():
        raise ValueError("metal cannot be empty")
    
    # Check if metal is a valid element
    try:
        atomic_number = atomic_numbers[parameters.metal]
    except KeyError:
        raise ValueError(f"Invalid metal element '{parameters.metal}'. Must be a valid element symbol.")
    
    # Check if it's typically an FCC metal (common ones)
    fcc_metals = {
        "Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt", "Pb", "Rh", "Ir", 
        "Ca", "Sr", "Yb", "Ce", "Th", "Ac"
    }
    if parameters.metal not in fcc_metals:
        # Warning but don't fail - user might know what they're doing
        pass
    
    # Miller index validation
    if not isinstance(parameters.miller_index, (tuple, list)):
        raise TypeError("miller_index must be a tuple or list")
    
    if len(parameters.miller_index) != 3:
        raise ValueError("miller_index must have exactly 3 integers (h, k, l)")
    
    if not all(isinstance(x, int) for x in parameters.miller_index):
        raise TypeError("miller_index must contain only integers")
    
    if all(x == 0 for x in parameters.miller_index):
        raise ValueError("miller_index cannot be (0, 0, 0)")
    
    # Normalize miller index to tuple
    parameters.miller_index = tuple(parameters.miller_index)
    
    # Size validation
    if not isinstance(parameters.size, (tuple, list)):
        raise TypeError("size must be a tuple or list")
    
    if len(parameters.size) != 2:
        raise ValueError("size must have exactly 2 integers (nx, ny)")
    
    if not all(isinstance(x, int) for x in parameters.size):
        raise TypeError("size must contain only integers")
    
    if not all(x >= 1 for x in parameters.size):
        raise ValueError("size must contain positive integers")
    
    if any(x > 20 for x in parameters.size):
        raise ValueError("size too large (>20). Consider computational cost.")
    
    # Normalize size to tuple
    parameters.size = tuple(parameters.size)
    
    # Number of layers validation
    if not isinstance(parameters.n_layers, int):
        raise TypeError("n_layers must be an integer")
    
    if parameters.n_layers < 3:
        raise ValueError("n_layers must be at least 3 for proper surface representation")
    
    if parameters.n_layers > 20:
        raise ValueError("n_layers too large (>20). Consider computational cost.")
    
    # Vacuum validation
    if not isinstance(parameters.vacuum, (int, float)):
        raise TypeError("vacuum must be numeric")
    
    if parameters.vacuum < 5.0:
        raise ValueError("vacuum too small (<5 Å). Minimum recommended is 5 Å.")
    
    if parameters.vacuum > 50.0:
        raise ValueError("vacuum too large (>50 Å). Consider computational cost.")
    
    # Output file validation
    if not isinstance(parameters.output_file, str):
        raise TypeError("output_file must be a string")
    
    if not parameters.output_file.strip():
        raise ValueError("output_file cannot be empty")
    
    # Lattice constant validation
    if parameters.lattice_constant is not None:
        if not isinstance(parameters.lattice_constant, (int, float)):
            raise TypeError("lattice_constant must be numeric or None")
        
        if parameters.lattice_constant <= 0:
            raise ValueError("lattice_constant must be positive")
        
        if parameters.lattice_constant < 2.0 or parameters.lattice_constant > 6.0:
            raise ValueError("lattice_constant outside reasonable range (2.0-6.0 Å)")
    
    # Center slab validation
    if not isinstance(parameters.center_slab, bool):
        raise TypeError("center_slab must be a boolean")
    
    # Orthogonalize validation
    if not isinstance(parameters.orthogonalize, bool):
        raise TypeError("orthogonalize must be a boolean")
    
    # Fix bottom layers validation
    if not isinstance(parameters.fix_bottom_layers, int):
        raise TypeError("fix_bottom_layers must be an integer")
    
    if parameters.fix_bottom_layers < 0:
        raise ValueError("fix_bottom_layers must be non-negative")
    
    if parameters.fix_bottom_layers >= parameters.n_layers - 1:
        raise ValueError(
            f"fix_bottom_layers ({parameters.fix_bottom_layers}) must be less than n_layers-1 ({parameters.n_layers-1}). "
            "Need at least 2 free layers."
        )
    
    # Adsorbate validation
    if parameters.add_adsorbate is not None:
        validate_adsorbate_params(parameters.add_adsorbate)
    
    # Supercell validation
    if parameters.supercell is not None:
        if not isinstance(parameters.supercell, (tuple, list)):
            raise TypeError("supercell must be a tuple or list")
        
        if len(parameters.supercell) != 3:
            raise ValueError("supercell must have exactly 3 integers (nx, ny, nz)")
        
        if not all(isinstance(x, int) for x in parameters.supercell):
            raise TypeError("supercell must contain only integers")
        
        if not all(x >= 1 for x in parameters.supercell):
            raise ValueError("supercell must contain positive integers")
        
        if any(x > 5 for x in parameters.supercell):
            raise ValueError("supercell too large (>5). Consider computational cost.")
        
        # Normalize to tuple
        parameters.supercell = tuple(parameters.supercell)
    
    # Output format validation
    if parameters.output_format is not None:
        if not isinstance(parameters.output_format, str):
            raise TypeError("output_format must be a string or None")
        
        valid_formats = ["xyz", "cif", "vasp", "lammps"]
        if parameters.output_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid output_format '{parameters.output_format}'. "
                f"Supported formats: {', '.join(valid_formats)}"
            )
    
    # Logging parameters validation
    if not isinstance(parameters.log, bool):
        raise TypeError("log must be a boolean")
    
    if parameters.logger is not None:
        # Import here to avoid circular imports
        try:
            from ...utils.logger import MLIPLogger
            if not isinstance(parameters.logger, MLIPLogger):
                raise TypeError("logger must be an MLIPLogger instance or None")
        except ImportError:
            raise ImportError("MLIPLogger not available. Check utils.logger module.")


def validate_adsorbate_params(adsorbate_params: dict) -> None:
    """
    Validate adsorbate parameters.
    
    Args:
        adsorbate_params: Adsorbate parameter dictionary
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameter types are incorrect
    """
    if not isinstance(adsorbate_params, dict):
        raise TypeError("add_adsorbate must be a dictionary")
    
    # Required parameter: element
    if "element" not in adsorbate_params:
        raise ValueError("add_adsorbate must contain 'element'")
    
    element = adsorbate_params["element"]
    if not isinstance(element, str):
        raise TypeError("adsorbate element must be a string")
    
    # Check if element is valid
    try:
        atomic_numbers[element]
    except KeyError:
        raise ValueError(f"Invalid adsorbate element '{element}'. Must be a valid element symbol.")
    
    # Optional parameter: position
    if "position" in adsorbate_params:
        position = adsorbate_params["position"]
        if not isinstance(position, str):
            raise TypeError("adsorbate position must be a string")
        
        valid_positions = ["top", "bridge", "hollow", "fcc", "hcp"]
        if position.lower() not in valid_positions:
            raise ValueError(
                f"Invalid adsorbate position '{position}'. "
                f"Valid positions: {', '.join(valid_positions)}"
            )
    
    # Optional parameter: height
    if "height" in adsorbate_params:
        height = adsorbate_params["height"]
        if not isinstance(height, (int, float)):
            raise TypeError("adsorbate height must be numeric")
        
        if height <= 0:
            raise ValueError("adsorbate height must be positive")
        
        if height > 10.0:
            raise ValueError("adsorbate height too large (>10 Å)")
    
    # Optional parameter: coverage
    if "coverage" in adsorbate_params:
        coverage = adsorbate_params["coverage"]
        if not isinstance(coverage, (int, float)):
            raise TypeError("adsorbate coverage must be numeric")
        
        if coverage <= 0 or coverage > 1.0:
            raise ValueError("adsorbate coverage must be between 0 and 1")