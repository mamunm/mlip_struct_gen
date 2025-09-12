"""Parameter validation for metal-water interface generation."""

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_parameters import MetalWaterParameters

try:
    from ase.data import atomic_numbers
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

from ..templates.water_models import get_water_model


def validate_parameters(parameters: "MetalWaterParameters") -> None:
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
    
    # Metal size validation
    if not isinstance(parameters.metal_size, (tuple, list)):
        raise TypeError("metal_size must be a tuple or list")
    
    if len(parameters.metal_size) != 2:
        raise ValueError("metal_size must have exactly 2 integers (nx, ny)")
    
    if not all(isinstance(x, int) for x in parameters.metal_size):
        raise TypeError("metal_size must contain only integers")
    
    if not all(x >= 2 for x in parameters.metal_size):
        raise ValueError("metal_size must contain integers >= 2")
    
    if any(x > 10 for x in parameters.metal_size):
        raise ValueError("metal_size too large (>10). Consider computational cost.")
    
    # Normalize size to tuple
    parameters.metal_size = tuple(parameters.metal_size)
    
    # Number of metal layers validation
    if not isinstance(parameters.n_metal_layers, int):
        raise TypeError("n_metal_layers must be an integer")
    
    if parameters.n_metal_layers < 3:
        raise ValueError("n_metal_layers must be at least 3 for proper surface representation")
    
    if parameters.n_metal_layers > 10:
        raise ValueError("n_metal_layers too large (>10). Consider computational cost.")
    
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
    
    # Fix bottom layers validation
    if not isinstance(parameters.fix_bottom_layers, int):
        raise TypeError("fix_bottom_layers must be an integer")
    
    if parameters.fix_bottom_layers < 0:
        raise ValueError("fix_bottom_layers must be non-negative")
    
    if parameters.fix_bottom_layers >= parameters.n_metal_layers - 1:
        raise ValueError(
            f"fix_bottom_layers ({parameters.fix_bottom_layers}) must be less than n_metal_layers-1 ({parameters.n_metal_layers-1}). "
            "Need at least 2 free layers."
        )
    
    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")
    
    # Validate water model exists and is supported
    try:
        get_water_model(parameters.water_model)
    except ValueError as e:
        raise ValueError(f"Invalid water model: {e}")
    
    # Water thickness validation
    if parameters.water_thickness is not None:
        if not isinstance(parameters.water_thickness, (int, float)):
            raise TypeError("water_thickness must be numeric or None")
        
        if parameters.water_thickness <= 0:
            raise ValueError("water_thickness must be positive")
        
        if parameters.water_thickness < 10.0:
            raise ValueError("water_thickness too small (<10 Å). Minimum recommended is 10 Å.")
        
        if parameters.water_thickness > 100.0:
            raise ValueError("water_thickness too large (>100 Å). Consider computational cost.")
    
    # Number of water molecules validation
    if parameters.n_water_molecules is not None:
        if not isinstance(parameters.n_water_molecules, int):
            raise TypeError("n_water_molecules must be an integer or None")
        
        if parameters.n_water_molecules <= 0:
            raise ValueError("n_water_molecules must be positive")
        
        if parameters.n_water_molecules > 10000:
            raise ValueError("n_water_molecules too large (>10000). Consider computational cost.")
        
        if parameters.n_water_molecules < 10:
            raise ValueError("n_water_molecules too small (<10). Need reasonable water layer.")
    
    # Water density validation
    if parameters.water_density is not None:
        if not isinstance(parameters.water_density, (int, float)):
            raise TypeError("water_density must be numeric or None")
        
        if parameters.water_density <= 0:
            raise ValueError("water_density must be positive")
        
        if parameters.water_density > 1.5:
            raise ValueError("water_density too high (>1.5 g/cm³). Water density is ~1 g/cm³.")
        
        if parameters.water_density < 0.5:
            raise ValueError("water_density too low (<0.5 g/cm³). Check units.")
    
    # Metal-water gap validation
    if not isinstance(parameters.metal_water_gap, (int, float)):
        raise TypeError("metal_water_gap must be numeric")
    
    if parameters.metal_water_gap <= 0:
        raise ValueError("metal_water_gap must be positive")
    
    if parameters.metal_water_gap < 1.5:
        raise ValueError("metal_water_gap too small (<1.5 Å). Risk of atomic overlap.")
    
    if parameters.metal_water_gap > 10.0:
        raise ValueError("metal_water_gap too large (>10 Å). Consider physical relevance.")
    
    # Vacuum above water validation
    if not isinstance(parameters.vacuum_above_water, (int, float)):
        raise TypeError("vacuum_above_water must be numeric")
    
    if parameters.vacuum_above_water <= 0:
        raise ValueError("vacuum_above_water must be positive")
    
    if parameters.vacuum_above_water < 5.0:
        raise ValueError("vacuum_above_water too small (<5 Å). Minimum recommended is 5 Å.")
    
    if parameters.vacuum_above_water > 50.0:
        raise ValueError("vacuum_above_water too large (>50 Å). Consider computational cost.")
    
    # Packmol tolerance validation
    if not isinstance(parameters.packmol_tolerance, (int, float)):
        raise TypeError("packmol_tolerance must be numeric")
    
    if parameters.packmol_tolerance <= 0:
        raise ValueError("packmol_tolerance must be positive")
    
    if parameters.packmol_tolerance < 1.0:
        raise ValueError("packmol_tolerance too small (<1.0 Å). Risk of molecular overlap.")
    
    if parameters.packmol_tolerance > 3.0:
        raise ValueError("packmol_tolerance too large (>3.0 Å). May affect packing efficiency.")
    
    # Packmol seed validation
    if not isinstance(parameters.packmol_seed, int):
        raise TypeError("packmol_seed must be an integer")
    
    if parameters.packmol_seed < 0:
        raise ValueError("packmol_seed must be non-negative")
    
    # Packmol executable validation
    if not isinstance(parameters.packmol_executable, str):
        raise TypeError("packmol_executable must be a string")
    
    if not parameters.packmol_executable.strip():
        raise ValueError("packmol_executable cannot be empty")
    
    # Output format validation
    if parameters.output_format is not None:
        if not isinstance(parameters.output_format, str):
            raise TypeError("output_format must be a string or None")
        
        valid_formats = ["xyz", "vasp", "lammps"]
        if parameters.output_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid output_format '{parameters.output_format}'. "
                f"Supported formats: {', '.join(valid_formats)}"
            )
    
    # Water orientation validation
    if not isinstance(parameters.water_orientation, str):
        raise TypeError("water_orientation must be a string")
    
    valid_orientations = ["random", "ordered", "bulk"]
    if parameters.water_orientation.lower() not in valid_orientations:
        raise ValueError(
            f"Invalid water_orientation '{parameters.water_orientation}'. "
            f"Valid options: {', '.join(valid_orientations)}"
        )
    
    # Surface coverage validation
    if not isinstance(parameters.surface_coverage, (int, float)):
        raise TypeError("surface_coverage must be numeric")
    
    if parameters.surface_coverage <= 0 or parameters.surface_coverage > 1.0:
        raise ValueError("surface_coverage must be between 0 and 1")
    
    # Surface hydroxyl validation
    if not isinstance(parameters.add_surface_hydroxyl, bool):
        raise TypeError("add_surface_hydroxyl must be a boolean")
    
    # Hydroxyl coverage validation
    if not isinstance(parameters.hydroxyl_coverage, (int, float)):
        raise TypeError("hydroxyl_coverage must be numeric")
    
    if parameters.hydroxyl_coverage < 0 or parameters.hydroxyl_coverage > 1.0:
        raise ValueError("hydroxyl_coverage must be between 0 and 1")
    
    # Center system validation
    if not isinstance(parameters.center_system, bool):
        raise TypeError("center_system must be a boolean")
    
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
    
    # Cross-parameter validation
    if parameters.n_water_molecules is not None and parameters.water_thickness is not None:
        # Both specified - warn but allow (water_thickness takes precedence for region calculation)
        pass
    
    if parameters.n_water_molecules is None and parameters.water_thickness is None:
        raise ValueError("Either n_water_molecules or water_thickness must be specified")
    
    # Check system size reasonableness
    estimated_metal_atoms = (parameters.metal_size[0] * parameters.metal_size[1] * 
                           parameters.n_metal_layers * 4)  # Rough estimate for FCC
    
    estimated_water_molecules = parameters.n_water_molecules or 100  # Default estimate
    total_atoms = estimated_metal_atoms + estimated_water_molecules * 3
    
    if total_atoms > 50000:
        raise ValueError(
            f"System too large (~{total_atoms} atoms). "
            "Consider reducing metal_size, n_metal_layers, or water content."
        )
    
    if total_atoms < 50:
        raise ValueError(
            f"System too small (~{total_atoms} atoms). "
            "Consider increasing system size."
        )