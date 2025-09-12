"""Parameter validation for salt water box generation."""

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_parameters import SaltWaterBoxGeneratorParameters

from ..templates.water_models import get_water_model, get_water_density
from ..templates.salt_models import get_salt_model, get_available_salts


def validate_parameters(parameters: "SaltWaterBoxGeneratorParameters") -> None:
    """
    Comprehensive parameter validation and normalization.
    
    Args:
        parameters: Parameters to validate and normalize
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
    """
    # Box size validation and normalization
    if parameters.box_size is None:
        # Box size will be computed from n_water_molecules - validate n_water_molecules is provided
        if parameters.n_water_molecules is None:
            raise ValueError("Either box_size or n_water_molecules must be provided")
        # box_size will be computed later in the generator
    elif isinstance(parameters.box_size, (int, float)):
        # Convert single number to cubic box
        if parameters.box_size <= 0:
            raise ValueError("Box size must be positive")
        parameters.box_size = (
            float(parameters.box_size), 
            float(parameters.box_size), 
            float(parameters.box_size)
        )
    elif isinstance(parameters.box_size, (list, tuple)):
        # Validate and normalize list/tuple
        if len(parameters.box_size) != 3:
            raise ValueError("box_size as tuple/list must have exactly 3 dimensions")
        
        if not all(isinstance(s, (int, float)) for s in parameters.box_size):
            raise TypeError("All box_size dimensions must be numeric")
        
        if not all(s > 0 for s in parameters.box_size):
            raise ValueError("All box dimensions must be positive")
        
        parameters.box_size = tuple(float(s) for s in parameters.box_size)
    else:
        raise TypeError("box_size must be a number (cubic), tuple/list of 3 numbers, or None")
    
    # Additional box size validation (only if box_size is provided)
    if parameters.box_size is not None:
        if any(s > 1000.0 for s in parameters.box_size):
            raise ValueError("Box dimensions too large (>1000 Å). Check units.")
        
        if any(s < 5.0 for s in parameters.box_size):
            raise ValueError("Box dimensions too small (<5 Å). Minimum recommended size is 5 Å.")
    
    # Output file validation
    if not isinstance(parameters.output_file, str):
        raise TypeError("output_file must be a string")
    
    if not parameters.output_file.strip():
        raise ValueError("output_file cannot be empty")
    
    # Add extension based on format if not provided
    output_path = Path(parameters.output_file)
    if not output_path.suffix:
        # Add appropriate extension based on format
        format_extensions = {
            "xyz": ".xyz",
            "lammps": ".data",
            "poscar": ""  # POSCAR doesn't use an extension
        }
        extension = format_extensions.get(parameters.output_format, ".xyz")
        parameters.output_file = str(output_path) + extension
    
    # Salt type validation
    if parameters.salt_type is not None and parameters.custom_salt_params is None:
        if not isinstance(parameters.salt_type, str):
            raise TypeError("salt_type must be a string")
        
        # Validate salt type exists and is supported
        available_salts = get_available_salts()
        if parameters.salt_type not in available_salts:
            raise ValueError(
                f"Invalid salt type '{parameters.salt_type}'. "
                f"Available salts: {', '.join(available_salts)}"
            )
    elif parameters.custom_salt_params is not None:
        # Validate custom salt parameters
        validate_custom_salt_params(parameters.custom_salt_params)
    else:
        raise ValueError("Either salt_type or custom_salt_params must be provided")
    
    # Number of salt molecules validation
    if not isinstance(parameters.n_salt_molecules, int):
        raise TypeError("n_salt_molecules must be an integer")
    
    if parameters.n_salt_molecules <= 0:
        raise ValueError("n_salt_molecules must be positive")
    
    if parameters.n_salt_molecules > 10000:
        raise ValueError("n_salt_molecules too large (>10000). Consider a smaller number.")
    
    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")
    
    # Validate water model exists and is supported
    try:
        get_water_model(parameters.water_model)
    except ValueError as e:
        raise ValueError(f"Invalid water model: {e}")
    
    # Number of water molecules validation
    if parameters.n_water_molecules is not None:
        if not isinstance(parameters.n_water_molecules, int):
            raise TypeError("n_water_molecules must be an integer or None")
        
        if parameters.n_water_molecules <= 0:
            raise ValueError("n_water_molecules must be positive")
        
        if parameters.n_water_molecules > 1000000:
            raise ValueError("n_water_molecules too large (>1M). Consider using density instead.")
    
    # Water density validation
    if parameters.water_density is not None:
        if not isinstance(parameters.water_density, (int, float)):
            raise TypeError("water_density must be numeric or None")
        
        if parameters.water_density <= 0:
            raise ValueError("water_density must be positive")
        
        if parameters.water_density > 5.0:
            raise ValueError("water_density too high (>5 g/cm³). Water density is ~1 g/cm³.")
        
        if parameters.water_density < 0.1:
            raise ValueError("water_density too low (<0.1 g/cm³). Check units.")
    
    # Cannot specify both n_water_molecules and water_density
    if parameters.n_water_molecules is not None and parameters.water_density is not None:
        raise ValueError("Cannot specify both n_water_molecules and water_density. Choose one.")
    
    # Tolerance validation
    if not isinstance(parameters.tolerance, (int, float)):
        raise TypeError("tolerance must be numeric")
    
    if parameters.tolerance <= 0:
        raise ValueError("tolerance must be positive")
    
    if parameters.tolerance > 10.0:
        raise ValueError("tolerance too large (>10 Å). Typical values are 1-3 Å.")
    
    # Seed validation
    if not isinstance(parameters.seed, int):
        raise TypeError("seed must be an integer")
    
    if parameters.seed < 0:
        raise ValueError("seed must be non-negative")
    
    # Packmol executable validation
    if not isinstance(parameters.packmol_executable, str):
        raise TypeError("packmol_executable must be a string")
    
    if not parameters.packmol_executable.strip():
        raise ValueError("packmol_executable cannot be empty")
    
    # Output format validation
    if not isinstance(parameters.output_format, str):
        raise TypeError("output_format must be a string")
    
    valid_formats = ["xyz", "lammps", "poscar"]
    if parameters.output_format not in valid_formats:
        raise ValueError(
            f"Invalid output_format '{parameters.output_format}'. "
            f"Supported formats: {', '.join(valid_formats)}"
        )
    
    # Neutralize validation
    if not isinstance(parameters.neutralize, bool):
        raise TypeError("neutralize must be a boolean")
    
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
    


def validate_custom_salt_params(custom_params: dict) -> None:
    """
    Validate custom salt parameters.
    
    Args:
        custom_params: Custom salt parameter dictionary
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameter types are incorrect
    """
    required_keys = ["name", "cation", "anion"]
    for key in required_keys:
        if key not in custom_params:
            raise ValueError(f"custom_salt_params must contain '{key}'")
    
    # Validate name
    if not isinstance(custom_params["name"], str):
        raise TypeError("custom_salt_params['name'] must be a string")
    
    # Validate cation
    ion_required_keys = ["element", "charge", "mass"]
    for ion_type in ["cation", "anion"]:
        ion_data = custom_params[ion_type]
        
        if not isinstance(ion_data, dict):
            raise TypeError(f"custom_salt_params['{ion_type}'] must be a dictionary")
        
        for key in ion_required_keys:
            if key not in ion_data:
                raise ValueError(f"custom_salt_params['{ion_type}'] must contain '{key}'")
        
        # Validate element
        if not isinstance(ion_data["element"], str):
            raise TypeError(f"custom_salt_params['{ion_type}']['element'] must be a string")
        
        # Validate charge
        if not isinstance(ion_data["charge"], (int, float)):
            raise TypeError(f"custom_salt_params['{ion_type}']['charge'] must be numeric")
        
        # Validate mass
        if not isinstance(ion_data["mass"], (int, float)):
            raise TypeError(f"custom_salt_params['{ion_type}']['mass'] must be numeric")
        
        if ion_data["mass"] <= 0:
            raise ValueError(f"custom_salt_params['{ion_type}']['mass'] must be positive")
    
    # Validate charges
    if custom_params["cation"]["charge"] <= 0:
        raise ValueError("Cation charge must be positive")
    
    if custom_params["anion"]["charge"] >= 0:
        raise ValueError("Anion charge must be negative")
    
    # Validate optional stoichiometry
    if "stoichiometry" in custom_params:
        stoich = custom_params["stoichiometry"]
        if not isinstance(stoich, dict):
            raise TypeError("custom_salt_params['stoichiometry'] must be a dictionary")
        
        if "cation" not in stoich or "anion" not in stoich:
            raise ValueError("stoichiometry must contain 'cation' and 'anion' keys")
        
        for key in ["cation", "anion"]:
            if not isinstance(stoich[key], int):
                raise TypeError(f"stoichiometry['{key}'] must be an integer")
            
            if stoich[key] <= 0:
                raise ValueError(f"stoichiometry['{key}'] must be positive")