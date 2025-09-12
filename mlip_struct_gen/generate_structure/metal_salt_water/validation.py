"""Validation functions for metal-salt-water interface generation parameters."""

import numpy as np
from typing import Any
from .input_parameters import MetalSaltWaterParameters
from ..templates.salt_models import get_salt_model


def validate_parameters(params: MetalSaltWaterParameters) -> None:
    """
    Validate all parameters for metal-salt-water interface generation.
    
    Args:
        params: Parameters to validate
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
    """
    _validate_basic_types(params)
    _validate_metal_parameters(params)
    _validate_salt_parameters(params)
    _validate_water_parameters(params)
    _validate_interface_parameters(params)
    _validate_packmol_parameters(params)
    _validate_output_parameters(params)
    _validate_advanced_parameters(params)


def _validate_basic_types(params: MetalSaltWaterParameters) -> None:
    """Validate basic parameter types."""
    if not isinstance(params.metal, str):
        raise TypeError("metal must be a string")
    
    if not isinstance(params.miller_index, (tuple, list)) or len(params.miller_index) != 3:
        raise TypeError("miller_index must be a tuple or list of 3 integers")
    
    if not all(isinstance(x, int) for x in params.miller_index):
        raise TypeError("miller_index values must be integers")
    
    if not isinstance(params.metal_size, (tuple, list)) or len(params.metal_size) != 2:
        raise TypeError("metal_size must be a tuple or list of 2 integers")
    
    if not all(isinstance(x, int) for x in params.metal_size):
        raise TypeError("metal_size values must be integers")
    
    if not isinstance(params.n_metal_layers, int):
        raise TypeError("n_metal_layers must be an integer")
    
    if not isinstance(params.output_file, str):
        raise TypeError("output_file must be a string")


def _validate_metal_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate metal surface parameters."""
    # Validate metal element
    valid_metals = [
        "Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt", "Pb", "Rh", "Ir", 
        "Ca", "Sr", "Fe", "Co", "Zn", "Cd", "Hg"
    ]
    if params.metal not in valid_metals:
        raise ValueError(f"Unsupported metal '{params.metal}'. Supported: {', '.join(valid_metals)}")
    
    # Validate Miller indices
    if any(abs(h) > 10 for h in params.miller_index):
        raise ValueError("Miller indices should be between -10 and 10")
    
    if all(h == 0 for h in params.miller_index):
        raise ValueError("Miller indices cannot all be zero")
    
    # Validate metal size
    if any(s < 1 or s > 20 for s in params.metal_size):
        raise ValueError("metal_size values must be between 1 and 20")
    
    # Validate number of layers
    if params.n_metal_layers < 3 or params.n_metal_layers > 20:
        raise ValueError("n_metal_layers must be between 3 and 20")
    
    # Validate lattice constant
    if params.lattice_constant is not None:
        if not isinstance(params.lattice_constant, (int, float)):
            raise TypeError("lattice_constant must be a number")
        if params.lattice_constant <= 0 or params.lattice_constant > 10:
            raise ValueError("lattice_constant must be between 0 and 10 Å")
    
    # Validate fix_bottom_layers
    if not isinstance(params.fix_bottom_layers, int):
        raise TypeError("fix_bottom_layers must be an integer")
    if params.fix_bottom_layers < 0 or params.fix_bottom_layers >= params.n_metal_layers:
        raise ValueError("fix_bottom_layers must be between 0 and n_metal_layers-1")


def _validate_salt_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate salt parameters."""
    # Validate salt type or custom parameters
    if params.custom_salt_params is None:
        if params.salt_type is None:
            raise ValueError("Either salt_type or custom_salt_params must be provided")
        
        # Validate built-in salt type
        try:
            get_salt_model(params.salt_type)
        except ValueError as e:
            raise ValueError(f"Invalid salt_type: {e}")
    else:
        # Validate custom salt parameters
        if not isinstance(params.custom_salt_params, dict):
            raise TypeError("custom_salt_params must be a dictionary")
        
        required_keys = ["cation", "anion"]
        for key in required_keys:
            if key not in params.custom_salt_params:
                raise ValueError(f"custom_salt_params must contain '{key}' key")
        
        # Validate ion parameters
        for ion_type in ["cation", "anion"]:
            ion_data = params.custom_salt_params[ion_type]
            if not isinstance(ion_data, dict):
                raise TypeError(f"custom_salt_params['{ion_type}'] must be a dictionary")
            
            required_ion_keys = ["element", "charge", "mass"]
            for key in required_ion_keys:
                if key not in ion_data:
                    raise ValueError(f"custom_salt_params['{ion_type}'] must contain '{key}' key")
            
            # Validate ion values
            if not isinstance(ion_data["element"], str):
                raise TypeError(f"{ion_type} element must be a string")
            if not isinstance(ion_data["charge"], (int, float)):
                raise TypeError(f"{ion_type} charge must be a number")
            if not isinstance(ion_data["mass"], (int, float)):
                raise TypeError(f"{ion_type} mass must be a number")
            if ion_data["mass"] <= 0:
                raise ValueError(f"{ion_type} mass must be positive")
    
    # Validate number of salt molecules
    if not isinstance(params.n_salt_molecules, int):
        raise TypeError("n_salt_molecules must be an integer")
    if params.n_salt_molecules < 0 or params.n_salt_molecules > 10000:
        raise ValueError("n_salt_molecules must be between 0 and 10000")
    
    # Validate neutralize flag
    if not isinstance(params.neutralize, bool):
        raise TypeError("neutralize must be a boolean")


def _validate_water_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate water model parameters."""
    # Validate water model
    valid_models = ["SPC/E", "TIP3P", "TIP4P"]
    if params.water_model not in valid_models:
        raise ValueError(f"Unsupported water model '{params.water_model}'. Supported: {', '.join(valid_models)}")
    
    # Validate solution thickness
    if params.solution_thickness is not None:
        if not isinstance(params.solution_thickness, (int, float)):
            raise TypeError("solution_thickness must be a number")
        if params.solution_thickness < 5.0 or params.solution_thickness > 200.0:
            raise ValueError("solution_thickness must be between 5.0 and 200.0 Å")
    
    # Validate number of water molecules
    if params.n_water_molecules is not None:
        if not isinstance(params.n_water_molecules, int):
            raise TypeError("n_water_molecules must be an integer")
        if params.n_water_molecules < 1 or params.n_water_molecules > 100000:
            raise ValueError("n_water_molecules must be between 1 and 100000")
    
    # Validate water density
    if params.water_density is not None:
        if not isinstance(params.water_density, (int, float)):
            raise TypeError("water_density must be a number")
        if params.water_density < 0.1 or params.water_density > 5.0:
            raise ValueError("water_density must be between 0.1 and 5.0 g/cm³")
    
    # Check conflicting parameters
    if params.n_water_molecules is not None and params.water_density is not None:
        raise ValueError("Cannot specify both n_water_molecules and water_density")


def _validate_interface_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate interface parameters."""
    # Validate metal-solution gap
    if not isinstance(params.metal_solution_gap, (int, float)):
        raise TypeError("metal_solution_gap must be a number")
    if params.metal_solution_gap < 0.5 or params.metal_solution_gap > 20.0:
        raise ValueError("metal_solution_gap must be between 0.5 and 20.0 Å")
    
    # Validate vacuum above solution
    if not isinstance(params.vacuum_above_solution, (int, float)):
        raise TypeError("vacuum_above_solution must be a number")
    if params.vacuum_above_solution < 0.0 or params.vacuum_above_solution > 100.0:
        raise ValueError("vacuum_above_solution must be between 0.0 and 100.0 Å")


def _validate_packmol_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate Packmol parameters."""
    # Validate tolerance
    if not isinstance(params.packmol_tolerance, (int, float)):
        raise TypeError("packmol_tolerance must be a number")
    if params.packmol_tolerance < 0.1 or params.packmol_tolerance > 10.0:
        raise ValueError("packmol_tolerance must be between 0.1 and 10.0 Å")
    
    # Validate seed
    if not isinstance(params.packmol_seed, int):
        raise TypeError("packmol_seed must be an integer")
    if params.packmol_seed < 0:
        raise ValueError("packmol_seed must be non-negative")
    
    # Validate executable
    if not isinstance(params.packmol_executable, str):
        raise TypeError("packmol_executable must be a string")
    if not params.packmol_executable.strip():
        raise ValueError("packmol_executable cannot be empty")


def _validate_output_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate output parameters."""
    # Validate output file
    if not params.output_file.strip():
        raise ValueError("output_file cannot be empty")
    
    # Validate output format
    if params.output_format is not None:
        valid_formats = ["xyz", "vasp", "lammps-data"]
        if params.output_format.lower() not in valid_formats:
            raise ValueError(f"Unsupported output format '{params.output_format}'. Supported: {', '.join(valid_formats)}")


def _validate_advanced_parameters(params: MetalSaltWaterParameters) -> None:
    """Validate advanced parameters."""
    # Validate surface coverage
    if not isinstance(params.surface_coverage, (int, float)):
        raise TypeError("surface_coverage must be a number")
    if params.surface_coverage <= 0.0 or params.surface_coverage > 1.0:
        raise ValueError("surface_coverage must be between 0.0 and 1.0")
    
    # Validate hydroxyl parameters
    if not isinstance(params.add_surface_hydroxyl, bool):
        raise TypeError("add_surface_hydroxyl must be a boolean")
    
    if not isinstance(params.hydroxyl_coverage, (int, float)):
        raise TypeError("hydroxyl_coverage must be a number")
    if params.hydroxyl_coverage < 0.0 or params.hydroxyl_coverage > 1.0:
        raise ValueError("hydroxyl_coverage must be between 0.0 and 1.0")
    
    # Validate center system
    if not isinstance(params.center_system, bool):
        raise TypeError("center_system must be a boolean")
    
    # Validate logging parameters
    if not isinstance(params.log, bool):
        raise TypeError("log must be a boolean")
    
    # Logger validation is optional since it's handled by the generator