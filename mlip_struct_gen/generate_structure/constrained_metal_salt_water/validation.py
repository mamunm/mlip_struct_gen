"""Parameter validation for constrained metal-salt-water interface generation."""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_parameters import ConstrainedMetalSaltWaterParameters

from ..templates.water_models import get_water_model

# Supported FCC metals
SUPPORTED_METALS: set[str] = {"Cu", "Pt"}

# Lattice constants for FCC metals (Angstroms)
# Cu and Pt use training data lattice constants
METAL_LATTICE_CONSTANTS = {
    "Cu": 3.5364568957690099,
    "Pt": 3.9005639173935944,
}

# Supported salt types
SUPPORTED_SALTS = {
    "NaCl": {"cation": "Na", "anion": "Cl", "cation_count": 1, "anion_count": 1},
    "KCl": {"cation": "K", "anion": "Cl", "cation_count": 1, "anion_count": 1},
    "LiCl": {"cation": "Li", "anion": "Cl", "cation_count": 1, "anion_count": 1},
    "CsCl": {"cation": "Cs", "anion": "Cl", "cation_count": 1, "anion_count": 1},
}

# Valid water elements for constraints
VALID_WATER_ELEMENTS = {"O", "H"}

# Valid ion elements for constraints
VALID_ION_ELEMENTS = {"Na", "K", "Li", "Cs", "Cl"}

# All valid elements for constraints
VALID_CONSTRAINT_ELEMENTS = VALID_WATER_ELEMENTS | VALID_ION_ELEMENTS

# Water model parameters
WATER_MODELS = {
    "SPC/E": {
        "OH_distance": 1.0,
        "HOH_angle": 109.47,
    },
    "TIP3P": {
        "OH_distance": 0.9572,
        "HOH_angle": 104.52,
    },
    "TIP4P": {
        "OH_distance": 0.9572,
        "HOH_angle": 104.52,
    },
    "SPC/Fw": {
        "OH_distance": 1.012,
        "HOH_angle": 113.24,
    },
}


def validate_parameters(parameters: "ConstrainedMetalSaltWaterParameters") -> None:
    """
    Validate parameters for constrained metal-salt-water interface generation.

    Args:
        parameters: Parameters to validate

    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
        RuntimeError: If required external tools are missing
    """
    # Check for PACKMOL availability
    if not shutil.which(parameters.packmol_executable):
        raise RuntimeError(
            f"PACKMOL executable '{parameters.packmol_executable}' not found. "
            f"Please install PACKMOL and ensure it's in your PATH, "
            f"or specify the full path to the executable."
        )

    # Metal validation
    if not isinstance(parameters.metal, str):
        raise TypeError("metal must be a string")
    if parameters.metal not in SUPPORTED_METALS:
        raise ValueError(
            f"Metal '{parameters.metal}' not supported. "
            f"Supported metals: {', '.join(sorted(SUPPORTED_METALS))}"
        )

    # Size validation
    if not parameters.size or len(parameters.size) != 3:
        raise ValueError("size must be a tuple of 3 integers (nx, ny, nz)")

    nx, ny, nz = parameters.size
    if not all(isinstance(x, int) for x in parameters.size):
        raise ValueError("size values must be integers")
    if nx < 1 or ny < 1:
        raise ValueError(f"Lateral dimensions (nx={nx}, ny={ny}) must be at least 1")
    if nz < 3:
        raise ValueError(
            f"Number of layers (nz={nz}) must be at least 3 for proper surface representation"
        )

    # Lattice constant validation
    if parameters.lattice_constant is not None:
        if not isinstance(parameters.lattice_constant, int | float):
            raise TypeError("lattice_constant must be numeric")
        if parameters.lattice_constant <= 0:
            raise ValueError("lattice_constant must be positive")
        if parameters.lattice_constant < 2.0 or parameters.lattice_constant > 7.0:
            raise ValueError(
                f"lattice_constant ({parameters.lattice_constant} A) should be between 2.0 and 7.0 A"
            )

    # fix_bottom_layers validation
    if not isinstance(parameters.fix_bottom_layers, int):
        raise TypeError("fix_bottom_layers must be an integer")
    if parameters.fix_bottom_layers < 0:
        raise ValueError("fix_bottom_layers must be non-negative")
    if parameters.fix_bottom_layers >= nz:
        raise ValueError(
            f"fix_bottom_layers ({parameters.fix_bottom_layers}) must be less than "
            f"the number of layers ({nz})"
        )

    # Salt validation
    if not isinstance(parameters.salt_type, str):
        raise TypeError("salt_type must be a string")
    if parameters.salt_type not in SUPPORTED_SALTS:
        raise ValueError(
            f"Salt type '{parameters.salt_type}' not supported. "
            f"Supported salts: {', '.join(SUPPORTED_SALTS.keys())}"
        )
    if not isinstance(parameters.n_salt, int):
        raise TypeError("n_salt must be an integer")
    if parameters.n_salt < 0:
        raise ValueError("n_salt must be non-negative")

    # Output file validation
    if not isinstance(parameters.output_file, str):
        raise TypeError("output_file must be a string")
    if not parameters.output_file.strip():
        raise ValueError("output_file cannot be empty")

    output_path = Path(parameters.output_file)
    if not output_path.suffix:
        parameters.output_file = str(output_path) + ".data"

    # Model files validation
    if not isinstance(parameters.model_files, list):
        raise TypeError("model_files must be a list")
    for mf in parameters.model_files:
        if not isinstance(mf, str):
            raise TypeError("All model_files must be strings")

    # Water parameters validation
    if not isinstance(parameters.n_water, int):
        raise TypeError("n_water must be an integer")
    if parameters.n_water < 1:
        raise ValueError("n_water must be at least 1")
    if parameters.n_water > 10000:
        raise ValueError("n_water too large (>10000)")

    if not isinstance(parameters.density, int | float):
        raise TypeError("density must be numeric")
    if parameters.density <= 0:
        raise ValueError("density must be positive")
    if parameters.density > 2.0:
        raise ValueError("density too high (>2 g/cm3)")

    if parameters.gap < 0:
        raise ValueError("gap must be non-negative")

    if parameters.vacuum_above_water < 0:
        raise ValueError("vacuum_above_water must be non-negative")

    if parameters.no_salt_zone < 0 or parameters.no_salt_zone >= 0.5:
        raise ValueError("no_salt_zone must be between 0 and 0.5")

    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")
    try:
        get_water_model(parameters.water_model)
    except ValueError as e:
        raise ValueError(f"Invalid water model: {e}") from e

    # Metal-water distance constraints validation
    for constraint in parameters.metal_water_distance_constraints:
        if constraint.water_element not in VALID_WATER_ELEMENTS:
            raise ValueError(
                f"Invalid water_element: {constraint.water_element}. "
                f"Must be one of: {sorted(VALID_WATER_ELEMENTS)}"
            )
        if not isinstance(constraint.count, int | str):
            raise TypeError("Metal-water distance constraint count must be int or 'all'")
        if isinstance(constraint.count, str) and constraint.count != "all":
            raise ValueError("Metal-water distance constraint count string must be 'all'")
        if isinstance(constraint.count, int) and constraint.count <= 0:
            raise ValueError("Metal-water distance constraint count must be positive")
        if not isinstance(constraint.distance, int | float):
            raise TypeError("Metal-water distance constraint distance must be numeric")
        if constraint.distance <= 0:
            raise ValueError("Metal-water distance constraint distance must be positive")
        if constraint.distance > 10.0:
            raise ValueError("Metal-water distance constraint distance too large (>10 A)")

    # Metal-water angle constraints validation
    for constraint in parameters.metal_water_angle_constraints:
        if not isinstance(constraint.count, int | str):
            raise TypeError("Metal-water angle constraint count must be int or 'all'")
        if isinstance(constraint.count, str) and constraint.count != "all":
            raise ValueError("Metal-water angle constraint count string must be 'all'")
        if isinstance(constraint.count, int) and constraint.count <= 0:
            raise ValueError("Metal-water angle constraint count must be positive")
        if not isinstance(constraint.angle, int | float):
            raise TypeError("Metal-water angle constraint angle must be numeric")
        if constraint.angle <= 0 or constraint.angle >= 180:
            raise ValueError("Metal-water angle constraint angle must be between 0 and 180 degrees")

    # Metal-ion distance constraints validation
    for constraint in parameters.metal_ion_distance_constraints:
        if constraint.ion_element not in VALID_ION_ELEMENTS:
            raise ValueError(
                f"Invalid ion_element: {constraint.ion_element}. "
                f"Must be one of: {sorted(VALID_ION_ELEMENTS)}"
            )
        if not isinstance(constraint.count, int | str):
            raise TypeError("Metal-ion distance constraint count must be int or 'all'")
        if isinstance(constraint.count, str) and constraint.count != "all":
            raise ValueError("Metal-ion distance constraint count string must be 'all'")
        if isinstance(constraint.count, int) and constraint.count <= 0:
            raise ValueError("Metal-ion distance constraint count must be positive")
        if not isinstance(constraint.distance, int | float):
            raise TypeError("Metal-ion distance constraint distance must be numeric")
        if constraint.distance <= 0:
            raise ValueError("Metal-ion distance constraint distance must be positive")
        if constraint.distance > 10.0:
            raise ValueError("Metal-ion distance constraint distance too large (>10 A)")

    # General distance constraints validation (O-H, O-O, Na-O, Cl-O, Na-Cl)
    for constraint in parameters.distance_constraints:
        if not isinstance(constraint.element1, str):
            raise TypeError("Distance constraint element1 must be a string")
        if not isinstance(constraint.element2, str):
            raise TypeError("Distance constraint element2 must be a string")
        if constraint.element1 not in VALID_CONSTRAINT_ELEMENTS:
            raise ValueError(
                f"Invalid element: {constraint.element1}. "
                f"Must be one of: {sorted(VALID_CONSTRAINT_ELEMENTS)}"
            )
        if constraint.element2 not in VALID_CONSTRAINT_ELEMENTS:
            raise ValueError(
                f"Invalid element: {constraint.element2}. "
                f"Must be one of: {sorted(VALID_CONSTRAINT_ELEMENTS)}"
            )
        if not isinstance(constraint.count, int | str):
            raise TypeError("Distance constraint count must be int or 'all'")
        if isinstance(constraint.count, str) and constraint.count != "all":
            raise ValueError("Distance constraint count string must be 'all'")
        if isinstance(constraint.count, int) and constraint.count <= 0:
            raise ValueError("Distance constraint count must be positive")
        if not isinstance(constraint.distance, int | float):
            raise TypeError("Distance constraint distance must be numeric")
        if constraint.distance <= 0:
            raise ValueError("Distance constraint distance must be positive")
        if constraint.distance > 10.0:
            raise ValueError("Distance constraint distance too large (>10 A)")

    # Angle constraints validation
    for constraint in parameters.angle_constraints:
        if not isinstance(constraint.count, int | str):
            raise TypeError("Angle constraint count must be int or 'all'")
        if isinstance(constraint.count, str) and constraint.count != "all":
            raise ValueError("Angle constraint count string must be 'all'")
        if isinstance(constraint.count, int) and constraint.count <= 0:
            raise ValueError("Angle constraint count must be positive")
        if not isinstance(constraint.angle, int | float):
            raise TypeError("Angle constraint angle must be numeric")
        if constraint.angle <= 0 or constraint.angle >= 180:
            raise ValueError("Angle constraint angle must be between 0 and 180 degrees")

    # MD parameters validation
    if not isinstance(parameters.nsteps, int) or parameters.nsteps <= 0:
        raise ValueError("nsteps must be a positive integer")
    if not isinstance(parameters.temp, int | float) or parameters.temp <= 0:
        raise ValueError("temp must be positive")
    if not isinstance(parameters.pres, int | float) or parameters.pres <= 0:
        raise ValueError("pres must be positive")
    if not isinstance(parameters.timestep, int | float) or parameters.timestep <= 0:
        raise ValueError("timestep must be positive")
    if not isinstance(parameters.dump_freq, int) or parameters.dump_freq <= 0:
        raise ValueError("dump_freq must be a positive integer")
    if not isinstance(parameters.thermo_freq, int) or parameters.thermo_freq <= 0:
        raise ValueError("thermo_freq must be a positive integer")

    if not isinstance(parameters.minimize, bool):
        raise TypeError("minimize must be a boolean")

    # Packmol parameters
    if (
        not isinstance(parameters.packmol_tolerance, int | float)
        or parameters.packmol_tolerance <= 0
    ):
        raise ValueError("packmol_tolerance must be positive")
    if not isinstance(parameters.seed, int) or parameters.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(parameters.constraint_seed, int) or parameters.constraint_seed < 0:
        raise ValueError("constraint_seed must be a non-negative integer")
    if not isinstance(parameters.packmol_executable, str):
        raise TypeError("packmol_executable must be a string")

    # Elements validation
    if not isinstance(parameters.elements, list):
        raise TypeError("elements must be a list")
    for elem in parameters.elements:
        if not isinstance(elem, str):
            raise TypeError("All elements must be strings")

    # Logging validation
    if not isinstance(parameters.log, bool):
        raise TypeError("log must be a boolean")


def get_lattice_constant(metal: str, custom_lattice: float | None = None) -> float:
    """
    Get lattice constant for a metal.

    Args:
        metal: Metal element symbol
        custom_lattice: Custom lattice constant (optional)

    Returns:
        Lattice constant in Angstroms

    Raises:
        ValueError: If metal is not supported and no custom lattice is provided
    """
    if custom_lattice is not None:
        return custom_lattice

    if metal in METAL_LATTICE_CONSTANTS:
        return METAL_LATTICE_CONSTANTS[metal]

    raise ValueError(
        f"No default lattice constant for metal '{metal}'. "
        f"Please provide a custom lattice constant."
    )


def get_salt_info(salt_type: str) -> dict:
    """
    Get salt information including stoichiometry and ion types.

    Args:
        salt_type: Salt type name

    Returns:
        Dictionary with salt information

    Raises:
        ValueError: If salt is not supported
    """
    if salt_type not in SUPPORTED_SALTS:
        raise ValueError(f"Unknown salt type: {salt_type}")

    return SUPPORTED_SALTS[salt_type]


def get_water_model_params(model: str) -> dict:
    """
    Get water model parameters.

    Args:
        model: Water model name

    Returns:
        Dictionary with water model parameters

    Raises:
        ValueError: If model is not supported
    """
    if model not in WATER_MODELS:
        raise ValueError(f"Unknown water model: {model}")

    return WATER_MODELS[model]
