"""Validation for metal-water interface generation parameters."""

import shutil
from pathlib import Path

from .input_parameters import MetalWaterParameters

# Supported FCC metals with experimental lattice constants (Angstroms)
SUPPORTED_METALS: set[str] = {
    "Al",
    "Au",
    "Ag",
    "Cu",
    "Ni",
    "Pd",
    "Pt",
    "Pb",
    "Rh",
    "Ir",
    "Ca",
    "Sr",
    "Yb",
}

# Default lattice constants for common FCC metals (Angstroms)
DEFAULT_LATTICE_CONSTANTS = {
    "Al": 4.050,
    "Au": 4.078,
    "Ag": 4.085,
    "Cu": 3.615,
    "Ni": 3.524,
    "Pd": 3.890,
    "Pt": 3.924,
    "Pb": 4.950,
    "Rh": 3.803,
    "Ir": 3.839,
    "Ca": 5.588,
    "Sr": 6.085,
    "Yb": 5.485,
}

# Water model parameters
WATER_MODELS = {
    "SPC/E": {
        "OH_distance": 1.0,  # Angstroms
        "HOH_angle": 109.47,  # degrees
    },
    "TIP3P": {
        "OH_distance": 0.9572,  # Angstroms
        "HOH_angle": 104.52,  # degrees
    },
    "TIP4P": {
        "OH_distance": 0.9572,  # Angstroms
        "HOH_angle": 104.52,  # degrees
    },
    "SPC/Fw": {
        "OH_distance": 1.012,  # Angstroms
        "HOH_angle": 113.24,  # degrees
    },
}


def validate_parameters(params: MetalWaterParameters) -> None:
    """
    Validate metal-water interface generation parameters.

    Args:
        params: Parameters to validate

    Raises:
        ValueError: If any parameter is invalid
        RuntimeError: If required external tools are missing
    """
    # Check for PACKMOL availability
    if not shutil.which(params.packmol_executable):
        raise RuntimeError(
            f"PACKMOL executable '{params.packmol_executable}' not found. "
            f"Please install PACKMOL and ensure it's in your PATH, "
            f"or specify the full path to the executable."
        )

    # Validate metal
    if not params.metal:
        raise ValueError("Metal element symbol is required")

    if params.metal not in SUPPORTED_METALS:
        raise ValueError(
            f"Metal '{params.metal}' not supported. "
            f"Supported metals: {', '.join(sorted(SUPPORTED_METALS))}"
        )

    # Validate size
    if not params.size or len(params.size) != 3:
        raise ValueError("size must be a tuple of 3 integers (nx, ny, nz)")

    nx, ny, nz = params.size

    if not all(isinstance(x, int) for x in params.size):
        raise ValueError("size values must be integers")

    if nx < 1 or ny < 1:
        raise ValueError(f"Lateral dimensions (nx={nx}, ny={ny}) must be at least 1")

    if nz < 3:
        raise ValueError(
            f"Number of layers (nz={nz}) must be at least 3 for proper surface representation"
        )

    if nx > 20 or ny > 20:
        raise ValueError(
            f"Lateral dimensions (nx={nx}, ny={ny}) should not exceed 20 for computational efficiency"
        )

    if nz > 20:
        raise ValueError(
            f"Number of layers (nz={nz}) should not exceed 20 for computational efficiency"
        )

    # Validate water parameters
    if params.n_water < 1:
        raise ValueError(f"n_water ({params.n_water}) must be at least 1")

    if params.n_water > 10000:
        raise ValueError(
            f"n_water ({params.n_water}) is very large (>10000). Consider computational cost."
        )

    if params.density <= 0:
        raise ValueError(f"density ({params.density} g/cm³) must be positive")

    if params.density < 0.5 or params.density > 1.5:
        raise ValueError(f"density ({params.density} g/cm³) should be between 0.5 and 1.5 g/cm³")

    # Validate water model
    if params.water_model not in WATER_MODELS:
        raise ValueError(
            f"Water model '{params.water_model}' not supported. "
            f"Supported models: {', '.join(WATER_MODELS.keys())}"
        )

    # Validate gap and vacuum
    if params.gap_above_metal < 0:
        raise ValueError(f"gap_above_metal ({params.gap_above_metal} Å) must be non-negative")

    if params.gap_above_metal > 10:
        raise ValueError(f"gap_above_metal ({params.gap_above_metal} Å) is very large (>10 Å)")

    if params.vacuum_above_water < 0:
        raise ValueError(f"vacuum_above_water ({params.vacuum_above_water} Å) must be non-negative")

    if params.vacuum_above_water > 50:
        raise ValueError(
            f"vacuum_above_water ({params.vacuum_above_water} Å) should not exceed 50 Å"
        )

    # Validate lattice constant if provided
    if params.lattice_constant is not None:
        if params.lattice_constant <= 0:
            raise ValueError(f"Lattice constant ({params.lattice_constant} Å) must be positive")

        if params.lattice_constant < 2.0 or params.lattice_constant > 7.0:
            raise ValueError(
                f"Lattice constant ({params.lattice_constant} Å) should be between 2.0 and 7.0 Å "
                f"for FCC metals"
            )

    # Validate fix_bottom_layers
    if params.fix_bottom_layers < 0:
        raise ValueError(f"fix_bottom_layers ({params.fix_bottom_layers}) must be non-negative")

    if params.fix_bottom_layers >= nz:
        raise ValueError(
            f"fix_bottom_layers ({params.fix_bottom_layers}) must be less than "
            f"the number of layers ({nz})"
        )

    # Validate PACKMOL parameters
    if params.packmol_tolerance <= 0:
        raise ValueError(f"packmol_tolerance ({params.packmol_tolerance} Å) must be positive")

    if params.packmol_tolerance < 1.0:
        print(
            f"Warning: Small packmol_tolerance ({params.packmol_tolerance} Å) may cause packing failures"
        )

    if params.packmol_tolerance > 3.0:
        print(
            f"Warning: Large packmol_tolerance ({params.packmol_tolerance} Å) may result in poor packing"
        )

    # Validate seed
    if params.seed < 0:
        raise ValueError(f"seed ({params.seed}) must be non-negative")

    # Validate output file
    if not params.output_file:
        raise ValueError("Output file path is required")

    output_path = Path(params.output_file)

    # Check if parent directory exists
    parent_dir = output_path.parent
    if parent_dir != Path(".") and not parent_dir.exists():
        raise ValueError(f"Output directory does not exist: {parent_dir}")

    # Validate output format
    if params.output_format:
        valid_formats = {"xyz", "vasp", "poscar", "lammps", "lammps/dpmd", "data", "lammpstrj"}
        if params.output_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid output format '{params.output_format}'. "
                f"Supported formats: {', '.join(valid_formats)}"
            )
    else:
        # Check if file extension is recognizable
        suffix = output_path.suffix.lower()
        valid_extensions = {".xyz", ".vasp", ".poscar", ".lammps", ".data", ".lmp", ".lammpstrj"}
        if suffix and suffix not in valid_extensions and output_path.name.upper() != "POSCAR":
            print(
                f"Warning: Unrecognized file extension '{suffix}'. Will use LAMMPS format by default."
            )


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

    if metal in DEFAULT_LATTICE_CONSTANTS:
        return DEFAULT_LATTICE_CONSTANTS[metal]

    raise ValueError(
        f"No default lattice constant for metal '{metal}'. "
        f"Please provide a custom lattice constant."
    )


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
