"""Validation for metal-salt-water interface generation parameters."""

import shutil
from pathlib import Path
from typing import Any

from .input_parameters import MetalSaltWaterParameters

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

# Supported salt types with their stoichiometry
SUPPORTED_SALTS = {
    "NaCl": {"cation": "Na", "anion": "Cl", "cation_count": 1, "anion_count": 1},
    "KCl": {"cation": "K", "anion": "Cl", "cation_count": 1, "anion_count": 1},
    "LiCl": {"cation": "Li", "anion": "Cl", "cation_count": 1, "anion_count": 1},
    "CaCl2": {"cation": "Ca", "anion": "Cl", "cation_count": 1, "anion_count": 2},
    "MgCl2": {"cation": "Mg", "anion": "Cl", "cation_count": 1, "anion_count": 2},
    "NaBr": {"cation": "Na", "anion": "Br", "cation_count": 1, "anion_count": 1},
    "KBr": {"cation": "K", "anion": "Br", "cation_count": 1, "anion_count": 1},
    "CsCl": {"cation": "Cs", "anion": "Cl", "cation_count": 1, "anion_count": 1},
}

# Ion parameters (masses in g/mol, radii in Angstroms)
ION_PARAMETERS = {
    "Na": {"mass": 22.990, "charge": 1.0, "vdw_radius": 2.27},
    "K": {"mass": 39.098, "charge": 1.0, "vdw_radius": 2.75},
    "Li": {"mass": 6.941, "charge": 1.0, "vdw_radius": 1.82},
    "Ca": {"mass": 40.078, "charge": 2.0, "vdw_radius": 2.31},
    "Mg": {"mass": 24.305, "charge": 2.0, "vdw_radius": 1.73},
    "Cs": {"mass": 132.905, "charge": 1.0, "vdw_radius": 3.43},
    "Cl": {"mass": 35.453, "charge": -1.0, "vdw_radius": 1.81},
    "Br": {"mass": 79.904, "charge": -1.0, "vdw_radius": 1.96},
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
}


def validate_parameters(params: MetalSaltWaterParameters) -> None:
    """
    Validate metal-salt-water interface generation parameters.

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

    # Validate salt parameters
    if params.salt_type not in SUPPORTED_SALTS:
        raise ValueError(
            f"Salt type '{params.salt_type}' not supported. "
            f"Supported salts: {', '.join(SUPPORTED_SALTS.keys())}"
        )

    if params.n_salt < 0:
        raise ValueError(f"n_salt ({params.n_salt}) must be non-negative")

    if params.n_salt > 1000:
        raise ValueError(
            f"n_salt ({params.n_salt}) is very large (>1000). Consider computational cost."
        )

    # Validate water parameters
    if params.n_water < 1:
        raise ValueError(f"n_water ({params.n_water}) must be at least 1")

    if params.n_water > 10000:
        raise ValueError(
            f"n_water ({params.n_water}) is very large (>10000). Consider computational cost."
        )

    if params.density <= 0:
        raise ValueError(f"density ({params.density} g/cm^3) must be positive")

    if params.density < 0.5 or params.density > 1.5:
        raise ValueError(f"density ({params.density} g/cm^3) should be between 0.5 and 1.5 g/cm^3")

    # Validate water model
    if params.water_model not in WATER_MODELS:
        raise ValueError(
            f"Water model '{params.water_model}' not supported. "
            f"Supported models: {', '.join(WATER_MODELS.keys())}"
        )

    # Validate gap and vacuum
    if params.gap < 0:
        raise ValueError(f"gap ({params.gap} Angstroms) must be non-negative")

    if params.gap > 10:
        raise ValueError(f"gap ({params.gap} Angstroms) is very large (>10 Angstroms)")

    if params.vacuum_above_water < 0:
        raise ValueError(
            f"vacuum_above_water ({params.vacuum_above_water} Angstroms) must be non-negative"
        )

    if params.vacuum_above_water > 50:
        raise ValueError(
            f"vacuum_above_water ({params.vacuum_above_water} Angstroms) should not exceed 50 Angstroms"
        )

    # Validate lattice constant if provided
    if params.lattice_constant is not None:
        if params.lattice_constant <= 0:
            raise ValueError(
                f"Lattice constant ({params.lattice_constant} Angstroms) must be positive"
            )

        if params.lattice_constant < 2.0 or params.lattice_constant > 7.0:
            raise ValueError(
                f"Lattice constant ({params.lattice_constant} Angstroms) should be between 2.0 and 7.0 Angstroms "
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
        raise ValueError(
            f"packmol_tolerance ({params.packmol_tolerance} Angstroms) must be positive"
        )

    if params.packmol_tolerance < 1.0:
        print(
            f"Warning: Small packmol_tolerance ({params.packmol_tolerance} Angstroms) may cause packing failures"
        )

    if params.packmol_tolerance > 3.0:
        print(
            f"Warning: Large packmol_tolerance ({params.packmol_tolerance} Angstroms) may result in poor packing"
        )

    # Validate seed
    if params.seed is not None and params.seed < 0:
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
        valid_formats = {"xyz", "vasp", "poscar", "lammps", "data"}
        if params.output_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid output format '{params.output_format}'. "
                f"Supported formats: {', '.join(valid_formats)}"
            )
    else:
        # Check if file extension is recognizable
        suffix = output_path.suffix.lower()
        valid_extensions = {".xyz", ".vasp", ".poscar", ".lammps", ".data", ".lmp"}
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


def get_salt_info(salt_type: str) -> dict[str, Any]:
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


def get_ion_params(ion: str) -> dict[str, Any]:
    """
    Get ion parameters including mass, charge, and radius.

    Args:
        ion: Ion element symbol

    Returns:
        Dictionary with ion parameters

    Raises:
        ValueError: If ion is not supported
    """
    if ion not in ION_PARAMETERS:
        raise ValueError(f"Unknown ion: {ion}")

    return ION_PARAMETERS[ion]
