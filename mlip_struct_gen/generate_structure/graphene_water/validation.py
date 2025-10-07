"""Validation utilities for graphene-water interface generation."""

from pathlib import Path

from ..templates.water_models import WATER_MODELS
from .input_parameters import GrapheneWaterParameters

# Supported output formats
SUPPORTED_FORMATS = {
    ".xyz": "xyz",
    ".vasp": "poscar",
    "POSCAR": "poscar",
    ".lammps": "lammps",
    ".data": "lammps",
    ".lammpstrj": "lammpstrj",
}


def validate_parameters(params: GrapheneWaterParameters) -> None:
    """
    Validate graphene-water interface generation parameters.

    Args:
        params: Parameters to validate

    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate size
    if not isinstance(params.size, tuple) or len(params.size) != 2:
        raise ValueError("size must be a tuple of 2 integers (nx, ny)")

    nx, ny = params.size
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError("size dimensions must be integers")

    if nx < 1 or ny < 1:
        raise ValueError(f"Invalid size {params.size}. Minimum size is (1, 1)")

    if nx > 1000 or ny > 1000:
        raise ValueError(f"Size {params.size} is too large. Maximum is (1000, 1000)")

    # Validate n_water
    if params.n_water < 1:
        raise ValueError(f"n_water must be positive, got {params.n_water}")

    if params.n_water > 100000:
        raise ValueError(f"n_water {params.n_water} is too large (maximum 100000)")

    # Validate lattice constant
    if params.a <= 0:
        raise ValueError(f"Lattice constant must be positive, got {params.a}")

    if params.a < 2.0 or params.a > 3.0:
        raise ValueError(
            f"Lattice constant {params.a} Å is unusual for graphene. "
            "Typical range is 2.0-3.0 Å (default 2.46 Å)"
        )

    # Validate thickness
    if params.thickness < 0:
        raise ValueError(f"Thickness must be non-negative, got {params.thickness}")

    if params.thickness > 10.0:
        raise ValueError(f"Thickness {params.thickness} Å is too large for graphene")

    # Validate graphene_vacuum
    if params.graphene_vacuum < 0:
        raise ValueError(f"graphene_vacuum must be non-negative, got {params.graphene_vacuum}")

    if params.graphene_vacuum > 20.0:
        raise ValueError(f"graphene_vacuum {params.graphene_vacuum} Å is too large")

    # Validate water density
    if params.water_density <= 0:
        raise ValueError(f"Water density must be positive, got {params.water_density}")

    if params.water_density < 0.5 or params.water_density > 1.5:
        raise ValueError(
            f"Water density {params.water_density} g/cm³ is unusual. "
            "Typical range is 0.5-1.5 g/cm³"
        )

    # Validate gaps
    if params.gap_above_graphene < 0:
        raise ValueError(
            f"gap_above_graphene must be non-negative, got {params.gap_above_graphene}"
        )

    if params.gap_above_graphene > 50:
        raise ValueError(f"gap_above_graphene {params.gap_above_graphene} Å is too large")

    if params.vacuum_above_water < 0:
        raise ValueError(
            f"vacuum_above_water must be non-negative, got {params.vacuum_above_water}"
        )

    if params.vacuum_above_water > 200:
        raise ValueError(f"vacuum_above_water {params.vacuum_above_water} Å is too large")

    # Validate water model
    if params.water_model not in WATER_MODELS:
        available = ", ".join(WATER_MODELS.keys())
        raise ValueError(f"Unknown water model '{params.water_model}'. Available: {available}")

    # Validate packmol tolerance
    if params.packmol_tolerance <= 0:
        raise ValueError(f"packmol_tolerance must be positive, got {params.packmol_tolerance}")

    if params.packmol_tolerance < 1.0:
        raise ValueError(
            f"packmol_tolerance {params.packmol_tolerance} Å is too small. "
            "Minimum recommended is 1.0 Å"
        )

    if params.packmol_tolerance > 5.0:
        raise ValueError(
            f"packmol_tolerance {params.packmol_tolerance} Å is too large. "
            "Maximum recommended is 5.0 Å"
        )

    # Validate output file
    output_path = Path(params.output_file)

    # Check if parent directory exists
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise ValueError(f"Parent path {output_path.parent} exists but is not a directory")

    # Determine output format
    if params.output_format:
        # User specified format explicitly
        valid_formats = ["xyz", "poscar", "vasp", "lammps", "lammps/dpmd", "lammpstrj"]
        if params.output_format not in valid_formats:
            raise ValueError(
                f"Unknown output format '{params.output_format}'. "
                f"Valid formats: {', '.join(valid_formats)}"
            )
    else:
        # Detect from file extension
        suffix = output_path.suffix.lower()
        name = output_path.name

        # Special case for POSCAR
        if name == "POSCAR" or name.startswith("POSCAR"):
            params.output_format = "poscar"
        elif suffix in SUPPORTED_FORMATS:
            params.output_format = SUPPORTED_FORMATS[suffix]
        else:
            raise ValueError(
                f"Cannot determine output format from file extension '{suffix}'. "
                "Please specify --output-format explicitly or use a supported extension: "
                ".xyz, .vasp, .lammps, .data, .lammpstrj"
            )

    # Validate elements list if provided
    if params.elements is not None:
        if not isinstance(params.elements, list):
            raise ValueError("elements must be a list of element symbols")

        if len(params.elements) == 0:
            raise ValueError("elements list cannot be empty")

        # Check for valid element symbols (basic check)
        valid_elements = {
            "H",
            "C",
            "N",
            "O",
            "F",
            "Na",
            "Cl",
            "K",
            "Ca",
            "Mg",
            "Li",
            "Br",
            "I",
            "Cs",
            "Rb",
            "Sr",
            "Ba",
            "S",
            "P",
        }
        for elem in params.elements:
            if elem not in valid_elements:
                raise ValueError(f"Invalid element symbol: {elem}")

        # Check for duplicates
        if len(params.elements) != len(set(params.elements)):
            raise ValueError("elements list contains duplicates")

        # Ensure C, O, H are in the list for graphene-water
        required = {"C", "O", "H"}
        provided = set(params.elements)
        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(
                f"elements list must contain C, O, and H for graphene-water. "
                f"Missing: {', '.join(missing)}"
            )


def get_water_model_params(water_model: str) -> dict:
    """
    Get water model parameters.

    Args:
        water_model: Name of the water model

    Returns:
        Dictionary with water model parameters

    Raises:
        ValueError: If water model is unknown
    """
    if water_model not in WATER_MODELS:
        raise ValueError(f"Unknown water model: {water_model}")

    return WATER_MODELS[water_model]
