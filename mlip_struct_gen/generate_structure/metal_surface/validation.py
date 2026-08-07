"""Validation for metal surface generation parameters."""

from pathlib import Path

from ..composition import (  # noqa: F401  (re-exported for compatibility)
    DEFAULT_LATTICE_CONSTANTS,
    SUPPORTED_METALS,
    parse_composition,
    vegard_lattice_constant,
)
from .input_parameters import MetalSurfaceParameters


def validate_parameters(params: MetalSurfaceParameters) -> None:
    """
    Validate metal surface generation parameters.

    Args:
        params: Parameters to validate

    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate metal (single element or alloy composition string)
    composition = parse_composition(params.metal)

    unsupported = [el for el in composition if el not in SUPPORTED_METALS]
    if unsupported:
        raise ValueError(
            f"Metal(s) {', '.join(unsupported)} not supported. "
            f"Supported metals: {', '.join(sorted(SUPPORTED_METALS))}"
        )

    # Validate size
    if not params.size or len(params.size) != 3:
        raise ValueError("Size must be a tuple of 3 integers (nx, ny, nz)")

    nx, ny, nz = params.size

    if not all(isinstance(x, int) for x in params.size):
        raise ValueError("Size values must be integers")

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

    # Validate vacuum
    if params.vacuum < 0:
        raise ValueError(f"Vacuum ({params.vacuum} Å) must be non-negative")

    if params.vacuum > 50:
        raise ValueError(f"Vacuum ({params.vacuum} Å) should not exceed 50 Å")

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

    if params.fix_bottom_layers > nz - 1:
        raise ValueError(
            f"fix_bottom_layers ({params.fix_bottom_layers}) must leave at least "
            f"1 free layer (total layers: {nz})"
        )

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
                f"Warning: Unrecognized file extension '{suffix}'. Will use XYZ format by default."
            )


def get_lattice_constant(metal: str, custom_lattice: float | None = None) -> float:
    """
    Get lattice constant for a metal or alloy composition.

    For alloys the default is the Vegard's-law (composition-weighted)
    average of the per-element lattice constants.

    Args:
        metal: Metal element symbol or composition string (e.g. "CoCrFeMnNi")
        custom_lattice: Custom lattice constant (optional)

    Returns:
        Lattice constant in Angstroms

    Raises:
        ValueError: If metal is not supported and no custom lattice is provided
    """
    return vegard_lattice_constant(parse_composition(metal), custom_lattice)
