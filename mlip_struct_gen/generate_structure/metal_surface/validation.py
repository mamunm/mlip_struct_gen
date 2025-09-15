"""Validation for metal surface generation parameters."""

from pathlib import Path
from typing import Set

from .input_parameters import MetalSurfaceParameters


# Supported FCC metals with experimental lattice constants (Angstroms)
SUPPORTED_METALS: Set[str] = {
    "Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt",
    "Pb", "Rh", "Ir", "Ca", "Sr", "Yb"
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
    "Yb": 5.485
}


def validate_parameters(params: MetalSurfaceParameters) -> None:
    """
    Validate metal surface generation parameters.

    Args:
        params: Parameters to validate

    Raises:
        ValueError: If any parameter is invalid
    """
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
        raise ValueError("Size must be a tuple of 3 integers (nx, ny, nz)")

    nx, ny, nz = params.size

    if not all(isinstance(x, int) for x in params.size):
        raise ValueError("Size values must be integers")

    if nx < 1 or ny < 1:
        raise ValueError(f"Lateral dimensions (nx={nx}, ny={ny}) must be at least 1")

    if nz < 3:
        raise ValueError(f"Number of layers (nz={nz}) must be at least 3 for proper surface representation")

    if nx > 20 or ny > 20:
        raise ValueError(f"Lateral dimensions (nx={nx}, ny={ny}) should not exceed 20 for computational efficiency")

    if nz > 20:
        raise ValueError(f"Number of layers (nz={nz}) should not exceed 20 for computational efficiency")

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
        valid_formats = {"xyz", "vasp", "poscar", "lammps", "data"}
        if params.output_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid output format '{params.output_format}'. "
                f"Supported formats: {', '.join(valid_formats)}"
            )
    else:
        # Check if file extension is recognizable
        suffix = output_path.suffix.lower()
        valid_extensions = {".xyz", ".vasp", ".poscar", ".lammps", ".data"}
        if suffix and suffix not in valid_extensions and output_path.name.upper() != "POSCAR":
            print(f"Warning: Unrecognized file extension '{suffix}'. Will use XYZ format by default.")


def get_lattice_constant(metal: str, custom_lattice: float = None) -> float:
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