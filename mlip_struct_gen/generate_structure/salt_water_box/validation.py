"""Parameter validation for salt water box generation."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .input_parameters import SaltWaterBoxGeneratorParameters

from ..templates.salt_models import get_salt_model
from ..templates.water_models import get_water_model


def validate_parameters(parameters: "SaltWaterBoxGeneratorParameters") -> None:
    """
    Comprehensive parameter validation and normalization for salt water box.

    Args:
        parameters: Parameters to validate and normalize

    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
    """
    # Box size validation and normalization (same as water_box)
    if parameters.box_size is None:
        # Box size will be computed from n_water
        if parameters.n_water is None:
            raise ValueError("Either box_size or n_water must be provided")
        # box_size will be computed later in the generator
    elif isinstance(parameters.box_size, int | float):
        # Convert single number to cubic box
        if parameters.box_size <= 0:
            raise ValueError("Box size must be positive")
        parameters.box_size = (
            float(parameters.box_size),
            float(parameters.box_size),
            float(parameters.box_size),
        )
    elif isinstance(parameters.box_size, list | tuple):
        # Validate and normalize list/tuple
        if len(parameters.box_size) != 3:
            raise ValueError("box_size as tuple/list must have exactly 3 dimensions")

        if not all(isinstance(s, int | float) for s in parameters.box_size):
            raise TypeError("All box_size dimensions must be numeric")

        if not all(s > 0 for s in parameters.box_size):
            raise ValueError("All box dimensions must be positive")

        parameters.box_size = tuple(float(s) for s in parameters.box_size)  # type: ignore[assignment]
    else:
        raise TypeError("box_size must be a number (cubic), tuple/list of 3 numbers, or None")

    # Additional box size validation (only if box_size is provided)
    if parameters.box_size is not None:
        if any(s > 1000.0 for s in parameters.box_size):
            raise ValueError("Box dimensions too large (>1000 Å). Check units.")

        # Minimum 10 Å for salt water boxes (larger than pure water)
        if any(s < 10.0 for s in parameters.box_size):
            raise ValueError(
                "Box dimensions too small (<10 Å). Minimum recommended size is 10 Å for salt solutions."
            )

    # Output file validation
    if not isinstance(parameters.output_file, str):
        raise TypeError("output_file must be a string")

    if not parameters.output_file.strip():
        raise ValueError("output_file cannot be empty")

    # Add extension based on format if not provided
    output_path = Path(parameters.output_file)
    if not output_path.suffix:
        format_extensions = {
            "xyz": ".xyz",
            "lammps": ".data",
            "poscar": "",  # POSCAR doesn't use an extension
        }
        extension = format_extensions.get(parameters.output_format, ".xyz")
        parameters.output_file = str(output_path) + extension

    # Salt type validation
    if not isinstance(parameters.salt_type, str):
        raise TypeError("salt_type must be a string")

    # Validate salt model exists
    try:
        get_salt_model(parameters.salt_type)
    except ValueError as e:
        raise ValueError(f"Invalid salt type: {e}") from e

    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")

    try:
        get_water_model(parameters.water_model)
    except ValueError as e:
        raise ValueError(f"Invalid water model: {e}") from e

    # Number of salt molecules validation
    if not isinstance(parameters.n_salt, int):
        raise TypeError("n_salt must be an integer")

    if parameters.n_salt < 0:
        raise ValueError("n_salt must be non-negative")

    if parameters.n_salt > 100000:
        raise ValueError("n_salt too large (>100000)")

    # Number of water molecules validation
    if parameters.n_water is not None:
        if not isinstance(parameters.n_water, int):
            raise TypeError("n_water must be an integer or None")

        if parameters.n_water <= 0:
            raise ValueError("n_water must be positive")

        if parameters.n_water > 1000000:
            raise ValueError("n_water too large (>1M)")

    # Density validation
    if parameters.density is not None:
        if not isinstance(parameters.density, int | float):
            raise TypeError("density must be numeric or None")

        if parameters.density <= 0:
            raise ValueError("density must be positive")

        # Wider range for salt solutions
        if parameters.density > 2.0:
            raise ValueError("density too high (>2.0 g/cm³)")

        if parameters.density < 0.5:
            raise ValueError("density too low (<0.5 g/cm³)")

    # Validate parameter combinations (same as water_box)
    # Valid combinations:
    # 1. box_size only (uses default density)
    # 2. box_size + density (custom density)
    # 3. box_size + n_water (exact water molecules)
    # 4. n_water only (uses default density to compute box)
    # 5. n_water + density (computes box for molecules at density)

    # Invalid: all three specified
    if (
        parameters.box_size is not None
        and parameters.n_water is not None
        and parameters.density is not None
    ):
        raise ValueError(
            "Cannot specify all three: box_size, n_water, and density. Choose at most two."
        )

    # Volume handling validation
    if not isinstance(parameters.include_salt_volume, bool):
        raise TypeError("include_salt_volume must be a boolean")

    # Tolerance validation
    if not isinstance(parameters.tolerance, int | float):
        raise TypeError("tolerance must be numeric")

    if parameters.tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if parameters.tolerance > 10.0:
        raise ValueError("tolerance too large (>10 Å)")

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

    # Logging parameters validation
    if not isinstance(parameters.log, bool):
        raise TypeError("log must be a boolean")

    if parameters.logger is not None:
        try:
            from ...utils.logger import MLIPLogger

            if not isinstance(parameters.logger, MLIPLogger):
                raise TypeError("logger must be an MLIPLogger instance or None")
        except ImportError:
            raise ImportError("MLIPLogger not available. Check utils.logger module.") from None

    # Check for reasonable salt concentration (only if box_size is provided)
    if parameters.box_size is not None and parameters.n_salt > 0:
        box_volume_l = np.prod(parameters.box_size) * 1e-27  # Å³ to L

        # Rough concentration check
        na = 6.022e23
        concentration = parameters.n_salt / (na * box_volume_l)

        if concentration > 10.0:
            raise ValueError(
                f"Salt concentration too high (~{concentration:.1f} M). "
                "Maximum reasonable concentration is ~10 M."
            )

        # Warning for very high concentrations
        if concentration > 5.0 and not parameters.include_salt_volume:
            import warnings

            warnings.warn(
                f"High salt concentration (~{concentration:.1f} M) detected. "
                "Consider setting include_salt_volume=True for better accuracy.",
                UserWarning,
                stacklevel=2,
            )
