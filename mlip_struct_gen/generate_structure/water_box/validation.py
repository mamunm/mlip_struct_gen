"""Parameter validation for water box generation."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .input_parameters import WaterBoxGeneratorParameters

from ..templates.water_models import get_water_density, get_water_model


def validate_parameters(parameters: "WaterBoxGeneratorParameters") -> None:
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
        # Box size will be computed from n_water - validate n_water is provided
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

        if any(s < 4.0 for s in parameters.box_size):
            raise ValueError("Box dimensions too small (<4 Å). Minimum recommended size is 4 Å.")

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
            "lammps/dpmd": ".data",  # Same extension as regular LAMMPS
            "poscar": "",  # POSCAR doesn't use an extension
            "lammpstrj": ".lammpstrj",
        }
        extension = format_extensions.get(parameters.output_format, ".xyz")
        parameters.output_file = str(output_path) + extension

    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")

    # Validate water model exists and is supported
    try:
        get_water_model(parameters.water_model)
    except ValueError as e:
        raise ValueError(f"Invalid water model: {e}") from e

    # Number of molecules validation
    if parameters.n_water is not None:
        if not isinstance(parameters.n_water, int):
            raise TypeError("n_water must be an integer or None")

        if parameters.n_water <= 0:
            raise ValueError("n_water must be positive")

        if parameters.n_water > 1000000:
            raise ValueError("n_water too large (>1M). Consider using density instead.")

    # Density validation
    if parameters.density is not None:
        if not isinstance(parameters.density, int | float):
            raise TypeError("density must be numeric or None")

        if parameters.density <= 0:
            raise ValueError("density must be positive")

        if parameters.density > 5.0:
            raise ValueError("density too high (>5 g/cm³). Water density is ~1 g/cm³.")

        if parameters.density < 0.1:
            raise ValueError("density too low (<0.1 g/cm³). Check units.")

    # Validate parameter combinations
    # Valid combinations:
    # 1. box_size only (uses default density)
    # 2. box_size + density (custom density)
    # 3. box_size + n_water (fills box with exact n_water)
    # 4. n_water only (uses default density to compute box)
    # 5. n_water + density (computes box for n_water at specified density)

    # Invalid: all three specified
    if (
        parameters.box_size is not None
        and parameters.n_water is not None
        and parameters.density is not None
    ):
        raise ValueError(
            "Cannot specify all three of box_size, n_water, and density. Choose at most two."
        )

    # Tolerance validation
    if not isinstance(parameters.tolerance, int | float):
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

    valid_formats = ["xyz", "lammps", "lammps/dpmd", "poscar", "lammpstrj"]
    if parameters.output_format not in valid_formats:
        raise ValueError(
            f"Invalid output_format '{parameters.output_format}'. "
            f"Supported formats: {', '.join(valid_formats)}"
        )

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
            raise ImportError("MLIPLogger not available. Check utils.logger module.") from None

    # Check for reasonable molecule density (only if box_size is provided)
    if (
        parameters.box_size is not None
        and parameters.n_water is None
        and parameters.density is None
    ):
        # Will use water model's default density - check if reasonable
        model_density = get_water_density(parameters.water_model)
        box_volume_cm3 = np.prod(parameters.box_size) * 1e-24

        # Calculate estimated molecules using model density
        water_molar_mass = 18.015  # g/mol
        na = 6.022e23  # Avogadro's number
        mass_g = model_density * box_volume_cm3
        moles = mass_g / water_molar_mass
        estimated_molecules = int(moles * na)

        if estimated_molecules == 0:
            raise ValueError("Box too small to fit any water molecules")

        if estimated_molecules > 100000:
            raise ValueError(
                f"Box size will result in {estimated_molecules} molecules using {parameters.water_model} density. "
                "Consider using a smaller box or specify n_water explicitly."
            )
