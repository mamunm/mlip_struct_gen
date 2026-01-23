"""Parameter validation for constrained water box generation."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_parameters import ConstrainedWaterBoxParameters

from ..templates.water_models import get_water_model


def validate_parameters(parameters: "ConstrainedWaterBoxParameters") -> None:
    """
    Validate and normalize parameters for constrained water box generation.

    Args:
        parameters: Parameters to validate and normalize

    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
    """
    # Count how many of the 3 main parameters are provided
    params_count = sum(
        [
            parameters.box_size is not None,
            parameters.n_water is not None,
            parameters.density is not None,
        ]
    )

    if params_count < 2:
        raise ValueError("Must specify exactly 2 of: box_size, n_water, density")
    if params_count > 2:
        raise ValueError(
            "Cannot specify all three of box_size, n_water, and density. Choose exactly 2."
        )

    # Box size validation and normalization
    if parameters.box_size is not None:
        if isinstance(parameters.box_size, int | float):
            if parameters.box_size <= 0:
                raise ValueError("Box size must be positive")
            parameters.box_size = (
                float(parameters.box_size),
                float(parameters.box_size),
                float(parameters.box_size),
            )
        elif isinstance(parameters.box_size, list | tuple):
            if len(parameters.box_size) != 3:
                raise ValueError("box_size as tuple/list must have exactly 3 dimensions")
            if not all(isinstance(s, int | float) for s in parameters.box_size):
                raise TypeError("All box_size dimensions must be numeric")
            if not all(s > 0 for s in parameters.box_size):
                raise ValueError("All box dimensions must be positive")
            parameters.box_size = tuple(float(s) for s in parameters.box_size)
        else:
            raise TypeError("box_size must be a number, tuple/list of 3 numbers, or None")

        if any(s > 1000.0 for s in parameters.box_size):
            raise ValueError("Box dimensions too large (>1000 A)")
        if any(s < 4.0 for s in parameters.box_size):
            raise ValueError("Box dimensions too small (<4 A)")

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

    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")
    try:
        get_water_model(parameters.water_model)
    except ValueError as e:
        raise ValueError(f"Invalid water model: {e}") from e

    # n_water validation
    if parameters.n_water is not None:
        if not isinstance(parameters.n_water, int):
            raise TypeError("n_water must be an integer")
        if parameters.n_water <= 0:
            raise ValueError("n_water must be positive")
        if parameters.n_water > 1000000:
            raise ValueError("n_water too large (>1M)")

    # Density validation
    if parameters.density is not None:
        if not isinstance(parameters.density, int | float):
            raise TypeError("density must be numeric")
        if parameters.density <= 0:
            raise ValueError("density must be positive")
        if parameters.density > 5.0:
            raise ValueError("density too high (>5 g/cm3)")
        if parameters.density < 0.1:
            raise ValueError("density too low (<0.1 g/cm3)")

    # Distance constraints validation
    for constraint in parameters.distance_constraints:
        if not isinstance(constraint.element1, str):
            raise TypeError("Distance constraint element1 must be a string")
        if not isinstance(constraint.element2, str):
            raise TypeError("Distance constraint element2 must be a string")
        if constraint.element1 not in ["O", "H"]:
            raise ValueError(f"Invalid element: {constraint.element1}. Must be O or H")
        if constraint.element2 not in ["O", "H"]:
            raise ValueError(f"Invalid element: {constraint.element2}. Must be O or H")
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
    if not isinstance(parameters.tolerance, int | float) or parameters.tolerance <= 0:
        raise ValueError("tolerance must be positive")
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
