"""Validation for spring salt water box parameters."""

from .input_parameters import SpringSaltWaterBoxParameters


def validate_parameters(params: SpringSaltWaterBoxParameters) -> None:
    """
    Validate spring salt water box parameters.

    Args:
        params: Parameters to validate

    Raises:
        ValueError: If parameters are invalid
    """
    # Count specified box parameters
    specified = sum(
        [
            params.box_size is not None,
            params.n_water is not None,
            params.density is not None,
        ]
    )

    if specified < 2:
        raise ValueError("Must specify at least 2 of: box_size, n_water, density")

    if specified > 2:
        raise ValueError("Cannot specify all three: box_size, n_water, density")

    # Validate spring constraints
    if not params.spring_constraints:
        raise ValueError("Must specify at least one spring constraint")

    for constraint in params.spring_constraints:
        if constraint.distance <= 0:
            raise ValueError(f"Spring distance must be positive: {constraint.distance}")
        if constraint.k_spring < 0:
            raise ValueError(f"Spring constant must be non-negative: {constraint.k_spring}")

    # Validate MD parameters
    if params.nsteps <= 0:
        raise ValueError(f"nsteps must be positive: {params.nsteps}")

    if params.temp <= 0:
        raise ValueError(f"Temperature must be positive: {params.temp}")

    if params.timestep <= 0:
        raise ValueError(f"Timestep must be positive: {params.timestep}")
