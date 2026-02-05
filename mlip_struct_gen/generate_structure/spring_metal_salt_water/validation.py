"""Validation for spring metal-salt-water parameters."""

from .input_parameters import SpringMetalSaltWaterParameters

# Default lattice constants in Angstroms
LATTICE_CONSTANTS = {
    "Cu": 3.536,
    "Pt": 3.901,
}


def get_lattice_constant(metal: str, custom_value: float | None = None) -> float:
    """Get lattice constant for a metal."""
    if custom_value is not None:
        return custom_value
    if metal not in LATTICE_CONSTANTS:
        raise ValueError(f"Unknown metal: {metal}. Provide --lattice-constant.")
    return LATTICE_CONSTANTS[metal]


def validate_parameters(params: SpringMetalSaltWaterParameters) -> None:
    """
    Validate spring metal-salt-water parameters.

    Args:
        params: Parameters to validate

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate metal
    if params.metal not in LATTICE_CONSTANTS and params.lattice_constant is None:
        raise ValueError(f"Unknown metal: {params.metal}. Provide lattice_constant.")

    # Validate spring constraints exist
    has_constraints = (
        params.metal_water_spring_constraints
        or params.metal_ion_spring_constraints
        or params.spring_constraints
    )
    if not has_constraints:
        raise ValueError("Must specify at least one spring constraint")

    # Validate spring constraint values
    for constraint in params.metal_water_spring_constraints:
        if constraint.distance <= 0:
            raise ValueError(f"Spring distance must be positive: {constraint.distance}")
        if constraint.k_spring < 0:
            raise ValueError(f"Spring constant must be non-negative: {constraint.k_spring}")

    for constraint in params.metal_ion_spring_constraints:
        if constraint.distance <= 0:
            raise ValueError(f"Spring distance must be positive: {constraint.distance}")
        if constraint.k_spring < 0:
            raise ValueError(f"Spring constant must be non-negative: {constraint.k_spring}")

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
