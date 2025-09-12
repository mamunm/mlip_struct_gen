"""Parameter validation for LAMMPS MD simulation setup."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_parameters import LAMMPSInputParameters


def validate_parameters(parameters: "LAMMPSInputParameters") -> None:
    """
    Comprehensive parameter validation and normalization.
    
    Args:
        parameters: Parameters to validate and normalize
        
    Raises:
        ValueError: If any parameter is invalid
        TypeError: If parameter types are incorrect
        FileNotFoundError: If required files don't exist
    """
    # LAMMPS data file validation
    if not isinstance(parameters.lammps_data_file, str):
        raise TypeError("lammps_data_file must be a string")
    
    if not parameters.lammps_data_file.strip():
        raise ValueError("lammps_data_file cannot be empty")
    
    # Output file validation and normalization
    if not isinstance(parameters.output_file, str):
        raise TypeError("output_file must be a string")
    
    if not parameters.output_file.strip():
        raise ValueError("output_file cannot be empty")
    
    # Add .in extension if not provided
    output_path = Path(parameters.output_file)
    if not output_path.suffix:
        parameters.output_file = str(output_path) + ".in"
    
    # Water model validation
    if not isinstance(parameters.water_model, str):
        raise TypeError("water_model must be a string")
    
    valid_water_models = ["SPC/E", "TIP3P", "TIP4P"]
    if parameters.water_model not in valid_water_models:
        raise ValueError(
            f"Invalid water_model '{parameters.water_model}'. "
            f"Supported models: {', '.join(valid_water_models)}"
        )
    
    # Ensemble validation
    if not isinstance(parameters.ensemble, str):
        raise TypeError("ensemble must be a string")
    
    valid_ensembles = ["NVE", "NVT", "NPT"]
    if parameters.ensemble not in valid_ensembles:
        raise ValueError(
            f"Invalid ensemble '{parameters.ensemble}'. "
            f"Supported ensembles: {', '.join(valid_ensembles)}"
        )
    
    # Temperature validation
    if not isinstance(parameters.temperature, (int, float)):
        raise TypeError("temperature must be numeric")
    
    if parameters.temperature <= 0:
        raise ValueError("temperature must be positive")
    
    if parameters.temperature < 250 or parameters.temperature > 400:
        raise ValueError(
            "temperature out of range (250-400 K). "
            "Use caution with extreme temperatures for water simulations."
        )
    
    # Pressure validation (only for NPT)
    if parameters.ensemble == "NPT":
        if not isinstance(parameters.pressure, (int, float)):
            raise TypeError("pressure must be numeric")
        
        if parameters.pressure <= 0:
            raise ValueError("pressure must be positive")
        
        if parameters.pressure < 0.1 or parameters.pressure > 10.0:
            raise ValueError("pressure out of range (0.1-10.0 atm)")
    
    # Time validation
    if not isinstance(parameters.equilibration_time, (int, float)):
        raise TypeError("equilibration_time must be numeric")
    
    if parameters.equilibration_time <= 0:
        raise ValueError("equilibration_time must be positive")
    
    if parameters.equilibration_time < 10.0:
        raise ValueError("equilibration_time too small (<10 ps). Minimum recommended: 10 ps")
    
    if parameters.equilibration_time > 1000.0:
        raise ValueError("equilibration_time too large (>1000 ps). Consider shorter equilibration.")
    
    if not isinstance(parameters.production_time, (int, float)):
        raise TypeError("production_time must be numeric")
    
    if parameters.production_time <= 0:
        raise ValueError("production_time must be positive")
    
    if parameters.production_time < 50.0:
        raise ValueError("production_time too small (<50 ps). Minimum recommended: 50 ps")
    
    if parameters.production_time > 10000.0:
        raise ValueError("production_time too large (>10000 ps). Consider shorter simulation.")
    
    # Dump frequency validation
    if not isinstance(parameters.dump_freq, (int, float)):
        raise TypeError("dump_freq must be numeric")
    
    if parameters.dump_freq <= 0:
        raise ValueError("dump_freq must be positive")
    
    if parameters.dump_freq < 0.1 or parameters.dump_freq > 100.0:
        raise ValueError("dump_freq out of range (0.1-100.0 ps)")
    
    # Timestep validation
    if not isinstance(parameters.timestep, (int, float)):
        raise TypeError("timestep must be numeric")
    
    if parameters.timestep <= 0:
        raise ValueError("timestep must be positive")
    
    if parameters.timestep < 0.5 or parameters.timestep > 2.0:
        raise ValueError("timestep out of range (0.5-2.0 fs)")
    
    # Pair style validation
    if not isinstance(parameters.pair_style, str):
        raise TypeError("pair_style must be a string")
    
    if not parameters.pair_style.strip():
        raise ValueError("pair_style cannot be empty")
    
    # Common LAMMPS pair styles for molecular simulations
    valid_pair_styles = [
        "lj/cut/coul/long",
        "lj/cut/coul/cut", 
        "lj/cut/coul/debye",
        "lj/charmm/coul/long",
        "lj/charmm/coul/charmm",
        "lj/cut",
        "lj/expand"
    ]
    
    if parameters.pair_style not in valid_pair_styles:
        # Allow custom pair styles but warn about common ones
        import warnings
        warnings.warn(
            f"Pair style '{parameters.pair_style}' is not in the list of common styles: "
            f"{', '.join(valid_pair_styles)}. Make sure it's a valid LAMMPS pair style."
        )
    
    # Check if long-range electrostatics are needed
    needs_kspace = "coul/long" in parameters.pair_style
    
    # Pair style cutoff validation
    if not isinstance(parameters.pair_style_cutoff, (int, float)):
        raise TypeError("pair_style_cutoff must be numeric")
    
    if parameters.pair_style_cutoff <= 0:
        raise ValueError("pair_style_cutoff must be positive")
    
    if parameters.pair_style_cutoff < 6.0 or parameters.pair_style_cutoff > 15.0:
        raise ValueError("pair_style_cutoff out of range (6.0-15.0 Å)")
    
    # Auto-determine SHAKE usage if not specified
    if parameters.use_shake is None:
        # All common water models benefit from SHAKE constraints
        parameters.use_shake = True
    
    if not isinstance(parameters.use_shake, bool):
        raise TypeError("use_shake must be a boolean")
    
    # SHAKE tolerance validation
    if parameters.use_shake:
        if not isinstance(parameters.shake_tolerance, (int, float)):
            raise TypeError("shake_tolerance must be numeric")
        
        if parameters.shake_tolerance <= 0:
            raise ValueError("shake_tolerance must be positive")
        
        if parameters.shake_tolerance > 1e-4:
            raise ValueError("shake_tolerance too large (>1e-4). Recommended: 1e-6")
    
    # Thermostat damping validation (for NVT and NPT)
    if parameters.ensemble in ["NVT", "NPT"]:
        if not isinstance(parameters.thermostat_damping, (int, float)):
            raise TypeError("thermostat_damping must be numeric")
        
        if parameters.thermostat_damping <= 0:
            raise ValueError("thermostat_damping must be positive")
        
        if parameters.thermostat_damping < 0.05 or parameters.thermostat_damping > 1.0:
            raise ValueError("thermostat_damping out of range (0.05-1.0 ps)")
    
    # Barostat damping validation (for NPT only)
    if parameters.ensemble == "NPT":
        if not isinstance(parameters.barostat_damping, (int, float)):
            raise TypeError("barostat_damping must be numeric")
        
        if parameters.barostat_damping <= 0:
            raise ValueError("barostat_damping must be positive")
        
        if parameters.barostat_damping < 0.5 or parameters.barostat_damping > 5.0:
            raise ValueError("barostat_damping out of range (0.5-5.0 ps)")
    
    # Frequency validations
    for freq_name, freq_value in [
        ("output_frequency", parameters.output_frequency),
        ("thermo_frequency", parameters.thermo_frequency)
    ]:
        if not isinstance(freq_value, int):
            raise TypeError(f"{freq_name} must be an integer")
        
        if freq_value <= 0:
            raise ValueError(f"{freq_name} must be positive")
        
        # Convert production time to steps for comparison
        production_steps = int(parameters.production_time * 1000 / parameters.timestep)
        if freq_value > production_steps:
            raise ValueError(f"{freq_name} cannot be larger than production steps ({production_steps})")
    
    # Seed validation
    if not isinstance(parameters.seed, int):
        raise TypeError("seed must be an integer")
    
    if parameters.seed < 0:
        raise ValueError("seed must be non-negative")
    
    # Logging parameters validation
    if not isinstance(parameters.log, bool):
        raise TypeError("log must be a boolean")
    
    if parameters.logger is not None:
        # Import here to avoid circular imports
        try:
            from ..utils.logger import MLIPLogger
            if not isinstance(parameters.logger, MLIPLogger):
                raise TypeError("logger must be an MLIPLogger instance or None")
        except ImportError:
            raise ImportError("MLIPLogger not available. Check utils.logger module.")
    
    # Consistency checks
    production_steps = int(parameters.production_time * 1000 / parameters.timestep)
    equilibration_steps = int(parameters.equilibration_time * 1000 / parameters.timestep)
    
    if parameters.output_frequency > production_steps // 10:
        raise ValueError(
            "output_frequency too large. Should be at most production_steps/10 "
            "to ensure reasonable trajectory sampling."
        )
    
    if parameters.thermo_frequency > equilibration_steps // 10:
        raise ValueError(
            "thermo_frequency too large. Should be at most equilibration_steps/10 "
            "to monitor equilibration progress."
        )