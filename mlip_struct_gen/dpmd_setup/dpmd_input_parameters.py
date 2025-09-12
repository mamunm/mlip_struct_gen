"""Data classes for DPMD simulation input parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Any


@dataclass
class DPMDInputParameters:
    """Parameters for DPMD-based LAMMPS simulations."""
    
    # Required parameters
    dpmd_model_path: Union[str, Path]
    lammps_data_file: Union[str, Path]
    
    # Ensemble parameters
    ensemble: str = "NVT"
    temperature: float = 300.0  # Kelvin
    pressure: float = 1.0  # atm (only for NPT)
    
    # Time parameters
    timestep: float = 0.5  # femtoseconds
    equilibration_time: float = 100.0  # picoseconds
    production_time: float = 1000.0  # picoseconds
    
    # Output parameters
    output_file: str = "dpmd_water.in"
    dump_freq: float = 1.0  # picoseconds
    thermo_frequency: int = 100  # steps
    
    # Thermostat/barostat parameters
    thermostat_damping: float = 0.1  # picoseconds
    barostat_damping: float = 1.0  # picoseconds
    
    # Other parameters
    seed: int = 12345
    logger: Optional[Any] = None
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.dpmd_model_path = Path(self.dpmd_model_path)
        self.lammps_data_file = Path(self.lammps_data_file)
        self.ensemble = self.ensemble.upper()
    
    def validate(self):
        """Validate input parameters."""
        # Check if files exist
        if not self.dpmd_model_path.exists():
            raise FileNotFoundError(f"DPMD model file not found: {self.dpmd_model_path}")
        
        if not self.lammps_data_file.exists():
            raise FileNotFoundError(f"LAMMPS data file not found: {self.lammps_data_file}")
        
        # Check ensemble
        if self.ensemble not in ["NVE", "NVT", "NPT"]:
            raise ValueError(f"Invalid ensemble: {self.ensemble}. Must be NVE, NVT, or NPT")
        
        # Check numerical parameters
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive: {self.temperature}")
        
        if self.timestep <= 0:
            raise ValueError(f"Timestep must be positive: {self.timestep}")
        
        if self.equilibration_time < 0:
            raise ValueError(f"Equilibration time cannot be negative: {self.equilibration_time}")
        
        if self.production_time <= 0:
            raise ValueError(f"Production time must be positive: {self.production_time}")
        
        if self.dump_freq <= 0:
            raise ValueError(f"Dump frequency must be positive: {self.dump_freq}")
        
        if self.thermo_frequency <= 0:
            raise ValueError(f"Thermo frequency must be positive: {self.thermo_frequency}")
        
        if self.thermostat_damping <= 0:
            raise ValueError(f"Thermostat damping must be positive: {self.thermostat_damping}")
        
        if self.barostat_damping <= 0:
            raise ValueError(f"Barostat damping must be positive: {self.barostat_damping}")
        
        if self.ensemble == "NPT" and self.pressure <= 0:
            raise ValueError(f"Pressure must be positive for NPT ensemble: {self.pressure}")


@dataclass 
class DPMDWaterInputParameters(DPMDInputParameters):
    """Specialized parameters for DPMD water simulations.
    
    Inherits all parameters from DPMDInputParameters with
    water-specific defaults.
    """
    
    # Water-specific defaults
    timestep: float = 0.5  # Smaller timestep for water
    thermostat_damping: float = 0.1  # Fast thermostat for water
    
    # Additional water-specific parameters
    compute_rdf: bool = False  # Whether to compute radial distribution function
    rdf_cutoff: float = 8.0  # Angstroms
    rdf_nbins: int = 100
    
    compute_msd: bool = False  # Whether to compute mean squared displacement
    msd_sample_freq: int = 10  # Sample every N steps
    
    compute_hbonds: bool = False  # Whether to analyze hydrogen bonds
    hbond_cutoff: float = 3.5  # Angstroms
    hbond_angle: float = 30.0  # Degrees from 180


@dataclass
class DPMDSaltWaterInputParameters(DPMDWaterInputParameters):
    """Parameters for DPMD salt water (NaCl solution) simulations.
    
    Extends water parameters with salt-specific options.
    """
    
    # Salt concentration
    n_salt_pairs: int = 0  # Number of NaCl pairs
    
    # Ion-specific analysis
    compute_ion_rdf: bool = False  # Ion-water and ion-ion RDFs
    compute_ion_msd: bool = False  # Ion diffusion coefficients
    compute_coordination: bool = False  # Ion coordination numbers
    
    # Solvation shell analysis
    first_shell_cutoff: float = 3.2  # Angstroms for Na-O
    second_shell_cutoff: float = 5.5  # Angstroms
    
    def get_system_name(self) -> str:
        """Generate a descriptive system name."""
        if self.n_salt_pairs == 0:
            return "pure_water"
        else:
            return f"water_{self.n_salt_pairs}NaCl"