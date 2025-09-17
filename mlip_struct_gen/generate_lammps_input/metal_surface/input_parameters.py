"""Input parameters for metal surface LAMMPS simulations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class MetalSurfaceLAMMPSParameters:
    """Parameters for metal surface LAMMPS simulations.

    Parameters optimized for MLIP training data generation from metal surfaces.
    """

    # Required parameters
    data_file: str | Path
    metal_type: str  # Metal element (e.g., "Pt", "Au", "Cu")

    # Simulation ensemble and conditions
    ensemble: str = "NVT"  # NPT, NVT, or NVE
    temperatures: list[float] | float = field(default_factory=lambda: [330.0])  # K
    pressure: float = 1.0  # bar (for NPT)

    # Simulation times (in ps)
    equilibration_time: float = 100.0  # ps
    production_time: float = 500.0  # ps
    dump_frequency: float = 1.0  # ps for MLIP snapshots

    # Metal potential parameters
    lj_cutoff: float = 10.0  # LJ cutoff in Angstrom

    # Surface constraints
    fix_bottom_layers: int = 0  # Number of bottom layers to fix

    # Thermostat/barostat settings
    thermostat_damping: float = 100.0  # fs
    barostat_damping: float = 1000.0  # fs

    # Output options
    output_file: str | None = None  # Auto-generated if not specified
    coulomb_accuracy: float = 1.0e-4  # If charges are present

    # Advanced settings
    timestep: float = 1.0  # fs
    seed: int = 12345  # Random seed
    use_velocity_scaling: bool = False  # Use velocity scaling instead of Nose-Hoover
    log: bool = False  # Enable logging

    # MLIP-specific settings
    compute_stress: bool = True  # Compute per-atom stress for MLIP training
    compute_centro: bool = True  # Compute centrosymmetry parameter

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert temperatures to list if single value
        if not isinstance(self.temperatures, list):
            self.temperatures = [self.temperatures]

        # Convert data_file to Path
        self.data_file = Path(self.data_file)

        # Generate output filename if not specified
        if self.output_file is None:
            data_stem = self.data_file.stem
            if len(self.temperatures) == 1:
                self.output_file = f"in_{data_stem}_T{int(self.temperatures[0])}.lammps"
            else:
                # Will generate multiple files for different temperatures
                self.output_file = f"in_{data_stem}.lammps"
