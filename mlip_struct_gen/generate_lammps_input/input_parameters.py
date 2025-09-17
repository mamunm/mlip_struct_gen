"""Common LAMMPS input parameters for all system types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LAMMPSInputParameters:
    """Base parameters for LAMMPS input file generation focused on MLIP training data."""

    # Required input
    data_file: str | Path

    # Essential simulation parameters
    ensemble: str = "NVT"  # NPT, NVT, NVE (NVT default for better sampling)
    temperatures: list[float] | float = field(default_factory=lambda: [330.0])  # K
    pressure: float = 1.0  # bar (for NPT)

    # Simulation timeline (in picoseconds)
    equilibration_time: float = 100.0  # ps
    production_time: float = 500.0  # ps
    timestep: float = 1.0  # fs

    # MLIP sampling parameters (in picoseconds)
    dump_frequency: float = 1.0  # ps - how often to save snapshots for MLIP

    # Water model (if applicable)
    water_model: str = "SPC/E"  # SPC/E, TIP3P, TIP4P

    # Random seed for reproducibility
    seed: int = 12345

    # Output file (auto-generated if not specified)
    output_file: str | None = None

    # Advanced settings (usually not needed to change)
    thermostat_damping: float = 100.0  # fs
    barostat_damping: float = 1000.0  # fs
    coulomb_accuracy: float = 1.0e-4  # PPPM accuracy (1e-4 is sufficient for water)

    # Internal use (not user-facing)
    log: bool = False
    logger: Any = None  # MLIPLogger instance

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # Convert to Path if string
        if isinstance(self.data_file, str):
            self.data_file = Path(self.data_file)

        # Auto-generate output file name if not specified
        if self.output_file is None:
            data_stem = self.data_file.stem
            if isinstance(self.temperatures, list) and len(self.temperatures) > 1:
                self.output_file = f"in_{data_stem}_multiT.lammps"
            else:
                temp = (
                    self.temperatures[0]
                    if isinstance(self.temperatures, list)
                    else self.temperatures
                )
                self.output_file = f"in_{data_stem}_T{temp:.0f}.lammps"

        # Ensure temperatures is a list
        if not isinstance(self.temperatures, list):
            self.temperatures = [self.temperatures]

        # Convert times to steps
        self.equilibration_steps = int(self.equilibration_time * 1000 / self.timestep)
        self.production_steps = int(self.production_time * 1000 / self.timestep)
        self.dump_steps = int(self.dump_frequency * 1000 / self.timestep)

        # Set reasonable thermo frequency (every 1 ps)
        self.thermo_steps = int(1000 / self.timestep)

        # Validate ensemble
        if self.ensemble not in ["NPT", "NVT", "NVE"]:
            raise ValueError(f"Invalid ensemble: {self.ensemble}. Must be NPT, NVT, or NVE")

        # Validate temperatures
        for temp in self.temperatures:
            if temp <= 0:
                raise ValueError(f"Temperature must be positive: {temp}")
            if temp < 250 or temp > 400:
                print(f"Warning: Temperature {temp}K is outside typical range (250-400K) for water")

        # Validate times
        if self.equilibration_time < 10:
            print("Warning: Equilibration time < 10 ps may be too short")
        if self.production_time < 50:
            print("Warning: Production time < 50 ps may be too short for good sampling")
        if self.dump_frequency > self.production_time / 10:
            print("Warning: Dump frequency may be too low for good MLIP training data")
