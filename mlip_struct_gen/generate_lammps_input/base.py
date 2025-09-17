"""Base class for LAMMPS input file generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.logger import MLIPLogger


@dataclass
class BaseLAMMPSGenerator(ABC):
    """Abstract base class for LAMMPS input file generators."""

    parameters: Any
    logger: MLIPLogger | None = None

    def __post_init__(self) -> None:
        """Initialize the generator."""
        self.data_path = Path(self.parameters.data_file)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Parse data file to get system info
        self.system_info = self._parse_data_file()

        # Setup output paths
        self._setup_output_paths()

        # Initialize logger if needed
        if self.parameters.log and not self.logger:
            self.logger = MLIPLogger()

    def _parse_data_file(self) -> dict[str, Any]:
        """Parse LAMMPS data file to extract system information."""
        info: dict[str, Any] = {
            "n_atoms": 0,
            "n_bonds": 0,
            "n_angles": 0,
            "n_dihedrals": 0,
            "n_atom_types": 0,
            "n_bond_types": 0,
            "n_angle_types": 0,
            "box": {"xlo": 0.0, "xhi": 0.0, "ylo": 0.0, "yhi": 0.0, "zlo": 0.0, "zhi": 0.0},
            "masses": {},
            "has_charges": False,
            "has_molecules": False,
            "atom_style": None,
        }

        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse header section
                if "atoms" in line:
                    info["n_atoms"] = int(line.split()[0])
                elif "bonds" in line and "bond" not in line:
                    info["n_bonds"] = int(line.split()[0])
                elif "angles" in line and "angle" not in line:
                    info["n_angles"] = int(line.split()[0])
                elif "atom types" in line:
                    info["n_atom_types"] = int(line.split()[0])
                elif "bond types" in line:
                    info["n_bond_types"] = int(line.split()[0])
                elif "angle types" in line:
                    info["n_angle_types"] = int(line.split()[0])
                elif "xlo xhi" in line:
                    parts = line.split()
                    info["box"]["xlo"] = float(parts[0])
                    info["box"]["xhi"] = float(parts[1])
                elif "ylo yhi" in line:
                    parts = line.split()
                    info["box"]["ylo"] = float(parts[0])
                    info["box"]["yhi"] = float(parts[1])
                elif "zlo zhi" in line:
                    parts = line.split()
                    info["box"]["zlo"] = float(parts[0])
                    info["box"]["zhi"] = float(parts[1])
                elif "Masses" in line:
                    # Read mass section
                    next(f)  # Skip blank line
                    for _ in range(info["n_atom_types"]):
                        mass_line = next(f).strip()
                        if mass_line:
                            parts = mass_line.split()
                            info["masses"][int(parts[0])] = float(parts[1])
                elif "Atoms" in line:
                    # Determine atom style from first atom line
                    next(f)  # Skip blank line
                    first_atom = next(f).strip().split()
                    if len(first_atom) >= 6:
                        # Check if there's a charge column (usually 4th position)
                        try:
                            float(first_atom[3])
                            info["has_charges"] = True
                            info["atom_style"] = "full"
                        except (ValueError, IndexError):
                            info["atom_style"] = "atomic"

                    # Check for molecule IDs (usually 2nd position in full style)
                    if info["atom_style"] == "full" and len(first_atom) >= 2:
                        try:
                            mol_id = int(first_atom[1])
                            if mol_id > 0:
                                info["has_molecules"] = True
                        except (ValueError, IndexError):
                            pass
                    break

        return info

    def _setup_output_paths(self) -> None:
        """Setup output file paths based on parameters."""
        if isinstance(self.parameters.temperatures, list) and len(self.parameters.temperatures) > 1:
            # Multiple temperature files
            self.output_files = []
            data_stem = self.parameters.data_file.stem
            for temp in self.parameters.temperatures:
                output_file = f"in_{data_stem}_T{temp:.0f}.lammps"
                self.output_files.append(output_file)
        else:
            # Single output file
            self.output_files = [self.parameters.output_file]

    @abstractmethod
    def _generate_force_field_section(self, temperature: float) -> list[str]:
        """Generate force field definition section."""
        pass

    @abstractmethod
    def _generate_settings_section(self) -> list[str]:
        """Generate simulation settings section."""
        pass

    def _generate_initialization_section(self) -> list[str]:
        """Generate LAMMPS initialization section."""
        lines = []
        lines.append("# LAMMPS input file generated by MLIP Structure Generator")
        lines.append(f"# Data file: {self.parameters.data_file}")
        lines.append("")

        # Units and atom style
        lines.append("# Initialization")
        lines.append("units real")

        # Determine atom style from data file
        if self.system_info["has_molecules"]:
            lines.append("atom_style full")
        else:
            lines.append("atom_style atomic")

        # Boundary conditions
        if hasattr(self.parameters, "periodic_z") and not self.parameters.periodic_z:
            lines.append("boundary p p f")  # Fixed z for surfaces
        else:
            lines.append("boundary p p p")  # Fully periodic

        # Processor grid if specified
        if hasattr(self.parameters, "processors") and self.parameters.processors:
            px, py, pz = self.parameters.processors
            lines.append(f"processors {px} {py} {pz}")

        lines.append("")
        return lines

    def _generate_system_setup_section(self) -> list[str]:
        """Generate system setup section."""
        lines = []
        lines.append("# Read data file")
        lines.append(f"read_data {self.parameters.data_file}")
        lines.append("")

        return lines

    def _generate_groups_section(self) -> list[str]:
        """Generate atom groups section."""
        lines = []
        lines.append("# Groups")
        # Subclasses will override to add specific groups
        return lines

    def _generate_computes_section(self) -> list[str]:
        """Generate computes section."""
        lines = []
        lines.append("# Computes")
        lines.append("compute pe all pe")
        lines.append("compute ke all ke")
        lines.append("compute temp all temp")
        lines.append("compute press all pressure temp")
        lines.append("")
        return lines

    def _generate_equilibration_section(self, temperature: float) -> list[str]:
        """Generate equilibration section."""
        lines = []
        lines.append("# Equilibration")
        lines.append(f"velocity all create {temperature} {self.parameters.seed} dist gaussian")

        # Minimize first
        lines.append("minimize 1.0e-4 1.0e-6 1000 10000")
        lines.append("")

        # Equilibration run
        if self.parameters.ensemble == "NPT":
            lines.append(
                f"fix 1 all npt temp {temperature} {temperature} 100.0 iso {self.parameters.pressure} {self.parameters.pressure} 1000.0"
            )
        elif self.parameters.ensemble == "NVT":
            lines.append(f"fix 1 all nvt temp {temperature} {temperature} 100.0")
        else:  # NVE
            lines.append("fix 1 all nve")

        lines.append(f"timestep {self.parameters.timestep}")
        lines.append(f"thermo {self.parameters.thermo_frequency}")
        lines.append("thermo_style custom step temp press pe ke etotal density")

        # Restart file
        if self.parameters.restart_frequency > 0:
            lines.append(f"restart {self.parameters.restart_frequency} restart.*.rst")

        lines.append(f"run {self.parameters.equilibration_steps}")
        lines.append("unfix 1")
        lines.append("")

        return lines

    def _generate_production_section(self, temperature: float) -> list[str]:
        """Generate production section with MLIP training data output."""
        lines = []
        lines.append("# Production run for MLIP training data")

        # Production ensemble
        if self.parameters.ensemble == "NPT":
            lines.append(
                f"fix 1 all npt temp {temperature} {temperature} 100.0 iso {self.parameters.pressure} {self.parameters.pressure} 1000.0"
            )
        elif self.parameters.ensemble == "NVT":
            lines.append(f"fix 1 all nvt temp {temperature} {temperature} 100.0")
        else:  # NVE
            lines.append("fix 1 all nve")

        # Dump for MLIP training
        dump_items = ["id", "type", "x", "y", "z"]
        if self.parameters.dump_forces:
            dump_items.extend(["fx", "fy", "fz"])
        if self.parameters.dump_velocities:
            dump_items.extend(["vx", "vy", "vz"])

        # Trajectory file name
        if len(self.parameters.temperatures) > 1:
            traj_file = f"{self.parameters.trajectory_prefix}_T{temperature:.0f}.dump"
        else:
            traj_file = f"{self.parameters.trajectory_prefix}.dump"

        lines.append(
            f"dump 1 all custom {self.parameters.sample_frequency} {traj_file} {' '.join(dump_items)}"
        )
        lines.append(
            'dump_modify 1 sort id format line "%d %d %20.15f %20.15f %20.15f %20.15f %20.15f %20.15f"'
        )

        # Energy/stress output for MLIP
        if self.parameters.dump_stress:
            lines.append("compute stress all stress/atom NULL")
            lines.append(
                "dump 2 all custom 100 stress.dump id type c_stress[1] c_stress[2] c_stress[3] c_stress[4] c_stress[5] c_stress[6]"
            )

        lines.append(f"run {self.parameters.production_steps}")
        lines.append("")

        return lines

    def generate(self) -> None:
        """Generate LAMMPS input files."""
        temperatures = (
            self.parameters.temperatures
            if isinstance(self.parameters.temperatures, list)
            else [self.parameters.temperatures]
        )

        for temp, output_file in zip(temperatures, self.output_files, strict=False):
            if self.logger:
                self.logger.info(f"Generating LAMMPS input for T={temp}K: {output_file}")

            lines = []
            lines.extend(self._generate_initialization_section())
            lines.extend(self._generate_system_setup_section())
            lines.extend(self._generate_force_field_section(temp))
            lines.extend(self._generate_settings_section())
            lines.extend(self._generate_groups_section())
            lines.extend(self._generate_computes_section())
            lines.extend(self._generate_equilibration_section(temp))
            lines.extend(self._generate_production_section(temp))

            # Write file
            with open(output_file, "w") as f:
                f.write("\n".join(lines))

            if self.logger:
                self.logger.info(f"Written LAMMPS input file: {output_file}")

    def run(self) -> None:
        """Main entry point for generation."""
        try:
            self.generate()
            if self.logger:
                self.logger.info("LAMMPS input generation completed successfully")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during generation: {e}")
            raise
