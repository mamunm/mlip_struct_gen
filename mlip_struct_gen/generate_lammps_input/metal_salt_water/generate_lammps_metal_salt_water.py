"""LAMMPS input file generator for metal-salt-water interface simulations."""

import json
from dataclasses import dataclass

from ...potentials import LJ_PARAMS_FILE
from ..base import BaseLAMMPSGenerator
from .input_parameters import MetalSaltWaterLAMMPSParameters

# Water model parameters for metal units (eV, Angstrom, ps)
WATER_MODELS_METAL = {
    "SPC/E": {
        "O_mass": 15.9994,
        "H_mass": 1.008,
        "O_charge": -0.8476,
        "H_charge": 0.4238,
        "O_epsilon": 0.006808,  # 0.1553 kcal/mol = 0.006808 eV
        "O_sigma": 3.16556,  # Angstrom
        "H_epsilon": 0.0,
        "H_sigma": 0.0,
        "bond_k": 95.83,  # eV/Ang^2 (converted from SPC/E)
        "bond_r0": 1.0,  # Angstrom
        "angle_k": 10.42,  # eV/rad^2 (converted from SPC/E)
        "angle_theta0": 109.47,  # degrees
    },
    "TIP3P": {
        "O_mass": 15.9994,
        "H_mass": 1.008,
        "O_charge": -0.834,
        "H_charge": 0.417,
        "O_epsilon": 0.006661,  # 0.1521 kcal/mol = 0.006661 eV
        "O_sigma": 3.1507,  # Angstrom
        "H_epsilon": 0.0,
        "H_sigma": 0.0,
        "bond_k": 95.83,  # eV/Ang^2
        "bond_r0": 0.9572,  # Angstrom
        "angle_k": 12.0,  # eV/rad^2
        "angle_theta0": 104.52,  # degrees
    },
}

# Ion parameters for metal units (Joung-Cheatham for SPC/E water)
ION_PARAMS_METAL = {
    "Na": {
        "mass": 22.990,
        "charge": 1.0,
        "epsilon": 0.0001215,  # 0.00277614 kcal/mol = 0.0001215 eV
        "sigma": 2.73959,  # Angstrom
    },
    "K": {
        "mass": 39.098,
        "charge": 1.0,
        "epsilon": 0.0001526,  # 0.00348672 kcal/mol = 0.0001526 eV
        "sigma": 3.56359,  # Angstrom
    },
    "Li": {
        "mass": 6.941,
        "charge": 1.0,
        "epsilon": 0.0001332,  # 0.00304305 kcal/mol = 0.0001332 eV
        "sigma": 2.25923,  # Angstrom
    },
    "Cl": {
        "mass": 35.453,
        "charge": -1.0,
        "epsilon": 0.0311090,  # 0.71090000 kcal/mol = 0.0311090 eV
        "sigma": 3.78520,  # Angstrom
    },
    "F": {
        "mass": 18.998,
        "charge": -1.0,
        "epsilon": 0.0023320,  # 0.05329000 kcal/mol = 0.0023320 eV
        "sigma": 3.11814,  # Angstrom
    },
    "Br": {
        "mass": 79.904,
        "charge": -1.0,
        "epsilon": 0.0386160,  # 0.88210000 kcal/mol = 0.0386160 eV
        "sigma": 4.16524,  # Angstrom
    },
}


@dataclass
class MetalSaltWaterLAMMPSGenerator(BaseLAMMPSGenerator):
    """Generate LAMMPS input files for metal-salt-water interface simulations."""

    parameters: MetalSaltWaterLAMMPSParameters

    def __post_init__(self) -> None:
        """Initialize the metal-salt-water generator."""
        # Load LJ parameters for metals
        if LJ_PARAMS_FILE.exists():
            with open(LJ_PARAMS_FILE) as f:
                lj_data = json.load(f)
                self.metal_lj_params = lj_data.get("metals", {})
        else:
            self.metal_lj_params = {}

        # Get water model parameters for metal units
        self.water_params = WATER_MODELS_METAL.get(
            self.parameters.water_model, WATER_MODELS_METAL["SPC/E"]
        )

        # Parse salt type to get cation and anion
        self._parse_salt_type()

        # Get atomic masses
        self.metal_masses = {
            "Cu": 63.546,
            "Ag": 107.8682,
            "Au": 196.9666,
            "Ni": 58.6934,
            "Pd": 106.42,
            "Pt": 195.078,
            "Al": 26.9815,
        }

        # Initialize parent class
        super().__post_init__()

    def _parse_salt_type(self) -> None:
        """Parse salt type to identify cation and anion."""
        salt_compositions = {
            "NaCl": ("Na", "Cl"),
            "KCl": ("K", "Cl"),
            "LiCl": ("Li", "Cl"),
            "NaF": ("Na", "F"),
            "KF": ("K", "F"),
            "LiF": ("Li", "F"),
            "NaBr": ("Na", "Br"),
            "KBr": ("K", "Br"),
            "LiBr": ("Li", "Br"),
        }

        if self.parameters.salt_type in salt_compositions:
            self.cation, self.anion = salt_compositions[self.parameters.salt_type]
            self.cation_params = ION_PARAMS_METAL[self.cation]
            self.anion_params = ION_PARAMS_METAL[self.anion]
        else:
            # Default to NaCl if unknown
            self.cation, self.anion = "Na", "Cl"
            self.cation_params = ION_PARAMS_METAL["Na"]
            self.anion_params = ION_PARAMS_METAL["Cl"]

    def _generate_initialization_section(self) -> list[str]:
        """Generate LAMMPS initialization section for metal-salt-water interface."""
        from datetime import datetime

        lines = []
        # Header comments
        lines.append(
            f"# LAMMPS Input File: {self.parameters.metal_type}-{self.parameters.salt_type}-Water {self.parameters.ensemble} at {self.parameters.temperatures[0] if isinstance(self.parameters.temperatures, list) else self.parameters.temperatures}K"
        )
        lines.append(
            f"# Generated by mlip-struct-gen on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(
            f"# Atom types: 1={self.parameters.metal_type}, 2=O, 3=H, 4={self.cation}, 5={self.anion}"
        )
        lines.append(f"# Water model: {self.parameters.water_model}")
        lines.append(f"# Salt: {self.parameters.salt_type}")
        if self.parameters.metal_type in self.metal_lj_params:
            lj = self.metal_lj_params[self.parameters.metal_type]
            lines.append(
                f"# LJ potential: sigma={lj['sigma']:.3f} A, epsilon={lj['epsilon']:.6f} eV"
            )
        lines.append("")

        # Initialization
        lines.append("# Initialization")
        lines.append("units metal")  # metal units (eV, Angstrom, ps)
        lines.append("dimension 3")
        lines.append("boundary p p p")  # Periodic boundaries
        lines.append("atom_style full")  # Full style for molecular systems
        lines.append("")

        return lines

    def _generate_force_field_section(self, temperature: float) -> list[str]:
        """Generate metal-salt-water force field section."""
        lines = []

        # Set masses
        lines.append("# Set masses (required for metal units)")
        metal_mass = self.metal_masses.get(self.parameters.metal_type, 100.0)
        lines.append(f"mass 1 {metal_mass:.3f}  # {self.parameters.metal_type} atomic mass")
        lines.append(f"mass 2 {self.water_params['O_mass']:.4f}  # O atomic mass")
        lines.append(f"mass 3 {self.water_params['H_mass']:.3f}  # H atomic mass")
        lines.append(f"mass 4 {self.cation_params['mass']:.3f}  # {self.cation} atomic mass")
        lines.append(f"mass 5 {self.anion_params['mass']:.3f}  # {self.anion} atomic mass")
        lines.append("")

        # Define potentials
        lines.append(
            f"# Define potentials for {self.parameters.metal_type}-{self.parameters.salt_type}-water system"
        )
        lines.append(
            f"# {self.parameters.water_model} water parameters for metal units (eV, Angstrom)"
        )
        lines.append("pair_style lj/cut/coul/long 10.0 10.0")
        lines.append("")

        # Metal-metal interactions
        if self.parameters.metal_type in self.metal_lj_params:
            metal_lj = self.metal_lj_params[self.parameters.metal_type]
            lines.append(
                f"# {self.parameters.metal_type}-{self.parameters.metal_type} interactions (LJ potential)"
            )
            lines.append(
                f"pair_coeff 1 1 {metal_lj['epsilon']:.6f} {metal_lj['sigma']:.3f}   "
                f"# epsilon={metal_lj['epsilon']:.6f} eV, sigma={metal_lj['sigma']:.3f} A"
            )
        else:
            lines.append(f"# WARNING: No LJ parameters found for {self.parameters.metal_type}")
            lines.append("# Using default metal LJ parameters")
            lines.append("pair_coeff 1 1 0.5 2.5  # Default values")
        lines.append("")

        # Water-water interactions
        lines.append(
            f"# {self.parameters.water_model} water interactions (converted to metal units)"
        )
        lines.append("# O-O interactions")
        lines.append(
            f"pair_coeff 2 2 {self.water_params['O_epsilon']:.6f} {self.water_params['O_sigma']:.5f}   "
            f"# epsilon={self.water_params['O_epsilon']:.6f} eV"
        )
        lines.append("")

        lines.append("# H-H interactions (typically zero)")
        lines.append("pair_coeff 3 3 0.0000 0.0000")
        lines.append("")

        lines.append("# O-H interactions (zero LJ, only Coulomb)")
        lines.append("pair_coeff 2 3 0.0000 0.0000")
        lines.append("")

        # Ion-ion interactions
        lines.append("# Ion-ion interactions (Joung-Cheatham parameters)")
        lines.append(f"# {self.cation}-{self.cation} interactions")
        lines.append(
            f"pair_coeff 4 4 {self.cation_params['epsilon']:.6f} {self.cation_params['sigma']:.5f}   "
            f"# epsilon={self.cation_params['epsilon']:.6f} eV"
        )
        lines.append("")

        lines.append(f"# {self.anion}-{self.anion} interactions")
        lines.append(
            f"pair_coeff 5 5 {self.anion_params['epsilon']:.6f} {self.anion_params['sigma']:.5f}   "
            f"# epsilon={self.anion_params['epsilon']:.6f} eV"
        )
        lines.append("")

        # Ion-water interactions (Lorentz-Berthelot mixing)
        lines.append("# Ion-water interactions (Lorentz-Berthelot mixing)")

        # Cation-O
        co_epsilon = (self.cation_params["epsilon"] * self.water_params["O_epsilon"]) ** 0.5
        co_sigma = (self.cation_params["sigma"] + self.water_params["O_sigma"]) / 2
        lines.append(f"# {self.cation}-O interaction")
        lines.append(f"pair_coeff 2 4 {co_epsilon:.6f} {co_sigma:.5f}")

        # Cation-H (very weak, often set to zero)
        lines.append(f"# {self.cation}-H interaction (very weak)")
        lines.append("pair_coeff 3 4 0.0000 0.0000")

        # Anion-O
        ao_epsilon = (self.anion_params["epsilon"] * self.water_params["O_epsilon"]) ** 0.5
        ao_sigma = (self.anion_params["sigma"] + self.water_params["O_sigma"]) / 2
        lines.append(f"# {self.anion}-O interaction")
        lines.append(f"pair_coeff 2 5 {ao_epsilon:.6f} {ao_sigma:.5f}")

        # Anion-H (very weak)
        lines.append(f"# {self.anion}-H interaction (very weak)")
        lines.append("pair_coeff 3 5 0.0000 0.0000")
        lines.append("")

        # Ion-ion cross interaction
        ion_epsilon = (self.cation_params["epsilon"] * self.anion_params["epsilon"]) ** 0.5
        ion_sigma = (self.cation_params["sigma"] + self.anion_params["sigma"]) / 2
        lines.append(f"# {self.cation}-{self.anion} interaction")
        lines.append(f"pair_coeff 4 5 {ion_epsilon:.6f} {ion_sigma:.5f}")
        lines.append("")

        # Metal-water interactions (geometric mixing rule)
        lines.append("# Metal-water interactions (geometric mixing rule)")
        if self.parameters.metal_type in self.metal_lj_params:
            metal_lj = self.metal_lj_params[self.parameters.metal_type]

            # Metal-O interaction
            mo_epsilon = (metal_lj["epsilon"] * self.water_params["O_epsilon"]) ** 0.5
            mo_sigma = (metal_lj["sigma"] + self.water_params["O_sigma"]) / 2
            lines.append(f"# {self.parameters.metal_type}-O interaction")
            lines.append(f"pair_coeff 1 2 {mo_epsilon:.4f} {mo_sigma:.3f}")

            # Metal-H interaction (weak)
            mh_sigma = metal_lj["sigma"] / 2  # Much smaller for H
            lines.append(f"# {self.parameters.metal_type}-H interaction (weak)")
            lines.append(f"pair_coeff 1 3 0.0100 {mh_sigma:.3f}")

            # Metal-ion interactions
            # Metal-cation
            mc_epsilon = (metal_lj["epsilon"] * self.cation_params["epsilon"]) ** 0.5
            mc_sigma = (metal_lj["sigma"] + self.cation_params["sigma"]) / 2
            lines.append(f"# {self.parameters.metal_type}-{self.cation} interaction")
            lines.append(f"pair_coeff 1 4 {mc_epsilon:.6f} {mc_sigma:.3f}")

            # Metal-anion
            ma_epsilon = (metal_lj["epsilon"] * self.anion_params["epsilon"]) ** 0.5
            ma_sigma = (metal_lj["sigma"] + self.anion_params["sigma"]) / 2
            lines.append(f"# {self.parameters.metal_type}-{self.anion} interaction")
            lines.append(f"pair_coeff 1 5 {ma_epsilon:.6f} {ma_sigma:.3f}")
        else:
            lines.append("# Using default metal interactions")
            lines.append("pair_coeff 1 2 0.05 3.0  # Metal-O")
            lines.append("pair_coeff 1 3 0.01 1.5  # Metal-H")
            lines.append("pair_coeff 1 4 0.02 2.5  # Metal-cation")
            lines.append("pair_coeff 1 5 0.08 3.0  # Metal-anion")
        lines.append("")

        # Bond potential for water
        lines.append(f"# Bond potential for {self.parameters.water_model} water (O-H bonds)")
        lines.append("bond_style harmonic")
        lines.append(
            f"bond_coeff 1 {self.water_params['bond_k']:.2f} {self.water_params['bond_r0']:.1f}           "
            f"# k={self.water_params['bond_k']:.2f} eV/Ang^2, r0={self.water_params['bond_r0']:.1f} Ang"
        )
        lines.append("")

        # Angle potential for water
        lines.append(f"# Angle potential for {self.parameters.water_model} water (H-O-H angle)")
        lines.append("angle_style harmonic")
        lines.append(
            f"angle_coeff 1 {self.water_params['angle_k']:.2f} {self.water_params['angle_theta0']:.2f}       "
            f"# k={self.water_params['angle_k']:.2f} eV/rad^2, theta0={self.water_params['angle_theta0']:.2f} deg"
        )
        lines.append("")

        # Long-range Coulombics
        lines.append("# Long-range Coulombics")
        lines.append(f"kspace_style pppm {self.parameters.coulomb_accuracy}")
        lines.append("")

        lines.append("# Charges are already defined in data file")
        lines.append(
            f"# Water {self.parameters.water_model}: O: {self.water_params['O_charge']:.4f} e, H: {self.water_params['H_charge']:.4f} e"
        )
        lines.append(
            f"# Ions: {self.cation}: {self.cation_params['charge']:.1f} e, {self.anion}: {self.anion_params['charge']:.1f} e"
        )
        lines.append("")

        # Neighbor settings
        lines.append("# Neighbor settings")
        lines.append("neighbor 2.0 bin")
        lines.append("neigh_modify every 1 delay 0 check yes")
        lines.append("")

        return lines

    def _generate_settings_section(self) -> list[str]:
        """Generate simulation settings for metal-salt-water interface."""
        lines = []

        # Define groups for temperature computation
        lines.append("# Define groups for temperature computation")
        lines.append("group water type 2 3  # O and H atoms")
        lines.append("group metal type 1")
        lines.append("group ions type 4 5  # Cation and anion")
        lines.append("group solution type 2 3 4 5  # Water + ions")
        lines.append("")

        # Compute water temperature
        lines.append("# Compute temperatures")
        lines.append("compute temp_water water temp")
        lines.append("compute temp_solution solution temp")
        lines.append("")

        # Time integration and thermostat settings will be in equilibration/production
        # Timestep (convert fs to ps for metal units)
        self.parameters.timestep * 0.001  # Convert fs to ps

        # SHAKE constraints for rigid water molecules
        lines.append("# SHAKE constraints for rigid water molecules")
        lines.append(
            "fix shake all shake 0.0001 20 0 b 1 a 1          # constrain O-H bonds and H-O-H angles"
        )
        lines.append("")

        # Compute properties for MLIP training if needed
        if self.parameters.compute_stress:
            lines.append("# Compute per-atom stress for MLIP training")
            lines.append("compute stress_atom all stress/atom NULL")
            lines.append("")

        return lines

    def _generate_equilibration_section(self, temperature: float) -> list[str]:
        """Generate equilibration section for metal-salt-water interface."""
        lines = []

        # Timestep
        timestep_ps = self.parameters.timestep * 0.001  # Convert fs to ps
        lines.append("# Time integration setup")
        lines.append(
            f"timestep {timestep_ps:.3f}                                     # {self.parameters.timestep:.1f} fs timestep"
        )
        lines.append("")

        # Ensemble setup with fix bottom layers if needed
        if self.parameters.fix_bottom_layers > 0:
            lines.append("# Define groups for fixed layers")
            lines.append("group metal type 1")
            lines.append("variable zmin equal bound(metal,zmin)")
            lines.append(f"variable zfix equal ${{zmin}}+{self.parameters.fix_bottom_layers*3.0}")
            lines.append("region bottom_region block EDGE EDGE EDGE EDGE EDGE ${zfix}")
            lines.append("group bottom_metal region bottom_region")
            lines.append("group mobile_atoms subtract all bottom_metal")
            lines.append("fix freeze bottom_metal setforce 0.0 0.0 0.0")
            lines.append("")
            fix_group = "mobile_atoms"
        else:
            fix_group = "all"

        # Thermostat and barostat
        damping_ps = self.parameters.thermostat_damping * 0.001  # Convert fs to ps

        if self.parameters.ensemble == "NVT":
            lines.append("# Time integration and thermostat")
            lines.append(
                f"fix nvt {fix_group} nvt temp {temperature:.1f} {temperature:.1f} {damping_ps:.1f}              "
                f"# {damping_ps:.1f} ps damping time"
            )
        elif self.parameters.ensemble == "NPT":
            barostat_damping_ps = self.parameters.barostat_damping * 0.001
            lines.append("# Time integration with NPT ensemble")
            lines.append(
                f"fix npt {fix_group} npt temp {temperature:.1f} {temperature:.1f} {damping_ps:.1f} "
                f"iso {self.parameters.pressure:.1f} {self.parameters.pressure:.1f} {barostat_damping_ps:.1f}"
            )
        else:  # NVE
            lines.append("# Time integration with NVE ensemble")
            lines.append(f"fix nve {fix_group} nve")
        lines.append("")

        # Output settings
        lines.append("# Output settings")
        lines.append(
            "thermo_style custom step time temp c_temp_water c_temp_solution pe ke etotal press vol density"
        )
        lines.append(
            'thermo_modify colname c_temp_water "T_water" colname c_temp_solution "T_soln"'
        )
        eq_steps = int(self.parameters.equilibration_time / timestep_ps)
        thermo_out = min(1000, eq_steps // 100)  # Output ~100 times during equilibration
        lines.append(
            f"thermo {thermo_out}                                        # output every {thermo_out} steps"
        )
        lines.append("")

        # Run equilibration
        lines.append("# Equilibration phase")
        lines.append(f"# {self.parameters.equilibration_time:.1f} ps = {eq_steps} steps")
        lines.append(f"run {eq_steps}")
        lines.append("")

        return lines

    def _generate_production_section(self, temperature: float) -> list[str]:
        """Generate production section for metal-salt-water interface with MLIP training output."""
        lines = []

        # Reset timestep for production
        lines.append("# Reset for production run")
        lines.append("reset_timestep 0")
        lines.append("")

        # Calculate production steps
        timestep_ps = self.parameters.timestep * 0.001  # Convert fs to ps
        prod_steps = int(self.parameters.production_time / timestep_ps)
        dump_steps = int(self.parameters.dump_frequency / timestep_ps)
        restart_steps = min(50000, prod_steps // 10)  # Restart files every 10% of run
        thermo_steps = int(self.parameters.dump_frequency / timestep_ps)  # Same as dump frequency

        # Production thermo output settings
        lines.append("# Production thermo output")
        lines.append(f"thermo {thermo_steps}")
        lines.append(
            "thermo_style custom step time temp c_temp_water c_temp_solution pe ke etotal press vol density"
        )
        lines.append(
            'thermo_modify colname c_temp_water "T_water" colname c_temp_solution "T_soln"'
        )
        lines.append("")

        # Trajectory output
        lines.append("# Trajectory output with custom format")
        traj_file = "trajectory.lammpstrj"

        # Basic dump with positions
        lines.append(f"dump traj all custom {dump_steps} {traj_file} id type element x y z")
        # Add element mapping so trajectory file has proper element information
        # Type 1=metal, 2=O, 3=H, 4=cation, 5=anion
        lines.append(
            f"dump_modify traj element {self.parameters.metal_type} O H {self.cation} {self.anion}"
        )
        lines.append("")

        # Force dump if needed for MLIP
        if self.parameters.compute_stress:
            lines.append("# Force and stress output for MLIP training")
            lines.append(
                f"dump forces all custom {dump_steps} forces.dump id type x y z fx fy fz c_stress_atom[*]"
            )
            lines.append("")

        # Restart files
        lines.append("# Restart files")
        lines.append(
            f"restart {restart_steps} restart1.lmp restart2.lmp           # restart files every {restart_steps} steps"
        )
        lines.append("")

        # Run production
        lines.append("# Run simulation")
        lines.append(f"# {self.parameters.production_time:.1f} ps = {prod_steps} steps")
        lines.append(f"run {prod_steps}")
        lines.append("")

        # Write final configuration
        lines.append("# Write final configuration")
        lines.append("write_data final_configuration.data")
        lines.append("")

        lines.append(
            f'print "Simulation completed: {self.parameters.production_time:.1f} ps {self.parameters.ensemble} at {temperature:.1f}K"'
        )
        lines.append("")

        return lines
