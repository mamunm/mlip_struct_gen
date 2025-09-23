"""LAMMPS input file generator for salt-water box simulations."""

from dataclasses import dataclass

from ..base import BaseLAMMPSGenerator
from .input_parameters import SaltWaterBoxLAMMPSParameters

# Water model parameters (same as water_box)
WATER_MODELS = {
    "SPC/E": {
        "O_mass": 15.9994,
        "H_mass": 1.008,
        "O_charge": -0.8476,
        "H_charge": 0.4238,
        "O_epsilon": 0.1553,  # kcal/mol
        "O_sigma": 3.166,  # Angstrom
        "H_epsilon": 0.0,
        "H_sigma": 0.0,
        "bond_k": 554.1349,  # kcal/mol/A^2
        "bond_r0": 1.0,  # Angstrom
        "angle_k": 45.7696,  # kcal/mol/rad^2
        "angle_theta0": 109.47,  # degrees
    },
    "SPCE": {  # Alias for SPC/E
        "O_mass": 15.9994,
        "H_mass": 1.008,
        "O_charge": -0.8476,
        "H_charge": 0.4238,
        "O_epsilon": 0.1553,
        "O_sigma": 3.166,
        "H_epsilon": 0.0,
        "H_sigma": 0.0,
        "bond_k": 554.1349,
        "bond_r0": 1.0,
        "angle_k": 45.7696,
        "angle_theta0": 109.47,
    },
    "TIP3P": {
        "O_mass": 15.9994,
        "H_mass": 1.008,
        "O_charge": -0.834,
        "H_charge": 0.417,
        "O_epsilon": 0.1521,  # kcal/mol
        "O_sigma": 3.1507,  # Angstrom
        "H_epsilon": 0.0,
        "H_sigma": 0.0,
        "bond_k": 554.1349,
        "bond_r0": 0.9572,  # Angstrom
        "angle_k": 55.0,  # kcal/mol/rad^2
        "angle_theta0": 104.52,  # degrees
    },
    "TIP4P": {
        "O_mass": 15.9994,
        "H_mass": 1.008,
        "M_mass": 0.0,  # Massless site
        "O_charge": 0.0,
        "H_charge": 0.52,
        "M_charge": -1.04,
        "O_epsilon": 0.1550,  # kcal/mol
        "O_sigma": 3.1536,  # Angstrom
        "H_epsilon": 0.0,
        "H_sigma": 0.0,
        "M_epsilon": 0.0,
        "M_sigma": 0.0,
        "bond_k": 554.1349,
        "bond_r0": 0.9572,  # Angstrom
        "angle_k": 55.0,  # kcal/mol/rad^2
        "angle_theta0": 104.52,  # degrees
    },
}

# Joung-Cheatham ion parameters for monovalent ions with SPC/E water
# J. Phys. Chem. B 2008, 112, 9020-9041
ION_PARAMS_SPCE = {
    "Li": {"mass": 6.941, "charge": 1.0, "epsilon": 0.00535531, "sigma": 2.02590},
    "Na": {"mass": 22.990, "charge": 1.0, "epsilon": 0.00277614, "sigma": 2.73959},
    "K": {"mass": 39.098, "charge": 1.0, "epsilon": 0.17721700, "sigma": 3.13650},
    "Rb": {"mass": 85.468, "charge": 1.0, "epsilon": 0.39731940, "sigma": 3.29970},
    "Cs": {"mass": 132.905, "charge": 1.0, "epsilon": 0.73542900, "sigma": 3.52320},
    "F": {"mass": 18.998, "charge": -1.0, "epsilon": 0.71090000, "sigma": 2.30300},
    "Cl": {"mass": 35.453, "charge": -1.0, "epsilon": 0.71090000, "sigma": 3.78520},
    "Br": {"mass": 79.904, "charge": -1.0, "epsilon": 0.94513200, "sigma": 3.97620},
    "I": {"mass": 126.904, "charge": -1.0, "epsilon": 1.39500000, "sigma": 4.29520},
}

# Ion parameters for TIP3P water
ION_PARAMS_TIP3P = {
    "Li": {"mass": 6.941, "charge": 1.0, "epsilon": 0.00274200, "sigma": 2.12645},
    "Na": {"mass": 22.990, "charge": 1.0, "epsilon": 0.00359700, "sigma": 2.87547},
    "K": {"mass": 39.098, "charge": 1.0, "epsilon": 0.19377800, "sigma": 3.19354},
    "Rb": {"mass": 85.468, "charge": 1.0, "epsilon": 0.41896000, "sigma": 3.34179},
    "Cs": {"mass": 132.905, "charge": 1.0, "epsilon": 0.76173100, "sigma": 3.56047},
    "F": {"mass": 18.998, "charge": -1.0, "epsilon": 0.71090000, "sigma": 2.30300},
    "Cl": {"mass": 35.453, "charge": -1.0, "epsilon": 0.71090000, "sigma": 3.78520},
    "Br": {"mass": 79.904, "charge": -1.0, "epsilon": 0.94513200, "sigma": 3.97620},
    "I": {"mass": 126.904, "charge": -1.0, "epsilon": 1.39500000, "sigma": 4.29520},
}


@dataclass
class SaltWaterBoxLAMMPSGenerator(BaseLAMMPSGenerator):
    """Generator for salt-water box LAMMPS input files."""

    parameters: SaltWaterBoxLAMMPSParameters

    def __post_init__(self) -> None:
        """Initialize the salt-water box generator."""
        super().__post_init__()

        # Get water model parameters
        model_name = self.parameters.water_model.upper()
        if model_name not in WATER_MODELS:
            raise ValueError(f"Unknown water model: {self.parameters.water_model}")
        self.water_params = WATER_MODELS[model_name]

        # Get ion parameters based on water model
        if model_name in ["SPC/E", "SPCE"]:
            self.ion_params = ION_PARAMS_SPCE
        elif model_name == "TIP3P":
            self.ion_params = ION_PARAMS_TIP3P
        else:  # TIP4P uses SPC/E parameters as default
            self.ion_params = ION_PARAMS_SPCE

        # Determine atom type mapping
        self._setup_atom_types()

    def _generate_initialization_section(self) -> list[str]:
        """Generate LAMMPS initialization section with salt-water-specific headers."""
        from datetime import datetime

        lines = []
        # Header comments
        lines.append(
            f"# LAMMPS input script for {self.parameters.water_model} salt-water MD simulation"
        )
        lines.append(
            f"# Generated by mlip-struct-gen on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"# Ensemble: {self.parameters.ensemble}")
        temp = (
            self.parameters.temperatures[0]
            if isinstance(self.parameters.temperatures, list)
            else self.parameters.temperatures
        )
        lines.append(f"# Temperature: {temp} K")
        if self.parameters.ensemble == "NPT":
            lines.append(f"# Pressure: {self.parameters.pressure} atm")
        lines.append(f"# Water model: {self.parameters.water_model}")
        lines.append("# Ion parameters: Joung-Cheatham")
        lines.append(f"# SHAKE constraints: {'Yes' if self.parameters.use_shake else 'No'}")
        lines.append("")

        # Initialization
        lines.append("# Initialization")
        lines.append("clear")
        lines.append("units real")
        lines.append("atom_style full")
        lines.append("boundary p p p")

        lines.append("")
        return lines

    def _setup_atom_types(self) -> None:
        """Setup atom type mapping based on masses for water and ions."""
        self.atom_type_map = {}
        self.ion_type_map = {}

        # Map atom types based on masses
        for atom_type, mass in self.system_info["masses"].items():
            # Check water atoms first
            if abs(mass - self.water_params["O_mass"]) < 0.1:
                self.atom_type_map[atom_type] = "O"
            elif abs(mass - self.water_params["H_mass"]) < 0.1:
                self.atom_type_map[atom_type] = "H"
            elif "M_mass" in self.water_params and abs(mass - self.water_params["M_mass"]) < 0.1:
                self.atom_type_map[atom_type] = "M"
            else:
                # Check if it's an ion
                for ion_name, ion_data in self.ion_params.items():
                    if abs(mass - ion_data["mass"]) < 0.1:
                        self.ion_type_map[atom_type] = ion_name
                        break

    def _generate_force_field_section(self, temperature: float) -> list[str]:
        """Generate salt-water force field section."""
        lines = []
        lines.append("# Force field parameters")

        # Water model comment
        if self.parameters.water_model.upper() in ["SPC/E", "SPCE"]:
            lines.append("# Simple Point Charge/Extended model with Joung-Cheatham ions")
        elif self.parameters.water_model.upper() == "TIP3P":
            lines.append("# TIP3P water model with Joung-Cheatham ions")
        elif self.parameters.water_model.upper() == "TIP4P":
            lines.append("# TIP4P water model with ions")
        lines.append("")

        # Pair style and parameters
        lines.append("# Pair style and parameters")
        lines.append("pair_style lj/cut/coul/long 6.0")  # Reduced cutoff for efficiency
        lines.append("pair_modify mix arithmetic")

        # Water pair coefficients
        o_type = (
            [t for t, n in self.atom_type_map.items() if n == "O"][0] if self.atom_type_map else 1
        )
        h_type = (
            [t for t, n in self.atom_type_map.items() if n == "H"][0] if self.atom_type_map else 2
        )

        lines.append("")
        lines.append("# Water parameters")
        lines.append(
            f"pair_coeff {o_type} {o_type} {self.water_params['O_epsilon']:.4f} {self.water_params['O_sigma']:.4f}  # O-O"
        )
        lines.append(
            f"pair_coeff {h_type} {h_type} {self.water_params['H_epsilon']:.4f} {self.water_params['H_sigma']:.4f}  # H-H"
        )

        # Mixed O-H interaction
        o_h_epsilon = (self.water_params["O_epsilon"] * self.water_params["H_epsilon"]) ** 0.5
        o_h_sigma = (self.water_params["O_sigma"] + self.water_params["H_sigma"]) / 2.0
        if self.water_params["H_epsilon"] == 0:
            o_h_epsilon = self.water_params["O_epsilon"] / 2.0
        lines.append(f"pair_coeff {o_type} {h_type} {o_h_epsilon:.4f} {o_h_sigma:.4f}  # O-H")

        # Ion pair coefficients
        if self.ion_type_map:
            lines.append("")
            lines.append("# Ion parameters")
            for atom_type, ion_name in self.ion_type_map.items():
                ion_data = self.ion_params[ion_name]
                lines.append(
                    f"pair_coeff {atom_type} {atom_type} {ion_data['epsilon']:.6f} {ion_data['sigma']:.5f}  # {ion_name}"
                )

        lines.append("")

        # Bond parameters
        if self.system_info["n_bond_types"] > 0:
            lines.append("# Bond parameters")
            lines.append("bond_style harmonic")
            # Use stiff bond for SHAKE
            lines.append(f"bond_coeff 1 1000.0 {self.water_params['bond_r0']:.4f}  # O-H bond")
            lines.append("")

        # Angle parameters
        if self.system_info["n_angle_types"] > 0:
            lines.append("# Angle parameters")
            lines.append("angle_style harmonic")
            # Use stiff angle for SHAKE
            lines.append(
                f"angle_coeff 1 100.0 {self.water_params['angle_theta0']:.2f}  # H-O-H angle"
            )
            lines.append("")

        # Long-range electrostatics
        lines.append("# Long-range electrostatics")
        lines.append(f"kspace_style pppm {self.parameters.coulomb_accuracy}")
        lines.append("")

        # Neighbor list settings
        lines.append("# Neighbor list settings")
        lines.append("neighbor        2.0 bin")
        lines.append("neigh_modify    every 1 delay 10 check no")
        lines.append("")

        return lines

    def _generate_settings_section(self) -> list[str]:
        """Generate simulation settings for salt-water."""
        lines = []

        # SHAKE constraints
        if self.parameters.use_shake and self.system_info["n_bond_types"] > 0:
            lines.append("# SHAKE constraints")
            lines.append("fix shake_bonds all shake 1e-06 20 0 b 1 a 1")
            lines.append("")

        # Simulation setup
        lines.append("# Simulation setup")
        lines.append(f"timestep {self.parameters.timestep}")
        lines.append("")

        return lines

    def _generate_groups_section(self) -> list[str]:
        """Generate atom groups for salt-water."""
        lines = []
        lines.append("# Groups")

        # Water molecule groups
        o_types = [str(t) for t, n in self.atom_type_map.items() if n == "O"]
        h_types = [str(t) for t, n in self.atom_type_map.items() if n == "H"]

        if o_types:
            lines.append(f"group oxygen type {' '.join(o_types)}")
        if h_types:
            lines.append(f"group hydrogen type {' '.join(h_types)}")
        if o_types and h_types:
            lines.append("group water union oxygen hydrogen")

        # Ion groups
        cation_types = []
        anion_types = []
        for atom_type, ion_name in self.ion_type_map.items():
            if self.ion_params[ion_name]["charge"] > 0:
                cation_types.append(str(atom_type))
            else:
                anion_types.append(str(atom_type))

        if cation_types:
            lines.append(f"group cations type {' '.join(cation_types)}")
        if anion_types:
            lines.append(f"group anions type {' '.join(anion_types)}")
        if cation_types or anion_types:
            ion_types = cation_types + anion_types
            lines.append(f"group ions type {' '.join(ion_types)}")

        lines.append("")
        return lines

    def _generate_equilibration_section(self, temperature: float) -> list[str]:
        """Generate equilibration section for salt-water."""
        lines = []

        # Generate initial velocities
        lines.append("# Generate initial velocities")
        lines.append(f"velocity all create {temperature} {self.parameters.seed} dist uniform")
        lines.append("")

        # Output settings
        lines.append("# Output settings")
        lines.append("thermo 100")  # Every 100 steps during equilibration
        lines.append("thermo_style custom step temp press pe ke etotal vol density")
        lines.append("")

        # Calculate equilibration time in ps
        eq_time_ps = self.parameters.equilibration_steps * self.parameters.timestep / 1000.0

        lines.append("# Equilibration phase")
        lines.append(f"# {eq_time_ps:.1f} ps ({self.parameters.equilibration_steps} steps)")
        lines.append("")

        # Equilibration trajectory dump
        lines.append("# Equilibration trajectory output")

        # Element mapping for water and ions
        element_list = []
        for atom_type in sorted(self.system_info["masses"].keys()):
            if atom_type in self.atom_type_map:
                element_list.append(self.atom_type_map[atom_type])
            elif atom_type in self.ion_type_map:
                element_list.append(self.ion_type_map[atom_type])
            else:
                element_list.append("X")  # Unknown

        lines.append(
            f"dump eq_traj all custom {self.parameters.dump_frequency} eq_trajectory.lammpstrj id type element x y z"
        )
        lines.append(f"dump_modify eq_traj element {' '.join(element_list)}")
        lines.append("")

        # Equilibration ensemble
        if self.parameters.ensemble == "NPT":
            lines.append("# NPT ensemble with Nosé-Hoover thermostat and barostat")
            lines.append(
                f"fix npt_eq all npt temp {temperature} {temperature} {self.parameters.thermostat_damping} iso {self.parameters.pressure} {self.parameters.pressure} {self.parameters.barostat_damping}"
            )
        elif self.parameters.ensemble == "NVT":
            lines.append("# NVT ensemble with Nosé-Hoover thermostat")
            lines.append(
                f"fix nvt_eq all nvt temp {temperature} {temperature} {self.parameters.thermostat_damping}"
            )
        else:  # NVE
            lines.append("# NVE ensemble")
            lines.append("fix nve_eq all nve")

        lines.append("")
        lines.append("# Run equilibration")
        lines.append(f"run {self.parameters.equilibration_steps}")
        lines.append("")

        # Undump equilibration trajectory
        lines.append("undump eq_traj")
        lines.append("")

        # Unfix equilibration integrator
        lines.append("# Unfix equilibration integrator")
        if self.parameters.ensemble == "NPT":
            lines.append("unfix npt_eq")
        elif self.parameters.ensemble == "NVT":
            lines.append("unfix nvt_eq")
        else:
            lines.append("unfix nve_eq")

        lines.append("")
        return lines

    def _generate_production_section(self, temperature: float) -> list[str]:
        """Generate production section for salt-water with MLIP training output."""
        lines = []

        # Calculate production time in ps
        prod_time_ps = self.parameters.production_steps * self.parameters.timestep / 1000.0

        lines.append("# Production phase")
        lines.append(f"# {prod_time_ps:.1f} ps ({self.parameters.production_steps} steps)")
        lines.append("")

        lines.append("reset_timestep 0")
        lines.append("")

        # Production run output settings
        lines.append("# Production run output settings")
        lines.append(f"variable        dump_freq equal {self.parameters.dump_steps}")
        lines.append("thermo_style    custom step etotal pe ke ecoul evdwl elong etail density")
        lines.append("thermo          ${dump_freq}")
        lines.append("")

        # Trajectory output
        lines.append("# Trajectory output")

        # File naming based on temperature
        temps = (
            self.parameters.temperatures
            if isinstance(self.parameters.temperatures, list)
            else [self.parameters.temperatures]
        )
        if len(temps) > 1:
            traj_file = f"trajectory_T{temperature:.0f}.lammpstrj"
        else:
            traj_file = "trajectory.lammpstrj"

        # Main trajectory with charges for visualization
        lines.append(
            f"dump            1 all custom ${{dump_freq}} {traj_file} id mol type element q x y z"
        )

        # Element mapping for water and common ions
        element_list = []
        for atom_type in sorted(self.system_info["masses"].keys()):
            if atom_type in self.atom_type_map:
                element_list.append(self.atom_type_map[atom_type])
            elif atom_type in self.ion_type_map:
                # Use element symbol for ions
                ion_name = self.ion_type_map[atom_type]
                element_list.append(ion_name)
            else:
                element_list.append("X")  # Unknown

        lines.append(
            'dump_modify     1 sort id format line "%d %d %d %s %8.6f %20.15f %20.15f %20.15f"'
        )
        lines.append(f"dump_modify     1 element {' '.join(element_list)}")

        # Position dump for visualization
        pos_file = f"pos_T{temperature:.0f}.lammpstrj" if len(temps) > 1 else "pos.lammpstrj"
        lines.append(f"dump            movie all atom ${{dump_freq}} {pos_file}")
        lines.append("")

        # Production ensemble
        if self.parameters.ensemble == "NPT":
            lines.append("# NPT ensemble with Nosé-Hoover thermostat and barostat")
            lines.append(
                f"fix npt_prod all npt temp {temperature} {temperature} {self.parameters.thermostat_damping} iso {self.parameters.pressure} {self.parameters.pressure} {self.parameters.barostat_damping}"
            )
        elif self.parameters.ensemble == "NVT":
            lines.append("# NVT ensemble with Nosé-Hoover thermostat")
            lines.append(
                f"fix nvt_prod all nvt temp {temperature} {temperature} {self.parameters.thermostat_damping}"
            )
        else:  # NVE
            lines.append("# NVE ensemble")
            lines.append("fix nve_prod all nve")

        lines.append("")
        lines.append("# Run production")
        lines.append(f"run {self.parameters.production_steps}")
        lines.append("")

        # Final output
        lines.append("# Final output")
        temps = (
            self.parameters.temperatures
            if isinstance(self.parameters.temperatures, list)
            else [self.parameters.temperatures]
        )
        final_file = (
            f"final_structure_T{temperature:.0f}.data" if len(temps) > 1 else "final_structure.data"
        )
        lines.append(f"write_data {final_file}")
        lines.append('print "Simulation completed successfully"')
        lines.append("")

        return lines
