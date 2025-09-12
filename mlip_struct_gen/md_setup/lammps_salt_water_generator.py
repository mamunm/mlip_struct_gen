"""LAMMPS input file generator for salt water MD simulations."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import datetime

from .input_parameters import LAMMPSInputParameters
from .validation import validate_parameters
from ..generate_structure.templates.water_models import get_water_model
from ..generate_structure.templates.salt_models import get_salt_model, get_ion_lj_params


class LAMMPSSaltWaterGenerator:
    """Generate LAMMPS input files for salt water MD simulations with LJ parameters for ions."""
    
    # Force field parameters for different water models
    WATER_MODEL_PARAMETERS = {
        "SPC/E": {
            "description": "Simple Point Charge/Extended model",
            "lj_params": {
                "O": {"epsilon": 0.1553, "sigma": 3.166},  # kcal/mol, Angstrom
                "H": {"epsilon": 0.0, "sigma": 0.0}        # No LJ on hydrogens
            },
            "bond_params": {
                "OH": {"k": 1000.0, "r0": 1.0}  # Very stiff for SHAKE
            },
            "angle_params": {
                "HOH": {"k": 100.0, "theta0": 109.47}  # Degrees
            },
            "use_shake": True,
            "shake_bonds": "1",  # Bond type 1 (O-H)
            "shake_angles": "1"  # Angle type 1 (H-O-H)
        },
        "TIP3P": {
            "description": "Transferable Intermolecular Potential 3-Point model",
            "lj_params": {
                "O": {"epsilon": 0.1521, "sigma": 3.1507},
                "H": {"epsilon": 0.0, "sigma": 0.0}
            },
            "bond_params": {
                "OH": {"k": 450.0, "r0": 0.9572}
            },
            "angle_params": {
                "HOH": {"k": 55.0, "theta0": 104.52}
            },
            "use_shake": True,
            "shake_bonds": "1",
            "shake_angles": "1"
        },
        "TIP4P": {
            "description": "Transferable Intermolecular Potential 4-Point model",
            "lj_params": {
                "O": {"epsilon": 0.1550, "sigma": 3.1536},
                "H": {"epsilon": 0.0, "sigma": 0.0}
            },
            "bond_params": {
                "OH": {"k": 450.0, "r0": 0.9572}
            },
            "angle_params": {
                "HOH": {"k": 55.0, "theta0": 104.52}
            },
            "use_shake": True,
            "shake_bonds": "1", 
            "shake_angles": "1"
        }
    }
    
    def __init__(self, parameters: LAMMPSInputParameters, salt_type: str = "NaCl", lj_type: str = "charmm"):
        """
        Initialize LAMMPS salt water input generator.
        
        Args:
            parameters: LAMMPS simulation parameters
            salt_type: Type of salt (e.g., "NaCl", "KCl", "CaCl2")
            lj_type: Force field type for LJ parameters ("charmm" or "amber")
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If required files don't exist
        """
        self.parameters = parameters
        self.salt_type = salt_type
        self.lj_type = lj_type
        validate_parameters(self.parameters)
        
        # Get salt model parameters
        self.salt_model = get_salt_model(self.salt_type)
        
        # Get LJ parameters for the specified force field type
        self.ion_lj_params = get_ion_lj_params(self.salt_type, self.lj_type)
        
        # Setup logger
        self.logger: Optional["MLIPLogger"] = None
        if self.parameters.log:
            if self.parameters.logger is not None:
                self.logger = self.parameters.logger
            else:
                # Import here to avoid circular imports
                try:
                    from ..utils.logger import MLIPLogger
                    self.logger = MLIPLogger()
                except ImportError:
                    # Gracefully handle missing logger
                    self.logger = None
        
        if self.logger:
            self.logger.info("Initializing LAMMPSSaltWaterGenerator")
            self.logger.info(f"Salt type: {self.salt_type}")
            self.logger.info(f"LJ force field: {self.lj_type.upper()}")
            self.logger.info(f"Water model: {self.parameters.water_model}")
            self.logger.info(f"Ensemble: {self.parameters.ensemble}")
            self.logger.info(f"Temperature: {self.parameters.temperature} K")
    
    def generate(self) -> str:
        """
        Generate LAMMPS input file for salt water system.
        
        Returns:
            Path to generated LAMMPS input file
        """
        if self.logger:
            self.logger.info("Generating LAMMPS salt water input file")
        
        output_path = Path(self.parameters.output_file)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.in')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get water model parameters
        model_params = self.WATER_MODEL_PARAMETERS[self.parameters.water_model]
        
        with open(output_path, 'w') as f:
            self._write_header(f)
            self._write_initialization(f)
            self._write_force_field(f, model_params)
            self._write_setup(f)
            self._write_equilibration(f, model_params)
            self._write_production(f, model_params)
        
        if self.logger:
            self.logger.success(f"LAMMPS salt water input file generated: {output_path}")
        
        return str(output_path)
    
    def _get_atom_types_from_data(self) -> Dict[str, int]:
        """
        Parse LAMMPS data file to determine atom type mapping.
        
        Returns:
            Dictionary mapping element names to atom type IDs
        """
        data_path = Path(self.parameters.lammps_data_file)
        atom_types = {}
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        # Find masses section
        masses_start = None
        for i, line in enumerate(lines):
            if line.strip() == "Masses":
                masses_start = i + 2  # Skip "Masses" and empty line
                break
        
        if masses_start is None:
            # Default mapping based on common convention
            atom_types = {"O": 1, "H": 2}
            type_id = 3
            for ion_type in ["cation", "anion"]:
                element = self.salt_model[ion_type]["element"]
                if element not in atom_types:
                    atom_types[element] = type_id
                    type_id += 1
        else:
            # Parse masses section
            for i in range(masses_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    continue
                if line and not any(char.isdigit() for char in line.split()[0]):
                    break  # End of masses section
                
                parts = line.split()
                if len(parts) >= 3:
                    type_id = int(parts[0])
                    # Extract element from comment
                    if '#' in line:
                        comment = line.split('#')[1].strip()
                        element = comment.split()[0]
                        atom_types[element] = type_id
        
        return atom_types
    
    def _write_header(self, f):
        """Write file header with metadata."""
        f.write(f"# LAMMPS input script for {self.salt_type} + {self.parameters.water_model} water MD simulation\n")
        f.write(f"# Generated by mlip-struct-gen on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Salt: {self.salt_model['name']}\n")
        f.write(f"# LJ force field: {self.lj_type.upper()}\n")
        f.write(f"# Ensemble: {self.parameters.ensemble}\n")
        f.write(f"# Temperature: {self.parameters.temperature} K\n")
        if self.parameters.ensemble == "NPT":
            f.write(f"# Pressure: {self.parameters.pressure} atm\n")
        f.write(f"# Water model: {self.parameters.water_model}\n")
        f.write(f"# SHAKE constraints: {'Yes' if self.parameters.use_shake else 'No'}\n")
        f.write(f"# Ion LJ parameters: Arithmetic mixing rule\n")
        f.write("\n")
    
    def _write_initialization(self, f):
        """Write initialization commands."""
        f.write("# Initialization\n")
        f.write("clear\n")
        f.write("units real\n")
        f.write("atom_style full\n")
        f.write("boundary p p p\n")
        f.write("\n")
        
        f.write("# Read data file\n")
        f.write(f"read_data {self.parameters.lammps_data_file}\n")
        f.write("\n")
    
    def _write_force_field(self, f, model_params: Dict[str, Any]):
        """Write force field parameters for salt water system."""
        f.write("# Force field parameters\n")
        f.write(f"# Water: {model_params['description']}\n")
        f.write(f"# Salt: {self.salt_model['name']}\n")
        f.write("\n")
        
        # Get atom type mapping
        atom_types = self._get_atom_types_from_data()
        
        # Pair coefficients (Lennard-Jones)
        f.write("# Pair style and parameters\n")
        f.write(f"pair_style {self.parameters.pair_style} {self.parameters.pair_style_cutoff:.1f}\n")
        f.write("pair_modify mix arithmetic\n")  # Use arithmetic mixing rule
        f.write("\n")
        
        # Write pair coefficients for each atom type
        f.write("# LJ parameters for all atom types\n")
        
        # Water atoms
        if "O" in atom_types:
            o_params = model_params["lj_params"]["O"]
            f.write(f"pair_coeff {atom_types['O']} {atom_types['O']} {o_params['epsilon']:.4f} {o_params['sigma']:.4f}  # O-O (water)\n")
        
        if "H" in atom_types:
            h_params = model_params["lj_params"]["H"]
            f.write(f"pair_coeff {atom_types['H']} {atom_types['H']} {h_params['epsilon']:.4f} {h_params['sigma']:.4f}  # H-H (water)\n")
        
        # Ion atoms (using selected force field parameters)
        for ion_type in ["cation", "anion"]:
            ion_data = self.salt_model[ion_type]
            element = ion_data["element"]
            
            if element in atom_types:
                # Get LJ parameters for the selected force field
                lj_params = self.ion_lj_params[ion_type]
                epsilon = lj_params["epsilon"]
                sigma = lj_params["sigma"]
                f.write(f"pair_coeff {atom_types[element]} {atom_types[element]} {epsilon:.4f} {sigma:.4f}  # {element}-{element} (ion, {self.lj_type.upper()})\n")
        
        f.write("\n")
        f.write("# Cross interactions will be determined by arithmetic mixing rule\n")
        f.write("# epsilon_ij = (epsilon_i + epsilon_j) / 2\n")
        f.write("# sigma_ij = (sigma_i + sigma_j) / 2\n")
        f.write("\n")
        
        # Bond coefficients (only for water)
        f.write("# Bond parameters (water only)\n")
        bond_params = model_params["bond_params"]["OH"]
        f.write(f"bond_style harmonic\n")
        f.write(f"bond_coeff 1 {bond_params['k']:.1f} {bond_params['r0']:.4f}  # O-H bond\n")
        f.write("\n")
        
        # Angle coefficients (only for water)
        f.write("# Angle parameters (water only)\n")
        angle_params = model_params["angle_params"]["HOH"]
        f.write(f"angle_style harmonic\n")
        f.write(f"angle_coeff 1 {angle_params['k']:.1f} {angle_params['theta0']:.2f}  # H-O-H angle\n")
        f.write("\n")
        
        # Long-range electrostatics (important for ions)
        if "coul/long" in self.parameters.pair_style:
            f.write("# Long-range electrostatics (important for ions)\n")
            f.write("kspace_style pppm 1e-4\n")
            f.write("\n")
        
        # Neighbor list settings
        f.write("# Neighbor list settings\n")
        f.write("neighbor        2.0 bin\n")
        f.write("neigh_modify    every 1 delay 10 check no\n")
        f.write("\n")
        
        # SHAKE constraints (only for water)
        if self.parameters.use_shake:
            f.write("# SHAKE constraints (water molecules only)\n")
            f.write(f"fix shake_bonds all shake {self.parameters.shake_tolerance} 20 0 ")
            f.write(f"b {model_params['shake_bonds']} a {model_params['shake_angles']}\n")
            f.write("\n")
    
    def _write_setup(self, f):
        """Write simulation setup."""
        f.write("# Simulation setup\n")
        f.write(f"timestep {self.parameters.timestep}\n")
        f.write("\n")
        
        f.write("# Generate initial velocities\n")
        f.write(f"velocity all create {self.parameters.temperature} {self.parameters.seed} dist uniform\n")
        f.write("\n")
        
        f.write("# Output settings\n")
        f.write(f"thermo {self.parameters.thermo_frequency}\n")
        f.write("thermo_style custom step temp press pe ke etotal vol density\n")
        f.write("\n")
        
        # Define groups for analysis
        f.write("# Define groups for analysis\n")
        atom_types = self._get_atom_types_from_data()
        
        if "O" in atom_types and "H" in atom_types:
            f.write(f"group water type {atom_types['O']} {atom_types['H']}\n")
        
        ion_types = []
        for ion_type in ["cation", "anion"]:
            element = self.salt_model[ion_type]["element"]
            if element in atom_types:
                f.write(f"group {element.lower()} type {atom_types[element]}\n")
                ion_types.append(atom_types[element])
        
        if ion_types:
            f.write(f"group ions type {' '.join(map(str, ion_types))}\n")
        
        f.write("\n")
    
    def _write_equilibration(self, f, model_params: Dict[str, Any]):
        """Write equilibration phase."""
        # Convert time to steps
        equilibration_steps = int(self.parameters.equilibration_time * 1000 / self.parameters.timestep)
        
        f.write("# Equilibration phase\n")
        f.write(f"# {self.parameters.equilibration_time} ps ({equilibration_steps} steps)\n")
        f.write("\n")
        
        # Choose integrator based on ensemble
        if self.parameters.ensemble == "NVE":
            f.write("# NVE ensemble\n")
            f.write("fix nve_eq all nve\n")
        elif self.parameters.ensemble == "NVT":
            f.write("# NVT ensemble with Nosé-Hoover thermostat\n")
            damping_steps = self.parameters.thermostat_damping * 1000 / self.parameters.timestep
            f.write(f"fix nvt_eq all nvt temp {self.parameters.temperature} {self.parameters.temperature} {damping_steps:.1f}\n")
        elif self.parameters.ensemble == "NPT":
            f.write("# NPT ensemble with Nosé-Hoover thermostat and barostat\n")
            t_damping = self.parameters.thermostat_damping * 1000 / self.parameters.timestep
            p_damping = self.parameters.barostat_damping * 1000 / self.parameters.timestep
            f.write(f"fix npt_eq all npt temp {self.parameters.temperature} {self.parameters.temperature} {t_damping:.1f} ")
            f.write(f"iso {self.parameters.pressure} {self.parameters.pressure} {p_damping:.1f}\n")
        
        f.write("\n")
        f.write(f"# Run equilibration\n")
        f.write(f"run {equilibration_steps}\n")
        f.write("\n")
        
        f.write("# Unfix equilibration integrator\n")
        if self.parameters.ensemble == "NVE":
            f.write("unfix nve_eq\n")
        elif self.parameters.ensemble == "NVT":
            f.write("unfix nvt_eq\n")
        elif self.parameters.ensemble == "NPT":
            f.write("unfix npt_eq\n")
        f.write("\n")
    
    def _write_production(self, f, model_params: Dict[str, Any]):
        """Write production phase."""
        # Convert time to steps
        production_steps = int(self.parameters.production_time * 1000 / self.parameters.timestep)
        
        f.write("# Production phase\n")
        f.write(f"# {self.parameters.production_time} ps ({production_steps} steps)\n")
        f.write("\n")
        
        # Reset timestep counter
        f.write("reset_timestep 0\n")
        f.write("\n")
        
        # Production run output settings
        f.write("# Production run output settings\n")
        # Convert dump_freq from ps to steps
        dump_freq_steps = int(self.parameters.dump_freq * 1000 / self.parameters.timestep)
        f.write(f"variable        dump_freq equal {dump_freq_steps}\n")
        f.write("thermo_style    custom step etotal pe ke ecoul evdwl elong etail density\n")
        f.write("thermo        ${dump_freq}\n")
        f.write("\n")
        
        # Get atom types for element mapping
        atom_types = self._get_atom_types_from_data()
        elements = []
        for i in sorted(atom_types.values()):
            for elem, type_id in atom_types.items():
                if type_id == i:
                    elements.append(elem)
                    break
        
        f.write("# Trajectory output\n")
        f.write("dump            1 all custom ${dump_freq} trajectory.lammpstrj id mol type element q x y z\n")
        f.write('dump_modify     1 sort id format line "%d %d %d %s %8.6f %20.15f %20.15f %20.15f"\n')
        f.write(f"dump_modify     1 element {' '.join(elements)}\n")
        f.write("dump            movie all atom ${dump_freq} pos.lammpstrj\n")
        f.write("\n")
        
        # Production integrator (same as equilibration)
        if self.parameters.ensemble == "NVE":
            f.write("# NVE ensemble\n")
            f.write("fix nve_prod all nve\n")
        elif self.parameters.ensemble == "NVT":
            f.write("# NVT ensemble with Nosé-Hoover thermostat\n")
            damping_steps = self.parameters.thermostat_damping * 1000 / self.parameters.timestep
            f.write(f"fix nvt_prod all nvt temp {self.parameters.temperature} {self.parameters.temperature} {damping_steps:.1f}\n")
        elif self.parameters.ensemble == "NPT":
            f.write("# NPT ensemble with Nosé-Hoover thermostat and barostat\n")
            t_damping = self.parameters.thermostat_damping * 1000 / self.parameters.timestep
            p_damping = self.parameters.barostat_damping * 1000 / self.parameters.timestep
            f.write(f"fix npt_prod all npt temp {self.parameters.temperature} {self.parameters.temperature} {t_damping:.1f} ")
            f.write(f"iso {self.parameters.pressure} {self.parameters.pressure} {p_damping:.1f}\n")
        
        f.write("\n")
        f.write(f"# Run production\n")
        f.write(f"run {production_steps}\n")
        f.write("\n")
        
        f.write("# Final output\n")
        f.write("write_data final_structure.data\n")
        f.write('print \"Salt water MD simulation completed successfully\"\n')