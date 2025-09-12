"""
LAMMPS Input Generator for Metal-Water Interface Systems

This module generates LAMMPS input files optimized for metal-water interface simulations.
Combines metal surface potentials with water models for realistic interface studies.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import datetime

# from .validation import validate_parameters  # Water-specific validation, not suitable for metal-water
from ..utils.logger import MLIPLogger
from .lammps_metal_input_generator import METAL_POTENTIALS


@dataclass
class MetalWaterInputParameters:
    """Parameters for LAMMPS metal-water interface simulation input generation.
    
    Required Parameters:
        lammps_data_file (str): Path to LAMMPS data file containing the metal-water structure
        output_file (str): Path for the generated LAMMPS input file
        metal_type (str): Type of metal (Au, Pt, Cu, Ag, Al, etc.)
        water_model (str): Water model (SPC/E, TIP3P, TIP4P)
        ensemble (str): Statistical ensemble (NVE, NVT, NPT)
        temperature (float): Simulation temperature in Kelvin (250-400K)
    
    Optional Parameters:
        pressure (float): Pressure in atm (0.1-10), required for NPT
        timestep (float): Integration timestep in fs (0.5-1.0)
        equilibration_steps (int): Number of equilibration steps (50000-500000)
        production_steps (int): Number of production steps (500000-5000000)
        metal_potential_type (str): Metal potential type (EAM, MEAM)
        metal_potential_file (str): Path to metal potential parameter file
        fix_bottom_layers (int): Number of bottom metal layers to fix (0-5)
        thermostat_damping (float): Thermostat damping parameter in fs (50-200)
        barostat_damping (float): Barostat damping parameter in fs (500-2000)
        output_frequency (int): Trajectory output frequency (100-5000)
        restart_frequency (int): Restart file write frequency (10000-100000)
        thermo_frequency (int): Thermodynamic output frequency (100-5000)
        ewald_accuracy (float): PPPM accuracy (1e-4 to 1e-6)
        cutoff_lj (float): LJ cutoff distance in Angstroms (8.0-12.0)
        cutoff_coul (float): Coulombic cutoff distance in Angstroms (8.0-12.0)
    """
    
    # Required parameters
    lammps_data_file: str
    output_file: str
    metal_type: str
    water_model: str
    ensemble: str
    temperature: float
    
    # Optional parameters with defaults
    pressure: Optional[float] = 1.0
    timestep: float = 0.5
    equilibration_steps: int = 100000
    production_steps: int = 1000000
    metal_potential_type: str = "EAM"
    metal_potential_file: Optional[str] = None
    fix_bottom_layers: int = 2
    thermostat_damping: float = 100.0
    barostat_damping: float = 1000.0
    output_frequency: int = 2000
    restart_frequency: int = 50000
    thermo_frequency: int = 1000
    ewald_accuracy: float = 1e-5
    cutoff_lj: float = 10.0
    cutoff_coul: float = 10.0


# Water model parameters for metal-water interfaces
WATER_MODEL_PARAMETERS = {
    "SPC/E": {
        "description": "Simple Point Charge/Extended model - good for metal interfaces",
        "lj_params": {
            "O": {"epsilon": 0.1553, "sigma": 3.166},
            "H": {"epsilon": 0.0, "sigma": 0.0}
        },
        "charges": {"O": -0.8476, "H": 0.4238},
        "bond_params": {"OH": {"k": 1000.0, "r0": 1.0}},
        "angle_params": {"HOH": {"k": 100.0, "theta0": 109.47}},
        "use_shake": True,
        "shake_bonds": "1",
        "shake_angles": "1"
    },
    "TIP3P": {
        "description": "Transferable Intermolecular Potential 3-Point",
        "lj_params": {
            "O": {"epsilon": 0.1521, "sigma": 3.1507},
            "H": {"epsilon": 0.0, "sigma": 0.0}
        },
        "charges": {"O": -0.834, "H": 0.417},
        "bond_params": {"OH": {"k": 450.0, "r0": 0.9572}},
        "angle_params": {"HOH": {"k": 55.0, "theta0": 104.52}},
        "use_shake": True,
        "shake_bonds": "1",
        "shake_angles": "1"
    },
    "TIP4P": {
        "description": "Transferable Intermolecular Potential 4-Point",
        "lj_params": {
            "O": {"epsilon": 0.1550, "sigma": 3.1536},
            "H": {"epsilon": 0.0, "sigma": 0.0}
        },
        "charges": {"O": 0.0, "H": 0.520, "M": -1.040},  # M is virtual site
        "bond_params": {"OH": {"k": 450.0, "r0": 0.9572}},
        "angle_params": {"HOH": {"k": 55.0, "theta0": 104.52}},
        "use_shake": True,
        "shake_bonds": "1",
        "shake_angles": "1"
    }
}

# Metal-water cross interaction parameters (Lorentz-Berthelot mixing)
METAL_WATER_INTERACTIONS = {
    "Au": {
        "O": {"epsilon": 0.039, "sigma": 2.934},  # Au-O interaction
        "H": {"epsilon": 0.0, "sigma": 0.0}      # Au-H interaction (negligible)
    },
    "Pt": {
        "O": {"epsilon": 0.045, "sigma": 2.895},
        "H": {"epsilon": 0.0, "sigma": 0.0}
    },
    "Cu": {
        "O": {"epsilon": 0.042, "sigma": 2.845},
        "H": {"epsilon": 0.0, "sigma": 0.0}
    },
    "Ag": {
        "O": {"epsilon": 0.038, "sigma": 2.924},
        "H": {"epsilon": 0.0, "sigma": 0.0}
    },
    "Al": {
        "O": {"epsilon": 0.041, "sigma": 2.756},
        "H": {"epsilon": 0.0, "sigma": 0.0}
    }
}


class LAMMPSMetalWaterInputGenerator:
    """Generator for LAMMPS input files for metal-water interface simulations."""
    
    def __init__(self, parameters: MetalWaterInputParameters, logger: Optional[MLIPLogger] = None):
        """Initialize the metal-water input generator.
        
        Args:
            parameters: Input parameters for the simulation
            logger: Optional logger for tracking generation progress
        """
        self.parameters = self._validate_metal_water_parameters(parameters)
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Initializing LAMMPS metal-water input generator for {parameters.metal_type}-{parameters.water_model}")
    
    def _validate_metal_water_parameters(self, parameters: MetalWaterInputParameters) -> MetalWaterInputParameters:
        """Validate metal-water specific parameters."""
        # Basic type and value checks
        if not isinstance(parameters.lammps_data_file, str) or not parameters.lammps_data_file.strip():
            raise ValueError("lammps_data_file must be a non-empty string")
        
        if not isinstance(parameters.output_file, str) or not parameters.output_file.strip():
            raise ValueError("output_file must be a non-empty string")
        
        if parameters.metal_type not in METAL_POTENTIALS:
            available = ", ".join(METAL_POTENTIALS.keys())
            raise ValueError(f"Unsupported metal_type: {parameters.metal_type}. Available: {available}")
        
        if parameters.water_model not in WATER_MODEL_PARAMETERS:
            available = ", ".join(WATER_MODEL_PARAMETERS.keys())
            raise ValueError(f"Unsupported water_model: {parameters.water_model}. Available: {available}")
        
        if parameters.ensemble not in ["NVE", "NVT", "NPT"]:
            raise ValueError(f"Invalid ensemble: {parameters.ensemble}. Must be NVE, NVT, or NPT")
        
        if parameters.temperature <= 0 or parameters.temperature > 400:
            raise ValueError(f"Temperature out of range: {parameters.temperature}. Must be 0-400K for water systems")
        
        if parameters.ensemble == "NPT" and (parameters.pressure is None or parameters.pressure <= 0):
            raise ValueError("Pressure must be positive for NPT ensemble")
        
        if parameters.timestep <= 0 or parameters.timestep > 1.0:
            raise ValueError(f"Timestep out of range: {parameters.timestep}. Must be 0-1.0 fs for metal-water systems")
        
        # Normalize output file extension
        output_path = Path(parameters.output_file)
        if output_path.suffix != '.in':
            parameters.output_file = str(output_path.with_suffix('.in'))
        
        return parameters
    
    def generate(self) -> str:
        """Generate LAMMPS input file for metal-water interface simulation.
        
        Returns:
            str: Path to the generated input file
        """
        if self.logger:
            self.logger.step("Starting LAMMPS metal-water input file generation")
        
        output_path = Path(self.parameters.output_file)
        if output_path.suffix != '.in':
            output_path = output_path.with_suffix('.in')
        
        # Get parameters
        metal_params = self._get_metal_parameters()
        water_params = self._get_water_parameters()
        atom_types = self._get_atom_types_from_data()
        
        with open(output_path, 'w') as f:
            self._write_header(f)
            self._write_initialization(f)
            self._write_force_field(f, metal_params, water_params, atom_types)
            self._write_setup(f, water_params)
            self._write_equilibration(f, water_params)
            self._write_production(f)
        
        if self.logger:
            self.logger.success(f"Generated LAMMPS metal-water input file: {output_path}")
        
        return str(output_path)
    
    def _get_metal_parameters(self) -> Dict[str, Any]:
        """Get metal potential parameters."""
        if self.parameters.metal_type not in METAL_POTENTIALS:
            raise ValueError(f"Unsupported metal type: {self.parameters.metal_type}")
        
        metal_data = METAL_POTENTIALS[self.parameters.metal_type]
        if self.parameters.metal_potential_type not in metal_data:
            available = ", ".join(metal_data.keys())
            raise ValueError(f"Potential type {self.parameters.metal_potential_type} not available for {self.parameters.metal_type}. Available: {available}")
        
        return metal_data[self.parameters.metal_potential_type]
    
    def _get_water_parameters(self) -> Dict[str, Any]:
        """Get water model parameters."""
        if self.parameters.water_model not in WATER_MODEL_PARAMETERS:
            available = ", ".join(WATER_MODEL_PARAMETERS.keys())
            raise ValueError(f"Unsupported water model: {self.parameters.water_model}. Available: {available}")
        
        return WATER_MODEL_PARAMETERS[self.parameters.water_model]
    
    def _get_atom_types_from_data(self) -> Dict[str, int]:
        """Parse LAMMPS data file to determine atom types."""
        atom_types = {}
        
        try:
            with open(self.parameters.lammps_data_file, 'r') as f:
                lines = f.readlines()
            
            # Look for atom type information in data file
            in_masses = False
            type_counter = 1
            
            for line in lines:
                line = line.strip()
                if "Masses" in line:
                    in_masses = True
                    continue
                elif in_masses and line == "":
                    break
                elif in_masses and line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        type_id = int(parts[0])
                        # Infer atom type from mass
                        mass = float(parts[1])
                        if mass > 50:  # Likely metal
                            atom_types[self.parameters.metal_type] = type_id
                        elif 15 < mass < 17:  # Oxygen
                            atom_types["O"] = type_id
                        elif mass < 2:  # Hydrogen
                            atom_types["H"] = type_id
                        type_counter += 1
        
        except FileNotFoundError:
            # Default mapping if file not found
            atom_types = {self.parameters.metal_type: 1, "O": 2, "H": 3}
        
        return atom_types
    
    def _write_header(self, f):
        """Write file header with metadata."""
        f.write(f"# LAMMPS Input File for {self.parameters.metal_type}-{self.parameters.water_model} Interface Simulation\n")
        f.write(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Metal: {self.parameters.metal_type} ({self.parameters.metal_potential_type})\n")
        f.write(f"# Water Model: {self.parameters.water_model}\n")
        f.write(f"# Ensemble: {self.parameters.ensemble}\n")
        f.write(f"# Temperature: {self.parameters.temperature} K\n")
        if self.parameters.ensemble == "NPT":
            f.write(f"# Pressure: {self.parameters.pressure} atm\n")
        f.write("\n")
    
    def _write_initialization(self, f):
        """Write LAMMPS initialization commands."""
        f.write("# Initialization\n")
        f.write("units metal\n")
        f.write("atom_style full\n")
        f.write("boundary p p p\n")
        f.write("\n")
        
        f.write("# Read data file\n")
        f.write(f"read_data {self.parameters.lammps_data_file}\n")
        f.write("\n")
    
    def _write_force_field(self, f, metal_params: Dict[str, Any], water_params: Dict[str, Any], atom_types: Dict[str, int]):
        """Write force field parameters."""
        f.write("# Force Field\n")
        
        # Hybrid pair style for metal and water
        f.write("pair_style hybrid")
        if self.parameters.metal_potential_type == "EAM":
            f.write(" eam")
        elif self.parameters.metal_potential_type == "MEAM":
            f.write(" meam")
        f.write(f" lj/cut/coul/long {self.parameters.cutoff_lj} {self.parameters.cutoff_coul}\n")
        
        # Metal potential
        metal_type_id = atom_types.get(self.parameters.metal_type, 1)
        if self.parameters.metal_potential_type == "EAM":
            potential_file = self.parameters.metal_potential_file or metal_params["file"]
            f.write(f"pair_coeff {metal_type_id} {metal_type_id} eam {potential_file}\n")
        elif self.parameters.metal_potential_type == "MEAM":
            potential_file = self.parameters.metal_potential_file or metal_params["file"]
            f.write(f"pair_coeff {metal_type_id} {metal_type_id} meam {potential_file} {self.parameters.metal_type}\n")
        
        # Water-water interactions
        o_type = atom_types.get("O", 2)
        h_type = atom_types.get("H", 3)
        
        o_params = water_params["lj_params"]["O"]
        f.write(f"pair_coeff {o_type} {o_type} lj/cut/coul/long {o_params['epsilon']} {o_params['sigma']}\n")
        f.write(f"pair_coeff {h_type} {h_type} lj/cut/coul/long 0.0 0.0\n")
        f.write(f"pair_coeff {o_type} {h_type} lj/cut/coul/long 0.0 0.0\n")
        
        # Metal-water cross interactions
        if self.parameters.metal_type in METAL_WATER_INTERACTIONS:
            cross_params = METAL_WATER_INTERACTIONS[self.parameters.metal_type]
            mo_params = cross_params["O"]
            f.write(f"pair_coeff {metal_type_id} {o_type} lj/cut/coul/long {mo_params['epsilon']} {mo_params['sigma']}\n")
            f.write(f"pair_coeff {metal_type_id} {h_type} lj/cut/coul/long 0.0 0.0\n")
        
        f.write("\n")
        
        # Masses
        f.write("# Masses\n")
        f.write(f"mass {metal_type_id} {metal_params['mass']}\n")
        f.write(f"mass {o_type} 15.9994\n")
        f.write(f"mass {h_type} 1.008\n")
        f.write("\n")
        
        # Bond and angle parameters for water
        f.write("# Water bonds and angles\n")
        bond_params = water_params["bond_params"]["OH"]
        f.write(f"bond_style harmonic\n")
        f.write(f"bond_coeff 1 {bond_params['k']} {bond_params['r0']}\n")
        
        angle_params = water_params["angle_params"]["HOH"]
        f.write(f"angle_style harmonic\n")
        f.write(f"angle_coeff 1 {angle_params['k']} {angle_params['theta0']}\n")
        f.write("\n")
        
        # Electrostatics
        f.write("# Electrostatics\n")
        f.write(f"kspace_style pppm {self.parameters.ewald_accuracy}\n")
        f.write("\n")
        
        # SHAKE constraints for water
        if water_params["use_shake"]:
            f.write("# SHAKE constraints for water\n")
            f.write(f"fix shake all shake 0.0001 20 0 b {water_params['shake_bonds']} a {water_params['shake_angles']}\n")
            f.write("\n")
        
        # Fix bottom metal layers
        if self.parameters.fix_bottom_layers > 0:
            f.write("# Fix bottom metal layers\n")
            f.write(f"group metal type {metal_type_id}\n")
            f.write(f"region bottom block INF INF INF INF INF {self.parameters.fix_bottom_layers}\n")
            f.write("group bottom intersect metal region bottom\n")
            f.write("fix freeze bottom setforce 0.0 0.0 0.0\n")
            f.write("\n")
        
        # Group definitions
        f.write("# Group definitions\n")
        f.write(f"group metal type {metal_type_id}\n")
        f.write(f"group water type {o_type} {h_type}\n")
        f.write(f"group oxygen type {o_type}\n")
        f.write(f"group hydrogen type {h_type}\n")
        f.write("\n")
    
    def _write_setup(self, f, water_params: Dict[str, Any]):
        """Write simulation setup."""
        f.write("# Simulation Setup\n")
        f.write(f"timestep {self.parameters.timestep}\n")
        f.write("\n")
        
        f.write("# Initial velocities\n")
        f.write(f"velocity all create {self.parameters.temperature} 12345 dist gaussian\n")
        f.write("\n")
        
        f.write("# Output settings\n")
        f.write(f"thermo {self.parameters.thermo_frequency}\n")
        f.write("thermo_style custom step temp press pe ke etotal vol density\n")
        f.write("\n")
        
        f.write("# Restart files\n")
        f.write(f"restart {self.parameters.restart_frequency} restart.equil\n")
        f.write("\n")
    
    def _write_equilibration(self, f, water_params: Dict[str, Any]):
        """Write equilibration phase."""
        f.write("# Equilibration\n")
        f.write("fix integrate all nve\n")
        
        if self.parameters.ensemble in ["NVT", "NPT"]:
            f.write(f"fix thermostat all langevin {self.parameters.temperature} {self.parameters.temperature} {self.parameters.thermostat_damping} 48279\n")
        
        if self.parameters.ensemble == "NPT":
            f.write(f"fix barostat all press/berendsen iso {self.parameters.pressure} {self.parameters.pressure} {self.parameters.barostat_damping}\n")
        
        f.write(f"run {self.parameters.equilibration_steps}\n")
        f.write("\n")
        
        # Unfix integrators for production
        f.write("unfix integrate\n")
        if self.parameters.ensemble in ["NVT", "NPT"]:
            f.write("unfix thermostat\n")
        if self.parameters.ensemble == "NPT":
            f.write("unfix barostat\n")
        f.write("\n")
    
    def _write_production(self, f):
        """Write production phase."""
        f.write("# Production\n")
        f.write("reset_timestep 0\n")
        f.write("\n")
        
        # Set up integrator
        if self.parameters.ensemble == "NVE":
            f.write("fix integrate all nve\n")
        elif self.parameters.ensemble == "NVT":
            f.write(f"fix integrate all nvt temp {self.parameters.temperature} {self.parameters.temperature} {self.parameters.thermostat_damping}\n")
        elif self.parameters.ensemble == "NPT":
            f.write(f"fix integrate all npt temp {self.parameters.temperature} {self.parameters.temperature} {self.parameters.thermostat_damping} iso {self.parameters.pressure} {self.parameters.pressure} {self.parameters.barostat_damping}\n")
        
        f.write("\n")
        
        # Trajectory output
        f.write("# Trajectory output\n")
        output_file = Path(self.parameters.output_file).stem
        f.write(f"dump trajectory all custom {self.parameters.output_frequency} {output_file}.lammpstrj id type x y z vx vy vz fx fy fz\n")
        f.write(f"dump_modify trajectory element {self.parameters.metal_type} O H\n")
        f.write("\n")
        
        # Additional analysis outputs
        f.write("# Analysis outputs\n")
        f.write(f"compute rdf_mo all rdf 100 1 2  # Metal-Oxygen RDF\n")
        f.write(f"fix rdf_output all ave/time 100 10 {self.parameters.output_frequency} c_rdf_mo[*] file {output_file}_rdf.dat mode vector\n")
        f.write("\n")
        
        f.write("# Production run\n")
        f.write(f"run {self.parameters.production_steps}\n")
        f.write("\n")
        
        f.write("# Final output\n")
        f.write(f"write_data {output_file}_final.data\n")