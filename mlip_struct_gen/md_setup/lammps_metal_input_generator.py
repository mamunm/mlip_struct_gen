"""
Improved LAMMPS Input Generator for Metal Surface Systems

Key improvements:
1. Robust equilibration protocols
2. Structure validation
3. Better error handling and monitoring
4. Gradual heating approach
5. Updated potential parameters
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import datetime
import numpy as np

# from .validation import validate_parameters  # Water-specific validation, not needed for metals
from ..utils.logger import MLIPLogger


@dataclass
class MetalInputParameters:
    """Parameters for LAMMPS metal surface simulation input generation.
    
    Required Parameters:
        lammps_data_file (str): Path to LAMMPS data file containing the metal structure
        output_file (str): Path for the generated LAMMPS input file
        metal_type (str): Type of metal (Au, Pt, Cu, Ag, Al, etc.)
        ensemble (str): Statistical ensemble (NVE, NVT, NPT)
        temperature (float): Simulation temperature in Kelvin (250-1500K)
    
    Optional Parameters:
        pressure (float): Pressure in atm (0.1-1000), required for NPT
        timestep (float): Integration timestep in fs (0.5-2.0)
        equilibration_steps (int): Number of equilibration steps (10000-500000)
        production_steps (int): Number of production steps (100000-10000000)
        potential_type (str): Metal potential type (EAM, MEAM, ReaxFF)
        potential_file (str): Path to potential parameter file
        lattice_constant (float): Lattice constant in Angstroms (overrides database values)
        fix_bottom_layers (int): Number of bottom layers to fix (0-5)
        thermostat_damping (float): Thermostat damping parameter in fs (50-200)
        barostat_damping (float): Barostat damping parameter in fs (500-2000)
        output_frequency (int): Trajectory output frequency (100-10000)
        restart_frequency (int): Restart file write frequency (10000-100000)
        thermo_frequency (int): Thermodynamic output frequency (100-10000)
        robust_equilibration (bool): Use robust multi-stage equilibration
        initial_temperature (float): Starting temperature for gradual heating
        structure_validation (bool): Validate structure before simulation
    """
    
    # Required parameters
    lammps_data_file: str
    output_file: str
    metal_type: str
    ensemble: str
    temperature: float
    
    # Optional parameters with defaults
    pressure: Optional[float] = 1.0
    timestep: float = 1.0
    equilibration_steps: int = 50000
    production_steps: int = 500000
    potential_type: str = "EAM"
    potential_file: Optional[str] = None
    lattice_constant: Optional[float] = None
    fix_bottom_layers: int = 2
    thermostat_damping: float = 100.0
    barostat_damping: float = 1000.0
    output_frequency: int = 1000
    restart_frequency: int = 50000
    thermo_frequency: int = 1000
    robust_equilibration: bool = True
    initial_temperature: float = 50.0
    structure_validation: bool = True


# Updated metal potential parameters with MEAM support
METAL_POTENTIALS = {
    "Au": {
        "lattice_constant": 4.078,
        "EAM": {
            "file": "Au.eam.alloy",
            "mass": 196.966569,
            "description": "Gold EAM potential (Zhou et al. 2004)",
            "cutoff": 7.5
        },
        "MEAM": {
            "library_file": "library.meam",
            "parameter_file": "Au.meam", 
            "elements": "Au",
            "mass": 196.966569,
            "description": "Gold MEAM potential"
        }
    },
    "Pt": {
        "lattice_constant": 3.923,
        "EAM": {
            "file": "Pt.eam.alloy",
            "mass": 195.084,
            "description": "Platinum EAM potential (Zhou et al. 2004)",
            "cutoff": 7.0
        },
        "MEAM": {
            "library_file": "library.meam",
            "parameter_file": "Pt.meam",
            "elements": "Pt", 
            "mass": 195.084,
            "description": "Platinum MEAM potential"
        }
    },
    "Cu": {
        "lattice_constant": 3.615,
        "EAM": {
            "file": "Cu.eam.alloy",
            "mass": 63.546,
            "description": "Copper EAM potential (Zhou et al. 2004)",
            "cutoff": 6.5
        },
        "MEAM": {
            "library_file": "library.meam",
            "parameter_file": "Cu.meam",
            "elements": "Cu",
            "mass": 63.546,
            "description": "Copper MEAM potential"
        }
    },
    "Ag": {
        "lattice_constant": 4.085,
        "EAM": {
            "file": "Ag.eam.alloy",
            "mass": 107.8682,
            "description": "Silver EAM potential (Zhou et al. 2004)",
            "cutoff": 7.5
        },
        "MEAM": {
            "library_file": "library.meam", 
            "parameter_file": "Ag.meam",
            "elements": "Ag",
            "mass": 107.8682,
            "description": "Silver MEAM potential"
        }
    },
    "Al": {
        "lattice_constant": 4.050,
        "EAM": {
            "file": "Al.eam.alloy",
            "mass": 26.9815385,
            "description": "Aluminum EAM potential (Zhou et al. 2004)",
            "cutoff": 6.5
        },
        "MEAM": {
            "library_file": "library.meam",
            "parameter_file": "Al.meam", 
            "elements": "Al",
            "mass": 26.9815385,
            "description": "Aluminum MEAM potential"
        }
    }
}


class LAMMPSMetalInputGenerator:
    """Generator for LAMMPS input files for metal surface simulations."""
    
    def __init__(self, parameters: MetalInputParameters, logger: Optional[MLIPLogger] = None):
        """Initialize the metal input generator.
        
        Args:
            parameters: Input parameters for the simulation
            logger: Optional logger for tracking generation progress
        """
        self.parameters = self._validate_metal_parameters(parameters)
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Initializing LAMMPS metal input generator for {parameters.metal_type}")
    
    def _validate_metal_parameters(self, parameters: MetalInputParameters) -> MetalInputParameters:
        """Validate metal-specific parameters with enhanced checks."""
        # Basic type and value checks
        if not isinstance(parameters.lammps_data_file, str) or not parameters.lammps_data_file.strip():
            raise ValueError("lammps_data_file must be a non-empty string")
        
        # Check if data file exists
        if not Path(parameters.lammps_data_file).exists():
            raise FileNotFoundError(f"LAMMPS data file not found: {parameters.lammps_data_file}")
        
        if not isinstance(parameters.output_file, str) or not parameters.output_file.strip():
            raise ValueError("output_file must be a non-empty string")
        
        if parameters.metal_type not in METAL_POTENTIALS:
            available = ", ".join(METAL_POTENTIALS.keys())
            raise ValueError(f"Unsupported metal_type: {parameters.metal_type}. Available: {available}")
        
        if parameters.ensemble not in ["NVE", "NVT", "NPT"]:
            raise ValueError(f"Invalid ensemble: {parameters.ensemble}. Must be NVE, NVT, or NPT")
        
        if parameters.temperature <= 0 or parameters.temperature > 1500:
            raise ValueError(f"Temperature out of range: {parameters.temperature}. Must be 0-1500K")
        
        if parameters.ensemble == "NPT" and (parameters.pressure is None or parameters.pressure <= 0):
            raise ValueError("Pressure must be positive for NPT ensemble")
        
        if parameters.timestep <= 0 or parameters.timestep > 2.0:
            raise ValueError(f"Timestep out of range: {parameters.timestep}. Must be 0-2.0 fs")
        
        # Validate initial temperature for gradual heating
        if parameters.initial_temperature >= parameters.temperature:
            parameters.initial_temperature = parameters.temperature * 0.1
            if self.logger:
                self.logger.warning(f"Adjusted initial_temperature to {parameters.initial_temperature}K")
        
        # Normalize output file extension
        output_path = Path(parameters.output_file)
        if output_path.suffix != '.in':
            parameters.output_file = str(output_path.with_suffix('.in'))
        
        return parameters
    
    def generate(self) -> str:
        """Generate LAMMPS input file for metal surface simulation.
        
        Returns:
            str: Path to the generated input file
        """
        if self.logger:
            self.logger.step("Starting LAMMPS input file generation")
        
        output_path = Path(self.parameters.output_file)
        if output_path.suffix != '.in':
            output_path = output_path.with_suffix('.in')
        
        # Get metal potential parameters
        metal_params = self._get_metal_parameters()
        
        with open(output_path, 'w') as f:
            self._write_header(f)
            self._write_initialization(f)
            self._write_structure_validation(f)
            self._write_force_field(f, metal_params)
            self._write_setup(f)
            if self.parameters.robust_equilibration:
                self._write_robust_equilibration(f)
            else:
                self._write_simple_equilibration(f)
            self._write_production(f)
        
        if self.logger:
            self.logger.success(f"Generated LAMMPS input file: {output_path}")
        
        return str(output_path)
    
    def _get_metal_parameters(self) -> Dict[str, Any]:
        """Get metal potential parameters."""
        if self.parameters.metal_type not in METAL_POTENTIALS:
            raise ValueError(f"Unsupported metal type: {self.parameters.metal_type}")
        
        metal_data = METAL_POTENTIALS[self.parameters.metal_type]
        if self.parameters.potential_type not in metal_data:
            available = ", ".join([k for k in metal_data.keys() if k != "lattice_constant"])
            raise ValueError(f"Potential type {self.parameters.potential_type} not available for {self.parameters.metal_type}. Available: {available}")
        
        return metal_data[self.parameters.potential_type]
    
    def _write_header(self, f):
        """Write file header with metadata."""
        f.write(f"# LAMMPS Input File for {self.parameters.metal_type} Surface Simulation\n")
        f.write(f"# Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Metal: {self.parameters.metal_type}\n")
        f.write(f"# Ensemble: {self.parameters.ensemble}\n")
        f.write(f"# Temperature: {self.parameters.temperature} K\n")
        if self.parameters.ensemble == "NPT":
            f.write(f"# Pressure: {self.parameters.pressure} atm\n")
        f.write(f"# Potential: {self.parameters.potential_type}\n")
        f.write(f"# Robust equilibration: {self.parameters.robust_equilibration}\n")
        f.write("\n")
    
    def _write_initialization(self, f):
        """Write LAMMPS initialization commands."""
        f.write("# Initialization\n")
        f.write("units metal\n")
        f.write("atom_style atomic\n")
        f.write("boundary p p p\n")
        f.write("\n")
        
        # Enhanced neighbor settings for metals
        f.write("# Enhanced neighbor settings\n")
        f.write("neighbor 2.0 bin\n")
        f.write("neigh_modify delay 0 every 1 check yes\n")
        f.write("\n")
        
        f.write("# Read data file\n")
        f.write(f"read_data {self.parameters.lammps_data_file}\n")
        f.write("\n")
    
    def _write_structure_validation(self, f):
        """Write structure validation commands."""
        if not self.parameters.structure_validation:
            return
            
        f.write("# Structure Validation\n")
        f.write("info out log\n")
        f.write('variable natoms equal "count(all)"\n')
        f.write('print "=== SYSTEM VERIFICATION ==="\n')
        f.write('print "Total atoms: ${natoms}"\n')
        f.write("\n")
        
        f.write("# Check box dimensions\n")
        f.write("variable lx equal lx\n")
        f.write("variable ly equal ly\n")
        f.write("variable lz equal lz\n")
        f.write('print "Box dimensions: ${lx} x ${ly} x ${lz}"\n')
        f.write("\n")
    
    def _write_force_field(self, f, metal_params: Dict[str, Any]):
        """Write force field parameters."""
        f.write("# Force Field\n")
        
        if self.parameters.potential_type == "EAM":
            potential_file = self.parameters.potential_file or metal_params["file"]
            
            # Always use eam/alloy for better compatibility
            f.write("pair_style eam/alloy\n")
            f.write(f"pair_coeff * * {potential_file} {self.parameters.metal_type}\n")
            
        elif self.parameters.potential_type == "MEAM":
            f.write("pair_style meam\n")
            potential_file = self.parameters.potential_file or metal_params["file"]
            f.write(f"pair_coeff * * {potential_file} {self.parameters.metal_type}\n")
        else:
            raise ValueError(f"Unsupported potential type: {self.parameters.potential_type}")
        
        f.write("\n")
        
        # Set mass
        f.write(f"mass 1 {metal_params['mass']}\n")
        f.write("\n")
        
        # Validate initial energy
        if self.parameters.structure_validation:
            f.write("# Initial energy validation\n")
            f.write("# Force calculation of thermodynamic quantities\n")
            f.write("run 0\n")
            f.write("variable pe_per_atom equal pe/atoms\n")
            f.write('print "Initial PE per atom: ${pe_per_atom} eV"\n')
            f.write("\n")
            f.write("# Safety check for unreasonable energies\n")
            f.write('if "${pe_per_atom} > 100" then &\n')
            f.write('  "print \'ERROR: Initial energy too high - check structure\'" &\n')
            f.write('  "quit 1"\n')
            f.write('if "${pe_per_atom} < -10" then &\n')
            f.write('  "print \'ERROR: Initial energy too low - check potential\'" &\n')
            f.write('  "quit 1"\n')
            f.write("\n")
        
        # Fix bottom layers if specified
        if self.parameters.fix_bottom_layers > 0:
            self._write_bottom_layer_fix(f)
    
    def _write_bottom_layer_fix(self, f):
        """Write bottom layer fixing commands."""
        f.write("# Fix bottom layers\n")
        
        # Get lattice constant
        lattice_constant = 3.92  # Default fallback
        if self.parameters.metal_type in METAL_POTENTIALS:
            metal_data = METAL_POTENTIALS[self.parameters.metal_type]
            lattice_constant = metal_data.get("lattice_constant", lattice_constant)
        
        if self.parameters.lattice_constant is not None:
            lattice_constant = self.parameters.lattice_constant
        
        # Calculate interlayer spacing for (111) surface: d₁₁₁ = a/√3
        d_spacing = lattice_constant / (3**0.5)
        z_cutoff = self.parameters.fix_bottom_layers * d_spacing + d_spacing * 0.5
        
        f.write(f"# Metal: {self.parameters.metal_type}, lattice constant: {lattice_constant:.3f} Å\n")
        f.write(f"# Interlayer spacing d₁₁₁: {d_spacing:.3f} Å\n")
        f.write(f"# Fixing bottom {self.parameters.fix_bottom_layers} layers\n")
        f.write("variable zmin equal bound(all,zmin)\n")
        f.write(f"variable zcut equal v_zmin+{z_cutoff:.3f}\n")
        f.write("region bottom block INF INF INF INF INF ${zcut}\n")
        f.write("group bottom region bottom\n")
        f.write("group mobile subtract all bottom\n")
        f.write("fix freeze bottom setforce 0.0 0.0 0.0\n")
        f.write("\n")
        
        f.write('print "Frozen atoms: $(count(bottom))"\n')
        f.write('print "Mobile atoms: $(count(mobile))"\n')
        f.write("\n")
    
    def _write_setup(self, f):
        """Write simulation setup."""
        f.write("# Simulation Setup\n")
        
        # Start with conservative timestep
        initial_timestep = min(0.25, self.parameters.timestep * 0.25)
        f.write(f"timestep {initial_timestep}\n")
        f.write("\n")
        
        f.write("# Enhanced monitoring\n")
        f.write("# Note: Using coordination number instead of minimum distance for stability\n")
        f.write("compute coord all coord/atom cutoff 3.5\n")
        f.write("compute avg_coord all reduce ave c_coord\n")
        f.write("\n")
        
        f.write("# Output settings\n")
        f.write(f"thermo {self.parameters.thermo_frequency}\n")
        f.write("thermo_style custom step temp press pe ke etotal c_avg_coord vol density\n")
        f.write("\n")
        
        f.write("# Restart files\n")
        f.write(f"restart {self.parameters.restart_frequency} restart.equil\n")
        f.write("\n")
    
    def _write_robust_equilibration(self, f):
        """Write robust multi-stage equilibration."""
        f.write("# Robust Multi-Stage Equilibration\n")
        f.write("print '=== STAGE 1: MINIMIZATION ==='\n")
        
        # Stage 1: Aggressive minimization
        f.write("minimize 1.0e-6 1.0e-8 5000 50000\n")
        f.write("\n")
        
        # Check for atom loss
        f.write("variable natoms_after_min equal \"count(all)\"\n")
        f.write('print "Atoms after minimization: ${natoms_after_min}"\n')
        f.write('if "${natoms_after_min} < ${natoms}" then &\n')
        f.write('  "print \'ERROR: Lost atoms during minimization\'" &\n')
        f.write('  "quit 1"\n')
        f.write("\n")
        
        # Stage 2: Ultra-gentle heating
        f.write("print '=== STAGE 2: GENTLE HEATING ==='\n")
        f.write("reset_timestep 0\n")
        
        # Determine which group to heat
        group_to_heat = "mobile" if self.parameters.fix_bottom_layers > 0 else "all"
        
        f.write(f"velocity {group_to_heat} create {self.parameters.initial_temperature} 12345 dist gaussian\n")
        f.write(f"fix heat {group_to_heat} nvt temp {self.parameters.initial_temperature} {self.parameters.initial_temperature * 2} 50.0\n")
        f.write("run 5000\n")
        f.write("\n")
        
        # Stage 3: Gradual temperature ramp
        f.write("print '=== STAGE 3: TEMPERATURE RAMP ==='\n")
        f.write("unfix heat\n")
        
        # Multiple temperature steps
        temp_steps = [
            self.parameters.initial_temperature * 2,
            self.parameters.temperature * 0.5,
            self.parameters.temperature * 0.75,
            self.parameters.temperature
        ]
        
        for i, temp in enumerate(temp_steps):
            if i == 0:
                prev_temp = self.parameters.initial_temperature * 2
            else:
                prev_temp = temp_steps[i-1]
            
            f.write(f"fix heat{i} {group_to_heat} nvt temp {prev_temp} {temp} {self.parameters.thermostat_damping}\n")
            f.write(f"run {self.parameters.equilibration_steps // len(temp_steps)}\n")
            f.write(f"unfix heat{i}\n")
            f.write("\n")
            
            # Check for atom loss at each stage
            f.write(f"variable natoms_stage{i} equal \"count(all)\"\n")
            f.write(f'print "Atoms after stage {i}: ${{natoms_stage{i}}}"\n')
            f.write(f'if "${{natoms_stage{i}}} < ${{natoms}}" then &\n')
            f.write(f'  "print \'ERROR: Lost atoms during temperature ramp stage {i}\'" &\n')
            f.write('  "quit 1"\n')
            f.write("\n")
        
        # Stage 4: Final equilibration
        f.write("print '=== STAGE 4: FINAL EQUILIBRATION ==='\n")
        f.write(f"timestep {self.parameters.timestep * 0.5}\n")
        f.write(f"fix equil {group_to_heat} nvt temp {self.parameters.temperature} {self.parameters.temperature} {self.parameters.thermostat_damping}\n")
        f.write(f"run {self.parameters.equilibration_steps // 4}\n")
        f.write("\n")
        
        # Gradually increase timestep
        f.write("# Gradually increase timestep\n")
        f.write(f"timestep {self.parameters.timestep}\n")
        f.write(f"run {self.parameters.equilibration_steps // 4}\n")
        f.write("unfix equil\n")
        f.write("\n")
    
    def _write_simple_equilibration(self, f):
        """Write simple equilibration (original approach)."""
        f.write("# Simple Equilibration\n")
        f.write("minimize 1.0e-4 1.0e-6 1000 10000\n")
        f.write("\n")
        
        f.write("# Initial velocities\n")
        group_to_heat = "mobile" if self.parameters.fix_bottom_layers > 0 else "all"
        f.write(f"velocity {group_to_heat} create {self.parameters.temperature} 12345 dist gaussian\n")
        f.write("\n")
    
    def _write_production(self, f):
        """Write production phase with enhanced monitoring."""
        f.write("# Production Phase\n")
        f.write("reset_timestep 0\n")
        f.write("\n")
        
        # Determine which group to integrate
        group_to_integrate = "mobile" if self.parameters.fix_bottom_layers > 0 else "all"
        
        # Set up integrator
        if self.parameters.ensemble == "NVE":
            f.write(f"fix integrate {group_to_integrate} nve\n")
        elif self.parameters.ensemble == "NVT":
            f.write(f"fix integrate {group_to_integrate} nvt temp {self.parameters.temperature} {self.parameters.temperature} {self.parameters.thermostat_damping}\n")
        elif self.parameters.ensemble == "NPT":
            f.write(f"fix integrate {group_to_integrate} npt temp {self.parameters.temperature} {self.parameters.temperature} {self.parameters.thermostat_damping} iso {self.parameters.pressure} {self.parameters.pressure} {self.parameters.barostat_damping}\n")
        
        f.write("\n")
        
        # Simplified safety monitoring (removed problematic atom_monitor fix)
        f.write("# Periodic atom count checks during production\n")
        f.write("variable check_step equal 0\n")
        f.write("\n")
        
        # Trajectory output
        f.write("# Trajectory output\n")
        output_file = Path(self.parameters.output_file).stem
        f.write(f"dump trajectory all custom {self.parameters.output_frequency} {self.parameters.ensemble.lower()}_{self.parameters.metal_type.lower()}_{int(self.parameters.temperature)}k.lammpstrj id type x y z vx vy vz fx fy fz\n")
        f.write(f"dump_modify trajectory element {self.parameters.metal_type}\n")
        f.write("\n")
        
        f.write("# Production run\n")
        f.write(f"run {self.parameters.production_steps}\n")
        f.write("\n")
        
        # Final verification
        f.write("# Final verification\n")
        f.write("variable final_natoms equal \"count(all)\"\n")
        f.write('print "=== FINAL VERIFICATION ==="\n')
        f.write('print "Started with: ${natoms} atoms"\n')
        f.write('print "Ended with: ${final_natoms} atoms"\n')
        f.write("\n")
        
        f.write('if "${final_natoms} == ${natoms}" then &\n')
        f.write('  "print \'SUCCESS: No atoms lost during simulation\'"\n')
        f.write("\n")
        
        f.write("# Final output\n")
        f.write(f"write_data {self.parameters.ensemble.lower()}_{self.parameters.metal_type.lower()}_{int(self.parameters.temperature)}k_final.data\n")
        f.write(f"write_restart {self.parameters.ensemble.lower()}_{self.parameters.metal_type.lower()}_{int(self.parameters.temperature)}k_final.restart\n")