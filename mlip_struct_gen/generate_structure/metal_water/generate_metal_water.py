"""Metal-water interface generation using ASE and Packmol."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    from ase import Atoms
    from ase.build import fcc111, fcc100, fcc110, surface, bulk
    from ase.io import write, read
    from ase.data import atomic_numbers, covalent_radii
    from ase.constraints import FixAtoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

from .input_parameters import MetalWaterParameters
from .validation import validate_parameters
from ..templates.water_models import (
    create_water_xyz,
    get_water_density,
    get_water_model
)


class MetalWaterGenerator:
    """Generate metal-water interfaces using ASE and Packmol."""
    
    # Experimental lattice constants for common FCC metals (Angstroms)
    LATTICE_CONSTANTS = {
        "Al": 4.050, "Au": 4.078, "Ag": 4.085, "Cu": 3.615,
        "Ni": 3.524, "Pd": 3.890, "Pt": 3.924, "Pb": 4.950,
        "Rh": 3.803, "Ir": 3.839, "Ca": 5.588, "Sr": 6.085
    }
    
    # Common Miller index to ASE function mapping
    SURFACE_BUILDERS = {
        (1, 1, 1): fcc111,
        (1, 0, 0): fcc100,
        (1, 1, 0): fcc110
    }
    
    def __init__(self, parameters: MetalWaterParameters):
        """
        Initialize metal-water interface generator.
        
        Args:
            parameters: Generation parameters
            
        Raises:
            ValueError: If parameters are invalid
            ImportError: If ASE is not available
        """
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE (Atomic Simulation Environment) is required for metal surface generation. "
                "Install with: pip install ase"
            )
        
        self.parameters = parameters
        validate_parameters(self.parameters)
        
        # Setup logger
        self.logger: Optional["MLIPLogger"] = None
        if self.parameters.log:
            if self.parameters.logger is not None:
                self.logger = self.parameters.logger
            else:
                try:
                    from ...utils.logger import MLIPLogger
                    self.logger = MLIPLogger()
                except ImportError:
                    self.logger = None
        
        # Get lattice constant
        if self.parameters.lattice_constant is not None:
            self.lattice_constant = self.parameters.lattice_constant
        elif self.parameters.metal in self.LATTICE_CONSTANTS:
            self.lattice_constant = self.LATTICE_CONSTANTS[self.parameters.metal]
        else:
            # Fallback to ASE data
            try:
                bulk_metal = bulk(self.parameters.metal, crystalstructure='fcc', a=4.0)
                self.lattice_constant = bulk_metal.cell.cellpar()[0]
            except:
                self.lattice_constant = 4.0
        
        if self.logger:
            self.logger.info("Initializing MetalWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Miller index: {self.parameters.miller_index}")
            self.logger.info(f"Metal-water gap: {self.parameters.metal_water_gap} Å")
            self.logger.info(f"Water model: {self.parameters.water_model}")
    
    def generate(self) -> str:
        """
        Generate metal-water interface structure.
        
        Returns:
            Path to generated structure file
            
        Raises:
            RuntimeError: If generation fails
        """
        if self.logger:
            self.logger.info("Starting metal-water interface generation")
        
        try:
            # Step 1: Generate metal surface
            if self.logger:
                self.logger.step("Creating metal surface")
            metal_atoms = self._create_metal_surface()
            
            # Step 2: Calculate water region and molecule count
            if self.logger:
                self.logger.step("Calculating water region parameters")
            water_region, n_water_molecules = self._calculate_water_region(metal_atoms)
            
            # Step 3: Generate water layer using Packmol
            if self.logger:
                self.logger.step("Generating water layer with Packmol")
            water_atoms = self._generate_water_layer(water_region, n_water_molecules)
            
            # Step 4: Add hydroxyl groups if requested
            if self.parameters.add_surface_hydroxyl:
                if self.logger:
                    self.logger.step("Adding surface hydroxyl groups")
                self._add_surface_hydroxyl_groups(metal_atoms)
            
            # Step 5: Combine metal and water
            if self.logger:
                self.logger.step("Combining metal and water structures")
            combined_atoms = self._combine_structures(metal_atoms, water_atoms)
            
            # Step 6: Apply constraints and final adjustments
            if self.parameters.fix_bottom_layers > 0:
                if self.logger:
                    self.logger.step(f"Fixing bottom {self.parameters.fix_bottom_layers} metal layers")
                self._add_constraints(combined_atoms, len(metal_atoms))
            
            # Step 7: Center system if requested
            if self.parameters.center_system:
                if self.logger:
                    self.logger.step("Centering system in unit cell")
                combined_atoms.center()
            
            # Step 8: Write output file
            output_path = Path(self.parameters.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_format = self._determine_output_format(output_path)
            
            if self.logger:
                self.logger.step(f"Writing structure to {output_format.upper()} format")
            
            self._write_structure(combined_atoms, output_path, output_format)
            
            if self.logger:
                self.logger.success("Metal-water interface generated successfully")
                self.logger.info(f"Output file: {output_path}")
                self.logger.info(f"Total atoms: {len(combined_atoms)}")
                self.logger.info(f"Metal atoms: {len(metal_atoms)}")
                self.logger.info(f"Water molecules: {n_water_molecules}")
                self.logger.info(f"Cell dimensions: {combined_atoms.cell.cellpar()[:3]}")
            
            return str(output_path)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to generate metal-water interface: {str(e)}")
            raise RuntimeError(f"Metal-water interface generation failed: {str(e)}")
    
    def _create_metal_surface(self) -> Atoms:
        """Create the metal surface using ASE."""
        miller = self.parameters.miller_index
        size = self.parameters.metal_size
        layers = self.parameters.n_metal_layers
        
        # Use specific builder if available
        if miller in self.SURFACE_BUILDERS:
            builder_func = self.SURFACE_BUILDERS[miller]
            metal_atoms = builder_func(
                symbol=self.parameters.metal,
                size=(*size, layers),
                a=self.lattice_constant,
                vacuum=0.0
            )
        else:
            # Use general surface builder
            bulk_atoms = bulk(
                self.parameters.metal,
                crystalstructure='fcc',
                a=self.lattice_constant
            )
            metal_atoms = surface(bulk_atoms, miller, layers, vacuum=0.0)
            if size != (1, 1):
                metal_atoms = metal_atoms.repeat((*size, 1))
        
        if self.logger:
            self.logger.info(f"Created {miller} metal surface with {len(metal_atoms)} atoms")
        
        return metal_atoms
    
    def _calculate_water_region(self, metal_atoms: Atoms) -> Tuple[Dict[str, float], int]:
        """Calculate water region bounds and number of molecules."""
        metal_positions = metal_atoms.get_positions()
        metal_cell = metal_atoms.get_cell()
        
        # Find top of metal surface
        z_max_metal = np.max(metal_positions[:, 2])
        
        # Calculate water region bounds
        z_min_water = z_max_metal + self.parameters.metal_water_gap
        
        if self.parameters.water_thickness is not None:
            z_max_water = z_min_water + self.parameters.water_thickness
        else:
            # Calculate from number of molecules if thickness not specified
            z_max_water = z_min_water + 20.0  # Default thickness
        
        # Get lateral dimensions from metal cell
        x_size = metal_cell[0, 0]
        y_size = metal_cell[1, 1]
        
        # Apply surface coverage
        if self.parameters.surface_coverage < 1.0:
            # Center the water region if partial coverage
            coverage_x = x_size * np.sqrt(self.parameters.surface_coverage)
            coverage_y = y_size * np.sqrt(self.parameters.surface_coverage)
            x_center = x_size / 2
            y_center = y_size / 2
            
            water_region = {
                'x_min': x_center - coverage_x/2,
                'x_max': x_center + coverage_x/2,
                'y_min': y_center - coverage_y/2,
                'y_max': y_center + coverage_y/2,
                'z_min': z_min_water,
                'z_max': z_max_water
            }
        else:
            water_region = {
                'x_min': 0.0,
                'x_max': x_size,
                'y_min': 0.0,
                'y_max': y_size,
                'z_min': z_min_water,
                'z_max': z_max_water
            }
        
        # Calculate number of water molecules
        if self.parameters.n_water_molecules is not None:
            n_water_molecules = self.parameters.n_water_molecules
        else:
            # Calculate from density and volume
            water_volume = ((water_region['x_max'] - water_region['x_min']) * 
                          (water_region['y_max'] - water_region['y_min']) * 
                          (water_region['z_max'] - water_region['z_min']))
            
            # Convert to cm³
            water_volume_cm3 = water_volume * 1e-24
            
            # Get water density
            if self.parameters.water_density is not None:
                density = self.parameters.water_density
            else:
                density = get_water_density(self.parameters.water_model)
            
            # Calculate number of molecules
            water_molar_mass = 18.015  # g/mol
            na = 6.022e23
            mass_g = density * water_volume_cm3
            moles = mass_g / water_molar_mass
            n_water_molecules = max(10, int(moles * na))
        
        if self.logger:
            self.logger.info(f"Water region: {water_region['x_max']:.1f} x {water_region['y_max']:.1f} x {water_region['z_max']-water_region['z_min']:.1f} Å")
            self.logger.info(f"Target water molecules: {n_water_molecules}")
        
        return water_region, n_water_molecules
    
    def _generate_water_layer(self, water_region: Dict[str, float], n_molecules: int) -> Atoms:
        """Generate water layer using Packmol."""
        # Create temporary directory for Packmol files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create water molecule template
            water_xyz = temp_path / "water.xyz"
            create_water_xyz(self.parameters.water_model, str(water_xyz))
            
            # Create Packmol input file
            input_file = temp_path / "packmol.inp"
            output_file = temp_path / "water_output.xyz"
            
            self._create_packmol_input(
                input_file, water_xyz, output_file,
                water_region, n_molecules
            )
            
            # Run Packmol
            self._run_packmol(input_file, temp_path)
            
            # Read generated water structure
            if output_file.exists():
                water_atoms = read(str(output_file))
                if self.logger:
                    self.logger.info(f"Generated {len(water_atoms)} water atoms ({len(water_atoms)//3} molecules)")
                return water_atoms
            else:
                raise RuntimeError("Packmol failed to generate water structure")
    
    def _create_packmol_input(self, input_file: Path, water_xyz: Path, 
                             output_file: Path, water_region: Dict[str, float], 
                             n_molecules: int) -> None:
        """Create Packmol input file for water layer."""
        with open(input_file, 'w') as f:
            f.write(f"tolerance {self.parameters.packmol_tolerance}\n")
            f.write("filetype xyz\n")
            f.write(f"output {output_file.name}\n")
            f.write(f"seed {self.parameters.packmol_seed}\n")
            f.write("\n")
            
            f.write(f"structure {water_xyz.name}\n")
            f.write(f"  number {n_molecules}\n")
            f.write(f"  inside box {water_region['x_min']:.6f} {water_region['y_min']:.6f} {water_region['z_min']:.6f} ")
            f.write(f"{water_region['x_max']:.6f} {water_region['y_max']:.6f} {water_region['z_max']:.6f}\n")
            f.write("end structure\n")
    
    def _run_packmol(self, input_file: Path, work_dir: Path) -> None:
        """Run Packmol with the given input file."""
        try:
            if self.logger:
                self.logger.debug(f"Executing: {self.parameters.packmol_executable}")
            
            result = subprocess.run(
                [self.parameters.packmol_executable],
                stdin=open(input_file, 'r'),
                capture_output=True,
                text=True,
                check=True,
                cwd=str(work_dir)
            )
            
            if self.logger:
                self.logger.success("Packmol execution completed successfully")
                
        except subprocess.CalledProcessError as e:
            if self.logger:
                self.logger.error(f"Packmol failed with return code {e.returncode}")
            raise RuntimeError(f"Packmol failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Packmol executable '{self.parameters.packmol_executable}' not found. "
                "Please install packmol or provide correct path."
            )
    
    def _add_surface_hydroxyl_groups(self, metal_atoms: Atoms) -> None:
        """Add hydroxyl groups to metal surface atoms."""
        metal_positions = metal_atoms.get_positions()
        z_coords = metal_positions[:, 2]
        
        # Find surface atoms (top layer)
        z_max = np.max(z_coords)
        surface_mask = z_coords > z_max - 1.0
        surface_indices = np.where(surface_mask)[0]
        
        # Select atoms for hydroxylation based on coverage
        n_hydroxyl = max(1, int(len(surface_indices) * self.parameters.hydroxyl_coverage))
        selected_indices = np.random.choice(surface_indices, n_hydroxyl, replace=False)
        
        # Add OH groups
        oh_height = 2.0  # Typical M-OH bond length
        for idx in selected_indices:
            metal_pos = metal_positions[idx]
            
            # Add oxygen
            o_pos = metal_pos.copy()
            o_pos[2] += oh_height
            o_atom = Atoms('O', positions=[o_pos])
            metal_atoms.extend(o_atom)
            
            # Add hydrogen
            h_pos = o_pos.copy()
            h_pos[2] += 1.0  # O-H bond length
            h_pos[0] += 0.5  # Slight offset
            h_atom = Atoms('H', positions=[h_pos])
            metal_atoms.extend(h_atom)
        
        if self.logger:
            self.logger.info(f"Added {n_hydroxyl} OH groups to surface")
    
    def _combine_structures(self, metal_atoms: Atoms, water_atoms: Atoms) -> Atoms:
        """Combine metal surface and water layer."""
        # Calculate total system size
        metal_cell = metal_atoms.get_cell()
        metal_positions = metal_atoms.get_positions()
        water_positions = water_atoms.get_positions()
        
        # Find bounds
        z_min = np.min(metal_positions[:, 2])
        z_max_water = np.max(water_positions[:, 2])
        total_z = z_max_water - z_min + self.parameters.vacuum_above_water
        
        # Create combined structure
        combined_atoms = metal_atoms.copy()
        combined_atoms.extend(water_atoms)
        
        # Set new cell dimensions
        new_cell = metal_cell.copy()
        new_cell[2, 2] = total_z
        combined_atoms.set_cell(new_cell, scale_atoms=False)
        
        return combined_atoms
    
    def _add_constraints(self, atoms: Atoms, n_metal_atoms: int) -> None:
        """Add constraints to fix bottom metal layers."""
        metal_positions = atoms.get_positions()[:n_metal_atoms]
        z_coords = metal_positions[:, 2]
        
        # Sort z-coordinates to find bottom layers
        unique_z = np.unique(np.round(z_coords, 3))
        unique_z.sort()
        
        if len(unique_z) < self.parameters.fix_bottom_layers:
            if self.logger:
                self.logger.warning(f"Cannot fix {self.parameters.fix_bottom_layers} layers, only {len(unique_z)} available")
            return
        
        # Find atoms in bottom layers to fix
        fixed_indices = []
        for i in range(self.parameters.fix_bottom_layers):
            layer_z = unique_z[i]
            layer_mask = np.abs(z_coords - layer_z) < 0.1
            fixed_indices.extend(np.where(layer_mask)[0])
        
        if fixed_indices:
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            
            if self.logger:
                self.logger.info(f"Fixed {len(fixed_indices)} metal atoms in bottom layers")
    
    def _determine_output_format(self, output_path: Path) -> str:
        """Determine output format from file extension or parameter."""
        if self.parameters.output_format:
            return self.parameters.output_format.lower()
        
        suffix = output_path.suffix.lower()
        format_map = {
            ".xyz": "xyz",
            ".vasp": "vasp",
            ".lammps": "lammps-data",
            ".data": "lammps-data"
        }
        
        return format_map.get(suffix, "xyz")
    
    def _write_structure(self, atoms: Atoms, output_path: Path, output_format: str) -> None:
        """Write structure to file in specified format."""
        if output_format == "lammps-data":
            self._write_lammps_data(atoms, output_path)
        else:
            write(str(output_path), atoms, format=output_format)
    
    def _write_lammps_data(self, atoms: Atoms, output_path: Path) -> None:
        """Write LAMMPS data file for metal-water system."""
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        # Get unique elements and assign types
        unique_elements = list(dict.fromkeys(symbols))  # Preserve order
        element_to_type = {elem: i+1 for i, elem in enumerate(unique_elements)}
        
        # Get masses
        from ase.data import atomic_masses
        masses = {elem: atomic_masses[atomic_numbers[elem]] for elem in unique_elements}
        
        # Count molecules and bonds (for water)
        water_model = get_water_model(self.parameters.water_model)
        atoms_per_water = len(water_model["atoms"])
        n_water_molecules = symbols.count('O')  # Assuming O only from water
        n_bonds = n_water_molecules * 2  # 2 O-H bonds per water
        n_angles = n_water_molecules  # 1 H-O-H angle per water
        
        with open(output_path, 'w') as f:
            f.write(f"# LAMMPS data file for {self.parameters.metal}-water interface\n")
            f.write(f"# Generated by mlip-struct-gen\n")
            f.write(f"# Miller index: {self.parameters.miller_index}\n")
            f.write(f"# Metal-water gap: {self.parameters.metal_water_gap} Å\n\n")
            
            # Counts
            f.write(f"{len(atoms)} atoms\n")
            if n_bonds > 0:
                f.write(f"{n_bonds} bonds\n")
            if n_angles > 0:
                f.write(f"{n_angles} angles\n")
            f.write(f"{len(unique_elements)} atom types\n")
            if n_bonds > 0:
                f.write("1 bond types\n")
            if n_angles > 0:
                f.write("1 angle types\n")
            f.write("\n")
            
            # Box bounds
            cell = atoms.cell
            f.write(f"0.0 {cell[0, 0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1, 1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2, 2]:.6f} zlo zhi\n\n")
            
            # Masses
            f.write("Masses\n\n")
            for elem, type_id in element_to_type.items():
                f.write(f"{type_id} {masses[elem]:.4f}  # {elem}\n")
            f.write("\n")
            
            # Atoms
            f.write("Atoms # atomic\n\n")
            for i, (symbol, pos) in enumerate(zip(symbols, positions)):
                type_id = element_to_type[symbol]
                f.write(f"{i+1:6d} {type_id:6d} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        
        if self.logger:
            self.logger.info(f"Written LAMMPS data file with {len(unique_elements)} atom types")