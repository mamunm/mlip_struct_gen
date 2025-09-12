"""Metal surface generation using ASE."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

try:
    from ase import Atoms
    from ase.build import fcc111, fcc100, fcc110, surface, bulk
    from ase.io import write
    from ase.data import atomic_numbers, covalent_radii
    from ase.build import add_adsorbate
    from ase.constraints import FixAtoms
except ImportError:
    raise ImportError(
        "ASE (Atomic Simulation Environment) is required for metal surface generation. "
        "Install with: pip install ase"
    )

from .input_parameters import MetalSurfaceParameters
from .validation import validate_parameters


class MetalSurfaceGenerator:
    """Generate metal surfaces using ASE."""
    
    # Experimental lattice constants for common FCC metals (Angstroms)
    LATTICE_CONSTANTS = {
        "Al": 4.050,
        "Au": 4.078,
        "Ag": 4.085,
        "Cu": 3.615,
        "Ni": 3.524,
        "Pd": 3.890,
        "Pt": 3.924,
        "Pb": 4.950,
        "Rh": 3.803,
        "Ir": 3.839,
        "Ca": 5.588,
        "Sr": 6.085,
        "Yb": 5.485
    }
    
    # Common Miller index to ASE function mapping
    SURFACE_BUILDERS = {
        (1, 1, 1): fcc111,
        (1, 0, 0): fcc100,
        (1, 1, 0): fcc110
    }
    
    def __init__(self, parameters: MetalSurfaceParameters):
        """
        Initialize metal surface generator.
        
        Args:
            parameters: Surface generation parameters
            
        Raises:
            ValueError: If parameters are invalid
            ImportError: If ASE is not available
        """
        self.parameters = parameters
        validate_parameters(self.parameters)
        
        # Setup logger
        self.logger: Optional["MLIPLogger"] = None
        if self.parameters.log:
            if self.parameters.logger is not None:
                self.logger = self.parameters.logger
            else:
                # Import here to avoid circular imports
                try:
                    from ...utils.logger import MLIPLogger
                    self.logger = MLIPLogger()
                except ImportError:
                    # Gracefully handle missing logger
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
                self.lattice_constant = 4.0  # Default fallback
        
        if self.logger:
            self.logger.info("Initializing MetalSurfaceGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Miller index: {self.parameters.miller_index}")
            self.logger.info(f"Size: {self.parameters.size}")
            self.logger.info(f"Layers: {self.parameters.n_layers}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.3f} Å")
    
    def generate(self) -> str:
        """
        Generate metal surface structure.
        
        Returns:
            Path to generated structure file
            
        Raises:
            RuntimeError: If surface generation fails
        """
        if self.logger:
            self.logger.info("Starting metal surface generation")
        
        try:
            # Generate the surface
            surface_atoms = self._create_surface()
            
            # Add vacuum
            if self.logger:
                self.logger.step("Adding vacuum region")
            surface_atoms.center(vacuum=self.parameters.vacuum/2, axis=2)
            
            # Center slab if requested
            if self.parameters.center_slab:
                if self.logger:
                    self.logger.step("Centering slab in unit cell")
                surface_atoms.center(axis=2)
            
            # Add constraints for fixed bottom layers
            if self.parameters.fix_bottom_layers > 0:
                if self.logger:
                    self.logger.step(f"Fixing bottom {self.parameters.fix_bottom_layers} layers")
                self._add_constraints(surface_atoms)
            
            # Add adsorbate if specified
            if self.parameters.add_adsorbate:
                if self.logger:
                    self.logger.step("Adding adsorbate")
                self._add_adsorbate(surface_atoms)
            
            # Apply supercell if specified
            if self.parameters.supercell:
                if self.logger:
                    self.logger.step(f"Creating supercell: {self.parameters.supercell}")
                surface_atoms = surface_atoms.repeat(self.parameters.supercell)
            
            # Write output file
            output_path = Path(self.parameters.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine output format
            output_format = self._determine_output_format(output_path)
            
            if self.logger:
                self.logger.step(f"Writing structure to {output_format.upper()} format")
            
            # Write file with appropriate format
            self._write_structure(surface_atoms, output_path, output_format)
            
            # Return the actual output path (which may have been modified with extension)
            actual_output_path = output_path
            if output_format == "lammps" and output_path.suffix != ".data":
                actual_output_path = output_path.with_suffix(".data")
            
            if self.logger:
                self.logger.success(f"Metal surface generated successfully")
                self.logger.info(f"Output file: {actual_output_path}")
                self.logger.info(f"Total atoms: {len(surface_atoms)}")
                self.logger.info(f"Cell dimensions: {surface_atoms.cell.cellpar()[:3]}")
            
            return str(actual_output_path)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to generate metal surface: {str(e)}")
            raise RuntimeError(f"Metal surface generation failed: {str(e)}")
    
    def _create_surface(self) -> Atoms:
        """Create the metal surface using ASE."""
        if self.logger:
            self.logger.step("Creating metal surface")
            if self.parameters.orthogonalize:
                self.logger.info("Using orthogonal cell")
        
        miller = self.parameters.miller_index
        size = self.parameters.size
        layers = self.parameters.n_layers
        
        # Use specific builder if available
        if miller in self.SURFACE_BUILDERS:
            builder_func = self.SURFACE_BUILDERS[miller]
            surface_atoms = builder_func(
                symbol=self.parameters.metal,
                size=(*size, layers),
                a=self.lattice_constant,
                vacuum=0.0,  # We'll add vacuum later
                orthogonal=self.parameters.orthogonalize
            )
        else:
            # Use general surface builder for arbitrary Miller indices
            if self.logger:
                self.logger.debug(f"Using general surface builder for {miller}")
            
            # Create bulk structure
            bulk_atoms = bulk(
                self.parameters.metal,
                crystalstructure='fcc',
                a=self.lattice_constant
            )
            
            # Create surface
            surface_atoms = surface(
                bulk_atoms,
                miller,
                layers,
                vacuum=0.0,
                orthogonal=self.parameters.orthogonalize
            )
            
            # Repeat to get desired size
            if size != (1, 1):
                surface_atoms = surface_atoms.repeat((*size, 1))
        
        if self.logger:
            self.logger.info(f"Created {miller} surface with {len(surface_atoms)} atoms")
        
        return surface_atoms
    
    def _add_constraints(self, atoms: Atoms) -> None:
        """Add constraints to fix bottom layers."""
        positions = atoms.get_positions()
        z_coords = positions[:, 2]
        
        # Sort z-coordinates to find bottom layers
        unique_z = np.unique(np.round(z_coords, 3))
        unique_z.sort()
        
        if len(unique_z) < self.parameters.fix_bottom_layers:
            if self.logger:
                self.logger.warning(f"Cannot fix {self.parameters.fix_bottom_layers} layers, only {len(unique_z)} layers available")
            return
        
        # Find atoms in bottom layers to fix
        fixed_indices = []
        for i in range(self.parameters.fix_bottom_layers):
            layer_z = unique_z[i]
            layer_mask = np.abs(z_coords - layer_z) < 0.1
            fixed_indices.extend(np.where(layer_mask)[0])
        
        # Add constraint
        if fixed_indices:
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            
            if self.logger:
                self.logger.info(f"Fixed {len(fixed_indices)} atoms in bottom {self.parameters.fix_bottom_layers} layers")
    
    def _add_adsorbate(self, atoms: Atoms) -> None:
        """Add adsorbate to the surface."""
        adsorbate_params = self.parameters.add_adsorbate
        
        element = adsorbate_params["element"]
        position = adsorbate_params.get("position", "top")
        height = adsorbate_params.get("height", 2.0)
        coverage = adsorbate_params.get("coverage", 0.25)
        
        if self.logger:
            self.logger.info(f"Adding {element} adsorbate at {position} sites")
            self.logger.info(f"Height: {height} Å, Coverage: {coverage}")
        
        # For simple implementation, add adsorbate at top site
        # More sophisticated site finding could be implemented for other positions
        positions = atoms.get_positions()
        z_max = np.max(positions[:, 2])
        
        # Find surface atoms (top layer)
        surface_mask = positions[:, 2] > z_max - 1.0
        surface_indices = np.where(surface_mask)[0]
        
        # Determine number of adsorbates based on coverage
        n_adsorbates = max(1, int(len(surface_indices) * coverage))
        
        # Select adsorption sites
        if n_adsorbates >= len(surface_indices):
            selected_sites = surface_indices
        else:
            # Distribute evenly
            step = len(surface_indices) // n_adsorbates
            selected_sites = surface_indices[::step][:n_adsorbates]
        
        # Add adsorbates
        for site_idx in selected_sites:
            site_pos = positions[site_idx]
            adsorbate_pos = site_pos.copy()
            adsorbate_pos[2] += height
            
            # Create adsorbate atom
            adsorbate = Atoms(element, positions=[adsorbate_pos])
            atoms.extend(adsorbate)
        
        if self.logger:
            self.logger.info(f"Added {len(selected_sites)} {element} adsorbates")
    
    def _determine_output_format(self, output_path: Path) -> str:
        """Determine output format from file extension or parameter."""
        if self.parameters.output_format:
            return self.parameters.output_format.lower()
        
        suffix = output_path.suffix.lower()
        
        format_map = {
            ".xyz": "xyz",
            ".cif": "cif", 
            ".vasp": "vasp",
            ".lammps": "lammps",
            ".data": "lammps"
        }
        
        return format_map.get(suffix, "xyz")
    
    def _write_structure(self, atoms: Atoms, output_path: Path, output_format: str) -> None:
        """Write structure to file in specified format."""
        if output_format == "lammps":
            # Custom LAMMPS data file writing
            # Add .data extension if not present
            if output_path.suffix != ".data":
                output_path = output_path.with_suffix(".data")
            self._write_lammps_data(atoms, output_path)
        else:
            # Use ASE's write function
            write(str(output_path), atoms, format=output_format)
    
    def _write_lammps_data(self, atoms: Atoms, output_path: Path) -> None:
        """Write LAMMPS data file for metal surface."""
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        # Get unique elements and assign types
        unique_elements = list(set(symbols))
        element_to_type = {elem: i+1 for i, elem in enumerate(unique_elements)}
        
        # Get masses from ASE
        from ase.data import atomic_masses
        masses = {elem: atomic_masses[atomic_numbers[elem]] for elem in unique_elements}
        
        with open(output_path, 'w') as f:
            f.write(f"# LAMMPS data file for {self.parameters.metal} surface\n")
            f.write(f"# Generated by mlip-struct-gen\n")
            f.write(f"# Miller index: {self.parameters.miller_index}\n\n")
            
            # Counts
            f.write(f"{len(atoms)} atoms\n")
            f.write(f"{len(unique_elements)} atom types\n\n")
            
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