"""Salt water box generation using Packmol."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .input_parameters import SaltWaterBoxGeneratorParameters
from .validation import validate_parameters
from ..templates.water_models import (
    create_water_xyz,
    calculate_water_molecules,
    get_water_density,
    get_water_model
)
from ..templates.salt_models import (
    get_salt_model,
    calculate_ion_numbers,
    create_ion_xyz
)


class SaltWaterBoxGenerator:
    """Generate salt water boxes using Packmol."""
    
    def __init__(self, parameters: SaltWaterBoxGeneratorParameters):
        """
        Initialize salt water box generator.
        
        Args:
            parameters: Generation parameters
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If packmol is not available
        """
        self.parameters = parameters
        validate_parameters(self.parameters)
        
        # Setup logger first
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
        
        # Get salt model
        if self.parameters.custom_salt_params:
            self.salt_model = self.parameters.custom_salt_params
        else:
            self.salt_model = get_salt_model(self.parameters.salt_type)
        
        # Compute box_size if not provided (after logger setup)
        if self.parameters.box_size is None:
            self._compute_box_size_from_molecules()
        
        if self.logger:
            self.logger.info("Initializing SaltWaterBoxGenerator")
            self.logger.info(f"Salt: {self.salt_model.get('name', self.parameters.salt_type)}")
            self.logger.info(f"Salt molecules: {self.parameters.n_salt_molecules}")
            self.logger.info(f"Water model: {self.parameters.water_model}")
            self.logger.info(f"Box size: {self.parameters.box_size}")
        
        self._check_packmol()
    
    def _check_packmol(self) -> None:
        """Check if packmol is available."""
        if self.logger:
            self.logger.step("Checking Packmol availability")
        
        try:
            # Packmol doesn't have --version, so we test with empty input and expect error 171
            result = subprocess.run(
                [self.parameters.packmol_executable], 
                input="",  # Empty input
                capture_output=True, 
                timeout=5,  # 5 second timeout
                text=True
            )
            # Packmol exits with code 171 when given empty input - this is expected
            if result.returncode == 171:
                if self.logger:
                    self.logger.success(f"Packmol found: {self.parameters.packmol_executable}")
                    # Extract version from stderr if available
                    if "Version" in result.stderr:
                        version_line = [line for line in result.stderr.split('\n') if "Version" in line]
                        if version_line:
                            self.logger.debug(f"Packmol {version_line[0].strip()}")
            else:
                if self.logger:
                    self.logger.warning(f"Packmol returned unexpected exit code: {result.returncode}")
                    self.logger.success(f"But Packmol executable appears to be working: {self.parameters.packmol_executable}")
                
        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.error(f"Packmol check timed out after 5 seconds")
            raise RuntimeError(
                f"Packmol executable '{self.parameters.packmol_executable}' timed out. "
                "This may indicate an installation issue."
            )
        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"Packmol executable '{self.parameters.packmol_executable}' not found in PATH")
            raise RuntimeError(
                f"Packmol executable '{self.parameters.packmol_executable}' not found. "
                "Please install packmol:\n"
                "  conda install -c conda-forge packmol\n"
                "Or compile from source: http://leandro.iqm.unicamp.br/m3g/packmol/"
            )
    
    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate a salt water box using Packmol.
        
        Args:
            save_artifacts: If True, save intermediate files (packmol.inp, water.xyz, ions)
            
        Returns:
            Path to generated salt water box file
            
        Raises:
            RuntimeError: If packmol execution fails
        """
        if self.logger:
            self.logger.info("Starting salt water box generation")
        
        # Calculate ion numbers based on n_salt_molecules and stoichiometry
        # Get stoichiometry
        if "stoichiometry" in self.salt_model:
            stoich = self.salt_model["stoichiometry"]
            n_cations = self.parameters.n_salt_molecules * stoich["cation"]
            n_anions = self.parameters.n_salt_molecules * stoich["anion"]
        else:
            # Default 1:1 stoichiometry
            n_cations = self.parameters.n_salt_molecules
            n_anions = self.parameters.n_salt_molecules
        
        # Neutralize if requested
        if self.parameters.neutralize:
            n_cations, n_anions = self._neutralize_system(n_cations, n_anions)
        
        if self.logger:
            self.logger.info(f"Target ions: {n_cations} cations, {n_anions} anions")
        
        # Calculate water molecules
        n_water_molecules = self._calculate_water_molecules(n_cations, n_anions)
        
        if self.logger:
            self.logger.info(f"Target water molecules: {n_water_molecules}")
        
        # Create output directory
        output_path = Path(self.parameters.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Output file: {output_path}")
        
        # Determine working directory
        if save_artifacts:
            # Save artifacts in current directory
            work_dir = Path("artifacts")
            work_dir.mkdir(exist_ok=True)
            temp_context = None
            if self.logger:
                self.logger.info(f"Saving artifacts to: {work_dir}")
        else:
            # Use temporary directory
            temp_context = tempfile.TemporaryDirectory()
            work_dir = Path(temp_context.name)
            if self.logger:
                self.logger.debug("Using temporary directory for intermediate files")
        
        try:
            # Create molecule templates
            if self.logger:
                self.logger.step("Creating molecule templates")
            
            # Water template
            water_xyz = work_dir / "water.xyz"
            create_water_xyz(self.parameters.water_model, str(water_xyz))
            
            # Ion templates
            cation_xyz = work_dir / "cation.xyz"
            anion_xyz = work_dir / "anion.xyz"
            
            create_ion_xyz(
                self.salt_model["cation"],
                [0.0, 0.0, 0.0],
                str(cation_xyz)
            )
            create_ion_xyz(
                self.salt_model["anion"],
                [0.0, 0.0, 0.0],
                str(anion_xyz)
            )
            
            # Create packmol input file
            if self.logger:
                self.logger.step("Generating Packmol input file")
            input_file = work_dir / "packmol.inp"
            temp_output = work_dir / "output.xyz"
            
            self._create_packmol_input(
                input_file, water_xyz, cation_xyz, anion_xyz, temp_output,
                self.parameters.box_size, n_water_molecules, n_cations, n_anions,
                self.parameters.tolerance, self.parameters.seed
            )
            
            # Run packmol
            if self.logger:
                self.logger.step("Running Packmol to generate XYZ format")
            self._run_packmol(input_file, work_dir)
            
            # Handle output file based on format
            if self.parameters.output_format == "lammps":
                # Convert XYZ to LAMMPS format with bonds and angles
                if self.logger:
                    self.logger.step("Converting XYZ to LAMMPS data format with topology")
                self._convert_xyz_to_lammps_with_topology(
                    temp_output, output_path, n_water_molecules, n_cations, n_anions
                )
            elif self.parameters.output_format == "poscar":
                # Convert XYZ to POSCAR format with sorted elements
                if self.logger:
                    self.logger.step("Converting XYZ to POSCAR format")
                self._convert_xyz_to_poscar(
                    temp_output, output_path, n_water_molecules, n_cations, n_anions
                )
            else:
                # Copy the XYZ file directly
                if self.logger:
                    self.logger.step("Copying XYZ output to final location")
                shutil.copy2(temp_output, output_path)
            
        finally:
            if temp_context is not None:
                temp_context.cleanup()
        
        if self.logger:
            self.logger.success(f"Salt water box generation completed successfully")
            self.logger.info(f"Output saved to: {output_path}")
            self.logger.info(f"System composition: {n_water_molecules} water, {n_cations} cations, {n_anions} anions")
        
        return str(output_path)
    
    def _neutralize_system(self, n_cations: int, n_anions: int) -> Tuple[int, int]:
        """
        Adjust ion counts to ensure electrical neutrality.
        
        Args:
            n_cations: Initial number of cations
            n_anions: Initial number of anions
            
        Returns:
            Tuple of (adjusted_n_cations, adjusted_n_anions)
        """
        cation_charge = self.salt_model["cation"]["charge"]
        anion_charge = self.salt_model["anion"]["charge"]
        
        total_positive = n_cations * cation_charge
        total_negative = n_anions * abs(anion_charge)
        
        if total_positive > total_negative:
            # Need more anions
            n_anions = int(np.ceil(total_positive / abs(anion_charge)))
        elif total_negative > total_positive:
            # Need more cations
            n_cations = int(np.ceil(total_negative / cation_charge))
        
        # Verify neutrality
        final_charge = n_cations * cation_charge + n_anions * anion_charge
        if abs(final_charge) > 0.01:
            if self.logger:
                self.logger.warning(f"System not perfectly neutral. Residual charge: {final_charge}")
        
        return n_cations, n_anions
    
    def _calculate_water_molecules(self, n_cations: int, n_anions: int) -> int:
        """Calculate number of water molecules accounting for ion volume."""
        if self.parameters.n_water_molecules is not None:
            return self.parameters.n_water_molecules
        
        # Calculate box volume
        box_volume = np.prod(self.parameters.box_size)
        
        # Only subtract ion volume if add_salt_volume is True
        if self.parameters.add_salt_volume:
            # Estimate ion volumes (rough approximation)
            cation_radius = self.salt_model["cation"].get("vdw_radius", 2.0)
            anion_radius = self.salt_model["anion"].get("vdw_radius", 2.0)
            
            cation_volume = (4/3) * np.pi * cation_radius**3
            anion_volume = (4/3) * np.pi * anion_radius**3
            
            total_ion_volume = n_cations * cation_volume + n_anions * anion_volume
            available_volume = box_volume - total_ion_volume
        else:
            # Use entire box volume for water calculation
            available_volume = box_volume
        
        # Use specified or default water density
        if self.parameters.water_density is not None:
            density = self.parameters.water_density
        else:
            density = get_water_density(self.parameters.water_model)
        
        # Calculate water molecules
        volume_cm3 = available_volume * 1e-24
        water_molar_mass = 18.015
        na = 6.022e23
        
        mass_g = density * volume_cm3
        moles = mass_g / water_molar_mass
        n_molecules = int(moles * na)
        
        return max(1, n_molecules)  # At least 1 water molecule
    
    def _compute_box_size_from_molecules(self) -> None:
        """Compute box_size from n_water_molecules and ion requirements."""
        if self.parameters.n_water_molecules is None:
            raise ValueError("n_water_molecules must be provided when box_size is None")
        
        # Estimate required volume
        # Start with water volume
        if self.parameters.water_density is not None:
            density = self.parameters.water_density
        else:
            density = get_water_density(self.parameters.water_model)
        
        water_molar_mass = 18.015
        na = 6.022e23
        
        # Water volume
        moles = self.parameters.n_water_molecules / na
        mass_g = moles * water_molar_mass
        water_volume_cm3 = mass_g / density
        water_volume_angstrom3 = water_volume_cm3 * 1e24
        
        # Get ion counts from n_salt_molecules and stoichiometry
        if "stoichiometry" in self.salt_model:
            stoich = self.salt_model["stoichiometry"]
            n_cations = self.parameters.n_salt_molecules * stoich["cation"]
            n_anions = self.parameters.n_salt_molecules * stoich["anion"]
        else:
            # Default 1:1 stoichiometry
            n_cations = self.parameters.n_salt_molecules
            n_anions = self.parameters.n_salt_molecules
        
        # Add ion volume only if add_salt_volume is True
        if self.parameters.add_salt_volume:
            cation_radius = self.salt_model["cation"].get("vdw_radius", 2.0)
            anion_radius = self.salt_model["anion"].get("vdw_radius", 2.0)
            
            cation_volume = (4/3) * np.pi * cation_radius**3
            anion_volume = (4/3) * np.pi * anion_radius**3
            
            total_ion_volume = n_cations * cation_volume + n_anions * anion_volume
            
            # Total volume
            total_volume = water_volume_angstrom3 + total_ion_volume
        else:
            # Only use water volume
            total_volume = water_volume_angstrom3
        box_size = total_volume ** (1/3)
        
        # Add 10% buffer for packing efficiency
        box_size *= 1.1
        
        # Set as cubic box
        self.parameters.box_size = (box_size, box_size, box_size)
        
        if self.logger:
            self.logger.info(f"Computed box size: {box_size:.2f} Å (cubic)")
            self.logger.info(f"For {self.parameters.n_water_molecules} water molecules + ions")
    
    def _create_packmol_input(
        self,
        input_file: Path,
        water_xyz: Path,
        cation_xyz: Path,
        anion_xyz: Path,
        output_file: Path,
        box_size: Tuple[float, float, float],
        n_water: int,
        n_cations: int,
        n_anions: int,
        tolerance: float,
        seed: int
    ) -> None:
        """Create Packmol input file for salt water system."""
        # Calculate box boundaries with tolerance buffer
        x_low = 0.5 * tolerance
        y_low = 0.5 * tolerance
        z_low = 0.5 * tolerance
        x_high = box_size[0] - 0.5 * tolerance
        y_high = box_size[1] - 0.5 * tolerance
        z_high = box_size[2] - 0.5 * tolerance
        
        with open(input_file, 'w') as f:
            f.write(f"tolerance {tolerance}\n")
            f.write(f"filetype xyz\n")
            f.write(f"output {output_file.name}\n")
            f.write(f"seed {seed}\n")
            f.write("\n")
            
            # Water molecules
            if n_water > 0:
                f.write(f"structure {water_xyz.name}\n")
                f.write(f"  number {n_water}\n")
                f.write(f"  inside box {x_low} {y_low} {z_low} {x_high} {y_high} {z_high}\n")
                f.write("end structure\n\n")
            
            # Cations
            if n_cations > 0:
                f.write(f"structure {cation_xyz.name}\n")
                f.write(f"  number {n_cations}\n")
                f.write(f"  inside box {x_low} {y_low} {z_low} {x_high} {y_high} {z_high}\n")
                f.write("end structure\n\n")
            
            # Anions
            if n_anions > 0:
                f.write(f"structure {anion_xyz.name}\n")
                f.write(f"  number {n_anions}\n")
                f.write(f"  inside box {x_low} {y_low} {z_low} {x_high} {y_high} {z_high}\n")
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
                self.logger.debug(f"stdout: {e.stdout}")
                self.logger.debug(f"stderr: {e.stderr}")
            
            raise RuntimeError(
                f"Packmol failed with return code {e.returncode}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
    
    def _convert_xyz_to_lammps_with_topology(
        self, 
        input_xyz: Path, 
        output_file: Path,
        n_water: int,
        n_cations: int,
        n_anions: int
    ) -> None:
        """
        Convert XYZ file to LAMMPS data format with bonds and angles for water.
        
        Args:
            input_xyz: Path to input XYZ file
            output_file: Path to output LAMMPS data file
            n_water: Number of water molecules
            n_cations: Number of cations
            n_anions: Number of anions
        """
        # Read XYZ file
        with open(input_xyz, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        
        # Parse atom positions
        atoms = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, x, y, z))
        
        # Get charges
        water_model = get_water_model(self.parameters.water_model)
        water_charges = {atom["element"]: atom["charge"] for atom in water_model["atoms"]}
        
        cation_charge = self.salt_model["cation"]["charge"]
        anion_charge = self.salt_model["anion"]["charge"]
        cation_element = self.salt_model["cation"]["element"]
        anion_element = self.salt_model["anion"]["element"]
        
        # Determine atom types
        atom_types = {}
        type_id = 1
        
        # Water atom types
        for atom in water_model["atoms"]:
            elem = atom["element"]
            if elem not in atom_types:
                atom_types[elem] = type_id
                type_id += 1
        
        # Ion atom types
        if cation_element not in atom_types:
            atom_types[cation_element] = type_id
            type_id += 1
        if anion_element not in atom_types:
            atom_types[anion_element] = type_id
            type_id += 1
        
        # Calculate bonds and angles (only for water)
        n_bonds = n_water * 2  # 2 O-H bonds per water
        n_angles = n_water     # 1 H-O-H angle per water
        
        # Write LAMMPS data file
        with open(output_file, 'w') as f:
            f.write(f"# LAMMPS data file for {self.salt_model.get('name', 'salt')} + {self.parameters.water_model} water\n")
            f.write(f"# Generated by mlip-struct-gen\n\n")
            
            # Counts section
            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{n_angles} angles\n")
            f.write("\n")
            
            # Types section
            f.write(f"{len(atom_types)} atom types\n")
            f.write("1 bond types\n") 
            f.write("1 angle types\n")
            f.write("\n")
            
            # Box dimensions
            f.write(f"0.0 {self.parameters.box_size[0]} xlo xhi\n")
            f.write(f"0.0 {self.parameters.box_size[1]} ylo yhi\n")
            f.write(f"0.0 {self.parameters.box_size[2]} zlo zhi\n")
            f.write("\n")
            
            # Masses section
            f.write("Masses\n\n")
            for element, type_id in sorted(atom_types.items(), key=lambda x: x[1]):
                if element == "O":
                    mass = 15.9994
                elif element == "H":
                    mass = 1.008
                elif element == "M":  # Virtual site for TIP4P
                    mass = 0.0
                else:
                    # Ion masses
                    if element == cation_element:
                        mass = self.salt_model["cation"]["mass"]
                    elif element == anion_element:
                        mass = self.salt_model["anion"]["mass"]
                    else:
                        mass = 1.0  # Default
                
                f.write(f"{type_id} {mass}  # {element}\n")
            f.write("\n")
            
            # Atoms section
            f.write("Atoms # full\n\n")
            
            atom_id = 1
            mol_id = 1
            
            # Write water molecules first
            water_atoms_per_molecule = len(water_model["atoms"])
            for i in range(n_water):
                for j in range(water_atoms_per_molecule):
                    element, x, y, z = atoms[i * water_atoms_per_molecule + j]
                    atom_type = atom_types[element]
                    charge = water_charges[element]
                    f.write(f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {charge:10.6f} "
                           f"{x:12.6f} {y:12.6f} {z:12.6f}\n")
                    atom_id += 1
                mol_id += 1
            
            # Write cations
            water_atoms_total = n_water * water_atoms_per_molecule
            for i in range(n_cations):
                element, x, y, z = atoms[water_atoms_total + i]
                atom_type = atom_types[cation_element]
                f.write(f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {cation_charge:10.6f} "
                       f"{x:12.6f} {y:12.6f} {z:12.6f}\n")
                atom_id += 1
                mol_id += 1
            
            # Write anions
            for i in range(n_anions):
                element, x, y, z = atoms[water_atoms_total + n_cations + i]
                atom_type = atom_types[anion_element]
                f.write(f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {anion_charge:10.6f} "
                       f"{x:12.6f} {y:12.6f} {z:12.6f}\n")
                atom_id += 1
                mol_id += 1
            
            f.write("\n")
            
            # Bonds section (only for water)
            if n_bonds > 0:
                f.write("Bonds\n\n")
                bond_id = 1
                for mol in range(n_water):
                    base_atom = mol * water_atoms_per_molecule + 1  # Oxygen atom ID
                    # O-H bonds
                    for h in range(1, water_atoms_per_molecule):
                        if atoms[(base_atom - 1) + h][0] == 'H':  # Check it's hydrogen
                            f.write(f"{bond_id:6d} 1 {base_atom:6d} {base_atom + h:6d}\n")
                            bond_id += 1
                f.write("\n")
            
            # Angles section (only for water)
            if n_angles > 0:
                f.write("Angles\n\n")
                angle_id = 1
                for mol in range(n_water):
                    base_atom = mol * water_atoms_per_molecule + 1  # Oxygen atom ID
                    # H-O-H angle
                    h_atoms = []
                    for h in range(1, water_atoms_per_molecule):
                        if atoms[(base_atom - 1) + h][0] == 'H':
                            h_atoms.append(base_atom + h)
                    
                    if len(h_atoms) >= 2:
                        f.write(f"{angle_id:6d} 1 {h_atoms[0]:6d} {base_atom:6d} {h_atoms[1]:6d}\n")
                        angle_id += 1
        
        if self.logger:
            self.logger.success(f"Successfully converted to LAMMPS data format")
            self.logger.info(f"System: {n_water} water molecules, {n_cations} cations, {n_anions} anions")
    
    def _convert_xyz_to_poscar(
        self, 
        input_xyz: Path, 
        output_file: Path,
        n_water: int,
        n_cations: int,
        n_anions: int
    ) -> None:
        """
        Convert XYZ file to POSCAR format with descending element ordering.
        
        Args:
            input_xyz: Path to input XYZ file
            output_file: Path to output POSCAR file
            n_water: Number of water molecules
            n_cations: Number of cations
            n_anions: Number of anions
        """
        try:
            from ase import io
            from ase import Atoms
        except ImportError:
            raise ImportError("ASE is required for POSCAR format. Install with: pip install ase")
        
        # Read the XYZ file
        atoms = io.read(str(input_xyz))
        
        # Set the cell dimensions and PBC
        atoms.set_cell([self.parameters.box_size[0], 
                       self.parameters.box_size[1], 
                       self.parameters.box_size[2]])
        atoms.set_pbc(True)
        
        # Get salt element symbols
        cation_element = self.salt_model["cation"]["element"]
        anion_element = self.salt_model["anion"]["element"]
        
        # Sort atoms by element in descending order for proper POSCAR grouping
        # Order will be determined alphabetically in descending order
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # Create a list of tuples (symbol, position)
        atom_data = [(s, p) for s, p in zip(symbols, positions)]
        
        # Sort in descending order - this will group elements together
        # and order them alphabetically in reverse (e.g., O, Na, H, Cl)
        atom_data.sort(key=lambda x: x[0], reverse=True)
        
        # Extract sorted symbols and positions
        sorted_symbols = [s for s, p in atom_data]
        sorted_positions = [p for s, p in atom_data]
        
        # Create new atoms object with sorted atoms
        sorted_atoms = Atoms(symbols=sorted_symbols, positions=sorted_positions,
                            cell=atoms.cell, pbc=True)
        
        # Write as POSCAR with proper formatting
        io.write(str(output_file), sorted_atoms, format='vasp', direct=False, sort=False)
        
        # Count each element type
        element_counts = {}
        for symbol in sorted_symbols:
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        
        if self.logger:
            self.logger.success(f"Successfully converted to POSCAR format")
            counts_str = ", ".join([f"{count} {elem}" for elem, count in element_counts.items()])
            self.logger.info(f"System: {counts_str}")
            self.logger.info(f"Total: {n_water} water, {n_cations} {cation_element}, {n_anions} {anion_element}")