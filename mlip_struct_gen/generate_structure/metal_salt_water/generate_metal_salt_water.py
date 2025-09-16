"""Metal-salt-water interface generation using ASE and PACKMOL."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    from ase.io import read, write
except ImportError:
    raise ImportError(
        "ASE (Atomic Simulation Environment) is required for metal-salt-water generation. "
        "Install with: pip install ase"
    )

from ...utils.water_models import WATER_MODELS
from .input_parameters import MetalSaltWaterParameters
from .validation import (
    get_ion_params,
    get_lattice_constant,
    get_salt_info,
    get_water_model_params,
    validate_parameters,
)


class MetalSaltWaterGenerator:
    """Generate FCC(111) metal surfaces with salt water layers using ASE and PACKMOL."""

    def __init__(self, parameters: MetalSaltWaterParameters):
        """
        Initialize metal-salt-water interface generator.

        Args:
            parameters: Interface generation parameters

        Raises:
            ValueError: If parameters are invalid
            ImportError: If ASE is not available
            RuntimeError: If PACKMOL is not available
        """
        self.parameters = parameters
        validate_parameters(self.parameters)

        # Setup logger
        self.logger: MLIPLogger | None = None
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
        self.lattice_constant = get_lattice_constant(
            self.parameters.metal, self.parameters.lattice_constant
        )

        # Get water model parameters
        self.water_params = get_water_model_params(self.parameters.water_model)

        # Get salt information
        self.salt_info = get_salt_info(self.parameters.salt_type)

        # Storage for intermediate structures
        self.metal_slab = None
        self.solution_atoms = None
        self.combined_system = None
        self.box_dimensions = None

        if self.logger:
            self.logger.info("Initializing MetalSaltWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Metal size: {self.parameters.size}")
            self.logger.info(
                f"Salt: {self.parameters.salt_type} ({self.parameters.n_salt} formula units)"
            )
            self.logger.info(f"Water molecules: {self.parameters.n_water}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.3f} Å")

    def generate(self) -> str:
        """
        Generate the metal-salt-water interface.

        Returns:
            Path to the output file

        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Use temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as tmpdir:
                # Build metal surface
                self._build_metal_surface()

                # Generate salt-water solution
                self._generate_salt_water_solution(tmpdir)

                # Combine metal and solution
                self._combine_metal_solution()

                # Adjust vacuum
                self._adjust_vacuum()

                # Write output file
                output_path = self._write_output()

                if self.logger:
                    self.logger.info(f"Successfully generated: {output_path}")

                return str(output_path)

        except Exception as e:
            error_msg = f"Failed to generate metal-salt-water interface: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _build_metal_surface(self) -> None:
        """Build the FCC(111) metal surface."""
        nx, ny, nz = self.parameters.size

        if self.logger:
            self.logger.info(f"Creating {self.parameters.metal}(111) surface")
            self.logger.info(f"  Dimensions: {nx}x{ny} unit cells, {nz} layers")

        # Build the surface using ASE's fcc111 function
        # orthogonal=True ensures LAMMPS compatibility
        self.metal_slab = fcc111(
            self.parameters.metal,
            size=(nx, ny, nz),
            a=self.lattice_constant,
            orthogonal=True,
            vacuum=0.0,
            periodic=True,
        )

        # Store box dimensions
        cell = self.metal_slab.get_cell()
        self.box_dimensions = {"x": cell[0, 0], "y": cell[1, 1], "z": cell[2, 2]}

        if self.logger:
            self.logger.info(f"Created surface with {len(self.metal_slab)} atoms")
            self.logger.info(
                f"Box dimensions: {self.box_dimensions['x']:.2f} x {self.box_dimensions['y']:.2f} x {self.box_dimensions['z']:.2f} Å"
            )

        # Apply constraints to fix bottom layers if requested
        if self.parameters.fix_bottom_layers > 0:
            self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply constraints to fix bottom metal layers."""
        positions = self.metal_slab.get_positions()
        z_positions = positions[:, 2]

        # Find unique z-layers
        z_unique = np.unique(np.round(z_positions, decimals=2))
        z_unique.sort()

        n_fix = min(self.parameters.fix_bottom_layers, len(z_unique) - 1)

        # Get z-threshold for fixed layers
        z_threshold = z_unique[n_fix] if n_fix < len(z_unique) else z_unique[-1]

        # Create mask for fixed atoms
        fixed_mask = z_positions < z_threshold + 0.01

        # Apply constraints
        constraint = FixAtoms(mask=fixed_mask)
        self.metal_slab.set_constraint(constraint)

        n_fixed = np.sum(fixed_mask)
        if self.logger:
            self.logger.info(f"Fixed {n_fixed} atoms in bottom {n_fix} layers")

    def _calculate_solution_height(self) -> float:
        """
        Calculate solution box height to achieve target density.

        Returns:
            Solution box height in Angstroms
        """
        # Water molecular weight: 18.015 g/mol
        water_mw = 18.015  # g/mol
        avogadro = 6.022e23  # molecules/mol

        # Calculate total mass
        total_mass_g = self.parameters.n_water * water_mw / avogadro

        # Add ion masses if include_salt_volume is True
        if self.parameters.include_salt_volume and self.parameters.n_salt > 0:
            # Get ion parameters
            cation_params = get_ion_params(self.salt_info["cation"])
            anion_params = get_ion_params(self.salt_info["anion"])

            # Calculate ion contributions
            n_cations = self.parameters.n_salt * self.salt_info["cation_count"]
            n_anions = self.parameters.n_salt * self.salt_info["anion_count"]

            cation_mass = n_cations * cation_params["mass"] / avogadro
            anion_mass = n_anions * anion_params["mass"] / avogadro

            total_mass_g += cation_mass + anion_mass

        # Convert density to g/Å³
        density_g_A3 = self.parameters.density * 1e-24  # g/cm³ to g/Å³

        # Calculate required volume
        volume_A3 = total_mass_g / density_g_A3

        # Calculate height from volume
        solution_height = volume_A3 / (self.box_dimensions["x"] * self.box_dimensions["y"])

        if self.logger:
            if self.parameters.include_salt_volume:
                self.logger.info(
                    f"Solution box height for {self.parameters.n_water} water + "
                    f"{self.parameters.n_salt} {self.parameters.salt_type} "
                    f"at {self.parameters.density} g/cm^3: {solution_height:.2f} Å"
                )
            else:
                self.logger.info(
                    f"Solution box height for {self.parameters.n_water} water molecules "
                    f"at {self.parameters.density} g/cm^3: {solution_height:.2f} Å"
                )

        return solution_height

    def _generate_salt_water_solution(self, tmpdir: str) -> None:
        """
        Generate salt-water solution using PACKMOL.

        Args:
            tmpdir: Temporary directory for intermediate files
        """
        if self.logger:
            self.logger.info("Generating salt-water solution with PACKMOL")

        # Calculate solution box dimensions
        metal_cell_z = self.metal_slab.get_cell()[2, 2]
        solution_height = self._calculate_solution_height()

        # Set margins for solution box
        margin_xy = 1.0  # margin from box edges in x and y
        margin_z_bottom = 3.0  # additional margin from metal top
        margin_z_top = 3.0  # margin from solution top to boundary

        solution_x = self.box_dimensions["x"] - 2 * margin_xy
        solution_y = self.box_dimensions["y"] - 2 * margin_xy
        solution_z_min = metal_cell_z + self.parameters.gap + margin_z_bottom
        solution_z_max = metal_cell_z + self.parameters.gap + solution_height - margin_z_top

        # Create molecule files
        water_xyz_path = os.path.join(tmpdir, "water_molecule.xyz")
        self._create_water_molecule_file(water_xyz_path)

        # Create PACKMOL input
        packmol_input_path = os.path.join(tmpdir, "pack_solution.inp")
        solution_output_path = os.path.join(tmpdir, "solution_box.xyz")

        # Build PACKMOL input with water and ions
        packmol_input = f"""
tolerance {self.parameters.packmol_tolerance}
filetype xyz
output {solution_output_path}
seed {self.parameters.seed}

# Water molecules
structure {water_xyz_path}
  number {self.parameters.n_water}
  inside box {margin_xy} {margin_xy} {solution_z_min} {solution_x + margin_xy} {solution_y + margin_xy} {solution_z_max}
end structure
"""

        # Add salt ions if present
        if self.parameters.n_salt > 0:
            # Calculate number of each ion type
            n_cations = self.parameters.n_salt * self.salt_info["cation_count"]
            n_anions = self.parameters.n_salt * self.salt_info["anion_count"]

            # Create ion files
            cation_xyz_path = os.path.join(tmpdir, f'{self.salt_info["cation"]}.xyz')
            anion_xyz_path = os.path.join(tmpdir, f'{self.salt_info["anion"]}.xyz')

            self._create_ion_file(self.salt_info["cation"], cation_xyz_path)
            self._create_ion_file(self.salt_info["anion"], anion_xyz_path)

            # Add ions to PACKMOL input
            packmol_input += f"""
# Cations
structure {cation_xyz_path}
  number {n_cations}
  inside box {margin_xy} {margin_xy} {solution_z_min} {solution_x + margin_xy} {solution_y + margin_xy} {solution_z_max}
end structure

# Anions
structure {anion_xyz_path}
  number {n_anions}
  inside box {margin_xy} {margin_xy} {solution_z_min} {solution_x + margin_xy} {solution_y + margin_xy} {solution_z_max}
end structure
"""

        with open(packmol_input_path, "w") as f:
            f.write(packmol_input)

        if self.logger:
            self.logger.info(
                f"Solution box: x=[{margin_xy:.1f}, {solution_x + margin_xy:.1f}], "
                f"y=[{margin_xy:.1f}, {solution_y + margin_xy:.1f}], "
                f"z=[{solution_z_min:.2f}, {solution_z_max:.2f}]"
            )
            if self.parameters.n_salt > 0:
                self.logger.info(
                    f"  Ions: {n_cations} {self.salt_info['cation']}+ and {n_anions} {self.salt_info['anion']}-"
                )

        # Run PACKMOL
        self._run_packmol(packmol_input_path, solution_output_path)

    def _create_water_molecule_file(self, filepath: str) -> None:
        """
        Create water molecule XYZ file based on water model.

        Args:
            filepath: Path to write the water molecule file
        """
        # Get water model geometry
        model_key = self.parameters.water_model.replace("/", "")  # Remove slash from SPC/E
        if model_key in WATER_MODELS:
            geometry = WATER_MODELS[model_key]["geometry"]
            water_xyz = f"""3
Water molecule {self.parameters.water_model}
O    {geometry['O'][0]:.4f}    {geometry['O'][1]:.4f}    {geometry['O'][2]:.4f}
H    {geometry['H1'][0]:.4f}   {geometry['H1'][1]:.4f}   {geometry['H1'][2]:.4f}
H    {geometry['H2'][0]:.4f}   {geometry['H2'][1]:.4f}   {geometry['H2'][2]:.4f}
"""
        else:
            # Fallback to SPC/E geometry
            water_xyz = """3
Water molecule SPC/E
O    0.0000    0.0000    0.0000
H    0.8164    0.0000    0.5773
H   -0.8164    0.0000    0.5773
"""

        with open(filepath, "w") as f:
            f.write(water_xyz)

    def _create_ion_file(self, ion: str, filepath: str) -> None:
        """
        Create ion XYZ file.

        Args:
            ion: Ion element symbol
            filepath: Path to write the ion file
        """
        ion_xyz = f"""1
{ion} ion
{ion}    0.0000    0.0000    0.0000
"""
        with open(filepath, "w") as f:
            f.write(ion_xyz)

    def _run_packmol(self, input_path: str, output_path: str) -> None:
        """
        Run PACKMOL to generate solution configuration.

        Args:
            input_path: Path to PACKMOL input file
            output_path: Path where PACKMOL will write the output

        Raises:
            RuntimeError: If PACKMOL fails
        """
        try:
            # Run PACKMOL using shell with input redirection
            cmd = f"{self.parameters.packmol_executable} < {input_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                if self.logger:
                    self.logger.info("PACKMOL completed successfully")
                # Read the generated solution configuration
                self.solution_atoms = read(output_path)
            else:
                raise RuntimeError(f"PACKMOL failed: {result.stderr}")

        except Exception as e:
            raise RuntimeError(f"Error running PACKMOL: {e}")

    def _combine_metal_solution(self) -> None:
        """Combine metal slab and salt-water solution."""
        if self.metal_slab is None or self.solution_atoms is None:
            raise ValueError("Both metal slab and solution must be generated first")

        # Combine atoms
        self.combined_system = self.metal_slab + self.solution_atoms

        # Set the cell from metal slab (will adjust z later)
        cell = self.metal_slab.get_cell()
        self.combined_system.set_cell(cell)
        self.combined_system.set_pbc([True, True, True])

        # Count components
        symbols = self.solution_atoms.get_chemical_symbols()
        n_water = symbols.count("O")

        if self.parameters.n_salt > 0:
            n_cations = symbols.count(self.salt_info["cation"])
            n_anions = symbols.count(self.salt_info["anion"])

            if self.logger:
                self.logger.info(
                    f"Combined system: {len(self.metal_slab)} {self.parameters.metal} atoms + "
                    f"{n_water} water molecules + {n_cations} {self.salt_info['cation']}+ + "
                    f"{n_anions} {self.salt_info['anion']}- ions"
                )
        else:
            if self.logger:
                self.logger.info(
                    f"Combined system: {len(self.metal_slab)} {self.parameters.metal} atoms + "
                    f"{n_water} water molecules"
                )

    def _adjust_vacuum(self) -> None:
        """Adjust the cell to have exact vacuum above the solution layer."""
        if self.combined_system is None:
            raise ValueError("Combine metal and solution first")

        metal_cell_z = self.metal_slab.get_cell()[2, 2]
        solution_height = self._calculate_solution_height()
        total_height = metal_cell_z + solution_height + self.parameters.gap

        # Update cell z-dimension
        cell = self.combined_system.get_cell()
        total_height = total_height + self.parameters.vacuum_above_water
        cell[2, 2] = total_height
        self.combined_system.set_cell(cell)

        if self.logger:
            self.logger.info(f"Adjusted cell z-dimension to {total_height:.2f} Å")
            self.logger.info(f"  Vacuum above solution: {self.parameters.vacuum_above_water:.2f} Å")

    def _write_output(self) -> Path:
        """
        Write the combined system to the output file.

        Returns:
            Path to the output file
        """
        output_path = Path(self.parameters.output_file)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine output format
        output_format = self._determine_format()

        if self.logger:
            self.logger.info(f"Writing output in {output_format} format to: {output_path}")

        if output_format == "lammps":
            self._write_lammps(output_path)
        elif output_format in ["vasp", "poscar"]:
            self._write_poscar(output_path)
        else:  # xyz
            write(str(output_path), self.combined_system, format="xyz")

        return output_path

    def _determine_format(self) -> str:
        """
        Determine the output format based on file extension or explicit format.

        Returns:
            Format string ("xyz", "vasp", "poscar", or "lammps")
        """
        if self.parameters.output_format:
            format_map = {
                "vasp": "vasp",
                "poscar": "poscar",
                "lammps": "lammps",
                "data": "lammps",
                "xyz": "xyz",
            }
            return format_map.get(self.parameters.output_format.lower(), "lammps")

        # Infer from file extension
        output_path = Path(self.parameters.output_file)
        suffix = output_path.suffix.lower()

        if suffix in [".vasp", ".poscar"] or output_path.name.upper() == "POSCAR":
            return "poscar"
        elif suffix in [".lammps", ".data"]:
            return "lammps"
        elif suffix == ".xyz":
            return "xyz"
        else:
            return "lammps"  # Default for metal-salt-water

    def _write_poscar(self, output_path: Path) -> None:
        """
        Write VASP POSCAR format with proper element grouping.

        Args:
            output_path: Output file path
        """
        # Get positions and cell
        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()

        # Build element list dynamically
        element_order = [self.parameters.metal]
        element_counts = [symbols.count(self.parameters.metal)]

        # Add ions if present
        if self.parameters.n_salt > 0:
            element_order.append(self.salt_info["cation"])
            element_counts.append(symbols.count(self.salt_info["cation"]))
            element_order.append(self.salt_info["anion"])
            element_counts.append(symbols.count(self.salt_info["anion"]))

        # Add water atoms
        element_order.extend(["O", "H"])
        element_counts.extend([symbols.count("O"), symbols.count("H")])

        # Sort atoms by element type
        sorted_positions = []
        for element in element_order:
            for i, sym in enumerate(symbols):
                if sym == element:
                    sorted_positions.append(positions[i])

        sorted_positions = np.array(sorted_positions)

        # Write POSCAR file
        with open(output_path, "w") as f:
            # Title
            n_water = symbols.count("O")
            f.write(
                f"{self.parameters.metal}(111) surface with {self.parameters.salt_type} and {n_water} water molecules\n"
            )

            # Scaling factor
            f.write("1.0\n")

            # Cell vectors
            for i in range(3):
                f.write(f"{cell[i,0]:20.16f} {cell[i,1]:20.16f} {cell[i,2]:20.16f}\n")

            # Element symbols
            f.write(" ".join(f"{elem:4s}" for elem in element_order) + "\n")

            # Number of each element
            f.write(" ".join(f"{count:4d}" for count in element_counts) + "\n")

            # Coordinate type
            f.write("Cartesian\n")

            # Atomic positions
            for pos in sorted_positions:
                f.write(f"{pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n")

    def _write_lammps(self, output_path: Path) -> None:
        """
        Write LAMMPS data file format with water topology and ions.

        Args:
            output_path: Output file path
        """
        # Get atomic data
        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()

        # Count atoms by type
        n_metal = symbols.count(self.parameters.metal)
        n_o = symbols.count("O")
        n_h = symbols.count("H")
        n_water = n_o

        # Count ions
        n_cations = 0
        n_anions = 0
        if self.parameters.n_salt > 0:
            n_cations = symbols.count(self.salt_info["cation"])
            n_anions = symbols.count(self.salt_info["anion"])

        # Total counts
        n_atoms = len(positions)
        n_bonds = n_water * 2  # 2 O-H bonds per water
        n_angles = n_water * 1  # 1 H-O-H angle per water

        # Determine number of atom types
        n_atom_types = 3  # Metal, O, H
        if self.parameters.n_salt > 0:
            n_atom_types = 5  # Metal, Cation, Anion, O, H

        # Get atomic masses
        from ase.data import atomic_masses, atomic_numbers

        metal_number = atomic_numbers[self.parameters.metal]
        metal_mass = atomic_masses[metal_number]

        # Get ion parameters
        cation_params = None
        anion_params = None
        if self.parameters.n_salt > 0:
            cation_params = get_ion_params(self.salt_info["cation"])
            anion_params = get_ion_params(self.salt_info["anion"])

        # Get water charges based on model
        model_key = self.parameters.water_model.replace("/", "")
        if model_key in WATER_MODELS:
            o_charge = WATER_MODELS[model_key]["charges"]["O"]
            h_charge = WATER_MODELS[model_key]["charges"]["H"]
        else:
            # Default to SPC/E charges
            o_charge = -0.8476
            h_charge = 0.4238

        with open(output_path, "w") as f:
            # Header
            f.write(f"LAMMPS data file for {self.parameters.metal}(111) surface with ")
            f.write(
                f"{self.parameters.salt_type} and {n_water} {self.parameters.water_model} water molecules\n\n"
            )

            # Counts
            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{n_angles} angles\n")
            f.write("0 dihedrals\n")
            f.write("0 impropers\n\n")

            # Types
            f.write(f"{n_atom_types} atom types\n")
            f.write("1 bond types\n")
            f.write("1 angle types\n\n")

            # Box dimensions
            f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n\n")

            # Masses
            f.write("Masses\n\n")
            if self.parameters.n_salt > 0:
                # With ions: 1=Metal, 2=Cation, 3=Anion, 4=O, 5=H
                f.write(f"1 {metal_mass:.4f}  # {self.parameters.metal}\n")
                f.write(f"2 {cation_params['mass']:.4f}  # {self.salt_info['cation']}\n")
                f.write(f"3 {anion_params['mass']:.4f}  # {self.salt_info['anion']}\n")
                f.write("4 15.9994  # O\n")
                f.write("5 1.00794  # H\n\n")
            else:
                # Without ions: 1=Metal, 2=O, 3=H
                f.write(f"1 {metal_mass:.4f}  # {self.parameters.metal}\n")
                f.write("2 15.9994  # O\n")
                f.write("3 1.00794  # H\n\n")

            # Atoms
            f.write("Atoms\n\n")
            atom_id = 1
            mol_id = 1

            # Track oxygen and hydrogen atom IDs for bonds/angles
            o_atoms = []
            h_atoms = []

            # Write metal atoms first (molecule ID 1)
            for i in range(len(symbols)):
                if symbols[i] == self.parameters.metal:
                    f.write(
                        f"{atom_id} {mol_id} 1 0.0 {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n"
                    )
                    atom_id += 1

            # Write ions if present (each ion gets its own molecule ID)
            if self.parameters.n_salt > 0:
                # Cations
                for i in range(len(symbols)):
                    if symbols[i] == self.salt_info["cation"]:
                        mol_id += 1
                        f.write(f"{atom_id} {mol_id} 2 {cation_params['charge']:.4f} ")
                        f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                        atom_id += 1

                # Anions
                for i in range(len(symbols)):
                    if symbols[i] == self.salt_info["anion"]:
                        mol_id += 1
                        f.write(f"{atom_id} {mol_id} 3 {anion_params['charge']:.4f} ")
                        f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                        atom_id += 1

            # Determine atom types for water
            o_type = 4 if self.parameters.n_salt > 0 else 2
            h_type = 5 if self.parameters.n_salt > 0 else 3

            # Write water molecules
            mol_id += 1
            water_mol_start = mol_id
            h_count = 0

            for i in range(len(symbols)):
                if symbols[i] == "O":
                    o_atoms.append(atom_id)
                    f.write(f"{atom_id} {mol_id} {o_type} {o_charge:.4f} ")
                    f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                    atom_id += 1
                elif symbols[i] == "H":
                    h_atoms.append(atom_id)
                    f.write(f"{atom_id} {mol_id} {h_type} {h_charge:.4f} ")
                    f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                    atom_id += 1
                    h_count += 1
                    # Increment molecule ID after every 2 H atoms (complete water molecule)
                    if h_count % 2 == 0:
                        mol_id += 1

            # Bonds
            if n_bonds > 0:
                f.write("\nBonds\n\n")
                bond_id = 1
                for i, o_id in enumerate(o_atoms):
                    # Each oxygen bonds to next two hydrogens
                    h1_id = h_atoms[2 * i]
                    h2_id = h_atoms[2 * i + 1]
                    f.write(f"{bond_id} 1 {o_id} {h1_id}\n")
                    bond_id += 1
                    f.write(f"{bond_id} 1 {o_id} {h2_id}\n")
                    bond_id += 1

            # Angles
            if n_angles > 0:
                f.write("\nAngles\n\n")
                angle_id = 1
                for i, o_id in enumerate(o_atoms):
                    h1_id = h_atoms[2 * i]
                    h2_id = h_atoms[2 * i + 1]
                    f.write(f"{angle_id} 1 {h1_id} {o_id} {h2_id}\n")
                    angle_id += 1

    def run(self, save_artifacts: bool = False) -> str:
        """
        Run the interface generation (compatibility method).

        Args:
            save_artifacts: Whether to save intermediate files (not used)

        Returns:
            Path to the output file
        """
        return self.generate()
