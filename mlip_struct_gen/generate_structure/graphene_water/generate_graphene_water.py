"""Graphene-water interface generation using ASE and PACKMOL."""

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..templates.water_models import create_water_xyz
from .input_parameters import GrapheneWaterParameters
from .validation import get_water_model_params, validate_parameters

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger

# Element masses in g/mol
ELEMENT_MASSES = {
    "H": 1.008,
    "O": 15.9994,
    "C": 12.011,
    "Na": 22.98977,
    "Cl": 35.453,
    "K": 39.0983,
    "Li": 6.941,
    "Ca": 40.078,
    "Mg": 24.305,
    "Br": 79.904,
    "Cs": 132.905,
}

try:
    from ase import Atoms
    from ase.build import graphene_nanoribbon
    from ase.io import read, write
except ImportError as e:
    raise ImportError(
        "ASE (Atomic Simulation Environment) is required for graphene-water generation. "
        "Install with: pip install ase"
    ) from e


class GrapheneWaterGenerator:
    """Generate graphene monolayer with water using ASE and PACKMOL."""

    def __init__(self, parameters: GrapheneWaterParameters):
        """
        Initialize graphene-water interface generator.

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
        self.logger: "MLIPLogger | None" = None
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

        # Get water model parameters
        self.water_params = get_water_model_params(self.parameters.water_model)

        if self.logger:
            self.logger.info("Initializing GrapheneWaterGenerator")
            self.logger.info(f"Graphene size: {self.parameters.size}")
            self.logger.info(f"Water molecules: {self.parameters.n_water}")
            self.logger.info(f"Water model: {self.parameters.water_model}")

        self._check_packmol()

        # Storage for intermediate structures
        self.graphene_atoms = None
        self.water_atoms = None
        self.combined_system = None
        self.box_dimensions = None

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
                text=True,
            )
            # Packmol exits with code 171 when given empty input - this is expected
            if result.returncode == 171:
                if self.logger:
                    self.logger.success(f"Packmol found: {self.parameters.packmol_executable}")
            else:
                if self.logger:
                    self.logger.warning(
                        f"Packmol returned unexpected exit code: {result.returncode}"
                    )
                    self.logger.success(
                        f"But Packmol executable appears to be working: {self.parameters.packmol_executable}"
                    )

        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.error("Packmol check timed out after 5 seconds")
            raise RuntimeError(
                f"Packmol executable '{self.parameters.packmol_executable}' timed out. "
                "This may indicate an installation issue."
            ) from None
        except FileNotFoundError:
            if self.logger:
                self.logger.error(
                    f"Packmol executable '{self.parameters.packmol_executable}' not found in PATH"
                )
            raise RuntimeError(
                f"Packmol executable '{self.parameters.packmol_executable}' not found. "
                "Please install packmol:\n"
                "  conda install -c conda-forge packmol\n"
                "Or compile from source: https://github.com/m3g/packmol"
            ) from None

    def generate(self) -> str:
        """
        Generate the graphene-water interface.

        Returns:
            Path to the output file

        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Step 1: Create graphene sheet
            if self.logger:
                self.logger.step("Creating graphene sheet")
            self._create_graphene()

            # Step 2: Calculate water box dimensions
            if self.logger:
                self.logger.step("Calculating water box dimensions")
            self._calculate_water_box_dimensions()

            # Step 3: Generate water using PACKMOL
            if self.logger:
                self.logger.step("Generating water molecules with PACKMOL")
            self._generate_water()

            # Step 4: Combine graphene and water
            if self.logger:
                self.logger.step("Combining graphene and water")
            self._combine_structures()

            # Step 5: Write output
            if self.logger:
                self.logger.step("Writing output file")
            output_path = self._write_output()

            if self.logger:
                self.logger.success(f"Successfully generated: {output_path}")

            return str(output_path)

        except Exception as e:
            error_msg = f"Failed to generate graphene-water interface: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from None

    def _create_graphene(self) -> None:
        """Create orthogonal graphene sheet using ASE's graphene_nanoribbon."""
        nx, ny = self.parameters.size

        # Calculate C-C bond length from lattice constant
        # For graphene: a = √3 * C_C_bond_length
        c_c_bond = self.parameters.a / np.sqrt(3)

        # Generate orthogonal graphene using graphene_nanoribbon
        # This creates a sheet in the x-z plane
        self.graphene_atoms = graphene_nanoribbon(
            n=nx,  # Width parameter
            m=ny,  # Length parameter
            type="armchair",
            saturated=False,  # No hydrogen termination
            C_C=c_c_bond,  # C-C bond length
            vacuum=self.parameters.graphene_vacuum,
            magnetic=False,
            initial_mag=0.0,
        )

        # graphene_nanoribbon creates sheet in x-z plane, we need it in x-y plane
        # So we need to swap y and z coordinates
        positions = self.graphene_atoms.get_positions()
        # Swap y and z coordinates: (x, y, z) -> (x, z, y)
        new_positions = positions[:, [0, 2, 1]]
        self.graphene_atoms.set_positions(new_positions)

        # Also swap the cell dimensions
        old_cell = self.graphene_atoms.get_cell()
        new_cell = np.zeros((3, 3))
        new_cell[0, 0] = old_cell[0, 0]  # x stays the same
        new_cell[1, 1] = old_cell[2, 2]  # y gets old z dimension
        new_cell[2, 2] = old_cell[1, 1] if old_cell[1, 1] > 0 else 10.0  # z gets old y (or default)
        self.graphene_atoms.set_cell(new_cell)

        # Get the cell dimensions - should now be orthogonal
        cell = self.graphene_atoms.get_cell()
        self.box_dimensions = [cell[0, 0], cell[1, 1], 0.0]

        if self.logger:
            self.logger.info(
                f"Created orthogonal graphene sheet with {len(self.graphene_atoms)} atoms"
            )
            self.logger.info(
                f"Graphene dimensions: {self.box_dimensions[0]:.2f} x {self.box_dimensions[1]:.2f} Å"
            )
            # Verify orthogonality
            if abs(cell[0, 1]) < 1e-10 and abs(cell[1, 0]) < 1e-10:
                self.logger.info("Cell is orthogonal (LAMMPS compatible)")
            else:
                self.logger.warning(
                    f"Cell may have non-orthogonal components: {cell[0,1]:.6f}, {cell[1,0]:.6f}"
                )

    def _calculate_water_box_dimensions(self) -> None:
        """Calculate the dimensions for the water box."""
        # Water box has same x,y as graphene
        water_box_x = self.box_dimensions[0]
        water_box_y = self.box_dimensions[1]

        # Calculate water box height from density
        water_molar_mass = 18.015  # g/mol
        na = 6.022e23  # Avogadro's number

        # Calculate required volume
        moles = self.parameters.n_water / na
        mass_g = moles * water_molar_mass
        volume_cm3 = mass_g / self.parameters.water_density
        volume_angstrom3 = volume_cm3 * 1e24

        # Calculate water box height
        box_area = water_box_x * water_box_y
        water_box_z = volume_angstrom3 / box_area

        # Store full dimensions including z
        self.box_dimensions[2] = (
            water_box_z + self.parameters.gap_above_graphene + self.parameters.vacuum_above_water
        )

        if self.logger:
            self.logger.info(
                f"Water box dimensions: {water_box_x:.2f} x {water_box_y:.2f} x {water_box_z:.2f} Å"
            )
            self.logger.info(f"Total cell height: {self.box_dimensions[2]:.2f} Å")

    def _generate_water(self) -> None:
        """Generate water molecules using PACKMOL."""
        # Create output directory
        output_path = Path(self.parameters.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            # Create water molecule template
            water_xyz = work_dir / "water.xyz"
            create_water_xyz(self.parameters.water_model, str(water_xyz))

            # Create PACKMOL input file
            input_file = work_dir / "packmol.inp"
            water_output = work_dir / "water.xyz"

            # Calculate water box boundaries
            # Water starts at graphene top + gap
            graphene_top_z = (
                self.parameters.thickness / 2.0 if self.parameters.thickness > 0 else 0.0
            )
            water_z_min = (
                graphene_top_z
                + self.parameters.gap_above_graphene
                + self.parameters.packmol_tolerance / 2
            )

            # Calculate water box height from volume
            water_box_x = self.box_dimensions[0]
            water_box_y = self.box_dimensions[1]

            water_molar_mass = 18.015
            na = 6.022e23
            moles = self.parameters.n_water / na
            mass_g = moles * water_molar_mass
            volume_cm3 = mass_g / self.parameters.water_density
            volume_angstrom3 = volume_cm3 * 1e24
            box_area = water_box_x * water_box_y
            water_box_height = volume_angstrom3 / box_area

            water_z_max = water_z_min + water_box_height - self.parameters.packmol_tolerance / 2

            # Write PACKMOL input
            with open(input_file, "w") as f:
                f.write(f"tolerance {self.parameters.packmol_tolerance}\n")
                f.write("filetype xyz\n")
                f.write(f"output {water_output.name}\n")
                f.write(f"seed {self.parameters.seed}\n")
                f.write("\n")
                f.write(f"structure {water_xyz.name}\n")
                f.write(f"  number {self.parameters.n_water}\n")
                f.write(f"  inside box 0.0 0.0 {water_z_min} ")
                f.write(f"{water_box_x} {water_box_y} {water_z_max}\n")
                f.write("end structure\n")

            # Run PACKMOL
            try:
                with open(input_file) as f:
                    subprocess.run(
                        [self.parameters.packmol_executable],
                        stdin=f,
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=str(work_dir),
                    )

                if self.logger:
                    self.logger.success("PACKMOL execution completed successfully")

            except subprocess.CalledProcessError as e:
                if self.logger:
                    self.logger.error(f"PACKMOL failed with return code {e.returncode}")
                raise RuntimeError(f"PACKMOL failed: {e.stderr}") from e

            # Read the generated water structure
            self.water_atoms = read(str(water_output))

    def _combine_structures(self) -> None:
        """Combine graphene and water into a single system."""
        # Get all atoms
        graphene_positions = self.graphene_atoms.get_positions()
        graphene_symbols = self.graphene_atoms.get_chemical_symbols()

        water_positions = self.water_atoms.get_positions()
        water_symbols = self.water_atoms.get_chemical_symbols()

        # Combine positions and symbols
        all_positions = np.vstack([graphene_positions, water_positions])
        all_symbols = graphene_symbols + water_symbols

        # Create combined system
        self.combined_system = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            cell=[self.box_dimensions[0], self.box_dimensions[1], self.box_dimensions[2]],
            pbc=[True, True, True],
        )

        if self.logger:
            n_c = sum(1 for s in all_symbols if s == "C")
            n_o = sum(1 for s in all_symbols if s == "O")
            n_h = sum(1 for s in all_symbols if s == "H")
            self.logger.info(f"Combined system: {n_c} C, {n_o} O, {n_h} H atoms")

    def _write_output(self) -> Path:
        """Write the combined system to output file."""
        output_path = Path(self.parameters.output_file)

        if self.parameters.output_format == "lammps":
            self._write_lammps_data(output_path, with_topology=True)
        elif self.parameters.output_format == "lammps/dpmd":
            self._write_lammps_data(output_path, with_topology=False)
        elif self.parameters.output_format == "poscar" or self.parameters.output_format == "vasp":
            self._write_poscar(output_path)
        elif self.parameters.output_format == "lammpstrj":
            self._write_lammpstrj(output_path)
        else:
            # Default to XYZ
            write(str(output_path), self.combined_system, format="xyz")

        return output_path

    def _write_lammps_data(self, output_path: Path, with_topology: bool = True) -> None:
        """
        Write LAMMPS data file.

        Args:
            output_path: Output file path
            with_topology: If True, include bonds and angles for water
        """
        atoms = self.combined_system
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        # Count atom types
        n_c = sum(1 for s in symbols if s == "C")
        n_o = sum(1 for s in symbols if s == "O")
        n_water = n_o  # Each water has 1 O

        # Determine atom types
        if self.parameters.elements:
            element_to_type = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}
            c_type = element_to_type.get("C", 1)
            o_type = element_to_type.get("O", 2)
            h_type = element_to_type.get("H", 3)
            max_atom_type = len(self.parameters.elements)
        else:
            c_type = 1
            o_type = 2
            h_type = 3
            max_atom_type = 3

        with open(output_path, "w") as f:
            f.write("# LAMMPS data file for graphene-water interface\n")
            f.write("# Generated by mlip-struct-gen\n\n")

            # Counts
            f.write(f"{len(atoms)} atoms\n")
            if with_topology:
                f.write(f"{n_water * 2} bonds\n")
                f.write(f"{n_water} angles\n")
            f.write("\n")

            # Types
            f.write(f"{max_atom_type} atom types\n")
            if with_topology:
                f.write("1 bond types\n")
                f.write("1 angle types\n")
            f.write("\n")

            # Box dimensions
            f.write(f"0.0 {cell[0, 0]} xlo xhi\n")
            f.write(f"0.0 {cell[1, 1]} ylo yhi\n")
            f.write(f"0.0 {cell[2, 2]} zlo zhi\n")
            f.write("\n")

            # Masses
            f.write("Masses\n\n")
            if self.parameters.elements:
                for i, elem in enumerate(self.parameters.elements, 1):
                    mass = ELEMENT_MASSES.get(elem, 1.0)
                    f.write(f"{i} {mass:<10.4f} # {elem}\n")
            else:
                f.write(f"{c_type} 12.011   # C\n")
                f.write(f"{o_type} 15.9994  # O\n")
                f.write(f"{h_type} 1.008    # H\n")
            f.write("\n")

            # Atoms
            if with_topology:
                f.write("Atoms # full\n\n")
                atom_id = 1
                mol_id = 1

                # Write graphene atoms (no molecule ID for graphene)
                for i in range(n_c):
                    symbol = symbols[i]
                    pos = positions[i]
                    f.write(f"{atom_id:6d} {0:6d} {c_type:6d} {0.0:10.6f} ")
                    f.write(f"{pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
                    atom_id += 1

                # Write water molecules
                water_start_idx = n_c
                for mol in range(n_water):
                    # Each water: O, H, H
                    base_idx = water_start_idx + mol * 3
                    for j in range(3):
                        idx = base_idx + j
                        symbol = symbols[idx]
                        pos = positions[idx]
                        atom_type = o_type if symbol == "O" else h_type

                        # Get charge from water model
                        charge = (
                            self.water_params["atoms"][0]["charge"]
                            if symbol == "O"
                            else self.water_params["atoms"][1]["charge"]
                        )

                        f.write(f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {charge:10.6f} ")
                        f.write(f"{pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
                        atom_id += 1
                    mol_id += 1
                f.write("\n")

                # Bonds
                f.write("Bonds\n\n")
                bond_id = 1
                first_water_atom = n_c + 1
                for mol in range(n_water):
                    base_atom = first_water_atom + mol * 3
                    # O-H1 bond
                    f.write(f"{bond_id:6d} 1 {base_atom:6d} {base_atom + 1:6d}\n")
                    bond_id += 1
                    # O-H2 bond
                    f.write(f"{bond_id:6d} 1 {base_atom:6d} {base_atom + 2:6d}\n")
                    bond_id += 1
                f.write("\n")

                # Angles
                f.write("Angles\n\n")
                angle_id = 1
                for mol in range(n_water):
                    base_atom = first_water_atom + mol * 3
                    # H-O-H angle
                    f.write(
                        f"{angle_id:6d} 1 {base_atom + 1:6d} {base_atom:6d} {base_atom + 2:6d}\n"
                    )
                    angle_id += 1

            else:
                # Atomic style (for DPMD)
                f.write("Atoms # atomic\n\n")
                for i, (symbol, pos) in enumerate(zip(symbols, positions, strict=False), 1):
                    if symbol == "C":
                        atom_type = c_type
                    elif symbol == "O":
                        atom_type = o_type
                    else:  # H
                        atom_type = h_type
                    f.write(f"{i:6d} {atom_type:6d} ")
                    f.write(f"{pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

    def _write_poscar(self, output_path: Path) -> None:
        """Write POSCAR format."""
        atoms = self.combined_system

        # Sort atoms by element (C, O, H order)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        # Group by element
        c_indices = [i for i, s in enumerate(symbols) if s == "C"]
        o_indices = [i for i, s in enumerate(symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(symbols) if s == "H"]

        # Reorder
        new_indices = c_indices + o_indices + h_indices
        sorted_positions = positions[new_indices]
        sorted_symbols = [symbols[i] for i in new_indices]

        # Create new atoms object
        sorted_atoms = Atoms(
            symbols=sorted_symbols, positions=sorted_positions, cell=atoms.cell, pbc=True
        )

        # Write POSCAR
        write(str(output_path), sorted_atoms, format="vasp", direct=False, sort=False)

    def _write_lammpstrj(self, output_path: Path) -> None:
        """Write LAMMPS trajectory format."""
        atoms = self.combined_system
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = atoms.get_cell()

        # Create type mapping
        unique_elements = ["C", "O", "H"]  # Fixed order
        type_map = {elem: i + 1 for i, elem in enumerate(unique_elements)}

        with open(output_path, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(atoms)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"0.0 {cell[0, 0]:.6f}\n")
            f.write(f"0.0 {cell[1, 1]:.6f}\n")
            f.write(f"0.0 {cell[2, 2]:.6f}\n")
            f.write("ITEM: ATOMS id type element x y z\n")

            for i, (symbol, pos) in enumerate(zip(symbols, positions, strict=False)):
                atom_type = type_map[symbol]
                f.write(f"{i+1} {atom_type} {symbol} ")
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
