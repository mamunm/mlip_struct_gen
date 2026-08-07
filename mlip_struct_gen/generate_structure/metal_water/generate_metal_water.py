"""Metal-water interface generation using ASE and PACKMOL."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...utils.water_models import WATER_MODELS
from ..composition import assign_random_symbols, get_element_mass, parse_composition
from .input_parameters import MetalWaterParameters
from .validation import get_lattice_constant, get_water_model_params, validate_parameters

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


def _mass_for(elem: str) -> float:
    """Mass for LAMMPS output; water masses match the historical values."""
    if elem == "H":
        return 1.00794
    return get_element_mass(elem)


try:
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    from ase.io import read, write
except ImportError as e:
    raise ImportError(
        "ASE (Atomic Simulation Environment) is required for metal-water generation. "
        "Install with: pip install ase"
    ) from e


class MetalWaterGenerator:
    """Generate FCC(111) metal surfaces with water layers using ASE and PACKMOL."""

    def __init__(self, parameters: MetalWaterParameters):
        """
        Initialize metal-water interface generator.

        Args:
            parameters: Interface generation parameters

        Raises:
            ValueError: If parameters are invalid
            ImportError: If ASE is not available
            RuntimeError: If PACKMOL is not available
        """
        self.parameters = parameters
        validate_parameters(self.parameters)

        # Parse metal specification (single element or alloy composition);
        # element order defines the canonical atom-type ordering
        self.composition = parse_composition(self.parameters.metal)
        self.metal_elements = list(self.composition)

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

        # Get lattice constant
        self.lattice_constant = get_lattice_constant(
            self.parameters.metal, self.parameters.lattice_constant
        )

        # Get water model parameters
        self.water_params = get_water_model_params(self.parameters.water_model)

        # Storage for intermediate structures
        self.metal_slab = None
        self.water_atoms = None
        self.combined_system = None
        self.box_dimensions = None

        if self.logger:
            self.logger.info("Initializing MetalWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Metal size: {self.parameters.size}")
            self.logger.info(f"Water molecules: {self.parameters.n_water}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.3f} Å")

    def generate(self) -> str:
        """
        Generate the metal-water interface.

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

                # Generate water configuration
                self._generate_water(tmpdir)

                # Combine metal and water
                self._combine_metal_water()

                # Adjust vacuum
                self._adjust_vacuum()

                # Write output file
                output_path = self._write_output()

                if self.logger:
                    self.logger.info(f"Successfully generated: {output_path}")

                return str(output_path)

        except Exception as e:
            error_msg = f"Failed to generate metal-water interface: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from None

    def _build_metal_surface(self) -> None:
        """Build the FCC(111) metal surface."""
        nx, ny, nz = self.parameters.size

        if self.logger:
            self.logger.info(f"Creating {self.parameters.metal}(111) surface")
            self.logger.info(f"  Dimensions: {nx}x{ny} unit cells, {nz} layers")

        # Build the surface using ASE's fcc111 function
        # orthogonal=True ensures LAMMPS compatibility
        # For alloys the first element is a placeholder; symbols are
        # randomly reassigned to match the composition below
        self.metal_slab = fcc111(
            self.metal_elements[0],
            size=(nx, ny, nz),
            a=self.lattice_constant,
            orthogonal=True,
            vacuum=0.0,
            periodic=True,
        )

        if len(self.composition) > 1:
            rng = np.random.default_rng(self.parameters.seed)
            assign_random_symbols(self.metal_slab, self.composition, rng)
            if self.logger:
                self.logger.info(
                    f"Assigned alloy composition {self.parameters.metal} "
                    f"(seed={self.parameters.seed})"
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

    def _calculate_water_height(self) -> float:
        """
        Calculate water box height to achieve target density.

        Returns:
            Water box height in Angstroms
        """
        # Water molecular weight: 18.015 g/mol
        water_mw = 18.015  # g/mol
        avogadro = 6.022e23  # molecules/mol

        # Convert density to g/Å³
        density_g_a3 = self.parameters.density * 1e-24  # g/cm³ to g/Å³

        # Calculate mass of water
        mass_g = self.parameters.n_water * water_mw / avogadro

        # Calculate required volume
        volume_a3 = mass_g / density_g_a3

        # Calculate height from volume
        water_height = volume_a3 / (self.box_dimensions["x"] * self.box_dimensions["y"])

        if self.logger:
            self.logger.info(
                f"Water box height for {self.parameters.n_water} molecules "
                f"at {self.parameters.density} g/cm³: {water_height:.2f} Å"
            )

        return water_height

    def _generate_water(self, tmpdir: str) -> None:
        """
        Generate water configuration using PACKMOL.

        Args:
            tmpdir: Temporary directory for intermediate files
        """
        if self.logger:
            self.logger.info("Generating water configuration with PACKMOL")

        # Calculate water box dimensions
        metal_cell_z = self.metal_slab.get_cell()[2, 2]
        water_height = self._calculate_water_height()

        # Set margins for water box
        margin_xy = 1.0  # margin from box edges in x and y
        margin_z_bottom = 3.0  # additional margin from metal top
        margin_z_top = 3.0  # margin from water top to boundary

        water_x = self.box_dimensions["x"] - 2 * margin_xy
        water_y = self.box_dimensions["y"] - 2 * margin_xy
        water_z_min = metal_cell_z + self.parameters.gap_above_metal + margin_z_bottom
        water_z_max = metal_cell_z + self.parameters.gap_above_metal + water_height - margin_z_top

        # Create water molecule file
        water_xyz_path = os.path.join(tmpdir, "water_molecule.xyz")
        self._create_water_molecule_file(water_xyz_path)

        # Create PACKMOL input
        packmol_input_path = os.path.join(tmpdir, "pack_water.inp")
        water_output_path = os.path.join(tmpdir, "water_box.xyz")

        packmol_input = f"""
tolerance {self.parameters.packmol_tolerance}
filetype xyz
output {water_output_path}
seed {self.parameters.seed}

structure {water_xyz_path}
  number {self.parameters.n_water}
  inside box {margin_xy} {margin_xy} {water_z_min} {water_x + margin_xy} {water_y + margin_xy} {water_z_max}
end structure
"""

        with open(packmol_input_path, "w") as f:
            f.write(packmol_input)

        if self.logger:
            self.logger.info(
                f"Water box: x=[{margin_xy:.1f}, {water_x + margin_xy:.1f}], "
                f"y=[{margin_xy:.1f}, {water_y + margin_xy:.1f}], "
                f"z=[{water_z_min:.2f}, {water_z_max:.2f}]"
            )

        # Run PACKMOL
        self._run_packmol(packmol_input_path, water_output_path)

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

    def _run_packmol(self, input_path: str, output_path: str) -> None:
        """
        Run PACKMOL to generate water configuration.

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
                # Read the generated water configuration
                self.water_atoms = read(output_path)
            else:
                raise RuntimeError(f"PACKMOL failed: {result.stderr}")

        except Exception as e:
            raise RuntimeError(f"Error running PACKMOL: {e}") from e

    def _combine_metal_water(self) -> None:
        """Combine metal slab and water molecules."""
        if self.metal_slab is None or self.water_atoms is None:
            raise ValueError("Both metal slab and water must be generated first")

        # Combine atoms
        self.combined_system = self.metal_slab + self.water_atoms

        # Set the cell from metal slab (will adjust z later)
        cell = self.metal_slab.get_cell()
        self.combined_system.set_cell(cell)
        self.combined_system.set_pbc([True, True, True])

        n_water = len(self.water_atoms) // 3
        if self.logger:
            self.logger.info(
                f"Combined system: {len(self.metal_slab)} {self.parameters.metal} atoms + "
                f"{len(self.water_atoms)} atoms ({n_water} water molecules)"
            )

    def _adjust_vacuum(self) -> None:
        """Adjust the cell to have exact vacuum above the water layer."""
        if self.combined_system is None:
            raise ValueError("Combine metal and water first")

        metal_cell_z = self.metal_slab.get_cell()[2, 2]
        water_height = self._calculate_water_height()
        total_height = metal_cell_z + water_height + self.parameters.gap_above_metal

        # Update cell z-dimension
        cell = self.combined_system.get_cell()
        total_height = total_height + self.parameters.vacuum_above_water
        cell[2, 2] = total_height
        self.combined_system.set_cell(cell)

        if self.logger:
            self.logger.info(f"Adjusted cell z-dimension to {total_height:.2f} Å")
            self.logger.info(f"  Vacuum above water: {self.parameters.vacuum_above_water:.2f} Å")

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
        elif output_format == "lammps/dpmd":
            self._write_lammps_atomic(output_path)
        elif output_format in ["vasp", "poscar"]:
            self._write_poscar(output_path)
        elif output_format == "lammpstrj":
            self._write_lammpstrj(output_path)
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
                "lammps/dpmd": "lammps/dpmd",
                "data": "lammps",
                "xyz": "xyz",
                "lammpstrj": "lammpstrj",
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
            return "lammps"  # Default for metal-water

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

        # Group atoms by element: metal(s) in composition order, then O, H
        element_order = self.metal_elements + ["O", "H"]
        n_o = symbols.count("O")

        counts = [symbols.count(elem) for elem in element_order]
        sorted_positions = np.array(
            [positions[i] for elem in element_order for i, sym in enumerate(symbols) if sym == elem]
        )

        # Write POSCAR file
        with open(output_path, "w") as f:
            # Title
            f.write(f"{self.parameters.metal}(111) surface with {n_o} water molecules\n")

            # Scaling factor
            f.write("1.0\n")

            # Cell vectors
            for i in range(3):
                f.write(f"{cell[i,0]:20.16f} {cell[i,1]:20.16f} {cell[i,2]:20.16f}\n")

            # Element symbols
            f.write(" ".join(f"{elem:4s}" for elem in element_order) + "\n")

            # Number of each element
            f.write(" ".join(f"{n:4d}" for n in counts) + "\n")

            # Coordinate type
            f.write("Cartesian\n")

            # Atomic positions
            for pos in sorted_positions:
                f.write(f"{pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n")

    def _write_lammps(self, output_path: Path) -> None:
        """
        Write LAMMPS data file format with water topology.

        Args:
            output_path: Output file path
        """
        # Get atomic data
        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()

        # Count atoms by type
        n_o = symbols.count("O")
        n_water = n_o

        # Total counts
        n_atoms = len(positions)
        n_bonds = n_water * 2  # 2 O-H bonds per water
        n_angles = n_water * 1  # 1 H-O-H angle per water

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
            f.write(
                f"LAMMPS data file for {self.parameters.metal}(111) surface with {n_water} {self.parameters.water_model} water molecules\n\n"
            )

            # Counts
            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{n_angles} angles\n")
            f.write("0 dihedrals\n")
            f.write("0 impropers\n\n")

            # Determine atom types: canonical order is metal element(s) in
            # composition order, then O, H; --elements overrides
            if self.parameters.elements:
                element_to_type = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}
            else:
                element_to_type = {}
            for elem in self.metal_elements + ["O", "H"]:
                if elem not in element_to_type:
                    element_to_type[elem] = len(element_to_type) + 1
            o_type = element_to_type["O"]
            h_type = element_to_type["H"]
            max_type = len(element_to_type)

            # Types
            f.write(f"{max_type} atom types\n")
            f.write("1 bond types\n")
            f.write("1 angle types\n\n")

            # Box dimensions
            f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n\n")

            # Masses
            f.write("Masses\n\n")
            for elem, type_id in sorted(element_to_type.items(), key=lambda x: x[1]):
                f.write(f"{type_id} {_mass_for(elem):.4f}  # {elem}\n")
            f.write("\n")

            # Atoms
            f.write("Atoms\n\n")
            atom_id = 1
            mol_id = 1

            # Track oxygen and hydrogen atom IDs for bonds/angles
            o_atoms = []
            h_atoms = []

            # Write metal atoms first (molecule ID 1)
            metal_set = set(self.metal_elements)
            for i in range(len(symbols)):
                if symbols[i] in metal_set:
                    f.write(
                        f"{atom_id} {mol_id} {element_to_type[symbols[i]]} 0.0 {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n"
                    )
                    atom_id += 1

            # Write water molecules (molecule IDs start from 2)
            mol_id = 2
            for i in range(len(symbols)):
                if symbols[i] == "O":
                    o_atoms.append(atom_id)
                    f.write(
                        f"{atom_id} {mol_id} {o_type} {o_charge:.4f} {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n"
                    )
                    atom_id += 1
                elif symbols[i] == "H":
                    h_atoms.append(atom_id)
                    f.write(
                        f"{atom_id} {mol_id} {h_type} {h_charge:.4f} {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n"
                    )
                    atom_id += 1
                    # Increment molecule ID after every 3 water atoms (O + 2H)
                    if len(h_atoms) % 2 == 0:
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

    def _write_lammpstrj(self, output_path: Path) -> None:
        """
        Write LAMMPS trajectory format.

        Args:
            output_path: Output file path
        """
        # Get cell dimensions
        cell = self.combined_system.get_cell()
        positions = self.combined_system.get_positions()
        symbols = self.combined_system.get_chemical_symbols()

        # Write custom LAMMPS dump format
        with open(output_path, "w") as f:
            # Header
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(self.combined_system)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"0.0 {cell[0, 0]:.6f}\n")
            f.write(f"0.0 {cell[1, 1]:.6f}\n")
            f.write(f"0.0 {cell[2, 2]:.6f}\n")
            f.write("ITEM: ATOMS id type element x y z\n")

            # Create type mapping (canonical order, matching the data files)
            if self.parameters.elements:
                unique_elements = list(self.parameters.elements)
            else:
                unique_elements = self.metal_elements + ["O", "H"]
            for elem in symbols:
                if elem not in unique_elements:
                    unique_elements.append(elem)

            type_map = {elem: i + 1 for i, elem in enumerate(unique_elements)}

            # Write atoms
            for i, (symbol, pos) in enumerate(zip(symbols, positions, strict=False)):
                atom_type = type_map.get(symbol, len(type_map) + 1)
                f.write(f"{i+1} {atom_type} {symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    def _write_lammps_atomic(self, output_path: Path) -> None:
        """
        Write LAMMPS data file in atomic format (no charges, bonds, angles).
        Suitable for DPMD simulations.

        Args:
            output_path: Output file path
        """
        # Get atomic data
        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()
        n_atoms = len(self.combined_system)

        # Count atoms by type
        element_counts = {}
        for symbol in symbols:
            element_counts[symbol] = element_counts.get(symbol, 0) + 1

        # Count water molecules (assuming 3 atoms per water: O, H, H)
        n_water = sum(1 for s in symbols if s == "O")
        n_metal = sum(element_counts.get(elem, 0) for elem in self.metal_elements)

        with open(output_path, "w") as f:
            # Header
            f.write(
                f"# LAMMPS data file for {self.parameters.metal}-water interface (atomic style for DPMD)\n"
            )
            f.write("# Generated by mlip-struct-gen\n\n")

            # Counts section - only atoms
            f.write(f"{n_atoms} atoms\n\n")

            # Determine atom types: canonical order is metal element(s) in
            # composition order, then O, H; --elements overrides
            if self.parameters.elements:
                # Use predefined element order
                element_to_type = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}
                # Ensure all present elements have types
                for elem in element_counts:
                    if elem not in element_to_type:
                        element_to_type[elem] = len(element_to_type) + 1
                max_atom_type = len(self.parameters.elements)
            else:
                unique_elements = self.metal_elements + ["O", "H"]
                for elem in symbols:
                    if elem not in unique_elements:
                        unique_elements.append(elem)
                element_to_type = {elem: i + 1 for i, elem in enumerate(unique_elements)}
                max_atom_type = len(unique_elements)

            # Types section
            f.write(f"{max_atom_type} atom types\n\n")

            # Box dimensions
            f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n\n")

            # Masses section
            f.write("Masses\n\n")
            if self.parameters.elements:
                # Write masses for all elements in predefined order
                for i, elem in enumerate(self.parameters.elements, 1):
                    f.write(f"{i} {get_element_mass(elem):<10.4f} # {elem}\n")
            else:
                # Write masses for elements present in structure
                for elem, type_id in sorted(element_to_type.items(), key=lambda x: x[1]):
                    f.write(f"{type_id} {get_element_mass(elem):<10.4f} # {elem}\n")
            f.write("\n")

            # Atoms section - atomic style (no molecule ID, no charge)
            f.write("Atoms # atomic\n\n")
            for atom_id, (symbol, pos) in enumerate(zip(symbols, positions, strict=False), 1):
                atom_type = element_to_type[symbol]
                f.write(
                    f"{atom_id:6d} {atom_type:6d} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n"
                )

        if self.logger:
            self.logger.success("Successfully wrote LAMMPS atomic format for DPMD")
            self.logger.info(
                f"System: {n_metal} {self.parameters.metal} atoms, {n_water} water molecules"
            )

    def run(self, save_artifacts: bool = False) -> str:
        """
        Run the interface generation (compatibility method).

        Args:
            save_artifacts: Whether to save intermediate files (not used)

        Returns:
            Path to the output file
        """
        return self.generate()
