"""Walled metal-salt-water interface generation using ASE and PACKMOL."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    from ase.io import read, write
except ImportError as e:
    raise ImportError(
        "ASE (Atomic Simulation Environment) is required for walled metal-salt-water generation. "
        "Install with: pip install ase"
    ) from e

from ...utils.water_models import WATER_MODELS
from .input_parameters import WalledMetalSaltWaterParameters
from .validation import (
    get_ion_params,
    get_lattice_constant,
    get_salt_info,
    get_water_model_params,
    validate_parameters,
)

# Element masses in g/mol
ELEMENT_MASSES = {
    "H": 1.008,
    "O": 15.9994,
    "Na": 22.98977,
    "Cl": 35.453,
    "K": 39.0983,
    "Li": 6.941,
    "Ca": 40.078,
    "Mg": 24.305,
    "Br": 79.904,
    "Cs": 132.905,
    "Pt": 195.078,
    "Au": 196.967,
    "Ag": 107.868,
    "Cu": 63.546,
    "Ni": 58.693,
    "Pd": 106.42,
    "Al": 26.982,
    "Fe": 55.845,
}

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


class WalledMetalSaltWaterGenerator:
    """Generate walled FCC(111) metal surfaces with salt water between them."""

    def __init__(self, parameters: WalledMetalSaltWaterParameters):
        """
        Initialize walled metal-salt-water interface generator.

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
                try:
                    from ...utils.logger import MLIPLogger

                    self.logger = MLIPLogger()
                except ImportError:
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
        self.bottom_wall_top = None
        self.top_wall_z = None
        self.solution_height = None
        self.n_bottom = None
        self.n_top = None
        self.bottom_indices = None
        self.top_indices = None

        if self.logger:
            self.logger.info("Initializing WalledMetalSaltWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Metal size: {self.parameters.size}")
            self.logger.info(
                f"Salt: {self.parameters.salt_type} ({self.parameters.n_salt} formula units)"
            )
            self.logger.info(f"Water molecules: {self.parameters.n_water}")
            self.logger.info(f"Box z-dimension: {self.parameters.box_z} Angstroms")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.3f} Angstroms")

    def generate(self) -> str:
        """
        Generate the walled metal-salt-water interface.

        Returns:
            Path to the output file

        Raises:
            RuntimeError: If generation fails
        """
        try:
            if self.parameters.save_artifacts:
                output_path = Path(self.parameters.output_file)
                artifacts_dir = output_path.parent / f"{output_path.stem}_artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                tmpdir = str(artifacts_dir)
                if self.logger:
                    self.logger.info(f"Saving artifacts to: {artifacts_dir}")
                self._run_generation(tmpdir)
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    self._run_generation(tmpdir)

            if self.logger:
                self.logger.info(f"Successfully generated: {self.parameters.output_file}")

            return str(self.parameters.output_file)

        except Exception as e:
            error_msg = f"Failed to generate walled metal-salt-water interface: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from None

    def _run_generation(self, tmpdir: str) -> None:
        """Run the actual generation steps."""
        self._build_metal_walls()
        self._generate_salt_water_solution(tmpdir)
        self._combine_metal_solution()
        self._adjust_vacuum()
        self._write_output()

    def _build_metal_walls(self) -> None:
        """Build two metal walls by splitting a full FCC(111) slab."""
        nx, ny, nz = self.parameters.size

        if self.logger:
            self.logger.info(f"Creating {self.parameters.metal}(111) walled surface")
            self.logger.info(f"  Dimensions: {nx}x{ny} unit cells, {nz} total layers")

        full_slab = fcc111(
            self.parameters.metal,
            size=(nx, ny, nz),
            a=self.lattice_constant,
            orthogonal=True,
            vacuum=0.0,
            periodic=True,
        )

        cell = full_slab.get_cell()
        self.box_dimensions = {"x": cell[0, 0], "y": cell[1, 1], "z": cell[2, 2]}

        positions = full_slab.get_positions()
        z_positions = positions[:, 2]
        z_unique = np.unique(np.round(z_positions, decimals=2))
        z_unique.sort()

        self.n_bottom = (nz + 1) // 2
        self.n_top = nz - self.n_bottom

        if self.logger:
            self.logger.info(f"  Bottom wall: {self.n_bottom} layers")
            self.logger.info(f"  Top wall: {self.n_top} layers")

        bottom_threshold = z_unique[self.n_bottom - 1] + 0.01
        bottom_mask = z_positions < bottom_threshold
        top_mask = ~bottom_mask

        self.bottom_indices = np.where(bottom_mask)[0]
        self.top_indices = np.where(top_mask)[0]

        self.bottom_wall_top = z_unique[self.n_bottom - 1]

        self.solution_height = self._calculate_solution_height()

        self.top_wall_z = (
            self.bottom_wall_top
            + self.parameters.gap_above_metal
            + self.solution_height
            + self.parameters.vacuum_above_water
        )

        original_top_bottom = z_unique[self.n_bottom]
        z_shift = self.top_wall_z - original_top_bottom

        new_positions = positions.copy()
        new_positions[self.top_indices, 2] += z_shift
        full_slab.set_positions(new_positions)

        top_metal_height = z_unique[nz - 1] - z_unique[self.n_bottom]
        top_wall_top_z = self.top_wall_z + top_metal_height

        if self.parameters.box_z < top_wall_top_z:
            raise ValueError(
                f"box_z ({self.parameters.box_z:.2f} Angstroms) is too small. "
                f"Top wall extends to {top_wall_top_z:.2f} Angstroms. "
                f"Minimum box_z required: {top_wall_top_z:.2f} Angstroms."
            )

        if self.logger:
            self.logger.info(f"  Bottom wall top at z = {self.bottom_wall_top:.2f} Angstroms")
            self.logger.info(f"  Top wall bottom at z = {self.top_wall_z:.2f} Angstroms")
            self.logger.info(f"  Top wall top at z = {top_wall_top_z:.2f} Angstroms")
            self.logger.info(f"  Box z = {self.parameters.box_z:.2f} Angstroms")

        self.metal_slab = full_slab

        if self.logger:
            self.logger.info(f"Created walled surface with {len(self.metal_slab)} atoms")

        if self.parameters.fix_bottom_layers > 0:
            self._apply_symmetric_constraints()

    def _apply_symmetric_constraints(self) -> None:
        """Fix bottom N layers of bottom wall AND top N layers of top wall."""
        positions = self.metal_slab.get_positions()
        z_positions = positions[:, 2]
        z_unique = np.unique(np.round(z_positions, decimals=2))
        z_unique.sort()

        n_fix = self.parameters.fix_bottom_layers

        bottom_fix_threshold = z_unique[n_fix - 1] + 0.01
        bottom_fixed = z_positions < bottom_fix_threshold

        top_z_positions = z_positions[self.top_indices]
        top_z_unique = np.unique(np.round(top_z_positions, decimals=2))
        top_z_unique.sort()
        top_fix_threshold = top_z_unique[len(top_z_unique) - n_fix] - 0.01
        top_fixed = z_positions > top_fix_threshold

        fixed_mask = bottom_fixed | top_fixed
        constraint = FixAtoms(mask=fixed_mask)
        self.metal_slab.set_constraint(constraint)

        n_fixed = np.sum(fixed_mask)
        if self.logger:
            self.logger.info(f"Fixed {n_fixed} atoms: {n_fix} bottom layers + {n_fix} top layers")

    def _calculate_solution_height(self) -> float:
        """Calculate solution box height to achieve target density."""
        water_mw = 18.015
        avogadro = 6.022e23

        total_mass_g = self.parameters.n_water * water_mw / avogadro

        if self.parameters.include_salt_volume and self.parameters.n_salt > 0:
            cation_params = get_ion_params(self.salt_info["cation"])
            anion_params = get_ion_params(self.salt_info["anion"])

            n_cations = self.parameters.n_salt * self.salt_info["cation_count"]
            n_anions = self.parameters.n_salt * self.salt_info["anion_count"]

            cation_mass = n_cations * cation_params["mass"] / avogadro
            anion_mass = n_anions * anion_params["mass"] / avogadro

            total_mass_g += cation_mass + anion_mass

        density_g_a3 = self.parameters.density * 1e-24
        volume_a3 = total_mass_g / density_g_a3
        solution_height = volume_a3 / (self.box_dimensions["x"] * self.box_dimensions["y"])

        if self.logger:
            if self.parameters.include_salt_volume:
                self.logger.info(
                    f"Solution box height for {self.parameters.n_water} water + "
                    f"{self.parameters.n_salt} {self.parameters.salt_type} "
                    f"at {self.parameters.density} g/cm^3: {solution_height:.2f} Angstroms"
                )
            else:
                self.logger.info(
                    f"Solution box height for {self.parameters.n_water} water molecules "
                    f"at {self.parameters.density} g/cm^3: {solution_height:.2f} Angstroms"
                )

        return solution_height

    def _generate_salt_water_solution(self, tmpdir: str) -> None:
        """Generate salt-water solution using PACKMOL between the two walls."""
        if self.logger:
            self.logger.info("Generating salt-water solution with PACKMOL")

        margin_xy = 1.0
        margin_z_bottom = 3.0
        margin_z_top = 3.0

        solution_x = self.box_dimensions["x"] - 2 * margin_xy
        solution_y = self.box_dimensions["y"] - 2 * margin_xy
        solution_z_min = self.bottom_wall_top + self.parameters.gap_above_metal + margin_z_bottom
        solution_z_max = (
            self.bottom_wall_top
            + self.parameters.gap_above_metal
            + self.solution_height
            - margin_z_top
        )

        # Restricted z-boundaries for salt ions based on no_salt_zone
        solution_box_height = solution_z_max - solution_z_min
        salt_z_min = solution_z_min + (self.parameters.no_salt_zone * solution_box_height)
        salt_z_max = solution_z_max - (self.parameters.no_salt_zone * solution_box_height)

        # Create molecule files
        water_xyz_path = os.path.join(tmpdir, "water_molecule.xyz")
        self._create_water_molecule_file(water_xyz_path)

        packmol_input_path = os.path.join(tmpdir, "pack_solution.inp")
        solution_output_path = os.path.join(tmpdir, "solution_box.xyz")

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

        if self.parameters.n_salt > 0:
            n_cations = self.parameters.n_salt * self.salt_info["cation_count"]
            n_anions = self.parameters.n_salt * self.salt_info["anion_count"]

            cation_xyz_path = os.path.join(tmpdir, f'{self.salt_info["cation"]}.xyz')
            anion_xyz_path = os.path.join(tmpdir, f'{self.salt_info["anion"]}.xyz')

            self._create_ion_file(self.salt_info["cation"], cation_xyz_path)
            self._create_ion_file(self.salt_info["anion"], anion_xyz_path)

            packmol_input += f"""
# Cations
structure {cation_xyz_path}
  number {n_cations}
  inside box {margin_xy} {margin_xy} {salt_z_min} {solution_x + margin_xy} {solution_y + margin_xy} {salt_z_max}
end structure

# Anions
structure {anion_xyz_path}
  number {n_anions}
  inside box {margin_xy} {margin_xy} {salt_z_min} {solution_x + margin_xy} {solution_y + margin_xy} {salt_z_max}
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
                    f"  Ions: {n_cations} {self.salt_info['cation']}+ and "
                    f"{n_anions} {self.salt_info['anion']}-"
                )
                if self.parameters.no_salt_zone > 0:
                    self.logger.info(
                        f"  Ion exclusion zone: {self.parameters.no_salt_zone:.1%} from top/bottom "
                        f"(ions in z=[{salt_z_min:.2f}, {salt_z_max:.2f}])"
                    )

        self._run_packmol(packmol_input_path, solution_output_path)

    def _create_water_molecule_file(self, filepath: str) -> None:
        """Create water molecule XYZ file based on water model."""
        model_key = self.parameters.water_model.replace("/", "")
        if model_key in WATER_MODELS:
            geometry = WATER_MODELS[model_key]["geometry"]
            water_xyz = f"""3
Water molecule {self.parameters.water_model}
O    {geometry['O'][0]:.4f}    {geometry['O'][1]:.4f}    {geometry['O'][2]:.4f}
H    {geometry['H1'][0]:.4f}   {geometry['H1'][1]:.4f}   {geometry['H1'][2]:.4f}
H    {geometry['H2'][0]:.4f}   {geometry['H2'][1]:.4f}   {geometry['H2'][2]:.4f}
"""
        else:
            water_xyz = """3
Water molecule SPC/E
O    0.0000    0.0000    0.0000
H    0.8164    0.0000    0.5773
H   -0.8164    0.0000    0.5773
"""
        with open(filepath, "w") as f:
            f.write(water_xyz)

    def _create_ion_file(self, ion: str, filepath: str) -> None:
        """Create ion XYZ file."""
        ion_xyz = f"""1
{ion} ion
{ion}    0.0000    0.0000    0.0000
"""
        with open(filepath, "w") as f:
            f.write(ion_xyz)

    def _run_packmol(self, input_path: str, output_path: str) -> None:
        """Run PACKMOL to generate solution configuration."""
        try:
            cmd = f"{self.parameters.packmol_executable} < {input_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                if self.logger:
                    self.logger.info("PACKMOL completed successfully")
                self.solution_atoms = read(output_path)
            else:
                raise RuntimeError(f"PACKMOL failed: {result.stderr}")

        except Exception as e:
            raise RuntimeError(f"Error running PACKMOL: {e}") from e

    def _combine_metal_solution(self) -> None:
        """Combine metal walls and salt-water solution."""
        if self.metal_slab is None or self.solution_atoms is None:
            raise ValueError("Both metal walls and solution must be generated first")

        self.combined_system = self.metal_slab + self.solution_atoms

        cell = self.metal_slab.get_cell()
        self.combined_system.set_cell(cell)
        self.combined_system.set_pbc([True, True, True])

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
        """Set the cell z-dimension to the user-specified box_z."""
        if self.combined_system is None:
            raise ValueError("Combine metal and solution first")

        cell = self.combined_system.get_cell()
        cell[2, 2] = self.parameters.box_z
        self.combined_system.set_cell(cell)

        if self.logger:
            self.logger.info(f"Set cell z-dimension to {self.parameters.box_z:.2f} Angstroms")

    def _write_output(self) -> Path:
        """Write the combined system to the output file."""
        output_path = Path(self.parameters.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
        else:
            write(str(output_path), self.combined_system, format="xyz")

        return output_path

    def _determine_format(self) -> str:
        """Determine the output format based on file extension or explicit format."""
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

        output_path = Path(self.parameters.output_file)
        suffix = output_path.suffix.lower()

        if suffix in [".vasp", ".poscar"] or output_path.name.upper() == "POSCAR":
            return "poscar"
        elif suffix in [".lammps", ".data"]:
            return "lammps"
        elif suffix == ".xyz":
            return "xyz"
        else:
            return "lammps"

    def _write_poscar(self, output_path: Path) -> None:
        """Write VASP POSCAR format with proper element grouping."""
        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()

        element_order = [self.parameters.metal]
        element_counts = [symbols.count(self.parameters.metal)]

        if self.parameters.n_salt > 0:
            element_order.append(self.salt_info["cation"])
            element_counts.append(symbols.count(self.salt_info["cation"]))
            element_order.append(self.salt_info["anion"])
            element_counts.append(symbols.count(self.salt_info["anion"]))

        element_order.extend(["O", "H"])
        element_counts.extend([symbols.count("O"), symbols.count("H")])

        sorted_positions = []
        for element in element_order:
            for i, sym in enumerate(symbols):
                if sym == element:
                    sorted_positions.append(positions[i])

        sorted_positions = np.array(sorted_positions)

        with open(output_path, "w") as f:
            n_water = symbols.count("O")
            f.write(
                f"{self.parameters.metal}(111) walled surface with "
                f"{self.parameters.salt_type} and {n_water} water molecules\n"
            )
            f.write("1.0\n")
            for i in range(3):
                f.write(f"{cell[i,0]:20.16f} {cell[i,1]:20.16f} {cell[i,2]:20.16f}\n")
            f.write(" ".join(f"{elem:4s}" for elem in element_order) + "\n")
            f.write(" ".join(f"{count:4d}" for count in element_counts) + "\n")
            f.write("Cartesian\n")
            for pos in sorted_positions:
                f.write(f"{pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n")

    def _write_lammps(self, output_path: Path) -> None:
        """Write LAMMPS data file format with water topology and ions."""
        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()

        symbols.count(self.parameters.metal)
        n_o = symbols.count("O")
        symbols.count("H")
        n_water = n_o

        if self.parameters.n_salt > 0:
            symbols.count(self.salt_info["cation"])
            symbols.count(self.salt_info["anion"])

        n_atoms = len(positions)
        n_bonds = n_water * 2
        n_angles = n_water * 1

        n_atom_types = 3
        if self.parameters.n_salt > 0:
            n_atom_types = 5

        from ase.data import atomic_masses, atomic_numbers

        metal_number = atomic_numbers[self.parameters.metal]
        metal_mass = atomic_masses[metal_number]

        cation_params = None
        anion_params = None
        if self.parameters.n_salt > 0:
            cation_params = get_ion_params(self.salt_info["cation"])
            anion_params = get_ion_params(self.salt_info["anion"])

        model_key = self.parameters.water_model.replace("/", "")
        if model_key in WATER_MODELS:
            o_charge = WATER_MODELS[model_key]["charges"]["O"]
            h_charge = WATER_MODELS[model_key]["charges"]["H"]
        else:
            o_charge = -0.8476
            h_charge = 0.4238

        with open(output_path, "w") as f:
            f.write(f"LAMMPS data file for {self.parameters.metal}(111) walled surface with ")
            f.write(
                f"{self.parameters.salt_type} and {n_water} "
                f"{self.parameters.water_model} water molecules\n\n"
            )

            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{n_angles} angles\n")
            f.write("0 dihedrals\n")
            f.write("0 impropers\n\n")

            f.write(f"{n_atom_types} atom types\n")
            f.write("1 bond types\n")
            f.write("1 angle types\n\n")

            f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n\n")

            f.write("Masses\n\n")
            if self.parameters.n_salt > 0:
                f.write(f"1 {metal_mass:.4f}  # {self.parameters.metal}\n")
                f.write("2 15.9994  # O\n")
                f.write("3 1.00794  # H\n")
                f.write(f"4 {cation_params['mass']:.4f}  # {self.salt_info['cation']}\n")
                f.write(f"5 {anion_params['mass']:.4f}  # {self.salt_info['anion']}\n\n")
            else:
                f.write(f"1 {metal_mass:.4f}  # {self.parameters.metal}\n")
                f.write("2 15.9994  # O\n")
                f.write("3 1.00794  # H\n\n")

            f.write("Atoms\n\n")
            atom_id = 1
            mol_id = 1

            o_atoms = []
            h_atoms = []

            for i in range(len(symbols)):
                if symbols[i] == self.parameters.metal:
                    f.write(
                        f"{atom_id} {mol_id} 1 0.0 "
                        f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n"
                    )
                    atom_id += 1

            mol_id += 1
            h_count = 0

            for i in range(len(symbols)):
                if symbols[i] == "O":
                    o_atoms.append(atom_id)
                    f.write(f"{atom_id} {mol_id} 2 {o_charge:.4f} ")
                    f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                    atom_id += 1
                elif symbols[i] == "H":
                    h_atoms.append(atom_id)
                    f.write(f"{atom_id} {mol_id} 3 {h_charge:.4f} ")
                    f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                    atom_id += 1
                    h_count += 1
                    if h_count % 2 == 0:
                        mol_id += 1

            if self.parameters.n_salt > 0:
                for i in range(len(symbols)):
                    if symbols[i] == self.salt_info["cation"]:
                        mol_id += 1
                        f.write(f"{atom_id} {mol_id} 4 {cation_params['charge']:.4f} ")
                        f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                        atom_id += 1

                for i in range(len(symbols)):
                    if symbols[i] == self.salt_info["anion"]:
                        mol_id += 1
                        f.write(f"{atom_id} {mol_id} 5 {anion_params['charge']:.4f} ")
                        f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
                        atom_id += 1

            if n_bonds > 0:
                f.write("\nBonds\n\n")
                bond_id = 1
                for i, o_id in enumerate(o_atoms):
                    h1_id = h_atoms[2 * i]
                    h2_id = h_atoms[2 * i + 1]
                    f.write(f"{bond_id} 1 {o_id} {h1_id}\n")
                    bond_id += 1
                    f.write(f"{bond_id} 1 {o_id} {h2_id}\n")
                    bond_id += 1

            if n_angles > 0:
                f.write("\nAngles\n\n")
                angle_id = 1
                for i, o_id in enumerate(o_atoms):
                    h1_id = h_atoms[2 * i]
                    h2_id = h_atoms[2 * i + 1]
                    f.write(f"{angle_id} 1 {h1_id} {o_id} {h2_id}\n")
                    angle_id += 1

    def _write_lammpstrj(self, output_path: Path) -> None:
        """Write LAMMPS trajectory format."""
        if self.combined_system is None:
            raise ValueError("Combined system not yet generated")

        cell = np.array(self.combined_system.get_cell())
        positions = self.combined_system.get_positions()
        symbols = self.combined_system.get_chemical_symbols()

        with open(str(output_path), "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(self.combined_system)}\n")

            is_orthogonal = (
                abs(cell[0, 1]) < 1e-10
                and abs(cell[0, 2]) < 1e-10
                and abs(cell[1, 0]) < 1e-10
                and abs(cell[1, 2]) < 1e-10
                and abs(cell[2, 0]) < 1e-10
                and abs(cell[2, 1]) < 1e-10
            )

            if is_orthogonal:
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write(f"0.0 {float(cell[0, 0]):.6f}\n")
                f.write(f"0.0 {float(cell[1, 1]):.6f}\n")
                f.write(f"0.0 {float(cell[2, 2]):.6f}\n")
            else:
                f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                f.write(f"0.0 {float(cell[0, 0]):.6f} {float(cell[1, 0]):.6f}\n")
                f.write(f"0.0 {float(cell[1, 1]):.6f} {float(cell[2, 0]):.6f}\n")
                f.write(f"0.0 {float(cell[2, 2]):.6f} {float(cell[2, 1]):.6f}\n")

            f.write("ITEM: ATOMS id type element x y z\n")

            unique_elements = sorted(set(symbols))
            type_map = {elem: i + 1 for i, elem in enumerate(unique_elements)}

            for i in range(len(symbols)):
                symbol = symbols[i]
                pos = positions[i]
                atom_type = type_map.get(symbol, len(type_map) + 1)
                f.write(
                    f"{i+1} {atom_type} {symbol} "
                    f"{float(pos[0]):.6f} {float(pos[1]):.6f} {float(pos[2]):.6f}\n"
                )

    def _write_lammps_atomic(self, output_path: Path) -> None:
        """Write LAMMPS data file in atomic format (no charges, bonds, angles)."""
        if self.combined_system is None:
            raise ValueError("Combined system not yet generated")

        positions = self.combined_system.get_positions()
        cell = self.combined_system.get_cell()
        symbols = self.combined_system.get_chemical_symbols()
        n_atoms = len(self.combined_system)

        element_counts = {}
        for symbol in symbols:
            element_counts[symbol] = element_counts.get(symbol, 0) + 1

        n_water = sum(1 for s in symbols if s == "O")
        n_metal = element_counts.get(self.parameters.metal, 0)

        from ..templates.salt_models import get_salt_model

        salt_model = get_salt_model(self.parameters.salt_type)
        cation_element = salt_model["cation"]["element"]
        anion_element = salt_model["anion"]["element"]
        n_cations = element_counts.get(cation_element, 0)
        n_anions = element_counts.get(anion_element, 0)

        with open(output_path, "w") as f:
            f.write(
                f"# LAMMPS data file for {self.parameters.metal}-{self.parameters.salt_type}-water "
                f"walled interface (atomic style for DPMD)\n"
            )
            f.write("# Generated by mlip-struct-gen\n\n")

            f.write(f"{n_atoms} atoms\n\n")

            if self.parameters.elements:
                element_to_type = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}
                for elem in element_counts:
                    if elem not in element_to_type:
                        element_to_type[elem] = len(element_to_type) + 1
                max_atom_type = len(self.parameters.elements)
            else:
                unique_elements = sorted(set(symbols))
                ordered_elements = []
                if self.parameters.metal in unique_elements:
                    ordered_elements.append(self.parameters.metal)
                    unique_elements.remove(self.parameters.metal)
                if "O" in unique_elements:
                    ordered_elements.append("O")
                    unique_elements.remove("O")
                if "H" in unique_elements:
                    ordered_elements.append("H")
                    unique_elements.remove("H")
                ordered_elements.extend(sorted(unique_elements))
                element_to_type = {elem: i + 1 for i, elem in enumerate(ordered_elements)}
                max_atom_type = len(ordered_elements)

            f.write(f"{max_atom_type} atom types\n\n")

            f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n\n")

            f.write("Masses\n\n")
            if self.parameters.elements:
                for i, elem in enumerate(self.parameters.elements, 1):
                    mass = ELEMENT_MASSES.get(elem, 1.0)
                    f.write(f"{i} {mass:<10.4f} # {elem}\n")
            else:
                for elem, type_id in sorted(element_to_type.items(), key=lambda x: x[1]):
                    mass = ELEMENT_MASSES.get(elem, 1.0)
                    f.write(f"{type_id} {mass:<10.4f} # {elem}\n")
            f.write("\n")

            f.write("Atoms # atomic\n\n")
            for atom_id, (symbol, pos) in enumerate(zip(symbols, positions, strict=False), 1):
                atom_type = element_to_type[symbol]
                f.write(
                    f"{atom_id:6d} {atom_type:6d} "
                    f"{pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n"
                )

        if self.logger:
            self.logger.success("Successfully wrote LAMMPS atomic format for DPMD")
            self.logger.info(
                f"System: {n_metal} {self.parameters.metal} atoms, {n_water} water molecules, "
                f"{n_cations} {cation_element}, {n_anions} {anion_element}"
            )

    def run(self, save_artifacts: bool = False) -> str:
        """Run the interface generation (compatibility method)."""
        return self.generate()
