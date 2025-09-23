"""Salt water box generation using Packmol."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..templates.salt_models import create_ion_xyz, get_salt_model, get_salt_stoichiometry
from ..templates.water_models import create_water_xyz, get_water_density, get_water_model
from .input_parameters import SaltWaterBoxGeneratorParameters
from .validation import validate_parameters

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
    "Fe": 55.845,
}

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


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

        # Get salt model and calculate ion numbers
        self.salt_model = get_salt_model(self.parameters.salt_type)
        self._calculate_ion_numbers()

        # Compute box_size if not provided
        if self.parameters.box_size is None:
            self._compute_box_size_from_molecules()

        if self.logger:
            self.logger.info("Initializing SaltWaterBoxGenerator")
            self.logger.info(f"Water model: {self.parameters.water_model}")
            self.logger.info(
                f"Salt: {self.salt_model['name']} ({self.parameters.n_salt} formula units)"
            )
            self.logger.info(f"Ions: {self.n_cations} cations, {self.n_anions} anions")
            self.logger.info(f"Box size: {self.parameters.box_size}")

        self._check_packmol()

    def _calculate_ion_numbers(self) -> None:
        """Calculate number of cations and anions from salt formula units."""
        n_cation_per_formula, n_anion_per_formula = get_salt_stoichiometry(
            self.parameters.salt_type
        )
        self.n_cations = self.parameters.n_salt * n_cation_per_formula
        self.n_anions = self.parameters.n_salt * n_anion_per_formula

    def _check_packmol(self) -> None:
        """Check if packmol is available."""
        if self.logger:
            self.logger.step("Checking Packmol availability")

        try:
            result = subprocess.run(
                [self.parameters.packmol_executable],
                input="",
                capture_output=True,
                timeout=5,
                text=True,
            )
            # Packmol exits with code 171 when given empty input
            if result.returncode == 171:
                if self.logger:
                    self.logger.success(f"Packmol found: {self.parameters.packmol_executable}")
            else:
                if self.logger:
                    self.logger.warning(
                        f"Packmol returned unexpected exit code: {result.returncode}"
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
                    f"Packmol executable '{self.parameters.packmol_executable}' not found"
                )
            raise RuntimeError(
                f"Packmol executable '{self.parameters.packmol_executable}' not found. "
                "Please install packmol:\n"
                "  conda install -c conda-forge packmol\n"
                "Or compile from source: https://github.com/m3g/packmol"
            ) from None

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate a salt water box using Packmol.

        Args:
            save_artifacts: If True, save intermediate files

        Returns:
            Path to generated salt water box file

        Raises:
            RuntimeError: If packmol execution fails
        """
        if self.logger:
            self.logger.info("Starting salt water box generation")

        # Calculate water molecules
        n_water = self._calculate_water_molecules()

        if self.logger:
            self.logger.info(
                f"Target molecules: {n_water} water, {self.n_cations} cations, {self.n_anions} anions"
            )

        # Create output directory
        output_path = Path(self.parameters.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup working directory
        if save_artifacts:
            work_dir = Path("artifacts")
            work_dir.mkdir(exist_ok=True)
            temp_context = None
            if self.logger:
                self.logger.info(f"Saving artifacts to: {work_dir}")
        else:
            temp_context = tempfile.TemporaryDirectory()
            work_dir = Path(temp_context.name)

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

            create_ion_xyz(self.salt_model["cation"], [0.0, 0.0, 0.0], str(cation_xyz))
            create_ion_xyz(self.salt_model["anion"], [0.0, 0.0, 0.0], str(anion_xyz))

            # Create packmol input
            if self.logger:
                self.logger.step("Generating Packmol input file")

            input_file = work_dir / "packmol.inp"
            temp_output = work_dir / "output.xyz"

            self._create_packmol_input(
                input_file,
                water_xyz,
                cation_xyz,
                anion_xyz,
                temp_output,
                n_water,
                self.n_cations,
                self.n_anions,
            )

            # Run packmol
            if self.logger:
                self.logger.step("Running Packmol to generate XYZ format")

            self._run_packmol(input_file, work_dir)

            # Convert to desired output format
            if self.parameters.output_format == "lammps":
                if self.logger:
                    self.logger.step("Converting XYZ to LAMMPS data format")
                self._convert_xyz_to_lammps(temp_output, output_path, n_water)
            elif self.parameters.output_format == "poscar":
                if self.logger:
                    self.logger.step("Converting XYZ to POSCAR format")
                self._convert_xyz_to_poscar(temp_output, output_path)
            else:
                # Just copy XYZ
                shutil.copy2(temp_output, output_path)

        finally:
            if temp_context is not None:
                temp_context.cleanup()

        if self.logger:
            self.logger.success("Salt water box generation completed successfully")
            self.logger.info(f"Output saved to: {output_path}")

        return str(output_path)

    def _calculate_water_molecules(self) -> int:
        """Calculate number of water molecules based on parameters."""
        # If explicitly specified, use that
        if self.parameters.n_water is not None:
            return self.parameters.n_water

        # Otherwise calculate from box size and density
        assert self.parameters.box_size is not None  # Validated earlier
        box_volume = np.prod(self.parameters.box_size)

        # Account for ion volume if requested
        if self.parameters.include_salt_volume and self.parameters.n_salt > 0:
            # Calculate ion volumes using VDW radii
            cation_radius = self.salt_model["cation"]["vdw_radius"]
            anion_radius = self.salt_model["anion"]["vdw_radius"]

            cation_volume = (4 / 3) * np.pi * cation_radius**3
            anion_volume = (4 / 3) * np.pi * anion_radius**3

            total_ion_volume = self.n_cations * cation_volume + self.n_anions * anion_volume
            available_volume = box_volume - total_ion_volume

            if self.logger:
                self.logger.info(
                    f"Ion volume: {total_ion_volume:.1f} Ų ({100*total_ion_volume/box_volume:.1f}% of box)"
                )
        else:
            available_volume = box_volume

        # Use specified or default density
        if self.parameters.density is not None:
            density = self.parameters.density
        else:
            density = get_water_density(self.parameters.water_model)

        # Calculate water molecules
        volume_cm3 = available_volume * 1e-24
        water_molar_mass = 18.015
        na = 6.022e23

        mass_g = density * volume_cm3
        moles = mass_g / water_molar_mass
        n_molecules = int(moles * na)

        return max(1, n_molecules)

    def _compute_box_size_from_molecules(self) -> None:
        """Compute box size from n_water and density."""
        if self.parameters.n_water is None:
            raise ValueError("n_water must be provided when box_size is None")

        # Determine density
        if self.parameters.density is not None:
            density = self.parameters.density
        else:
            density = get_water_density(self.parameters.water_model)

        # Calculate water volume
        water_molar_mass = 18.015
        na = 6.022e23

        moles = self.parameters.n_water / na
        mass_g = moles * water_molar_mass
        water_volume_cm3 = mass_g / density
        water_volume_angstrom3 = water_volume_cm3 * 1e24

        # Add ion volume if requested
        total_volume = water_volume_angstrom3
        if self.parameters.include_salt_volume and self.parameters.n_salt > 0:
            cation_radius = self.salt_model["cation"]["vdw_radius"]
            anion_radius = self.salt_model["anion"]["vdw_radius"]

            cation_volume = (4 / 3) * np.pi * cation_radius**3
            anion_volume = (4 / 3) * np.pi * anion_radius**3

            total_ion_volume = self.n_cations * cation_volume + self.n_anions * anion_volume
            total_volume += total_ion_volume

        # Calculate cubic box size
        box_size = total_volume ** (1 / 3)

        # Set as cubic box
        self.parameters.box_size = (box_size, box_size, box_size)

        if self.logger:
            self.logger.info(f"Computed box size: {box_size:.2f} Å (cubic)")

    def _create_packmol_input(
        self,
        input_file: Path,
        water_xyz: Path,
        cation_xyz: Path,
        anion_xyz: Path,
        output_file: Path,
        n_water: int,
        n_cations: int,
        n_anions: int,
    ) -> None:
        """Create Packmol input file."""
        assert self.parameters.box_size is not None
        box = self.parameters.box_size
        tol = self.parameters.tolerance

        # Calculate boundaries with tolerance buffer
        x_low = 0.5 * tol
        y_low = 0.5 * tol
        z_low = 0.5 * tol
        x_high = box[0] - 0.5 * tol
        y_high = box[1] - 0.5 * tol
        z_high = box[2] - 0.5 * tol

        with open(input_file, "w") as f:
            f.write(f"tolerance {tol}\n")
            f.write("filetype xyz\n")
            f.write(f"output {output_file.name}\n")
            f.write(f"seed {self.parameters.seed}\n")
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
                self.logger.success("Packmol execution completed successfully")

        except subprocess.CalledProcessError as e:
            if self.logger:
                self.logger.error(f"Packmol failed with return code {e.returncode}")
            raise RuntimeError(
                f"Packmol failed with return code {e.returncode}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e

    def _convert_xyz_to_lammps(self, input_xyz: Path, output_file: Path, n_water: int) -> None:
        """Convert XYZ to LAMMPS data format with bonds and angles for water."""
        # Read XYZ file
        with open(input_xyz) as f:
            lines = f.readlines()

        n_atoms = int(lines[0].strip())

        # Parse atoms
        atoms = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, x, y, z))

        # Get models
        water_model = get_water_model(self.parameters.water_model)
        water_charges = {atom["element"]: atom["charge"] for atom in water_model["atoms"]}

        cation_charge = self.salt_model["cation"]["charge"]
        anion_charge = self.salt_model["anion"]["charge"]
        cation_element = self.salt_model["cation"]["element"]
        anion_element = self.salt_model["anion"]["element"]

        # Determine atom types
        if self.parameters.elements:
            # Use predefined element order
            atom_types = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}
            # Ensure all atoms in the structure have types
            max_type = len(self.parameters.elements)
            for atom in water_model["atoms"]:
                elem = atom["element"]
                if elem not in atom_types:
                    max_type += 1
                    atom_types[elem] = max_type
            if cation_element not in atom_types:
                max_type += 1
                atom_types[cation_element] = max_type
            if anion_element not in atom_types:
                max_type += 1
                atom_types[anion_element] = max_type
        else:
            # Sequential numbering
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
        water_atoms_per_molecule = len(water_model["atoms"])
        n_bonds = n_water * 2  # 2 O-H bonds per water
        n_angles = n_water  # 1 H-O-H angle per water

        # Write LAMMPS data file
        assert self.parameters.box_size is not None
        box = self.parameters.box_size

        with open(output_file, "w") as f:
            f.write(
                f"# LAMMPS data file for {self.salt_model['name']} + {self.parameters.water_model} water\n"
            )
            f.write("# Generated by mlip-struct-gen\n\n")

            # Counts
            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{n_angles} angles\n")
            f.write("\n")

            # Types
            f.write(f"{len(atom_types)} atom types\n")
            f.write("1 bond types\n")
            f.write("1 angle types\n")
            f.write("\n")

            # Box
            f.write(f"0.0 {box[0]} xlo xhi\n")
            f.write(f"0.0 {box[1]} ylo yhi\n")
            f.write(f"0.0 {box[2]} zlo zhi\n")
            f.write("\n")

            # Masses
            f.write("Masses\n\n")
            if self.parameters.elements:
                # Write masses for all defined elements
                for i, elem in enumerate(self.parameters.elements, 1):
                    if elem in ELEMENT_MASSES:
                        mass = ELEMENT_MASSES[elem]
                    elif elem == cation_element:
                        mass = self.salt_model["cation"]["mass"]
                    elif elem == anion_element:
                        mass = self.salt_model["anion"]["mass"]
                    else:
                        mass = 1.0  # Default
                    f.write(f"{i} {mass:<10.4f} # {elem}\n")
                # Add any additional types not in elements list
                for element, tid in sorted(atom_types.items(), key=lambda x: x[1]):
                    if tid > len(self.parameters.elements):
                        if element in ELEMENT_MASSES:
                            mass = ELEMENT_MASSES[element]
                        elif element == cation_element:
                            mass = self.salt_model["cation"]["mass"]
                        elif element == anion_element:
                            mass = self.salt_model["anion"]["mass"]
                        else:
                            mass = 1.0
                        f.write(f"{tid} {mass:<10.4f} # {element}\n")
            else:
                # Sequential numbering
                for element, tid in sorted(atom_types.items(), key=lambda x: x[1]):
                    if element in ELEMENT_MASSES:
                        mass = ELEMENT_MASSES[element]
                    elif element == cation_element:
                        mass = self.salt_model["cation"]["mass"]
                    elif element == anion_element:
                        mass = self.salt_model["anion"]["mass"]
                    else:
                        mass = 1.0  # Default
                    f.write(f"{tid} {mass}  # {element}\n")
            f.write("\n")

            # Atoms
            f.write("Atoms # full\n\n")

            atom_id = 1
            mol_id = 1

            # Water molecules first
            for i in range(n_water):
                for j in range(water_atoms_per_molecule):
                    element, x, y, z = atoms[i * water_atoms_per_molecule + j]
                    atom_type = atom_types[element]
                    charge = water_charges[element]
                    f.write(
                        f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {charge:10.6f} "
                        f"{x:12.6f} {y:12.6f} {z:12.6f}\n"
                    )
                    atom_id += 1
                mol_id += 1

            # Cations
            water_atoms_total = n_water * water_atoms_per_molecule
            for i in range(self.n_cations):
                element, x, y, z = atoms[water_atoms_total + i]
                atom_type = atom_types[cation_element]
                f.write(
                    f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {cation_charge:10.6f} "
                    f"{x:12.6f} {y:12.6f} {z:12.6f}\n"
                )
                atom_id += 1
                mol_id += 1

            # Anions
            for i in range(self.n_anions):
                element, x, y, z = atoms[water_atoms_total + self.n_cations + i]
                atom_type = atom_types[anion_element]
                f.write(
                    f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {anion_charge:10.6f} "
                    f"{x:12.6f} {y:12.6f} {z:12.6f}\n"
                )
                atom_id += 1
                mol_id += 1

            f.write("\n")

            # Bonds (only for water)
            if n_bonds > 0:
                f.write("Bonds\n\n")
                bond_id = 1
                for mol in range(n_water):
                    base_atom = mol * water_atoms_per_molecule + 1
                    # O-H bonds
                    for h in range(1, water_atoms_per_molecule):
                        if atoms[(base_atom - 1) + h][0] == "H":
                            f.write(f"{bond_id:6d} 1 {base_atom:6d} {base_atom + h:6d}\n")
                            bond_id += 1
                f.write("\n")

            # Angles (only for water)
            if n_angles > 0:
                f.write("Angles\n\n")
                angle_id = 1
                for mol in range(n_water):
                    base_atom = mol * water_atoms_per_molecule + 1
                    # H-O-H angle
                    h_atoms = []
                    for h in range(1, water_atoms_per_molecule):
                        if atoms[(base_atom - 1) + h][0] == "H":
                            h_atoms.append(base_atom + h)

                    if len(h_atoms) >= 2:
                        f.write(f"{angle_id:6d} 1 {h_atoms[0]:6d} {base_atom:6d} {h_atoms[1]:6d}\n")
                        angle_id += 1

        if self.logger:
            self.logger.success("Successfully converted to LAMMPS data format")
            self.logger.info(
                f"System: {n_water} water, {self.n_cations} {cation_element}, {self.n_anions} {anion_element}"
            )

    def _convert_xyz_to_poscar(self, input_xyz: Path, output_file: Path) -> None:
        """Convert XYZ to POSCAR format."""
        try:
            from ase import Atoms, io
        except ImportError:
            raise ImportError(
                "ASE is required for POSCAR format. Install with: pip install ase"
            ) from None

        # Read XYZ
        atoms = io.read(str(input_xyz))  # type: ignore[assignment]

        # Set cell and PBC
        assert self.parameters.box_size is not None
        box = self.parameters.box_size
        atoms.set_cell([box[0], box[1], box[2]])  # type: ignore
        atoms.set_pbc(True)  # type: ignore

        # Sort atoms by element
        symbols = atoms.get_chemical_symbols()  # type: ignore
        positions = atoms.get_positions()  # type: ignore

        atom_data = list(zip(symbols, positions, strict=False))
        atom_data.sort(key=lambda x: x[0], reverse=True)

        sorted_symbols = [s for s, _ in atom_data]
        sorted_positions = [p for _, p in atom_data]

        sorted_atoms = Atoms(
            symbols=sorted_symbols, positions=sorted_positions, cell=atoms.cell, pbc=True
        )  # type: ignore

        # Write POSCAR
        io.write(str(output_file), sorted_atoms, format="vasp", direct=False, sort=False)

        if self.logger:
            self.logger.success("Successfully converted to POSCAR format")
