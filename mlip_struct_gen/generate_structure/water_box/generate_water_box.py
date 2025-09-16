"""Water box generation using Packmol."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..templates.water_models import create_water_xyz, get_water_density
from .input_parameters import WaterBoxGeneratorParameters
from .validation import validate_parameters


class WaterBoxGenerator:
    """Generate water boxes using Packmol."""

    def __init__(self, parameters: WaterBoxGeneratorParameters):
        """
        Initialize water box generator.

        Args:
            parameters: Generation parameters

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If packmol is not available
        """
        self.parameters = parameters
        validate_parameters(self.parameters)

        # Setup logger first
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from ...utils.logger import MLIPLogger
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

        # Compute box_size if not provided (after logger setup)
        if self.parameters.box_size is None:
            self._compute_box_size_from_molecules()

        if self.logger:
            self.logger.info("Initializing WaterBoxGenerator")
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
                text=True,
            )
            # Packmol exits with code 171 when given empty input - this is expected
            if result.returncode == 171:
                if self.logger:
                    self.logger.success(f"Packmol found: {self.parameters.packmol_executable}")
                    # Extract version from stderr if available
                    if "Version" in result.stderr:
                        version_line = [
                            line for line in result.stderr.split("\n") if "Version" in line
                        ]
                        if version_line:
                            self.logger.debug(f"Packmol {version_line[0].strip()}")
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

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate a water box using Packmol.

        Args:
            save_artifacts: If True, save intermediate files (packmol.inp, water.xyz)

        Returns:
            Path to generated water box file

        Raises:
            RuntimeError: If packmol execution fails
        """
        if self.logger:
            self.logger.info("Starting water box generation")

        # Calculate number of molecules if not provided
        n_water = self.parameters.n_water
        if n_water is None:
            if self.parameters.density is not None:
                # Use user-specified density
                if self.logger:
                    self.logger.info(f"Using custom density: {self.parameters.density} g/cm³")
                assert self.parameters.box_size is not None  # Type guard for mypy
                # After validation, box_size is always tuple[float, float, float]
                n_water = self._calculate_molecules_custom_density(
                    self.parameters.box_size,
                    self.parameters.density,  # type: ignore[arg-type]
                )
            else:
                # Use water model's default density
                model_density = get_water_density(self.parameters.water_model)
                if self.logger:
                    self.logger.info(
                        f"Using {self.parameters.water_model} default density: {model_density} g/cm³"
                    )
                assert self.parameters.box_size is not None  # Type guard for mypy
                # After validation, box_size is always tuple[float, float, float]
                n_water = self._calculate_molecules_custom_density(
                    self.parameters.box_size,
                    model_density,  # type: ignore[arg-type]
                )

        if self.logger:
            self.logger.info(f"Target molecules: {n_water}")

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
            # Create water molecule template (always XYZ format)
            if self.logger:
                self.logger.step("Creating water molecule template")
            water_xyz = work_dir / "water.xyz"
            create_water_xyz(self.parameters.water_model, str(water_xyz))

            # Create packmol input file (always XYZ output)
            if self.logger:
                self.logger.step("Generating Packmol input file")
            input_file = work_dir / "packmol.inp"
            temp_output = work_dir / "output.xyz"

            assert self.parameters.box_size is not None  # Type guard for mypy
            # After validation, box_size is always tuple[float, float, float]
            self._create_packmol_input(
                input_file,
                water_xyz,
                temp_output,
                self.parameters.box_size,  # type: ignore[arg-type]
                n_water,
                self.parameters.tolerance,
                self.parameters.seed,
            )

            # Run packmol (always generates XYZ)
            if self.logger:
                self.logger.step("Running Packmol to generate XYZ format")
            self._run_packmol(input_file, work_dir)

            # Handle output file based on format
            if self.parameters.output_format == "lammps":
                # Convert XYZ to LAMMPS format with bonds and angles
                if self.logger:
                    self.logger.step("Converting XYZ to LAMMPS data format with bonds and angles")
                self._convert_xyz_to_lammps_with_topology(temp_output, output_path)
            elif self.parameters.output_format == "poscar":
                # Convert XYZ to POSCAR format with sorted elements
                if self.logger:
                    self.logger.step("Converting XYZ to POSCAR format")
                self._convert_xyz_to_poscar(temp_output, output_path)
            else:
                # Copy the XYZ file directly
                if self.logger:
                    self.logger.step("Copying XYZ output to final location")
                shutil.copy2(temp_output, output_path)

        finally:
            if temp_context is not None:
                temp_context.cleanup()

        if self.logger:
            self.logger.success("Water box generation completed successfully")
            self.logger.info(f"Output saved to: {output_path}")

        return str(output_path)

    def _calculate_molecules_custom_density(
        self, box_size: tuple[float, float, float], density: float
    ) -> int:
        """Calculate number of molecules for custom density."""
        # Box volume in cm³
        volume_cm3 = (box_size[0] * box_size[1] * box_size[2]) * 1e-24

        # Water molar mass (g/mol)
        water_molar_mass = 18.015

        # Avogadro's number
        na = 6.022e23

        # Calculate number of molecules
        mass_g = density * volume_cm3
        moles = mass_g / water_molar_mass
        n_water = int(moles * na)

        return n_water

    def _compute_box_size_from_molecules(self) -> None:
        """Compute box_size from n_water and density."""
        if self.parameters.n_water is None:
            raise ValueError("n_water must be provided when box_size is None")

        # Determine density to use
        if self.parameters.density is not None:
            density = self.parameters.density
        else:
            # Use water model's default density
            density = get_water_density(self.parameters.water_model)

        # Calculate required volume
        water_molar_mass = 18.015  # g/mol
        na = 6.022e23  # Avogadro's number

        # Calculate mass and volume
        moles = self.parameters.n_water / na
        mass_g = moles * water_molar_mass
        volume_cm3 = mass_g / density
        volume_angstrom3 = volume_cm3 * 1e24

        # Calculate cubic box size
        box_size = volume_angstrom3 ** (1 / 3)

        # Set as cubic box
        self.parameters.box_size = (box_size, box_size, box_size)

        # Log the computed box size
        if self.logger:
            self.logger.info(
                f"Computed box size from {self.parameters.n_water} molecules: {box_size:.2f} Å (cubic)"
            )
            self.logger.info(f"Using density: {density:.3f} g/cm³")

        # Validate the computed box size
        if any(s > 1000.0 for s in self.parameters.box_size):
            raise ValueError(
                f"Computed box dimensions too large ({box_size:.1f} Å). Check n_water or density."
            )

        if any(s < 5.0 for s in self.parameters.box_size):
            raise ValueError(
                f"Computed box dimensions too small ({box_size:.1f} Å). Check n_water or density."
            )

    def _create_packmol_input(
        self,
        input_file: Path,
        water_xyz: Path,
        output_file: Path,
        box_size: tuple[float, float, float],
        n_water: int,
        tolerance: float,
        seed: int,
    ) -> None:
        """Create Packmol input file (always XYZ format)."""
        # Calculate box boundaries with tolerance buffer
        x_low = 0.5 * tolerance
        y_low = 0.5 * tolerance
        z_low = 0.5 * tolerance
        x_high = box_size[0] - 0.5 * tolerance
        y_high = box_size[1] - 0.5 * tolerance
        z_high = box_size[2] - 0.5 * tolerance

        with open(input_file, "w") as f:
            f.write(f"tolerance {tolerance}\n")
            f.write("filetype xyz\n")
            f.write(f"output {output_file.name}\n")  # Use relative path
            f.write(f"seed {seed}\n")
            f.write("\n")
            f.write(f"structure {water_xyz.name}\n")  # Use relative path
            f.write(f"  number {n_water}\n")
            f.write(f"  inside box {x_low} {y_low} {z_low} {x_high} {y_high} {z_high}\n")
            f.write("end structure\n")

    def _run_packmol(self, input_file: Path, work_dir: Path) -> None:
        """Run Packmol with the given input file."""
        try:
            if self.logger:
                self.logger.debug(f"Executing: {self.parameters.packmol_executable}")

            subprocess.run(
                [self.parameters.packmol_executable],
                stdin=open(input_file),
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
                self.logger.debug(f"stdout: {e.stdout}")
                self.logger.debug(f"stderr: {e.stderr}")

            raise RuntimeError(
                f"Packmol failed with return code {e.returncode}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e

    def estimate_box_size(
        self, n_water: int, aspect_ratio: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> tuple[float, float, float]:
        """
        Estimate box size needed for given number of water molecules.

        Args:
            n_water: Number of water molecules
            aspect_ratio: Relative box dimensions (x:y:z)

        Returns:
            Estimated box dimensions (x, y, z) in Angstroms
        """
        # Water molar mass (g/mol)
        water_molar_mass = 18.015

        # Avogadro's number
        na = 6.022e23

        # Density in g/cm³
        density = get_water_density(self.parameters.water_model)

        # Calculate required volume
        moles = n_water / na
        mass_g = moles * water_molar_mass
        volume_cm3 = mass_g / density
        volume_angstrom3 = volume_cm3 * 1e24

        # Calculate box dimensions with given aspect ratio
        ratio_product = aspect_ratio[0] * aspect_ratio[1] * aspect_ratio[2]
        scale = (volume_angstrom3 / ratio_product) ** (1 / 3)

        box_size = (scale * aspect_ratio[0], scale * aspect_ratio[1], scale * aspect_ratio[2])

        return box_size

    def _convert_xyz_to_lammps_with_topology(self, input_xyz: Path, output_file: Path) -> None:
        """
        Convert XYZ file to LAMMPS data format with bonds and angles.

        Args:
            input_xyz: Path to input XYZ file
            output_file: Path to output LAMMPS data file
        """
        # Read XYZ file manually to preserve water molecule ordering
        with open(input_xyz) as f:
            lines = f.readlines()

        n_atoms = int(lines[0].strip())
        n_water = n_atoms // 3  # Each water molecule has 3 atoms

        # Parse atom positions
        atoms = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, x, y, z))

        # Get water model charges
        from ..templates.water_models import get_water_model

        water_model = get_water_model(self.parameters.water_model)
        charges = {atom["element"]: atom["charge"] for atom in water_model["atoms"]}

        # Write LAMMPS data file
        with open(output_file, "w") as f:
            f.write(f"# LAMMPS data file for {self.parameters.water_model} water box\n")
            f.write("# Generated by mlip-struct-gen\n\n")

            # Counts section
            f.write(f"{n_atoms} atoms\n")
            f.write(f"{n_water * 2} bonds\n")  # 2 bonds per water molecule (O-H, O-H)
            f.write(f"{n_water} angles\n")  # 1 angle per water molecule (H-O-H)
            f.write("\n")

            # Types section
            f.write("2 atom types\n")
            f.write("1 bond types\n")
            f.write("1 angle types\n")
            f.write("\n")

            # Box dimensions
            assert self.parameters.box_size is not None  # Type guard for mypy
            # After validation, box_size is always tuple[float, float, float]
            box: tuple[float, float, float] = self.parameters.box_size  # type: ignore[assignment]
            f.write(f"0.0 {box[0]} xlo xhi\n")
            f.write(f"0.0 {box[1]} ylo yhi\n")
            f.write(f"0.0 {box[2]} zlo zhi\n")
            f.write("\n")

            # Masses section
            f.write("Masses\n\n")
            f.write("1 15.9994  # O\n")
            f.write("2 1.008    # H\n")
            f.write("\n")

            # Atoms section
            f.write("Atoms # full\n\n")
            atom_id = 1
            for mol_id in range(1, n_water + 1):
                for j in range(3):  # O, H, H for each molecule
                    element, x, y, z = atoms[(mol_id - 1) * 3 + j]
                    atom_type = 1 if element == "O" else 2
                    charge = charges[element]
                    f.write(
                        f"{atom_id:6d} {mol_id:6d} {atom_type:6d} {charge:10.6f} "
                        f"{x:12.6f} {y:12.6f} {z:12.6f}\n"
                    )
                    atom_id += 1
            f.write("\n")

            # Bonds section
            f.write("Bonds\n\n")
            bond_id = 1
            for mol_id in range(1, n_water + 1):
                base_atom = (mol_id - 1) * 3 + 1  # Oxygen atom ID
                # O-H1 bond
                f.write(f"{bond_id:6d} 1 {base_atom:6d} {base_atom + 1:6d}\n")
                bond_id += 1
                # O-H2 bond
                f.write(f"{bond_id:6d} 1 {base_atom:6d} {base_atom + 2:6d}\n")
                bond_id += 1
            f.write("\n")

            # Angles section
            f.write("Angles\n\n")
            angle_id = 1
            for mol_id in range(1, n_water + 1):
                base_atom = (mol_id - 1) * 3 + 1  # Oxygen atom ID
                # H-O-H angle
                f.write(f"{angle_id:6d} 1 {base_atom + 1:6d} {base_atom:6d} {base_atom + 2:6d}\n")
                angle_id += 1

        if self.logger:
            self.logger.success(
                f"Successfully converted to LAMMPS data format with {n_water * 2} bonds and {n_water} angles"
            )

    def _convert_xyz_to_poscar(self, input_xyz: Path, output_file: Path) -> None:
        """
        Convert XYZ file to POSCAR format with descending element ordering.

        Args:
            input_xyz: Path to input XYZ file
            output_file: Path to output POSCAR file
        """
        try:
            from ase import Atoms, io
        except ImportError:
            raise ImportError(
                "ASE is required for POSCAR format. Install with: pip install ase"
            ) from None

        # Read the XYZ file
        atoms = io.read(str(input_xyz))  # type: ignore[assignment]

        # Set the cell dimensions and PBC
        assert self.parameters.box_size is not None  # Type guard for mypy
        # After validation, box_size is always tuple[float, float, float]
        box: tuple[float, float, float] = self.parameters.box_size  # type: ignore[assignment]
        atoms.set_cell([box[0], box[1], box[2]])  # type: ignore
        atoms.set_pbc(True)  # type: ignore

        # Sort atoms by element in descending order (O before H)
        symbols = atoms.get_chemical_symbols()  # type: ignore
        positions = atoms.get_positions()  # type: ignore

        # Create a list of tuples (symbol, position) and sort in descending order
        atom_data = [(s, p) for s, p in zip(symbols, positions, strict=False)]
        atom_data.sort(key=lambda x: x[0], reverse=True)  # O comes before H in descending order

        # Extract sorted symbols and positions
        sorted_symbols = [s for s, _ in atom_data]
        sorted_positions = [p for _, p in atom_data]

        # Create new atoms object with sorted atoms
        sorted_atoms = Atoms(
            symbols=sorted_symbols, positions=sorted_positions, cell=atoms.cell, pbc=True
        )  # type: ignore

        # Write as POSCAR with proper formatting
        io.write(str(output_file), sorted_atoms, format="vasp", direct=False, sort=False)

        # Count O and H atoms
        n_oxygen = sum(1 for s in sorted_symbols if s == "O")
        n_hydrogen = sum(1 for s in sorted_symbols if s == "H")

        if self.logger:
            self.logger.success("Successfully converted to POSCAR format")
            self.logger.info(f"System: {n_oxygen} O atoms, {n_hydrogen} H atoms")
