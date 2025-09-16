"""Metal FCC(111) surface generation using ASE."""

from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    from ase.io import write
except ImportError:
    raise ImportError(
        "ASE (Atomic Simulation Environment) is required for metal surface generation. "
        "Install with: pip install ase"
    )

from .input_parameters import MetalSurfaceParameters
from .validation import get_lattice_constant, validate_parameters


class MetalSurfaceGenerator:
    """Generate FCC(111) metal surfaces using ASE."""

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

        if self.logger:
            self.logger.info("Initializing MetalSurfaceGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Size: {self.parameters.size}")
            self.logger.info(f"Vacuum: {self.parameters.vacuum} Å")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.3f} Å")

    def generate(self) -> str:
        """
        Generate the metal surface.

        Returns:
            Path to the output file

        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Create the FCC(111) surface
            nx, ny, nz = self.parameters.size

            if self.logger:
                self.logger.info(f"Creating {self.parameters.metal}(111) surface")
                self.logger.info(f"  Dimensions: {nx}x{ny} unit cells, {nz} layers")

            # Build the surface using ASE's fcc111 function
            # orthogonal=True ensures LAMMPS compatibility
            slab = fcc111(
                self.parameters.metal,
                size=(nx, ny, nz),
                a=self.lattice_constant,
                orthogonal=self.parameters.orthogonalize,
                vacuum=self.parameters.vacuum,
                periodic=True,
            )

            if self.logger:
                self.logger.info(f"Created surface with {len(slab)} atoms")
                cell = slab.get_cell()
                self.logger.info(
                    f"Cell dimensions: {cell[0,0]:.2f} x {cell[1,1]:.2f} x {cell[2,2]:.2f} Å"
                )

            # Apply constraints to fix bottom layers if requested
            if self.parameters.fix_bottom_layers > 0:
                self._apply_constraints(slab)

            # Write the output file
            output_path = self._write_output(slab)

            if self.logger:
                self.logger.info(f"Successfully generated: {output_path}")

            return str(output_path)

        except Exception as e:
            error_msg = f"Failed to generate metal surface: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _apply_constraints(self, slab: Atoms) -> None:
        """
        Apply constraints to fix bottom layers.

        Args:
            slab: The atoms object to apply constraints to
        """
        if self.parameters.fix_bottom_layers == 0:
            return

        positions = slab.get_positions()
        z_positions = positions[:, 2]

        # Find unique z-layers
        z_unique = np.unique(np.round(z_positions, decimals=2))
        z_unique.sort()

        if len(z_unique) < self.parameters.fix_bottom_layers:
            if self.logger:
                self.logger.warning(
                    f"Only {len(z_unique)} layers found, but requested to fix "
                    f"{self.parameters.fix_bottom_layers} layers. Fixing all but top layer."
                )
            n_fix = len(z_unique) - 1
        else:
            n_fix = self.parameters.fix_bottom_layers

        # Get z-threshold for fixed layers
        z_threshold = z_unique[n_fix] if n_fix < len(z_unique) else z_unique[-1]

        # Create mask for fixed atoms
        fixed_mask = z_positions < z_threshold + 0.01

        # Apply constraints
        constraint = FixAtoms(mask=fixed_mask)
        slab.set_constraint(constraint)

        n_fixed = np.sum(fixed_mask)
        if self.logger:
            self.logger.info(f"Fixed {n_fixed} atoms in bottom {n_fix} layers")

    def _write_output(self, slab: Atoms) -> Path:
        """
        Write the slab to the output file.

        Args:
            slab: The atoms object to write

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
            self._write_lammps(slab, output_path)
        elif output_format in ["vasp", "poscar"]:
            self._write_poscar(slab, output_path)
        else:  # xyz
            write(str(output_path), slab, format="xyz")

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
            return format_map.get(self.parameters.output_format.lower(), "xyz")

        # Infer from file extension
        output_path = Path(self.parameters.output_file)
        suffix = output_path.suffix.lower()

        if suffix in [".vasp", ".poscar"] or output_path.name.upper() == "POSCAR":
            return "poscar"
        elif suffix in [".lammps", ".data"]:
            return "lammps"
        else:
            return "xyz"

    def _write_poscar(self, slab: Atoms, output_path: Path) -> None:
        """
        Write VASP POSCAR format.

        Args:
            slab: Atoms object to write
            output_path: Output file path
        """
        write(str(output_path), slab, format="vasp", direct=False, vasp5=True)

    def _write_lammps(self, slab: Atoms, output_path: Path) -> None:
        """
        Write LAMMPS data file format.

        Args:
            slab: Atoms object to write
            output_path: Output file path
        """
        # Get atomic data
        positions = slab.get_positions()
        cell = slab.get_cell()
        symbols = slab.get_chemical_symbols()
        metal = self.parameters.metal

        # Count atoms
        n_atoms = len(positions)

        # Get atomic mass
        from ase.data import atomic_masses, atomic_numbers

        atomic_number = atomic_numbers[metal]
        atomic_mass = atomic_masses[atomic_number]

        with open(output_path, "w") as f:
            # Header
            f.write(f"LAMMPS data file for {metal}(111) surface\n\n")

            # Counts
            f.write(f"{n_atoms} atoms\n")
            f.write("0 bonds\n")
            f.write("0 angles\n")
            f.write("0 dihedrals\n")
            f.write("0 impropers\n\n")

            # Types
            f.write("1 atom types\n\n")

            # Box dimensions
            f.write(f"0.0 {cell[0,0]:.6f} xlo xhi\n")
            f.write(f"0.0 {cell[1,1]:.6f} ylo yhi\n")
            f.write(f"0.0 {cell[2,2]:.6f} zlo zhi\n")

            # Check for non-orthogonal cell
            if not self.parameters.orthogonalize:
                # Add tilt factors if cell is not orthogonal
                xy = cell[1, 0]
                xz = cell[2, 0]
                yz = cell[2, 1]
                if abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6:
                    f.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n")
            f.write("\n")

            # Masses
            f.write("Masses\n\n")
            f.write(f"1 {atomic_mass:.4f}  # {metal}\n\n")

            # Atoms
            f.write("Atoms\n\n")
            for i in range(n_atoms):
                # atom_id mol_id atom_type charge x y z
                f.write(
                    f"{i+1} 1 1 0.0 {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n"
                )

    def run(self, save_artifacts: bool = False) -> str:
        """
        Run the surface generation (compatibility method).

        Args:
            save_artifacts: Whether to save intermediate files (not used)

        Returns:
            Path to the output file
        """
        return self.generate()
