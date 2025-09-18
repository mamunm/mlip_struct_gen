"""Convert LAMMPS trajectory files to POSCAR files for VASP calculations."""

from pathlib import Path

from ase.io import read, write


class TrajectoryToPOSCAR:
    """Convert LAMMPS trajectory files to individual POSCAR files."""

    def __init__(
        self,
        trajectory_file: str | Path,
        output_dir: str | Path = "snapshots",
        prefix: str = "snapshot",
    ):
        """
        Initialize trajectory converter.

        Args:
            trajectory_file: Path to the LAMMPS trajectory file
            output_dir: Directory to store the snapshot folders
            prefix: Prefix for snapshot folder names

        Raises:
            FileNotFoundError: If trajectory file doesn't exist
        """
        self.trajectory_file = Path(trajectory_file)
        self.output_dir = Path(output_dir)
        self.prefix = prefix

        # Setup logger
        try:
            from ..utils.logger import MLIPLogger

            self.logger: MLIPLogger | None = MLIPLogger()
        except ImportError:
            self.logger = None

        if not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_file}")

    def _sort_atoms_by_element(self, atoms):
        """
        Sort atoms by element type for proper POSCAR format.
        Order: metals, O, H, cations, anions

        Args:
            atoms: ASE Atoms object

        Returns:
            ASE Atoms object with sorted atoms
        """
        from ase import Atoms
        from ase.data import atomic_numbers

        # Define element categories
        # Common FCC metals
        metals = {"Pt", "Au", "Ag", "Pd", "Cu", "Ni", "Rh", "Ir", "Al", "Pb", "Ca", "Sr", "Yb"}

        # Common cations (alkali and alkaline earth metals)
        cations = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}

        # Common anions
        anions = {"F", "Cl", "Br", "I"}

        # Get symbols and create sorting key
        symbols = atoms.get_chemical_symbols()
        indices = list(range(len(symbols)))

        def sort_key(idx):
            """Define sorting priority for atoms."""
            symbol = symbols[idx]

            # Priority order: 1=metal, 2=O, 3=H, 4=cation, 5=anion, 6=other
            if symbol in metals:
                # Sort metals by atomic number within category
                return (1, atomic_numbers[symbol])
            elif symbol == "O":
                return (2, 0)
            elif symbol == "H":
                return (3, 0)
            elif symbol in cations:
                # Sort cations by atomic number within category
                return (4, atomic_numbers[symbol])
            elif symbol in anions:
                # Sort anions by atomic number within category
                return (5, atomic_numbers[symbol])
            else:
                # Other elements sorted by atomic number at the end
                return (6, atomic_numbers[symbol])

        # Sort indices based on the key
        sorted_indices = sorted(indices, key=sort_key)

        # Create new atoms object with sorted order
        sorted_atoms = Atoms(
            symbols=[symbols[i] for i in sorted_indices],
            positions=atoms.positions[sorted_indices],
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

        # Log the element ordering if logger is available
        if self.logger:
            unique_symbols = []
            seen = set()
            for i in sorted_indices:
                symbol = symbols[i]
                if symbol not in seen:
                    unique_symbols.append(symbol)
                    seen.add(symbol)
            self.logger.debug(f"Element order in POSCAR: {' '.join(unique_symbols)}")

        return sorted_atoms

    def convert(self) -> int:
        """
        Convert LAMMPS trajectory to individual POSCAR files.

        Returns:
            Number of snapshots converted
        """
        if self.logger:
            self.logger.info("Starting trajectory to POSCAR conversion")
            self.logger.info(f"Reading trajectory from: {self.trajectory_file}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read all snapshots from trajectory
        if self.logger:
            self.logger.step("Reading trajectory snapshots")

        snapshots = read(str(self.trajectory_file), index=":")

        if self.logger:
            self.logger.info(f"Found {len(snapshots)} snapshots in trajectory")

        # Determine padding width based on number of snapshots
        padding_width = len(str(len(snapshots)))

        # Process each snapshot
        for i, atoms in enumerate(snapshots, 1):
            if self.logger:
                self.logger.step(f"Processing snapshot {i:0{padding_width}d}")

            # Create folder for this snapshot
            snapshot_dir = self.output_dir / f"{self.prefix}_{i:0{padding_width}d}"
            snapshot_dir.mkdir(exist_ok=True)

            # Sort atoms by element (metal, O, H, cation, anion order)
            sorted_atoms = self._sort_atoms_by_element(atoms)

            # Write POSCAR file
            poscar_file = snapshot_dir / "POSCAR"
            write(str(poscar_file), sorted_atoms, format="vasp")

            if self.logger:
                self.logger.debug(f"Written snapshot {i:0{padding_width}d} to {poscar_file}")

        if self.logger:
            self.logger.success(
                f"Conversion complete. {len(snapshots)} POSCAR files created in {self.output_dir}"
            )

        return len(snapshots)
