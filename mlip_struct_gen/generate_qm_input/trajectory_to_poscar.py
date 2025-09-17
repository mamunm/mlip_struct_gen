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
        Groups atoms by element with H first, then O.

        Args:
            atoms: ASE Atoms object

        Returns:
            ASE Atoms object with sorted atoms
        """
        import numpy as np
        from ase import Atoms

        # Get indices sorted by element (H=1, O=8, so H comes first naturally)
        symbols = atoms.get_chemical_symbols()
        sorted_indices = np.argsort(symbols)

        # Create new atoms object with sorted order
        sorted_atoms = Atoms(
            symbols=[symbols[i] for i in sorted_indices],
            positions=atoms.positions[sorted_indices],
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

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

            # Sort atoms by element (H first, then O for water)
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
