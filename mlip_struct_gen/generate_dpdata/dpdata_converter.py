"""Convert VASP OUTCARs to dpdata format with manual data handling to ensure consistency."""

import json
from pathlib import Path

import dpdata
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from mlip_struct_gen.utils.logger import get_logger


class CompositionData:
    """Store data for a unique composition."""

    def __init__(self, composition: str, type_map: list[str]):
        self.composition = composition
        self.type_map = type_map

        # Data storage - each element is from one OUTCAR
        self.coords: list[np.ndarray] = []  # List of (n_frames, 3*n_atoms) arrays
        self.energies: list[np.ndarray] = []  # List of (n_frames, 1) arrays
        self.forces: list[np.ndarray] = []  # List of (n_frames, 3*n_atoms) arrays
        self.cells: list[np.ndarray] = []  # List of (n_frames, 9) arrays
        self.virials: list[np.ndarray] = []  # List of (n_frames, 9) arrays

        # Atom information - should be same for all systems with this composition
        self.atom_types: np.ndarray | None = None  # Atom type indices
        self.atom_numbs: list[int] | None = None  # Number of each atom type
        self.n_atoms: int = 0
        self.total_frames: int = 0

        # Track frame counts for consistency
        self.frame_counts: list[int] = []

    def add_system(self, system_data: dict, source_path: Path) -> bool:
        """
        Add data from a parsed system to this composition.

        Args:
            system_data: Data dictionary from dpdata.LabeledSystem
            source_path: Path to source OUTCAR for logging

        Returns:
            True if successfully added, False otherwise
        """
        # Check frame consistency within this system
        n_frames = len(system_data["coords"])
        n_frames_energy = len(system_data["energies"])
        n_frames_force = len(system_data["forces"])
        n_frames_cell = len(system_data["cells"])

        if not (n_frames == n_frames_energy == n_frames_force == n_frames_cell):
            logger = get_logger()
            logger.warning(
                f"Frame inconsistency in {source_path}: "
                f"coords={n_frames}, energy={n_frames_energy}, "
                f"force={n_frames_force}, cell={n_frames_cell}"
            )
            return False

        # Get atom information
        n_atoms = sum(system_data["atom_numbs"])
        atom_types = system_data["atom_types"]

        # Initialize atom information on first system
        if self.atom_types is None:
            self.atom_types = atom_types
            self.atom_numbs = system_data["atom_numbs"]
            self.n_atoms = n_atoms
        else:
            # Verify consistency with previous systems
            if not np.array_equal(self.atom_types, atom_types):
                logger = get_logger()
                logger.warning(f"Atom type mismatch in {source_path}")
                return False

        # Reshape data to dpdata format
        coords_flat = system_data["coords"].reshape(n_frames, -1)  # (n_frames, 3*n_atoms)
        energies_reshaped = system_data["energies"].reshape(n_frames, 1)  # (n_frames, 1)
        forces_flat = system_data["forces"].reshape(n_frames, -1)  # (n_frames, 3*n_atoms)
        cells_flat = system_data["cells"].reshape(n_frames, 9)  # (n_frames, 9)

        # Add to storage
        self.coords.append(coords_flat)
        self.energies.append(energies_reshaped)
        self.forces.append(forces_flat)
        self.cells.append(cells_flat)

        # Handle virials if present
        if "virials" in system_data and system_data["virials"] is not None:
            virials_flat = system_data["virials"].reshape(n_frames, 9)
            self.virials.append(virials_flat)

        # Update counters
        self.frame_counts.append(n_frames)
        self.total_frames += n_frames

        return True

    def save_to_disk(self, output_dir: Path, verbose: bool = False) -> None:
        """
        Save this composition's data to disk in dpdata format.

        Args:
            output_dir: Base output directory
            verbose: Whether to print debug information
        """
        logger = get_logger()

        # Create directory structure
        comp_dir = output_dir / self.composition
        set_dir = comp_dir / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate all data
        if len(self.coords) == 0:
            logger.warning(f"No data to save for {self.composition}")
            return

        # Stack all frames
        all_coords = np.vstack(self.coords) if self.coords else np.empty((0, 3 * self.n_atoms))
        all_energies = np.vstack(self.energies) if self.energies else np.empty((0, 1))
        all_forces = np.vstack(self.forces) if self.forces else np.empty((0, 3 * self.n_atoms))
        all_cells = np.vstack(self.cells) if self.cells else np.empty((0, 9))

        # Verify shapes
        assert all_coords.shape == (self.total_frames, 3 * self.n_atoms)
        assert all_energies.shape == (self.total_frames, 1)
        assert all_forces.shape == (self.total_frames, 3 * self.n_atoms)
        assert all_cells.shape == (self.total_frames, 9)

        # Save npy files in set.000
        np.save(set_dir / "coord.npy", all_coords)
        np.save(set_dir / "energy.npy", all_energies)
        np.save(set_dir / "force.npy", all_forces)
        np.save(set_dir / "box.npy", all_cells)

        # Save virials if present
        if self.virials:
            all_virials = np.vstack(self.virials)
            assert all_virials.shape == (self.total_frames, 9)
            np.save(set_dir / "virial.npy", all_virials)

        # Write type_map.raw at composition level
        with open(comp_dir / "type_map.raw", "w") as f:
            for element in self.type_map:
                f.write(f"{element}\n")

        # Write type.raw - atom types for all atoms in order
        with open(comp_dir / "type.raw", "w") as f:
            for atom_type in self.atom_types:
                f.write(f"{atom_type}\n")

        if verbose:
            logger.info(
                f"  Saved {self.composition}: {self.total_frames} frames, "
                f"{self.n_atoms} atoms, {len(self.frame_counts)} systems"
            )


class DPDataConverter:
    """Convert VASP OUTCARs to dpdata format with manual data handling."""

    def __init__(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        type_map: list[str],
        recursive: bool = True,
        verbose: bool = False,
        save_file_loc: Path | str | None = None,
    ):
        """
        Initialize DPDataConverter.

        Args:
            input_dir: Directory containing VASP OUTCARs
            output_dir: Output directory for dpdata
            type_map: List of element symbols (e.g., ["Cu", "O", "H", "Na", "Cl"])
                     Only systems containing subsets of these elements will be processed
            recursive: Whether to search recursively for OUTCARs
            verbose: Whether to show debug messages
            save_file_loc: Optional file path to save OUTCAR locations
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.type_map = type_map
        self.recursive = recursive
        self.verbose = verbose
        self.save_file_loc = Path(save_file_loc) if save_file_loc else None
        self.logger = get_logger()

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        # Storage for composition data
        self.composition_data: dict[str, CompositionData] = {}

    def find_outcars(self) -> list[Path]:
        """
        Find all OUTCAR files in the input directory.

        Returns:
            List of paths to OUTCAR files (sorted)
        """
        if self.recursive:
            outcar_files = sorted(self.input_dir.rglob("OUTCAR"))
        else:
            outcar_files = sorted(self.input_dir.glob("*/OUTCAR"))

        self.logger.info(f"Found {len(outcar_files)} OUTCAR files")
        return outcar_files

    def check_elements_valid(self, system: dpdata.LabeledSystem) -> bool:
        """
        Check if all elements in the system are in the type_map.

        Args:
            system: dpdata LabeledSystem

        Returns:
            True if all elements are in type_map, False otherwise
        """
        try:
            system_elements = set(system.data.get("atom_names", []))
            allowed_elements = set(self.type_map)

            # Check if system elements are a subset of allowed elements
            is_valid = system_elements.issubset(allowed_elements)

            if not is_valid and self.verbose:
                self.logger.debug(
                    f"System has elements {system_elements} not all in {allowed_elements}"
                )

            return is_valid
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Error checking elements: {e}")
            return False

    def get_composition_string(self, system: dpdata.LabeledSystem) -> str:
        """
        Get composition string with zero-padding for missing elements.

        Args:
            system: dpdata LabeledSystem with type_map applied

        Returns:
            Composition string like "Cu48O32H64Na0Cl0"
        """
        data = system.data
        atom_numbs = data["atom_numbs"]

        # Build composition string using type_map order
        composition_parts = []
        for i, element in enumerate(self.type_map):
            count = atom_numbs[i] if i < len(atom_numbs) else 0
            composition_parts.append(f"{element}{count}")

        return "".join(composition_parts)

    def run(self) -> None:
        """Run the conversion process."""
        self.logger.step("Starting OUTCAR to dpdata conversion (Manual Mode)")
        self.logger.info(f"Type map: {self.type_map}")
        self.logger.info(f"Output directory: {self.output_dir}")

        if self.save_file_loc:
            self.logger.info(f"Saving OUTCAR locations to: {self.save_file_loc}")

        # Find all OUTCARs
        outcar_files = self.find_outcars()
        if not outcar_files:
            self.logger.error("No OUTCAR files found")
            return

        # Process counters
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        inconsistent_count = 0

        # Store processed OUTCAR locations
        processed_outcar_paths = []

        # Process all OUTCARs
        self.logger.step(f"Processing {len(outcar_files)} OUTCAR files")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing OUTCARs...", total=len(outcar_files))

            for outcar_path in outcar_files:
                progress.update(
                    task,
                    description=f"[cyan]Processing: {outcar_path.parent.name}/{outcar_path.name}",
                )

                try:
                    # Load system from OUTCAR
                    system = dpdata.LabeledSystem(str(outcar_path), fmt="vasp/outcar")

                    # Check if elements are valid (subset of type_map)
                    if not self.check_elements_valid(system):
                        skipped_count += 1
                        progress.advance(task)
                        continue

                    # Apply type_map to ensure consistent ordering
                    system.apply_type_map(self.type_map)

                    # Get composition string
                    composition = self.get_composition_string(system)

                    # Create or get composition data container
                    if composition not in self.composition_data:
                        self.composition_data[composition] = CompositionData(
                            composition, self.type_map
                        )

                    # Add system data to composition
                    success = self.composition_data[composition].add_system(
                        system.data, outcar_path
                    )

                    if success:
                        processed_count += 1
                        processed_outcar_paths.append(outcar_path.parent.resolve())
                    else:
                        inconsistent_count += 1
                        if self.verbose:
                            self.logger.debug(f"Skipped inconsistent system: {outcar_path}")

                except Exception as e:
                    if self.verbose:
                        self.logger.debug(f"Failed to process {outcar_path}: {e}")
                    failed_count += 1

                progress.advance(task)

        # Summary of processing
        self.logger.info("\nProcessing summary:")
        self.logger.info(f"  Processed: {processed_count} systems")
        self.logger.info(f"  Skipped (wrong elements): {skipped_count} systems")
        self.logger.info(f"  Skipped (inconsistent frames): {inconsistent_count} systems")
        self.logger.info(f"  Failed: {failed_count} systems")

        if processed_count == 0:
            self.logger.error("No systems were successfully processed")
            return

        # Save all composition data
        self.logger.step("Saving to dpdata format")
        self.logger.info(f"Found {len(self.composition_data)} unique compositions")

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save each composition
        total_frames = 0
        for _, comp_data in self.composition_data.items():
            comp_data.save_to_disk(self.output_dir, verbose=self.verbose)
            total_frames += comp_data.total_frames

        self.logger.success("\nConversion completed successfully!")
        self.logger.info(f"Output saved to: {self.output_dir}")

        # Show what was created
        self.logger.info("\nGenerated compositions:")
        for comp_dir in sorted(self.output_dir.iterdir()):
            if comp_dir.is_dir() and (comp_dir / "set.000").exists():
                # Get frame count from saved data
                coord_file = comp_dir / "set.000" / "coord.npy"
                if coord_file.exists():
                    n_frames = np.load(coord_file).shape[0]
                    type_raw = comp_dir / "type.raw"
                    if type_raw.exists():
                        with open(type_raw) as f:
                            n_atoms = len(f.readlines())
                    else:
                        n_atoms = "?"
                    self.logger.info(f"  {comp_dir.name}: {n_frames} frames, {n_atoms} atoms")

        # Save metadata
        metadata = {
            "type_map": self.type_map,
            "total_systems_processed": processed_count,
            "systems_skipped": skipped_count,
            "systems_failed": failed_count,
            "systems_inconsistent": inconsistent_count,
            "unique_compositions": len(self.composition_data),
            "total_frames": int(total_frames),  # Convert numpy int to Python int
            "compositions": {
                comp: {
                    "n_frames": int(comp_data.total_frames),  # Convert to Python int
                    "n_atoms": int(comp_data.n_atoms),  # Convert to Python int
                    "n_systems": len(comp_data.frame_counts),
                }
                for comp, comp_data in self.composition_data.items()
            },
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"\nMetadata saved to: {self.output_dir / 'metadata.json'}")

        # Save OUTCAR parent directory locations if requested
        if self.save_file_loc and processed_outcar_paths:
            with open(self.save_file_loc, "w") as f:
                for path in processed_outcar_paths:
                    f.write(str(path) + "\n")
            self.logger.info(f"OUTCAR directory locations saved to: {self.save_file_loc}")
            self.logger.info(f"  Total directories saved: {len(processed_outcar_paths)}")
