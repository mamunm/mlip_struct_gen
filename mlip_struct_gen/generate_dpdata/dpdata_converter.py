"""Convert VASP OUTCARs to dpdata format using dpdata MultiSystems."""

from pathlib import Path

import dpdata
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from mlip_struct_gen.utils.logger import get_logger


class DPDataConverter:
    """Convert VASP OUTCARs to dpdata format using dpdata MultiSystems."""

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

    def find_outcars(self) -> list[Path]:
        """
        Find all OUTCAR files in the input directory.

        Returns:
            List of paths to OUTCAR files
        """
        if self.recursive:
            outcar_files = list(self.input_dir.rglob("OUTCAR"))
        else:
            outcar_files = list(self.input_dir.glob("*/OUTCAR"))

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

    def run(self) -> None:
        """Run the conversion process."""
        self.logger.step("Starting OUTCAR to dpdata conversion")
        self.logger.info(f"Type map: {self.type_map}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.save_file_loc:
            self.logger.info(f"Saving OUTCAR locations to: {self.save_file_loc}")

        # Find all OUTCARs
        outcar_files = self.find_outcars()
        if not outcar_files:
            self.logger.error("No OUTCAR files found")
            return

        # Initialize MultiSystems
        ms = dpdata.MultiSystems()

        # Process counters
        processed_count = 0
        skipped_count = 0
        failed_count = 0

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

                    # Add to MultiSystems
                    ms.append(system)
                    processed_count += 1

                    # Store the parent directory containing the OUTCAR
                    processed_outcar_paths.append(outcar_path.parent)

                except Exception as e:
                    if self.verbose:
                        self.logger.debug(f"Failed to process {outcar_path}: {e}")
                    failed_count += 1

                progress.advance(task)

        # Summary
        self.logger.info("\nProcessing summary:")
        self.logger.info(f"  Processed: {processed_count} systems")
        self.logger.info(f"  Skipped (wrong elements): {skipped_count} systems")
        self.logger.info(f"  Failed: {failed_count} systems")

        if processed_count == 0:
            self.logger.error("No systems were successfully processed")
            return

        # Save MultiSystems
        self.logger.step("Saving to dpdata format")
        self.logger.info(
            f"MultiSystems contains {len(ms.systems)} unique compositions with {sum(len(s) for s in ms.systems)} total frames"
        )

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save in deepmd/npy format
        ms.to_deepmd_npy(str(self.output_dir))

        self.logger.success("\nConversion completed successfully!")
        self.logger.info(f"Output saved to: {self.output_dir}")

        # Show what was created
        self.logger.info("\nGenerated compositions:")
        for comp_dir in sorted(self.output_dir.iterdir()):
            if comp_dir.is_dir():
                # Try to get frame count
                try:
                    test_sys = dpdata.LabeledSystem(str(comp_dir), fmt="deepmd/npy")
                    n_frames = len(test_sys)
                    n_atoms = test_sys.get_natoms()
                    self.logger.info(f"  {comp_dir.name}: {n_frames} frames, {n_atoms} atoms")
                except Exception:
                    self.logger.info(f"  {comp_dir.name}/")

        # Save metadata
        import json

        metadata = {
            "type_map": self.type_map,
            "total_systems_processed": processed_count,
            "systems_skipped": skipped_count,
            "systems_failed": failed_count,
            "unique_compositions": len(ms.systems),
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
