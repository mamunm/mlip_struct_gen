"""Convert Wannier center outputs to dpdata format with atomic dipole properties."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from mlip_struct_gen.utils.logger import get_logger


class WCDPDataConverter:
    """Convert Wannier center outputs (wc_out.npy) to dpdata format with atomic dipoles."""

    def __init__(
        self,
        input_file_loc: Path | str,
        output_dir: Path | str,
        type_map: list[str] | None = None,
        verbose: bool = False,
    ):
        """
        Initialize WCDPDataConverter.

        Args:
            input_file_loc: Path to text file containing directories with wc_out.npy files
            output_dir: Output directory for dpdata (e.g., DATA/water)
            type_map: Optional list of element symbols (e.g., ["Pt", "O", "H"])
                     Used to handle missing elements - if POSCAR has O2H4 and type_map
                     is ["Pt", "O", "H"], it will create Pt0O2H4 with zero Pt atoms
            verbose: Whether to show debug messages
        """
        self.input_file_loc = Path(input_file_loc)
        self.output_dir = Path(output_dir)
        self.type_map = type_map
        self.verbose = verbose
        self.logger = get_logger()

        if not self.input_file_loc.exists():
            raise ValueError(f"Input file does not exist: {self.input_file_loc}")

    def read_directory_list(self) -> list[Path]:
        """
        Read the list of directories from the input file.

        Returns:
            List of directory paths containing wc_out.npy files
        """
        directories = []
        with open(self.input_file_loc) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    dir_path = Path(line)
                    if dir_path.exists():
                        directories.append(dir_path)
                    else:
                        self.logger.warning(f"Directory not found: {dir_path}")

        self.logger.info(f"Found {len(directories)} valid directories")
        return directories

    def load_wannier_data(self, directory: Path) -> np.ndarray | None:
        """
        Load wannier center data from wc_out.npy file.

        Args:
            directory: Directory containing wc_out.npy file

        Returns:
            Loaded numpy array or None if failed
        """
        wc_file = directory / "wc_out.npy"
        if not wc_file.exists():
            if self.verbose:
                self.logger.debug(f"wc_out.npy not found in {directory}")
            return None

        try:
            data = np.load(wc_file, allow_pickle=True)
            return data
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Failed to load {wc_file}: {e}")
            return None

    def extract_composition_from_poscar(
        self, directory: Path
    ) -> tuple[dict[str, int], list[str]] | None:
        """
        Extract atomic composition from POSCAR file.

        Args:
            directory: Directory containing POSCAR file

        Returns:
            Tuple of (element counts dict, ordered element list) or None if failed
        """
        poscar_file = directory / "POSCAR"
        if not poscar_file.exists():
            if self.verbose:
                self.logger.debug(f"POSCAR not found in {directory}")
            return None

        try:
            atoms = read(poscar_file)
            symbols = atoms.get_chemical_symbols()

            # Count elements while preserving order
            element_counts = Counter(symbols)

            # Get unique elements in order of first appearance
            seen = set()
            ordered_elements = []
            for symbol in symbols:
                if symbol not in seen:
                    ordered_elements.append(symbol)
                    seen.add(symbol)

            return element_counts, ordered_elements
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Failed to read POSCAR from {directory}: {e}")
            return None

    def get_composition_string_ordered(
        self,
        element_counts: dict[str, int],
        ordered_elements: list[str],
        type_map: list[str] | None = None,
    ) -> str:
        """
        Generate composition string with proper formatting based on element order.

        Args:
            element_counts: Dictionary of element counts
            ordered_elements: List of elements in order from POSCAR
            type_map: Optional type map to include additional elements with count 0

        Returns:
            Composition string (e.g., "Pt0O2H4" or "O2H4")
        """
        if type_map:
            # Include all elements from type_map, even with count 0
            composition_parts = []
            for element in type_map:
                count = element_counts.get(element, 0)
                composition_parts.append(f"{element}{count}")
            return "".join(composition_parts)
        else:
            # Use the order from POSCAR (ordered_elements)
            composition_parts = []
            for element in ordered_elements:
                count = element_counts[element]
                composition_parts.append(f"{element}{count}")
            return "".join(composition_parts)

    def get_composition_string(
        self, element_counts: dict[str, int], type_map: list[str] | None = None
    ) -> str:
        """
        Generate composition string with proper formatting.

        Args:
            element_counts: Dictionary of element counts
            type_map: Optional type map to include additional elements with count 0

        Returns:
            Composition string (e.g., "Pt0O2H4" or "O2H4")
        """
        if type_map:
            # Include all elements from type_map, even with count 0
            composition_parts = []
            for element in type_map:
                count = element_counts.get(element, 0)
                composition_parts.append(f"{element}{count}")
            return "".join(composition_parts)
        else:
            # Define standard element order: metals, O, H, cations, anions
            # Common FCC metals
            metals = {"Pt", "Au", "Ag", "Pd", "Cu", "Ni", "Rh", "Ir", "Al", "Pb", "Ca", "Sr", "Yb"}
            # Common cations
            cations = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}
            # Common anions
            anions = {"F", "Cl", "Br", "I"}

            # Sort elements by priority
            def element_priority(elem):
                if elem in metals:
                    return (1, elem)
                elif elem == "O":
                    return (2, elem)
                elif elem == "H":
                    return (3, elem)
                elif elem in cations:
                    return (4, elem)
                elif elem in anions:
                    return (5, elem)
                else:
                    return (6, elem)

            sorted_elements = sorted(element_counts.keys(), key=element_priority)

            composition_parts = []
            for element in sorted_elements:
                count = element_counts[element]
                composition_parts.append(f"{element}{count}")
            return "".join(composition_parts)

    def extract_wannier_centroids(self, wc_data: np.ndarray) -> np.ndarray:
        """
        Extract wannier centroids from wc_out.npy data.

        Args:
            wc_data: Loaded wannier center data array

        Returns:
            Concatenated wannier centroids array
        """
        centroids = []
        for atom_data in wc_data:
            if isinstance(atom_data, dict) and "wannier_centroid" in atom_data:
                centroid = atom_data["wannier_centroid"]
                # Ensure it's a numpy array with shape (3,)
                if isinstance(centroid, np.ndarray) and centroid.shape == (3,):
                    centroids.append(centroid)
                else:
                    self.logger.warning(
                        f"Invalid wannier_centroid shape for atom {atom_data.get('atom_index', '?')}"
                    )
                    centroids.append(np.zeros(3))
            else:
                self.logger.warning("Missing wannier_centroid for atom")
                centroids.append(np.zeros(3))

        # Concatenate all centroids into a single flat array
        return np.concatenate(centroids)

    def organize_data_by_composition(self, all_data: dict) -> dict:
        """
        Organize collected data by composition.

        Args:
            all_data: Dictionary mapping directories to their data

        Returns:
            Dictionary organized by composition
        """
        organized = defaultdict(list)

        for _, data in all_data.items():
            composition = data["composition"]
            organized[composition].append(data)

        return dict(organized)

    def load_structure_data(self, directory: Path) -> dict | None:
        """
        Load structure data from POSCAR for coordinates, box, and types.

        Args:
            directory: Directory containing POSCAR

        Returns:
            Dictionary with structure data or None if failed
        """
        poscar_file = directory / "POSCAR"
        if not poscar_file.exists():
            return None

        try:
            atoms = read(poscar_file)

            # Get coordinates
            coords = atoms.get_positions()

            # Get box/cell
            cell = atoms.get_cell()

            # Get chemical symbols
            symbols = atoms.get_chemical_symbols()

            return {
                "coords": coords,
                "cell": cell,
                "symbols": symbols,
                "n_atoms": len(atoms),
            }
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Failed to load structure from {directory}: {e}")
            return None

    def save_dpdata_format(self, composition_data: list[dict], composition: str) -> None:
        """
        Save data in dpdata format for a specific composition.

        Args:
            composition_data: List of data dictionaries for this composition
            composition: Composition string (e.g., "O2H4")
        """
        # Create output directory structure
        comp_dir = self.output_dir / composition / "set.000"
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Collect all atomic dipoles
        atomic_dipoles = []
        coords_list = []
        cells_list = []

        for data in composition_data:
            atomic_dipoles.append(data["atomic_dipole"])

            # Load structure data if available
            struct_data = self.load_structure_data(data["directory"])
            if struct_data:
                coords_list.append(struct_data["coords"].flatten())
                cells_list.append(struct_data["cell"].flatten())

        # Stack into arrays
        atomic_dipoles_array = np.array(atomic_dipoles)

        # Save atomic dipoles
        np.save(comp_dir / "atomic_dipole.npy", atomic_dipoles_array)
        self.logger.info(f"  Saved atomic_dipole.npy with shape {atomic_dipoles_array.shape}")

        # If we have structure data, save it too
        if coords_list and cells_list:
            coords_array = np.array(coords_list).reshape(len(coords_list), -1, 3)
            cells_array = np.array(cells_list).reshape(len(cells_list), 3, 3)

            np.save(comp_dir / "coord.npy", coords_array)
            np.save(comp_dir / "box.npy", cells_array)

            # Create type array based on composition
            if self.type_map:
                # Parse composition to get counts
                import re

                pattern = r"([A-Z][a-z]?)(\d+)"
                matches = re.findall(pattern, composition)

                type_array = []
                for element, count in matches:
                    if element in self.type_map:
                        type_idx = self.type_map.index(element)
                        type_array.extend([type_idx] * int(count))

                type_array = np.array(type_array)
                np.save(comp_dir / "type.npy", np.tile(type_array, (len(coords_list), 1)))

                # Save type_map
                with open(comp_dir / "type_map.raw", "w") as f:
                    for element in self.type_map:
                        f.write(f"{element}\n")

    def run(self) -> None:
        """Run the conversion process."""
        self.logger.step("Starting Wannier center to dpdata conversion")

        if self.type_map:
            self.logger.info(f"Type map: {self.type_map}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Read directory list
        directories = self.read_directory_list()
        if not directories:
            self.logger.error("No valid directories found")
            return

        # Process counters
        processed_count = 0
        skipped_count = 0
        failed_count = 0

        # Store all data
        all_data = {}

        # Process all directories
        self.logger.step(f"Processing {len(directories)} directories")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing directories...", total=len(directories))

            for directory in directories:
                progress.update(
                    task,
                    description=f"[cyan]Processing: {directory.name}",
                )

                # Load wannier data
                wc_data = self.load_wannier_data(directory)
                if wc_data is None:
                    skipped_count += 1
                    progress.advance(task)
                    continue

                # Extract composition from POSCAR
                comp_result = self.extract_composition_from_poscar(directory)
                if comp_result is None:
                    skipped_count += 1
                    progress.advance(task)
                    continue

                element_counts, ordered_elements = comp_result

                # Get composition string
                composition = self.get_composition_string_ordered(
                    element_counts, ordered_elements, self.type_map
                )

                # Extract wannier centroids
                try:
                    atomic_dipole = self.extract_wannier_centroids(wc_data)

                    # Store data
                    all_data[directory] = {
                        "directory": directory,
                        "composition": composition,
                        "atomic_dipole": atomic_dipole,
                        "element_counts": element_counts,
                        "ordered_elements": ordered_elements,
                    }

                    processed_count += 1

                except Exception as e:
                    if self.verbose:
                        self.logger.debug(f"Failed to process {directory}: {e}")
                    failed_count += 1

                progress.advance(task)

        # Summary
        self.logger.info("\nProcessing summary:")
        self.logger.info(f"  Processed: {processed_count} directories")
        self.logger.info(f"  Skipped: {skipped_count} directories")
        self.logger.info(f"  Failed: {failed_count} directories")

        if processed_count == 0:
            self.logger.error("No directories were successfully processed")
            return

        # Organize by composition
        organized_data = self.organize_data_by_composition(all_data)

        # Save data for each composition
        self.logger.step("Saving to dpdata format")
        self.logger.info(f"Found {len(organized_data)} unique compositions")

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for composition, comp_data in organized_data.items():
            self.logger.info(f"\n{composition}: {len(comp_data)} samples")
            self.save_dpdata_format(comp_data, composition)

        # Save metadata
        metadata = {
            "type_map": self.type_map,
            "total_directories_processed": processed_count,
            "directories_skipped": skipped_count,
            "directories_failed": failed_count,
            "unique_compositions": list(organized_data.keys()),
            "samples_per_composition": {comp: len(data) for comp, data in organized_data.items()},
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.success("\nConversion completed successfully!")
        self.logger.info(f"Output saved to: {self.output_dir}")
        self.logger.info(f"Metadata saved to: {self.output_dir / 'metadata.json'}")
