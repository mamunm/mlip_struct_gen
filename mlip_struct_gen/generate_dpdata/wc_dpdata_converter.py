"""Unified converter: VASP OUTCAR + Wannier centroids -> dpdata with aligned atomic dipoles.

For every OUTCAR discovered under ``input_dir``, this converter expects the same
directory to contain ``wannier90_centres.xyz``. Wannier centroids are computed
inline from that file against the post-``apply_type_map`` coordinates; no
intermediate ``wc_out.npy`` is read or written. Energies, forces, coordinates,
cells, virials and atomic dipoles are all written in the same pass so row and
atom ordering are guaranteed to match under the user-provided ``type_map``.
"""

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

from mlip_struct_gen.generate_dpdata.dpdata_converter import CompositionData
from mlip_struct_gen.utils.logger import get_logger
from mlip_struct_gen.wannier_centroid.compute_wannier_centroid import parse_wannier_centers
from mlip_struct_gen.wannier_centroid.pbc_utils import find_k_nearest_neighbors

# Atoms whose atomic dipole is labelled from Wannier centers.
WANNIER_TARGET_ATOMS = ("O", "Na", "Cl", "K", "Li", "Cs")

# Number of nearest Wannier centers averaged per atom type (from
# compute_wannier_centroid.py:111).
WC_N_NEIGHBORS = {"O": 4, "Na": 4, "Cl": 4, "K": 4, "Li": 1, "Cs": 4}


class WCCompositionData(CompositionData):
    """CompositionData extended with per-frame atomic dipoles for selected atoms."""

    def __init__(self, composition: str, type_map: list[str], sel_type: list[str]):
        super().__init__(composition, type_map)
        self.sel_type = sel_type
        self.dipoles: list[np.ndarray] = []
        self.n_sel_atoms: int | None = None

    def add_dipole_frame(self, dipole_flat: np.ndarray, source_path: Path) -> bool:
        """Append a single-frame flat dipole vector of shape (3 * n_sel_atoms,)."""
        logger = get_logger()
        n_vals = dipole_flat.shape[0]
        if self.n_sel_atoms is None:
            self.n_sel_atoms = n_vals // 3
        elif n_vals != 3 * self.n_sel_atoms:
            logger.warning(
                f"Dipole length mismatch in {source_path}: got {n_vals}, "
                f"expected {3 * self.n_sel_atoms}"
            )
            return False
        self.dipoles.append(dipole_flat.reshape(1, -1))
        return True

    def save_to_disk(self, output_dir: Path, verbose: bool = False) -> None:
        super().save_to_disk(output_dir, verbose=verbose)
        if not self.dipoles:
            return
        set_dir = output_dir / self.composition / "set.000"
        all_dipoles = np.vstack(self.dipoles)
        assert all_dipoles.shape == (self.total_frames, 3 * self.n_sel_atoms)
        np.save(set_dir / "atomic_dipole.npy", all_dipoles)
        if verbose:
            get_logger().info(
                f"  Saved atomic_dipole.npy for {self.composition}: " f"{all_dipoles.shape}"
            )


class WCDPDataConverter:
    """Unified OUTCAR + Wannier converter producing dpdata with atomic dipoles."""

    def __init__(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        type_map: list[str],
        sel_type: list[str] | None = None,
        recursive: bool = True,
        verbose: bool = False,
        save_file_loc: Path | str | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.type_map = list(type_map)
        self.recursive = recursive
        self.verbose = verbose
        self.save_file_loc = Path(save_file_loc) if save_file_loc else None
        self.logger = get_logger()

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        if sel_type is None:
            sel_type = [t for t in WANNIER_TARGET_ATOMS if t in self.type_map]
        unknown = [t for t in sel_type if t not in self.type_map]
        if unknown:
            raise ValueError(f"sel_type contains elements not in type_map: {unknown}")
        self.sel_type = list(sel_type)
        self.sel_type_indices = {self.type_map.index(t) for t in self.sel_type}

        self.composition_data: dict[str, WCCompositionData] = {}

    def find_outcars(self) -> list[Path]:
        if self.recursive:
            outcars = sorted(self.input_dir.rglob("OUTCAR"))
        else:
            outcars = sorted(self.input_dir.glob("*/OUTCAR"))
        self.logger.info(f"Found {len(outcars)} OUTCAR files")
        return outcars

    def _get_composition_string(self, atom_numbs: list[int]) -> str:
        parts = []
        for i, element in enumerate(self.type_map):
            count = atom_numbs[i] if i < len(atom_numbs) else 0
            parts.append(f"{element}{count}")
        return "".join(parts)

    def _compute_atomic_dipoles(
        self,
        coords: np.ndarray,
        atom_types: np.ndarray,
        cell: np.ndarray,
        wannier_positions: np.ndarray,
    ) -> np.ndarray:
        """Compute flat dipole vector for selected atoms in ``atom_types`` order."""
        inv_cell = np.linalg.inv(cell)
        sel_indices = np.where(
            np.array([t in self.sel_type_indices for t in atom_types], dtype=bool)
        )[0]
        if sel_indices.size == 0:
            return np.empty(0, dtype=float)

        centroids = np.empty((sel_indices.size, 3), dtype=float)
        for out_i, atom_i in enumerate(sel_indices):
            symbol = self.type_map[atom_types[atom_i]]
            k = WC_N_NEIGHBORS[symbol]
            _, _, vectors = find_k_nearest_neighbors(coords[atom_i], wannier_positions, cell, k)
            # Minimum-image each (wc_i - atom) vector.
            vec_frac = vectors @ inv_cell
            vectors = (vec_frac - np.round(vec_frac)) @ cell
            # Mean, then minimum-image the mean as well.
            centroid = vectors.mean(axis=0)
            centroid = centroid - np.round(centroid @ inv_cell) @ cell
            centroids[out_i] = centroid
        return centroids.reshape(-1)

    def _process_outcar(self, outcar_path: Path) -> tuple[str, dict] | None:
        """Load one OUTCAR + compute its dipoles. Returns (composition, payload) or a skip tag."""
        directory = outcar_path.parent

        wannier_file = directory / "wannier90_centres.xyz"
        if not wannier_file.exists():
            if self.verbose:
                self.logger.debug(f"No wannier90_centres.xyz in {directory}, skipping")
            return ("__skip_no_wannier__", {})

        try:
            system = dpdata.LabeledSystem(str(outcar_path), fmt="vasp/outcar")
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Failed to load {outcar_path}: {e}")
            return ("__skip_failed__", {})

        n_frames = len(system.data["coords"])
        if n_frames != 1:
            if self.verbose:
                self.logger.debug(
                    f"Multi-frame OUTCAR ({n_frames} frames) at {outcar_path}, skipping"
                )
            return ("__skip_multiframe__", {})

        system_elements = set(system.data.get("atom_names", []))
        if not system_elements.issubset(set(self.type_map)):
            if self.verbose:
                self.logger.debug(
                    f"Elements {system_elements} not subset of type_map in {outcar_path}"
                )
            return ("__skip_bad_elements__", {})

        system.apply_type_map(self.type_map)

        try:
            wannier_positions = parse_wannier_centers(wannier_file)
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Failed to parse {wannier_file}: {e}")
            return ("__skip_failed__", {})

        try:
            dipole_flat = self._compute_atomic_dipoles(
                system.data["coords"][0],
                np.asarray(system.data["atom_types"]),
                np.asarray(system.data["cells"][0]),
                wannier_positions,
            )
        except Exception as e:
            self.logger.warning(f"{outcar_path}: dipole computation failed: {e}")
            return ("__skip_failed__", {})

        composition = self._get_composition_string(list(system.data["atom_numbs"]))
        payload = {
            "system_data": system.data,
            "dipole_flat": dipole_flat,
            "source": outcar_path,
            "directory": directory.resolve(),
        }
        return (composition, payload)

    def run(self) -> None:
        self.logger.step("Starting unified OUTCAR + Wannier -> dpdata conversion")
        self.logger.info(f"Type map: {self.type_map}")
        self.logger.info(f"Sel type (dipole atoms): {self.sel_type}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.save_file_loc:
            self.logger.info(f"Saving OUTCAR locations to: {self.save_file_loc}")

        outcars = self.find_outcars()
        if not outcars:
            self.logger.error("No OUTCAR files found")
            return

        processed = 0
        skipped_no_wannier = 0
        skipped_multiframe = 0
        skipped_bad_elements = 0
        failed = 0
        processed_paths: list[Path] = []

        self.logger.step(f"Processing {len(outcars)} OUTCAR files")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing OUTCARs...", total=len(outcars))

            for outcar_path in outcars:
                progress.update(
                    task,
                    description=f"[cyan]Processing: {outcar_path.parent.name}/{outcar_path.name}",
                )
                result = self._process_outcar(outcar_path)
                progress.advance(task)

                if result is None:
                    failed += 1
                    continue
                comp, payload = result
                if comp == "__skip_no_wannier__":
                    skipped_no_wannier += 1
                    continue
                if comp == "__skip_multiframe__":
                    skipped_multiframe += 1
                    continue
                if comp == "__skip_bad_elements__":
                    skipped_bad_elements += 1
                    continue
                if comp == "__skip_failed__":
                    failed += 1
                    continue

                if comp not in self.composition_data:
                    self.composition_data[comp] = WCCompositionData(
                        comp, self.type_map, self.sel_type
                    )
                bucket = self.composition_data[comp]
                if not bucket.add_system(payload["system_data"], payload["source"]):
                    failed += 1
                    continue
                if not bucket.add_dipole_frame(payload["dipole_flat"], payload["source"]):
                    failed += 1
                    continue

                processed += 1
                processed_paths.append(payload["directory"])

        self.logger.info("\nProcessing summary:")
        self.logger.info(f"  Processed: {processed}")
        self.logger.info(f"  Skipped (no Wannier data): {skipped_no_wannier}")
        self.logger.info(f"  Skipped (multi-frame OUTCAR): {skipped_multiframe}")
        self.logger.info(f"  Skipped (elements not in type_map): {skipped_bad_elements}")
        self.logger.info(f"  Failed: {failed}")

        if processed == 0:
            self.logger.error("No systems were successfully processed")
            return

        self.logger.step("Saving to dpdata format")
        self.logger.info(f"Found {len(self.composition_data)} unique compositions")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        total_frames = 0
        for comp_data in self.composition_data.values():
            comp_data.save_to_disk(self.output_dir, verbose=self.verbose)
            total_frames += comp_data.total_frames

        self.logger.success("\nConversion completed successfully!")
        self.logger.info(f"Output saved to: {self.output_dir}")

        self.logger.info("\nGenerated compositions:")
        for comp, comp_data in self.composition_data.items():
            self.logger.info(
                f"  {comp}: {comp_data.total_frames} frames, "
                f"{comp_data.n_atoms} atoms, "
                f"{comp_data.n_sel_atoms or 0} dipole atoms"
            )

        metadata = {
            "type_map": self.type_map,
            "sel_type": self.sel_type,
            "n_outcars_seen": len(outcars),
            "processed": processed,
            "skipped_no_wannier": skipped_no_wannier,
            "skipped_multiframe": skipped_multiframe,
            "skipped_bad_elements": skipped_bad_elements,
            "failed": failed,
            "unique_compositions": len(self.composition_data),
            "total_frames": int(total_frames),
            "compositions": {
                comp: {
                    "n_frames": int(cd.total_frames),
                    "n_atoms": int(cd.n_atoms),
                    "n_sel_atoms": int(cd.n_sel_atoms or 0),
                    "n_systems": len(cd.frame_counts),
                }
                for comp, cd in self.composition_data.items()
            },
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"\nMetadata saved to: {self.output_dir / 'metadata.json'}")

        if self.save_file_loc and processed_paths:
            with open(self.save_file_loc, "w") as f:
                for p in processed_paths:
                    f.write(str(p) + "\n")
            self.logger.info(
                f"OUTCAR directory locations saved to: {self.save_file_loc} "
                f"({len(processed_paths)} dirs)"
            )
