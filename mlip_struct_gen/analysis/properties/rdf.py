"""Radial Distribution Function (RDF) calculator."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .._analysis_core import RDFCalculator, TrajectoryReader
except ImportError:
    raise ImportError(
        "C++ extension not compiled. Please build the extension first using: "
        "python -m pip install -e . or build manually with CMake."
    ) from None

from .base import PropertyCalculator


class RDF(PropertyCalculator):
    """
    Radial Distribution Function calculator.

    Computes g(r) and coordination numbers from LAMMPS trajectory files.
    """

    def __init__(
        self,
        trajectory_file: str | Path,
        type_map: dict[int, str] | None = None,
    ):
        """
        Initialize RDF calculator.

        Args:
            trajectory_file: Path to LAMMPS trajectory file
            type_map: Mapping of atom type indices to element symbols
                     (e.g., {1: "Pt", 2: "O", 3: "H"})
        """
        super().__init__(trajectory_file)
        self.type_map = type_map or {}
        self._calculator = RDFCalculator()
        self._reader = None

    def compute(
        self,
        pairs: list[tuple[str, str]],
        rmax: float = 10.0,
        nbins: int = 200,
        n_frames: int | None = None,
    ) -> dict[str, Any]:
        """
        Compute RDF for specified element pairs.

        Args:
            pairs: List of element pairs to compute RDF for
                  (e.g., [("O", "O"), ("O", "H"), ("Na", "Cl")])
            rmax: Maximum distance for RDF computation (Angstroms)
            nbins: Number of bins for histogram
            n_frames: Number of frames to use (None = all frames)

        Returns:
            Dictionary with RDF results for each pair
        """
        # Set parameters
        self._calculator.set_rmax(rmax)
        self._calculator.set_nbins(nbins)

        # Read trajectory
        self._reader = TrajectoryReader(str(self.trajectory_file))
        if self.type_map:
            self._reader.set_type_map(self.type_map)

        # Read frames
        if n_frames is None:
            frames = self._reader.read_all_frames()
        else:
            frames = self._reader.read_frames(n_frames)

        if not frames:
            raise RuntimeError("No frames read from trajectory")

        # Compute RDFs
        results_list = self._calculator.compute_multiple_rdfs(frames, pairs)

        # Convert to dictionary
        self._results = {}
        for result in results_list:
            result_dict = result.to_numpy()
            pair_name = result_dict["pair"]
            self._results[pair_name] = {
                "r": np.array(result_dict["r"]),
                "gr": np.array(result_dict["gr"]),
                "coordination": np.array(result_dict["coordination"]),
                "n_frames": result_dict["n_frames"],
                "rmax": result_dict["rmax"],
                "nbins": result_dict["nbins"],
            }

        return self._results

    def plot(
        self,
        backend: str = "matplotlib",
        pairs: list[str] | None = None,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot RDF results.

        Args:
            backend: Plotting backend ("matplotlib" or "plotly")
            pairs: List of pairs to plot (None = all pairs)
            output: Output file path (None = display only)
            **kwargs: Additional plotting parameters

        Returns:
            Plot object (depends on backend)
        """
        if self._results is None:
            raise RuntimeError("No results to plot. Run compute() first.")

        # Select pairs to plot
        if pairs is None:
            pairs = list(self._results.keys())

        # Import plotting backend
        if backend == "matplotlib":
            from ..plotting.matplotlib_backend import plot_rdf_matplotlib

            return plot_rdf_matplotlib(self._results, pairs, output, **kwargs)
        elif backend == "plotly":
            from ..plotting.plotly_backend import plot_rdf_plotly

            return plot_rdf_plotly(self._results, pairs, output, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _save_csv(self, output_path: Path) -> None:
        """Save RDF results as CSV files (one per pair)."""
        for pair_name, data in self._results.items():
            # Create filename
            safe_name = pair_name.replace("-", "_")
            csv_path = output_path.parent / f"{output_path.stem}_{safe_name}.csv"

            # Write CSV
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["r (Angstrom)", "g(r)", "Coordination Number"])

                for r, gr, cn in zip(data["r"], data["gr"], data["coordination"], strict=False):
                    writer.writerow([r, gr, cn])

    def _save_json(self, output_path: Path) -> None:
        """Save RDF results as JSON."""
        # Convert numpy arrays to lists
        json_data = {}
        for pair_name, data in self._results.items():
            json_data[pair_name] = {
                "r": data["r"].tolist(),
                "gr": data["gr"].tolist(),
                "coordination": data["coordination"].tolist(),
                "n_frames": int(data["n_frames"]),
                "rmax": float(data["rmax"]),
                "nbins": int(data["nbins"]),
            }

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    def _save_npz(self, output_path: Path) -> None:
        """Save RDF results as compressed numpy format."""
        # Flatten dictionary for npz
        save_dict = {}
        for pair_name, data in self._results.items():
            safe_name = pair_name.replace("-", "_")
            save_dict[f"{safe_name}_r"] = data["r"]
            save_dict[f"{safe_name}_gr"] = data["gr"]
            save_dict[f"{safe_name}_coordination"] = data["coordination"]

        np.savez_compressed(output_path, **save_dict)

    def get_first_peak(self, pair: str) -> tuple[float, float]:
        """
        Get position and height of first peak in g(r).

        Args:
            pair: Pair name (e.g., "O-O")

        Returns:
            Tuple of (r_peak, gr_peak)
        """
        if self._results is None or pair not in self._results:
            raise RuntimeError(f"No results for pair {pair}")

        data = self._results[pair]
        idx = np.argmax(data["gr"])
        return data["r"][idx], data["gr"][idx]

    def get_coordination_number(self, pair: str, r_cutoff: float) -> float:
        """
        Get coordination number at a specific cutoff distance.

        Args:
            pair: Pair name (e.g., "O-O")
            r_cutoff: Cutoff distance (Angstroms)

        Returns:
            Coordination number at r_cutoff
        """
        if self._results is None or pair not in self._results:
            raise RuntimeError(f"No results for pair {pair}")

        data = self._results[pair]
        idx = np.searchsorted(data["r"], r_cutoff)
        if idx >= len(data["coordination"]):
            idx = len(data["coordination"]) - 1

        return data["coordination"][idx]
