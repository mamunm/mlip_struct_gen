"""Base class for physical property calculations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PropertyCalculator(ABC):
    """
    Abstract base class for all property calculators.

    This provides a common interface for computing physical properties
    from trajectory data.
    """

    def __init__(self, trajectory_file: str | Path):
        """
        Initialize the property calculator.

        Args:
            trajectory_file: Path to the trajectory file
        """
        self.trajectory_file = Path(trajectory_file)
        if not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

        self._results = None

    @abstractmethod
    def compute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Compute the property.

        Args:
            **kwargs: Property-specific parameters

        Returns:
            Dictionary containing computed results
        """
        pass

    @abstractmethod
    def plot(self, backend: str = "matplotlib", **kwargs: Any) -> Any:
        """
        Plot the computed property.

        Args:
            backend: Plotting backend ("matplotlib" or "plotly")
            **kwargs: Backend-specific plotting parameters

        Returns:
            Plot object (depends on backend)
        """
        pass

    def save(self, output_file: str | Path, format: str = "csv") -> None:
        """
        Save results to file.

        Args:
            output_file: Output file path
            format: Output format ("csv", "json", "npz")
        """
        if self._results is None:
            raise RuntimeError("No results to save. Run compute() first.")

        output_path = Path(output_file)

        if format == "csv":
            self._save_csv(output_path)
        elif format == "json":
            self._save_json(output_path)
        elif format == "npz":
            self._save_npz(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @abstractmethod
    def _save_csv(self, output_path: Path) -> None:
        """Save results as CSV."""
        pass

    @abstractmethod
    def _save_json(self, output_path: Path) -> None:
        """Save results as JSON."""
        pass

    @abstractmethod
    def _save_npz(self, output_path: Path) -> None:
        """Save results as NPZ (numpy compressed)."""
        pass

    @property
    def results(self) -> dict[str, Any] | None:
        """Get computed results."""
        return self._results
