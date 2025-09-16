"""Utility functions for structure generation."""

from pathlib import Path

from ase import Atoms
from ase.io import read, write


def save_structure(atoms: Atoms, filename: str | Path, format: str | None = None, **kwargs) -> None:
    """
    Save structure to file.

    Args:
        atoms: Structure to save
        filename: Output filename
        format: File format (auto-detected if None)
        **kwargs: Additional arguments for ASE write
    """
    write(filename, atoms, format=format, **kwargs)


def load_structure(
    filename: str | Path, index: int | str = -1, format: str | None = None
) -> Atoms | list[Atoms]:
    """
    Load structure from file.

    Args:
        filename: Input filename
        index: Frame index or slice
        format: File format (auto-detected if None)

    Returns:
        Single Atoms object or list of Atoms
    """
    return read(filename, index=index, format=format)
