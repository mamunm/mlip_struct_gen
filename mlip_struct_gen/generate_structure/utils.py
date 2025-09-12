"""Utility functions for structure generation."""

from typing import Optional, Union, List
from ase import Atoms
from ase.io import read, write
from pathlib import Path


def save_structure(
    atoms: Atoms,
    filename: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
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
    filename: Union[str, Path],
    index: Union[int, str] = -1,
    format: Optional[str] = None
) -> Union[Atoms, List[Atoms]]:
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