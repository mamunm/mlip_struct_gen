"""MLIP Structure Generation Package."""

__version__ = "0.1.0"
__author__ = "MLIP Structure Generation Team"
__description__ = "MLIP initial structure generation and MD workflow management"

from . import generate_structure, utils

__all__ = [
    "generate_structure",
    "utils",
    "__version__",
    "__author__",
    "__description__",
]
