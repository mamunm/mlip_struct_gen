"""dpdata setup module for converting VASP outputs to DeepMD format."""

from .deepmd_npy_converter import convert_to_dpdata

__all__ = ["convert_to_dpdata"]