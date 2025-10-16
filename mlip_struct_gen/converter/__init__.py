"""
MLIP Converter Module

Tools for converting and processing LAMMPS data files for machine learning interatomic potentials.
"""

from .mlip_sr_lr_convert import (
    ELEMENT_MASSES,
    LAMMPSDataProcessor,
    parse_charge_map,
    parse_duplication_spec,
)

__all__ = ["LAMMPSDataProcessor", "parse_duplication_spec", "parse_charge_map", "ELEMENT_MASSES"]

__version__ = "1.0.0"
