"""Potential files storage for LAMMPS simulations."""

from pathlib import Path

# Get the potentials directory path
POTENTIALS_DIR = Path(__file__).parent
LJ_PARAMS_FILE = POTENTIALS_DIR / "lj_parameters.json"

__all__ = ["POTENTIALS_DIR", "LJ_PARAMS_FILE"]
