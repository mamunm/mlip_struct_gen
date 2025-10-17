"""Physical property analysis module for MLIP trajectory data.

This module provides fast C++-accelerated computation of physical properties
from LAMMPS trajectories, including:
- Radial Distribution Function (RDF)
- Coordination numbers
- (Future: Mean Square Displacement, Diffusion coefficients, etc.)
"""

from .properties.rdf import RDF

__all__ = ["RDF"]

__version__ = "0.1.0"
