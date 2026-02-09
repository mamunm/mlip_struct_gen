"""Input parameters for walled metal-water interface generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class WalledMetalWaterParameters:
    """
    Parameters for walled FCC(111) metal surface with water layer generation.

    Creates a structure with metal walls on both bottom and top, sandwiching
    water molecules between them. The metal slab is split into two walls:
    - Bottom wall: floor((nz+1)/2) layers (more for odd nz)
    - Top wall: floor((nz-1)/2) layers (fewer for odd nz)

    Z-axis layout (bottom to top):
        z=0: bottom metal wall
        gap_above_metal: gap between bottom wall and water
        water region: computed from n_water and density
        vacuum_above_water: gap between water and top wall
        top metal wall
        remaining space up to box_z

    Args:
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu", "Ag", "Pd", "Ni", "Al").

        size: Surface size as (nx, ny, nz) unit cells.
            nz is the TOTAL number of layers across both walls.

        n_water: Number of water molecules to pack between the walls.

        output_file: Path to output structure file.

        box_z: Total z-dimension of the simulation box in Angstroms.
            Must be large enough to fit both walls, gaps, and water.

        density: Target density of water in g/cm^3. Default: 1.0

        gap_above_metal: Gap between bottom metal wall top and water bottom
            in Angstroms. Applied symmetrically (same gap below top wall).
            Default: 0.0

        vacuum_above_water: Gap between water top and top metal wall bottom
            in Angstroms. Default: 0.0

        water_model: Water model geometry. Default: "SPC/E"

        lattice_constant: Custom lattice constant in Angstroms. Default: None

        fix_bottom_layers: Number of layers to fix in each wall (symmetric).
            Fixes bottom N layers of bottom wall and top N layers of top wall.
            Default: 0

        packmol_executable: Path to packmol executable. Default: "packmol"

        packmol_tolerance: Packmol packing tolerance in Angstroms. Default: 2.0

        seed: Random seed for reproducible configurations. Default: 12345

        output_format: Output format override. Default: None (infer from extension)

        elements: Element order for LAMMPS atom types. Default: None

        save_artifacts: Save intermediate files. Default: False

        log: Enable logging. Default: False

        logger: Custom MLIPLogger instance. Default: None
    """

    metal: str
    size: tuple[int, int, int]  # (nx, ny, nz) - nz = total layers across both walls
    n_water: int
    output_file: str
    box_z: float  # total z-dimension of the simulation box
    density: float = 1.0  # g/cm^3
    gap_above_metal: float = 0.0  # Angstroms
    vacuum_above_water: float = 0.0  # Angstroms
    water_model: str = "SPC/E"
    lattice_constant: float | None = None
    fix_bottom_layers: int = 0
    packmol_executable: str = "packmol"
    packmol_tolerance: float = 2.0
    seed: int = 12345
    output_format: str | None = None
    elements: list[str] | None = None
    save_artifacts: bool = False
    log: bool = False
    logger: Optional["MLIPLogger"] = None
