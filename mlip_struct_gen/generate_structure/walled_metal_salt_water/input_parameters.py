"""Input parameters for walled metal-salt-water interface generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class WalledMetalSaltWaterParameters:
    """
    Parameters for walled FCC(111) metal surface with salt-water layer generation.

    Creates a structure with metal walls on both bottom and top, sandwiching
    salt-water solution between them. The metal slab is split into two walls:
    - Bottom wall: floor((nz+1)/2) layers (more for odd nz)
    - Top wall: floor((nz-1)/2) layers (fewer for odd nz)

    Z-axis layout (bottom to top):
        z=0: bottom metal wall
        gap_above_metal: gap between bottom wall and solution
        solution region: computed from n_water, density, and optionally salt volume
        vacuum_above_water: gap between solution and top wall
        top metal wall
        remaining space up to box_z

    Args:
        metal: Metal element symbol (e.g., "Au", "Pt", "Cu").

        size: Surface size as (nx, ny, nz) unit cells.
            nz is the TOTAL number of layers across both walls.

        n_water: Number of water molecules.

        salt_type: Salt type (NaCl, KCl, LiCl, CaCl2, MgCl2, NaBr, KBr, CsCl).

        n_salt: Number of salt formula units.

        output_file: Path to output structure file.

        box_z: Total z-dimension of the simulation box in Angstroms.

        include_salt_volume: Account for ion volume in density calculation.

        density: Target density in g/cm^3. Default: 1.0

        gap_above_metal: Gap between bottom metal wall and solution in Angstroms.
            Default: 0.0

        vacuum_above_water: Gap between solution top and top metal wall in Angstroms.
            Default: 0.0

        water_model: Water model geometry. Default: "SPC/E"

        lattice_constant: Custom lattice constant. Default: None

        fix_bottom_layers: Number of layers to fix in each wall (symmetric).
            Default: 0

        packmol_executable: Path to packmol. Default: "packmol"

        packmol_tolerance: Packmol tolerance. Default: 2.0

        seed: Random seed. Default: 12345

        no_salt_zone: Fraction of solution height to exclude ions from top/bottom.
            Default: 0.2

        save_artifacts: Save intermediate files. Default: False

        output_format: Output format override. Default: None

        elements: Element order for LAMMPS atom types. Default: None

        log: Enable logging. Default: False

        logger: Custom MLIPLogger instance. Default: None
    """

    metal: str
    size: tuple[int, int, int]  # (nx, ny, nz) - nz = total layers across both walls
    n_water: int
    salt_type: str
    n_salt: int
    output_file: str
    box_z: float  # total z-dimension of the simulation box
    include_salt_volume: bool = False
    density: float = 1.0  # g/cm^3
    gap_above_metal: float = 0.0  # Angstroms
    vacuum_above_water: float = 0.0  # Angstroms
    water_model: str = "SPC/E"
    lattice_constant: float | None = None
    fix_bottom_layers: int = 0
    packmol_executable: str = "packmol"
    packmol_tolerance: float = 2.0
    seed: int = 12345
    no_salt_zone: float = 0.2
    save_artifacts: bool = False
    output_format: str | None = None
    elements: list[str] | None = None
    log: bool = False
    logger: Optional["MLIPLogger"] = None
