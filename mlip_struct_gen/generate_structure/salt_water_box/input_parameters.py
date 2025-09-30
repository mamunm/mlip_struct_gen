"""Input parameters for salt water box generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...utils.logger import MLIPLogger


@dataclass
class SaltWaterBoxGeneratorParameters:
    """
    Parameters for salt water box generation using Packmol.

    This dataclass defines all parameters needed to generate a salt water box
    (aqueous salt solution) using Packmol. Similar to water box generation,
    you can specify any 2 of 3 parameters (box_size, n_water, density).

    Args:
        output_file: Path to output file. Extension determines format if not specified.

        box_size: Box dimensions in Angstroms. Can be:
            - Single number (float): Creates cubic box
            - Tuple/list of 3 numbers: Creates rectangular box (x, y, z)
            - None: Automatically computed from n_water and density
            Valid range: 10.0 - 1000.0 Å per dimension

        n_water: Number of water molecules. If None, calculated
            from box_size and density (accounting for ion volume if enabled).
            Valid range: 1 - 1,000,000 molecules

        density: Total solution density in g/cm³. If None, calculated
            from water density and salt contribution.
            Valid range: 0.5 - 2.0 g/cm³

        salt_type: Type of salt to add. Supported salts:
            - "NaCl": Sodium chloride (default)
            - "KCl": Potassium chloride
            - "LiCl": Lithium chloride
            - "CaCl2": Calcium chloride (2:1 ratio)
            - "MgCl2": Magnesium chloride (2:1 ratio)

        n_salt: Number of salt formula units (e.g., for CaCl2,
            one formula unit = 1 Ca²⁺ + 2 Cl⁻).
            Valid range: 0 - 100000

        include_salt_volume: If True, subtracts ion volumes from available
            space using VDW radii. Default False for simplicity.
            Useful for high salt concentrations.

        water_model: Water model for molecular geometry.
            Supported: "SPC/E" (default), "TIP3P", "TIP4P", "SPC/Fw"

        tolerance: Packmol tolerance in Angstroms.
            Valid range: 0.5 - 10.0 Å (typical: 1.5 - 2.5 Å)

        seed: Random seed for reproducibility.

        packmol_executable: Path to Packmol executable.

        output_format: Output format.
            - "lammps": LAMMPS data file with bonds/angles (default)
            - "xyz": Simple coordinate format
            - "poscar": VASP POSCAR format

        elements: List of elements defining atom type order for LAMMPS format.
            When specified, atom types are assigned based on the order in this list.
            For example: ["Pt", "O", "H", "Na", "Cl"] will assign:
            - Pt = type 1, O = type 2, H = type 3, Na = type 4, Cl = type 5
            Elements not in the structure will still have their masses defined.
            If None (default), uses sequential numbering based on occurrence.
            Only applies when output_format is "lammps".

        log: Enable logging output.

        logger: Custom MLIPLogger instance.

    Examples:
        Simple NaCl solution with default settings:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     box_size=40.0,
        ...     output_file="nacl_solution.data",
        ...     salt_type="NaCl",
        ...     n_salt=100
        ... )

        CaCl2 with ion volume accounting:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     box_size=50.0,
        ...     output_file="cacl2.xyz",
        ...     salt_type="CaCl2",
        ...     n_salt=50,
        ...     include_salt_volume=True,
        ...     output_format="xyz"
        ... )

        Specify water molecules and density (box computed):
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     n_water=1000,
        ...     density=1.1,
        ...     output_file="salt_water.data",
        ...     salt_type="KCl",
        ...     n_salt=30
        ... )

        High concentration with ion volume:
        >>> params = SaltWaterBoxGeneratorParameters(
        ...     box_size=30.0,
        ...     output_file="concentrated.poscar",
        ...     salt_type="NaCl",
        ...     n_salt=200,
        ...     include_salt_volume=True,
        ...     output_format="poscar"
        ... )
    """

    # Required
    output_file: str

    # Box parameters (pick any 2 of 3, same as water_box)
    box_size: float | tuple[float, float, float] | None = None
    n_water: int | None = None
    density: float | None = None  # Total solution density in g/cm³

    # Salt parameters
    salt_type: str = "NaCl"
    n_salt: int = 0  # Number of salt formula units

    # Volume handling
    include_salt_volume: bool = False  # Default: don't account for ion volume

    # Models
    water_model: str = "SPC/E"

    # Packmol parameters
    tolerance: float = 2.0
    seed: int = 12345
    packmol_executable: str = "packmol"

    # Output
    output_format: str = "lammps"
    elements: list[str] | None = None

    # Logging
    log: bool = False
    logger: Optional["MLIPLogger"] = None
