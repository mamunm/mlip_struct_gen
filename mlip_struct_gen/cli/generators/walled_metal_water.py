"""CLI interface for walled metal-water interface generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.walled_metal_water import (
    WalledMetalWaterGenerator,
    WalledMetalWaterParameters,
)
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the walled-metal-water subcommand parser."""
    parser = subparsers.add_parser(
        "walled-metal-water",
        help="Generate walled metal-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate walled metal-water interface with metal walls on top and bottom",
        epilog="""
Examples:
  1. Basic walled metal-water interface:
     mlip-struct-gen generate walled-metal-water --metal Pt --size 4 4 6 --n-water 500 \\
         --box-z 60 --output interface.data

  2. With symmetric gap:
     mlip-struct-gen generate walled-metal-water --metal Au --size 5 5 8 --n-water 600 \\
         --gap 3.0 --vacuum 3.0 --box-z 80 --output interface.xyz

  3. Fixed layers in both walls:
     mlip-struct-gen generate walled-metal-water --metal Cu --size 4 4 6 --n-water 500 \\
         --fix-bottom-layers 1 --box-z 60 --output interface.lammps

Output formats:
  - .data, .lammps: LAMMPS data file with topology
  - .xyz: XYZ format (no topology information)
  - .poscar, POSCAR: VASP POSCAR format
        """,
    )

    _add_arguments(parser)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments to the parser."""
    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path",
    )

    parser.add_argument(
        "--metal",
        "-m",
        type=str,
        required=True,
        choices=["Al", "Cu", "Ni", "Pd", "Ag", "Pt", "Au"],
        help="Metal element for the surface",
    )

    parser.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Surface size: nx ny nz (total layers across both walls)",
    )

    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        required=True,
        help="Number of water molecules to pack between walls",
    )

    parser.add_argument(
        "--box-z",
        type=float,
        required=True,
        help="Total z-dimension of the simulation box in Angstroms",
    )

    # Optional arguments
    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
        help="Water model (default: SPC/E)",
    )

    parser.add_argument(
        "--density",
        "-d",
        type=float,
        default=0.997,
        help="Water density in g/cm^3 (default: 0.997)",
    )

    parser.add_argument(
        "--gap",
        "-g",
        type=float,
        default=0,
        help="Gap between bottom metal wall and water in Angstroms (default: 0)",
    )

    parser.add_argument(
        "--vacuum",
        "-v",
        type=float,
        default=0,
        help="Gap between water top and top metal wall in Angstroms (default: 0)",
    )

    parser.add_argument(
        "--lattice-constant",
        "-a",
        type=float,
        default=None,
        help="Metal lattice constant in Angstroms (default: element-specific)",
    )

    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        help="Number of layers to fix in each wall symmetrically (default: 0)",
    )

    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "lammps/dpmd", "poscar", "lammpstrj"],
        help="Output file format (inferred from extension if not specified)",
    )

    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        metavar="ELEM",
        help="Element order for LAMMPS atom types (e.g., Pt O H)",
    )

    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=2.0,
        help="Packmol tolerance in Angstroms (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for Packmol (default: 12345)",
    )

    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable (default: packmol)",
    )

    parser.add_argument(
        "--log",
        "-l",
        action="store_true",
        help="Enable detailed logging",
    )

    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save intermediate files in 'artifacts' directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to input_params.json",
    )


def infer_output_format(output_path: Path) -> str:
    """Infer output format from file extension."""
    suffix = output_path.suffix.lower()
    name = output_path.name.lower()

    if suffix in [".data", ".lammps", ".lmp"]:
        return "lammps"
    elif suffix == ".xyz":
        return "xyz"
    elif suffix == ".poscar" or name == "poscar":
        return "poscar"
    else:
        return "lammps"


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' already exists. Use --force to overwrite")
        sys.exit(1)

    if len(args.size) != 3:
        logger.error("--size must have exactly 3 values (nx ny nz)")
        sys.exit(1)

    nx, ny, nz = args.size
    if nx < 1 or ny < 1:
        logger.error(f"Lateral dimensions (nx={nx}, ny={ny}) must be at least 1")
        sys.exit(1)

    if nz < 3:
        logger.error(f"Number of layers (nz={nz}) must be at least 3")
        sys.exit(1)

    if args.n_water < 1:
        logger.error(f"--n-water ({args.n_water}) must be at least 1")
        sys.exit(1)

    if args.box_z <= 0:
        logger.error(f"--box-z ({args.box_z}) must be positive")
        sys.exit(1)

    if args.density <= 0:
        logger.error(f"--density ({args.density}) must be positive")
        sys.exit(1)

    n_bottom = (nz + 1) // 2
    n_top = nz - n_bottom
    max_fixable = min(n_bottom, n_top) - 1
    if args.fix_bottom_layers > max_fixable:
        logger.error(
            f"--fix-bottom-layers ({args.fix_bottom_layers}) too large for walled geometry. "
            f"Bottom wall: {n_bottom} layers, top wall: {n_top} layers. Max fixable: {max(0, max_fixable)}"
        )
        sys.exit(1)

    if args.lattice_constant is not None and args.lattice_constant <= 0:
        logger.error(f"--lattice-constant ({args.lattice_constant}) must be positive")
        sys.exit(1)

    if args.gap < 0:
        logger.error(f"--gap ({args.gap}) must be non-negative")
        sys.exit(1)

    if args.vacuum < 0:
        logger.error(f"--vacuum ({args.vacuum}) must be non-negative")
        sys.exit(1)

    if args.output_format is None:
        args.output_format = infer_output_format(output_path)


def handle_command(args: argparse.Namespace) -> int:
    """Handle the walled-metal-water generation command."""
    validate_args(args)

    nz = args.size[2]
    n_bottom = (nz + 1) // 2
    n_top = nz - n_bottom

    if args.dry_run:
        logger.info("Dry run - would generate walled metal-water interface with:")
        logger.info(f"  Metal: {args.metal}")
        logger.info(f"  Metal size: {args.size[0]}x{args.size[1]} cells, {nz} total layers")
        logger.info(f"  Bottom wall: {n_bottom} layers, Top wall: {n_top} layers")
        logger.info(f"  Water molecules: {args.n_water}")
        logger.info(f"  Water model: {args.water_model}")
        logger.info(f"  Water density: {args.density} g/cm^3")
        logger.info(f"  Gap above metal: {args.gap} Angstroms")
        logger.info(f"  Vacuum above water: {args.vacuum} Angstroms")
        logger.info(f"  Box z: {args.box_z} Angstroms")
        if args.fix_bottom_layers > 0:
            logger.info(f"  Fixed layers per wall: {args.fix_bottom_layers}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Format: {args.output_format}")
        return 0

    try:
        param_logger = logger if args.log else None

        params = WalledMetalWaterParameters(
            output_file=args.output,
            metal=args.metal,
            size=tuple(args.size),
            n_water=args.n_water,
            box_z=args.box_z,
            water_model=args.water_model,
            density=args.density,
            gap_above_metal=args.gap,
            vacuum_above_water=args.vacuum,
            lattice_constant=args.lattice_constant,
            fix_bottom_layers=args.fix_bottom_layers,
            packmol_tolerance=args.tolerance,
            seed=args.seed,
            packmol_executable=args.packmol_executable,
            output_format=args.output_format,
            elements=args.elements if hasattr(args, "elements") else None,
            save_artifacts=getattr(args, "save_artifacts", False),
            log=args.log,
            logger=param_logger,
        )

        if args.save_input:
            save_parameters_to_json(params)

        generator = WalledMetalWaterGenerator(params)

        if not getattr(args, "quiet", False):
            logger.info(f"Generating walled {args.metal}-water interface...")
            logger.info(f"  Bottom wall: {n_bottom} layers, Top wall: {n_top} layers")

        output_file = generator.run(save_artifacts=getattr(args, "save_artifacts", False))

        if not getattr(args, "quiet", False):
            logger.info(f"Successfully generated: {output_file}")
            logger.info(f"  Metal: {args.metal} ({args.size[0]}x{args.size[1]}x{args.size[2]})")
            logger.info(f"  Water: {args.n_water} {args.water_model} molecules")
            logger.info(f"  Box z: {args.box_z} Angstroms")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Standalone entry point for walled metal-water generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-walled-metal-water",
        description="Generate walled metal-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Basic walled metal-water interface:
     mlip-walled-metal-water --metal Pt --size 4 4 6 --n-water 50 \\
         --box-z 60 --output interface.data

  2. With symmetric gap:
     mlip-walled-metal-water --metal Au --size 5 5 8 --n-water 60 \\
         --gap 3.0 --vacuum 3.0 --box-z 80 --output interface.xyz

Output formats:
  - .data, .lammps: LAMMPS data file with topology
  - .xyz: XYZ format (no topology information)
  - .poscar, POSCAR: VASP POSCAR format
        """,
    )

    _add_arguments(parser)

    parser.add_argument("--verbose", "-vv", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
