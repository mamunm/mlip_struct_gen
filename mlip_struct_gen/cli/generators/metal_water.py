"""CLI interface for metal-water interface generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.metal_water import MetalWaterGenerator, MetalWaterParameters
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the metal-water subcommand parser."""
    parser = subparsers.add_parser(
        "metal-water",
        help="Generate metal-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate metal-water interface structures with ASE and Packmol",
        epilog="""
Examples:
  1. Basic metal-water interface:
     mlip-struct-gen generate metal-water --metal Pt --size 4 4 6 --n-water 500 --output interface.data

  2. With custom density and gap:
     mlip-struct-gen generate metal-water --metal Au --size 5 5 8 --n-water 600 \\
         --density 1.1 --gap 3.0 --output interface.xyz

  3. Fixed bottom layers for dynamics:
     mlip-struct-gen generate metal-water --metal Cu --size 4 4 6 --n-water 500 \\
         --fix-bottom-layers 2 --output interface.lammps

  4. Custom lattice constant:
     mlip-struct-gen generate metal-water --metal Ag --size 4 4 6 --n-water 500 \\
         --lattice-constant 4.085 --output interface.data

Output formats:
  - .data, .lammps: LAMMPS data file with topology
  - .xyz: XYZ format (no topology information)
  - .poscar, POSCAR: VASP POSCAR format
        """,
    )

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
        help="Surface size: nx ny nz (unit cells in x, y, and layers in z)",
    )

    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        required=True,
        help="Number of water molecules to add",
    )

    # Optional arguments
    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P"],
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
        help="Gap between metal surface and water in Angstroms (default: 0)",
    )

    parser.add_argument(
        "--vacuum",
        "-v",
        type=float,
        default=0,
        help="Vacuum space above water in Angstroms (default: 0)",
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
        help="Number of bottom metal layers to fix (default: 0)",
    )

    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "poscar"],
        help="Output file format (inferred from extension if not specified)",
    )

    # Packmol parameters
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

    # Options
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
        # Default to LAMMPS format
        return "lammps"


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' already exists. Use --force to overwrite")
        sys.exit(1)

    # Check size parameters
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

    # Check water molecules
    if args.n_water < 1:
        logger.error(f"--n-water ({args.n_water}) must be at least 1")
        sys.exit(1)

    if args.density <= 0:
        logger.error(f"--density ({args.density}) must be positive")
        sys.exit(1)

    # Check fix layers
    if args.fix_bottom_layers >= nz:
        logger.error(
            f"--fix-bottom-layers ({args.fix_bottom_layers}) must be less than total layers ({nz})"
        )
        sys.exit(1)

    # Check lattice constant
    if args.lattice_constant is not None and args.lattice_constant <= 0:
        logger.error(f"--lattice-constant ({args.lattice_constant}) must be positive")
        sys.exit(1)

    # Check gap and vacuum
    if args.gap < 0:
        logger.error(f"--gap ({args.gap}) must be non-negative")
        sys.exit(1)

    if args.vacuum < 0:
        logger.error(f"--vacuum ({args.vacuum}) must be non-negative")
        sys.exit(1)

    # Infer output format
    if args.output_format is None:
        args.output_format = infer_output_format(output_path)


def handle_command(args: argparse.Namespace) -> int:
    """Handle the metal-water generation command."""
    # Validate arguments
    validate_args(args)

    # Check if metal is supported
    supported_metals = ["Al", "Cu", "Ni", "Pd", "Ag", "Pt", "Au"]
    if args.metal not in supported_metals:
        logger.error(f"Unknown metal '{args.metal}'. Supported: Al, Cu, Ni, Pd, Ag, Pt, Au")
        sys.exit(1)

    # Get default lattice constants
    default_lattice_constants = {
        "Al": 4.046,
        "Cu": 3.615,
        "Ni": 3.524,
        "Pd": 3.891,
        "Ag": 4.085,
        "Pt": 3.924,
        "Au": 4.078,
    }

    # Warn if custom lattice constant differs significantly from default
    if args.lattice_constant is not None:
        default_lc = default_lattice_constants[args.metal]
        if abs(args.lattice_constant - default_lc) > 0.1:
            if not args.dry_run:
                logger.warning(
                    f"Specified lattice constant {args.lattice_constant} Å differs from "
                    f"default {default_lc} Å for {args.metal}"
                )

    # Dry run
    if args.dry_run:
        logger.info("Dry run - would generate metal-water interface with:")
        logger.info(f"  Metal: {args.metal}")
        logger.info(
            f"  Metal size: {args.size[0]}x{args.size[1]} unit cells, {args.size[2]} layers"
        )
        logger.info(f"  Water molecules: {args.n_water}")
        logger.info(f"  Water model: {args.water_model}")
        logger.info(f"  Water density: {args.density} g/cm^3")
        logger.info(f"  Gap above metal: {args.gap} Angstroms")
        logger.info(f"  Vacuum above water: {args.vacuum} Angstroms")
        if args.lattice_constant:
            logger.info(f"  Lattice constant: {args.lattice_constant} Angstroms")
        if args.fix_bottom_layers > 0:
            logger.info(f"  Fixed bottom layers: {args.fix_bottom_layers}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Format: {args.output_format}")
        return 0

    try:
        # Create logger if requested
        param_logger = logger if args.log else None

        # Create parameters
        params = MetalWaterParameters(
            output_file=args.output,
            metal=args.metal,
            size=tuple(args.size),  # Pass as tuple (nx, ny, nz)
            n_water=args.n_water,
            water_model=args.water_model,
            density=args.density,
            gap_above_metal=args.gap,
            vacuum_above_water=args.vacuum,
            lattice_constant=args.lattice_constant,
            fix_bottom_layers=args.fix_bottom_layers,
            packmol_tolerance=args.tolerance,  # Changed from tolerance to packmol_tolerance
            seed=args.seed,
            packmol_executable=args.packmol_executable,
            output_format=args.output_format,
            log=args.log,
            logger=param_logger,
        )

        # Save input parameters if requested
        if args.save_input:
            save_parameters_to_json(params)

        # Create and run generator
        generator = MetalWaterGenerator(params)

        if not getattr(args, "quiet", False):
            logger.info(f"Generating {args.metal}-water interface...")
            logger.info(f"  Building {args.metal}(111) surface...")

        output_file = generator.run(save_artifacts=args.save_artifacts)

        if not getattr(args, "quiet", False):
            logger.info(f"Successfully generated: {output_file}")

            # Print summary
            logger.info(f"  Metal: {args.metal} ({args.size[0]}x{args.size[1]}x{args.size[2]})")
            logger.info(f"  Water: {args.n_water} {args.water_model} molecules")
            logger.info(f"  Density: {args.density} g/cm^3")
            if args.fix_bottom_layers > 0:
                logger.info(f"  Fixed bottom layers: {args.fix_bottom_layers}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Standalone entry point for metal-water generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-metal-water",
        description="Generate metal-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Basic metal-water interface:
     mlip-metal-water --metal Pt --size 4 4 6 --n-water 500 --output interface.data

  2. With custom density and gap:
     mlip-metal-water --metal Au --size 5 5 8 --n-water 600 \\
         --density 1.1 --gap 3.0 --output interface.xyz

  3. Fixed bottom layers for dynamics:
     mlip-metal-water --metal Cu --size 4 4 6 --n-water 500 \\
         --fix-bottom-layers 2 --output interface.lammps

  4. Custom lattice constant:
     mlip-metal-water --metal Ag --size 4 4 6 --n-water 500 \\
         --lattice-constant 4.085 --output interface.data

Output formats:
  - .data, .lammps: LAMMPS data file with topology
  - .xyz: XYZ format (no topology information)
  - .poscar, POSCAR: VASP POSCAR format
        """,
    )

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
        help="Surface size: nx ny nz (unit cells in x, y, and layers in z)",
    )

    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        required=True,
        help="Number of water molecules to add",
    )

    # Optional arguments
    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P"],
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
        help="Gap between metal surface and water in Angstroms (default: 0)",
    )

    parser.add_argument(
        "--vacuum",
        "-v",
        type=float,
        default=0,
        help="Vacuum space above water in Angstroms (default: 0)",
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
        help="Number of bottom metal layers to fix (default: 0)",
    )

    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "poscar"],
        help="Output file format (inferred from extension if not specified)",
    )

    # Packmol parameters
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

    # Options
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

    # Add verbose/quiet flags for standalone
    parser.add_argument("--verbose", "-vv", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
