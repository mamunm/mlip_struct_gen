"""CLI interface for water box generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.water_box import WaterBoxGenerator, WaterBoxGeneratorParameters
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the water-box subcommand parser."""
    parser = subparsers.add_parser(
        "water-box",
        help="Generate water box for MD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate water box structures using Packmol",
        epilog="""
Parameter Combinations:
  You can specify any 2 of these 3 parameters:
    - box-size: Dimensions of the simulation box
    - n-water: Number of water molecules
    - density: Water density in g/cm³

Examples:
  1. Box size only (uses default density):
     mlip-struct-gen generate water-box --box-size 30 --output water.xyz

  2. Box size + custom density:
     mlip-struct-gen generate water-box --box-size 30 --density 1.1 --output water.data

  3. Box size + exact molecules:
     mlip-struct-gen generate water-box --box-size 30 --n-water 500 --output water.xyz

  4. Number of molecules only (computes box):
     mlip-struct-gen generate water-box --n-water 1000 --output water.data

  5. Molecules + density (computes box):
     mlip-struct-gen generate water-box --n-water 500 --density 0.92 --output ice.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., water.xyz, water.data, POSCAR)",
    )

    # Box size (can be 1 or 3 values)
    parser.add_argument(
        "--box-size",
        "-b",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Box dimensions in Angstroms. Single value for cubic box, or 3 values for rectangular (x y z)",
    )

    # Number of molecules
    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        metavar="N",
        help="Number of water molecules to generate",
    )

    # Density
    parser.add_argument(
        "--density",
        "-d",
        type=float,
        metavar="RHO",
        help="Water density in g/cm³ (default: model-specific, ~0.997)",
    )

    # Water model
    parser.add_argument(
        "--water-model",
        "-m",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
        default="SPC/E",
        help="Water model to use (default: SPC/E)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "lammps/dpmd", "poscar", "lammpstrj"],
        help="Output file format. If not specified, inferred from file extension",
    )

    # Elements list for LAMMPS
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        metavar="ELEM",
        help="Element order for LAMMPS atom types (e.g., Pt O H Na Cl). Only for LAMMPS format",
    )

    # Packmol parameters
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=2.0,
        metavar="TOL",
        help="Packmol tolerance in Angstroms (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
    )

    # Packmol executable
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
        help="Save intermediate files (packmol.inp, water.xyz) in 'artifacts' directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually running",
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


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check box-size format
    if args.box_size is not None:
        if len(args.box_size) not in [1, 3]:
            logger.error("--box-size must have 1 value (cubic) or 3 values (rectangular)")
            sys.exit(1)
        if len(args.box_size) == 3:
            args.box_size = tuple(args.box_size)
        else:
            args.box_size = args.box_size[0]

    # Count how many of the 3 main parameters are provided
    params_count = sum(
        [
            args.box_size is not None,
            args.n_water is not None,
            args.density is not None,
        ]
    )

    # Check valid combinations
    if params_count == 0:
        logger.error("Must specify at least one of: --box-size, --n-water, --density")
        sys.exit(1)
    elif params_count == 3:
        logger.error("Cannot specify all three: --box-size, --n-water, and --density")
        logger.error("Please specify only 2 of these 3 parameters")
        sys.exit(1)

    # Need at least box_size OR n_molecules
    if args.box_size is None and args.n_water is None:
        logger.error("Must specify either --box-size or --n-water")
        sys.exit(1)

    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' already exists. Use --force to overwrite")
        sys.exit(1)

    # Infer output format from extension if not specified
    if args.output_format is None:
        suffix = output_path.suffix.lower()
        if suffix == ".xyz":
            args.output_format = "xyz"
        elif suffix == ".data":
            args.output_format = "lammps"
        elif suffix == "" or suffix == ".poscar" or output_path.name == "POSCAR":
            args.output_format = "poscar"
        else:
            # Default to xyz if can't infer
            args.output_format = "xyz"
            if args.log:
                logger.warning(f"Could not infer format from '{suffix}', using XYZ format")


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the water-box generation command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Validate arguments
    validate_args(args)

    # Dry run - just show what would be done
    if args.dry_run:
        logger.info("Dry run - would generate water box with:")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Format: {args.output_format}")
        logger.info(f"  Water model: {args.water_model}")
        if args.box_size is not None:
            logger.info(f"  Box size: {args.box_size}")
        if args.n_water is not None:
            logger.info(f"  Molecules: {args.n_water}")
        if args.density is not None:
            logger.info(f"  Density: {args.density} g/cm³")
        logger.info(f"  Tolerance: {args.tolerance} Å")
        logger.info(f"  Seed: {args.seed}")
        return 0

    try:
        # Create parameters
        params = WaterBoxGeneratorParameters(
            output_file=args.output,
            box_size=args.box_size,
            water_model=args.water_model,
            n_water=args.n_water,
            density=args.density,
            tolerance=args.tolerance,
            seed=args.seed,
            packmol_executable=args.packmol_executable,
            output_format=args.output_format,
            elements=args.elements if hasattr(args, "elements") else None,
            log=args.log,
            logger=logger if args.log else None,
        )

        # Save input parameters if requested
        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        # Create generator
        generator = WaterBoxGenerator(params)

        # Generate water box
        if not getattr(args, "quiet", False):
            logger.info("Generating water box...")

        output_file = generator.run(save_artifacts=args.save_artifacts)

        if not getattr(args, "quiet", False):
            logger.info(f"Successfully generated: {output_file}")

            # Print summary information
            if args.box_size is not None:
                if isinstance(args.box_size, tuple):
                    logger.info(
                        f"  Box size: {args.box_size[0]} x {args.box_size[1]} x {args.box_size[2]} Å"
                    )
                else:
                    logger.info(
                        f"  Box size: {args.box_size} x {args.box_size} x {args.box_size} Å"
                    )

            if args.save_artifacts:
                logger.info("  Artifacts saved in 'artifacts' directory")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Standalone entry point for water-box generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-water-box",
        description="Generate water box structures using Packmol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter Combinations:
  You can specify any 2 of these 3 parameters:
    - box-size: Dimensions of the simulation box
    - n-water: Number of water molecules
    - density: Water density in g/cm³

Examples:
  1. Box size only (uses default density):
     mlip-water-box --box-size 30 --output water.xyz

  2. Box size + custom density:
     mlip-water-box --box-size 30 --density 1.1 --output water.data

  3. Box size + exact molecules:
     mlip-water-box --box-size 30 --n-water 500 --output water.xyz

  4. Number of molecules only (computes box):
     mlip-water-box --n-water 1000 --output water.data

  5. Molecules + density (computes box):
     mlip-water-box --n-water 500 --density 0.92 --output ice.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., water.xyz, water.data, POSCAR)",
    )

    # Box size (can be 1 or 3 values)
    parser.add_argument(
        "--box-size",
        "-b",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Box dimensions in Angstroms. Single value for cubic box, or 3 values for rectangular (x y z)",
    )

    # Number of molecules
    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        metavar="N",
        help="Number of water molecules to generate",
    )

    # Density
    parser.add_argument(
        "--density",
        "-d",
        type=float,
        metavar="RHO",
        help="Water density in g/cm³ (default: model-specific, ~0.997)",
    )

    # Water model
    parser.add_argument(
        "--water-model",
        "-m",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
        default="SPC/E",
        help="Water model to use (default: SPC/E)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "lammps/dpmd", "poscar", "lammpstrj"],
        help="Output file format. If not specified, inferred from file extension",
    )

    # Elements list for LAMMPS
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        metavar="ELEM",
        help="Element order for LAMMPS atom types (e.g., Pt O H Na Cl). Only for LAMMPS format",
    )

    # Packmol parameters
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=2.0,
        metavar="TOL",
        help="Packmol tolerance in Angstroms (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
    )

    # Packmol executable
    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable (default: packmol)",
    )

    # Logging and output control
    parser.add_argument(
        "--log",
        "-l",
        action="store_true",
        help="Enable detailed logging",
    )

    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save intermediate files (packmol.inp, water.xyz) in 'artifacts' directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually running",
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
