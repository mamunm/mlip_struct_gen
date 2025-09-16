"""CLI interface for salt water box generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.salt_water_box import (
    SaltWaterBoxGenerator,
    SaltWaterBoxGeneratorParameters,
)
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the salt-water-box subcommand parser."""
    parser = subparsers.add_parser(
        "salt-water-box",
        help="Generate salt water box for MD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate salt water box structures using Packmol",
        epilog="""
Parameter Combinations:
  Same as water-box - specify any 2 of these 3 parameters:
    - box-size: Dimensions of the simulation box
    - n-water: Number of water molecules
    - density: Total solution density in g/cm³

Salt Parameters:
  - salt-type: Type of salt (NaCl, KCl, CaCl2, MgCl2, etc.)
  - n-salt: Number of salt formula units

Examples:
  1. Box size with NaCl:
     mlip-struct-gen generate salt-water-box --box-size 40 --salt-type NaCl \\
       --n-salt 100 --output nacl_solution.data

  2. Include ion volume for concentrated solution:
     mlip-struct-gen generate salt-water-box --box-size 30 --salt-type NaCl \\
       --n-salt 200 --include-salt-volume --output concentrated.xyz

  3. CaCl2 solution (2:1 stoichiometry handled automatically):
     mlip-struct-gen generate salt-water-box --box-size 50 --salt-type CaCl2 \\
       --n-salt 50 --output cacl2_solution.data

  4. Specify water molecules and density (box computed):
     mlip-struct-gen generate salt-water-box --n-water 1000 --density 1.1 \\
       --salt-type KCl --n-salt 30 --output kcl_solution.poscar
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., salt_water.xyz, salt_water.data, POSCAR)",
    )

    # Box parameters (same as water-box)
    parser.add_argument(
        "--box-size",
        "-b",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Box dimensions in Angstroms. Single value for cubic, or 3 values for rectangular",
    )

    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        metavar="N",
        help="Number of water molecules",
    )

    parser.add_argument(
        "--density",
        "-d",
        type=float,
        metavar="RHO",
        help="Total solution density in g/cm³ (default: water model density)",
    )

    # Salt parameters
    parser.add_argument(
        "--salt-type",
        "-s",
        type=str,
        choices=["NaCl", "KCl", "LiCl", "CaCl2", "MgCl2", "NaBr", "KBr", "CsCl"],
        default="NaCl",
        help="Type of salt to add (default: NaCl)",
    )

    parser.add_argument(
        "--n-salt",
        type=int,
        default=0,
        metavar="N",
        help="Number of salt formula units (default: 0)",
    )

    parser.add_argument(
        "--include-salt-volume",
        action="store_true",
        help="Account for ion volume using VDW radii (default: False)",
    )

    # Water model
    parser.add_argument(
        "--water-model",
        "-m",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P"],
        default="SPC/E",
        help="Water model to use (default: SPC/E)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "poscar"],
        help="Output file format. If not specified, inferred from extension",
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
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
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
            print(
                "Error: --box-size must have 1 value (cubic) or 3 values (rectangular)",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(args.box_size) == 3:
            args.box_size = tuple(args.box_size)
        else:
            args.box_size = args.box_size[0]

    # Count main parameters
    params_count = sum(
        [
            args.box_size is not None,
            args.n_water is not None,
            args.density is not None,
        ]
    )

    # Check valid combinations
    if params_count == 0:
        print(
            "Error: Must specify at least one of: --box-size, --n-water, --density", file=sys.stderr
        )
        sys.exit(1)
    elif params_count == 3:
        print(
            "Error: Cannot specify all three: --box-size, --n-water, and --density", file=sys.stderr
        )
        print("       Please specify only 2 of these 3 parameters", file=sys.stderr)
        sys.exit(1)

    # Need at least box_size OR n_water
    if args.box_size is None and args.n_water is None:
        print("Error: Must specify either --box-size or --n-water", file=sys.stderr)
        sys.exit(1)

    # Check that include-salt-volume is not used with box-size
    if args.box_size is not None and args.include_salt_volume:
        print(
            "Error: Cannot use --include-salt-volume when --box-size is specified", file=sys.stderr
        )
        print(
            "       Ion volume adjustment is only available when box size is computed",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(
            f"Error: Output file '{args.output}' already exists. Use --force to overwrite",
            file=sys.stderr,
        )
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
            # Default to lammps for salt water
            args.output_format = "lammps"
            if args.log:
                print(
                    f"Warning: Could not infer format from '{suffix}', using LAMMPS format",
                    file=sys.stderr,
                )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the salt-water-box generation command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Validate arguments
    validate_args(args)

    # Dry run
    if args.dry_run:
        print("Dry run - would generate salt water box with:")
        print(f"  Output: {args.output}")
        print(f"  Format: {args.output_format}")
        print(f"  Water model: {args.water_model}")
        print(f"  Salt: {args.salt_type} ({args.n_salt} formula units)")
        if args.box_size is not None:
            print(f"  Box size: {args.box_size}")
        if args.n_water is not None:
            print(f"  Water molecules: {args.n_water}")
        if args.density is not None:
            print(f"  Density: {args.density} g/cm³")
        if args.include_salt_volume:
            print("  Including ion volume in calculations")
        print(f"  Tolerance: {args.tolerance} Å")
        print(f"  Seed: {args.seed}")
        return 0

    try:
        # Create logger if requested
        logger = MLIPLogger() if args.log else None

        # Create parameters
        params = SaltWaterBoxGeneratorParameters(
            output_file=args.output,
            box_size=args.box_size,
            n_water=args.n_water,
            density=args.density,
            salt_type=args.salt_type,
            n_salt=args.n_salt,
            include_salt_volume=args.include_salt_volume,
            water_model=args.water_model,
            tolerance=args.tolerance,
            seed=args.seed,
            packmol_executable=args.packmol_executable,
            output_format=args.output_format,
            log=args.log,
            logger=logger,
        )

        # Save input parameters if requested
        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        # Create generator
        generator = SaltWaterBoxGenerator(params)

        # Generate salt water box
        if not getattr(args, "quiet", False):
            print("Generating salt water box...")

        output_file = generator.run(save_artifacts=args.save_artifacts)

        if not getattr(args, "quiet", False):
            print(f"Successfully generated: {output_file}")

            # Print summary
            if args.box_size is not None:
                if isinstance(args.box_size, tuple):
                    print(
                        f"  Box size: {args.box_size[0]} x {args.box_size[1]} x {args.box_size[2]} Å"
                    )
                else:
                    print(f"  Box size: {args.box_size} x {args.box_size} x {args.box_size} Å")

            print(f"  Salt: {args.salt_type} ({args.n_salt} formula units)")

            if args.save_artifacts:
                print("  Artifacts saved in 'artifacts' directory")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Standalone entry point for salt-water-box generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-salt-water-box",
        description="Generate salt water box structures using Packmol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter Combinations:
  Same as water-box - specify any 2 of these 3 parameters:
    - box-size: Dimensions of the simulation box
    - n-water: Number of water molecules
    - density: Total solution density in g/cm³

Salt Parameters:
  - salt-type: Type of salt (NaCl, KCl, CaCl2, MgCl2, etc.)
  - n-salt: Number of salt formula units

Examples:
  1. Box size with NaCl:
     mlip-salt-water-box --box-size 40 --salt-type NaCl \\
       --n-salt 100 --output nacl_solution.data

  2. Include ion volume for concentrated solution:
     mlip-salt-water-box --box-size 30 --salt-type NaCl \\
       --n-salt 200 --include-salt-volume --output concentrated.xyz

  3. CaCl2 solution (2:1 stoichiometry handled automatically):
     mlip-salt-water-box --box-size 50 --salt-type CaCl2 \\
       --n-salt 50 --output cacl2_solution.data

  4. Specify water molecules and density (box computed):
     mlip-salt-water-box --n-water 1000 --density 1.1 \\
       --salt-type KCl --n-salt 30 --output kcl_solution.poscar
        """,
    )

    # Add verbose/quiet flags for standalone
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., salt_water.xyz, salt_water.data, POSCAR)",
    )

    # Box parameters (same as water-box)
    parser.add_argument(
        "--box-size",
        "-b",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Box dimensions in Angstroms. Single value for cubic, or 3 values for rectangular",
    )

    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        metavar="N",
        help="Number of water molecules",
    )

    parser.add_argument(
        "--density",
        "-d",
        type=float,
        metavar="RHO",
        help="Total solution density in g/cm³ (default: water model density)",
    )

    # Salt parameters
    parser.add_argument(
        "--salt-type",
        "-s",
        type=str,
        choices=["NaCl", "KCl", "LiCl", "CaCl2", "MgCl2", "NaBr", "KBr", "CsCl"],
        default="NaCl",
        help="Type of salt to add (default: NaCl)",
    )

    parser.add_argument(
        "--n-salt",
        type=int,
        default=0,
        metavar="N",
        help="Number of salt formula units (default: 0)",
    )

    parser.add_argument(
        "--include-salt-volume",
        action="store_true",
        help="Account for ion volume using VDW radii (default: False)",
    )

    # Water model
    parser.add_argument(
        "--water-model",
        "-m",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P"],
        default="SPC/E",
        help="Water model to use (default: SPC/E)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "lammps", "poscar"],
        help="Output file format. If not specified, inferred from extension",
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
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
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

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
