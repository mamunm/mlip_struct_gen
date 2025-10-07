"""CLI interface for graphene-water interface generation."""

import argparse
import sys

from ...generate_structure.graphene_water import GrapheneWaterGenerator, GrapheneWaterParameters
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the graphene-water subcommand parser."""
    parser = subparsers.add_parser(
        "graphene-water",
        help="Generate graphene-water interface for MD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate graphene monolayer with water using ASE and PACKMOL",
        epilog="""
Examples:
  1. Basic graphene-water interface:
     mlip-struct-gen generate graphene-water --size 20 20 --n-water 500 --output graphene_water.xyz

  2. With custom gap and density:
     mlip-struct-gen generate graphene-water --size 30 30 --n-water 1000 --gap 3.3 --density 0.997 --output graphene_water.vasp

  3. Custom lattice constant:
     mlip-struct-gen generate graphene-water --size 25 25 --n-water 750 --a 2.45 --output graphene_water.data

  4. With vacuum above water:
     mlip-struct-gen generate graphene-water --size 20 20 --n-water 500 --vacuum 10.0 --output graphene_water.lammps

  5. Using TIP3P water model:
     mlip-struct-gen generate graphene-water --size 20 20 --n-water 500 --water-model TIP3P --output graphene_water.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path. Extension determines format (.xyz, .vasp, .lammps, .data, .lammpstrj)",
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        required=True,
        help="Graphene size as (nx, ny) unit cells. Example: --size 20 20",
    )

    parser.add_argument(
        "--n-water",
        type=int,
        required=True,
        help="Number of water molecules to add above graphene",
    )

    # Graphene parameters
    parser.add_argument(
        "--a",
        type=float,
        default=2.46,
        help="Graphene lattice constant in Angstroms (default: 2.46)",
    )

    parser.add_argument(
        "--thickness",
        type=float,
        default=0.0,
        help="Graphene thickness in Angstroms (default: 0.0 for true 2D sheet)",
    )

    parser.add_argument(
        "--graphene-vacuum",
        type=float,
        default=0.0,
        help="In-plane vacuum around graphene edges in Angstroms (default: 0.0)",
    )

    # Water parameters
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Target water density in g/cm³ (default: 1.0)",
    )

    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
        help="Water model to use (default: SPC/E)",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Gap between graphene and water in Angstroms (default: 0.0)",
    )

    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum space above water in Angstroms (default: 0.0)",
    )

    # Packmol parameters
    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable (default: packmol)",
    )

    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol tolerance for packing in Angstroms (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducible water configurations (default: 12345)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "lammps/dpmd", "lammpstrj"],
        help="Output format (default: auto-detect from file extension)",
    )

    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        help="Element order for LAMMPS atom types. Example: --elements C O H Na Cl",
    )

    # Additional options
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save parameters to JSON file",
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging output",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the graphene-water generation command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Create parameters
        params = GrapheneWaterParameters(
            size=tuple(args.size),
            n_water=args.n_water,
            output_file=args.output,
            a=args.a,
            thickness=args.thickness,
            graphene_vacuum=args.graphene_vacuum,
            water_density=args.density,
            gap_above_graphene=args.gap,
            vacuum_above_water=args.vacuum,
            water_model=args.water_model,
            packmol_executable=args.packmol_executable,
            packmol_tolerance=args.packmol_tolerance,
            seed=args.seed,
            output_format=args.output_format,
            elements=args.elements,
            log=not args.no_log,
            logger=logger if not args.no_log else None,
        )

        # Save JSON if requested
        if args.save_json:
            save_parameters_to_json(params, args.save_json)
            if not args.no_log:
                logger.info(f"Parameters saved to: {args.save_json}")

        # Create generator and run
        if not args.no_log:
            logger.info("Starting graphene-water interface generation")

        generator = GrapheneWaterGenerator(params)
        output_file = generator.generate()

        if not args.no_log:
            logger.success(f"Successfully created graphene-water interface: {output_file}")

        return 0

    except KeyboardInterrupt:
        if not args.no_log:
            logger.warning("Generation interrupted by user")
        return 130

    except Exception as e:
        if not args.no_log:
            logger.error(f"Generation failed: {e}")
        return 1


def main() -> int:
    """Standalone entry point for graphene-water generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-graphene-water",
        description="Generate graphene monolayer with water using ASE and PACKMOL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Basic graphene-water interface:
     mlip-graphene-water --size 20 20 --n-water 500 --output graphene_water.xyz

  2. With custom gap and density:
     mlip-graphene-water --size 30 30 --n-water 1000 --gap 3.3 --density 0.997 --output graphene_water.vasp

  3. Custom lattice constant:
     mlip-graphene-water --size 25 25 --n-water 750 --a 2.45 --output graphene_water.data

  4. With vacuum above water:
     mlip-graphene-water --size 20 20 --n-water 500 --vacuum 10.0 --output graphene_water.lammps

  5. Using TIP3P water model:
     mlip-graphene-water --size 20 20 --n-water 500 --water-model TIP3P --output graphene_water.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path. Extension determines format (.xyz, .vasp, .lammps, .data, .lammpstrj)",
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        required=True,
        help="Graphene size as (nx, ny) unit cells. Example: --size 20 20",
    )

    parser.add_argument(
        "--n-water",
        type=int,
        required=True,
        help="Number of water molecules to add above graphene",
    )

    # Graphene parameters
    parser.add_argument(
        "--a",
        type=float,
        default=2.46,
        help="Graphene lattice constant in Angstroms (default: 2.46)",
    )

    parser.add_argument(
        "--thickness",
        type=float,
        default=0.0,
        help="Graphene thickness in Angstroms (default: 0.0 for true 2D sheet)",
    )

    parser.add_argument(
        "--graphene-vacuum",
        type=float,
        default=0.0,
        help="In-plane vacuum around graphene edges in Angstroms (default: 0.0)",
    )

    # Water parameters
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Target water density in g/cm³ (default: 1.0)",
    )

    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
        help="Water model to use (default: SPC/E)",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Gap between graphene and water in Angstroms (default: 0.0)",
    )

    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum space above water in Angstroms (default: 0.0)",
    )

    # Packmol parameters
    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable (default: packmol)",
    )

    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol tolerance for packing in Angstroms (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducible water configurations (default: 12345)",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "lammps/dpmd", "lammpstrj"],
        help="Output format (default: auto-detect from file extension)",
    )

    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        help="Element order for LAMMPS atom types. Example: --elements C O H Na Cl",
    )

    # Additional options
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save parameters to JSON file",
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
