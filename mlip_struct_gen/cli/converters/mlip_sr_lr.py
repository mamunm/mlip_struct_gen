"""CLI interface for MLIP SR-LR converter."""

import argparse
import sys
from pathlib import Path

from ...converter.mlip_sr_lr_convert import (
    LAMMPSDataProcessor,
    parse_charge_map,
    parse_duplication_spec,
)
from ...utils.logger import get_logger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the mlip-sr-lr converter subcommand parser."""
    parser = subparsers.add_parser(
        "mlip-sr-lr",
        help="Convert LAMMPS data files for SR/LR splitting in MLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Process LAMMPS data files for short-range/long-range splitting in machine learning interatomic potentials",
        epilog="""
Examples:
  Duplicate only type 2 as type 4 with bond type 1:
    mlip-struct-gen convert mlip-sr-lr input.data output.data \\
      --type-map C H O --duplicate "2:4:1"

  Multiple duplications with charges:
    mlip-struct-gen convert mlip-sr-lr input.data output.data \\
      --type-map Pt O H Na Cl \\
      --duplicate "2:6:1" "4:7:2" "5:8:3" \\
      --charge-map "1:0" "2:-2" "3:1" "4:1" "5:-1"

  No duplication, just reformat with charges:
    mlip-struct-gen convert mlip-sr-lr input.data output.data \\
      --type-map C H O --charge-map 0.0 0.5 -0.5

  Duplicate with bond displacement:
    mlip-struct-gen convert mlip-sr-lr input.data output.data \\
      --type-map C H O --duplicate "2:4:1" --bond-length 1.5

Notes:
  - Duplication format: "original_type:new_type:bond_type"
  - Charge format: "type:charge" or positional values
  - Duplicated atoms inherit charges from originals unless overridden
  - Output is in LAMMPS atom style full format
        """,
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Input LAMMPS data file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output LAMMPS data file",
    )
    parser.add_argument(
        "--type-map",
        nargs="+",
        required=True,
        help="Element names for atom types (e.g., Pt O H Na Cl)",
    )

    # Optional arguments
    parser.add_argument(
        "--duplicate",
        nargs="*",
        default=[],
        help='Duplication specs: "orig:new:bond" (e.g., "2:6:1" "4:7:2")',
    )
    parser.add_argument(
        "--charge-map",
        nargs="*",
        default=[],
        help='Charge specs: "type:charge" or positional (e.g., "1:0" "2:-1" or 0 -1 1)',
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.0,
        help="Displacement for duplicated atoms in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the mlip-sr-lr conversion command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger = get_logger()

    # Set verbose mode if global verbose flag is set
    verbose = getattr(args, "verbose", False)
    if verbose:
        import logging

        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("MLIP SR-LR Converter")
    logger.info("=" * 60)
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Type map: {args.type_map}")
    if args.duplicate:
        logger.info(f"Duplication specs: {args.duplicate}")
    if args.charge_map:
        logger.info(f"Charge map: {args.charge_map}")
    if args.bond_length > 0:
        logger.info(f"Bond displacement: {args.bond_length} Å")

    # Check input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file '{args.input_file}' not found")
        return 1

    # Check output file
    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite:
        response = input(f"Output file '{args.output_file}' exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            logger.info("Operation cancelled.")
            return 1

    try:
        # Parse duplication configuration
        duplication_config = parse_duplication_spec(args.duplicate)

        # Create temporary processor to get atom type info for charge parsing
        temp_processor = LAMMPSDataProcessor(
            str(input_path), str(output_path), args.type_map, [], {}
        )
        temp_processor.read_input_file()

        # Get max type for charge parsing
        max_type = (
            max(atom[1] for atom in temp_processor.atoms)
            if temp_processor.atoms
            else len(args.type_map)
        )

        # Parse charge map
        charge_map = parse_charge_map(args.charge_map, duplication_config, max_type)

        # Create processor with full configuration
        processor = LAMMPSDataProcessor(
            str(input_path),
            str(output_path),
            args.type_map,
            duplication_config,
            charge_map,
            args.bond_length,
        )

        if args.validate_only:
            processor.read_input_file()
            processor.validate_configuration()
            logger.success("Configuration validated successfully!")
            return 0
        else:
            # Process the file
            processor.process()
            return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for standalone mlip-sr-lr-convert command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mlip-sr-lr-convert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Process LAMMPS data files for short-range/long-range splitting in machine learning interatomic potentials",
        epilog="""
Examples:
  Duplicate only type 2 as type 4 with bond type 1:
    mlip-sr-lr-convert input.data output.data \\
      --type-map C H O --duplicate "2:4:1"

  Multiple duplications with charges:
    mlip-sr-lr-convert input.data output.data \\
      --type-map Pt O H Na Cl \\
      --duplicate "2:6:1" "4:7:2" "5:8:3" \\
      --charge-map "1:0" "2:-2" "3:1" "4:1" "5:-1"

  No duplication, just reformat with charges:
    mlip-sr-lr-convert input.data output.data \\
      --type-map C H O --charge-map 0.0 0.5 -0.5

  Duplicate with bond displacement:
    mlip-sr-lr-convert input.data output.data \\
      --type-map C H O --duplicate "2:4:1" --bond-length 1.5

Notes:
  - Duplication format: "original_type:new_type:bond_type"
  - Charge format: "type:charge" or positional values
  - Duplicated atoms inherit charges from originals unless overridden
  - Output is in LAMMPS atom style full format
        """,
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Input LAMMPS data file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output LAMMPS data file",
    )
    parser.add_argument(
        "--type-map",
        nargs="+",
        required=True,
        help="Element names for atom types (e.g., Pt O H Na Cl)",
    )

    # Optional arguments
    parser.add_argument(
        "--duplicate",
        nargs="*",
        default=[],
        help='Duplication specs: "orig:new:bond" (e.g., "2:6:1" "4:7:2")',
    )
    parser.add_argument(
        "--charge-map",
        nargs="*",
        default=[],
        help='Charge specs: "type:charge" or positional (e.g., "1:0" "2:-1" or 0 -1 1)',
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.0,
        help="Displacement for duplicated atoms in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
