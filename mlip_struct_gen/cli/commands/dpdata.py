"""DPData command for converting VASP OUTCARs to dpdata format."""

import argparse

from mlip_struct_gen.generate_dpdata.dpdata_converter import DPDataConverter
from mlip_struct_gen.utils.logger import get_logger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the dpdata command parser."""
    parser = subparsers.add_parser(
        "dpdata",
        help="Convert VASP OUTCARs to dpdata format",
        description="Convert VASP OUTCARs to dpdata format using MultiSystems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all Cu/O/H/Na/Cl systems
  mlip-struct-gen dpdata --input-dir . \\
              --type-map Cu O H Na Cl \\
              --output-dir DATA/

  # Process only water systems
  mlip-struct-gen dpdata --input-dir . \\
              --type-map O H \\
              --output-dir DATA/water_only

  # Process Cu-water systems (no salt)
  mlip-struct-gen dpdata --input-dir . \\
              --type-map Cu O H \\
              --output-dir DATA/cu_water

  # Non-recursive (only immediate subdirectories)
  mlip-struct-gen dpdata --input-dir ./snapshots \\
              --type-map Cu O H Na Cl \\
              --output-dir DATA/ \\
              --no-recursive

Output structure:
  DATA/
  ├── 32Water/          # Pure water systems
  ├── Cu48_32Water/     # Metal-water systems
  ├── 32Water_4NaCl/    # Salt-water systems
  └── Cu32_32Water_4NaCl/  # Metal-salt-water systems

Notes:
  - Systems with elements NOT in type-map will be skipped
  - Each composition is automatically grouped into a MultiSystems
  - Type ordering is preserved from the type-map argument
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Directory containing VASP OUTCARs",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./DATA",
        help="Output directory for dpdata format (default: ./DATA)",
    )

    parser.add_argument(
        "--type-map",
        "-t",
        nargs="+",
        type=str,
        required=True,
        help="Element type map - only systems with these elements will be processed. "
        "Order matters for DeePMD. Example: Cu O H Na Cl",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively for OUTCARs (only immediate subdirectories)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without converting",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the dpdata command.

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
    logger.info("DPData Conversion using MultiSystems")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Type map: {args.type_map}")
    logger.info(f"Recursive search: {not args.no_recursive}")

    if args.dry_run:
        logger.info("\nDRY RUN MODE - No conversion will be performed")
        logger.info("Searching for OUTCAR files...")

        converter = DPDataConverter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            type_map=args.type_map,
            recursive=not args.no_recursive,
            verbose=verbose,
        )

        outcar_files = converter.find_outcars()
        if outcar_files:
            logger.info(f"\nFound {len(outcar_files)} OUTCAR files")

            # Sample a few to check compositions
            logger.info("\nSample OUTCAR locations:")
            for outcar_path in outcar_files[:5]:
                logger.info(f"  - {outcar_path}")
            if len(outcar_files) > 5:
                logger.info(f"  ... and {len(outcar_files) - 5} more")

            # Try to load a few to show compositions
            logger.info("\nSample compositions (first 3 valid systems):")
            sample_count = 0
            for outcar_path in outcar_files[:20]:  # Check up to 20 to find 3 valid
                try:
                    import dpdata

                    sys_test = dpdata.LabeledSystem(str(outcar_path), fmt="vasp/outcar")
                    elements = set(sys_test.data.get("atom_names", []))
                    allowed_elements = set(args.type_map)

                    if elements.issubset(allowed_elements):
                        logger.info(f"  {outcar_path.parent.name}: {elements}")
                        sample_count += 1
                        if sample_count >= 3:
                            break
                except Exception:
                    continue

        else:
            logger.info("No OUTCAR files found")

        return 0

    try:
        # Create and run converter
        converter = DPDataConverter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            type_map=args.type_map,
            recursive=not args.no_recursive,
            verbose=verbose,
        )

        converter.run()
        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1
