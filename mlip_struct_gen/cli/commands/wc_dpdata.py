"""Wannier center to DPData conversion command."""

import argparse
from pathlib import Path

from mlip_struct_gen.generate_dpdata.wc_dpdata_converter import WCDPDataConverter
from mlip_struct_gen.utils.logger import get_logger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the wc-dpdata command parser."""
    parser = subparsers.add_parser(
        "wc-dpdata",
        help="Convert Wannier center outputs to dpdata format",
        description="Convert Wannier center outputs (wc_out.npy) to dpdata format with atomic dipoles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  mlip-struct-gen wc-dpdata \\
      --input-file-loc directories.txt \\
      --out-dir DATA/water

  # With type map for handling missing elements
  # If POSCAR has O2H4 but type-map is Pt O H, creates Pt0O2H4
  mlip-struct-gen wc-dpdata \\
      --input-file-loc directories.txt \\
      --out-dir DATA/water \\
      --type-map Pt O H

  # Dry run to preview what will be processed
  mlip-struct-gen wc-dpdata \\
      --input-file-loc directories.txt \\
      --out-dir DATA/water \\
      --dry-run

Input file format (directories.txt):
  /path/to/snapshot1/
  /path/to/snapshot2/
  /path/to/snapshot3/
  # Comments are allowed

Each directory should contain:
  - wc_out.npy: Wannier center output from compute-wc
  - POSCAR: Structure file for composition determination

Output structure:
  DATA/water/
  ├── O2H4/
  │   └── set.000/
  │       ├── atomic_dipole.npy  # Shape: (n_samples, n_atoms*3)
  │       ├── coord.npy
  │       ├── box.npy
  │       ├── type.npy
  │       └── type_map.raw
  └── Pt0O2H4/  # If type-map includes Pt
      └── set.000/
          └── ...

Notes:
  - The atomic_dipole.npy contains concatenated wannier centroids
  - For O2H4: 2 O atoms × 3 + 4 H atoms × 3 = 18 values per sample
  - Type-map allows consistent element ordering across different compositions
        """,
    )

    parser.add_argument(
        "--input-file-loc",
        "-i",
        type=str,
        required=True,
        help="Path to text file containing list of directories with wc_out.npy files",
    )

    parser.add_argument(
        "--out-dir",
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
        help="Element type map for consistent ordering (e.g., Pt O H). "
        "If a composition lacks an element from type-map, it's included with count 0.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without converting",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the wc-dpdata command.

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
    logger.info("Wannier Center to DPData Conversion")
    logger.info("=" * 60)
    logger.info(f"Input file: {args.input_file_loc}")
    logger.info(f"Output directory: {args.out_dir}")
    if args.type_map:
        logger.info(f"Type map: {args.type_map}")

    # Check input file exists
    input_file = Path(args.input_file_loc)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    if args.dry_run:
        logger.info("\nDRY RUN MODE - No conversion will be performed")
        logger.info("Reading directory list...")

        # Read and validate directories
        directories = []
        missing_wc = []
        missing_poscar = []

        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    dir_path = Path(line)
                    if dir_path.exists():
                        directories.append(dir_path)

                        # Check for required files
                        if not (dir_path / "wc_out.npy").exists():
                            missing_wc.append(dir_path)
                        if not (dir_path / "POSCAR").exists():
                            missing_poscar.append(dir_path)

        logger.info(f"\nFound {len(directories)} directories")

        if missing_wc:
            logger.warning(f"Directories missing wc_out.npy: {len(missing_wc)}")
            for d in missing_wc[:5]:
                logger.warning(f"  - {d}")
            if len(missing_wc) > 5:
                logger.warning(f"  ... and {len(missing_wc) - 5} more")

        if missing_poscar:
            logger.warning(f"Directories missing POSCAR: {len(missing_poscar)}")
            for d in missing_poscar[:5]:
                logger.warning(f"  - {d}")
            if len(missing_poscar) > 5:
                logger.warning(f"  ... and {len(missing_poscar) - 5} more")

        valid_dirs = [d for d in directories if d not in missing_wc and d not in missing_poscar]
        logger.info(f"\nDirectories ready for processing: {len(valid_dirs)}")

        if valid_dirs:
            logger.info("\nSample directories to be processed:")
            for d in valid_dirs[:10]:
                logger.info(f"  - {d}")
            if len(valid_dirs) > 10:
                logger.info(f"  ... and {len(valid_dirs) - 10} more")

        return 0

    try:
        # Create and run converter
        converter = WCDPDataConverter(
            input_file_loc=args.input_file_loc,
            output_dir=args.out_dir,
            type_map=args.type_map,
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


def main() -> None:
    """
    Main entry point for standalone mlip-wc-dpdata command.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="mlip-wc-dpdata",
        description="Convert Wannier center outputs (wc_out.npy) to dpdata format with atomic dipoles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-file-loc",
        "-i",
        type=str,
        required=True,
        help="Path to text file containing list of directories with wc_out.npy files",
    )

    parser.add_argument(
        "--out-dir",
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
        help="Element type map for consistent ordering (e.g., Pt O H). "
        "If a composition lacks an element from type-map, it's included with count 0.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without converting",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    exit_code = handle_command(args)
    sys.exit(exit_code)
