"""CLI interface for LAMMPS trajectory to POSCAR conversion."""

import argparse
import sys
from pathlib import Path

from ...generate_qm_input.trajectory_to_poscar import TrajectoryToPOSCAR
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the trajectory-to-poscar subcommand parser."""
    parser = subparsers.add_parser(
        "trajectory-to-poscar",
        help="Convert LAMMPS trajectory to VASP POSCAR files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert LAMMPS trajectory files to individual POSCAR files for VASP calculations",
        epilog="""
Examples:
  1. Convert trajectory with default settings:
     mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj

  2. Specify output directory:
     mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --output-dir vasp_snapshots

  3. Custom prefix for snapshot folders:
     mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --prefix frame

  4. Convert specific frame range:
     mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --start-frame 100 --end-frame 200

  5. Sample every nth frame:
     mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --stride 10

Output Structure:
  Creates a directory with individual snapshot folders:
    output_dir/
      snapshot_001/POSCAR
      snapshot_002/POSCAR
      snapshot_003/POSCAR
      ...
        """,
    )

    # Required arguments
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to LAMMPS trajectory file (.lammpstrj or similar)",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="snapshots",
        help="Output directory for POSCAR files (default: snapshots)",
    )

    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="snapshot",
        help="Prefix for snapshot folder names (default: snapshot)",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        help="Starting frame index (0-based, inclusive)",
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        help="Ending frame index (0-based, inclusive)",
    )

    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=1,
        help="Process every nth frame (default: 1, process all frames)",
    )

    parser.add_argument(
        "--sort-elements",
        action="store_true",
        help="Sort atoms by element type in POSCAR (required for VASP)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversion without creating files",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the trajectory-to-poscar conversion command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Validate input file
        trajectory_path = Path(args.trajectory)
        if not trajectory_path.exists():
            logger.error(f"Trajectory file not found: {args.trajectory}")
            return 1

        # Log conversion start
        logger.info("Starting LAMMPS trajectory to POSCAR conversion")
        logger.info(f"Input trajectory: {args.trajectory}")
        logger.info(f"Output directory: {args.output_dir}")

        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be created")

        # Create converter instance
        converter = TrajectoryToPOSCAR(
            trajectory_file=args.trajectory,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )

        # Handle frame selection
        if args.verbose:
            logger.info("Reading trajectory file...")

        # For now, we'll use the existing convert method which processes all frames
        # In a future enhancement, we could add frame selection to the TrajectoryToPOSCAR class
        if args.start_frame or args.end_frame or args.stride != 1:
            logger.warning(
                "Frame selection options (--start-frame, --end-frame, --stride) "
                "not yet implemented. Processing all frames."
            )

        if args.dry_run:
            # Preview mode - just count frames
            from ase.io import read

            snapshots = read(str(trajectory_path), index=":")
            logger.info(f"Would convert {len(snapshots)} frames")
            logger.info(f"Output structure: {args.output_dir}/{args.prefix}_XXX/POSCAR")
            return 0

        # Perform conversion
        n_converted = converter.convert()

        logger.success(f"Successfully converted {n_converted} frames to POSCAR format")
        logger.info(f"Output files saved in: {args.output_dir}/")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Make sure ASE is installed: pip install ase")
        return 1
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for standalone mlip-trajectory-to-poscar command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mlip-trajectory-to-poscar",
        description="Convert LAMMPS trajectory files to individual POSCAR files for VASP calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Convert trajectory with default settings:
     mlip-trajectory-to-poscar trajectory.lammpstrj

  2. Specify output directory:
     mlip-trajectory-to-poscar trajectory.lammpstrj --output-dir vasp_snapshots

  3. Custom prefix for snapshot folders:
     mlip-trajectory-to-poscar trajectory.lammpstrj --prefix frame

Output Structure:
  Creates a directory with individual snapshot folders:
    output_dir/
      snapshot_001/POSCAR
      snapshot_002/POSCAR
      ...
        """,
    )

    # Add arguments directly to parser
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to LAMMPS trajectory file (.lammpstrj or similar)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="snapshots",
        help="Output directory for POSCAR files (default: snapshots)",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="snapshot",
        help="Prefix for snapshot folder names (default: snapshot)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        help="Starting frame index (0-based, inclusive)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="Ending frame index (0-based, inclusive)",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=1,
        help="Process every nth frame (default: 1, process all frames)",
    )
    parser.add_argument(
        "--sort-elements",
        action="store_true",
        help="Sort atoms by element type in POSCAR (required for VASP)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversion without creating files",
    )

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
