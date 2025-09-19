"""Wannier centroid computation command."""

import argparse
import sys
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the wannier command parser."""
    parser = subparsers.add_parser(
        "compute-wc",
        help="Compute wannier centroids from wannier centers",
        description="Compute wannier centroids for atoms from POSCAR and wannier90_centres.xyz files",
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing POSCAR and wannier90_centres.xyz files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information during computation",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the wannier centroid computation command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from ...wannier_centroid import compute_wannier_centroid

        folder_path = Path(args.folder)

        if not folder_path.exists():
            print(f"Error: Folder '{folder_path}' does not exist")
            return 1

        poscar_path = folder_path / "POSCAR"
        wannier_path = folder_path / "wannier90_centres.xyz"

        if not poscar_path.exists():
            print(f"Error: POSCAR not found at {poscar_path}")
            return 1

        if not wannier_path.exists():
            print(f"Error: wannier90_centres.xyz not found at {wannier_path}")
            return 1

        compute_wannier_centroid.main(str(folder_path), verbose=args.verbose)

        return 0

    except Exception as e:
        print(f"Error during wannier centroid computation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> None:
    """
    Main entry point for standalone mlip-compute-wc command.
    """
    parser = argparse.ArgumentParser(
        prog="mlip-compute-wc",
        description="Compute wannier centroids for atoms from POSCAR and wannier90_centres.xyz files",
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing POSCAR and wannier90_centres.xyz files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information during computation",
    )

    args = parser.parse_args()
    exit_code = handle_command(args)
    sys.exit(exit_code)
