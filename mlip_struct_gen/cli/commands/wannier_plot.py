"""Wannier centroid plotting and analysis command."""

import argparse
import sys
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the wannier plot command parser."""
    parser = subparsers.add_parser(
        "wc-plot",
        help="Generate wannier centroid distribution plots and reports",
        description="Analyze wannier centroid data from wc_out.txt files and generate statistical reports with plots",
    )

    parser.add_argument(
        "directory",
        type=str,
        default=".",
        nargs="?",
        help="Root directory to search for wc_out.txt files (default: current directory)",
    )

    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="wc_report",
        help="Output base filename for report and plots (default: wc_report)",
    )

    parser.add_argument(
        "--bins",
        "-b",
        type=int,
        default=50,
        help="Number of bins for histograms (default: 50)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the wannier centroid plotting command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from ...wannier_centroid.plot_wannier_report import generate_report

        root_dir = Path(args.directory)

        if not root_dir.exists():
            print(f"Error: Directory '{root_dir}' does not exist")
            return 1

        generate_report(
            root_dir=root_dir, output_name=args.out, bins=args.bins, verbose=args.verbose
        )

        return 0

    except ImportError as e:
        print("Error: Missing required dependencies. Please install matplotlib and scipy.")
        print("  pip install matplotlib scipy")
        if args.verbose:
            print(f"  Import error: {e}")
        return 1

    except Exception as e:
        print(f"Error during wannier centroid analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> None:
    """
    Main entry point for standalone mlip-wc-plot command.
    """
    parser = argparse.ArgumentParser(
        prog="mlip-wc-plot",
        description="Analyze wannier centroid data from wc_out.txt files and generate statistical reports with plots",
    )

    parser.add_argument(
        "directory",
        type=str,
        default=".",
        nargs="?",
        help="Root directory to search for wc_out.txt files (default: current directory)",
    )

    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="wc_report",
        help="Output base filename for report and plots (default: wc_report)",
    )

    parser.add_argument(
        "--bins",
        "-b",
        type=int,
        default=50,
        help="Number of bins for histograms (default: 50)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()
    exit_code = handle_command(args)
    sys.exit(exit_code)
