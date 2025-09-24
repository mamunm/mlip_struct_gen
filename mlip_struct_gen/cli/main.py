#!/usr/bin/env python3
"""Main entry point for mlip-struct-gen CLI."""

import argparse
import sys

from .commands import convert, dpdata, generate, wannier, wannier_plot, wc_dpdata


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlip-struct-gen",
        description="Machine Learning Interatomic Potential Structure Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate water box with default density:
    mlip-struct-gen generate water-box --box-size 30 --output water.xyz

  Generate water box with custom density:
    mlip-struct-gen generate water-box --box-size 30 --density 1.1 --output water.data

  Generate water box with exact molecules:
    mlip-struct-gen generate water-box --n-molecules 500 --output water.xyz

For more help on specific commands:
    mlip-struct-gen generate --help
    mlip-struct-gen generate water-box --help
        """,
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        help="Available commands",
        required=True,
    )

    # Add generate command
    generate.add_parser(subparsers)

    # Add convert command
    convert.add_parser(subparsers)

    # Add dpdata command
    dpdata.add_parser(subparsers)

    # Add wannier command
    wannier.add_parser(subparsers)

    # Add wannier plot command
    wannier_plot.add_parser(subparsers)

    # Add wannier center to dpdata command
    wc_dpdata.add_parser(subparsers)

    # Future commands can be added here:
    # analyze.add_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (for testing). If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()

    # Parse arguments
    args = parser.parse_args(argv)

    # Set up logging based on verbose/quiet flags
    if args.quiet and args.verbose:
        parser.error("Cannot use --quiet and --verbose together")

    # Handle commands
    if args.command == "generate":
        return generate.handle_command(args)
    elif args.command == "convert":
        return convert.handle_command(args)
    elif args.command == "dpdata":
        return dpdata.handle_command(args)
    elif args.command == "compute-wc":
        return wannier.handle_command(args)
    elif args.command == "wc-plot":
        return wannier_plot.handle_command(args)
    elif args.command == "wc-dpdata":
        return wc_dpdata.handle_command(args)

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
