"""Convert command for file format conversions."""

import argparse

from ..converters import mlip_sr_lr, trajectory_to_poscar


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the convert command parser."""
    parser = subparsers.add_parser(
        "convert",
        help="Convert between different file formats",
        description="Convert molecular structure files between different formats",
    )

    # Add subcommands for different conversion types
    convert_subparsers = parser.add_subparsers(
        title="Conversion Types",
        dest="conversion_type",
        help="Type of conversion to perform",
        required=True,
    )

    # Add trajectory-to-poscar subcommand
    trajectory_to_poscar.add_parser(convert_subparsers)

    # Add mlip-sr-lr subcommand
    mlip_sr_lr.add_parser(convert_subparsers)

    # Future conversion types can be added here:
    # poscar_to_xyz.add_parser(convert_subparsers)
    # lammps_to_xyz.add_parser(convert_subparsers)


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the convert command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args.conversion_type == "trajectory-to-poscar":
        return trajectory_to_poscar.handle_command(args)
    elif args.conversion_type == "mlip-sr-lr":
        return mlip_sr_lr.handle_command(args)
    # Future conversion types:
    # elif args.conversion_type == "poscar-to-xyz":
    #     return poscar_to_xyz.handle_command(args)
    else:
        print(f"Unknown conversion type: {args.conversion_type}")
        return 1
