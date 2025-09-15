"""Generate command for creating various structures."""

import argparse

from ..generators import metal_surface, metal_water, salt_water_box, water_box


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the generate command parser."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate molecular structures",
        description="Generate various molecular structures for simulations",
    )

    # Add subcommands for different structure types
    structure_subparsers = parser.add_subparsers(
        title="Structure Types",
        dest="structure_type",
        help="Type of structure to generate",
        required=True,
    )

    # Add water-box subcommand
    water_box.add_parser(structure_subparsers)

    # Add salt-water-box subcommand
    salt_water_box.add_parser(structure_subparsers)

    # Add metal-surface subcommand
    metal_surface.add_parser(structure_subparsers)

    # Add metal-water subcommand
    metal_water.add_parser(structure_subparsers)

    # Future structure types can be added here:
    # polymer.add_parser(structure_subparsers)


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the generate command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args.structure_type == "water-box":
        return water_box.handle_command(args)
    elif args.structure_type == "salt-water-box":
        return salt_water_box.handle_command(args)
    elif args.structure_type == "metal-surface":
        return metal_surface.handle_command(args)
    elif args.structure_type == "metal-water":
        return metal_water.handle_command(args)
    # Future structure types:
    #     return interface.handle_command(args)
    else:
        print(f"Unknown structure type: {args.structure_type}")
        return 1