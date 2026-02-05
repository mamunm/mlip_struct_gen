"""Generate command for creating various structures."""

import argparse

from ..generators import (
    constrained_metal_salt_water,
    constrained_metal_water,
    constrained_salt_water_box,
    constrained_water_box,
    graphene_water,
    metal_salt_water,
    metal_surface,
    metal_water,
    salt_water_box,
    spring_metal_salt_water,
    spring_metal_water,
    spring_salt_water_box,
    spring_water_box,
    water_box,
)


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

    # Add metal-salt-water subcommand
    metal_salt_water.add_parser(structure_subparsers)

    # Add graphene-water subcommand
    graphene_water.add_parser(structure_subparsers)

    # Add constrained-water subcommand
    constrained_water_box.add_parser(structure_subparsers)

    # Add spring-water subcommand
    spring_water_box.add_parser(structure_subparsers)

    # Add constrained-salt-water subcommand
    constrained_salt_water_box.add_parser(structure_subparsers)

    # Add spring-salt-water subcommand
    spring_salt_water_box.add_parser(structure_subparsers)

    # Add constrained-metal-water subcommand
    constrained_metal_water.add_parser(structure_subparsers)

    # Add spring-metal-water subcommand
    spring_metal_water.add_parser(structure_subparsers)

    # Add constrained-metal-salt-water subcommand
    constrained_metal_salt_water.add_parser(structure_subparsers)

    # Add spring-metal-salt-water subcommand
    spring_metal_salt_water.add_parser(structure_subparsers)


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
    elif args.structure_type == "metal-salt-water":
        return metal_salt_water.handle_command(args)
    elif args.structure_type == "graphene-water":
        return graphene_water.handle_command(args)
    elif args.structure_type == "constrained-water":
        return constrained_water_box.handle_command(args)
    elif args.structure_type == "spring-water":
        return spring_water_box.handle_command(args)
    elif args.structure_type == "constrained-salt-water":
        return constrained_salt_water_box.handle_command(args)
    elif args.structure_type == "spring-salt-water":
        return spring_salt_water_box.handle_command(args)
    elif args.structure_type == "constrained-metal-water":
        return constrained_metal_water.handle_command(args)
    elif args.structure_type == "spring-metal-water":
        return spring_metal_water.handle_command(args)
    elif args.structure_type == "constrained-metal-salt-water":
        return constrained_metal_salt_water.handle_command(args)
    elif args.structure_type == "spring-metal-salt-water":
        return spring_metal_salt_water.handle_command(args)
    else:
        print(f"Unknown structure type: {args.structure_type}")
        return 1
