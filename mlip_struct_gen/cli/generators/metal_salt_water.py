#!/usr/bin/env python3
"""Command-line interface for metal-salt-water interface generation."""

import argparse
import sys

from mlip_struct_gen.generate_structure.metal_salt_water.generate_metal_salt_water import (
    MetalSaltWaterGenerator,
)
from mlip_struct_gen.generate_structure.metal_salt_water.input_parameters import (
    MetalSaltWaterParameters,
)
from mlip_struct_gen.generate_structure.metal_salt_water.validation import validate_parameters
from mlip_struct_gen.utils.json_utils import save_parameters_to_json


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the metal-salt-water subcommand to the main parser.

    Args:
        subparsers: The subparsers object from the main argument parser.
    """
    parser = subparsers.add_parser(
        "metal-salt-water",
        help="Generate metal-salt-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate metal-salt-water interface structures using ASE and Packmol",
        epilog="""
Examples:
  # Basic metal-salt-water interface with Pt metal and NaCl salt
  mlip generate metal-salt-water -o interface.lmp --metal Pt --salt NaCl \\
      --n-salt 10 --n-water 500 --size 4 4 6

  # Interface with custom lattice constant and fixed layers
  mlip generate metal-salt-water -o interface.lmp --metal Au --salt KCl \\
      --n-salt 10 --n-water 500 --size 4 4 6 --lattice-constant 4.078 \\
      --fix-bottom-layers 2

  # Include salt volume in density calculation
  mlip generate metal-salt-water -o interface.lmp --metal Cu --salt LiCl \\
      --n-salt 10 --n-water 500 --size 4 4 6 --include-salt-volume

  # Use different water model with CaCl2 (2:1 stoichiometry)
  mlip generate metal-salt-water -o interface.lmp --metal Ag --salt CaCl2 \\
      --n-salt 5 --n-water 500 --size 4 4 6 --water-model TIP4P

Output formats:
  - .lmp, .lammps, .data: LAMMPS data file with topology
  - .xyz: XYZ format (no topology)
  - .poscar, .vasp: VASP POSCAR format (no topology)
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (determines format by extension)",
    )

    # Metal parameters
    parser.add_argument(
        "--metal",
        type=str,
        required=True,
        help="FCC metal element symbol (e.g., Pt, Au, Ag, Cu, Ni, Pd, Al)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Size of metal slab in unit cells (nx ny nz)",
    )

    # Salt parameters
    parser.add_argument(
        "--salt",
        type=str,
        required=True,
        help="Salt type (NaCl, KCl, LiCl, CaCl2, MgCl2, NaBr, KBr, CsCl)",
    )
    parser.add_argument(
        "--n-salt",
        type=int,
        required=True,
        help="Number of salt molecules",
    )

    # Water parameters
    parser.add_argument(
        "--n-water",
        type=int,
        required=True,
        help="Number of water molecules",
    )
    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P"],
        help="Water model to use (default: SPC/E)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Target water density in g/cm^3 (default: 1.0)",
    )
    parser.add_argument(
        "--include-salt-volume",
        action="store_true",
        help="Include salt ions in density calculation for solution height",
    )

    # Optional metal parameters
    parser.add_argument(
        "--lattice-constant",
        type=float,
        help="Metal lattice constant in Angstroms (uses default if not specified)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Gap between metal surface and water in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum spacing above water in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        help="Number of bottom metal layers to fix (default: 0)",
    )
    parser.add_argument(
        "--orthogonalize",
        action="store_true",
        default=True,
        help="Orthogonalize the unit cell for LAMMPS (default: True)",
    )

    # Packing parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for Packmol (default: 12345)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Minimum distance between molecules in Angstroms (default: 2.0)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "data"],
        help="Output file format (overrides extension detection)",
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to input_params.json",
    )

    parser.set_defaults(func=handle_command)


def handle_command(args: argparse.Namespace) -> int:
    """Execute metal-salt-water interface generation from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Create parameters object
        params = MetalSaltWaterParameters(
            output_file=args.output,
            metal=args.metal,
            metal_size=tuple(args.size),
            salt_type=args.salt,
            n_salt_molecules=args.n_salt,
            n_water_molecules=args.n_water,
            water_model=args.water_model,
            water_density=args.density,
            include_salt_volume=args.include_salt_volume,
            lattice_constant=args.lattice_constant,
            gap=args.gap,
            vacuum_above_water=args.vacuum,
            fix_bottom_layers=args.fix_bottom_layers,
            seed=args.seed,
            packmol_tolerance=args.tolerance,
            output_format=args.output_format,
        )

        # Validate parameters
        try:
            validate_parameters(params)
        except (ValueError, RuntimeError) as e:
            error_message = str(e)
            print(f"Error: {error_message}", file=sys.stderr)
            return 1

        # Save input parameters if requested
        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        # Generate structure
        generator = MetalSaltWaterGenerator(params)
        generator.generate()

        print(f"Successfully generated metal-salt-water interface: {args.output}")
        return 0

    except Exception as e:
        print(f"Error generating metal-salt-water interface: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Standalone entry point for metal-salt-water interface generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-metal-salt-water",
        description="Generate metal-salt-water interface structures using ASE and Packmol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic metal-salt-water interface with Pt metal and NaCl salt
  mlip-metal-salt-water -o interface.lmp --metal Pt --salt NaCl \\
      --n-salt 10 --n-water 500 --size 4 4 6

  # Interface with custom lattice constant and fixed layers
  mlip-metal-salt-water -o interface.lmp --metal Au --salt KCl \\
      --n-salt 10 --n-water 500 --size 4 4 6 --lattice-constant 4.078 \\
      --fix-bottom-layers 2

  # Include salt volume in density calculation
  mlip-metal-salt-water -o interface.lmp --metal Cu --salt LiCl \\
      --n-salt 10 --n-water 500 --size 4 4 6 --include-salt-volume

  # Use different water model with CaCl2 (2:1 stoichiometry)
  mlip-metal-salt-water -o interface.lmp --metal Ag --salt CaCl2 \\
      --n-salt 5 --n-water 500 --size 4 4 6 --water-model TIP4P

Output formats:
  - .lmp, .lammps, .data: LAMMPS data file with topology
  - .xyz: XYZ format (no topology)
  - .poscar, .vasp: VASP POSCAR format (no topology)
        """,
    )

    # Add all arguments directly to parser
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (determines format by extension)",
    )

    # Metal parameters
    parser.add_argument(
        "--metal",
        type=str,
        required=True,
        help="FCC metal element symbol (e.g., Pt, Au, Ag, Cu, Ni, Pd, Al)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Size of metal slab in unit cells (nx ny nz)",
    )

    # Salt parameters
    parser.add_argument(
        "--salt",
        type=str,
        required=True,
        help="Salt type (NaCl, KCl, LiCl, CaCl2, MgCl2, NaBr, KBr, CsCl)",
    )
    parser.add_argument(
        "--n-salt",
        type=int,
        required=True,
        help="Number of salt molecules",
    )

    # Water parameters
    parser.add_argument(
        "--n-water",
        type=int,
        required=True,
        help="Number of water molecules",
    )
    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P"],
        help="Water model to use (default: SPC/E)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Target water density in g/cm^3 (default: 1.0)",
    )
    parser.add_argument(
        "--include-salt-volume",
        action="store_true",
        help="Include salt ions in density calculation for solution height",
    )

    # Optional metal parameters
    parser.add_argument(
        "--lattice-constant",
        type=float,
        help="Metal lattice constant in Angstroms (uses default if not specified)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Gap between metal surface and water in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum spacing above water in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        help="Number of bottom metal layers to fix (default: 0)",
    )
    parser.add_argument(
        "--orthogonalize",
        action="store_true",
        default=True,
        help="Orthogonalize the unit cell for LAMMPS (default: True)",
    )

    # Packing parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for Packmol (default: 12345)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Minimum distance between molecules in Angstroms (default: 2.0)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "data"],
        help="Output file format (overrides extension detection)",
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to input_params.json",
    )

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
