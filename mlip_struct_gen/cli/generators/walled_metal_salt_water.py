#!/usr/bin/env python3
"""Command-line interface for walled metal-salt-water interface generation."""

import argparse
import sys

from mlip_struct_gen.generate_structure.walled_metal_salt_water.generate_walled_metal_salt_water import (
    WalledMetalSaltWaterGenerator,
)
from mlip_struct_gen.generate_structure.walled_metal_salt_water.input_parameters import (
    WalledMetalSaltWaterParameters,
)
from mlip_struct_gen.generate_structure.walled_metal_salt_water.validation import (
    validate_parameters,
)
from mlip_struct_gen.utils.json_utils import save_parameters_to_json
from mlip_struct_gen.utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the walled-metal-salt-water subcommand to the main parser."""
    parser = subparsers.add_parser(
        "walled-metal-salt-water",
        help="Generate walled metal-salt-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate walled metal-salt-water interface with metal walls on top and bottom",
        epilog="""
Examples:
  # Basic walled metal-salt-water interface
  mlip-struct-gen generate walled-metal-salt-water -o interface.data --metal Pt --salt NaCl \\
      --n-salt 3 --n-water 50 --size 4 4 6 --box-z 60

  # With symmetric gap and fixed layers
  mlip-struct-gen generate walled-metal-salt-water -o interface.data --metal Au --salt KCl \\
      --n-salt 3 --n-water 50 --size 4 4 6 --box-z 80 --gap 3.0 --vacuum 3.0 \\
      --fix-bottom-layers 1

  # Include salt volume in density calculation
  mlip-struct-gen generate walled-metal-salt-water -o interface.data --metal Cu --salt LiCl \\
      --n-salt 3 --n-water 50 --size 4 4 6 --box-z 60 --include-salt-volume

Output formats:
  - .lmp, .lammps, .data: LAMMPS data file with topology
  - .xyz: XYZ format (no topology)
  - .poscar, .vasp: VASP POSCAR format (no topology)
        """,
    )

    _add_arguments(parser)
    parser.set_defaults(func=handle_command)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments to the parser."""
    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (determines format by extension)",
    )

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
        help="Size of metal slab in unit cells (total layers across both walls)",
    )

    parser.add_argument(
        "--box-z",
        type=float,
        required=True,
        help="Total z-dimension of the simulation box in Angstroms",
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
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
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
        help="Gap between bottom metal wall and solution in Angstroms (default: 0.0)",
    )

    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Gap between solution top and top metal wall in Angstroms (default: 0.0)",
    )

    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        help="Number of layers to fix in each wall symmetrically (default: 0)",
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
        "--no-salt-zone",
        type=float,
        default=0.2,
        help="Fraction of solution box height to exclude ions from top/bottom (default: 0.2)",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "lammps/dpmd", "data", "lammpstrj"],
        help="Output file format (overrides extension detection)",
    )

    parser.add_argument(
        "--elements",
        nargs="+",
        metavar="ELEM",
        help="Element order for LAMMPS atom types (e.g., Pt O H Na Cl)",
    )

    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to input_params.json",
    )

    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save intermediate files to <output>_artifacts/",
    )


def handle_command(args: argparse.Namespace) -> int:
    """Execute walled metal-salt-water interface generation."""
    try:
        nz = args.size[2]
        n_bottom = (nz + 1) // 2
        n_top = nz - n_bottom

        params = WalledMetalSaltWaterParameters(
            output_file=args.output,
            metal=args.metal,
            size=tuple(args.size),
            salt_type=args.salt,
            n_salt=args.n_salt,
            n_water=args.n_water,
            box_z=args.box_z,
            water_model=args.water_model,
            density=args.density,
            include_salt_volume=args.include_salt_volume,
            lattice_constant=args.lattice_constant,
            gap_above_metal=args.gap,
            vacuum_above_water=args.vacuum,
            fix_bottom_layers=args.fix_bottom_layers,
            seed=args.seed,
            packmol_tolerance=args.tolerance,
            no_salt_zone=args.no_salt_zone,
            save_artifacts=getattr(args, "save_artifacts", False),
            output_format=args.output_format,
            elements=args.elements if hasattr(args, "elements") else None,
        )

        try:
            validate_parameters(params)
        except (ValueError, RuntimeError) as e:
            logger.error(str(e))
            return 1

        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        logger.info(f"Generating walled {args.metal}-salt-water interface...")
        logger.info(
            f"  Building {args.metal}(111) walled surface "
            f"({args.size[0]}x{args.size[1]}x{args.size[2]})..."
        )
        logger.info(f"  Bottom wall: {n_bottom} layers, Top wall: {n_top} layers")
        logger.info(f"  Adding {args.n_water} {args.water_model} water molecules...")
        logger.info(f"  Adding {args.n_salt} {args.salt} formula units...")
        logger.info(f"  Box z: {args.box_z} Angstroms")

        generator = WalledMetalSaltWaterGenerator(params)
        generator.generate()

        logger.info(f"Successfully generated: {args.output}")
        logger.info(f"  Metal: {args.metal} ({args.size[0]}x{args.size[1]}x{args.size[2]})")
        logger.info(f"  Salt: {args.salt} ({args.n_salt} formula units)")
        logger.info(f"  Water: {args.n_water} {args.water_model} molecules")
        logger.info(f"  Density: {args.density} g/cm^3")
        logger.info(f"  Box z: {args.box_z} Angstroms")
        if args.fix_bottom_layers > 0:
            logger.info(f"  Fixed layers per wall: {args.fix_bottom_layers}")
        return 0

    except Exception as e:
        logger.error(f"Error generating walled metal-salt-water interface: {e}")
        return 1


def main() -> int:
    """Standalone entry point for walled metal-salt-water interface generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-walled-metal-salt-water",
        description="Generate walled metal-salt-water interface structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic walled metal-salt-water interface
  mlip-walled-metal-salt-water -o interface.lmp --metal Pt --salt NaCl \\
      --n-salt 10 --n-water 500 --size 4 4 6 --box-z 60

  # With symmetric gap and fixed layers
  mlip-walled-metal-salt-water -o interface.lmp --metal Au --salt KCl \\
      --n-salt 10 --n-water 500 --size 4 4 6 --box-z 80 --gap 3.0 --vacuum 3.0 \\
      --fix-bottom-layers 1

Output formats:
  - .lmp, .lammps, .data: LAMMPS data file with topology
  - .xyz: XYZ format (no topology)
  - .poscar, .vasp: VASP POSCAR format (no topology)
        """,
    )

    _add_arguments(parser)

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
