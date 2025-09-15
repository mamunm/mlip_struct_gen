"""CLI interface for metal-water interface generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.metal_water import (
    MetalWaterGenerator,
    MetalWaterParameters,
)
from ...utils.logger import MLIPLogger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the metal-water subcommand parser."""
    parser = subparsers.add_parser(
        "metal-water",
        help="Generate FCC(111) metal surfaces with water layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate FCC(111) metal surfaces with water layers using ASE and PACKMOL",
        epilog="""
Metal-Water Interface Generation:
  Creates FCC(111) metal surfaces with water molecules above them,
  suitable for metal-water interface MD simulations.

Supported Metals:
  Al, Au, Ag, Cu, Ni, Pd, Pt, Pb, Rh, Ir, Ca, Sr, Yb

Water Models:
  - SPC/E: Extended Simple Point Charge model (default)
  - TIP3P: Three-site transferable intermolecular potential
  - TIP4P: Four-site transferable intermolecular potential

Output Formats:
  Automatically detected from file extension:
    - .xyz: XYZ coordinate file
    - .vasp, .poscar, POSCAR: VASP POSCAR format
    - .lammps, .data: LAMMPS data file (includes water topology)

Examples:
  1. Basic Pt-water interface:
     mlip-struct-gen generate metal-water --metal Pt --metal-size 4 4 4 \\
       --n-water 100 --output pt_water.data

  2. Gold-water with custom parameters:
     mlip-struct-gen generate metal-water --metal Au --metal-size 5 5 6 \\
       --n-water 200 --water-density 0.997 --gap 3.5 --vacuum 10 \\
       --fix-bottom-layers 2 --output au_water.vasp

  3. Large copper-water system for MD:
     mlip-struct-gen generate metal-water --metal Cu --metal-size 10 10 8 \\
       --n-water 500 --water-model TIP3P --lattice-constant 3.615 \\
       --fix-bottom-layers 3 --output cu_water_large.lammps

  4. Aluminum-water with specific density:
     mlip-struct-gen generate metal-water --metal Al --metal-size 6 6 5 \\
       --n-water 300 --water-density 1.05 --gap 4.0 \\
       --output al_water_interface.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (e.g., metal_water.xyz, metal_water.data, POSCAR)",
    )

    # Metal parameters
    parser.add_argument(
        "--metal", "-m",
        type=str,
        default="Pt",
        choices=["Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt", "Pb", "Rh", "Ir", "Ca", "Sr", "Yb"],
        help="Metal element symbol (default: Pt)",
    )

    parser.add_argument(
        "--metal-size",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("NX", "NY", "NZ"),
        help="Metal surface size as nx ny nz (default: 4 4 4)",
    )

    # Water parameters
    parser.add_argument(
        "--n-water", "-n",
        type=int,
        required=True,
        metavar="N",
        help="Number of water molecules (required)",
    )

    parser.add_argument(
        "--water-density", "-d",
        type=float,
        default=1.0,
        metavar="RHO",
        help="Water density in g/cm^3 (default: 1.0)",
    )

    parser.add_argument(
        "--water-model", "-w",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P"],
        default="SPC/E",
        help="Water model (default: SPC/E)",
    )

    # Interface parameters
    parser.add_argument(
        "--gap", "-g",
        type=float,
        default=3.0,
        metavar="GAP",
        help="Gap between metal surface and water in Angstroms (default: 3.0)",
    )

    parser.add_argument(
        "--vacuum", "-v",
        type=float,
        default=0.0,
        metavar="VACUUM",
        help="Vacuum space above water in Angstroms (default: 0.0)",
    )

    # Optional parameters
    parser.add_argument(
        "--lattice-constant", "-a",
        type=float,
        metavar="A",
        help="Custom lattice constant in Angstroms (uses default if not specified)",
    )

    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        metavar="N",
        help="Number of bottom metal layers to fix (default: 0)",
    )

    # PACKMOL parameters
    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable (default: packmol)",
    )

    parser.add_argument(
        "--packmol-tolerance", "-t",
        type=float,
        default=2.0,
        metavar="TOL",
        help="Packmol tolerance in Angstroms (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
    )

    # Output format
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "data"],
        help="Output file format. If not specified, inferred from extension",
    )

    # Options
    parser.add_argument(
        "--log", "-l",
        action="store_true",
        help="Enable detailed logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Error: Output file '{args.output}' already exists. Use --force to overwrite", file=sys.stderr)
        sys.exit(1)

    # Validate metal_size
    if len(args.metal_size) != 3:
        print("Error: --metal-size must have exactly 3 values (nx ny nz)", file=sys.stderr)
        sys.exit(1)

    nx, ny, nz = args.metal_size
    if nx < 1 or ny < 1:
        print(f"Error: Lateral dimensions (nx={nx}, ny={ny}) must be at least 1", file=sys.stderr)
        sys.exit(1)

    if nz < 3:
        print(f"Error: Number of layers (nz={nz}) must be at least 3", file=sys.stderr)
        sys.exit(1)

    # Validate water parameters
    if args.n_water < 1:
        print(f"Error: --n-water ({args.n_water}) must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.water_density <= 0:
        print(f"Error: --water-density ({args.water_density}) must be positive", file=sys.stderr)
        sys.exit(1)

    # Validate fix_bottom_layers
    if args.fix_bottom_layers < 0:
        print(f"Error: --fix-bottom-layers ({args.fix_bottom_layers}) must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.fix_bottom_layers >= nz:
        print(f"Error: --fix-bottom-layers ({args.fix_bottom_layers}) must be less than nz ({nz})", file=sys.stderr)
        sys.exit(1)

    # Validate gap and vacuum
    if args.gap < 0:
        print(f"Error: --gap ({args.gap}) must be non-negative", file=sys.stderr)
        sys.exit(1)

    if args.vacuum < 0:
        print(f"Error: --vacuum ({args.vacuum}) must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Validate lattice constant if provided
    if args.lattice_constant is not None and args.lattice_constant <= 0:
        print(f"Error: --lattice-constant ({args.lattice_constant}) must be positive", file=sys.stderr)
        sys.exit(1)

    # Validate PACKMOL tolerance
    if args.packmol_tolerance <= 0:
        print(f"Error: --packmol-tolerance ({args.packmol_tolerance}) must be positive", file=sys.stderr)
        sys.exit(1)

    # Infer output format from extension if not specified
    if args.output_format is None:
        suffix = output_path.suffix.lower()
        if suffix == ".xyz":
            args.output_format = "xyz"
        elif suffix in [".vasp", ".poscar"] or output_path.name.upper() == "POSCAR":
            args.output_format = "poscar"
        elif suffix in [".lammps", ".data"]:
            args.output_format = "lammps"
        else:
            # Default to lammps for metal-water
            args.output_format = "lammps"
            if args.log:
                print(f"Warning: Could not infer format from '{suffix}', using LAMMPS format", file=sys.stderr)


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the metal-water generation command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Validate arguments
    validate_args(args)

    # Dry run
    if args.dry_run:
        print("Dry run - would generate metal-water interface with:")
        print(f"  Metal: {args.metal}")
        print(f"  Metal size: {args.metal_size[0]}x{args.metal_size[1]} unit cells, {args.metal_size[2]} layers")
        print(f"  Water molecules: {args.n_water}")
        print(f"  Water model: {args.water_model}")
        print(f"  Water density: {args.water_density} g/cm^3")
        print(f"  Gap above metal: {args.gap} �")
        print(f"  Vacuum above water: {args.vacuum} �")
        if args.lattice_constant:
            print(f"  Lattice constant: {args.lattice_constant} �")
        if args.fix_bottom_layers > 0:
            print(f"  Fixed bottom layers: {args.fix_bottom_layers}")
        print(f"  Output: {args.output}")
        print(f"  Format: {args.output_format}")
        return 0

    try:
        # Create logger if requested
        logger = MLIPLogger() if args.log else None

        # Create parameters
        params = MetalWaterParameters(
            metal=args.metal,
            metal_size=tuple(args.metal_size),
            n_water_molecules=args.n_water,
            output_file=args.output,
            water_density=args.water_density,
            gap_above_metal=args.gap,
            vacuum_above_water=args.vacuum,
            water_model=args.water_model,
            lattice_constant=args.lattice_constant,
            fix_bottom_layers=args.fix_bottom_layers,
            packmol_executable=args.packmol_executable,
            packmol_tolerance=args.packmol_tolerance,
            seed=args.seed,
            output_format=args.output_format,
            log=args.log,
            logger=logger,
        )

        # Create generator
        generator = MetalWaterGenerator(params)

        # Generate interface
        if not getattr(args, 'quiet', False):
            print(f"Generating {args.metal}-water interface...")
            print(f"  Building {args.metal}(111) surface...")

        output_file = generator.generate()

        if not getattr(args, 'quiet', False):
            print(f"Successfully generated: {output_file}")

            # Print summary
            print(f"  Metal: {args.metal} ({args.metal_size[0]}x{args.metal_size[1]}x{args.metal_size[2]})")
            print(f"  Water: {args.n_water} {args.water_model} molecules")
            print(f"  Density: {args.water_density} g/cm^3")
            if args.fix_bottom_layers > 0:
                print(f"  Fixed bottom layers: {args.fix_bottom_layers}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Standalone entry point for metal-water generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-metal-water",
        description="Generate FCC(111) metal surfaces with water layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metal-Water Interface Generation:
  Creates FCC(111) metal surfaces with water molecules above them.

Supported Metals:
  Al, Au, Ag, Cu, Ni, Pd, Pt, Pb, Rh, Ir, Ca, Sr, Yb

Water Models: SPC/E (default), TIP3P, TIP4P

Examples:
  1. Basic Pt-water interface:
     mlip-metal-water --metal Pt --metal-size 4 4 4 --n-water 100 \\
       --output pt_water.data

  2. Gold-water with custom parameters:
     mlip-metal-water --metal Au --metal-size 5 5 6 --n-water 200 \\
       --water-density 0.997 --gap 3.5 --vacuum 10 \\
       --fix-bottom-layers 2 --output au_water.vasp

  3. Large copper-water system:
     mlip-metal-water --metal Cu --metal-size 10 10 8 --n-water 500 \\
       --water-model TIP3P --lattice-constant 3.615 \\
       --output cu_water_large.lammps
        """,
    )

    # Add verbose/quiet flags for standalone
    parser.add_argument("--verbose", "-V", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    # Required arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path",
    )

    # Metal parameters
    parser.add_argument(
        "--metal", "-m",
        type=str,
        default="Pt",
        choices=["Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt", "Pb", "Rh", "Ir", "Ca", "Sr", "Yb"],
        help="Metal element symbol (default: Pt)",
    )

    parser.add_argument(
        "--metal-size",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("NX", "NY", "NZ"),
        help="Metal surface size (default: 4 4 4)",
    )

    # Water parameters
    parser.add_argument(
        "--n-water", "-n",
        type=int,
        required=True,
        help="Number of water molecules",
    )

    parser.add_argument(
        "--water-density", "-d",
        type=float,
        default=1.0,
        help="Water density in g/cm^3 (default: 1.0)",
    )

    parser.add_argument(
        "--water-model", "-w",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P"],
        default="SPC/E",
        help="Water model (default: SPC/E)",
    )

    # Interface parameters
    parser.add_argument(
        "--gap", "-g",
        type=float,
        default=3.0,
        help="Gap between metal and water in � (default: 3.0)",
    )

    parser.add_argument(
        "--vacuum", "-v",
        type=float,
        default=0.0,
        help="Vacuum above water in � (default: 0.0)",
    )

    # Optional parameters
    parser.add_argument(
        "--lattice-constant", "-a",
        type=float,
        help="Custom lattice constant in �",
    )

    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        help="Number of bottom metal layers to fix (default: 0)",
    )

    # PACKMOL parameters
    parser.add_argument(
        "--packmol-executable",
        type=str,
        default="packmol",
        help="Path to packmol executable",
    )

    parser.add_argument(
        "--packmol-tolerance", "-t",
        type=float,
        default=2.0,
        help="Packmol tolerance in � (default: 2.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345)",
    )

    # Output format
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "data"],
        help="Output file format",
    )

    # Options
    parser.add_argument(
        "--log", "-l",
        action="store_true",
        help="Enable detailed logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())