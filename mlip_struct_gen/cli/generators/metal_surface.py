"""CLI interface for metal surface generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.metal_surface import MetalSurfaceGenerator, MetalSurfaceParameters
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the metal-surface subcommand parser."""
    parser = subparsers.add_parser(
        "metal-surface",
        help="Generate FCC(111) metal surfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate FCC(111) metal surfaces using ASE",
        epilog="""
Supported Metals:
  Al, Au, Ag, Cu, Ni, Pd, Pt, Pb, Rh, Ir, Ca, Sr, Yb

Size Parameter:
  Specify surface dimensions as three integers:
    - First two: lateral dimensions (nx, ny) in unit cells
    - Third: number of atomic layers (nz)
  Example: --size 4 4 6 creates a 4x4 surface with 6 layers

Output Formats:
  Automatically detected from file extension:
    - .xyz: XYZ coordinate file
    - .vasp, .poscar, POSCAR: VASP POSCAR format
    - .lammps, .data: LAMMPS data file

Examples:
  1. Basic Pt(111) surface:
     mlip-struct-gen generate metal-surface --metal Pt --size 4 4 5 \\
       --vacuum 15 --output pt_111.xyz

  2. Gold surface for LAMMPS with fixed bottom layers:
     mlip-struct-gen generate metal-surface --metal Au --size 5 5 6 \\
       --vacuum 12 --fix-bottom-layers 2 --output au_111.data

  3. Copper surface with custom lattice constant:
     mlip-struct-gen generate metal-surface --metal Cu --size 3 3 4 \\
       --vacuum 10 --lattice-constant 3.62 --output cu_111.vasp

  4. Large aluminum surface:
     mlip-struct-gen generate metal-surface --metal Al --size 10 10 8 \\
       --vacuum 20 --orthogonalize --output al_111_large.xyz
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., metal.xyz, metal.data, POSCAR)",
    )

    # Metal selection
    parser.add_argument(
        "--metal",
        "-m",
        type=str,
        default="Pt",
        choices=["Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt", "Pb", "Rh", "Ir", "Ca", "Sr", "Yb"],
        help="Metal element symbol (default: Pt)",
    )

    # Size parameters
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("NX", "NY", "NZ"),
        help="Surface size as nx ny nz (default: 4 4 4)",
    )

    # Vacuum
    parser.add_argument(
        "--vacuum",
        "-v",
        type=float,
        default=15.0,
        metavar="VACUUM",
        help="Vacuum space above surface in Angstroms (default: 15.0)",
    )

    # Optional parameters
    parser.add_argument(
        "--lattice-constant",
        "-a",
        type=float,
        metavar="A",
        help="Custom lattice constant in Angstroms (uses default if not specified)",
    )

    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        metavar="N",
        help="Number of bottom layers to fix (default: 0)",
    )

    parser.add_argument(
        "--orthogonalize",
        action="store_true",
        default=True,
        help="Create orthogonal unit cell (default: True, required for LAMMPS)",
    )

    parser.add_argument(
        "--no-orthogonalize",
        action="store_false",
        dest="orthogonalize",
        help="Keep non-orthogonal unit cell",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "data"],
        help="Output file format. If not specified, inferred from extension",
    )

    # Options
    parser.add_argument(
        "--log",
        "-l",
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

    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to input_params.json",
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
        print(
            f"Error: Output file '{args.output}' already exists. Use --force to overwrite",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate size
    if len(args.size) != 3:
        print("Error: --size must have exactly 3 values (nx ny nz)", file=sys.stderr)
        sys.exit(1)

    nx, ny, nz = args.size
    if nx < 1 or ny < 1:
        print(f"Error: Lateral dimensions (nx={nx}, ny={ny}) must be at least 1", file=sys.stderr)
        sys.exit(1)

    if nz < 3:
        print(f"Error: Number of layers (nz={nz}) must be at least 3", file=sys.stderr)
        sys.exit(1)

    # Validate fix_bottom_layers
    if args.fix_bottom_layers < 0:
        print(
            f"Error: --fix-bottom-layers ({args.fix_bottom_layers}) must be non-negative",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.fix_bottom_layers >= nz:
        print(
            f"Error: --fix-bottom-layers ({args.fix_bottom_layers}) must be less than nz ({nz})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate vacuum
    if args.vacuum < 0:
        print(f"Error: --vacuum ({args.vacuum}) must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Validate lattice constant if provided
    if args.lattice_constant is not None and args.lattice_constant <= 0:
        print(
            f"Error: --lattice-constant ({args.lattice_constant}) must be positive", file=sys.stderr
        )
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
            # Default to xyz
            args.output_format = "xyz"
            if args.log:
                print(
                    f"Warning: Could not infer format from '{suffix}', using XYZ format",
                    file=sys.stderr,
                )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the metal-surface generation command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Validate arguments
    validate_args(args)

    # Dry run
    if args.dry_run:
        print("Dry run - would generate metal surface with:")
        print(f"  Metal: {args.metal}")
        print(f"  Size: {args.size[0]}x{args.size[1]} unit cells, {args.size[2]} layers")
        print(f"  Vacuum: {args.vacuum} Å")
        if args.lattice_constant:
            print(f"  Lattice constant: {args.lattice_constant} Å")
        if args.fix_bottom_layers > 0:
            print(f"  Fixed bottom layers: {args.fix_bottom_layers}")
        print(f"  Orthogonal cell: {args.orthogonalize}")
        print(f"  Output: {args.output}")
        print(f"  Format: {args.output_format}")
        return 0

    try:
        # Create logger if requested
        logger = MLIPLogger() if args.log else None

        # Create parameters
        params = MetalSurfaceParameters(
            metal=args.metal,
            size=tuple(args.size),
            vacuum=args.vacuum,
            output_file=args.output,
            lattice_constant=args.lattice_constant,
            fix_bottom_layers=args.fix_bottom_layers,
            orthogonalize=args.orthogonalize,
            output_format=args.output_format,
            log=args.log,
            logger=logger,
        )

        # Save input parameters if requested
        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        # Create generator
        generator = MetalSurfaceGenerator(params)

        # Generate surface
        if not getattr(args, "quiet", False):
            print(f"Generating {args.metal}(111) surface...")

        output_file = generator.generate()

        if not getattr(args, "quiet", False):
            print(f"Successfully generated: {output_file}")

            # Print summary
            print(f"  Metal: {args.metal}")
            print(f"  Size: {args.size[0]}x{args.size[1]} unit cells, {args.size[2]} layers")
            print(f"  Vacuum: {args.vacuum} Å")
            if args.fix_bottom_layers > 0:
                print(f"  Fixed bottom layers: {args.fix_bottom_layers}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Standalone entry point for metal-surface generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-metal-surface",
        description="Generate FCC(111) metal surfaces using ASE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Metals:
  Al, Au, Ag, Cu, Ni, Pd, Pt, Pb, Rh, Ir, Ca, Sr, Yb

Size Parameter:
  Specify surface dimensions as three integers:
    - First two: lateral dimensions (nx, ny) in unit cells
    - Third: number of atomic layers (nz)
  Example: --size 4 4 6 creates a 4x4 surface with 6 layers

Examples:
  1. Basic Pt(111) surface:
     mlip-metal-surface --metal Pt --size 4 4 5 --vacuum 15 --output pt_111.xyz

  2. Gold surface with fixed layers:
     mlip-metal-surface --metal Au --size 5 5 6 --vacuum 12 \\
       --fix-bottom-layers 2 --output au_111.data

  3. Copper surface with custom lattice:
     mlip-metal-surface --metal Cu --size 3 3 4 --vacuum 10 \\
       --lattice-constant 3.62 --output cu_111.vasp
        """,
    )

    # Add verbose/quiet flags for standalone
    parser.add_argument("--verbose", "-V", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (e.g., metal.xyz, metal.data, POSCAR)",
    )

    # Metal selection
    parser.add_argument(
        "--metal",
        "-m",
        type=str,
        default="Pt",
        choices=["Al", "Au", "Ag", "Cu", "Ni", "Pd", "Pt", "Pb", "Rh", "Ir", "Ca", "Sr", "Yb"],
        help="Metal element symbol (default: Pt)",
    )

    # Size parameters
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("NX", "NY", "NZ"),
        help="Surface size as nx ny nz (default: 4 4 4)",
    )

    # Vacuum
    parser.add_argument(
        "--vacuum",
        "-v",
        type=float,
        default=15.0,
        metavar="VACUUM",
        help="Vacuum space above surface in Angstroms (default: 15.0)",
    )

    # Optional parameters
    parser.add_argument(
        "--lattice-constant",
        "-a",
        type=float,
        metavar="A",
        help="Custom lattice constant in Angstroms",
    )

    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        metavar="N",
        help="Number of bottom layers to fix (default: 0)",
    )

    parser.add_argument(
        "--orthogonalize",
        action="store_true",
        default=True,
        help="Create orthogonal unit cell (default: True)",
    )

    parser.add_argument(
        "--no-orthogonalize",
        action="store_false",
        dest="orthogonalize",
        help="Keep non-orthogonal unit cell",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        choices=["xyz", "vasp", "poscar", "lammps", "data"],
        help="Output file format",
    )

    # Options
    parser.add_argument(
        "--log",
        "-l",
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

    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to input_params.json",
    )

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
