"""Analysis commands for trajectory properties."""

import argparse
import sys
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the analyze command parser."""
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze trajectory properties (RDF, MSD, etc.)",
        description="Compute and plot physical properties from LAMMPS trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compute O-O RDF:
    mlip-struct-gen analyze rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl

  Multiple pairs with interactive plot:
    mlip-struct-gen analyze rdf trajectory.lammpstrj --pairs "O-O" "Na-Cl" \\
        --type-map Pt O H Na Cl --plot-backend plotly --output rdf.html

  RDF with coordination number:
    mlip-struct-gen analyze rdf trajectory.lammpstrj --pairs "O-O" \\
        --type-map Pt O H Na Cl --show-coordination --output rdf.png

For more details on a specific analysis:
    mlip-struct-gen analyze rdf --help
        """,
    )

    # Create subcommands for different analysis types
    analyze_subparsers = parser.add_subparsers(
        title="Available analyses",
        dest="analysis_type",
        help="Type of physical property to analyze",
        required=True,
        metavar="ANALYSIS",
    )

    # RDF subcommand
    add_rdf_parser(analyze_subparsers)


def add_rdf_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add RDF analysis subcommand."""
    parser = subparsers.add_parser(
        "rdf",
        help="Compute radial distribution function g(r)",
        description="Compute radial distribution function (RDF) and coordination numbers from LAMMPS trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic RDF for O-O pairs:
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl

  Multiple pairs (O-O, O-H, Na-Cl):
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" "O-H" "Na-Cl" \\
        --type-map Pt O H Na Cl --rmax 10.0 --nbins 200 --output rdf.png

  Interactive plotly plot:
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl \\
        --plot-backend plotly --output rdf.html

  With coordination number and data export:
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl \\
        --show-coordination --output rdf.png --save-data rdf_data.csv

  Use only first 1000 frames:
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl \\
        --n-frames 1000 --output rdf.png

Notes:
  - Pairs format: "Element1-Element2" (e.g., "O-O", "Na-Cl")
  - Type map order must match atom types in trajectory (1->Pt, 2->O, etc.)
  - Default rmax=10.0 Å, nbins=200
  - Supports both matplotlib (static) and plotly (interactive) backends
        """,
    )

    parser.add_argument(
        "trajectory",
        type=str,
        metavar="TRAJECTORY",
        help="LAMMPS trajectory file path (.lammpstrj format)",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        metavar="PAIR",
        help='Element pairs in "Elem1-Elem2" format (e.g., "O-O" "O-H" "Na-Cl"). '
        "Multiple pairs can be specified.",
    )

    parser.add_argument(
        "--type-map",
        nargs="+",
        metavar="ELEM",
        help="Element symbols for atom types in order (type 1, 2, 3, ...). "
        "Example: Pt O H Na Cl maps type 1->Pt, 2->O, 3->H, etc.",
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=10.0,
        metavar="DIST",
        help="Maximum distance for RDF calculation in Angstroms (default: 10.0)",
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=200,
        metavar="N",
        help="Number of histogram bins for distance discretization (default: 200)",
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        metavar="N",
        help="Number of frames to analyze from trajectory. If not specified, uses all frames.",
    )

    parser.add_argument(
        "--plot-backend",
        choices=["matplotlib", "plotly", "both"],
        default="matplotlib",
        metavar="BACKEND",
        help="Plotting backend: 'matplotlib' (static), 'plotly' (interactive), or 'both' (default: matplotlib)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="FILE",
        help="Output file for plot. Extension determines format (.png, .pdf, .svg for matplotlib; .html for plotly)",
    )

    parser.add_argument(
        "--save-data",
        type=str,
        metavar="FILE",
        help="Save computed RDF data to file. Supports CSV (.csv), JSON (.json), or NumPy (.npz) formats",
    )

    parser.add_argument(
        "--show-coordination",
        action="store_true",
        help="Include running coordination number plot alongside g(r)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )


def handle_command(args: argparse.Namespace) -> int:
    """
    Handle the analyze command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args.analysis_type == "rdf":
        return handle_rdf_command(args)

    return 1


def handle_rdf_command(args: argparse.Namespace) -> int:
    """
    Handle RDF analysis command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from ...analysis.properties.rdf import RDF
        from ...utils.logger import get_logger

        logger = get_logger()

        # Validate trajectory file
        trajectory_path = Path(args.trajectory)
        if not trajectory_path.exists():
            logger.error(f"Trajectory file not found: {args.trajectory}")
            return 1

        # Parse type map if provided
        type_map = None
        if args.type_map:
            type_map = {i + 1: elem for i, elem in enumerate(args.type_map)}
            if args.verbose:
                logger.info(f"Type map: {type_map}")

        # Parse pairs
        pairs = []
        for pair_str in args.pairs:
            if "-" not in pair_str:
                logger.error(
                    f'Invalid pair format: {pair_str}. Expected format: "Element1-Element2"'
                )
                return 1
            elem1, elem2 = pair_str.split("-", 1)
            pairs.append((elem1.strip(), elem2.strip()))

        logger.step(f"Computing RDF for trajectory: {trajectory_path.name}")
        logger.info(f"Pairs: {[f'{e1}-{e2}' for e1, e2 in pairs]}")
        logger.info(f"Parameters: rmax={args.rmax} Å, nbins={args.nbins}")

        # Create RDF calculator
        rdf = RDF(trajectory_path, type_map=type_map)

        # Compute RDF
        logger.step("Reading trajectory and computing RDF...")
        results = rdf.compute(pairs=pairs, rmax=args.rmax, nbins=args.nbins, n_frames=args.n_frames)

        logger.success(f"RDF computation complete for {len(results)} pair(s)")

        # Print summary
        for pair_name, data in results.items():
            logger.info(f"\nPair: {pair_name}")
            logger.info(f"  Frames used: {data['n_frames']}")
            logger.info(f"  First peak: r={rdf.get_first_peak(pair_name)[0]:.3f} Å")

        # Save data if requested
        if args.save_data:
            save_path = Path(args.save_data)
            format_ext = save_path.suffix.lstrip(".")
            if format_ext not in ["csv", "json", "npz"]:
                logger.warning(f"Unknown format '{format_ext}', defaulting to CSV")
                format_ext = "csv"

            logger.step(f"Saving RDF data to {save_path}")
            rdf.save(save_path, format=format_ext)
            logger.success("Data saved successfully")

        # Plot results
        if args.plot_backend in ["matplotlib", "both"]:
            logger.step("Generating matplotlib plot...")
            output_mpl = args.output if args.plot_backend == "matplotlib" else None
            if args.plot_backend == "both" and args.output:
                output_mpl = Path(args.output).with_suffix(".png")

            rdf.plot(
                backend="matplotlib",
                output=output_mpl,
                show_coordination=args.show_coordination,
            )

        if args.plot_backend in ["plotly", "both"]:
            logger.step("Generating plotly plot...")
            output_plotly = args.output if args.plot_backend == "plotly" else None
            if args.plot_backend == "both" and args.output:
                output_plotly = Path(args.output).with_suffix(".html")

            rdf.plot(
                backend="plotly",
                output=output_plotly,
                show_coordination=args.show_coordination,
            )

        logger.success("Analysis complete!")
        return 0

    except ImportError as e:
        logger = get_logger() if "logger" in locals() else None
        error_msg = f"Import error: {e}\n\nThe C++ extension may not be compiled. Please build it first with:\n  python -m pip install -e ."
        if logger:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}", file=sys.stderr)
        return 1

    except Exception as e:
        logger = get_logger() if "logger" in locals() else None
        if logger:
            logger.error(f"Error during RDF analysis: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """
    Main entry point for standalone mlip-analyze command.
    """
    parser = argparse.ArgumentParser(
        prog="mlip-analyze",
        description="Analyze physical properties from LAMMPS trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compute radial distribution function:
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl

  Multiple pairs with interactive plot:
    mlip-analyze rdf trajectory.lammpstrj --pairs "O-O" "Na-Cl" \\
        --type-map Pt O H Na Cl --plot-backend plotly --output rdf.html

For detailed help on specific analysis:
    mlip-analyze rdf --help

Available analyses:
  rdf     Radial distribution function and coordination numbers
        """,
    )

    # Add subparsers
    subparsers = parser.add_subparsers(
        title="Available analyses",
        dest="analysis_type",
        required=True,
        metavar="ANALYSIS",
    )

    # Add RDF parser
    add_rdf_parser(subparsers)

    args = parser.parse_args()
    exit_code = handle_command(args)
    sys.exit(exit_code)


def main_rdf() -> None:
    """
    Direct entry point for mlip-analyze-rdf command.
    """
    parser = argparse.ArgumentParser(
        prog="mlip-analyze-rdf",
        description="Compute radial distribution function (RDF) and coordination numbers from LAMMPS trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic RDF for O-O pairs:
    mlip-analyze-rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl

  Multiple pairs (O-O, O-H, Na-Cl):
    mlip-analyze-rdf trajectory.lammpstrj --pairs "O-O" "O-H" "Na-Cl" \\
        --type-map Pt O H Na Cl --rmax 10.0 --nbins 200 --output rdf.png

  Interactive plotly plot:
    mlip-analyze-rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl \\
        --plot-backend plotly --output rdf.html

  With coordination number and data export:
    mlip-analyze-rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl \\
        --show-coordination --output rdf.png --save-data rdf_data.csv

  Use only first 1000 frames:
    mlip-analyze-rdf trajectory.lammpstrj --pairs "O-O" --type-map Pt O H Na Cl \\
        --n-frames 1000 --output rdf.png

Notes:
  - Pairs format: "Element1-Element2" (e.g., "O-O", "Na-Cl")
  - Type map order must match atom types in trajectory (1->Pt, 2->O, etc.)
  - Default rmax=10.0 Å, nbins=200
  - Supports both matplotlib (static) and plotly (interactive) backends
        """,
    )

    parser.add_argument(
        "trajectory",
        type=str,
        metavar="TRAJECTORY",
        help="LAMMPS trajectory file path (.lammpstrj format)",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        metavar="PAIR",
        help='Element pairs in "Elem1-Elem2" format (e.g., "O-O" "O-H" "Na-Cl"). '
        "Multiple pairs can be specified.",
    )

    parser.add_argument(
        "--type-map",
        nargs="+",
        metavar="ELEM",
        help="Element symbols for atom types in order (type 1, 2, 3, ...). "
        "Example: Pt O H Na Cl maps type 1->Pt, 2->O, 3->H, etc.",
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=10.0,
        metavar="DIST",
        help="Maximum distance for RDF calculation in Angstroms (default: 10.0)",
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=200,
        metavar="N",
        help="Number of histogram bins for distance discretization (default: 200)",
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        metavar="N",
        help="Number of frames to analyze from trajectory. If not specified, uses all frames.",
    )

    parser.add_argument(
        "--plot-backend",
        choices=["matplotlib", "plotly", "both"],
        default="matplotlib",
        metavar="BACKEND",
        help="Plotting backend: 'matplotlib' (static), 'plotly' (interactive), or 'both' (default: matplotlib)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="FILE",
        help="Output file for plot. Extension determines format (.png, .pdf, .svg for matplotlib; .html for plotly)",
    )

    parser.add_argument(
        "--save-data",
        type=str,
        metavar="FILE",
        help="Save computed RDF data to file. Supports CSV (.csv), JSON (.json), or NumPy (.npz) formats",
    )

    parser.add_argument(
        "--show-coordination",
        action="store_true",
        help="Include running coordination number plot alongside g(r)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )

    args = parser.parse_args()

    # Set analysis_type for handle_rdf_command
    args.analysis_type = "rdf"

    exit_code = handle_rdf_command(args)
    sys.exit(exit_code)
