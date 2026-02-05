"""CLI interface for spring-restrained metal-water interface generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.spring_metal_water import (
    MetalWaterSpringConstraint,
    SpringConstraint,
    SpringMetalWaterGenerator,
    SpringMetalWaterParameters,
)
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the spring-metal-water subcommand parser."""
    parser = subparsers.add_parser(
        "spring-metal-water",
        help="Generate metal-water interface with spring bond restraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate spring-restrained metal-water interface structures for MLIP training",
        epilog="""
Examples:
  1. Add spring restraint to 5 Pt-O distances:
     mlip-struct-gen generate spring-metal-water \\
         --metal Pt --size 4 4 4 --n-water 50 \\
         --spring-metal-water O 5 2.5 \\
         --type-map Pt O H --model graph.000.pb \\
         --output pt_water_spring.data

  2. Add spring to Metal-O and O-H:
     mlip-struct-gen generate spring-metal-water \\
         --metal Cu --size 4 4 4 --n-water 50 \\
         --spring-metal-water O 3 2.3 \\
         --spring O H 2 1.02 \\
         --k-spring 100 \\
         --type-map Cu O H --model graph.000.pb \\
         --output cu_water_spring.data
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output LAMMPS data file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["graph.000.pb", "graph.001.pb", "graph.002.pb", "graph.003.pb", "graph.004.pb"],
        help="DeepMD model files",
    )

    # Metal surface parameters
    parser.add_argument("--metal", type=str, required=True, choices=["Cu", "Pt"])
    parser.add_argument("--size", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--lattice-constant", type=float, default=None)
    parser.add_argument("--fix-bottom-layers", type=int, default=0)

    # Water parameters
    parser.add_argument("--n-water", "-n", type=int, required=True, metavar="N")
    parser.add_argument("--density", "-d", type=float, default=1.0)
    parser.add_argument("--gap-above-metal", type=float, default=3.0)
    parser.add_argument("--vacuum-above-water", type=float, default=0.0)

    # Metal-water spring constraint arguments
    parser.add_argument(
        "--spring-metal-water",
        nargs=3,
        action="append",
        metavar=("WATER_ELEM", "COUNT", "DISTANCE"),
        help="Metal-water spring: O 5 2.5 (5 Metal-O pairs at 2.5 A)",
    )

    # Water-only spring constraint arguments
    parser.add_argument(
        "--spring",
        nargs=4,
        action="append",
        metavar=("ELEM1", "ELEM2", "COUNT", "DISTANCE"),
        help="Water spring: O H 1 1.02, O O 1 2.8",
    )

    parser.add_argument(
        "--k-spring", type=float, default=50.0, help="Spring constant (default: 50.0)"
    )
    parser.add_argument("--minimize", action="store_true")

    # MD parameters
    parser.add_argument("--ensemble", type=str, choices=["nvt", "npt"], default="npt")
    parser.add_argument("--nsteps", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=300.0)
    parser.add_argument("--pres", type=float, default=1.0)
    parser.add_argument("--timestep", type=float, default=0.0005)
    parser.add_argument("--dump-freq", type=int, default=10)
    parser.add_argument("--thermo-freq", type=int, default=10)

    # Other
    parser.add_argument("--water-model", "-m", type=str, default="SPC/E")
    parser.add_argument("--tolerance", "-t", type=float, default=2.0)
    parser.add_argument("--seed", "-s", type=int, default=12345)
    parser.add_argument("--constraint-seed", type=int, default=42)

    # Type map
    parser.add_argument("--type-map", type=str, nargs="+", metavar="ELEM")

    # Options
    parser.add_argument("--log", "-l", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-input", action="store_true")


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' exists. Use --force to overwrite")
        sys.exit(1)

    has_constraints = args.spring_metal_water or args.spring
    if not has_constraints:
        logger.error(
            "Must specify at least one spring constraint (--spring-metal-water or --spring)"
        )
        sys.exit(1)


def get_default_elements(metal: str) -> list[str]:
    """Get default element ordering for a metal type."""
    return [metal, "O", "H"]


def parse_constraints(args: argparse.Namespace) -> tuple[list, list]:
    """Parse spring constraint arguments."""
    metal_water_spring_constraints = []
    spring_constraints = []

    if args.spring_metal_water:
        for water_elem, count_str, dist_str in args.spring_metal_water:
            count = count_str if count_str == "all" else int(count_str)
            distance = float(dist_str)
            metal_water_spring_constraints.append(
                MetalWaterSpringConstraint(water_elem, count, distance, k_spring=args.k_spring)
            )

    if args.spring:
        for elem1, elem2, count_str, dist_str in args.spring:
            count = count_str if count_str == "all" else int(count_str)
            distance = float(dist_str)
            spring_constraints.append(
                SpringConstraint(elem1, elem2, count, distance, k_spring=args.k_spring)
            )

    return metal_water_spring_constraints, spring_constraints


def handle_command(args: argparse.Namespace) -> int:
    """Handle the spring-metal-water generation command."""
    validate_args(args)

    if args.dry_run:
        logger.info("Dry run - would generate spring-restrained metal-water interface with:")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Metal: {args.metal}")
        logger.info(f"  Size: {args.size}")
        logger.info(f"  N water: {args.n_water}")
        if args.spring_metal_water:
            for c in args.spring_metal_water:
                logger.info(
                    f"  Metal-water spring: {args.metal}-{c[0]} x{c[1]} -> {c[2]} A (K={args.k_spring})"
                )
        if args.spring:
            for c in args.spring:
                logger.info(
                    f"  Water spring: {c[0]}-{c[1]} x{c[2]} -> {c[3]} A (K={args.k_spring})"
                )
        return 0

    try:
        metal_water_spring_constraints, spring_constraints = parse_constraints(args)
        elements = args.type_map if args.type_map else get_default_elements(args.metal)

        params = SpringMetalWaterParameters(
            output_file=args.output,
            model_files=args.model,
            metal=args.metal,
            size=tuple(args.size),
            lattice_constant=args.lattice_constant,
            fix_bottom_layers=args.fix_bottom_layers,
            n_water=args.n_water,
            density=args.density,
            gap_above_metal=args.gap_above_metal,
            vacuum_above_water=args.vacuum_above_water,
            water_model=args.water_model,
            metal_water_spring_constraints=metal_water_spring_constraints,
            spring_constraints=spring_constraints,
            constraint_seed=args.constraint_seed,
            minimize=args.minimize,
            ensemble=args.ensemble,
            nsteps=args.nsteps,
            temp=args.temp,
            pres=args.pres,
            timestep=args.timestep,
            dump_freq=args.dump_freq,
            thermo_freq=args.thermo_freq,
            packmol_tolerance=args.tolerance,
            seed=args.seed,
            elements=elements,
            log=args.log,
            logger=logger if args.log else None,
        )

        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        generator = SpringMetalWaterGenerator(params)

        if not getattr(args, "quiet", False):
            logger.info("Generating spring-restrained metal-water interface...")

        output_file = generator.run()

        if not getattr(args, "quiet", False):
            logger.info(f"Successfully generated: {output_file}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1
