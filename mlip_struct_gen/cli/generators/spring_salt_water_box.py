"""CLI interface for spring-restrained salt water box generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.spring_salt_water_box import (
    SpringConstraint,
    SpringSaltWaterBoxGenerator,
    SpringSaltWaterBoxParameters,
)
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the spring-salt-water subcommand parser."""
    parser = subparsers.add_parser(
        "spring-salt-water",
        help="Generate salt water box with spring bond restraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate spring-restrained salt water box structures for MLIP training",
        epilog="""
Examples:
  1. Add spring restraint to 1 O-H bond in salt water:
     mlip-struct-gen generate spring-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --spring O H 1 1.02 \\
         --model graph.000.pb --output spring_salt.data

  2. Add spring restraint to Na-O distance:
     mlip-struct-gen generate spring-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --spring Na O 1 2.3 --k-spring 100 \\
         --model graph.000.pb --output spring_salt.data

  3. Multiple spring restraints:
     mlip-struct-gen generate spring-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --spring O H 2 1.02 \\
         --spring Na Cl 1 2.8 \\
         --model graph.000.pb --output spring_salt.data
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

    # Water box parameters
    parser.add_argument("--box-size", "-b", type=float, nargs="+", metavar="SIZE")
    parser.add_argument("--n-water", "-n", type=int, metavar="N")
    parser.add_argument("--density", "-d", type=float, metavar="RHO")

    # Salt parameters
    parser.add_argument(
        "--salt-type",
        type=str,
        default="NaCl",
        choices=["NaCl", "KCl", "LiCl", "CaCl2", "MgCl2", "NaBr", "KBr", "CsCl"],
    )
    parser.add_argument("--n-salt", type=int, default=0)
    parser.add_argument("--include-salt-volume", action="store_true")

    # Spring constraint arguments
    parser.add_argument(
        "--spring",
        nargs=4,
        action="append",
        metavar=("ELEM1", "ELEM2", "COUNT", "DISTANCE"),
        help="Spring restraint: O H 1 1.02, Na O 1 2.3, Na Cl 1 2.8",
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
    if args.box_size is not None:
        if len(args.box_size) not in [1, 3]:
            logger.error("--box-size must have 1 value (cubic) or 3 values (rectangular)")
            sys.exit(1)
        if len(args.box_size) == 3:
            args.box_size = tuple(args.box_size)
        else:
            args.box_size = args.box_size[0]

    params_count = sum(
        [
            args.box_size is not None,
            args.n_water is not None,
            args.density is not None,
        ]
    )

    if params_count < 2:
        logger.error("Must specify exactly 2 of: --box-size, --n-water, --density")
        sys.exit(1)
    elif params_count > 2:
        logger.error("Cannot specify all three: --box-size, --n-water, --density")
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' exists. Use --force to overwrite")
        sys.exit(1)

    if not args.spring:
        logger.error("Must specify at least one spring constraint (--spring)")
        sys.exit(1)


def get_default_elements(salt_type: str) -> list[str]:
    """Get default element ordering for a salt type."""
    salt_elements = {
        "NaCl": ["O", "H", "Na", "Cl"],
        "KCl": ["O", "H", "K", "Cl"],
        "LiCl": ["O", "H", "Li", "Cl"],
        "CaCl2": ["O", "H", "Ca", "Cl"],
        "MgCl2": ["O", "H", "Mg", "Cl"],
        "NaBr": ["O", "H", "Na", "Br"],
        "KBr": ["O", "H", "K", "Br"],
        "CsCl": ["O", "H", "Cs", "Cl"],
    }
    return salt_elements.get(salt_type, ["O", "H", "Na", "Cl"])


def parse_spring_constraints(args: argparse.Namespace) -> list:
    """Parse spring constraint arguments."""
    constraints = []
    if args.spring:
        for elem1, elem2, count_str, dist_str in args.spring:
            count = count_str if count_str == "all" else int(count_str)
            distance = float(dist_str)
            constraints.append(
                SpringConstraint(elem1, elem2, count, distance, k_spring=args.k_spring)
            )
    return constraints


def handle_command(args: argparse.Namespace) -> int:
    """Handle the spring-salt-water generation command."""
    validate_args(args)

    if args.dry_run:
        logger.info("Dry run - would generate spring-restrained salt water box with:")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Models: {args.model}")
        if args.box_size:
            logger.info(f"  Box size: {args.box_size}")
        if args.n_water:
            logger.info(f"  N water: {args.n_water}")
        if args.density:
            logger.info(f"  Density: {args.density} g/cm3")
        logger.info(f"  Salt type: {args.salt_type}")
        logger.info(f"  N salt: {args.n_salt}")
        if args.spring:
            for c in args.spring:
                logger.info(
                    f"  Spring restraint: {c[0]}-{c[1]} x{c[2]} -> {c[3]} A (K={args.k_spring})"
                )
        return 0

    try:
        spring_constraints = parse_spring_constraints(args)
        elements = args.type_map if args.type_map else get_default_elements(args.salt_type)

        params = SpringSaltWaterBoxParameters(
            output_file=args.output,
            model_files=args.model,
            box_size=args.box_size,
            n_water=args.n_water,
            density=args.density,
            salt_type=args.salt_type,
            n_salt=args.n_salt,
            include_salt_volume=args.include_salt_volume,
            water_model=args.water_model,
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
            tolerance=args.tolerance,
            seed=args.seed,
            elements=elements,
            log=args.log,
            logger=logger if args.log else None,
        )

        if getattr(args, "save_input", False):
            save_parameters_to_json(params)

        generator = SpringSaltWaterBoxGenerator(params)

        if not getattr(args, "quiet", False):
            logger.info("Generating spring-restrained salt water box...")

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
