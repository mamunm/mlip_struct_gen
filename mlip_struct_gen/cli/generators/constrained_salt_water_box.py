"""CLI interface for constrained salt water box generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.constrained_salt_water_box import (
    AngleConstraint,
    ConstrainedSaltWaterBoxGenerator,
    ConstrainedSaltWaterBoxParameters,
    DistanceConstraint,
)
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the constrained-salt-water subcommand parser."""
    parser = subparsers.add_parser(
        "constrained-salt-water",
        help="Generate salt water box with constrained bonds/angles/distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate constrained salt water box structures for MLIP training",
        epilog="""
Examples:
  1. Constrain 1 O-H bond to 0.7 A in salt water:
     mlip-struct-gen generate constrained-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance O H 1 0.7 \\
         --model graph.000.pb --output constrained_salt.data

  2. Constrain Na-O distance (move ion closer to water):
     mlip-struct-gen generate constrained-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance Na O 1 2.0 \\
         --model graph.000.pb --output constrained_salt.data

  3. Multiple constraints with harmonic potential:
     mlip-struct-gen generate constrained-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance O H 2 0.7 \\
         --constrain-distance Na O 1 2.0 \\
         --constrain-angle 1 100 \\
         --constraint-type harmonic --harmonic-k 50 \\
         --model graph.000.pb --output constrained_salt.data

  4. Constrain Na-Cl distance:
     mlip-struct-gen generate constrained-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance Na Cl 1 2.5 \\
         --model graph.000.pb --output constrained_salt.data

  5. With custom type-map for LAMMPS:
     mlip-struct-gen generate constrained-salt-water \\
         --n-water 32 --density 1.0 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance O H 1 0.7 \\
         --type-map Cu O H Na Cl \\
         --model graph.000.pb --output constrained_salt.data
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output LAMMPS data file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["graph.000.pb", "graph.001.pb", "graph.002.pb", "graph.003.pb", "graph.004.pb"],
        help="DeepMD model files (default: graph.000.pb to graph.004.pb)",
    )

    # Water box parameters (need 2 of 3)
    parser.add_argument(
        "--box-size",
        "-b",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Box dimensions in Angstroms (1 value for cubic, 3 for rectangular)",
    )
    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        metavar="N",
        help="Number of water molecules",
    )
    parser.add_argument(
        "--density",
        "-d",
        type=float,
        metavar="RHO",
        help="Water density in g/cm3",
    )

    # Salt parameters
    parser.add_argument(
        "--salt-type",
        type=str,
        default="NaCl",
        choices=["NaCl", "KCl", "LiCl", "CaCl2", "MgCl2", "NaBr", "KBr", "CsCl"],
        help="Type of salt (default: NaCl)",
    )
    parser.add_argument(
        "--n-salt",
        type=int,
        default=0,
        help="Number of salt formula units",
    )
    parser.add_argument(
        "--include-salt-volume",
        action="store_true",
        help="Account for ion volume when computing box size",
    )

    # Constraint arguments
    parser.add_argument(
        "--constrain-distance",
        nargs=4,
        action="append",
        metavar=("ELEM1", "ELEM2", "COUNT", "DISTANCE"),
        help="Distance constraint: O H 1 0.7 (O-H bond), Na O 1 2.0 (ion-water), Na Cl 1 2.5 (ion-ion)",
    )
    parser.add_argument(
        "--constrain-angle",
        nargs=2,
        action="append",
        metavar=("COUNT", "ANGLE"),
        help="H-O-H angle constraint: 1 100 (1 angle to 100 degrees)",
    )

    # Constraint type
    parser.add_argument(
        "--constraint-type",
        type=str,
        choices=["rigid", "harmonic"],
        default="rigid",
        help="Constraint type: rigid (K=10000) or harmonic (default: rigid)",
    )
    parser.add_argument(
        "--harmonic-k",
        type=float,
        default=50.0,
        help="Spring constant for harmonic constraints (default: 50.0)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Add energy minimization before MD",
    )

    # MD parameters
    parser.add_argument(
        "--ensemble",
        type=str,
        choices=["nvt", "npt"],
        default="npt",
        help="MD ensemble: nvt or npt (default: npt)",
    )
    parser.add_argument("--nsteps", type=int, default=1000, help="Number of MD steps")
    parser.add_argument("--temp", type=float, default=300.0, help="Temperature in K")
    parser.add_argument("--pres", type=float, default=1.0, help="Pressure in bar")
    parser.add_argument("--timestep", type=float, default=0.0005, help="Timestep in ps")
    parser.add_argument("--dump-freq", type=int, default=10, help="Trajectory dump frequency")
    parser.add_argument("--thermo-freq", type=int, default=10, help="Thermo output frequency")

    # Other parameters
    parser.add_argument(
        "--water-model",
        "-m",
        type=str,
        choices=["SPC/E", "TIP3P", "TIP4P", "SPC/Fw"],
        default="SPC/E",
        help="Water model (default: SPC/E)",
    )
    parser.add_argument("--tolerance", "-t", type=float, default=2.0, help="Packmol tolerance")
    parser.add_argument("--seed", "-s", type=int, default=12345, help="Random seed for Packmol")
    parser.add_argument(
        "--constraint-seed", type=int, default=42, help="Seed for constraint selection"
    )

    # Type map for LAMMPS
    parser.add_argument(
        "--type-map",
        type=str,
        nargs="+",
        metavar="ELEM",
        help="Element type mapping for LAMMPS (e.g., --type-map O H Na Cl assigns O=1, H=2, Na=3, Cl=4). "
        "Elements not in structure still get types defined.",
    )

    # Options
    parser.add_argument("--log", "-l", action="store_true", help="Enable detailed logging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    parser.add_argument("--save-input", action="store_true", help="Save parameters to JSON")


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check box-size format
    if args.box_size is not None:
        if len(args.box_size) not in [1, 3]:
            logger.error("--box-size must have 1 value (cubic) or 3 values (rectangular)")
            sys.exit(1)
        if len(args.box_size) == 3:
            args.box_size = tuple(args.box_size)
        else:
            args.box_size = args.box_size[0]

    # Count how many of the 3 main parameters are provided
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

    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' exists. Use --force to overwrite")
        sys.exit(1)

    # Check at least one constraint is specified
    if not args.constrain_distance and not args.constrain_angle:
        logger.error(
            "Must specify at least one constraint (--constrain-distance or --constrain-angle)"
        )
        sys.exit(1)


def parse_constraints(args: argparse.Namespace) -> tuple[list, list]:
    """Parse constraint arguments into constraint objects."""
    distance_constraints = []
    angle_constraints = []

    if args.constrain_distance:
        for elem1, elem2, count_str, dist_str in args.constrain_distance:
            count = count_str if count_str == "all" else int(count_str)
            distance = float(dist_str)
            distance_constraints.append(DistanceConstraint(elem1, elem2, count, distance))

    if args.constrain_angle:
        for count_str, angle_str in args.constrain_angle:
            count = count_str if count_str == "all" else int(count_str)
            angle = float(angle_str)
            angle_constraints.append(AngleConstraint(count, angle))

    return distance_constraints, angle_constraints


def get_default_elements(salt_type: str) -> list[str]:
    """Get default element ordering for a salt type."""
    # Map salt types to their cation/anion
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


def handle_command(args: argparse.Namespace) -> int:
    """Handle the constrained-salt-water generation command."""
    validate_args(args)

    if args.dry_run:
        logger.info("Dry run - would generate constrained salt water box with:")
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
        if args.constrain_distance:
            for c in args.constrain_distance:
                logger.info(f"  Distance constraint: {c[0]}-{c[1]} x{c[2]} -> {c[3]} A")
        if args.constrain_angle:
            for c in args.constrain_angle:
                logger.info(f"  Angle constraint: H-O-H x{c[0]} -> {c[1]} deg")
        logger.info(f"  Constraint type: {args.constraint_type}")
        logger.info(f"  Ensemble: {args.ensemble}")
        if args.minimize:
            logger.info("  Minimization: enabled")
        if args.type_map:
            logger.info(f"  Type map: {args.type_map}")
        return 0

    try:
        distance_constraints, angle_constraints = parse_constraints(args)

        # Use type-map if provided, otherwise get default based on salt type
        elements = args.type_map if args.type_map else get_default_elements(args.salt_type)

        params = ConstrainedSaltWaterBoxParameters(
            output_file=args.output,
            model_files=args.model,
            box_size=args.box_size,
            n_water=args.n_water,
            density=args.density,
            salt_type=args.salt_type,
            n_salt=args.n_salt,
            include_salt_volume=args.include_salt_volume,
            water_model=args.water_model,
            distance_constraints=distance_constraints,
            angle_constraints=angle_constraints,
            constraint_seed=args.constraint_seed,
            constraint_type=args.constraint_type,
            harmonic_k=args.harmonic_k,
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

        generator = ConstrainedSaltWaterBoxGenerator(params)

        if not getattr(args, "quiet", False):
            logger.info("Generating constrained salt water box...")

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


def main() -> int:
    """Standalone entry point for constrained-salt-water generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-constrained-salt-water",
        description="Generate constrained salt water box structures for MLIP training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file path")
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
    parser.add_argument("--salt-type", type=str, default="NaCl")
    parser.add_argument("--n-salt", type=int, default=0)
    parser.add_argument("--include-salt-volume", action="store_true")

    # Constraints
    parser.add_argument(
        "--constrain-distance",
        nargs=4,
        action="append",
        metavar=("ELEM1", "ELEM2", "COUNT", "DISTANCE"),
    )
    parser.add_argument("--constrain-angle", nargs=2, action="append", metavar=("COUNT", "ANGLE"))

    # Constraint type
    parser.add_argument(
        "--constraint-type", type=str, choices=["rigid", "harmonic"], default="rigid"
    )
    parser.add_argument("--harmonic-k", type=float, default=50.0)
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
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
