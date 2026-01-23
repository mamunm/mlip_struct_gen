"""CLI interface for constrained metal-salt-water interface generation."""

import argparse
import sys
from pathlib import Path

from ...generate_structure.constrained_metal_salt_water import (
    AngleConstraint,
    ConstrainedMetalSaltWaterGenerator,
    ConstrainedMetalSaltWaterParameters,
    DistanceConstraint,
    MetalIonDistanceConstraint,
    MetalWaterAngleConstraint,
    MetalWaterDistanceConstraint,
)
from ...utils.json_utils import save_parameters_to_json
from ...utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the constrained-metal-salt-water subcommand parser."""
    parser = subparsers.add_parser(
        "constrained-metal-salt-water",
        help="Generate metal-salt-water interface with constrained bonds/angles/distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate constrained metal-salt-water interface structures for MLIP training",
        epilog="""
Examples:
  1. Constrain Metal-O and Metal-Na distances:
     mlip-struct-gen generate constrained-metal-salt-water \\
         --metal Pt --size 4 4 4 --n-water 50 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-metal-water-distance O 5 2.5 \\
         --constrain-metal-ion-distance Na 3 2.8 \\
         --type-map Pt O H Na Cl --model graph.000.pb \\
         --output pt_nacl_water_constrained.data

  2. Constrain ion-water distances (Na-O solvation):
     mlip-struct-gen generate constrained-metal-salt-water \\
         --metal Cu --size 4 4 4 --n-water 50 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance Na O 3 2.3 \\
         --type-map Cu O H Na Cl --model graph.000.pb \\
         --output cu_nacl_water_constrained.data

  3. Constrain ion-ion distances (Na-Cl):
     mlip-struct-gen generate constrained-metal-salt-water \\
         --metal Pt --size 4 4 4 --n-water 50 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-distance Na Cl 2 2.5 \\
         --type-map Pt O H Na Cl --model graph.000.pb \\
         --output pt_nacl_constrained.data

  4. Multiple constraints:
     mlip-struct-gen generate constrained-metal-salt-water \\
         --metal Pt --size 4 4 4 --n-water 50 \\
         --salt-type NaCl --n-salt 5 \\
         --constrain-metal-water-distance O 5 2.5 \\
         --constrain-metal-ion-distance Na 3 2.8 \\
         --constrain-metal-ion-distance Cl 2 3.0 \\
         --constrain-distance Na O 2 2.3 \\
         --constrain-distance O H 2 0.85 \\
         --type-map Pt O H Na Cl --model graph.000.pb \\
         --output constrained.data
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

    # Metal surface parameters
    parser.add_argument(
        "--metal",
        type=str,
        required=True,
        choices=["Cu", "Pt"],
        help="Metal element (Cu or Pt)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Surface size as (nx, ny, nz) unit cells",
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=None,
        help="Custom lattice constant in Angstroms (default: Cu=3.536, Pt=3.901)",
    )
    parser.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=0,
        help="Number of bottom metal layers to fix (default: 0)",
    )

    # Salt parameters
    parser.add_argument(
        "--salt-type",
        type=str,
        default="NaCl",
        choices=["NaCl", "KCl", "LiCl", "CsCl"],
        help="Type of salt (default: NaCl)",
    )
    parser.add_argument(
        "--n-salt",
        type=int,
        default=5,
        help="Number of salt formula units (default: 5)",
    )
    parser.add_argument(
        "--include-salt-volume",
        action="store_true",
        help="Account for ion volume when computing box size",
    )
    parser.add_argument(
        "--no-salt-zone",
        type=float,
        default=0.2,
        help="Fraction of box height where ions are excluded (default: 0.2)",
    )

    # Water parameters
    parser.add_argument(
        "--n-water",
        "-n",
        type=int,
        required=True,
        metavar="N",
        help="Number of water molecules",
    )
    parser.add_argument(
        "--density",
        "-d",
        type=float,
        default=1.0,
        metavar="RHO",
        help="Solution density in g/cm3 (default: 1.0)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Gap between metal surface and solution in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--vacuum-above-water",
        type=float,
        default=0.0,
        help="Vacuum space above water in Angstroms (default: 0.0)",
    )

    # Metal-water constraint arguments
    parser.add_argument(
        "--constrain-metal-water-distance",
        nargs=3,
        action="append",
        metavar=("WATER_ELEM", "COUNT", "DISTANCE"),
        help="Metal-water distance constraint: O 5 2.5 (5 Metal-O pairs at 2.5 A)",
    )
    parser.add_argument(
        "--constrain-metal-water-angle",
        nargs=2,
        action="append",
        metavar=("COUNT", "ANGLE"),
        help="Metal-O-H angle constraint: 3 120 (3 Metal-O-H angles at 120 degrees)",
    )

    # Metal-ion constraint arguments
    parser.add_argument(
        "--constrain-metal-ion-distance",
        nargs=3,
        action="append",
        metavar=("ION_ELEM", "COUNT", "DISTANCE"),
        help="Metal-ion distance constraint: Na 3 2.8 (3 Metal-Na pairs at 2.8 A)",
    )

    # General distance/angle constraint arguments
    parser.add_argument(
        "--constrain-distance",
        nargs=4,
        action="append",
        metavar=("ELEM1", "ELEM2", "COUNT", "DISTANCE"),
        help="Distance constraint: Na O 2 2.3 (ion-water), Na Cl 2 2.5 (ion-ion), O H 2 0.85 (water)",
    )
    parser.add_argument(
        "--constrain-angle",
        nargs=2,
        action="append",
        metavar=("COUNT", "ANGLE"),
        help="H-O-H angle constraint: 1 100 (1 angle to 100 degrees)",
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
        help="Element type mapping for LAMMPS (e.g., --type-map Pt O H Na Cl)",
    )

    # Options
    parser.add_argument("--log", "-l", action="store_true", help="Enable detailed logging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    parser.add_argument("--save-input", action="store_true", help="Save parameters to JSON")


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        logger.error(f"Output file '{args.output}' exists. Use --force to overwrite")
        sys.exit(1)

    # Check at least one constraint is specified
    has_constraints = (
        args.constrain_metal_water_distance
        or args.constrain_metal_water_angle
        or args.constrain_metal_ion_distance
        or args.constrain_distance
        or args.constrain_angle
    )
    if not has_constraints:
        logger.error(
            "Must specify at least one constraint (--constrain-metal-water-distance, "
            "--constrain-metal-water-angle, --constrain-metal-ion-distance, "
            "--constrain-distance, or --constrain-angle)"
        )
        sys.exit(1)


def parse_constraints(
    args: argparse.Namespace,
) -> tuple[list, list, list, list, list]:
    """Parse constraint arguments into constraint objects."""
    metal_water_distance_constraints = []
    metal_water_angle_constraints = []
    metal_ion_distance_constraints = []
    distance_constraints = []
    angle_constraints = []

    if args.constrain_metal_water_distance:
        for water_elem, count_str, dist_str in args.constrain_metal_water_distance:
            count = count_str if count_str == "all" else int(count_str)
            distance = float(dist_str)
            metal_water_distance_constraints.append(
                MetalWaterDistanceConstraint(water_elem, count, distance)
            )

    if args.constrain_metal_water_angle:
        for count_str, angle_str in args.constrain_metal_water_angle:
            count = count_str if count_str == "all" else int(count_str)
            angle = float(angle_str)
            metal_water_angle_constraints.append(MetalWaterAngleConstraint(count, angle))

    if args.constrain_metal_ion_distance:
        for ion_elem, count_str, dist_str in args.constrain_metal_ion_distance:
            count = count_str if count_str == "all" else int(count_str)
            distance = float(dist_str)
            metal_ion_distance_constraints.append(
                MetalIonDistanceConstraint(ion_elem, count, distance)
            )

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

    return (
        metal_water_distance_constraints,
        metal_water_angle_constraints,
        metal_ion_distance_constraints,
        distance_constraints,
        angle_constraints,
    )


def get_default_elements(metal: str, salt_type: str) -> list[str]:
    """Get default element ordering for a metal and salt type."""
    salt_ions = {
        "NaCl": ["Na", "Cl"],
        "KCl": ["K", "Cl"],
        "LiCl": ["Li", "Cl"],
        "CsCl": ["Cs", "Cl"],
    }
    ions = salt_ions.get(salt_type, ["Na", "Cl"])
    return [metal, "O", "H"] + ions


def handle_command(args: argparse.Namespace) -> int:
    """Handle the constrained-metal-salt-water generation command."""
    validate_args(args)

    if args.dry_run:
        logger.info("Dry run - would generate constrained metal-salt-water interface with:")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Models: {args.model}")
        logger.info(f"  Metal: {args.metal}")
        logger.info(f"  Size: {args.size}")
        logger.info(f"  N water: {args.n_water}")
        logger.info(f"  Salt type: {args.salt_type}")
        logger.info(f"  N salt: {args.n_salt}")
        if args.constrain_metal_water_distance:
            for c in args.constrain_metal_water_distance:
                logger.info(f"  Metal-water distance: {args.metal}-{c[0]} x{c[1]} -> {c[2]} A")
        if args.constrain_metal_water_angle:
            for c in args.constrain_metal_water_angle:
                logger.info(f"  Metal-water angle: {args.metal}-O-H x{c[0]} -> {c[1]} deg")
        if args.constrain_metal_ion_distance:
            for c in args.constrain_metal_ion_distance:
                logger.info(f"  Metal-ion distance: {args.metal}-{c[0]} x{c[1]} -> {c[2]} A")
        if args.constrain_distance:
            for c in args.constrain_distance:
                logger.info(f"  Distance: {c[0]}-{c[1]} x{c[2]} -> {c[3]} A")
        if args.constrain_angle:
            for c in args.constrain_angle:
                logger.info(f"  Angle: H-O-H x{c[0]} -> {c[1]} deg")
        logger.info(f"  Ensemble: {args.ensemble}")
        if args.minimize:
            logger.info("  Minimization: enabled")
        if args.type_map:
            logger.info(f"  Type map: {args.type_map}")
        return 0

    try:
        (
            metal_water_distance_constraints,
            metal_water_angle_constraints,
            metal_ion_distance_constraints,
            distance_constraints,
            angle_constraints,
        ) = parse_constraints(args)

        # Use type-map if provided, otherwise get default
        elements = (
            args.type_map if args.type_map else get_default_elements(args.metal, args.salt_type)
        )

        params = ConstrainedMetalSaltWaterParameters(
            output_file=args.output,
            model_files=args.model,
            metal=args.metal,
            size=tuple(args.size),
            lattice_constant=args.lattice_constant,
            fix_bottom_layers=args.fix_bottom_layers,
            salt_type=args.salt_type,
            n_salt=args.n_salt,
            include_salt_volume=args.include_salt_volume,
            no_salt_zone=args.no_salt_zone,
            n_water=args.n_water,
            density=args.density,
            gap=args.gap,
            vacuum_above_water=args.vacuum_above_water,
            water_model=args.water_model,
            metal_water_distance_constraints=metal_water_distance_constraints,
            metal_water_angle_constraints=metal_water_angle_constraints,
            metal_ion_distance_constraints=metal_ion_distance_constraints,
            distance_constraints=distance_constraints,
            angle_constraints=angle_constraints,
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

        generator = ConstrainedMetalSaltWaterGenerator(params)

        if not getattr(args, "quiet", False):
            logger.info("Generating constrained metal-salt-water interface...")

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
    """Standalone entry point for constrained-metal-salt-water generation."""
    parser = argparse.ArgumentParser(
        prog="mlip-constrained-metal-salt-water",
        description="Generate constrained metal-salt-water interface structures for MLIP training",
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

    # Metal surface parameters
    parser.add_argument("--metal", type=str, required=True, choices=["Cu", "Pt"])
    parser.add_argument("--size", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--lattice-constant", type=float, default=None)
    parser.add_argument("--fix-bottom-layers", type=int, default=0)

    # Salt parameters
    parser.add_argument(
        "--salt-type", type=str, default="NaCl", choices=["NaCl", "KCl", "LiCl", "CsCl"]
    )
    parser.add_argument("--n-salt", type=int, default=5)
    parser.add_argument("--include-salt-volume", action="store_true")
    parser.add_argument("--no-salt-zone", type=float, default=0.2)

    # Water parameters
    parser.add_argument("--n-water", "-n", type=int, required=True, metavar="N")
    parser.add_argument("--density", "-d", type=float, default=1.0, metavar="RHO")
    parser.add_argument("--gap", type=float, default=0.0)
    parser.add_argument("--vacuum-above-water", type=float, default=0.0)

    # Constraints
    parser.add_argument(
        "--constrain-metal-water-distance",
        nargs=3,
        action="append",
        metavar=("WATER_ELEM", "COUNT", "DISTANCE"),
    )
    parser.add_argument(
        "--constrain-metal-water-angle",
        nargs=2,
        action="append",
        metavar=("COUNT", "ANGLE"),
    )
    parser.add_argument(
        "--constrain-metal-ion-distance",
        nargs=3,
        action="append",
        metavar=("ION_ELEM", "COUNT", "DISTANCE"),
    )
    parser.add_argument(
        "--constrain-distance",
        nargs=4,
        action="append",
        metavar=("ELEM1", "ELEM2", "COUNT", "DISTANCE"),
    )
    parser.add_argument("--constrain-angle", nargs=2, action="append", metavar=("COUNT", "ANGLE"))

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
