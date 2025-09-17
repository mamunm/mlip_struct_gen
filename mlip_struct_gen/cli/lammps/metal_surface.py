"""CLI for generating LAMMPS input for metal surface systems."""

import argparse

from mlip_struct_gen.generate_lammps_input.metal_surface import (
    MetalSurfaceLAMMPSGenerator,
    MetalSurfaceLAMMPSParameters,
)
from mlip_struct_gen.utils.logger import MLIPLogger

logger = MLIPLogger()


def setup_parser(subparsers):
    """Set up the parser for metal surface LAMMPS input generation.

    Args:
        subparsers: The subparsers object from argparse
    """
    parser = subparsers.add_parser(
        "mlip-metal-surface",
        help="Generate LAMMPS input for metal surface simulations",
        description="Generate LAMMPS input files for metal surface simulations with LJ potentials",
    )

    # Required arguments
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the LAMMPS data file containing metal surface structure",
    )

    parser.add_argument(
        "--metal",
        "-m",
        type=str,
        required=True,
        choices=["Cu", "Ag", "Au", "Ni", "Pd", "Pt", "Al"],
        help="Metal type for the surface",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="input.lammps",
        help="Output LAMMPS input filename (default: input.lammps)",
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NPT", "NVT", "NVE"],
        help="Simulation ensemble (default: NVT)",
    )

    parser.add_argument(
        "--temperature",
        "-T",
        nargs="+",
        type=float,
        default=[330.0],
        help="Temperature(s) in K for sampling (default: 330.0)",
    )

    parser.add_argument(
        "--pressure",
        "-P",
        type=float,
        default=1.0,
        help="Pressure in bar for NPT ensemble (default: 1.0)",
    )

    parser.add_argument(
        "--equilibration-time",
        type=float,
        default=100.0,
        help="Equilibration time in ps (default: 100.0)",
    )

    parser.add_argument(
        "--production-time",
        type=float,
        default=500.0,
        help="Production time in ps (default: 500.0)",
    )

    parser.add_argument(
        "--dump-frequency",
        type=float,
        default=1.0,
        help="Trajectory dump frequency in ps (default: 1.0)",
    )

    parser.add_argument(
        "--fix-layers",
        type=int,
        default=2,
        help="Number of bottom layers to fix (default: 2)",
    )

    parser.add_argument(
        "--lj-cutoff",
        type=float,
        default=10.0,
        help="LJ cutoff distance in Angstrom (default: 10.0)",
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Timestep in fs (default: 1.0)",
    )

    parser.add_argument(
        "--compute-stress",
        action="store_true",
        default=True,
        help="Compute per-atom stress for MLIP training (default: True)",
    )

    parser.add_argument(
        "--compute-centro",
        action="store_true",
        default=True,
        help="Compute centrosymmetry parameter (default: True)",
    )

    parser.set_defaults(func=run_metal_surface)


def run_metal_surface(args):
    """Run the metal surface LAMMPS input generation.

    Args:
        args: Command line arguments
    """
    # Create parameters
    parameters = MetalSurfaceLAMMPSParameters(
        data_file=args.data_file,
        metal_type=args.metal,
        ensemble=args.ensemble,
        temperatures=args.temperature,
        pressure=args.pressure,
        equilibration_time=args.equilibration_time,
        production_time=args.production_time,
        dump_frequency=args.dump_frequency,
        fix_bottom_layers=args.fix_layers,
        lj_cutoff=args.lj_cutoff,
        timestep=args.timestep,
        compute_stress=args.compute_stress,
        compute_centro=args.compute_centro,
        output_file=args.output,
    )

    # Create generator
    generator = MetalSurfaceLAMMPSGenerator(parameters)

    # Generate LAMMPS input file
    generator.generate()

    logger.info(f"LAMMPS input file written to: {args.output}")
    logger.info(f"Metal: {args.metal}")
    logger.info(f"Ensemble: {args.ensemble}")
    logger.info(f"Temperature(s): {args.temperature} K")
    if args.ensemble == "NPT":
        logger.info(f"Pressure: {args.pressure} bar")
    logger.info(f"Fixed bottom layers: {args.fix_layers}")
    logger.info(f"Equilibration time: {args.equilibration_time} ps")
    logger.info(f"Production time: {args.production_time} ps")
    logger.info(f"Dump frequency: {args.dump_frequency} ps")
    logger.info(f"LJ cutoff: {args.lj_cutoff} Angstrom")


def main():
    """Main entry point for the mlip-lammps-metal-surface command."""
    parser = argparse.ArgumentParser(
        prog="mlip-lammps-metal-surface",
        description="Generate LAMMPS input files for metal surface simulations with LJ potentials",
    )

    # Required arguments
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the LAMMPS data file containing metal surface structure",
    )

    parser.add_argument(
        "--metal",
        "-m",
        type=str,
        required=True,
        choices=["Cu", "Ag", "Au", "Ni", "Pd", "Pt", "Al"],
        help="Metal type for the surface",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="input.lammps",
        help="Output LAMMPS input filename (default: input.lammps)",
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NPT", "NVT", "NVE"],
        help="Simulation ensemble (default: NVT)",
    )

    parser.add_argument(
        "--temperature",
        "-T",
        nargs="+",
        type=float,
        default=[330.0],
        help="Temperature(s) in K for sampling (default: 330.0)",
    )

    parser.add_argument(
        "--pressure",
        "-P",
        type=float,
        default=1.0,
        help="Pressure in bar for NPT ensemble (default: 1.0)",
    )

    parser.add_argument(
        "--equilibration-time",
        type=float,
        default=100.0,
        help="Equilibration time in ps (default: 100.0)",
    )

    parser.add_argument(
        "--production-time",
        type=float,
        default=500.0,
        help="Production time in ps (default: 500.0)",
    )

    parser.add_argument(
        "--dump-frequency",
        type=float,
        default=1.0,
        help="Trajectory dump frequency in ps (default: 1.0)",
    )

    parser.add_argument(
        "--fix-layers",
        type=int,
        default=2,
        help="Number of bottom layers to fix (default: 2)",
    )

    parser.add_argument(
        "--lj-cutoff",
        type=float,
        default=10.0,
        help="LJ cutoff distance in Angstrom (default: 10.0)",
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Timestep in fs (default: 1.0)",
    )

    parser.add_argument(
        "--compute-stress",
        action="store_true",
        default=True,
        help="Compute per-atom stress for MLIP training (default: True)",
    )

    parser.add_argument(
        "--compute-centro",
        action="store_true",
        default=True,
        help="Compute centrosymmetry parameter (default: True)",
    )

    args = parser.parse_args()
    run_metal_surface(args)


if __name__ == "__main__":
    main()
