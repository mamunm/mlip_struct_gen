"""CLI for generating LAMMPS input for salt-water systems."""

import argparse
from pathlib import Path

from mlip_struct_gen.generate_lammps_input.salt_water_box import SaltWaterBoxLAMMPSGenerator
from mlip_struct_gen.generate_lammps_input.salt_water_box.input_parameters import SaltWaterBoxLAMMPSParameters
from mlip_struct_gen.utils.logger import MLIPLogger

logger = MLIPLogger()


def setup_parser(subparsers):
    """Set up the parser for salt-water box LAMMPS input generation.

    Args:
        subparsers: The subparsers object from argparse
    """
    parser = subparsers.add_parser(
        "mlip-salt-water-box",
        help="Generate LAMMPS input for salt-water systems",
        description="Generate LAMMPS input files for salt-water systems with ion parameters"
    )

    # Required arguments
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the LAMMPS data file containing salt-water system"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="input.lammps",
        help="Output LAMMPS input filename (default: input.lammps)"
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NPT", "NVT", "NVE"],
        help="Simulation ensemble (default: NVT)"
    )

    parser.add_argument(
        "--temperature",
        "-T",
        nargs="+",
        type=float,
        default=[330.0],
        help="Temperature(s) in K (default: 330.0)"
    )

    parser.add_argument(
        "--pressure",
        "-P",
        type=float,
        default=1.0,
        help="Pressure in bar for NPT ensemble (default: 1.0)"
    )

    parser.add_argument(
        "--equilibration-time",
        type=float,
        default=100.0,
        help="Equilibration time in ps (default: 100.0)"
    )

    parser.add_argument(
        "--production-time",
        type=float,
        default=500.0,
        help="Production time in ps (default: 500.0)"
    )

    parser.add_argument(
        "--dump-frequency",
        type=float,
        default=1.0,
        help="Trajectory dump frequency in ps (default: 1.0)"
    )

    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P"],
        help="Water model to use (default: SPC/E)"
    )

    parser.add_argument(
        "--coulomb-accuracy",
        type=float,
        default=1.0e-4,
        help="PPPM accuracy for Coulomb interactions (default: 1.0e-4)"
    )

    parser.set_defaults(func=run_salt_water_box)


def run_salt_water_box(args):
    """Run the salt-water box LAMMPS input generation.

    Args:
        args: Command line arguments
    """
    # Create parameters
    parameters = SaltWaterBoxLAMMPSParameters(
        data_file=args.data_file,
        ensemble=args.ensemble,
        temperatures=args.temperature,
        pressure=args.pressure,
        equilibration_time=args.equilibration_time,
        production_time=args.production_time,
        dump_frequency=args.dump_frequency,
        water_model=args.water_model,
        coulomb_accuracy=args.coulomb_accuracy,
    )

    # Create generator
    generator = SaltWaterBoxLAMMPSGenerator(parameters)

    # Set output file in parameters
    parameters.output_file = args.output

    # Generate and write input file
    generator.generate()

    logger.info(f"LAMMPS input file written to: {args.output}")
    logger.info(f"Ensemble: {args.ensemble}")
    logger.info(f"Temperature(s): {args.temperature} K")
    if args.ensemble == "NPT":
        logger.info(f"Pressure: {args.pressure} bar")
    logger.info(f"Water model: {args.water_model}")
    logger.info(f"Equilibration time: {args.equilibration_time} ps")
    logger.info(f"Production time: {args.production_time} ps")
    logger.info(f"Dump frequency: {args.dump_frequency} ps")


def main():
    """Main entry point for the mlip-lammps-salt-water command."""
    parser = argparse.ArgumentParser(
        prog="mlip-lammps-salt-water",
        description="Generate LAMMPS input files for salt-water systems with ion parameters"
    )

    # Required arguments
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the LAMMPS data file containing salt-water system"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="input.lammps",
        help="Output LAMMPS input filename (default: input.lammps)"
    )

    parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NPT", "NVT", "NVE"],
        help="Simulation ensemble (default: NVT)"
    )

    parser.add_argument(
        "--temperature",
        "-T",
        nargs="+",
        type=float,
        default=[330.0],
        help="Temperature(s) in K (default: 330.0)"
    )

    parser.add_argument(
        "--pressure",
        "-P",
        type=float,
        default=1.0,
        help="Pressure in bar for NPT ensemble (default: 1.0)"
    )

    parser.add_argument(
        "--equilibration-time",
        type=float,
        default=100.0,
        help="Equilibration time in ps (default: 100.0)"
    )

    parser.add_argument(
        "--production-time",
        type=float,
        default=500.0,
        help="Production time in ps (default: 500.0)"
    )

    parser.add_argument(
        "--dump-frequency",
        type=float,
        default=1.0,
        help="Trajectory dump frequency in ps (default: 1.0)"
    )

    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "TIP3P", "TIP4P"],
        help="Water model to use (default: SPC/E)"
    )

    parser.add_argument(
        "--coulomb-accuracy",
        type=float,
        default=1.0e-4,
        help="PPPM accuracy for Coulomb interactions (default: 1.0e-4)"
    )

    args = parser.parse_args()
    run_salt_water_box(args)