#!/usr/bin/env python3
"""CLI for generating LAMMPS input files for water box simulations."""

import argparse
import sys
from pathlib import Path

from mlip_struct_gen.generate_lammps_input.water_box import (
    WaterBoxLAMMPSGenerator,
    WaterBoxLAMMPSParameters,
)
from mlip_struct_gen.utils.json_utils import save_parameters_to_json
from mlip_struct_gen.utils.logger import MLIPLogger

logger = MLIPLogger()


def add_parser(parser: argparse.ArgumentParser) -> None:
    """Add water box LAMMPS input arguments to parser."""

    # Required input
    parser.add_argument(
        "data_file",
        type=str,
        help="LAMMPS data file containing the water box structure",
    )

    # Essential parameters
    parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NPT", "NVT", "NVE"],
        help="Ensemble for simulation (default: NVT)",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        nargs="+",
        default=[330.0],
        help="Temperature(s) in K for sampling. Multiple values for multi-T MLIP training (default: 330)",
    )
    parser.add_argument(
        "--pressure",
        "-P",
        type=float,
        default=1.0,
        help="Pressure in bar for NPT ensemble (default: 1.0)",
    )

    # Simulation times
    parser.add_argument(
        "--equilibration-time",
        type=float,
        default=100.0,
        help="Equilibration time in ps (default: 100)",
    )
    parser.add_argument(
        "--production-time",
        type=float,
        default=500.0,
        help="Production time in ps (default: 500)",
    )
    parser.add_argument(
        "--dump-frequency",
        type=float,
        default=1.0,
        help="Frequency for saving MLIP training snapshots in ps (default: 1.0)",
    )

    # Water model
    parser.add_argument(
        "--water-model",
        type=str,
        default="SPC/E",
        choices=["SPC/E", "SPCE", "TIP3P", "TIP4P"],
        help="Water model to use (default: SPC/E)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for velocity initialization (default: 12345)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output LAMMPS input file name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Save input parameters to lammps_params.json",
    )


def main() -> int:
    """Main entry point for water box LAMMPS input generation."""

    examples_text = """
Examples:
  # Basic NVT simulation at 330K for MLIP training
  mlip-lammps-water water.data

  # Multi-temperature sampling for robust MLIP training
  mlip-lammps-water water.data --temperature 300 330 360

  # NPT ensemble at different pressure
  mlip-lammps-water water.data --ensemble NPT --pressure 100

  # Longer production run with more frequent sampling
  mlip-lammps-water water.data --production-time 1000 --dump-frequency 0.5

  # TIP3P water model
  mlip-lammps-water water.data --water-model TIP3P

  # Save parameters for reproducibility
  mlip-lammps-water water.data --temperature 330 360 --save-input
"""

    parser = argparse.ArgumentParser(
        description="Generate LAMMPS input files for water box MD simulations optimized for MLIP training data generation",
        epilog=examples_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_parser(parser)
    args = parser.parse_args()

    try:
        # Create parameters object
        params = WaterBoxLAMMPSParameters(
            data_file=args.data_file,
            ensemble=args.ensemble,
            temperatures=(
                args.temperature if isinstance(args.temperature, list) else [args.temperature]
            ),
            pressure=args.pressure,
            equilibration_time=args.equilibration_time,
            production_time=args.production_time,
            dump_frequency=args.dump_frequency,
            water_model=args.water_model,
            seed=args.seed,
            output_file=args.output,
        )

        # Save input parameters if requested
        if args.save_input:
            save_parameters_to_json(params, "lammps_params.json")

        # Generate LAMMPS input files
        generator = WaterBoxLAMMPSGenerator(params)
        generator.run()

        # Log summary
        if len(params.temperatures) > 1:
            logger.info(f"Successfully generated {len(params.temperatures)} LAMMPS input files:")
            for temp in params.temperatures:
                data_stem = Path(params.data_file).stem
                logger.info(f"  - in_{data_stem}_T{temp:.0f}.lammps")
        else:
            logger.info(f"Successfully generated LAMMPS input file: {params.output_file}")

        logger.info(f"Ensemble: {params.ensemble}")
        logger.info(f"Temperature(s): {params.temperatures} K")
        if params.ensemble == "NPT":
            logger.info(f"Pressure: {params.pressure} bar")
        logger.info(f"Equilibration: {params.equilibration_time} ps")
        logger.info(f"Production: {params.production_time} ps")
        logger.info(f"MLIP sampling: every {params.dump_frequency} ps")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
