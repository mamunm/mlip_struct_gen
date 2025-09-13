#!/usr/bin/env python3
"""
Example 1: Generate water box with specified box size only.

This example demonstrates generating a water box when only the box size is
specified. The system will use the default density for the chosen water model.

Parameter combination: box_size only
- Uses water model's default density (SPC/E: 0.997 g/cm³)
- Automatically calculates the number of molecules to achieve this density
"""

import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from mlip_struct_gen.generate_structure.water_box import (
    WaterBoxGenerator,
    WaterBoxGeneratorParameters,
)


def main():
    """Generate water box with specified box size."""

    # Example 1a: Cubic box (single dimension)
    params_cubic = WaterBoxGeneratorParameters(
        box_size=30.0,  # Creates 30x30x30 Å cubic box
        output_file="water_cubic_30A.xyz",
        water_model="SPC/E",  # Default model
        tolerance=2.0,  # Default tolerance
        seed=42,  # For reproducibility
        log=True,  # Enable logging to see details
    )

    generator_cubic = WaterBoxGenerator(params_cubic)
    output_cubic = generator_cubic.run()
    print(f"Generated cubic water box: {output_cubic}")

    # Example 1b: Rectangular box (3 dimensions)
    params_rect = WaterBoxGeneratorParameters(
        box_size=(40.0, 30.0, 25.0),  # Creates rectangular box
        output_file="water_rectangular.xyz",
        water_model="TIP3P",  # Different water model
        tolerance=2.0,
        seed=42,
        log=True,
    )

    generator_rect = WaterBoxGenerator(params_rect)
    output_rect = generator_rect.run()
    print(f"Generated rectangular water box: {output_rect}")

    # Example 1c: Generate LAMMPS format output
    params_lammps = WaterBoxGeneratorParameters(
        box_size=25.0,
        output_file="water_box.data",  # LAMMPS data file
        water_model="SPC/E",
        output_format="lammps",  # LAMMPS format with bonds/angles
        seed=42,
        log=True,
    )

    generator_lammps = WaterBoxGenerator(params_lammps)
    output_lammps = generator_lammps.run()
    print(f"Generated LAMMPS water box: {output_lammps}")

    print("\n" + "="*60)
    print("Summary:")
    print("- Cubic box: Uses SPC/E default density (0.997 g/cm³)")
    print("- Rectangular box: Uses TIP3P default density (0.997 g/cm³)")
    print("- LAMMPS format: Includes bonds and angles for MD simulations")
    print("="*60)


if __name__ == "__main__":
    main()