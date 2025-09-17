#!/usr/bin/env python3
"""
Example 4: Generate water box with only number of molecules specified.

This example demonstrates generating a water box when only the number of
molecules is specified. The system will use the water model's default
density to compute the required cubic box size.

Parameter combination: n_molecules only
- Automatically computes cubic box size
- Uses water model's default density
- Ideal when you need a specific number of molecules for computational reasons
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
    """Generate water box with specified number of molecules only."""

    # Example 4a: Small system for testing (XYZ format)
    params_test = WaterBoxGeneratorParameters(
        n_molecules=216,  # 6x6x6 = 216, common test size
        output_file="water_216_test.xyz",
        water_model="SPC/E",
        output_format="xyz",  # Simple XYZ for visualization
        tolerance=2.0,
        seed=11111,
        log=True,
    )

    generator_test = WaterBoxGenerator(params_test)
    output_test = generator_test.run()
    print(f"Generated test system (216 molecules): {output_test}")
    print("Box size auto-computed based on SPC/E density (0.997 g/cm³)")

    # Example 4b: Standard MD simulation size (LAMMPS format)
    params_md = WaterBoxGeneratorParameters(
        n_molecules=1000,  # Common MD simulation size
        output_file="water_1000_md.data",
        water_model="TIP3P",
        output_format="lammps",  # LAMMPS with topology
        tolerance=2.0,
        seed=22222,
        log=True,
    )

    generator_md = WaterBoxGenerator(params_md)
    output_md = generator_md.run()
    print(f"\nGenerated MD system (1000 molecules): {output_md}")
    print("Box size auto-computed based on TIP3P density (0.997 g/cm³)")

    # Example 4c: Large system (POSCAR format)
    params_large = WaterBoxGeneratorParameters(
        n_molecules=2000,  # Larger system
        output_file="water_2000_large",  # No extension for POSCAR
        water_model="TIP4P",
        output_format="poscar",  # VASP format
        tolerance=1.8,
        seed=33333,
        log=True,
    )

    generator_large = WaterBoxGenerator(params_large)
    output_large = generator_large.run()
    print(f"\nGenerated large system (2000 molecules): {output_large}")
    print("Box size auto-computed based on TIP4P density (0.997 g/cm³)")

    # Example 4d: Minimal system for debugging
    params_minimal = WaterBoxGeneratorParameters(
        n_molecules=50,  # Very small system
        output_file="water_50_minimal.xyz",
        water_model="SPC/E",
        output_format="xyz",
        tolerance=2.5,
        seed=44444,
        log=True,
    )

    generator_minimal = WaterBoxGenerator(params_minimal)
    output_minimal = generator_minimal.run()
    print(f"\nGenerated minimal system (50 molecules): {output_minimal}")

    # Example 4e: Save artifacts for inspection
    params_artifacts = WaterBoxGeneratorParameters(
        n_molecules=300,
        output_file="water_300_with_artifacts.data",
        water_model="TIP3P",
        output_format="lammps",
        tolerance=2.0,
        seed=55555,
        log=True,
    )

    generator_artifacts = WaterBoxGenerator(params_artifacts)
    output_artifacts = generator_artifacts.run(save_artifacts=True)
    print(f"\nGenerated system with artifacts (300 molecules): {output_artifacts}")
    print("Check 'artifacts' directory for packmol.inp and water.xyz files")

    print("\n" + "=" * 60)
    print("Summary of auto-computed box sizes:")
    print("- All boxes are cubic when auto-computed")
    print("- Box size = (volume)^(1/3) where volume = mass/density")
    print("- Default densities: SPC/E=0.997, TIP3P=0.997, TIP4P=0.997 g/cm³")
    print("\nOutput formats used:")
    print("- XYZ: Simple coordinate format (examples 1, 4)")
    print("- LAMMPS: Full topology with bonds/angles (examples 2, 5)")
    print("- POSCAR: VASP format with sorted elements (example 3)")
    print("=" * 60)


if __name__ == "__main__":
    main()
