#!/usr/bin/env python3
"""
Example 3: Generate water box with specified box size and exact number of molecules.

This example demonstrates generating a water box when both box size and
number of molecules are specified. The density parameter is ignored, and
the system will pack exactly the specified number of molecules in the box.

Parameter combination: box_size + n_molecules
- Ignores density completely
- Useful for specific system sizes or computational constraints
- May result in non-physical densities if not carefully chosen
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
    """Generate water box with exact number of molecules."""

    # Example 3a: Small system with POSCAR output
    params_small = WaterBoxGeneratorParameters(
        box_size=20.0,  # 20x20x20 Å cubic box
        n_molecules=100,  # Exactly 100 water molecules
        output_file="water_100_molecules",  # POSCAR has no extension
        water_model="TIP3P",
        output_format="poscar",  # VASP POSCAR format
        tolerance=2.0,
        seed=12345,
        log=True,
    )

    generator_small = WaterBoxGenerator(params_small)
    output_small = generator_small.run()
    print(f"Generated small system (100 molecules): {output_small}")

    # Example 3b: Medium system with LAMMPS output
    params_medium = WaterBoxGeneratorParameters(
        box_size=(30.0, 30.0, 40.0),  # Rectangular box
        n_molecules=500,  # Exactly 500 water molecules
        output_file="water_500_molecules.data",  # LAMMPS data file
        water_model="SPC/E",
        output_format="lammps",  # Includes bonds and angles
        tolerance=1.8,
        seed=54321,
        log=True,
    )

    generator_medium = WaterBoxGenerator(params_medium)
    output_medium = generator_medium.run()
    print(f"Generated medium system (500 molecules): {output_medium}")

    # Example 3c: Large sparse system with XYZ output
    params_sparse = WaterBoxGeneratorParameters(
        box_size=50.0,  # Large 50x50x50 Å box
        n_molecules=200,  # Only 200 molecules (sparse)
        output_file="water_sparse.xyz",
        water_model="TIP4P",
        output_format="xyz",  # Simple XYZ format
        tolerance=2.5,
        seed=99999,
        log=True,
    )

    generator_sparse = WaterBoxGenerator(params_sparse)
    output_sparse = generator_sparse.run()
    print(f"Generated sparse system (200 molecules in 50Å box): {output_sparse}")

    # Example 3d: Dense packing test
    params_dense = WaterBoxGeneratorParameters(
        box_size=(25.0, 25.0, 25.0),
        n_molecules=1000,  # 1000 molecules in relatively small box
        output_file="water_dense_packed.xyz",
        water_model="SPC/E",
        output_format="xyz",
        tolerance=1.5,  # Tighter tolerance for dense packing
        seed=7777,
        log=True,
    )

    generator_dense = WaterBoxGenerator(params_dense)
    output_dense = generator_dense.run()
    print(f"Generated dense system (1000 molecules): {output_dense}")

    print("\n" + "="*60)
    print("Summary of exact molecule specifications:")
    print("- Small: 100 molecules in 20Å box → POSCAR format")
    print("- Medium: 500 molecules in 30x30x40Å → LAMMPS format")
    print("- Sparse: 200 molecules in 50Å box → XYZ format")
    print("- Dense: 1000 molecules in 25Å box → XYZ format")
    print("\nNote: Actual density depends on box size and molecule count")
    print("="*60)


if __name__ == "__main__":
    main()