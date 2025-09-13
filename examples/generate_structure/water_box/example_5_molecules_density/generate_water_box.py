#!/usr/bin/env python3
"""
Example 5: Generate water box with specified molecules and custom density.

This example demonstrates generating a water box when both number of molecules
and density are specified. The system will compute the required cubic box size
to accommodate the exact number of molecules at the specified density.

Parameter combination: n_molecules + density
- Automatically computes cubic box size for given molecules at specified density
- Useful for studying specific densities with exact molecule counts
- Box size = (volume)^(1/3) where volume = (n_molecules * M_water) / (density * N_A)
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
    """Generate water box with specified molecules and density."""

    # Example 5a: Ice conditions with POSCAR output
    params_ice = WaterBoxGeneratorParameters(
        n_molecules=512,  # 8x8x8 = 512 molecules
        density=0.92,  # Ice density at 0°C
        output_file="water_512_ice_density",  # POSCAR format (no extension)
        water_model="TIP4P",  # TIP4P often used for ice
        output_format="poscar",  # VASP format
        tolerance=2.2,
        seed=10001,
        log=True,
    )

    generator_ice = WaterBoxGenerator(params_ice)
    output_ice = generator_ice.run()
    print(f"Generated ice-density system (512 molecules at 0.92 g/cm³): {output_ice}")
    print("Box size computed to be larger due to lower density")

    # Example 5b: Room temperature water with LAMMPS output
    params_room = WaterBoxGeneratorParameters(
        n_molecules=864,  # Specific molecule count
        density=0.997,  # Water at 25°C and 1 atm
        output_file="water_864_room_temp.data",
        water_model="SPC/E",
        output_format="lammps",  # LAMMPS with full topology
        tolerance=2.0,
        seed=20002,
        log=True,
    )

    generator_room = WaterBoxGenerator(params_room)
    output_room = generator_room.run()
    print(f"\nGenerated room-temp system (864 molecules at 0.997 g/cm³): {output_room}")

    # Example 5c: High temperature/low density with XYZ output
    params_hot = WaterBoxGeneratorParameters(
        n_molecules=300,
        density=0.6,  # Hot water/steam-like density
        output_file="water_300_hot.xyz",
        water_model="TIP3P",
        output_format="xyz",  # Simple XYZ format
        tolerance=2.5,  # Larger tolerance for sparse system
        seed=30003,
        log=True,
    )

    generator_hot = WaterBoxGenerator(params_hot)
    output_hot = generator_hot.run()
    print(f"\nGenerated hot/sparse system (300 molecules at 0.6 g/cm³): {output_hot}")
    print("Larger box computed due to low density")

    # Example 5d: Compressed water with XYZ output
    params_compressed = WaterBoxGeneratorParameters(
        n_molecules=1000,
        density=1.2,  # High pressure conditions
        output_file="water_1000_compressed.xyz",
        water_model="SPC/E",
        output_format="xyz",
        tolerance=1.5,  # Tighter tolerance for dense packing
        seed=40004,
        log=True,
    )

    generator_compressed = WaterBoxGenerator(params_compressed)
    output_compressed = generator_compressed.run()
    print(f"\nGenerated compressed system (1000 molecules at 1.2 g/cm³): {output_compressed}")
    print("Smaller box computed due to high density")

    # Example 5e: Supercritical conditions with LAMMPS output
    params_super = WaterBoxGeneratorParameters(
        n_molecules=150,
        density=0.3,  # Supercritical water density
        output_file="water_150_supercritical.data",
        water_model="TIP3P",
        output_format="lammps",
        tolerance=3.0,  # Very loose tolerance for extremely sparse system
        seed=50005,
        log=True,
    )

    generator_super = WaterBoxGenerator(params_super)
    output_super = generator_super.run()
    print(f"\nGenerated supercritical system (150 molecules at 0.3 g/cm³): {output_super}")
    print("Very large box computed due to extremely low density")

    print("\n" + "="*60)
    print("Summary of n_molecules + density combinations:")
    print("- Ice: 512 molecules @ 0.92 g/cm³ → Larger box")
    print("- Room temp: 864 molecules @ 0.997 g/cm³ → Standard box")
    print("- Hot/sparse: 300 molecules @ 0.6 g/cm³ → Larger box")
    print("- Compressed: 1000 molecules @ 1.2 g/cm³ → Smaller box")
    print("- Supercritical: 150 molecules @ 0.3 g/cm³ → Very large box")
    print("\nNote: Box size inversely proportional to density for fixed n_molecules")
    print("="*60)


if __name__ == "__main__":
    main()