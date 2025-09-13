#!/usr/bin/env python3
"""
Example 2: Generate water box with specified box size and custom density.

This example demonstrates generating a water box when both box size and
density are specified. The system will calculate the number of molecules
needed to achieve the specified density in the given box.

Parameter combination: box_size + density
- Custom density overrides the water model's default
- Useful for studying density effects or matching experimental conditions
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
    """Generate water box with custom density."""

    # Example 2a: Ice-like density (lower than normal)
    params_ice = WaterBoxGeneratorParameters(
        box_size=30.0,  # 30x30x30 Å cubic box
        output_file="water_ice_density.xyz",
        water_model="TIP4P",
        density=0.92,  # Ice density at 0°C (g/cm³)
        tolerance=2.0,
        seed=42,
        log=True,
    )

    generator_ice = WaterBoxGenerator(params_ice)
    output_ice = generator_ice.run()
    print(f"Generated ice-density water box: {output_ice}")

    # Example 2b: High pressure density
    params_high = WaterBoxGeneratorParameters(
        box_size=(35.0, 35.0, 35.0),
        output_file="water_high_pressure.xyz",
        water_model="SPC/E",
        density=1.1,  # Higher density (compressed water)
        tolerance=1.8,  # Tighter tolerance for denser packing
        seed=42,
        log=True,
    )

    generator_high = WaterBoxGenerator(params_high)
    output_high = generator_high.run()
    print(f"Generated high-pressure water box: {output_high}")

    # Example 2c: Supercritical water density
    params_super = WaterBoxGeneratorParameters(
        box_size=40.0,
        output_file="water_supercritical.data",
        water_model="SPC/E",
        density=0.3,  # Supercritical water density
        output_format="lammps",
        tolerance=2.5,  # Looser tolerance for sparse system
        seed=42,
        log=True,
    )

    generator_super = WaterBoxGenerator(params_super)
    output_super = generator_super.run()
    print(f"Generated supercritical water box: {output_super}")

    # Example 2d: Standard conditions with POSCAR output
    params_poscar = WaterBoxGeneratorParameters(
        box_size=25.0,
        output_file="water_standard",  # Will add no extension for POSCAR
        water_model="TIP3P",
        density=1.0,  # Standard water density at 25°C
        output_format="poscar",  # VASP POSCAR format
        seed=42,
        log=True,
    )

    generator_poscar = WaterBoxGenerator(params_poscar)
    output_poscar = generator_poscar.run()
    print(f"Generated POSCAR water box: {output_poscar}")

    print("\n" + "="*60)
    print("Summary of density variations:")
    print("- Ice density (0.92 g/cm³): Fewer molecules, ice-like")
    print("- High pressure (1.1 g/cm³): More molecules, compressed")
    print("- Supercritical (0.3 g/cm³): Very sparse, gas-like")
    print("- Standard (1.0 g/cm³): Normal liquid water at 25°C")
    print("="*60)


if __name__ == "__main__":
    main()