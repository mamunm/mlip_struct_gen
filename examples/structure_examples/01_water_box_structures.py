#!/usr/bin/env python
"""
Example: Generate water box structures.

This example demonstrates how to:
1. Create water boxes with different densities
2. Use different water models (SPC/E, TIP3P, TIP4P)
3. Generate structures in various output formats
4. Create systems of different sizes for scaling studies
"""

import subprocess
from pathlib import Path

# Create output directory
Path("water_structures").mkdir(exist_ok=True)

# Step 1: Basic water box with different sizes
print("Step 1: Generating water boxes of different sizes...")
sizes = [
    (100, "small"),
    (500, "medium"),
    (1000, "large"),
    (5000, "xlarge"),
]

for n_water, label in sizes:
    print(f"\nGenerating {label} water box ({n_water} molecules)...")
    cmd = [
        "mlip-struct-gen", "generate", "water-box",
        "--n-water", str(n_water),
        "--density", "0.997",
        "--water-model", "SPC/E",
        "--output", f"water_structures/water_{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created water_{label}.data")

# Step 2: Different water models
print("\nStep 2: Different water models (500 molecules each)...")
models = ["SPC/E", "SPCE", "TIP3P", "TIP4P"]

for model in models:
    safe_name = model.replace("/", "_")
    print(f"\nGenerating {model} water box...")
    cmd = [
        "mlip-struct-gen", "generate", "water-box",
        "--n-water", "500",
        "--density", "0.997",
        "--water-model", model,
        "--output", f"water_structures/water_{safe_name}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created water_{safe_name}.data")

# Step 3: Density variations (important for phase studies)
print("\nStep 3: Water boxes at different densities...")
densities = [
    (0.8, "vapor"),
    (0.997, "liquid"),
    (1.1, "compressed"),
]

for density, phase in densities:
    print(f"\nGenerating {phase} phase water (density={density} g/cm³)...")
    cmd = [
        "mlip-struct-gen", "generate", "water-box",
        "--n-water", "500",
        "--density", str(density),
        "--output", f"water_structures/water_{phase}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created water_{phase}.data")

# Step 4: Different output formats
print("\nStep 4: Different output formats for visualization/analysis...")
formats = [
    ("water.xyz", "XYZ for visualization"),
    ("water.data", "LAMMPS data file"),
    ("POSCAR", "VASP POSCAR format"),
]

for filename, description in formats:
    print(f"\nGenerating {description}...")
    cmd = [
        "mlip-struct-gen", "generate", "water-box",
        "--n-water", "300",
        "--density", "0.997",
        "--output", f"water_structures/{filename}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {filename}")

# Step 5: Custom box dimensions
print("\nStep 5: Water box with custom dimensions...")
cmd = [
    "mlip-struct-gen", "generate", "water-box",
    "--n-water", "500",
    "--box", "30", "30", "60",  # Elongated box for interface studies
    "--output", "water_structures/water_elongated.data"
]
subprocess.run(cmd, check=True)
print("✓ Created elongated water box (30x30x60 Å)")

# Step 6: Generate multiple configurations for ensemble averaging
print("\nStep 6: Multiple configurations with different random seeds...")
for i in range(5):
    cmd = [
        "mlip-struct-gen", "generate", "water-box",
        "--n-water", "500",
        "--density", "0.997",
        "--seed", str(12345 + i),
        "--output", f"water_structures/water_config_{i+1}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created configuration {i+1}")

# Step 7: Ice-like (low density) initial configuration
print("\nStep 7: Ice-like low density configuration...")
cmd = [
    "mlip-struct-gen", "generate", "water-box",
    "--n-water", "512",  # Perfect cube number
    "--density", "0.92",  # Ice density
    "--output", "water_structures/water_ice_like.data"
]
subprocess.run(cmd, check=True)
print("✓ Created ice-like configuration")

# Summary
print("\n" + "="*60)
print("STRUCTURE GENERATION COMPLETE!")
print("="*60)
print("\nGenerated structures in water_structures/:")
print("  - Different sizes: 100 to 5000 molecules")
print("  - Different models: SPC/E, TIP3P, TIP4P")
print("  - Different densities: vapor, liquid, compressed")
print("  - Different formats: .data, .xyz, POSCAR")
print("  - Multiple configurations for statistics")

print("\nNext steps:")
print("1. Visualize with OVITO: ovito water_structures/*.xyz")
print("2. Generate LAMMPS inputs: mlip-lammps-water water_structures/*.data")
print("3. Check structure quality with ASE or MDAnalysis")

print("\nUseful analysis:")
print("  - Radial distribution functions")
print("  - Hydrogen bond networks")
print("  - Density profiles")
print("  - Molecular orientations")
