#!/usr/bin/env python
"""
Example: Generate LAMMPS input for salt-water simulation.

This example demonstrates how to:
1. Create a salt-water box structure
2. Generate LAMMPS input with proper ion parameters
3. Set up simulations at different salt concentrations
"""

import subprocess
from pathlib import Path

# Step 1: Generate salt-water box structures at different concentrations
print("Step 1: Generating salt-water structures at different concentrations...")

concentrations = [
    (0, "pure_water"),      # Pure water reference
    (10, "0.5M_NaCl"),      # ~0.5 M NaCl
    (20, "1.0M_NaCl"),      # ~1.0 M NaCl
    (40, "2.0M_NaCl"),      # ~2.0 M NaCl
]

for n_salt, label in concentrations:
    print(f"\nGenerating {label}...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", str(n_salt),
        "--salt", "NaCl",
        "--density", "1.0",
        "--output", f"{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 2: Generate LAMMPS inputs for each concentration
print("\nStep 2: Generating LAMMPS input files...")

for n_salt, label in concentrations:
    if n_salt == 0:
        # Use water generator for pure water
        cmd = [
            "mlip-lammps-water", f"{label}.data",
            "--ensemble", "NPT",
            "--temperature", "298.15",
            "--output", f"in.{label}"
        ]
    else:
        # Use salt-water generator
        cmd = [
            "mlip-lammps-salt-water", f"{label}.data",
            "--salt", "NaCl",
            "--ensemble", "NPT",
            "--temperature", "298.15",
            "--output", f"in.{label}"
        ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.{label}")

# Step 3: Different salt types
print("\nStep 3: Different salt types at same concentration...")

salts = ["NaCl", "KCl", "LiCl", "NaF"]
for salt in salts:
    print(f"\nGenerating {salt} solution...")
    # Generate structure
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", "20",
        "--salt", salt,
        "--output", f"{salt}_solution.data"
    ]
    subprocess.run(cmd, check=True)

    # Generate LAMMPS input
    cmd = [
        "mlip-lammps-salt-water", f"{salt}_solution.data",
        "--salt", salt,
        "--temperature", "330",
        "--output", f"in.{salt}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {salt} simulation files")

print("\nStep 4: To run simulations:")
print("  for file in in.*.lammps; do")
print("    lmp -i $file")
print("  done")

print("\nExample complete! Generated LAMMPS inputs for salt-water simulations.")
