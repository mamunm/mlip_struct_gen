#!/usr/bin/env python
"""
Example: Generate LAMMPS input for water box simulation.

This example demonstrates how to:
1. Create a water box structure
2. Generate LAMMPS input for MD simulation
3. Run the simulation (if LAMMPS is installed)
"""

import subprocess
from pathlib import Path

# Step 1: Generate water box structure
print("Step 1: Generating water box structure...")
cmd = [
    "mlip-struct-gen", "generate", "water-box",
    "--n-water", "500",
    "--density", "0.997",
    "--water-model", "SPC/E",
    "--output", "water_box.data"
]
subprocess.run(cmd, check=True)
print("✓ Created water_box.data")

# Step 2: Generate LAMMPS input file
print("\nStep 2: Generating LAMMPS input file...")
cmd = [
    "mlip-lammps-water", "water_box.data",
    "--water-model", "SPC/E",
    "--ensemble", "NPT",
    "--temperature", "300",
    "--pressure", "1.0",
    "--equilibration-time", "50",
    "--production-time", "200",
    "--dump-frequency", "1.0",
    "--output", "in.water"
]
subprocess.run(cmd, check=True)
print("✓ Created in.water")

# Step 3: Show how to run simulation
print("\nStep 3: To run the simulation:")
print("  lmp -i in.water")
print("\nThis will produce:")
print("  - trajectory.lammpstrj: atomic trajectories")
print("  - final.data: final configuration")
print("  - Thermo output showing temperature, energy, pressure")

# Step 4: Multiple temperature sampling
print("\nStep 4: Generate inputs for multiple temperatures:")
for temp in [280, 300, 320, 340]:
    cmd = [
        "mlip-lammps-water", "water_box.data",
        "--water-model", "SPC/E",
        "--temperature", str(temp),
        "--output", f"in.water_T{temp}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.water_T{temp}")

print("\nExample complete! Generated LAMMPS inputs for water box simulations.")
