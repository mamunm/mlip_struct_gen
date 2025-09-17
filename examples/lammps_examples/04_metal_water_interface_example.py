#!/usr/bin/env python
"""
Example: Generate LAMMPS input for metal-water interface simulation.

This example demonstrates how to:
1. Create metal-water interface structures
2. Generate LAMMPS input with proper metal-water interactions
3. Monitor water temperature separately from metal
"""

import subprocess
from pathlib import Path

# Step 1: Generate metal-water interface structures
print("Step 1: Generating metal-water interface structures...")

interfaces = [
    ("Pt", 100, "pt_water_small"),
    ("Pt", 300, "pt_water_medium"),
    ("Au", 200, "au_water"),
    ("Ag", 200, "ag_water"),
]

for metal, n_water, label in interfaces:
    print(f"\nGenerating {label}...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", metal,
        "--size", "5", "5", "8",
        "--n-water", str(n_water),
        "--gap", "2.5",
        "--vacuum", "20",
        "--output", f"{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 2: Generate LAMMPS inputs with fixed bottom layers
print("\nStep 2: Generating LAMMPS input files...")

for metal, n_water, label in interfaces:
    cmd = [
        "mlip-lammps-metal-water", f"{label}.data",
        "--metal", metal,
        "--ensemble", "NVT",
        "--temperature", "300",
        "--fix-layers", "2",  # Fix bottom 2 metal layers
        "--equilibration-time", "100",
        "--production-time", "500",
        "--output", f"in.{label}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.{label}")

# Step 3: Temperature study - water behavior at interface
print("\nStep 3: Temperature-dependent interface study...")

temps = [280, 300, 330, 360]
for temp in temps:
    label = f"pt_water_T{temp}"
    # Use existing structure
    cmd = [
        "mlip-lammps-metal-water", "pt_water_medium.data",
        "--metal", "Pt",
        "--temperature", str(temp),
        "--fix-layers", "3",
        "--output", f"in.{label}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.{label}")

# Step 4: Different water models
print("\nStep 4: Different water models at interface...")

# First create structure with TIP3P water
cmd = [
    "mlip-struct-gen", "generate", "metal-water",
    "--metal", "Au",
    "--size", "6", "6", "10",
    "--n-water", "250",
    "--water-model", "TIP3P",
    "--output", "au_tip3p.data"
]
subprocess.run(cmd, check=True)

# Generate LAMMPS inputs for different water models
for model in ["SPC/E", "TIP3P"]:
    data_file = "au_water.data" if model == "SPC/E" else "au_tip3p.data"
    cmd = [
        "mlip-lammps-metal-water", data_file,
        "--metal", "Au",
        "--water-model", model,
        "--temperature", "330",
        "--output", f"in.au_{model.replace('/', '_')}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created Au-water with {model}")

# Step 5: NPT simulation for interface
print("\nStep 5: NPT simulation for interface equilibration...")

cmd = [
    "mlip-lammps-metal-water", "pt_water_medium.data",
    "--metal", "Pt",
    "--ensemble", "NPT",
    "--temperature", "298.15",
    "--pressure", "1.0",
    "--fix-layers", "2",
    "--equilibration-time", "200",
    "--production-time", "1000",
    "--output", "in.pt_water_npt"
]
subprocess.run(cmd, check=True)
print("✓ Created NPT interface simulation")

print("\nStep 6: Analysis tips:")
print("  - Monitor T_water column in thermo output")
print("  - Water density profile: analyze trajectory.lammpstrj")
print("  - Interface structure: first water layer orientation")
print("  - Hydrogen bonding at interface")

print("\nStep 7: To run and analyze:")
print("  lmp -i in.pt_water_medium")
print("  # Extract water temperature:")
print("  grep 'T_water' log.lammps > water_temp.dat")

print("\nExample complete! Generated LAMMPS inputs for metal-water interface simulations.")
