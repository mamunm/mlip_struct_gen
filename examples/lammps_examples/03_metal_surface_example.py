#!/usr/bin/env python
"""
Example: Generate LAMMPS input for metal surface simulation.

This example demonstrates how to:
1. Create metal surface structures
2. Generate LAMMPS input with LJ potentials
3. Set up simulations with fixed bottom layers
"""

import subprocess
from pathlib import Path

# Step 1: Generate metal surfaces
print("Step 1: Generating metal surface structures...")

metals = ["Au", "Pt", "Ag", "Cu"]
for metal in metals:
    print(f"\nGenerating {metal}(111) surface...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", metal,
        "--size", "6", "6", "8",  # 6x6 unit cells, 8 layers
        "--vacuum", "20",
        "--output", f"{metal.lower()}_111.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {metal.lower()}_111.data")

# Step 2: Generate LAMMPS inputs with different settings
print("\nStep 2: Generating LAMMPS input files...")

# Standard NVT simulation
for metal in metals:
    cmd = [
        "mlip-lammps-metal-surface", f"{metal.lower()}_111.data",
        "--metal", metal,
        "--ensemble", "NVT",
        "--temperature", "300",
        "--fix-layers", "0",  # No fixed layers for bulk metal
        "--output", f"in.{metal.lower()}_nvt"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.{metal.lower()}_nvt")

# Step 3: Surface with fixed bottom layers (for interface simulations)
print("\nStep 3: Surface simulations with fixed bottom layers...")

metal = "Pt"
cmd = [
    "mlip-struct-gen", "generate", "metal-surface",
    "--metal", metal,
    "--size", "8", "8", "12",  # Larger surface for interface
    "--vacuum", "30",
    "--output", "pt_surface_large.data"
]
subprocess.run(cmd, check=True)

# Generate LAMMPS with fixed layers
cmd = [
    "mlip-lammps-metal-surface", "pt_surface_large.data",
    "--metal", metal,
    "--ensemble", "NVT",
    "--temperature", "330",
    "--fix-layers", "3",  # Fix bottom 3 layers
    "--output", "in.pt_surface_fixed"
]
subprocess.run(cmd, check=True)
print("✓ Created Pt surface with fixed bottom layers")

# Step 4: Temperature-dependent simulations
print("\nStep 4: Temperature-dependent simulations...")

temps = [200, 300, 400, 500, 600]
for temp in temps:
    cmd = [
        "mlip-lammps-metal-surface", "au_111.data",
        "--metal", "Au",
        "--temperature", str(temp),
        "--equilibration-time", "50",
        "--production-time", "200",
        "--output", f"in.au_T{temp}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.au_T{temp}")

# Step 5: NPT simulation for thermal expansion
print("\nStep 5: NPT simulation for thermal expansion study...")

cmd = [
    "mlip-lammps-metal-surface", "cu_111.data",
    "--metal", "Cu",
    "--ensemble", "NPT",
    "--temperature", "500",
    "--pressure", "0",  # Zero pressure
    "--output", "in.cu_npt_expansion"
]
subprocess.run(cmd, check=True)
print("✓ Created NPT simulation for thermal expansion")

print("\nStep 6: Analysis suggestions:")
print("  - trajectory.lammpstrj: visualize with OVITO")
print("  - Thermo output: extract lattice constants vs temperature")
print("  - Surface reconstruction: monitor top layer positions")

print("\nExample complete! Generated LAMMPS inputs for metal surface simulations.")
