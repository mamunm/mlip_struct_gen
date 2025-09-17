#!/usr/bin/env python
"""
Example: Generate LAMMPS input for metal-salt-water interface simulation.

This example demonstrates how to:
1. Create complex metal-salt-water interface structures
2. Generate LAMMPS input with proper interactions for all components
3. Monitor temperatures of water and solution separately
"""

import subprocess
from pathlib import Path

# Step 1: Generate metal-salt-water interface structures
print("Step 1: Generating metal-salt-water interface structures...")

systems = [
    ("Pt", "NaCl", 10, 200, "pt_nacl_dilute"),
    ("Pt", "NaCl", 20, 200, "pt_nacl_concentrated"),
    ("Au", "KCl", 15, 250, "au_kcl"),
    ("Ag", "LiCl", 10, 200, "ag_licl"),
]

for metal, salt, n_salt, n_water, label in systems:
    print(f"\nGenerating {label}...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", metal,
        "--size", "6", "6", "10",
        "--n-water", str(n_water),
        "--n-salt", str(n_salt),
        "--salt", salt,
        "--gap", "3.0",
        "--vacuum", "25",
        "--output", f"{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 2: Generate LAMMPS inputs
print("\nStep 2: Generating LAMMPS input files...")

for metal, salt, n_salt, n_water, label in systems:
    cmd = [
        "mlip-lammps-metal-salt-water", f"{label}.data",
        "--metal", metal,
        "--salt", salt,
        "--ensemble", "NVT",
        "--temperature", "330",
        "--fix-layers", "2",
        "--equilibration-time", "150",
        "--production-time", "500",
        "--output", f"in.{label}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created in.{label}")

# Step 3: Concentration study - effect of salt concentration
print("\nStep 3: Salt concentration study at Pt interface...")

concentrations = [5, 10, 20, 40]
for n_salt in concentrations:
    label = f"pt_nacl_{n_salt}ions"

    # Generate structure
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", "Pt",
        "--size", "7", "7", "12",
        "--n-water", "300",
        "--n-salt", str(n_salt),
        "--salt", "NaCl",
        "--output", f"{label}.data"
    ]
    subprocess.run(cmd, check=True)

    # Generate LAMMPS input
    cmd = [
        "mlip-lammps-metal-salt-water", f"{label}.data",
        "--metal", "Pt",
        "--salt", "NaCl",
        "--temperature", "298.15",
        "--fix-layers", "3",
        "--output", f"in.{label}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label} (~{n_salt/20:.1f}M concentration)")

# Step 4: Different salt types at same metal interface
print("\nStep 4: Different salts at Au interface...")

salts = ["NaCl", "KCl", "LiCl", "NaF"]
for salt_type in salts:
    label = f"au_{salt_type.lower()}_interface"

    # Generate structure
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", "Au",
        "--size", "5", "5", "10",
        "--n-water", "200",
        "--n-salt", "15",
        "--salt", salt_type,
        "--output", f"{label}.data"
    ]
    subprocess.run(cmd, check=True)

    # Generate LAMMPS input
    cmd = [
        "mlip-lammps-metal-salt-water", f"{label}.data",
        "--metal", "Au",
        "--salt", salt_type,
        "--temperature", "330",
        "--output", f"in.{label}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created Au-{salt_type}-water interface")

# Step 5: Temperature study for electrochemical conditions
print("\nStep 5: Temperature study for electrochemical conditions...")

temps = [298, 323, 348, 373]  # 25°C, 50°C, 75°C, 100°C
for temp in temps:
    cmd = [
        "mlip-lammps-metal-salt-water", "pt_nacl_concentrated.data",
        "--metal", "Pt",
        "--salt", "NaCl",
        "--temperature", str(temp),
        "--fix-layers", "2",
        "--output", f"in.pt_nacl_T{temp}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created Pt-NaCl-water at {temp}K ({temp-273:.0f}°C)")

# Step 6: NPT simulation for realistic interface
print("\nStep 6: NPT simulation for realistic interface equilibration...")

cmd = [
    "mlip-lammps-metal-salt-water", "pt_nacl_concentrated.data",
    "--metal", "Pt",
    "--salt", "NaCl",
    "--ensemble", "NPT",
    "--temperature", "298.15",
    "--pressure", "1.0",
    "--fix-layers", "3",
    "--equilibration-time", "200",
    "--production-time", "1000",
    "--output", "in.pt_nacl_npt"
]
subprocess.run(cmd, check=True)
print("✓ Created NPT simulation for Pt-NaCl-water")

print("\nStep 7: Analysis suggestions:")
print("  - Monitor T_water and T_soln columns separately")
print("  - Ion distribution at interface: density profiles")
print("  - Double layer structure: ion accumulation near metal")
print("  - Water orientation at interface with ions present")

print("\nStep 8: Example analysis commands:")
print("  # Run simulation")
print("  lmp -i in.pt_nacl_concentrated")
print("")
print("  # Extract temperatures")
print("  grep -E 'Step|T_water|T_soln' log.lammps > temperatures.dat")
print("")
print("  # Visualize with OVITO")
print("  ovito trajectory.lammpstrj")

print("\nStep 9: Expected phenomena:")
print("  - Ion accumulation/depletion at metal interface")
print("  - Modified water structure due to ions")
print("  - Temperature gradients in solution")
print("  - Electric double layer formation")

print("\nExample complete! Generated LAMMPS inputs for metal-salt-water interface simulations.")
