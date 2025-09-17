#!/usr/bin/env python
"""
Example: Generate salt-water solution structures.

This example demonstrates how to:
1. Create salt solutions at various concentrations
2. Use different salt types (NaCl, KCl, LiCl, etc.)
3. Generate structures for conductivity studies
4. Create systems for ion-specific effects studies
"""

import subprocess
from pathlib import Path

# Create output directory
Path("salt_water_structures").mkdir(exist_ok=True)

# Step 1: NaCl solutions at different concentrations
print("Step 1: NaCl solutions at different concentrations...")
print("(Approximate molar concentrations for 500 water molecules)")

concentrations = [
    (0, 0.0, "pure_water"),
    (5, 0.5, "0.5M_NaCl"),
    (10, 1.0, "1.0M_NaCl"),
    (20, 2.0, "2.0M_NaCl"),
    (30, 3.0, "3.0M_NaCl"),
    (50, 5.0, "5.0M_NaCl"),
]

for n_salt, molarity, label in concentrations:
    print(f"\nGenerating {label} (~{molarity}M)...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", str(n_salt),
        "--salt", "NaCl",
        "--density", "1.0" if n_salt == 0 else str(1.0 + 0.04 * molarity),  # Adjust density
        "--output", f"salt_water_structures/{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 2: Different salt types at same concentration
print("\nStep 2: Different salt types at ~1M concentration...")
salts = [
    ("NaCl", "sodium_chloride"),
    ("KCl", "potassium_chloride"),
    ("LiCl", "lithium_chloride"),
    ("NaF", "sodium_fluoride"),
    ("KF", "potassium_fluoride"),
    ("LiF", "lithium_fluoride"),
    ("NaBr", "sodium_bromide"),
    ("KBr", "potassium_bromide"),
]

for salt, name in salts:
    print(f"\nGenerating {salt} solution...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", "10",
        "--salt", salt,
        "--output", f"salt_water_structures/{name}_1M.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {name}_1M.data")

# Step 3: Large-scale systems for property calculations
print("\nStep 3: Large-scale salt solutions for bulk properties...")
sizes = [
    (1000, 20, "large_dilute"),
    (2000, 80, "large_concentrated"),
    (5000, 200, "xlarge_bulk"),
]

for n_water, n_salt, label in sizes:
    print(f"\nGenerating {label} ({n_water} water, {n_salt} ion pairs)...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", str(n_water),
        "--n-salt", str(n_salt),
        "--salt", "NaCl",
        "--output", f"salt_water_structures/{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 4: Mixed salt solutions
print("\nStep 4: Mixed salt solutions (sea water composition)...")
# Simplified seawater: mainly NaCl with some KCl
print("\nGenerating seawater-like composition...")
# First create NaCl solution
cmd = [
    "mlip-struct-gen", "generate", "salt-water-box",
    "--n-water", "1000",
    "--n-salt", "35",  # ~3.5% salinity
    "--salt", "NaCl",
    "--output", "salt_water_structures/seawater_nacl.data"
]
subprocess.run(cmd, check=True)
print("✓ Created simplified seawater structure")

# Step 5: Different box shapes for interface studies
print("\nStep 5: Salt solutions in different box geometries...")
geometries = [
    ([40, 40, 40], "cubic"),
    ([30, 30, 60], "elongated"),
    ([50, 50, 25], "slab"),
]

for box, shape in geometries:
    print(f"\nGenerating {shape} box...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", "15",
        "--salt", "KCl",
        "--box", str(box[0]), str(box[1]), str(box[2]),
        "--output", f"salt_water_structures/kcl_{shape}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {shape} KCl solution")

# Step 6: Temperature series preparation (different densities)
print("\nStep 6: Preparing structures for temperature studies...")
temps_densities = [
    (273, 1.00, "0C"),
    (298, 0.997, "25C"),
    (323, 0.988, "50C"),
    (348, 0.975, "75C"),
    (373, 0.958, "100C"),
]

for temp, density, label in temps_densities:
    print(f"\nGenerating structure for {temp}K ({label})...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", "10",
        "--salt", "NaCl",
        "--density", str(density),
        "--output", f"salt_water_structures/nacl_T{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created structure for {temp}K")

# Step 7: Different water models with salt
print("\nStep 7: Different water models with NaCl...")
water_models = ["SPC/E", "TIP3P"]

for model in water_models:
    safe_name = model.replace("/", "_")
    print(f"\nGenerating NaCl solution with {model} water...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "500",
        "--n-salt", "10",
        "--salt", "NaCl",
        "--water-model", model,
        "--output", f"salt_water_structures/nacl_{safe_name}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created NaCl with {model}")

# Step 8: Output in different formats
print("\nStep 8: Different output formats...")
formats = [
    "nacl_viz.xyz",
    "nacl_lammps.data",
    "POSCAR_nacl",
]

for filename in formats:
    print(f"\nGenerating {filename}...")
    cmd = [
        "mlip-struct-gen", "generate", "salt-water-box",
        "--n-water", "200",
        "--n-salt", "5",
        "--salt", "LiCl",
        "--output", f"salt_water_structures/{filename}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {filename}")

# Summary
print("\n" + "="*60)
print("SALT-WATER STRUCTURE GENERATION COMPLETE!")
print("="*60)
print("\nGenerated structures in salt_water_structures/:")
print("  - Concentration series: 0M to 5M NaCl")
print("  - Different salts: NaCl, KCl, LiCl, NaF, KF, LiF, NaBr, KBr")
print("  - Large systems: up to 5000 water molecules")
print("  - Different geometries: cubic, elongated, slab")
print("  - Temperature series with adjusted densities")

print("\nPhysical properties to study:")
print("  - Ionic conductivity vs concentration")
print("  - Ion-specific effects (Hofmeister series)")
print("  - Activity coefficients")
print("  - Viscosity changes")
print("  - Dielectric constant modifications")

print("\nAnalysis suggestions:")
print("  - Ion-ion radial distribution functions")
print("  - Ion-water coordination numbers")
print("  - Water structure around ions")
print("  - Ion clustering at high concentrations")
print("  - Diffusion coefficients")

print("\nNext steps:")
print("1. Generate LAMMPS inputs: mlip-lammps-salt-water <structure>.data --salt <type>")
print("2. Run MD simulations for equilibration")
print("3. Calculate transport properties")
print("4. Compare with experimental data")
