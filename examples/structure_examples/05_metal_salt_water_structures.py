#!/usr/bin/env python
"""
Example: Generate metal-salt-water interface structures.

This example demonstrates how to:
1. Create complex three-component systems
2. Control salt concentration at metal interfaces
3. Generate structures for electrochemical studies
4. Create systems for corrosion and battery research
5. Build realistic electrode-electrolyte interfaces
"""

import subprocess
from pathlib import Path

# Create output directory
Path("metal_salt_water_structures").mkdir(exist_ok=True)

# Step 1: Basic metal-salt-water interfaces
print("Step 1: Basic metal-salt-water interfaces with common salts...")
systems = [
    ("Pt", "NaCl", 10, 200, "pt_nacl"),
    ("Au", "KCl", 10, 200, "au_kcl"),
    ("Ag", "LiCl", 10, 200, "ag_licl"),
    ("Cu", "NaF", 10, 200, "cu_naf"),
]

for metal, salt, n_salt, n_water, label in systems:
    print(f"\nGenerating {label} interface...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", metal,
        "--size", "6", "6", "8",
        "--n-water", str(n_water),
        "--n-salt", str(n_salt),
        "--salt", salt,
        "--gap", "3.0",
        "--vacuum", "25",
        "--output", f"metal_salt_water_structures/{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 2: Concentration gradient study
print("\nStep 2: Salt concentration effects at Pt interface...")
concentrations = [
    (0, "pure_water"),
    (5, "dilute"),
    (10, "moderate"),
    (20, "concentrated"),
    (40, "highly_concentrated"),
]

for n_salt, conc_label in concentrations:
    print(f"\nGenerating Pt-NaCl-water ({conc_label})...")
    if n_salt == 0:
        # Use metal-water generator for pure water
        cmd = [
            "mlip-struct-gen", "generate", "metal-water",
            "--metal", "Pt",
            "--size", "7", "7", "10",
            "--n-water", "300",
            "--output", f"metal_salt_water_structures/pt_water_{conc_label}.data"
        ]
    else:
        cmd = [
            "mlip-struct-gen", "generate", "metal-salt-water",
            "--metal", "Pt",
            "--size", "7", "7", "10",
            "--n-water", "300",
            "--n-salt", str(n_salt),
            "--salt", "NaCl",
            "--output", f"metal_salt_water_structures/pt_nacl_{conc_label}.data"
        ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{conc_label}.data")

# Step 3: Different salts at same metal (ion-specific effects)
print("\nStep 3: Different salts at Au interface...")
salts = ["NaCl", "KCl", "LiCl", "NaF", "KF", "NaBr"]

for salt_type in salts:
    print(f"\nGenerating Au-{salt_type}-water interface...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", "Au",
        "--size", "6", "6", "9",
        "--n-water", "250",
        "--n-salt", "15",
        "--salt", salt_type,
        "--output", f"metal_salt_water_structures/au_{salt_type.lower()}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created au_{salt_type.lower()}.data")

# Step 4: Large-scale electrochemical cells
print("\nStep 4: Large-scale electrochemical cells...")
large_cells = [
    ("Pt", [10, 10, 12], 1000, 50, "large_cathode"),
    ("Au", [12, 12, 14], 1500, 75, "large_anode"),
]

for metal, size, n_water, n_salt, label in large_cells:
    print(f"\nGenerating {label} ({metal}, {n_water} water, {n_salt} NaCl)...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", metal,
        "--size", str(size[0]), str(size[1]), str(size[2]),
        "--n-water", str(n_water),
        "--n-salt", str(n_salt),
        "--salt", "NaCl",
        "--gap", "3.5",
        "--vacuum", "40",
        "--output", f"metal_salt_water_structures/{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 5: Corrosion study systems
print("\nStep 5: Corrosion study systems...")
corrosion_systems = [
    ("Cu", "NaCl", 20, "copper_seawater"),
    ("Al", "KCl", 15, "aluminum_chloride"),
    ("Ag", "NaCl", 10, "silver_tarnish"),
]

for metal, salt, n_salt, label in corrosion_systems:
    print(f"\nGenerating {label} corrosion system...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", metal,
        "--size", "8", "8", "10",
        "--n-water", "400",
        "--n-salt", str(n_salt),
        "--salt", salt,
        "--fix-bottom-layers", "2",
        "--output", f"metal_salt_water_structures/{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}.data")

# Step 6: Battery electrode interfaces
print("\nStep 6: Battery electrode interface systems...")

# Lithium-ion battery cathode
print("\nGenerating Li-ion battery cathode interface...")
cmd = [
    "mlip-struct-gen", "generate", "metal-salt-water",
    "--metal", "Pt",  # Representing cathode
    "--size", "8", "8", "10",
    "--n-water", "300",
    "--n-salt", "30",
    "--salt", "LiCl",  # Li+ ions
    "--output", "metal_salt_water_structures/battery_cathode.data"
]
subprocess.run(cmd, check=True)
print("✓ Created battery cathode interface")

# Step 7: Different water layer thicknesses
print("\nStep 7: Varying water layer thickness with salt...")
water_thicknesses = [
    (100, "thin"),
    (300, "medium"),
    (600, "thick"),
    (1000, "bulk"),
]

for n_water, thickness in water_thicknesses:
    n_salt = n_water // 20  # Keep ~1M concentration
    print(f"\nGenerating Ag-NaCl with {thickness} water layer...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", "Ag",
        "--size", "7", "7", "10",
        "--n-water", str(n_water),
        "--n-salt", str(n_salt),
        "--salt", "NaCl",
        "--output", f"metal_salt_water_structures/ag_{thickness}_layer.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created ag_{thickness}_layer.data")

# Step 8: Mixed salt systems (sea water)
print("\nStep 8: Realistic sea water composition at metal surface...")
# Simplified: mainly NaCl with some KCl
# Real seawater: ~0.55M NaCl, ~0.01M KCl, traces of others

print("\nGenerating Pt in seawater...")
cmd = [
    "mlip-struct-gen", "generate", "metal-salt-water",
    "--metal", "Pt",
    "--size", "9", "9", "12",
    "--n-water", "800",
    "--n-salt", "45",  # High salt for seawater
    "--salt", "NaCl",
    "--output", "metal_salt_water_structures/pt_seawater.data"
]
subprocess.run(cmd, check=True)
print("✓ Created Pt in seawater")

# Step 9: Temperature series preparation
print("\nStep 9: Temperature series for phase behavior...")
temps = [
    (273, "0C"),
    (298, "25C"),
    (323, "50C"),
    (348, "75C"),
    (373, "100C"),
]

for temp, label in temps:
    print(f"\nGenerating Au-KCl-water for {temp}K ({label})...")
    # Adjust density slightly with temperature
    density_factor = 1.0 - 0.001 * (temp - 298)

    cmd = [
        "mlip-struct-gen", "generate", "metal-salt-water",
        "--metal", "Au",
        "--size", "5", "5", "8",
        "--n-water", "200",
        "--n-salt", "10",
        "--salt", "KCl",
        "--output", f"metal_salt_water_structures/au_kcl_T{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created structure for {temp}K")

# Step 10: Specific electrochemical configurations
print("\nStep 10: Specific electrochemical configurations...")

# Oxygen reduction reaction interface
print("\nGenerating ORR interface (Pt-dilute NaCl)...")
cmd = [
    "mlip-struct-gen", "generate", "metal-salt-water",
    "--metal", "Pt",
    "--size", "8", "8", "10",
    "--n-water", "400",
    "--n-salt", "5",  # Dilute for ORR
    "--salt", "NaCl",
    "--gap", "2.8",
    "--fix-bottom-layers", "3",
    "--output", "metal_salt_water_structures/pt_orr_interface.data"
]
subprocess.run(cmd, check=True)
print("✓ Created ORR interface")

# Hydrogen evolution reaction interface
print("\nGenerating HER interface (Pt-acidic)...")
cmd = [
    "mlip-struct-gen", "generate", "metal-salt-water",
    "--metal", "Pt",
    "--size", "7", "7", "9",
    "--n-water", "350",
    "--n-salt", "20",  # Higher concentration
    "--salt", "LiCl",  # Small cation
    "--output", "metal_salt_water_structures/pt_her_interface.data"
]
subprocess.run(cmd, check=True)
print("✓ Created HER interface")

# Summary
print("\n" + "="*60)
print("METAL-SALT-WATER INTERFACE STRUCTURE GENERATION COMPLETE!")
print("="*60)
print("\nGenerated structures in metal_salt_water_structures/:")
print("  - Multiple metals: Pt, Au, Ag, Cu, Al")
print("  - Various salts: NaCl, KCl, LiCl, NaF, KF, NaBr")
print("  - Concentration range: 0 to 40 ion pairs")
print("  - System sizes: up to 1500 water molecules")
print("  - Special systems: battery, corrosion, catalysis")

print("\nElectrochemical phenomena:")
print("  - Electric double layer structure")
print("  - Specific ion adsorption")
print("  - Potential of zero charge shifts")
print("  - Electrosorption valency")

print("\nApplications:")
print("  - Batteries and supercapacitors")
print("  - Corrosion and protection")
print("  - Electrocatalysis (ORR, HER, OER)")
print("  - Electroplating and deposition")
print("  - Desalination membranes")

print("\nAnalysis suggestions:")
print("  - Ion density profiles")
print("  - Double layer capacitance")
print("  - Ion-metal interaction energies")
print("  - Water structure perturbation")
print("  - Potential drop across interface")

print("\nSimulation workflow:")
print("1. Generate LAMMPS input: mlip-lammps-metal-salt-water <structure>.data --metal <M> --salt <S>")
print("2. Equilibrate system with NPT/NVT")
print("3. Apply electric field for realistic conditions")
print("4. Calculate electrochemical properties")
print("5. Compare with experimental cyclic voltammetry")
