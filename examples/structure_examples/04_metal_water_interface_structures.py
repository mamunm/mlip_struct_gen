#!/usr/bin/env python
"""
Example: Generate metal-water interface structures.

This example demonstrates how to:
1. Create metal-water interfaces with controlled water coverage
2. Vary the gap between metal surface and water
3. Generate interfaces for different metals
4. Control water orientation and packing at interface
5. Create systems for electrochemical studies
"""

import subprocess
from pathlib import Path

# Create output directory
Path("metal_water_structures").mkdir(exist_ok=True)

# Step 1: Basic metal-water interfaces for common metals
print("Step 1: Basic metal-water interfaces...")
metals = ["Pt", "Au", "Ag", "Cu", "Pd"]
n_water = 200  # Standard water amount

for metal in metals:
    print(f"\nGenerating {metal}-water interface...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", metal,
        "--size", "6", "6", "8",
        "--n-water", str(n_water),
        "--gap", "2.5",
        "--vacuum", "20",
        "--output", f"metal_water_structures/{metal.lower()}_water.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {metal.lower()}_water.data")

# Step 2: Water coverage study
print("\nStep 2: Different water coverages on Pt surface...")
water_amounts = [
    (50, "low_coverage"),
    (100, "medium_coverage"),
    (200, "high_coverage"),
    (400, "full_coverage"),
    (600, "thick_layer"),
]

for n_water, coverage in water_amounts:
    print(f"\nGenerating Pt with {coverage} ({n_water} molecules)...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Pt",
        "--size", "8", "8", "10",
        "--n-water", str(n_water),
        "--gap", "2.5",
        "--output", f"metal_water_structures/pt_{coverage}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{coverage}.data")

# Step 3: Gap distance study (important for double layer structure)
print("\nStep 3: Different metal-water gap distances...")
gaps = [
    (2.0, "close"),
    (2.5, "standard"),
    (3.0, "medium"),
    (4.0, "far"),
    (5.0, "separated"),
]

for gap, label in gaps:
    print(f"\nGenerating Au-water with {label} gap ({gap} Å)...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Au",
        "--size", "6", "6", "8",
        "--n-water", "150",
        "--gap", str(gap),
        "--output", f"metal_water_structures/au_gap_{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created au_gap_{label}.data")

# Step 4: Different water models at interface
print("\nStep 4: Different water models at Pt interface...")
water_models = ["SPC/E", "TIP3P", "TIP4P"]

for model in water_models:
    safe_name = model.replace("/", "_")
    print(f"\nGenerating Pt-water with {model} water...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Pt",
        "--size", "5", "5", "8",
        "--n-water", "150",
        "--water-model", model,
        "--output", f"metal_water_structures/pt_{safe_name}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{safe_name}.data")

# Step 5: Large interfaces for bulk water properties
print("\nStep 5: Large interfaces with bulk-like water...")
large_systems = [
    ("Pt", [10, 10, 12], 1000, "large_pt"),
    ("Au", [12, 12, 14], 1500, "large_au"),
    ("Ag", [8, 8, 10], 800, "large_ag"),
]

for metal, size, n_water, label in large_systems:
    print(f"\nGenerating {label} ({size[0]}x{size[1]}x{size[2]}, {n_water} water)...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", metal,
        "--size", str(size[0]), str(size[1]), str(size[2]),
        "--n-water", str(n_water),
        "--gap", "3.0",
        "--vacuum", "30",
        "--output", f"metal_water_structures/{label}_interface.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {label}_interface.data")

# Step 6: Different surface sizes with same water amount
print("\nStep 6: Size effects - same water on different surface areas...")
surface_sizes = [
    ([4, 4], "small"),
    ([6, 6], "medium"),
    ([8, 8], "large"),
    ([10, 10], "xlarge"),
]

for size_xy, label in surface_sizes:
    print(f"\nGenerating Cu {label} surface ({size_xy[0]}x{size_xy[1]}) with 200 water...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Cu",
        "--size", str(size_xy[0]), str(size_xy[1]), "8",
        "--n-water", "200",  # Same water amount
        "--output", f"metal_water_structures/cu_{label}_area.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created cu_{label}_area.data")

# Step 7: Asymmetric surfaces
print("\nStep 7: Asymmetric surfaces for flow studies...")
asymmetric = [
    ([4, 8], "2x1"),
    ([3, 9], "3x1"),
    ([6, 12], "2x1_large"),
]

for size, ratio in asymmetric:
    print(f"\nGenerating Ag surface {ratio} ratio with water...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Ag",
        "--size", str(size[0]), str(size[1]), "8",
        "--n-water", str(size[0] * size[1] * 4),  # Scale water with area
        "--output", f"metal_water_structures/ag_{ratio}_water.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created ag_{ratio}_water.data")

# Step 8: Fixed layer variations (for dynamics studies)
print("\nStep 8: Different fixed layer configurations...")
fixed_configs = [
    (0, "all_mobile"),
    (2, "bottom_2_fixed"),
    (4, "bottom_4_fixed"),
    (6, "mostly_fixed"),
]

for n_fixed, label in fixed_configs:
    print(f"\nGenerating Pt-water with {label}...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Pt",
        "--size", "6", "6", "10",
        "--n-water", "250",
        "--fix-bottom-layers", str(n_fixed),
        "--output", f"metal_water_structures/pt_{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{label}.data")

# Step 9: Custom lattice constants (strained interfaces)
print("\nStep 9: Strained metal surfaces with water...")
strains = [
    (3.884, "compressed"),
    (3.924, "equilibrium"),
    (3.963, "expanded"),
]

for lattice, strain in strains:
    print(f"\nGenerating {strain} Pt-water interface...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-water",
        "--metal", "Pt",
        "--size", "5", "5", "8",
        "--n-water", "150",
        "--lattice-constant", str(lattice),
        "--output", f"metal_water_structures/pt_{strain}_water.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{strain}_water.data")

# Step 10: Electrochemical cell setup
print("\nStep 10: Electrochemical cell configurations...")

# Double layer region
print("\nGenerating electrochemical double layer setup...")
cmd = [
    "mlip-struct-gen", "generate", "metal-water",
    "--metal", "Au",
    "--size", "10", "10", "12",
    "--n-water", "800",
    "--gap", "2.8",  # Optimal for first water layer
    "--vacuum", "40",  # Large vacuum for potential drop
    "--output", "metal_water_structures/au_double_layer.data"
]
subprocess.run(cmd, check=True)
print("✓ Created Au double layer structure")

# Catalytic interface
print("\nGenerating catalytic interface...")
cmd = [
    "mlip-struct-gen", "generate", "metal-water",
    "--metal", "Pt",
    "--size", "8", "8", "10",
    "--n-water", "400",
    "--gap", "2.5",
    "--fix-bottom-layers", "3",
    "--output", "metal_water_structures/pt_catalytic.data"
]
subprocess.run(cmd, check=True)
print("✓ Created Pt catalytic interface")

# Summary
print("\n" + "="*60)
print("METAL-WATER INTERFACE STRUCTURE GENERATION COMPLETE!")
print("="*60)
print("\nGenerated structures in metal_water_structures/:")
print("  - 5 different metals: Pt, Au, Ag, Cu, Pd")
print("  - Water coverage: 50 to 600 molecules")
print("  - Gap distances: 2.0 to 5.0 Å")
print("  - Surface sizes: 4x4 to 12x12 unit cells")
print("  - Different water models: SPC/E, TIP3P, TIP4P")

print("\nInterface phenomena to study:")
print("  - Water orientation at interface")
print("  - First layer structure and hydrogen bonding")
print("  - Electric double layer formation")
print("  - Water dissociation probability")
print("  - Interfacial tension")

print("\nElectrochemical applications:")
print("  - Electrode-electrolyte interfaces")
print("  - Catalytic water splitting")
print("  - Corrosion initiation")
print("  - Underpotential deposition")

print("\nAnalysis tools:")
print("  - Density profiles perpendicular to surface")
print("  - Orientation analysis (dipole, OH vectors)")
print("  - Hydrogen bond networks")
print("  - Residence times at surface")

print("\nNext steps:")
print("1. Equilibrate with MD: mlip-lammps-metal-water <structure>.data --metal <type>")
print("2. Apply electric fields for electrochemistry")
print("3. Add ions for realistic electrolyte")
print("4. Calculate work function changes")
