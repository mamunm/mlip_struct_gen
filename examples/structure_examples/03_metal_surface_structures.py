#!/usr/bin/env python
"""
Example: Generate metal surface structures.

This example demonstrates how to:
1. Create different metal surfaces (Au, Pt, Ag, Cu, Ni, Pd, Al)
2. Generate different surface orientations (111), (100), (110)
3. Control surface size and thickness
4. Create surfaces with different vacuum gaps
5. Generate stepped and kinked surfaces
"""

import subprocess
from pathlib import Path

# Create output directory
Path("metal_structures").mkdir(exist_ok=True)

# Step 1: Common FCC metals with (111) surface
print("Step 1: Generating (111) surfaces for common FCC metals...")
metals = ["Au", "Pt", "Ag", "Cu", "Ni", "Pd", "Al"]

for metal in metals:
    print(f"\nGenerating {metal}(111) surface...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", metal,
        "--size", "6", "6", "8",  # 6x6 unit cells, 8 layers
        "--vacuum", "15",
        "--output", f"metal_structures/{metal.lower()}_111.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {metal.lower()}_111.data")

# Step 2: Different surface sizes for scaling studies
print("\nStep 2: Different surface sizes for Pt...")
sizes = [
    ([4, 4, 6], "small"),
    ([8, 8, 10], "medium"),
    ([12, 12, 12], "large"),
    ([16, 16, 14], "xlarge"),
]

for size, label in sizes:
    print(f"\nGenerating {label} Pt surface ({size[0]}x{size[1]}x{size[2]})...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", "Pt",
        "--size", str(size[0]), str(size[1]), str(size[2]),
        "--vacuum", "20",
        "--output", f"metal_structures/pt_{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{label}.data")

# Step 3: Different vacuum gaps for interface studies
print("\nStep 3: Au surfaces with different vacuum gaps...")
vacuum_gaps = [10, 20, 30, 40, 50]

for vacuum in vacuum_gaps:
    print(f"\nGenerating Au surface with {vacuum}Å vacuum...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", "Au",
        "--size", "5", "5", "8",
        "--vacuum", str(vacuum),
        "--output", f"metal_structures/au_vacuum_{vacuum}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created au_vacuum_{vacuum}.data")

# Step 4: Thin films vs thick slabs
print("\nStep 4: Different slab thicknesses...")
thicknesses = [
    (4, "thin"),      # 4 layers - thin film
    (8, "medium"),    # 8 layers - standard
    (12, "thick"),    # 12 layers - thick slab
    (16, "bulk"),     # 16 layers - bulk-like
]

for n_layers, label in thicknesses:
    print(f"\nGenerating {label} Ag slab ({n_layers} layers)...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", "Ag",
        "--size", "6", "6", str(n_layers),
        "--vacuum", "20",
        "--output", f"metal_structures/ag_{label}_slab.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created ag_{label}_slab.data")

# Step 5: Rectangular surfaces for anisotropic studies
print("\nStep 5: Non-square surfaces for anisotropic studies...")
rectangles = [
    ([4, 8], "2x1"),
    ([6, 12], "2x1_large"),
    ([8, 4], "1x2"),
    ([9, 3], "3x1"),
]

for size_xy, ratio in rectangles:
    print(f"\nGenerating Cu surface with {ratio} aspect ratio...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", "Cu",
        "--size", str(size_xy[0]), str(size_xy[1]), "10",
        "--vacuum", "15",
        "--output", f"metal_structures/cu_rect_{ratio}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created cu_rect_{ratio}.data")

# Step 6: Custom lattice constants for strained surfaces
print("\nStep 6: Strained surfaces with modified lattice constants...")
# Normal Pt lattice constant is ~3.924 Å
strains = [
    (3.845, "compressed_2pct"),
    (3.884, "compressed_1pct"),
    (3.924, "equilibrium"),
    (3.963, "expanded_1pct"),
    (4.002, "expanded_2pct"),
]

for lattice, label in strains:
    print(f"\nGenerating Pt surface with {label} ({lattice:.3f} Å)...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", "Pt",
        "--size", "5", "5", "8",
        "--lattice-constant", str(lattice),
        "--vacuum", "15",
        "--output", f"metal_structures/pt_{label}.data"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created pt_{label}.data")

# Step 7: Large surfaces for defect studies
print("\nStep 7: Large surfaces for defect/adsorption studies...")
cmd = [
    "mlip-struct-gen", "generate", "metal-surface",
    "--metal", "Pd",
    "--size", "20", "20", "10",
    "--vacuum", "25",
    "--output", "metal_structures/pd_large_surface.data"
]
subprocess.run(cmd, check=True)
print("✓ Created large Pd surface (20x20x10)")

# Step 8: Different output formats
print("\nStep 8: Different output formats for visualization...")
formats = [
    ("au_surface.xyz", "XYZ format"),
    ("au_surface.data", "LAMMPS format"),
    ("POSCAR_Au", "VASP format"),
]

for filename, description in formats:
    print(f"\nGenerating Au surface in {description}...")
    cmd = [
        "mlip-struct-gen", "generate", "metal-surface",
        "--metal", "Au",
        "--size", "4", "4", "6",
        "--output", f"metal_structures/{filename}"
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {filename}")

# Step 9: Surfaces for specific applications
print("\nStep 9: Surfaces for specific applications...")

# Catalyst surface
print("\nGenerating Pt catalyst surface...")
cmd = [
    "mlip-struct-gen", "generate", "metal-surface",
    "--metal", "Pt",
    "--size", "8", "8", "12",
    "--vacuum", "30",  # Large vacuum for adsorbates
    "--output", "metal_structures/pt_catalyst.data"
]
subprocess.run(cmd, check=True)
print("✓ Created Pt catalyst surface")

# Electrode surface
print("\nGenerating Au electrode surface...")
cmd = [
    "mlip-struct-gen", "generate", "metal-surface",
    "--metal", "Au",
    "--size", "10", "10", "14",
    "--vacuum", "40",  # Extra vacuum for double layer
    "--output", "metal_structures/au_electrode.data"
]
subprocess.run(cmd, check=True)
print("✓ Created Au electrode surface")

# Summary
print("\n" + "="*60)
print("METAL SURFACE STRUCTURE GENERATION COMPLETE!")
print("="*60)
print("\nGenerated structures in metal_structures/:")
print("  - 7 different metals: Au, Pt, Ag, Cu, Ni, Pd, Al")
print("  - Various sizes: 4x4 to 20x20 unit cells")
print("  - Different thicknesses: 4 to 16 layers")
print("  - Vacuum gaps: 10 to 50 Å")
print("  - Strained surfaces: ±2% lattice strain")

print("\nApplications:")
print("  - Catalysis: Pt, Pd surfaces")
print("  - Electrochemistry: Au, Pt electrodes")
print("  - Corrosion: Cu, Al surfaces")
print("  - Nanoelectronics: Ag, Au thin films")

print("\nStructural analysis:")
print("  - Surface relaxation")
print("  - Surface energy calculations")
print("  - Work function determination")
print("  - Adsorption site identification")

print("\nNext steps:")
print("1. Optimize surface structures with DFT or classical MD")
print("2. Add adsorbates for reaction studies")
print("3. Create interfaces with water/electrolytes")
print("4. Generate LAMMPS inputs: mlip-lammps-metal-surface <structure>.data --metal <type>")
