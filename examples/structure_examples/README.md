# Structure Generation Examples

This directory contains comprehensive examples for generating various molecular structures using the mlip-struct-gen tools.

## Examples Overview

### 01_water_box_structures.py
Generate pure water box structures with:
- Different system sizes (100 to 5000 molecules)
- Various water models (SPC/E, TIP3P, TIP4P)
- Different densities (vapor, liquid, compressed)
- Multiple output formats (LAMMPS, XYZ, POSCAR)
- Custom box dimensions for interface studies

### 02_salt_water_structures.py
Create salt-water solutions featuring:
- Concentration series (0M to 5M)
- Different salt types (NaCl, KCl, LiCl, NaF, etc.)
- Large-scale systems for bulk properties
- Temperature-dependent structures
- Different box geometries

### 03_metal_surface_structures.py
Generate metal surface structures with:
- Common FCC metals (Au, Pt, Ag, Cu, Ni, Pd, Al)
- Variable surface sizes and thicknesses
- Different vacuum gaps
- Strained surfaces
- Large surfaces for defect studies

### 04_metal_water_interface_structures.py
Create metal-water interfaces featuring:
- Controlled water coverage
- Variable metal-water gap distances
- Different water models at interfaces
- Fixed metal layers for dynamics
- Electrochemical cell configurations

### 05_metal_salt_water_structures.py
Generate complex three-component systems:
- Metal-electrolyte interfaces
- Concentration gradients at surfaces
- Corrosion study systems
- Battery electrode interfaces
- Realistic seawater compositions

## Quick Start

1. **Run an example script:**
```bash
python 01_water_box_structures.py
```

2. **View generated structures:**
```bash
# Using OVITO
ovito water_structures/*.xyz

# Using VMD
vmd water_structures/*.data
```

3. **Generate LAMMPS inputs:**
```bash
# For water
mlip-lammps-water water_structures/water_medium.data

# For salt-water
mlip-lammps-salt-water salt_water_structures/1.0M_NaCl.data --salt NaCl

# For metal-water
mlip-lammps-metal-water metal_water_structures/pt_water.data --metal Pt
```

## Structure Properties

### Water Models
| Model | OH Bond (Å) | HOH Angle (°) | Features |
|-------|-------------|---------------|----------|
| SPC/E | 1.0 | 109.47 | Most common, good properties |
| TIP3P | 0.9572 | 104.52 | Fast, reasonable accuracy |
| TIP4P | 0.9572 | 104.52 | 4-site model, better properties |

### Metal Lattice Constants
| Metal | a₀ (Å) | Crystal Structure |
|-------|--------|-------------------|
| Au | 4.078 | FCC |
| Pt | 3.924 | FCC |
| Ag | 4.085 | FCC |
| Cu | 3.615 | FCC |
| Ni | 3.524 | FCC |
| Pd | 3.891 | FCC |
| Al | 4.046 | FCC |

### Salt Parameters (Joung-Cheatham for SPC/E)
| Ion | σ (Å) | ε (kcal/mol) | Charge |
|-----|-------|--------------|---------|
| Na⁺ | 2.740 | 0.00277 | +1 |
| K⁺ | 3.564 | 0.00349 | +1 |
| Li⁺ | 2.259 | 0.00304 | +1 |
| Cl⁻ | 3.785 | 0.71090 | -1 |
| F⁻ | 3.118 | 0.05329 | -1 |
| Br⁻ | 4.165 | 0.88210 | -1 |

## System Size Guidelines

### Small Systems (Testing)
- 100-500 water molecules
- 4x4x6 metal surface
- 5-10 ion pairs

### Medium Systems (Production)
- 500-2000 water molecules
- 8x8x10 metal surface
- 10-50 ion pairs

### Large Systems (Bulk Properties)
- 2000-10000 water molecules
- 12x12x14 metal surface
- 50-200 ion pairs

## Common Use Cases

### 1. Solvation Studies
```python
# Generate water box
mlip-struct-gen generate water-box --n-water 1000 --output water.data

# Add ions
mlip-struct-gen generate salt-water-box --n-water 1000 --n-salt 20 --salt NaCl --output nacl.data
```

### 2. Interface Studies
```python
# Metal-water interface
mlip-struct-gen generate metal-water --metal Pt --size 8 8 10 --n-water 500 --output interface.data

# With electrolyte
mlip-struct-gen generate metal-salt-water --metal Au --n-water 500 --n-salt 20 --salt KCl --output electrode.data
```

### 3. Concentration Series
```python
for conc in [0, 10, 20, 40]:
    mlip-struct-gen generate salt-water-box --n-water 500 --n-salt {conc} --output nacl_{conc}.data
```

## Visualization Tools

### OVITO
- Best for trajectories and crystalline structures
- Excellent for metal surfaces
- Good particle analysis tools

### VMD
- Best for molecular systems
- Good for water and biomolecules
- Excellent selection language

### ASE GUI
- Simple and fast
- Good for quick checks
- Python integration

## Analysis Suggestions

### Radial Distribution Functions
- g(r) for O-O, O-H, H-H in water
- Ion-water coordination numbers
- Metal-water first layer structure

### Density Profiles
- Water density vs distance from metal
- Ion accumulation at interface
- Double layer structure

### Hydrogen Bonding
- Average H-bonds per water
- H-bond lifetime at interface
- Network topology changes

### Orientation Analysis
- Water dipole orientation
- OH vector alignment
- Surface coverage maps

## Troubleshooting

### Overlapping Atoms
- Increase Packmol tolerance
- Reduce density slightly
- Check box dimensions

### Memory Issues
- Reduce system size
- Use fewer atoms
- Split into smaller systems

### Packmol Failures
- Check if packmol is installed
- Increase tolerance parameter
- Verify input parameters

## References

1. Water Models: Berendsen et al., J. Phys. Chem. 91, 6269 (1987)
2. Ion Parameters: Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008)
3. Metal Surfaces: ASE documentation
4. Packmol: Martinez et al., J. Comput. Chem. 30, 2157 (2009)
