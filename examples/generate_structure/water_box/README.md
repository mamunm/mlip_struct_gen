# Water Box Generation Examples

This directory contains 5 comprehensive examples demonstrating different parameter combinations for water box generation using the `mlip_struct_gen` package.

## Overview of Parameter Combinations

The water box generator supports flexible parameter specification. You can provide any 2 of these 3 parameters:
- `box_size`: Dimensions of the simulation box (Å)
- `n_molecules`: Number of water molecules
- `density`: Water density (g/cm³)

## Examples

### Example 1: Box Size Only
**Directory:** `example_1_box_size_only/`
- **Parameters:** Only `box_size` specified
- **Behavior:** Uses water model's default density (e.g., SPC/E: 0.997 g/cm³)
- **Use Case:** Standard water box at model's recommended density
- **Output Formats:** XYZ, LAMMPS (.data), POSCAR

### Example 2: Box Size + Custom Density
**Directory:** `example_2_box_size_density/`
- **Parameters:** Both `box_size` and `density` specified
- **Behavior:** Calculates number of molecules for given density
- **Use Case:** Studying density effects, matching experimental conditions
- **Demonstrates:** Ice density (0.92), high pressure (1.1), supercritical (0.3) g/cm³
- **Output Formats:** XYZ, LAMMPS (.data), POSCAR

### Example 3: Box Size + Number of Molecules
**Directory:** `example_3_box_size_molecules/`
- **Parameters:** Both `box_size` and `n_molecules` specified
- **Behavior:** Packs exact number of molecules, ignores density
- **Use Case:** Specific system sizes for computational constraints
- **Demonstrates:** 100-1000 molecules in various box sizes
- **Output Formats:** XYZ, LAMMPS (.data), POSCAR

### Example 4: Number of Molecules Only
**Directory:** `example_4_molecules_only/`
- **Parameters:** Only `n_molecules` specified
- **Behavior:** Computes cubic box size using default density
- **Use Case:** When you need exact molecule count, box size flexible
- **Demonstrates:** 50-2000 molecules with auto-computed boxes
- **Output Formats:** XYZ, LAMMPS (.data), POSCAR

### Example 5: Number of Molecules + Custom Density
**Directory:** `example_5_molecules_density/`
- **Parameters:** Both `n_molecules` and `density` specified
- **Behavior:** Computes cubic box size for given molecules at specified density
- **Use Case:** Exact molecule count at specific thermodynamic conditions
- **Demonstrates:** Various density conditions from ice to supercritical
- **Output Formats:** XYZ, LAMMPS (.data), POSCAR

## Output Formats

Each example demonstrates different output formats:

- **XYZ** (`.xyz`): Simple coordinate format for visualization
- **LAMMPS** (`.data`): Full atom style with bonds and angles for MD simulations
- **POSCAR** (no extension): VASP format with sorted elements (O before H)

## Water Models

Examples use different water models:
- **SPC/E**: Simple Point Charge/Extended (default)
- **TIP3P**: Transferable Intermolecular Potential 3-Point
- **TIP4P**: Transferable Intermolecular Potential 4-Point

All models have default density of 0.997 g/cm³.

## Running the Examples

1. Ensure `packmol` is installed:
   ```bash
   conda install -c conda-forge packmol
   ```

2. Run any example:
   ```bash
   cd example_1_box_size_only
   python generate_water_box.py
   ```

3. Check the generated output files in the same directory.

## Key Features Demonstrated

- **Flexible parameter specification** (5 different combinations)
- **Multiple output formats** (XYZ, LAMMPS, POSCAR)
- **Different water models** (SPC/E, TIP3P, TIP4P)
- **Various density conditions** (ice, liquid, supercritical)
- **Logging support** for debugging and verification
- **Artifact saving** (Example 4 shows how to save intermediate files)

## Notes

- Box dimensions must be between 5-1000 Å
- Density must be between 0.1-5.0 g/cm³
- Number of molecules must be between 1-1,000,000
- Tolerance parameter controls minimum atom separation (typical: 1.5-2.5 Å)
- Use seeds for reproducible results