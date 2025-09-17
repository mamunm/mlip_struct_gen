# MLIP Structure Generator

![MLIP-STRUCT-GEN](./assets/mlip-animated.svg)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PACKMOL](https://img.shields.io/badge/requires-PACKMOL-green.svg)](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml)
[![ASE](https://img.shields.io/badge/requires-ASE-green.svg)](https://wiki.fysik.dtu.dk/ase/)

A modular Python framework for generating molecular structures for machine learning interatomic potential (MLIP) training and molecular dynamics simulations.

## Author

Osman Mamun - mamun.che06@gmail.com

## Features

### Available Structure Generators

- **Water Box**: Configurable water boxes with SPC/E, TIP3P, TIP4P models
- **Salt-Water Box**: Aqueous electrolyte solutions with various salts
- **Metal Surface**: FCC(111) metal surfaces for catalysis and electrochemistry
- **Metal-Water Interface**: Metal-water interfaces for surface science
- **Metal-Salt-Water Interface**: Electrochemical interfaces with electrolytes

### Key Capabilities

- Multiple output formats: LAMMPS, VASP/POSCAR, XYZ
- Flexible parameter specification with automatic constraint solving
- Reproducible structure generation with seed control
- Integrated PACKMOL for optimal molecular packing
- Support for fixed layers in surface simulations

## Installation

### Prerequisites

1. **Python 3.11+** - Required for modern type hints
2. **PACKMOL** - Required for molecular packing
   ```bash
   conda install -c conda-forge packmol
   ```
3. **ASE** - Required for metal surface generation
   ```bash
   pip install ase
   ```

### Package Installation

```bash
# Development installation with pip
pip install -e .

# Or using uv for faster dependency resolution
uv pip install -e .
```

## Quick Start

### Structure Generation

The package provides a unified CLI interface through `mlip-struct-gen`:

```bash
# Generate a water box with 500 molecules
mlip-struct-gen generate water-box --n-water 500 --density 0.997 --output water.data

# Create a 1M NaCl solution
mlip-struct-gen generate salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output nacl_1M.data

# Build a Pt(111) surface (6x6x8 unit cells)
mlip-struct-gen generate metal-surface --metal Pt --size 6 6 8 --vacuum 20 --output pt_111.data

# Create a metal-water interface
mlip-struct-gen generate metal-water --metal Au --size 8 8 10 --n-water 400 --gap 2.5 --output au_water.data

# Build an electrochemical interface
mlip-struct-gen generate metal-salt-water --metal Pt --size 7 7 10 --n-water 500 --n-salt 25 --salt KCl --output electrode.data
```

### LAMMPS Input Generation

Generate optimized LAMMPS input files for molecular dynamics simulations:

```bash
# Water box simulation with NPT ensemble
mlip-lammps-water water.data --water-model SPC/E --ensemble NPT --temperature 300 --pressure 1.0 --output in.water

# Salt solution with proper ion parameters
mlip-lammps-salt-water nacl_1M.data --salt NaCl --temperature 298.15 --output in.nacl

# Metal surface with fixed bottom layers
mlip-lammps-metal-surface pt_111.data --metal Pt --temperature 330 --fix-layers 2 --output in.pt_surface

# Metal-water interface with water temperature monitoring
mlip-lammps-metal-water au_water.data --metal Au --temperature 300 --fix-layers 3 --output in.au_water

# Electrochemical interface with all interactions
mlip-lammps-metal-salt-water electrode.data --metal Pt --salt KCl --temperature 330 --output in.electrode
```

### Complete Workflow Example

```bash
# Step 1: Generate structure
mlip-struct-gen generate metal-water --metal Pt --size 6 6 8 --n-water 300 --output pt_water.data

# Step 2: Create LAMMPS input
mlip-lammps-metal-water pt_water.data --metal Pt --temperature 330 --equilibration-time 100 --production-time 500 --output in.pt_water

# Step 3: Run simulation (requires LAMMPS)
lmp -i in.pt_water

# Step 4: Output files ready for snapshot extraction
# - trajectory.lammpstrj (positions and forces)
# - final.data (equilibrated structure)
# - thermo output with T_water monitoring
```

### Python API

```python
from mlip_struct_gen.generate_structure.metal_water import MetalWaterGenerator, MetalWaterParameters
from mlip_struct_gen.generate_lammps_input.metal_water import MetalWaterLAMMPSGenerator, MetalWaterLAMMPSParameters

# Generate structure
struct_params = MetalWaterParameters(
    metal="Pt",
    size=(6, 6, 8),
    n_water=300,
    gap_above_metal=2.5,
    output_file="pt_water.data"
)
struct_gen = MetalWaterGenerator(struct_params)
struct_gen.run()

# Generate LAMMPS input
lammps_params = MetalWaterLAMMPSParameters(
    data_file="pt_water.data",
    metal_type="Pt",
    temperature=330.0,
    ensemble="NVT",
    output_file="in.pt_water"
)
lammps_gen = MetalWaterLAMMPSGenerator(lammps_params)
lammps_gen.generate()
```

## Parameter Specifications

### Water Box Generation

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--n-water` | Number of water molecules | `500` |
| `--box-size` | Box dimensions in Å (single value for cubic) | `30` or `30 30 40` |
| `--density` | Water density in g/cm³ | `1.0` |
| `--water-model` | Water model (SPC/E, TIP3P, TIP4P) | `SPC/E` |

Note: Specify any 2 of 3 parameters (n-water, box-size, density) and the third will be calculated.

### Salt-Water Box Generation

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--n-water` | Number of water molecules | `500` |
| `--n-salt` | Number of salt formula units | `10` |
| `--salt` | Salt type | `NaCl` |
| `--box-size` | Box dimensions in Å | `40` |
| `--density` | Solution density in g/cm³ | `1.1` |

### Metal Surface Generation

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--metal` | Metal element symbol | `Pt` |
| `--size` | Surface size (nx ny nz) | `4 4 6` |
| `--vacuum` | Vacuum space above surface in Å | `15` |
| `--fix-bottom-layers` | Number of fixed bottom layers | `2` |

### Metal-Water Interface

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--metal` | Metal element symbol | `Au` |
| `--size` | Metal surface size (nx ny nz) | `5 5 6` |
| `--n-water` | Number of water molecules | `200` |
| `--density` | Water density in g/cm³ | `1.0` |
| `--gap` | Gap between metal and water in Å | `3.5` |
| `--vacuum` | Vacuum above water in Å | `10` |

### Metal-Salt-Water Interface

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--metal` | Metal element symbol | `Cu` |
| `--size` | Metal surface size (nx ny nz) | `6 6 8` |
| `--n-water` | Number of water molecules | `300` |
| `--n-salt` | Number of salt formula units | `15` |
| `--salt` | Salt type | `KCl` |
| `--gap` | Gap between metal and solution in Å | `2.0` |

## Supported Systems

### Salts
- Simple salts: NaCl, KCl, LiCl, CsCl, NaBr, KBr
- Divalent salts: CaCl₂, MgCl₂

### FCC Metals
- Noble metals: Au, Ag, Pt, Pd
- Transition metals: Cu, Ni, Rh, Ir
- Other: Al, Pb, Ca, Sr, Yb

### Water Models
- SPC/E (default) - Extended Simple Point Charge
- TIP3P - Three-site Transferable Intermolecular Potential
- TIP4P - Four-site Transferable Intermolecular Potential

## Output Formats

| Format | Extension | Features |
|--------|-----------|----------|
| LAMMPS | `.lammps`, `.data` | Full topology with bonds, angles, and charges |
| VASP | `.vasp`, `POSCAR` | Crystal structure with sorted elements |
| XYZ | `.xyz` | Simple coordinate format |

The output format is automatically detected from the file extension, or can be explicitly specified with the `--output-format` flag.

## Advanced Features

### Reproducible Generation
All generators support a `--seed` parameter for reproducible random number generation:
```bash
mlip-water-box --n-water 500 --seed 42 --output water.data
```

### Dry Run Mode
Test parameters without generating files:
```bash
mlip-water-box --n-water 500 --output water.data --dry-run
```

### Artifact Saving
Save intermediate files for debugging:
```bash
mlip-salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output solution.data --save-artifacts
```

### Custom Lattice Constants
Override default lattice constants for metals:
```bash
mlip-metal-surface --metal Cu --size 4 4 6 --lattice-constant 3.615 --output cu.vasp
```

### Save Input Parameters
Save input parameters to JSON for reproducibility and documentation:
```bash
mlip-water-box --n-water 500 --output water.data --save-input
# Creates input_params.json with all parameters used

mlip-metal-salt-water --metal Au --size 5 5 6 --salt KCl --n-salt 8 --n-water 200 --output interface.lammps --save-input
# Saves complete configuration for complex interfaces
```
This feature is available for all generators and helps with:
- Reproducing exact structure generation runs
- Documenting simulation setups

## License

MIT License
