# MLIP Structure Generator

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

### Command Line Interface

The package provides standalone CLI tools for each generator:

```bash
# Generate a water box
mlip-water-box --n-water 500 --output water.data

# Create a salt-water electrolyte
mlip-salt-water-box --n-water 500 --salt NaCl --n-salt 10 --output electrolyte.data

# Build a Pt(111) surface
mlip-metal-surface --metal Pt --size 4 4 6 --output pt_surface.vasp

# Create a metal-water interface
mlip-metal-water --metal Pt --size 4 4 4 --n-water 100 --output pt_water.data

# Create an electrochemical interface
mlip-metal-salt-water --metal Au --size 5 5 6 --salt KCl --n-salt 8 --n-water 200 --output interface.lammps
```

### Python API

```python
from mlip_struct_gen.generate_structure.water_box import WaterBoxGenerator, WaterBoxGeneratorParameters

# Create parameters
params = WaterBoxGeneratorParameters(
    n_water=500,
    density=1.0,  # g/cm³
    output_file="water.lammps",
    water_model="SPC/E"
)

# Generate structure
generator = WaterBoxGenerator(params)
generator.run()
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

## Examples

Comprehensive examples for each generator are available in the `examples/` directory:
- `examples/generate_structure/water_box/` - Various water box configurations
- `examples/generate_structure/salt_water_box/` - Electrolyte solutions
- `examples/generate_structure/metal_surface/` - Metal surface generation
- `examples/generate_structure/metal_water/` - Metal-water interfaces
- `examples/generate_structure/metal_salt_water/` - Electrochemical interfaces

## License

MIT License