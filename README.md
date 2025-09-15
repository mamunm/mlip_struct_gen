# MLIP Structure Generator

A modular Python framework for generating molecular structures for machine learning interatomic potential (MLIP) training and molecular dynamics simulations.

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

### Package Installation

```bash
# Development installation with pip
pip install -e .

# Or using uv for faster dependency resolution
uv pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Generate a water box
mlip-struct-gen generate water-box --n-molecules 500 --output water.lammps

# Create a salt-water electrolyte
mlip-struct-gen generate salt-water-box --n-water 500 --salt NaCl --n-salt 10 --output electrolyte.data

# Build a Pt(111) surface
mlip-struct-gen generate metal-surface --metal Pt --size 4 4 6 --output pt_surface.vasp

# Create an electrochemical interface
mlip-struct-gen generate metal-salt-water --metal Au --size 5 5 6 --salt KCl --n-salt 8 --n-water 200 --output interface.lammps
```

### Python API

```python
from mlip_struct_gen.generate_structure.water_box import WaterBoxGenerator, WaterBoxGeneratorParameters

# Create parameters
params = WaterBoxGeneratorParameters(
    n_molecules=500,
    density=1.0,  # g/cm³
    output_file="water.lammps",
    water_model="SPC/E"
)

# Generate structure
generator = WaterBoxGenerator(params)
generator.run()
```

## Supported Systems

### Salts
- Simple salts: NaCl, KCl, LiCl, CsCl, NaBr, KBr
- Divalent salts: CaCl₂, MgCl₂

### FCC Metals
- Noble metals: Au, Ag, Pt, Pd
- Transition metals: Cu, Ni, Rh, Ir
- Other: Al, Pb, Ca, Sr, Yb

### Water Models
- SPC/E (default)
- TIP3P
- TIP4P

## Output Formats

| Format | Extension | Features |
|--------|-----------|----------|
| LAMMPS | `.lammps`, `.data` | Full topology, charges, bonds |
| VASP | `.vasp`, `POSCAR` | Crystal structure, sorted elements |
| XYZ | `.xyz` | Simple coordinates |

## Examples

Comprehensive examples for each generator are available in the `examples/` directory:
- `examples/generate_structure/water_box/` - Various water box configurations
- `examples/generate_structure/salt_water_box/` - Electrolyte solutions
- `examples/generate_structure/metal_surface/` - Metal surface generation

## Development

```bash
# Format code
black mlip_struct_gen/

# Lint
ruff check mlip_struct_gen/

# Type checking
mypy mlip_struct_gen/
```

## License

MIT License