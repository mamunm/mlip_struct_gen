# MLIP Structure Generator

A Python package for generating molecular structures for machine learning interatomic potential (MLIP) simulations.

## Installation

### Prerequisites

1. **Python 3.11+**: Required for modern type hints and features

2. **Packmol**: Required for water box generation
   ```bash
   conda install -c conda-forge packmol
   ```

### Install Package

```bash
# Install in development mode with uv
uv pip install -e .

# Or with pip
pip install -e .
```

## Command Line Interface

After installation, two CLI commands are available:

```bash
# Main hierarchical command
mlip-struct-gen generate water-box --box-size 30 --output water.xyz

# Direct shortcut for water box
mlip-water-box --box-size 30 --output water.xyz
```

## Water Box Generation

Generate water boxes for molecular dynamics simulations with flexible parameter specification.

### Quick Start

```python
from mlip_struct_gen.generate_structure.water_box import (
    WaterBoxGenerator,
    WaterBoxGeneratorParameters,
)

# Create a 30Å cubic water box
params = WaterBoxGeneratorParameters(
    box_size=30.0,
    output_file="water.xyz",
    water_model="SPC/E",  # or TIP3P, TIP4P
    log=True,  # Enable detailed logging
)

generator = WaterBoxGenerator(params)
output_path = generator.run()
```

### Parameter Combinations

You can specify any 2 of these 3 parameters:
- `box_size`: Box dimensions in Angstroms
- `n_molecules`: Number of water molecules
- `density`: Water density in g/cm³

### Output Formats

- **XYZ**: Simple coordinate format
- **LAMMPS**: Full atom style with bonds and angles
- **POSCAR**: VASP format with sorted elements

### Examples

See the `examples/generate_structure/water_box/` directory for comprehensive examples demonstrating all parameter combinations.

## Development

```bash
# Run linting
ruff check mlip_struct_gen/

# Type checking
mypy mlip_struct_gen/
```

## License

MIT License