# MLIP Structure Generator

A Python package for generating water box structures using Packmol for molecular simulations.

## Features

- **Water Box Generation**: Create water boxes with different water models (SPC/E, TIP3P, TIP4P)
- **Flexible Box Sizes**: Support for cubic and rectangular boxes
- **Rich Logging**: Beautiful console output with timing and file origin tracking
- **Comprehensive Validation**: Parameter validation with helpful error messages

## Installation

### Prerequisites

1. **Packmol**: Required for molecular packing
   ```bash
   conda install -c conda-forge packmol
   ```

2. **Python Environment**: Python 3.11 or higher
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Install Package

```bash
# Install in development mode
uv pip install -e .

# Install with development tools
uv pip install -e ".[dev]"
```

## Quick Start

```python
from mlip_struct_gen.generate_structure import (
    WaterBoxGenerator, 
    WaterBoxGeneratorParameters
)

# Create a 20x20x20 Å cubic water box with logging
params = WaterBoxGeneratorParameters(
    box_size=20.0,  # Cubic box
    output_file="water_box",  # Extension added based on format
    water_model="SPC/E",
    output_format="lammps",  # Creates water_box.data
    log=True  # Enable beautiful logging
)

generator = WaterBoxGenerator(params)
output_path = generator.run()
print(f"Water box saved to: {output_path}")
```

## Water Models

| Model | Description | Density (g/cm³) |
|-------|-------------|-----------------|
| SPC/E | Simple Point Charge/Extended | 0.997 |
| TIP3P | Transferable Intermolecular Potential 3-Point | 0.997 |
| TIP4P | Transferable Intermolecular Potential 4-Point | 0.997 |

## Examples

### Basic Cubic Box
```python
params = WaterBoxGeneratorParameters(
    box_size=20.0,  # 20x20x20 Å
    output_file="cubic_water",  # Creates cubic_water.data (LAMMPS default)
)
```

### Rectangular Box with TIP3P
```python
params = WaterBoxGeneratorParameters(
    box_size=(30.0, 25.0, 20.0),  # Rectangular
    output_file="rectangular_water",
    water_model="TIP3P",
    output_format="xyz"  # Creates rectangular_water.xyz
)
```

### Custom Density
```python
params = WaterBoxGeneratorParameters(
    box_size=25.0,
    output_file="dense_water",
    density=1.1,  # Higher than default
    output_format="xyz",  # Creates dense_water.xyz
    log=True
)
```

### Specific Number of Molecules
```python
params = WaterBoxGeneratorParameters(
    box_size=(40.0, 30.0, 30.0),
    output_file="500_molecules",  # Creates 500_molecules.data
    n_molecules=500,  # Exactly 500 molecules
    log=True
)
```

## Logging

The package includes a rich logging system with beautiful console output:

```python
from mlip_struct_gen.utils import MLIPLogger

# Custom logger
logger = MLIPLogger()
params = WaterBoxGeneratorParameters(
    box_size=20.0,
    output_file="water",  # Creates water.data
    log=True,
    logger=logger
)
```

### Sample Log Output
```
[14:32:15] (0.1s) {water_box/generate_water_box.py} INFO: Initializing WaterBoxGenerator
[14:32:15] (0.1s) {water_box/generate_water_box.py} INFO: Water model: TIP3P
[14:32:15] (0.2s) {water_box/generate_water_box.py} STEP: Checking Packmol availability
[14:32:16] (0.3s) {water_box/generate_water_box.py} SUCCESS: Packmol found: packmol
[14:32:16] (2.1s) {water_box/generate_water_box.py} SUCCESS: Water box generation completed
```

## Complete Example

See the comprehensive example in `experiments/water_box_example/`:

```bash
cd experiments/water_box_example
python water_box_example.py
```

This example demonstrates:
- Different water models
- Various box sizes and shapes
- Custom density and molecule counts
- Logging features
- Error handling
- Artifact saving

## API Reference

### WaterBoxGeneratorParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `box_size` | `float` or `tuple` | Required | Box dimensions in Å |
| `output_file` | `str` | Required | Output file name (extension added automatically) |
| `water_model` | `str` | `"SPC/E"` | Water model (SPC/E, TIP3P, TIP4P) |
| `n_molecules` | `int` | `None` | Exact number of molecules |
| `density` | `float` | `None` | Custom density (g/cm³) |
| `tolerance` | `float` | `2.0` | Packmol tolerance (Å) |
| `seed` | `int` | `12345` | Random seed |
| `output_format` | `str` | `"lammps"` | Output format (xyz, lammps) |
| `log` | `bool` | `False` | Enable logging |
| `logger` | `MLIPLogger` | `None` | Custom logger instance |

### WaterBoxGenerator Methods

- `__init__(parameters)`: Initialize generator
- `run(save_artifacts=False)`: Generate water box
- `estimate_box_size(n_molecules, aspect_ratio)`: Estimate required box size

## Development

### Code Quality Tools

```bash
# Linting and formatting
ruff check mlip_struct_gen/
ruff format mlip_struct_gen/
black mlip_struct_gen/

# Type checking
mypy mlip_struct_gen/

# Install pre-commit hooks
pre-commit install
```

### Project Structure

```
mlip_struct_gen/
├── utils/
│   ├── __init__.py
│   └── logger.py           # MLIPLogger with Rich formatting
└── generate_structure/
    ├── __init__.py
    ├── utils.py            # Basic I/O utilities
    ├── templates/
    │   ├── __init__.py
    │   └── water_models.py # Water model definitions
    └── water_box/
        ├── __init__.py
        ├── input_parameters.py    # WaterBoxGeneratorParameters
        ├── validation.py          # Parameter validation
        └── generate_water_box.py  # WaterBoxGenerator
```

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run code quality checks
5. Submit a pull request

## Requirements

- Python 3.11+
- numpy >= 1.24.0
- ase >= 3.22.0
- rich >= 13.0.0
- Packmol (external dependency)

## Troubleshooting

### Packmol Not Found
```bash
conda install -c conda-forge packmol
```

### Unicode Errors
Ensure all files are saved with UTF-8 encoding.

### Import Errors
Make sure the package is installed:
```bash
uv pip install -e .
```