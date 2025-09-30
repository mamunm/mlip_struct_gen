# MLIP Structure Generator

![MLIP-STRUCT-GEN](./assets/mlip-animated.svg)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PACKMOL](https://img.shields.io/badge/requires-PACKMOL-green.svg)](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml)
[![ASE](https://img.shields.io/badge/requires-ASE-green.svg)](https://wiki.fysik.dtu.dk/ase/)

A comprehensive Python framework for generating molecular structures, preparing LAMMPS simulations, and processing quantum chemistry data for machine learning interatomic potential (MLIP) training.

## Author

Osman Mamun - mamun.che06@gmail.com

## Overview

`mlip-struct-gen` provides an integrated workflow for MLIP training data generation:

- **Structure Generation**: Create atomistic structures for water, electrolytes, metal surfaces, and interfaces
- **LAMMPS Input Generation**: Prepare optimized molecular dynamics simulation inputs
- **Wannier Centroid Analysis**: Compute and analyze electronic wannier centers for dipole moment calculations
- **Data Conversion**: Convert between formats and prepare training datasets for DeePMD-kit and similar frameworks

## Features

### Structure Generators

#### Water Box
Generate pure water systems with configurable parameters and multiple water models.

**Supported Water Models:**
- **SPC/E** (default) - Extended Simple Point Charge model
- **TIP3P** - 3-site Transferable Intermolecular Potential
- **TIP4P** - 4-site model with virtual site
- **SPC/Fw** - Flexible SPC model

```bash
# Generate water box with 500 molecules
mlip-struct-gen generate water-box --n-water 500 --density 0.997 --output water.data

# Using direct command
mlip-water-box --n-water 500 --density 0.997 --output water.data

# Specify water model
mlip-water-box --n-water 500 --water-model TIP3P --output water.data
```

#### Salt-Water Box
Create aqueous electrolyte solutions with various salts and concentrations.

**Supported Salts:**
- Monovalent: NaCl, KCl, LiCl, CsCl, NaBr, KBr
- Divalent: CaCl₂, MgCl₂

```bash
# Generate 1M NaCl solution
mlip-struct-gen generate salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output nacl.data

# Using direct command
mlip-salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output nacl.data

# Different salt types
mlip-salt-water-box --n-water 500 --n-salt 5 --salt CaCl2 --output cacl2.data
```

#### Metal Surface
Generate FCC(111) metal surfaces for catalysis and electrochemistry studies.

**Supported FCC Metals:**
Al, Au, Ag, Cu, Ni, Pd, Pt, Pb, Rh, Ir, Ca, Sr, Yb

```bash
# Generate Pt(111) surface (6x6x8 unit cells)
mlip-struct-gen generate metal-surface --metal Pt --size 6 6 8 --vacuum 20 --output pt_111.data

# Using direct command
mlip-metal-surface --metal Pt --size 6 6 8 --vacuum 20 --output pt_111.data

# With fixed bottom layers
mlip-metal-surface --metal Au --size 8 8 10 --fix-bottom-layers 2 --output au_surface.data
```

#### Metal-Water Interface
Create metal-water interfaces for surface science and electrochemistry.

```bash
# Generate Au-water interface
mlip-struct-gen generate metal-water --metal Au --size 8 8 10 --n-water 400 --gap 2.5 --output au_water.data

# Using direct command
mlip-metal-water --metal Au --size 8 8 10 --n-water 400 --gap 2.5 --output au_water.data

# Control water density and gap
mlip-metal-water --metal Pt --size 6 6 8 --n-water 300 --density 1.0 --gap 3.0 --output pt_water.data
```

#### Metal-Salt-Water Interface
Generate three-component electrochemical interfaces.

```bash
# Generate Pt electrode with KCl electrolyte
mlip-struct-gen generate metal-salt-water --metal Pt --size 7 7 10 --n-water 500 --n-salt 25 --salt KCl --output electrode.data

# Using direct command
mlip-metal-salt-water --metal Pt --size 7 7 10 --n-water 500 --n-salt 25 --salt KCl --output electrode.data
```

### LAMMPS Input Generators

Generate production-ready LAMMPS input files optimized for MLIP training data collection.

**Key Features:**
- Multiple ensembles: NVT, NPT, NVE
- Multi-temperature sampling
- Fixed layer support for surfaces
- Automatic force/stress output for training
- Water temperature monitoring

#### Water Box LAMMPS Input

```bash
# NPT ensemble at 300K
mlip-lammps-water water.data --water-model SPC/E --ensemble NPT --temperature 300 --pressure 1.0 --output in.water

# NVT with multiple temperatures
mlip-lammps-water water.data --ensemble NVT --temperature 300 330 350 --output in.water_multiT

# Custom simulation time
mlip-lammps-water water.data --equilibration-time 100 --production-time 500 --output in.water
```

#### Salt-Water LAMMPS Input

```bash
# NaCl solution simulation
mlip-lammps-salt-water nacl.data --salt NaCl --temperature 298.15 --ensemble NPT --output in.nacl

# Multiple temperatures for enhanced sampling
mlip-lammps-salt-water nacl.data --salt NaCl --temperature 300 330 360 --output in.nacl_multiT
```

#### Metal Surface LAMMPS Input

```bash
# Pt surface with fixed bottom layers
mlip-lammps-metal-surface pt_111.data --metal Pt --temperature 330 --fix-layers 2 --output in.pt_surface

# NPT for thermal expansion studies
mlip-lammps-metal-surface pt_111.data --metal Pt --ensemble NPT --temperature 300 --output in.pt_npt
```

#### Metal-Water Interface LAMMPS Input

```bash
# Au-water interface with water temperature monitoring
mlip-lammps-metal-water au_water.data --metal Au --temperature 300 --fix-layers 3 --output in.au_water

# Multi-temperature sampling
mlip-lammps-metal-water pt_water.data --metal Pt --temperature 300 330 360 --output in.pt_water_multiT
```

#### Metal-Salt-Water Interface LAMMPS Input

```bash
# Full electrochemical interface
mlip-lammps-metal-salt-water electrode.data --metal Pt --salt KCl --temperature 330 --output in.electrode

# With custom equilibration and production times
mlip-lammps-metal-salt-water electrode.data --metal Pt --salt KCl --temperature 330 \
  --equilibration-time 200 --production-time 1000 --output in.electrode
```

### Wannier Centroid Analysis

Analyze electronic wannier centers from Wannier90 calculations to compute atomic dipole moments.

#### Compute Wannier Centroids

Parse wannier90_centres.xyz and POSCAR files to compute wannier centroids for each atom.

```bash
# Compute centroids for a single snapshot
mlip-struct-gen compute-wc /path/to/snapshot_folder

# Using direct command
mlip-compute-wc /path/to/snapshot_folder --verbose
```

**Required files in folder:**
- `POSCAR` - Atomic structure
- `wannier90_centres.xyz` - Wannier centers from Wannier90

**Output:** `wc_out.txt` and `wc_out.npy` containing centroid positions per atom

#### Plot Wannier Distributions

Generate statistical analysis and distribution plots of wannier centroid norms.

```bash
# Plot distributions from multiple snapshots
mlip-struct-gen wc-plot /path/to/root_directory --output wannier_report.png

# Using direct command
mlip-wc-plot /path/to/root_directory --output wannier_report.png --atom-types O Na Cl
```

**Features:**
- Histogram distributions by atom type
- Statistical summaries (mean, std, min, max)
- Dipole moment calculations
- Multi-snapshot analysis

#### Convert to DPData Format

Convert wannier centroid data to DeePMD-kit compatible format with atomic dipoles.

```bash
# Convert using file list
mlip-struct-gen wc-dpdata --input-file-loc directories.txt --out-dir DATA/water

# Using direct command
mlip-wc-dpdata --input-file-loc directories.txt --out-dir DATA/water

# With type map for consistent element ordering
mlip-wc-dpdata --input-file-loc directories.txt --out-dir DATA/ --type-map Pt O H Na Cl
```

**Input file format (directories.txt):**
```
/path/to/snapshot1/
/path/to/snapshot2/
/path/to/snapshot3/
```

**Output structure:**
```
DATA/
├── O2H4/
│   └── set.000/
│       ├── atomic_dipole.npy  # Wannier centroids as dipoles
│       ├── coord.npy
│       ├── box.npy
│       ├── type.npy
│       └── type_map.raw
```

### Data Conversion Tools

#### VASP to DPData Conversion

Convert VASP OUTCAR files to DeePMD-kit format using dpdata MultiSystems.

```bash
# Convert all OUTCARs with element filtering
mlip-struct-gen dpdata --input-dir . --type-map Cu O H Na Cl --output-dir DATA/

# Using direct command
mlip-dpdata --input-dir . --type-map Cu O H Na Cl --output-dir DATA/

# Process only water systems
mlip-dpdata --input-dir . --type-map O H --output-dir DATA/water_only

# Non-recursive search
mlip-dpdata --input-dir ./snapshots --type-map Cu O H --output-dir DATA/ --no-recursive

# Dry run to preview
mlip-dpdata --input-dir . --type-map O H --output-dir DATA/ --dry-run
```

**Features:**
- Automatic composition grouping (e.g., 32Water, Cu48_32Water, 32Water_4NaCl)
- Element filtering (skips systems with unlisted elements)
- Type ordering preservation
- Recursive directory search

**Output structure:**
```
DATA/
├── 32Water/          # Pure water systems
├── Cu48_32Water/     # Metal-water systems
├── 32Water_4NaCl/    # Salt-water systems
└── Cu32_32Water_4NaCl/  # Metal-salt-water systems
```

#### Trajectory to POSCAR Conversion

Extract snapshots from LAMMPS trajectories and convert to VASP POSCAR format.

```bash
# Convert entire trajectory
mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --output-dir snapshots

# Specify frame range
mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --start-frame 100 --end-frame 200 --output-dir snapshots

# Sample every 10th frame
mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --stride 10 --output-dir snapshots

# Sort atoms by element (required for VASP)
mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --sort-elements --output-dir snapshots
```

**Output structure:**
```
snapshots/
  snapshot_001/POSCAR
  snapshot_002/POSCAR
  snapshot_003/POSCAR
  ...
```

## CLI Usage Patterns

The package supports two command-line interface patterns:

### Unified CLI
All commands accessible through the main `mlip-struct-gen` entry point:

```bash
mlip-struct-gen generate water-box --n-water 500 --output water.data
mlip-struct-gen generate salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output nacl.data
mlip-struct-gen compute-wc /path/to/folder
mlip-struct-gen dpdata --input-dir . --type-map O H --output-dir DATA/
mlip-struct-gen convert trajectory-to-poscar trajectory.lammpstrj --output-dir snapshots
```

### Direct Commands
Individual commands for specific tasks:

```bash
# Structure generation
mlip-water-box --n-water 500 --output water.data
mlip-salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output nacl.data
mlip-metal-surface --metal Pt --size 6 6 8 --output pt.data
mlip-metal-water --metal Au --size 8 8 10 --n-water 400 --output au_water.data
mlip-metal-salt-water --metal Pt --size 7 7 10 --n-water 500 --n-salt 25 --salt KCl --output electrode.data

# LAMMPS input generation
mlip-lammps-water water.data --temperature 300 --output in.water
mlip-lammps-salt-water nacl.data --salt NaCl --temperature 300 --output in.nacl
mlip-lammps-metal-surface pt.data --metal Pt --temperature 330 --output in.pt
mlip-lammps-metal-water au_water.data --metal Au --temperature 300 --output in.au_water
mlip-lammps-metal-salt-water electrode.data --metal Pt --salt KCl --temperature 330 --output in.electrode

# Analysis and conversion
mlip-compute-wc /path/to/folder
mlip-wc-plot /path/to/root_dir --output report.png
mlip-wc-dpdata --input-file-loc dirs.txt --out-dir DATA/
mlip-dpdata --input-dir . --type-map O H --output-dir DATA/
mlip-trajectory-to-poscar trajectory.lammpstrj --output-dir snapshots
```

Both patterns are equivalent - choose based on your preference.

## Output Formats

The package supports multiple output formats, automatically detected from file extensions:

| Format | Extensions | Features | Use Case |
|--------|-----------|----------|----------|
| **LAMMPS** | `.data`, `.lammps` | Full topology with bonds, angles, charges | MD simulations |
| **LAMMPS/DeePMD** | `.lammps/dpmd` | DeePMD-compatible LAMMPS format | DeePMD-kit training |
| **LAMMPS Trajectory** | `.lammpstrj` | Trajectory format | Snapshots for QM |
| **VASP** | `.vasp`, `POSCAR` | VASP POSCAR format with sorted elements | DFT calculations |
| **XYZ** | `.xyz` | Simple coordinate format | Visualization |

Specify output format explicitly with `--output-format` flag if needed:

```bash
mlip-water-box --n-water 500 --output water.xyz --output-format xyz
mlip-water-box --n-water 500 --output water.data --output-format lammps
```

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

## Parameter Reference

### Water Box Generation

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--n-water` | Number of water molecules | - | `500` |
| `--box-size` | Box dimensions in Å (single or three values) | - | `30` or `30 30 40` |
| `--density` | Water density in g/cm³ | `0.997` | `1.0` |
| `--water-model` | Water model | `SPC/E` | `SPC/E`, `TIP3P`, `TIP4P`, `SPC/Fw` |
| `--output` | Output file path | Required | `water.data` |
| `--output-format` | Output format | Auto-detect | `lammps`, `xyz`, `vasp` |
| `--seed` | Random seed | `None` | `42` |

**Note:** Specify any 2 of 3 parameters (n-water, box-size, density) and the third will be calculated.

### Salt-Water Box Generation

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--n-water` | Number of water molecules | - | `500` |
| `--n-salt` | Number of salt formula units | - | `10` |
| `--salt` | Salt type | Required | `NaCl`, `KCl`, `CaCl2` |
| `--box-size` | Box dimensions in Å | Auto-calc | `40` |
| `--density` | Solution density in g/cm³ | Auto-calc | `1.1` |
| `--output` | Output file path | Required | `nacl.data` |

### Metal Surface Generation

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--metal` | Metal element symbol | Required | `Pt`, `Au`, `Cu` |
| `--size` | Surface size (nx ny nz) | Required | `6 6 8` |
| `--vacuum` | Vacuum space above surface in Å | `15.0` | `20` |
| `--fix-bottom-layers` | Number of fixed bottom layers | `0` | `2` |
| `--lattice-constant` | Custom lattice constant in Å | Auto | `3.924` |
| `--output` | Output file path | Required | `pt_111.data` |

### Metal-Water Interface Generation

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--metal` | Metal element symbol | Required | `Au`, `Pt` |
| `--size` | Metal surface size (nx ny nz) | Required | `8 8 10` |
| `--n-water` | Number of water molecules | Required | `400` |
| `--density` | Water density in g/cm³ | `1.0` | `1.0` |
| `--gap` | Gap between metal and water in Å | `2.5` | `3.0` |
| `--vacuum` | Vacuum above water in Å | `10.0` | `15` |
| `--output` | Output file path | Required | `au_water.data` |

### Metal-Salt-Water Interface Generation

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--metal` | Metal element symbol | Required | `Pt`, `Cu` |
| `--size` | Metal surface size (nx ny nz) | Required | `7 7 10` |
| `--n-water` | Number of water molecules | Required | `500` |
| `--n-salt` | Number of salt formula units | Required | `25` |
| `--salt` | Salt type | Required | `KCl`, `NaCl` |
| `--gap` | Gap between metal and solution in Å | `2.5` | `2.0` |
| `--vacuum` | Vacuum above solution in Å | `10.0` | `15` |
| `--output` | Output file path | Required | `electrode.data` |

### LAMMPS Input Generation (All Systems)

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--ensemble` | Ensemble type | `NVT` | `NVT`, `NPT`, `NVE` |
| `--temperature` | Temperature(s) in K (space-separated for multi-T) | `330.0` | `300` or `300 330 360` |
| `--pressure` | Pressure in bar (NPT only) | `1.0` | `1.0` |
| `--equilibration-time` | Equilibration time in ps | `100.0` | `200` |
| `--production-time` | Production time in ps | `500.0` | `1000` |
| `--timestep` | Timestep in fs | `1.0` | `0.5` |
| `--dump-frequency` | Snapshot frequency in ps | `1.0` | `2.0` |
| `--fix-layers` | Fixed bottom layers (surfaces only) | `0` | `2` |
| `--seed` | Random seed | `12345` | `42` |
| `--output` | Output file path | Auto | `in.water` |

## Supported Systems

### Water Models
| Model | OH Bond (Å) | HOH Angle (°) | Description |
|-------|-------------|---------------|-------------|
| **SPC/E** | 1.000 | 109.47 | Extended Simple Point Charge (most common) |
| **TIP3P** | 0.9572 | 104.52 | 3-site model, fast and reasonable |
| **TIP4P** | 0.9572 | 104.52 | 4-site with virtual site, better properties |
| **SPC/Fw** | 1.012 | 113.24 | Flexible SPC model |

### Salts
| Salt | Formula | Stoichiometry | Typical Concentration |
|------|---------|---------------|----------------------|
| **NaCl** | Sodium Chloride | 1:1 | 0.15 M |
| **KCl** | Potassium Chloride | 1:1 | 0.15 M |
| **LiCl** | Lithium Chloride | 1:1 | 0.15 M |
| **CsCl** | Cesium Chloride | 1:1 | 0.15 M |
| **NaBr** | Sodium Bromide | 1:1 | 0.15 M |
| **KBr** | Potassium Bromide | 1:1 | 0.15 M |
| **CaCl₂** | Calcium Chloride | 1:2 | 0.10 M |
| **MgCl₂** | Magnesium Chloride | 1:2 | 0.10 M |

### FCC Metals
| Metal | Lattice Constant (Å) | Applications |
|-------|---------------------|--------------|
| **Al** | 4.050 | Light metals, alloys |
| **Au** | 4.078 | Catalysis, electrochemistry |
| **Ag** | 4.085 | Plasmonic, catalysis |
| **Cu** | 3.615 | Catalysis, corrosion |
| **Ni** | 3.524 | Catalysis, magnetic |
| **Pd** | 3.891 | Hydrogen storage, catalysis |
| **Pt** | 3.924 | Electrocatalysis, fuel cells |
| **Pb** | 4.950 | Heavy metals |
| **Rh** | 3.803 | Catalysis |
| **Ir** | 3.839 | Catalysis |
| **Ca** | 5.588 | Alkaline earth |
| **Sr** | 6.085 | Alkaline earth |
| **Yb** | 5.449 | Rare earth |

## Python API

### Structure Generation Example

```python
from mlip_struct_gen.generate_structure.metal_water import (
    MetalWaterGenerator,
    MetalWaterParameters
)

# Generate metal-water interface structure
params = MetalWaterParameters(
    metal="Pt",
    size=(6, 6, 8),
    n_water=300,
    gap_above_metal=2.5,
    water_density=1.0,
    vacuum_above_water=10.0,
    output_file="pt_water.data",
    output_format="lammps",
    seed=42
)

generator = MetalWaterGenerator(params)
generator.run()
```

### LAMMPS Input Generation Example

```python
from mlip_struct_gen.generate_lammps_input.metal_water import (
    MetalWaterLAMMPSGenerator,
    MetalWaterLAMMPSParameters
)

# Generate LAMMPS input file
params = MetalWaterLAMMPSParameters(
    data_file="pt_water.data",
    metal_type="Pt",
    ensemble="NVT",
    temperatures=[300.0, 330.0, 360.0],
    equilibration_time=100.0,
    production_time=500.0,
    timestep=1.0,
    dump_frequency=1.0,
    fix_bottom_layers=3,
    output_file="in.pt_water"
)

generator = MetalWaterLAMMPSGenerator(params)
generator.generate()
```

### Wannier Centroid Computation Example

```python
from pathlib import Path
from mlip_struct_gen.wannier_centroid.compute_wannier_centroid import main

# Compute wannier centroids
folder_path = Path("/path/to/snapshot")
main(folder_path, verbose=True)

# Output: wc_out.txt and wc_out.npy
```

### DPData Conversion Example

```python
from mlip_struct_gen.generate_dpdata.dpdata_converter import DPDataConverter

# Convert VASP OUTCARs to dpdata format
converter = DPDataConverter(
    input_dir="./snapshots",
    output_dir="./DATA",
    type_map=["Cu", "O", "H", "Na", "Cl"],
    recursive=True,
    verbose=True
)

converter.run()
```

## Examples

The `examples/` directory contains comprehensive examples demonstrating all features:

### Structure Examples (`examples/structure_examples/`)
- `01_water_box_structures.py` - Water box generation with different models and sizes
- `02_salt_water_structures.py` - Electrolyte solutions with concentration series
- `03_metal_surface_structures.py` - Metal surface generation for various metals
- `04_metal_water_interface_structures.py` - Metal-water interfaces with variable coverage
- `05_metal_salt_water_structures.py` - Complex three-component electrochemical interfaces

### LAMMPS Examples (`examples/lammps_examples/`)
- `01_water_box_example.py` - Water simulations with different ensembles
- `02_salt_water_example.py` - Ion-water MD simulations
- `03_metal_surface_example.py` - Metal surface dynamics with fixed layers
- `04_metal_water_interface_example.py` - Interface simulations with temperature monitoring
- `05_metal_salt_water_example.py` - Full electrochemical interface simulations

Run any example:
```bash
cd examples/structure_examples
python 01_water_box_structures.py

cd ../lammps_examples
python 01_water_box_example.py
```

## Advanced Features

### Reproducible Generation
Use `--seed` for reproducible random number generation:
```bash
mlip-water-box --n-water 500 --seed 42 --output water.data
mlip-metal-water --metal Pt --size 6 6 8 --n-water 300 --seed 42 --output pt_water.data
```

### Dry Run Mode
Test parameters without generating files:
```bash
mlip-water-box --n-water 500 --output water.data --dry-run
mlip-dpdata --input-dir . --type-map O H --output-dir DATA/ --dry-run
```

### Artifact Saving
Save intermediate files for debugging:
```bash
mlip-salt-water-box --n-water 500 --n-salt 10 --salt NaCl --output solution.data --save-artifacts
```

### Save Input Parameters
Save all input parameters to JSON for reproducibility:
```bash
mlip-water-box --n-water 500 --output water.data --save-input
# Creates input_params.json

mlip-metal-salt-water --metal Au --size 5 5 6 --n-water 200 --n-salt 8 --salt KCl \
  --output interface.lammps --save-input
# Saves complete configuration for exact reproduction
```

### Multi-Temperature Sampling
Generate LAMMPS inputs that sample multiple temperatures sequentially:
```bash
# Single temperature
mlip-lammps-water water.data --temperature 300 --output in.water_300K

# Multiple temperatures (enhanced sampling)
mlip-lammps-water water.data --temperature 300 330 360 --output in.water_multiT
# Runs at 300K, then 330K, then 360K
```

### Custom Lattice Constants
Override default lattice constants for strained surfaces:
```bash
mlip-metal-surface --metal Cu --size 4 4 6 --lattice-constant 3.615 --output cu.vasp
```

## License

MIT License

Copyright (c) 2024 Osman Mamun
