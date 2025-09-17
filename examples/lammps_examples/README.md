# LAMMPS Examples for MLIP Structure Generator

This directory contains examples demonstrating how to generate LAMMPS input files for various systems using the mlip-struct-gen tools.

## Examples Overview

### 01_water_box_example.py
- Generate pure water box structures
- Create LAMMPS inputs for different water models (SPC/E, TIP3P, TIP4P)
- NPT and NVT ensemble simulations
- Multiple temperature sampling

### 02_salt_water_example.py
- Generate salt-water solutions at various concentrations
- Support for different salt types (NaCl, KCl, LiCl, etc.)
- Ion-water interactions using Joung-Cheatham parameters
- Concentration-dependent studies

### 03_metal_surface_example.py
- Generate metal surface structures (Au, Pt, Ag, Cu)
- LJ potentials for metals
- Fixed bottom layers for interface simulations
- NPT simulations for thermal expansion studies

### 04_metal_water_interface_example.py
- Create metal-water interface structures
- Proper metal-water interactions
- Monitor water temperature separately
- Compare different water models at interfaces

### 05_metal_salt_water_example.py
- Complex metal-salt-water interfaces
- Full interaction matrix (metal-water-ion)
- Monitor water and solution temperatures
- Electrochemical interface simulations

## Quick Start

1. **Install dependencies:**
```bash
pip install -e .
```

2. **Run an example:**
```bash
python 01_water_box_example.py
```

3. **Run LAMMPS simulation:**
```bash
lmp -i in.water
```

## Key Features

### Temperature Monitoring
- System temperature (all atoms)
- Water temperature (T_water)
- Solution temperature (T_soln) for salt solutions

### Atom Type Convention
Consistent atom type ordering across all generators:
- Type 1: Metal (when present)
- Type 2: Oxygen (O)
- Type 3: Hydrogen (H)
- Type 4: Cation (when salt present)
- Type 5: Anion (when salt present)

### Force Field Parameters
- **Water**: SPC/E, TIP3P parameters
- **Ions**: Joung-Cheatham parameters
- **Metals**: LJ parameters from literature
- **Cross interactions**: Lorentz-Berthelot mixing rules

### Simulation Settings
- **Units**: metal (eV, Angstrom, ps)
- **Ensembles**: NPT, NVT, NVE
- **Constraints**: SHAKE for rigid water molecules
- **Fixed layers**: Option to fix bottom metal layers

## Output Files

Each simulation produces:
- `trajectory.lammpstrj`: Atomic trajectories for visualization
- `forces.dump`: Forces and stress (if enabled) for MLIP training
- `final_configuration.data`: Final system configuration
- `restart*.lmp`: Restart files for continuing simulations
- `log.lammps`: Thermodynamic output

## Analysis Tips

### Visualizing Trajectories
```bash
# Use OVITO
ovito trajectory.lammpstrj

# Use VMD
vmd trajectory.lammpstrj
```

### Extracting Thermodynamic Data
```bash
# Extract temperature data
grep "T_water" log.lammps > water_temp.dat

# Extract all thermo data
grep "^[0-9]" log.lammps > thermo.dat
```

### Density Profiles
Use post-processing tools to calculate:
- Water density as function of distance from metal
- Ion distribution profiles
- Orientation of water molecules at interface

## Common Issues and Solutions

### Simulation Crashes
- Reduce timestep (default: 1.0 fs)
- Increase equilibration time
- Check for overlapping atoms in initial structure

### Temperature Drift
- Adjust thermostat damping parameter
- Use NPT for better equilibration
- Check if fixed layers are properly defined

### Memory Issues
- Reduce system size
- Use fewer atoms
- Adjust neighbor list settings

## References

- SPC/E water: Berendsen et al., J. Phys. Chem. 91, 6269 (1987)
- Joung-Cheatham ions: J. Phys. Chem. B 112, 9020 (2008)
- Metal LJ parameters: Various literature sources
- LAMMPS: https://www.lammps.org/
