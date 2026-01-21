"""Constrained metal-water interface generation."""

import random
import tempfile
from pathlib import Path

from ase import io as ase_io

from ..metal_water import MetalWaterGenerator, MetalWaterParameters
from .geometry_modifier import (
    find_hoh_angles,
    find_metal_water_angles,
    find_metal_water_pairs,
    find_nearest_oo_pairs,
    find_oh_bonds,
    find_surface_metal_atoms,
    find_water_molecules,
    get_current_angle,
    get_current_distance,
    modify_angle,
    modify_bond_distance,
    modify_intermolecular_distance,
    modify_metal_water_angle,
    move_water_molecule_to_metal_distance,
)
from .input_parameters import ConstrainedMetalWaterParameters
from .lammps_input import ELEMENT_MASSES, generate_lammps_input
from .validation import get_lattice_constant, validate_parameters


class ConstrainedMetalWaterGenerator:
    """Generate constrained metal-water interfaces for MLIP training."""

    def __init__(self, parameters: ConstrainedMetalWaterParameters):
        """
        Initialize constrained metal-water generator.

        Args:
            parameters: Generation parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.parameters = parameters
        validate_parameters(self.parameters)

        # Setup logger
        self.logger = None
        if self.parameters.log:
            if self.parameters.logger is not None:
                self.logger = self.parameters.logger
            else:
                try:
                    from ...utils.logger import MLIPLogger

                    self.logger = MLIPLogger()
                except ImportError:
                    self.logger = None

        # Get lattice constant
        self.lattice_constant = get_lattice_constant(
            self.parameters.metal, self.parameters.lattice_constant
        )

        # Track constrained atoms for LAMMPS fix generation
        self.constrained_atoms: dict = {"distance": [], "angle": []}

        if self.logger:
            self.logger.info("Initializing ConstrainedMetalWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Surface size: {self.parameters.size}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.6f} A")

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate constrained metal-water interface.

        Args:
            save_artifacts: Save intermediate files

        Returns:
            Path to generated output file
        """
        if self.logger:
            self.logger.info("Starting constrained metal-water interface generation")

        # Step 1: Generate initial metal-water structure
        if self.logger:
            self.logger.step("Generating initial metal-water interface with MetalWaterGenerator")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_data = Path(tmpdir) / "initial_metal_water.data"

            metal_water_params = MetalWaterParameters(
                output_file=str(temp_data),
                metal=self.parameters.metal,
                size=self.parameters.size,
                n_water=self.parameters.n_water,
                density=self.parameters.density,
                gap_above_metal=self.parameters.gap_above_metal,
                vacuum_above_water=self.parameters.vacuum_above_water,
                water_model=self.parameters.water_model,
                lattice_constant=self.lattice_constant,
                fix_bottom_layers=self.parameters.fix_bottom_layers,
                packmol_executable=self.parameters.packmol_executable,
                packmol_tolerance=self.parameters.packmol_tolerance,
                seed=self.parameters.seed,
                output_format="lammps/dpmd",
                elements=self.parameters.elements,
                log=self.parameters.log,
                logger=self.logger,
            )

            metal_water_generator = MetalWaterGenerator(metal_water_params)
            metal_water_generator.run()

            # Step 2: Load structure with ASE
            if self.logger:
                self.logger.step("Loading structure for constraint application")

            # Read LAMMPS data file
            atoms = ase_io.read(str(temp_data), format="lammps-data", style="atomic")

            # Get cell dimensions from the generator
            self.cell = metal_water_generator.combined_system.get_cell()

            # Step 3: Identify water molecules and surface metal atoms
            molecules = find_water_molecules(atoms)
            surface_metals = find_surface_metal_atoms(atoms, self.parameters.metal)

            if self.logger:
                self.logger.info(f"Found {len(molecules)} water molecules")
                self.logger.info(
                    f"Found {len(surface_metals)} surface {self.parameters.metal} atoms"
                )

            # Step 4: Apply metal-water distance constraints
            random.seed(self.parameters.constraint_seed)

            for constraint in self.parameters.metal_water_distance_constraints:
                self._apply_metal_water_distance_constraint(
                    atoms, molecules, surface_metals, constraint
                )

            # Step 5: Apply metal-water angle constraints
            for constraint in self.parameters.metal_water_angle_constraints:
                self._apply_metal_water_angle_constraint(
                    atoms, molecules, surface_metals, constraint
                )

            # Step 6: Apply water-only distance constraints
            for constraint in self.parameters.distance_constraints:
                self._apply_water_distance_constraint(atoms, molecules, constraint)

            # Step 7: Apply water-only angle constraints
            for constraint in self.parameters.angle_constraints:
                self._apply_water_angle_constraint(atoms, molecules, constraint)

            # Step 8: Write output files
            if self.logger:
                self.logger.step("Writing output files")

            self._write_output(atoms)

        if self.logger:
            self.logger.success("Constrained metal-water interface generation completed")
            self.logger.info(f"Output: {self.parameters.output_file}")

        return self.parameters.output_file

    def _apply_metal_water_distance_constraint(
        self, atoms, molecules, surface_metals, constraint
    ) -> None:
        """Apply metal-water distance constraint (metal fixed, water moves)."""
        target_dist = constraint.distance
        water_element = constraint.water_element

        if not surface_metals:
            if self.logger:
                self.logger.warning("No surface metal atoms found for constraint")
            return

        # Find metal-water pairs
        pairs = find_metal_water_pairs(
            atoms,
            self.parameters.metal,
            water_element,
            molecules,
            surface_metals,
            constraint.count,
        )

        if self.logger:
            self.logger.info(
                f"Applying {self.parameters.metal}-{water_element} distance constraint: "
                f"{len(pairs)} pairs to {target_dist} A"
            )

        for metal_idx, water_idx, mol_idx in pairs:
            mol_indices = molecules[mol_idx]
            old_dist = get_current_distance(atoms, metal_idx, water_idx)

            # Move entire water molecule as rigid body
            move_water_molecule_to_metal_distance(
                atoms, metal_idx, water_idx, mol_indices, target_dist
            )
            new_dist = get_current_distance(atoms, metal_idx, water_idx)

            # Store for LAMMPS fix (1-indexed)
            self.constrained_atoms["distance"].append((metal_idx + 1, water_idx + 1, target_dist))

            if self.logger:
                self.logger.debug(
                    f"  {self.parameters.metal}({metal_idx})-{water_element}({water_idx}): "
                    f"{old_dist:.3f} -> {new_dist:.3f} A"
                )

    def _apply_metal_water_angle_constraint(
        self, atoms, molecules, surface_metals, constraint
    ) -> None:
        """Apply metal-water angle constraint (Metal-O-H)."""
        target_angle = constraint.angle

        if not surface_metals:
            if self.logger:
                self.logger.warning("No surface metal atoms found for angle constraint")
            return

        # Find Metal-O-H angles
        angles = find_metal_water_angles(
            atoms, self.parameters.metal, molecules, surface_metals, constraint.count
        )

        if self.logger:
            self.logger.info(
                f"Applying {self.parameters.metal}-O-H angle constraint: "
                f"{len(angles)} angles to {target_angle} deg"
            )

        for metal_idx, o_idx, h_idx, mol_idx, h_num in angles:
            mol_indices = molecules[mol_idx]
            old_angle = get_current_angle(atoms, metal_idx, o_idx, h_idx)

            # Rotate water molecule around Metal-O axis
            modify_metal_water_angle(atoms, metal_idx, o_idx, h_idx, mol_indices, target_angle)
            new_angle = get_current_angle(atoms, metal_idx, o_idx, h_idx)

            # Store for LAMMPS fix (1-indexed)
            self.constrained_atoms["angle"].append(
                (metal_idx + 1, o_idx + 1, h_idx + 1, target_angle)
            )

            if self.logger:
                self.logger.debug(
                    f"  Mol {mol_idx} H{h_num}: {self.parameters.metal}-O-H "
                    f"{old_angle:.1f} -> {new_angle:.1f} deg"
                )

    def _apply_water_distance_constraint(self, atoms, molecules, constraint) -> None:
        """Apply water-only distance constraint (O-H or O-O)."""
        elem1, elem2 = constraint.element1, constraint.element2
        target_dist = constraint.distance

        # Determine if intramolecular (O-H) or intermolecular (O-O)
        is_intramolecular = (elem1 == "O" and elem2 == "H") or (elem1 == "H" and elem2 == "O")

        if is_intramolecular:
            # O-H bond constraint
            bonds = find_oh_bonds(atoms, molecules)

            if constraint.count == "all":
                selected_bonds = bonds
            else:
                count = min(int(constraint.count), len(bonds))
                selected_bonds = random.sample(bonds, count)

            if self.logger:
                self.logger.info(
                    f"Applying O-H distance constraint: {len(selected_bonds)} bonds to {target_dist} A"
                )

            for o_idx, h_idx, mol_idx in selected_bonds:
                old_dist = get_current_distance(atoms, o_idx, h_idx)
                modify_bond_distance(atoms, o_idx, h_idx, target_dist)
                new_dist = get_current_distance(atoms, o_idx, h_idx)

                # Store for LAMMPS fix (1-indexed)
                self.constrained_atoms["distance"].append((o_idx + 1, h_idx + 1, target_dist))

                if self.logger:
                    self.logger.debug(
                        f"  Molecule {mol_idx}: O-H {old_dist:.3f} -> {new_dist:.3f} A"
                    )
        else:
            # O-O intermolecular constraint
            pairs = find_nearest_oo_pairs(atoms, molecules, constraint.count)

            if self.logger:
                self.logger.info(
                    f"Applying O-O distance constraint: {len(pairs)} pairs to {target_dist} A"
                )

            for mol1_idx, mol2_idx in pairs:
                mol1 = molecules[mol1_idx]
                mol2 = molecules[mol2_idx]
                o1_idx, o2_idx = mol1[0], mol2[0]

                old_dist = get_current_distance(atoms, o1_idx, o2_idx)
                modify_intermolecular_distance(atoms, mol1, mol2, target_dist)
                new_dist = get_current_distance(atoms, o1_idx, o2_idx)

                # Store for LAMMPS fix (1-indexed)
                self.constrained_atoms["distance"].append((o1_idx + 1, o2_idx + 1, target_dist))

                if self.logger:
                    self.logger.debug(
                        f"  Molecules {mol1_idx}-{mol2_idx}: O-O {old_dist:.3f} -> {new_dist:.3f} A"
                    )

    def _apply_water_angle_constraint(self, atoms, molecules, constraint) -> None:
        """Apply H-O-H angle constraint."""
        target_angle = constraint.angle
        angles = find_hoh_angles(atoms, molecules)

        if constraint.count == "all":
            selected_angles = angles
        else:
            count = min(int(constraint.count), len(angles))
            selected_angles = random.sample(angles, count)

        if self.logger:
            self.logger.info(
                f"Applying H-O-H angle constraint: {len(selected_angles)} angles to {target_angle} deg"
            )

        for h1_idx, o_idx, h2_idx, mol_idx in selected_angles:
            old_angle = get_current_angle(atoms, h1_idx, o_idx, h2_idx)
            modify_angle(atoms, h1_idx, o_idx, h2_idx, target_angle)
            new_angle = get_current_angle(atoms, h1_idx, o_idx, h2_idx)

            # Store for LAMMPS fix (1-indexed)
            self.constrained_atoms["angle"].append(
                (h1_idx + 1, o_idx + 1, h2_idx + 1, target_angle)
            )

            if self.logger:
                self.logger.debug(
                    f"  Molecule {mol_idx}: H-O-H {old_angle:.1f} -> {new_angle:.1f} deg"
                )

    def _write_output(self, atoms) -> None:
        """Write LAMMPS data file and input file."""
        output_path = Path(self.parameters.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write LAMMPS data file
        self._write_lammps_data(atoms, output_path)

        # Generate LAMMPS input file
        lammps_input_file = output_path.parent / f"in.{output_path.stem}.lammps"
        generate_lammps_input(
            data_file=output_path.name,
            model_files=self.parameters.model_files,
            constrained_atoms=self.constrained_atoms,
            elements=self.parameters.elements,
            output_file=str(lammps_input_file),
            constraint_type=self.parameters.constraint_type,
            harmonic_k=self.parameters.harmonic_k,
            minimize=self.parameters.minimize,
            nsteps=self.parameters.nsteps,
            thermo_freq=self.parameters.thermo_freq,
            dump_freq=self.parameters.dump_freq,
            temp=self.parameters.temp,
            pres=self.parameters.pres,
            tau_t=self.parameters.tau_t,
            tau_p=self.parameters.tau_p,
            timestep=self.parameters.timestep,
            seed=self.parameters.seed,
        )

        if self.logger:
            self.logger.info(f"LAMMPS data file: {output_path}")
            self.logger.info(f"LAMMPS input file: {lammps_input_file}")

    def _write_lammps_data(self, atoms, output_path: Path) -> None:
        """Write LAMMPS data file in atomic style for DeepMD."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        n_atoms = len(atoms)

        # Create element to type mapping
        element_to_type = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}

        with open(output_path, "w") as f:
            f.write(
                f"# LAMMPS data file for constrained {self.parameters.metal}-water interface "
                f"(atomic style for DeepMD)\n"
            )
            f.write("# Generated by mlip-struct-gen\n\n")

            f.write(f"{n_atoms} atoms\n\n")
            f.write(f"{len(self.parameters.elements)} atom types\n\n")

            # Box dimensions from cell
            f.write(f"0.0 {self.cell[0, 0]:.6f} xlo xhi\n")
            f.write(f"0.0 {self.cell[1, 1]:.6f} ylo yhi\n")
            f.write(f"0.0 {self.cell[2, 2]:.6f} zlo zhi\n\n")

            # Masses
            f.write("Masses\n\n")
            for i, elem in enumerate(self.parameters.elements, 1):
                mass = ELEMENT_MASSES.get(elem, 1.0)
                f.write(f"{i} {mass:.4f}  # {elem}\n")
            f.write("\n")

            # Atoms (atomic style: id type x y z)
            f.write("Atoms # atomic\n\n")
            for atom_id, (symbol, pos) in enumerate(zip(symbols, positions, strict=False), 1):
                atom_type = element_to_type.get(symbol, 1)
                f.write(f"{atom_id} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
