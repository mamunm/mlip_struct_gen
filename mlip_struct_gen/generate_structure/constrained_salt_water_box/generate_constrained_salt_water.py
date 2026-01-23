"""Constrained salt water box generation."""

import random
import tempfile
from pathlib import Path

from ase import io as ase_io

from ..salt_water_box import SaltWaterBoxGenerator, SaltWaterBoxGeneratorParameters
from ..templates.salt_models import get_salt_model
from .geometry_modifier import (
    find_hoh_angles,
    find_ion_pairs,
    find_ion_water_pairs,
    find_ions,
    find_nearest_oo_pairs,
    find_oh_bonds,
    find_water_molecules,
    get_current_angle,
    get_current_distance,
    modify_angle,
    modify_bond_distance,
    modify_intermolecular_distance,
    move_ion_to_distance,
)
from .input_parameters import ConstrainedSaltWaterBoxParameters
from .lammps_input import ELEMENT_MASSES, generate_lammps_input
from .validation import validate_parameters


class ConstrainedSaltWaterBoxGenerator:
    """Generate constrained salt water boxes for MLIP training."""

    def __init__(self, parameters: ConstrainedSaltWaterBoxParameters):
        """
        Initialize constrained salt water box generator.

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

        # Get salt model info
        self.salt_model = get_salt_model(self.parameters.salt_type)
        self.cation_element = self.salt_model["cation"]["element"]
        self.anion_element = self.salt_model["anion"]["element"]

        # Track constrained atoms for LAMMPS fix generation
        # All constrained atoms are frozen in place using fix setforce 0 0 0
        self.constrained_atoms: dict = {"frozen_atoms": set()}

        if self.logger:
            self.logger.info("Initializing ConstrainedSaltWaterBoxGenerator")
            self.logger.info(f"Salt type: {self.parameters.salt_type}")

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate constrained salt water box.

        Args:
            save_artifacts: Save intermediate files

        Returns:
            Path to generated output file
        """
        if self.logger:
            self.logger.info("Starting constrained salt water box generation")

        # Step 1: Generate normal salt water box
        if self.logger:
            self.logger.step("Generating initial salt water box with Packmol")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_xyz = Path(tmpdir) / "initial_salt_water.xyz"

            salt_water_params = SaltWaterBoxGeneratorParameters(
                output_file=str(temp_xyz),
                box_size=self.parameters.box_size,
                water_model=self.parameters.water_model,
                n_water=self.parameters.n_water,
                density=self.parameters.density,
                salt_type=self.parameters.salt_type,
                n_salt=self.parameters.n_salt,
                include_salt_volume=self.parameters.include_salt_volume,
                tolerance=self.parameters.tolerance,
                seed=self.parameters.seed,
                packmol_executable=self.parameters.packmol_executable,
                output_format="xyz",
                log=self.parameters.log,
                logger=self.logger,
            )

            salt_water_generator = SaltWaterBoxGenerator(salt_water_params)
            salt_water_generator.run()

            # Get the computed box_size from the generator
            self.parameters.box_size = salt_water_generator.parameters.box_size

            # Step 2: Load structure with ASE
            if self.logger:
                self.logger.step("Loading structure for constraint application")

            atoms = ase_io.read(str(temp_xyz))

            # Step 3: Identify water molecules and ions
            molecules = find_water_molecules(atoms)
            ion_elements = [self.cation_element, self.anion_element]
            ions = find_ions(atoms, ion_elements)

            if self.logger:
                self.logger.info(f"Found {len(molecules)} water molecules")
                self.logger.info(
                    f"Found {len(ions[self.cation_element])} {self.cation_element} ions"
                )
                self.logger.info(f"Found {len(ions[self.anion_element])} {self.anion_element} ions")

            # Step 4: Apply distance constraints
            random.seed(self.parameters.constraint_seed)

            for constraint in self.parameters.distance_constraints:
                self._apply_distance_constraint(atoms, molecules, ions, constraint)

            # Step 5: Apply angle constraints
            for constraint in self.parameters.angle_constraints:
                self._apply_angle_constraint(atoms, molecules, constraint)

            # Step 6: Write output files
            if self.logger:
                self.logger.step("Writing output files")

            self._write_output(atoms)

        if self.logger:
            self.logger.success("Constrained salt water box generation completed")
            self.logger.info(f"Output: {self.parameters.output_file}")

        return self.parameters.output_file

    def _apply_distance_constraint(self, atoms, molecules, ions, constraint) -> None:
        """Apply a distance constraint (water, ion-water, or ion-ion)."""
        elem1, elem2 = constraint.element1, constraint.element2

        # Categorize constraint type
        water_elements = {"O", "H"}
        ion_elements = {self.cation_element, self.anion_element}

        is_water_only = elem1 in water_elements and elem2 in water_elements
        is_ion_water = (elem1 in ion_elements and elem2 in water_elements) or (
            elem1 in water_elements and elem2 in ion_elements
        )
        is_ion_ion = elem1 in ion_elements and elem2 in ion_elements

        if is_water_only:
            self._apply_water_distance_constraint(atoms, molecules, elem1, elem2, constraint)
        elif is_ion_water:
            self._apply_ion_water_constraint(atoms, molecules, ions, elem1, elem2, constraint)
        elif is_ion_ion:
            self._apply_ion_ion_constraint(atoms, ions, elem1, elem2, constraint)

    def _apply_water_distance_constraint(self, atoms, molecules, elem1, elem2, constraint) -> None:
        """Apply water-only distance constraint (O-H or O-O)."""
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
                    f"Applying O-H distance constraint: {len(selected_bonds)} bonds to {target_dist} A (freezing atoms)"
                )

            for o_idx, h_idx, mol_idx in selected_bonds:
                old_dist = get_current_distance(atoms, o_idx, h_idx)
                modify_bond_distance(atoms, o_idx, h_idx, target_dist)
                new_dist = get_current_distance(atoms, o_idx, h_idx)

                # Freeze entire water molecule - 1-indexed for LAMMPS
                mol_indices = molecules[mol_idx]
                for atom_idx in mol_indices:
                    self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

                if self.logger:
                    self.logger.debug(
                        f"  Molecule {mol_idx}: O-H {old_dist:.3f} -> {new_dist:.3f} A"
                    )
        else:
            # O-O intermolecular constraint - freeze entire water molecules
            pairs = find_nearest_oo_pairs(atoms, molecules, constraint.count)

            if self.logger:
                self.logger.info(
                    f"Applying O-O distance constraint: {len(pairs)} pairs to {target_dist} A (freezing atoms)"
                )

            for mol1_idx, mol2_idx in pairs:
                mol1 = molecules[mol1_idx]
                mol2 = molecules[mol2_idx]
                o1_idx, o2_idx = mol1[0], mol2[0]

                old_dist = get_current_distance(atoms, o1_idx, o2_idx)
                modify_intermolecular_distance(atoms, mol1, mol2, target_dist)
                new_dist = get_current_distance(atoms, o1_idx, o2_idx)

                # Freeze both water molecules (O and H atoms) - 1-indexed for LAMMPS
                for atom_idx in mol1:
                    self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)
                for atom_idx in mol2:
                    self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

                if self.logger:
                    self.logger.debug(
                        f"  Molecules {mol1_idx}-{mol2_idx}: O-O {old_dist:.3f} -> {new_dist:.3f} A"
                    )

    def _apply_ion_water_constraint(self, atoms, molecules, ions, elem1, elem2, constraint) -> None:
        """Apply ion-water distance constraint - freeze ion and water molecule."""
        target_dist = constraint.distance

        # Determine which is ion and which is water element
        ion_elements = {self.cation_element, self.anion_element}
        if elem1 in ion_elements:
            ion_element = elem1
            water_element = elem2
        else:
            ion_element = elem2
            water_element = elem1

        ion_indices = ions[ion_element]

        if not ion_indices:
            if self.logger:
                self.logger.warning(f"No {ion_element} ions found for constraint")
            return

        # Find ion-water pairs
        pairs = find_ion_water_pairs(
            atoms, ion_element, water_element, molecules, ion_indices, constraint.count
        )

        if self.logger:
            self.logger.info(
                f"Applying {ion_element}-{water_element} distance constraint: "
                f"{len(pairs)} pairs to {target_dist} A (freezing atoms)"
            )

        for ion_idx, water_idx, mol_idx in pairs:
            old_dist = get_current_distance(atoms, ion_idx, water_idx)

            # Move ion towards/away from water atom
            move_ion_to_distance(atoms, water_idx, ion_idx, target_dist)
            new_dist = get_current_distance(atoms, ion_idx, water_idx)

            # Freeze ion and entire water molecule (O and H atoms) - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(ion_idx + 1)
            mol_indices = molecules[mol_idx]
            for atom_idx in mol_indices:
                self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

            if self.logger:
                self.logger.debug(
                    f"  {ion_element}({ion_idx})-{water_element}({water_idx}): "
                    f"{old_dist:.3f} -> {new_dist:.3f} A"
                )

    def _apply_ion_ion_constraint(self, atoms, ions, elem1, elem2, constraint) -> None:
        """Apply ion-ion distance constraint - freeze both ions."""
        target_dist = constraint.distance

        indices1 = ions.get(elem1, [])
        indices2 = ions.get(elem2, [])

        if not indices1 or not indices2:
            if self.logger:
                self.logger.warning(f"Not enough ions for {elem1}-{elem2} constraint")
            return

        # Find ion-ion pairs
        pairs = find_ion_pairs(atoms, elem1, elem2, indices1, indices2, constraint.count)

        if self.logger:
            self.logger.info(
                f"Applying {elem1}-{elem2} distance constraint: {len(pairs)} pairs to {target_dist} A (freezing atoms)"
            )

        for idx1, idx2 in pairs:
            old_dist = get_current_distance(atoms, idx1, idx2)

            # Move second ion towards/away from first
            move_ion_to_distance(atoms, idx1, idx2, target_dist)
            new_dist = get_current_distance(atoms, idx1, idx2)

            # Freeze both ions - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(idx1 + 1)
            self.constrained_atoms["frozen_atoms"].add(idx2 + 1)

            if self.logger:
                self.logger.debug(
                    f"  {elem1}({idx1})-{elem2}({idx2}): {old_dist:.3f} -> {new_dist:.3f} A"
                )

    def _apply_angle_constraint(self, atoms, molecules, constraint) -> None:
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
                f"Applying H-O-H angle constraint: {len(selected_angles)} angles to {target_angle} deg (freezing atoms)"
            )

        for h1_idx, o_idx, h2_idx, mol_idx in selected_angles:
            old_angle = get_current_angle(atoms, h1_idx, o_idx, h2_idx)
            modify_angle(atoms, h1_idx, o_idx, h2_idx, target_angle)
            new_angle = get_current_angle(atoms, h1_idx, o_idx, h2_idx)

            # Freeze entire water molecule - 1-indexed for LAMMPS
            mol_indices = molecules[mol_idx]
            for atom_idx in mol_indices:
                self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

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
            minimize=self.parameters.minimize,
            ensemble=self.parameters.ensemble,
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
            f.write("# LAMMPS data file for constrained salt water box (atomic style for DeepMD)\n")
            f.write("# Generated by mlip-struct-gen\n\n")

            f.write(f"{n_atoms} atoms\n\n")
            f.write(f"{len(self.parameters.elements)} atom types\n\n")

            # Box dimensions
            box = self.parameters.box_size
            f.write(f"0.0 {box[0]:.6f} xlo xhi\n")
            f.write(f"0.0 {box[1]:.6f} ylo yhi\n")
            f.write(f"0.0 {box[2]:.6f} zlo zhi\n\n")

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
