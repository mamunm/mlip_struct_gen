"""Constrained metal-salt-water interface generation."""

import random
import tempfile
from pathlib import Path

from ase import io as ase_io

from ..metal_salt_water import MetalSaltWaterGenerator, MetalSaltWaterParameters
from .geometry_modifier import (
    find_hoh_angles,
    find_ion_ion_pairs,
    find_ion_water_pairs,
    find_metal_ion_pairs,
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
    move_ion_to_ion_distance,
    move_ion_to_metal_distance,
    move_water_molecule_to_metal_distance,
    move_water_to_ion_distance,
)
from .input_parameters import ConstrainedMetalSaltWaterParameters
from .lammps_input import ELEMENT_MASSES, generate_lammps_input
from .validation import get_lattice_constant, get_salt_info, validate_parameters


class ConstrainedMetalSaltWaterGenerator:
    """Generate constrained metal-salt-water interfaces for MLIP training."""

    def __init__(self, parameters: ConstrainedMetalSaltWaterParameters):
        """
        Initialize constrained metal-salt-water generator.

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

        # Get salt info
        self.salt_info = get_salt_info(self.parameters.salt_type)

        # Track constrained atoms for LAMMPS fix generation
        # "distance": O-H bond constraints (use harmonic restraints)
        # "angle": H-O-H angle constraints (use harmonic restraints)
        # "frozen_atoms": atoms to freeze in place (for O-O, metal-water, metal-ion, ion-water, ion-ion)
        self.constrained_atoms: dict = {"distance": [], "angle": [], "frozen_atoms": set()}

        if self.logger:
            self.logger.info("Initializing ConstrainedMetalSaltWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Surface size: {self.parameters.size}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.6f} A")
            self.logger.info(f"Salt: {self.parameters.salt_type} ({self.parameters.n_salt} units)")

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate constrained metal-salt-water interface.

        Args:
            save_artifacts: Save intermediate files

        Returns:
            Path to generated output file
        """
        if self.logger:
            self.logger.info("Starting constrained metal-salt-water interface generation")

        # Step 1: Generate initial metal-salt-water structure
        if self.logger:
            self.logger.step("Generating initial metal-salt-water interface")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_data = Path(tmpdir) / "initial_metal_salt_water.data"

            metal_salt_water_params = MetalSaltWaterParameters(
                output_file=str(temp_data),
                metal=self.parameters.metal,
                size=self.parameters.size,
                n_water=self.parameters.n_water,
                salt_type=self.parameters.salt_type,
                n_salt=self.parameters.n_salt,
                include_salt_volume=self.parameters.include_salt_volume,
                density=self.parameters.density,
                gap=self.parameters.gap,
                vacuum_above_water=self.parameters.vacuum_above_water,
                no_salt_zone=self.parameters.no_salt_zone,
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

            metal_salt_water_generator = MetalSaltWaterGenerator(metal_salt_water_params)
            metal_salt_water_generator.run()

            # Step 2: Load structure with ASE
            if self.logger:
                self.logger.step("Loading structure for constraint application")

            # Read LAMMPS data file
            atoms = ase_io.read(str(temp_data), format="lammps-data", style="atomic")

            # Get cell dimensions from the generator
            self.cell = metal_salt_water_generator.combined_system.get_cell()

            # Step 3: Identify components
            molecules = find_water_molecules(atoms)
            surface_metals = find_surface_metal_atoms(atoms, self.parameters.metal)

            if self.logger:
                self.logger.info(f"Found {len(molecules)} water molecules")
                self.logger.info(
                    f"Found {len(surface_metals)} surface {self.parameters.metal} atoms"
                )

            # Step 4: Apply constraints
            random.seed(self.parameters.constraint_seed)

            # Metal-water distance constraints
            for constraint in self.parameters.metal_water_distance_constraints:
                self._apply_metal_water_distance_constraint(
                    atoms, molecules, surface_metals, constraint
                )

            # Metal-water angle constraints
            for constraint in self.parameters.metal_water_angle_constraints:
                self._apply_metal_water_angle_constraint(
                    atoms, molecules, surface_metals, constraint
                )

            # Metal-ion distance constraints
            for constraint in self.parameters.metal_ion_distance_constraints:
                self._apply_metal_ion_distance_constraint(atoms, surface_metals, constraint)

            # General distance constraints (O-H, O-O, Ion-O, Ion-Ion)
            for constraint in self.parameters.distance_constraints:
                self._apply_distance_constraint(atoms, molecules, constraint)

            # Angle constraints (H-O-H)
            for constraint in self.parameters.angle_constraints:
                self._apply_water_angle_constraint(atoms, molecules, constraint)

            # Step 5: Write output files
            if self.logger:
                self.logger.step("Writing output files")

            self._write_output(atoms)

        if self.logger:
            self.logger.success("Constrained metal-salt-water interface generation completed")
            self.logger.info(f"Output: {self.parameters.output_file}")

        return self.parameters.output_file

    def _apply_metal_water_distance_constraint(
        self, atoms, molecules, surface_metals, constraint
    ) -> None:
        """Apply metal-water distance constraint - freeze metal and water molecule."""
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
                f"{len(pairs)} pairs to {target_dist} A (freezing atoms)"
            )

        for metal_idx, water_idx, mol_idx in pairs:
            mol_indices = molecules[mol_idx]
            old_dist = get_current_distance(atoms, metal_idx, water_idx)

            # Move entire water molecule as rigid body
            move_water_molecule_to_metal_distance(
                atoms, metal_idx, water_idx, mol_indices, target_dist
            )
            new_dist = get_current_distance(atoms, metal_idx, water_idx)

            # Freeze metal atom and entire water molecule - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(metal_idx + 1)
            for atom_idx in mol_indices:
                self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

            if self.logger:
                self.logger.debug(
                    f"  {self.parameters.metal}({metal_idx})-{water_element}({water_idx}): "
                    f"{old_dist:.3f} -> {new_dist:.3f} A"
                )

    def _apply_metal_water_angle_constraint(
        self, atoms, molecules, surface_metals, constraint
    ) -> None:
        """Apply metal-water angle constraint (Metal-O-H) - freeze metal and water molecule."""
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
                f"{len(angles)} angles to {target_angle} deg (freezing atoms)"
            )

        for metal_idx, o_idx, h_idx, mol_idx, h_num in angles:
            mol_indices = molecules[mol_idx]
            old_angle = get_current_angle(atoms, metal_idx, o_idx, h_idx)

            # Rotate water molecule around Metal-O axis
            modify_metal_water_angle(atoms, metal_idx, o_idx, h_idx, mol_indices, target_angle)
            new_angle = get_current_angle(atoms, metal_idx, o_idx, h_idx)

            # Freeze metal atom and entire water molecule - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(metal_idx + 1)
            for atom_idx in mol_indices:
                self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

            if self.logger:
                self.logger.debug(
                    f"  Mol {mol_idx} H{h_num}: {self.parameters.metal}-O-H "
                    f"{old_angle:.1f} -> {new_angle:.1f} deg"
                )

    def _apply_metal_ion_distance_constraint(self, atoms, surface_metals, constraint) -> None:
        """Apply metal-ion distance constraint - freeze both metal and ion."""
        target_dist = constraint.distance
        ion_element = constraint.ion_element

        if not surface_metals:
            if self.logger:
                self.logger.warning("No surface metal atoms found for constraint")
            return

        # Find metal-ion pairs
        pairs = find_metal_ion_pairs(
            atoms,
            self.parameters.metal,
            ion_element,
            surface_metals,
            constraint.count,
        )

        if self.logger:
            self.logger.info(
                f"Applying {self.parameters.metal}-{ion_element} distance constraint: "
                f"{len(pairs)} pairs to {target_dist} A (freezing atoms)"
            )

        for metal_idx, ion_idx in pairs:
            old_dist = get_current_distance(atoms, metal_idx, ion_idx)

            # Move ion
            move_ion_to_metal_distance(atoms, metal_idx, ion_idx, target_dist)
            new_dist = get_current_distance(atoms, metal_idx, ion_idx)

            # Freeze both metal and ion - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(metal_idx + 1)
            self.constrained_atoms["frozen_atoms"].add(ion_idx + 1)

            if self.logger:
                self.logger.debug(
                    f"  {self.parameters.metal}({metal_idx})-{ion_element}({ion_idx}): "
                    f"{old_dist:.3f} -> {new_dist:.3f} A"
                )

    def _apply_distance_constraint(self, atoms, molecules, constraint) -> None:
        """Apply general distance constraint (O-H, O-O, Ion-O, Ion-Ion)."""
        elem1, elem2 = constraint.element1, constraint.element2

        # Determine constraint type
        is_water_water = elem1 in {"O", "H"} and elem2 in {"O", "H"}
        is_ion_water = (elem1 in {"Na", "K", "Li", "Cs", "Cl"} and elem2 in {"O", "H"}) or (
            elem2 in {"Na", "K", "Li", "Cs", "Cl"} and elem1 in {"O", "H"}
        )
        is_ion_ion = elem1 in {"Na", "K", "Li", "Cs", "Cl"} and elem2 in {
            "Na",
            "K",
            "Li",
            "Cs",
            "Cl",
        }

        if is_water_water:
            self._apply_water_distance_constraint(atoms, molecules, constraint)
        elif is_ion_water:
            self._apply_ion_water_distance_constraint(atoms, molecules, constraint)
        elif is_ion_ion:
            self._apply_ion_ion_distance_constraint(atoms, constraint)

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

    def _apply_ion_water_distance_constraint(self, atoms, molecules, constraint) -> None:
        """Apply ion-water distance constraint - freeze ion and water molecule."""
        elem1, elem2 = constraint.element1, constraint.element2
        target_dist = constraint.distance

        # Determine which is ion and which is water element
        if elem1 in {"Na", "K", "Li", "Cs", "Cl"}:
            ion_element = elem1
            water_element = elem2
        else:
            ion_element = elem2
            water_element = elem1

        # Find ion-water pairs
        pairs = find_ion_water_pairs(atoms, ion_element, water_element, molecules, constraint.count)

        if self.logger:
            self.logger.info(
                f"Applying {ion_element}-{water_element} distance constraint: "
                f"{len(pairs)} pairs to {target_dist} A (freezing atoms)"
            )

        for ion_idx, water_idx, mol_idx in pairs:
            mol_indices = molecules[mol_idx]
            old_dist = get_current_distance(atoms, ion_idx, water_idx)

            # Move water molecule (ion stays fixed)
            move_water_to_ion_distance(atoms, ion_idx, water_idx, mol_indices, target_dist)
            new_dist = get_current_distance(atoms, ion_idx, water_idx)

            # Freeze ion and entire water molecule - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(ion_idx + 1)
            for atom_idx in mol_indices:
                self.constrained_atoms["frozen_atoms"].add(atom_idx + 1)

            if self.logger:
                self.logger.debug(
                    f"  {ion_element}({ion_idx})-{water_element}({water_idx}): "
                    f"{old_dist:.3f} -> {new_dist:.3f} A"
                )

    def _apply_ion_ion_distance_constraint(self, atoms, constraint) -> None:
        """Apply ion-ion distance constraint - freeze both ions."""
        elem1, elem2 = constraint.element1, constraint.element2
        target_dist = constraint.distance

        # Find ion-ion pairs
        pairs = find_ion_ion_pairs(atoms, elem1, elem2, constraint.count)

        if self.logger:
            self.logger.info(
                f"Applying {elem1}-{elem2} distance constraint: "
                f"{len(pairs)} pairs to {target_dist} A (freezing atoms)"
            )

        for ion1_idx, ion2_idx in pairs:
            old_dist = get_current_distance(atoms, ion1_idx, ion2_idx)

            # Move second ion (first stays fixed)
            move_ion_to_ion_distance(atoms, ion1_idx, ion2_idx, target_dist)
            new_dist = get_current_distance(atoms, ion1_idx, ion2_idx)

            # Freeze both ions - 1-indexed for LAMMPS
            self.constrained_atoms["frozen_atoms"].add(ion1_idx + 1)
            self.constrained_atoms["frozen_atoms"].add(ion2_idx + 1)

            if self.logger:
                self.logger.debug(
                    f"  {elem1}({ion1_idx})-{elem2}({ion2_idx}): "
                    f"{old_dist:.3f} -> {new_dist:.3f} A"
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
            f.write(
                f"# LAMMPS data file for constrained {self.parameters.metal}-{self.parameters.salt_type}-water interface "
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
