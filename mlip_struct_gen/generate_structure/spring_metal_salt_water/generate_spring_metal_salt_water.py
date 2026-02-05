"""Spring-restrained metal-salt-water interface generation."""

import random
import tempfile
from pathlib import Path

from ase import io as ase_io

from ..constrained_metal_salt_water.geometry_modifier import (
    find_ion_ion_pairs,
    find_ion_water_pairs,
    find_metal_ion_pairs,
    find_metal_water_pairs,
    find_nearest_oo_pairs,
    find_oh_bonds,
    find_surface_metal_atoms,
    find_water_molecules,
    get_current_distance,
)
from ..metal_salt_water import MetalSaltWaterGenerator, MetalSaltWaterParameters
from .input_parameters import SpringMetalSaltWaterParameters
from .lammps_input import ELEMENT_MASSES, generate_lammps_input
from .validation import get_lattice_constant, validate_parameters

# Salt info mapping
SALT_INFO = {
    "NaCl": {"cation": "Na", "anion": "Cl"},
    "KCl": {"cation": "K", "anion": "Cl"},
    "LiCl": {"cation": "Li", "anion": "Cl"},
    "CsCl": {"cation": "Cs", "anion": "Cl"},
    "CaCl2": {"cation": "Ca", "anion": "Cl"},
    "MgCl2": {"cation": "Mg", "anion": "Cl"},
    "NaBr": {"cation": "Na", "anion": "Br"},
    "KBr": {"cation": "K", "anion": "Br"},
}


class SpringMetalSaltWaterGenerator:
    """Generate metal-salt-water interfaces with spring bond restraints for MLIP training."""

    def __init__(self, parameters: SpringMetalSaltWaterParameters):
        """
        Initialize spring metal-salt-water generator.

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
        self.salt_info = SALT_INFO.get(self.parameters.salt_type, {"cation": "Na", "anion": "Cl"})

        # Track spring bonds for LAMMPS fix restrain generation
        self.spring_bonds: list[dict] = []

        if self.logger:
            self.logger.info("Initializing SpringMetalSaltWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Surface size: {self.parameters.size}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.6f} A")
            self.logger.info(f"Salt: {self.parameters.salt_type} ({self.parameters.n_salt} units)")

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate spring-restrained metal-salt-water interface.

        Args:
            save_artifacts: Save intermediate files

        Returns:
            Path to generated output file
        """
        if self.logger:
            self.logger.info("Starting spring-restrained metal-salt-water interface generation")

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
                self.logger.step("Loading structure for spring constraint application")

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

            # Step 4: Apply spring constraints
            random.seed(self.parameters.constraint_seed)

            # Metal-water spring constraints
            for constraint in self.parameters.metal_water_spring_constraints:
                self._apply_metal_water_spring_constraint(
                    atoms, molecules, surface_metals, constraint
                )

            # Metal-ion spring constraints
            for constraint in self.parameters.metal_ion_spring_constraints:
                self._apply_metal_ion_spring_constraint(atoms, surface_metals, constraint)

            # General spring constraints (O-H, O-O, Ion-O, Ion-Ion)
            for constraint in self.parameters.spring_constraints:
                self._apply_spring_constraint(atoms, molecules, constraint)

            # Step 5: Write output files
            if self.logger:
                self.logger.step("Writing output files")

            self._write_output(atoms)

        if self.logger:
            self.logger.success("Spring-restrained metal-salt-water interface generation completed")
            self.logger.info(f"Output: {self.parameters.output_file}")

        return self.parameters.output_file

    def _apply_metal_water_spring_constraint(
        self, atoms, molecules, surface_metals, constraint
    ) -> None:
        """Apply metal-water spring constraint."""
        target_dist = constraint.distance
        water_element = constraint.water_element
        k_spring = constraint.k_spring

        if not surface_metals:
            if self.logger:
                self.logger.warning("No surface metal atoms found for constraint")
            return

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
                f"Applying {self.parameters.metal}-{water_element} spring restraint: "
                f"{len(pairs)} pairs, k={k_spring}, r0={target_dist} A"
            )

        for metal_idx, water_idx, _mol_idx in pairs:
            current_dist = get_current_distance(atoms, metal_idx, water_idx)
            self.spring_bonds.append(
                {
                    "atom1_id": metal_idx + 1,
                    "atom2_id": water_idx + 1,
                    "k_spring": k_spring,
                    "distance": target_dist,
                }
            )
            if self.logger:
                self.logger.debug(
                    f"  {self.parameters.metal}({metal_idx})-{water_element}({water_idx}): "
                    f"spring at {current_dist:.3f} A -> r0={target_dist:.3f} A"
                )

    def _apply_metal_ion_spring_constraint(self, atoms, surface_metals, constraint) -> None:
        """Apply metal-ion spring constraint."""
        target_dist = constraint.distance
        ion_element = constraint.ion_element
        k_spring = constraint.k_spring

        if not surface_metals:
            if self.logger:
                self.logger.warning("No surface metal atoms found for constraint")
            return

        pairs = find_metal_ion_pairs(
            atoms,
            self.parameters.metal,
            ion_element,
            surface_metals,
            constraint.count,
        )

        if self.logger:
            self.logger.info(
                f"Applying {self.parameters.metal}-{ion_element} spring restraint: "
                f"{len(pairs)} pairs, k={k_spring}, r0={target_dist} A"
            )

        for metal_idx, ion_idx in pairs:
            current_dist = get_current_distance(atoms, metal_idx, ion_idx)
            self.spring_bonds.append(
                {
                    "atom1_id": metal_idx + 1,
                    "atom2_id": ion_idx + 1,
                    "k_spring": k_spring,
                    "distance": target_dist,
                }
            )
            if self.logger:
                self.logger.debug(
                    f"  {self.parameters.metal}({metal_idx})-{ion_element}({ion_idx}): "
                    f"spring at {current_dist:.3f} A -> r0={target_dist:.3f} A"
                )

    def _apply_spring_constraint(self, atoms, molecules, constraint) -> None:
        """Apply general spring constraint (O-H, O-O, Ion-O, Ion-Ion)."""
        elem1, elem2 = constraint.element1, constraint.element2

        # Determine constraint type
        is_water_water = elem1 in {"O", "H"} and elem2 in {"O", "H"}
        ion_elements = {"Na", "K", "Li", "Cs", "Cl", "Br", "Ca", "Mg"}
        is_ion_water = (elem1 in ion_elements and elem2 in {"O", "H"}) or (
            elem2 in ion_elements and elem1 in {"O", "H"}
        )
        is_ion_ion = elem1 in ion_elements and elem2 in ion_elements

        if is_water_water:
            self._apply_water_spring_constraint(atoms, molecules, constraint)
        elif is_ion_water:
            self._apply_ion_water_spring_constraint(atoms, molecules, constraint)
        elif is_ion_ion:
            self._apply_ion_ion_spring_constraint(atoms, constraint)

    def _apply_water_spring_constraint(self, atoms, molecules, constraint) -> None:
        """Apply water-only spring constraint (O-H or O-O)."""
        elem1, elem2 = constraint.element1, constraint.element2
        target_dist = constraint.distance
        k_spring = constraint.k_spring

        is_intramolecular = (elem1 == "O" and elem2 == "H") or (elem1 == "H" and elem2 == "O")

        if is_intramolecular:
            bonds = find_oh_bonds(atoms, molecules)

            if constraint.count == "all":
                selected_bonds = bonds
            else:
                count = min(int(constraint.count), len(bonds))
                selected_bonds = random.sample(bonds, count)

            if self.logger:
                self.logger.info(
                    f"Applying O-H spring restraint: {len(selected_bonds)} bonds, "
                    f"k={k_spring}, r0={target_dist} A"
                )

            for o_idx, h_idx, mol_idx in selected_bonds:
                current_dist = get_current_distance(atoms, o_idx, h_idx)
                self.spring_bonds.append(
                    {
                        "atom1_id": o_idx + 1,
                        "atom2_id": h_idx + 1,
                        "k_spring": k_spring,
                        "distance": target_dist,
                    }
                )
                if self.logger:
                    self.logger.debug(
                        f"  Molecule {mol_idx}: O-H spring at {current_dist:.3f} A -> r0={target_dist:.3f} A"
                    )
        else:
            pairs = find_nearest_oo_pairs(atoms, molecules, constraint.count)

            if self.logger:
                self.logger.info(
                    f"Applying O-O spring restraint: {len(pairs)} pairs, "
                    f"k={k_spring}, r0={target_dist} A"
                )

            for mol1_idx, mol2_idx in pairs:
                mol1 = molecules[mol1_idx]
                mol2 = molecules[mol2_idx]
                o1_idx, o2_idx = mol1[0], mol2[0]

                current_dist = get_current_distance(atoms, o1_idx, o2_idx)
                self.spring_bonds.append(
                    {
                        "atom1_id": o1_idx + 1,
                        "atom2_id": o2_idx + 1,
                        "k_spring": k_spring,
                        "distance": target_dist,
                    }
                )
                if self.logger:
                    self.logger.debug(
                        f"  Molecules {mol1_idx}-{mol2_idx}: O-O spring at {current_dist:.3f} A -> r0={target_dist:.3f} A"
                    )

    def _apply_ion_water_spring_constraint(self, atoms, molecules, constraint) -> None:
        """Apply ion-water spring constraint."""
        elem1, elem2 = constraint.element1, constraint.element2
        target_dist = constraint.distance
        k_spring = constraint.k_spring

        ion_elements = {"Na", "K", "Li", "Cs", "Cl", "Br", "Ca", "Mg"}
        if elem1 in ion_elements:
            ion_element = elem1
            water_element = elem2
        else:
            ion_element = elem2
            water_element = elem1

        pairs = find_ion_water_pairs(atoms, ion_element, water_element, molecules, constraint.count)

        if self.logger:
            self.logger.info(
                f"Applying {ion_element}-{water_element} spring restraint: "
                f"{len(pairs)} pairs, k={k_spring}, r0={target_dist} A"
            )

        for ion_idx, water_idx, _mol_idx in pairs:
            current_dist = get_current_distance(atoms, ion_idx, water_idx)
            self.spring_bonds.append(
                {
                    "atom1_id": ion_idx + 1,
                    "atom2_id": water_idx + 1,
                    "k_spring": k_spring,
                    "distance": target_dist,
                }
            )
            if self.logger:
                self.logger.debug(
                    f"  {ion_element}({ion_idx})-{water_element}({water_idx}): "
                    f"spring at {current_dist:.3f} A -> r0={target_dist:.3f} A"
                )

    def _apply_ion_ion_spring_constraint(self, atoms, constraint) -> None:
        """Apply ion-ion spring constraint."""
        elem1, elem2 = constraint.element1, constraint.element2
        target_dist = constraint.distance
        k_spring = constraint.k_spring

        pairs = find_ion_ion_pairs(atoms, elem1, elem2, constraint.count)

        if self.logger:
            self.logger.info(
                f"Applying {elem1}-{elem2} spring restraint: "
                f"{len(pairs)} pairs, k={k_spring}, r0={target_dist} A"
            )

        for ion1_idx, ion2_idx in pairs:
            current_dist = get_current_distance(atoms, ion1_idx, ion2_idx)
            self.spring_bonds.append(
                {
                    "atom1_id": ion1_idx + 1,
                    "atom2_id": ion2_idx + 1,
                    "k_spring": k_spring,
                    "distance": target_dist,
                }
            )
            if self.logger:
                self.logger.debug(
                    f"  {elem1}({ion1_idx})-{elem2}({ion2_idx}): "
                    f"spring at {current_dist:.3f} A -> r0={target_dist:.3f} A"
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
            spring_bonds=self.spring_bonds,
            output_file=str(lammps_input_file),
            minimize=self.parameters.minimize,
            ensemble=self.parameters.ensemble,
            elements=self.parameters.elements,
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

        element_to_type = {elem: i + 1 for i, elem in enumerate(self.parameters.elements)}

        with open(output_path, "w") as f:
            f.write(
                f"# LAMMPS data file for spring-restrained {self.parameters.metal}-{self.parameters.salt_type}-water interface "
                f"(atomic style for DeepMD)\n"
            )
            f.write("# Generated by mlip-struct-gen\n\n")

            f.write(f"{n_atoms} atoms\n\n")
            f.write(f"{len(self.parameters.elements)} atom types\n\n")

            f.write(f"0.0 {self.cell[0, 0]:.6f} xlo xhi\n")
            f.write(f"0.0 {self.cell[1, 1]:.6f} ylo yhi\n")
            f.write(f"0.0 {self.cell[2, 2]:.6f} zlo zhi\n\n")

            f.write("Masses\n\n")
            for i, elem in enumerate(self.parameters.elements, 1):
                mass = ELEMENT_MASSES.get(elem, 1.0)
                f.write(f"{i} {mass:.4f}  # {elem}\n")
            f.write("\n")

            f.write("Atoms # atomic\n\n")
            for atom_id, (symbol, pos) in enumerate(zip(symbols, positions, strict=False), 1):
                atom_type = element_to_type.get(symbol, 1)
                f.write(f"{atom_id} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
