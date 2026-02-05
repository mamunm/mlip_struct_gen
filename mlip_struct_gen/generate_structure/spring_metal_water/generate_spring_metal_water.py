"""Spring-restrained metal-water interface generation."""

import random
import tempfile
from pathlib import Path

from ase import io as ase_io

from ..constrained_metal_water.geometry_modifier import (
    find_metal_water_pairs,
    find_nearest_oo_pairs,
    find_oh_bonds,
    find_surface_metal_atoms,
    find_water_molecules,
    get_current_distance,
)
from ..metal_water import MetalWaterGenerator, MetalWaterParameters
from .input_parameters import SpringMetalWaterParameters
from .lammps_input import ELEMENT_MASSES, generate_lammps_input
from .validation import get_lattice_constant, validate_parameters


class SpringMetalWaterGenerator:
    """Generate metal-water interfaces with spring bond restraints for MLIP training."""

    def __init__(self, parameters: SpringMetalWaterParameters):
        """
        Initialize spring metal-water generator.

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

        # Track spring bonds for LAMMPS fix restrain generation
        self.spring_bonds: list[dict] = []

        if self.logger:
            self.logger.info("Initializing SpringMetalWaterGenerator")
            self.logger.info(f"Metal: {self.parameters.metal}")
            self.logger.info(f"Surface size: {self.parameters.size}")
            self.logger.info(f"Lattice constant: {self.lattice_constant:.6f} A")

    def run(self, save_artifacts: bool = False) -> str:
        """
        Generate spring-restrained metal-water interface.

        Args:
            save_artifacts: Save intermediate files

        Returns:
            Path to generated output file
        """
        if self.logger:
            self.logger.info("Starting spring-restrained metal-water interface generation")

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
                self.logger.step("Loading structure for spring constraint application")

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

            # Step 4: Apply spring constraints
            random.seed(self.parameters.constraint_seed)

            for constraint in self.parameters.metal_water_spring_constraints:
                self._apply_metal_water_spring_constraint(
                    atoms, molecules, surface_metals, constraint
                )

            for constraint in self.parameters.spring_constraints:
                self._apply_water_spring_constraint(atoms, molecules, constraint)

            # Step 5: Write output files
            if self.logger:
                self.logger.step("Writing output files")

            self._write_output(atoms)

        if self.logger:
            self.logger.success("Spring-restrained metal-water interface generation completed")
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
                f"# LAMMPS data file for spring-restrained {self.parameters.metal}-water interface "
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
