#!/usr/bin/env python3
"""
MLIP SR-LR Converter: LAMMPS data file processor for short-range/long-range splitting.
Duplicates specified atom types and creates bonds for ML interatomic potential applications.
"""

import argparse
import sys

from ..utils.logger import get_logger

# Atomic masses for common elements (in amu)
ELEMENT_MASSES = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.922,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98,
    "Ru": 101.07,
    "Rh": 102.906,
    "Pd": 106.42,
    "Ag": 107.868,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.904,
    "Xe": 131.293,
    "Cs": 132.905,
    "Ba": 137.327,
    "La": 138.905,
    "Ce": 140.116,
    "Pr": 140.908,
    "Nd": 144.242,
    "Pm": 145,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.925,
    "Dy": 162.500,
    "Ho": 164.930,
    "Er": 167.259,
    "Tm": 168.934,
    "Yb": 173.054,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.948,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.967,
    "Hg": 200.592,
    "Tl": 204.383,
    "Pb": 207.2,
    "Bi": 208.980,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226,
    "Ac": 227,
    "Th": 232.038,
    "Pa": 231.036,
    "U": 238.029,
}


def parse_duplication_spec(duplicate_args: list[str]) -> list[tuple[int, int, int]]:
    """
    Parse duplication specifications from command line arguments.

    Args:
        duplicate_args: List of strings in format "orig:new:bond"

    Returns:
        List of tuples (original_type, new_type, bond_type)
    """
    duplication_config = []

    if not duplicate_args:
        return duplication_config

    for spec in duplicate_args:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid duplication spec: {spec}. Expected format: 'orig:new:bond'")

        try:
            orig_type = int(parts[0])
            new_type = int(parts[1])
            bond_type = int(parts[2])
            duplication_config.append((orig_type, new_type, bond_type))
        except ValueError as err:
            raise ValueError(
                f"Invalid duplication spec: {spec}. All values must be integers."
            ) from err

    return duplication_config


def parse_charge_map(
    charge_args: list[str], duplication_config: list[tuple[int, int, int]], max_original_type: int
) -> dict[int, float]:
    """
    Parse charge specifications with support for inheritance.

    Args:
        charge_args: List of charge specifications ("type:charge" or positional)
        duplication_config: Duplication configuration
        max_original_type: Maximum atom type in original data

    Returns:
        Dictionary mapping type_id to charge
    """
    charges = {}

    if not charge_args:
        return charges

    # Check if all args are in type:charge format
    explicit_format = all(":" in arg for arg in charge_args)

    if explicit_format:
        # Explicit type:charge format
        for arg in charge_args:
            parts = arg.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid charge spec: {arg}. Expected format: 'type:charge'")

            try:
                type_id = int(parts[0])
                charge = float(parts[1])
                charges[type_id] = charge
            except ValueError as err:
                raise ValueError(
                    f"Invalid charge spec: {arg}. Type must be integer, charge must be float."
                ) from err
    else:
        # Positional format - assign to types 1, 2, 3, ...
        for i, arg in enumerate(charge_args, 1):
            try:
                charges[i] = float(arg)
            except ValueError as err:
                raise ValueError(f"Invalid charge value: {arg}. Must be a float.") from err

    # Apply inheritance for duplicated types
    for orig, new, _ in duplication_config:
        if new not in charges and orig in charges:
            charges[new] = charges[orig]  # Inherit from original

    return charges


class LAMMPSDataProcessor:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        type_map: list[str],
        duplication_config: list[tuple[int, int, int]] = None,
        charge_map: dict[int, float] = None,
        bond_length: float = 0.0,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.type_map = type_map
        self.duplication_config = duplication_config or []
        self.charge_map = charge_map or {}
        self.bond_length = bond_length
        self.logger = get_logger()

        # Build mappings from duplication config
        self.type_mapping = {orig: new for orig, new, _ in self.duplication_config}
        self.bond_type_mapping = {orig: bond for orig, _, bond in self.duplication_config}

        # Data storage
        self.atoms = []
        self.box_bounds = {}
        self.header_info = {}

    def read_input_file(self):
        """Read and parse the input LAMMPS data file."""
        with open(self.input_file) as f:
            lines = f.readlines()

        # Parse header
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "atoms" in line and "atom types" not in line:
                self.header_info["atoms"] = int(line.split()[0])
            elif "atom types" in line:
                self.header_info["atom_types"] = int(line.split()[0])
            elif "xlo xhi" in line:
                parts = line.split()
                self.box_bounds["xlo"] = float(parts[0])
                self.box_bounds["xhi"] = float(parts[1])
            elif "ylo yhi" in line:
                parts = line.split()
                self.box_bounds["ylo"] = float(parts[0])
                self.box_bounds["yhi"] = float(parts[1])
            elif "zlo zhi" in line:
                parts = line.split()
                self.box_bounds["zlo"] = float(parts[0])
                self.box_bounds["zhi"] = float(parts[1])
            elif "xy xz yz" in line:
                parts = line.split()
                self.box_bounds["xy"] = float(parts[0])
                self.box_bounds["xz"] = float(parts[1])
                self.box_bounds["yz"] = float(parts[2])
            elif line.startswith("Atoms"):
                # Find the start of atoms section
                atoms_start = i + 2  # Skip the "Atoms" line and empty line
                break

        # Parse atoms
        for i in range(atoms_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            if line.startswith("Bonds") or line.startswith("Velocities"):
                break

            parts = line.split()
            if len(parts) >= 5:
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                self.atoms.append([atom_id, atom_type, x, y, z])

    def validate_configuration(self):
        """Validate the duplication configuration against existing atom types."""
        if not self.duplication_config:
            return  # No duplication, nothing to validate

        # Get existing atom types
        existing_types = {atom[1] for atom in self.atoms}

        # Check for conflicts
        new_types = set()
        for orig, new, _ in self.duplication_config:
            # Check if original type exists
            if orig not in existing_types:
                self.logger.warning(
                    f"Original type {orig} not found in data file. No atoms will be duplicated for this type."
                )

            # Check for duplicate new types
            if new in new_types:
                raise ValueError(
                    f"Error: New type {new} specified multiple times in duplication config."
                )
            new_types.add(new)

            # Check if new type already exists
            if new in existing_types:
                raise ValueError(
                    f"Error: New type {new} already exists in the data file. Choose a different type number."
                )

    def duplicate_atoms(self) -> tuple[list, list]:
        """Duplicate atoms of specified types and create bonds."""
        duplicated_atoms = []
        bonds = []

        if not self.duplication_config:
            return duplicated_atoms, bonds

        # Get the maximum atom ID
        max_atom_id = max(atom[0] for atom in self.atoms) if self.atoms else 0
        bond_id = 0

        for atom in self.atoms:
            atom_id, atom_type, x, y, z = atom

            if atom_type in self.type_mapping:
                # Create a duplicate with new type
                new_atom_id = max_atom_id + 1
                new_atom_type = self.type_mapping[atom_type]

                # Apply displacement if specified
                if self.bond_length > 0:
                    # Add small random displacement in z direction (can be customized)
                    new_x = x
                    new_y = y
                    new_z = z + self.bond_length
                else:
                    # Perfect overlap
                    new_x, new_y, new_z = x, y, z

                duplicated_atoms.append([new_atom_id, new_atom_type, new_x, new_y, new_z])

                # Create a bond between original and duplicate
                bond_id += 1
                bond_type = self.bond_type_mapping[atom_type]
                bonds.append([bond_id, bond_type, atom_id, new_atom_id])

                max_atom_id += 1

        return duplicated_atoms, bonds

    def get_all_atom_types(self, duplicated_atoms: list) -> set:
        """Get all unique atom types (original + new)."""
        all_types = {atom[1] for atom in self.atoms}
        all_types.update(atom[1] for atom in duplicated_atoms)
        return all_types

    def get_masses(self, all_types: set) -> dict[int, float]:
        """Get masses for all atom types based on the type map."""
        masses = {}

        # Map original types from type_map
        for i, element in enumerate(self.type_map):
            type_id = i + 1
            if type_id in all_types:
                if element in ELEMENT_MASSES:
                    masses[type_id] = ELEMENT_MASSES[element]
                else:
                    self.logger.warning(f"Unknown element {element}, using default mass 1.0")
                    masses[type_id] = 1.0

        # Handle duplicated types (inherit mass from original)
        for orig, new, _ in self.duplication_config:
            if new in all_types:
                if orig in masses:
                    masses[new] = masses[orig]
                else:
                    # Try to get from type_map
                    if orig <= len(self.type_map):
                        element = self.type_map[orig - 1]
                        masses[new] = ELEMENT_MASSES.get(element, 1.0)
                    else:
                        masses[new] = 1.0
                        self.logger.warning(f"Could not determine mass for type {new}, using 1.0")

        return masses

    def get_charges(self, all_types: set) -> dict[int, float]:
        """Get charges for all atom types."""
        charges = {}

        # Assign charges for all types
        for type_id in all_types:
            charges[type_id] = self.charge_map.get(type_id, 0.0)

        return charges

    def write_output_file(self, duplicated_atoms: list, bonds: list):
        """Write the modified LAMMPS data file with atom style full."""
        all_atoms = self.atoms + duplicated_atoms
        total_atoms = len(all_atoms)
        total_bonds = len(bonds)

        # Get all unique atom types and bond types
        all_types = self.get_all_atom_types(duplicated_atoms)
        total_atom_types = len(all_types)

        bond_types = {bond[1] for bond in bonds} if bonds else set()
        total_bond_types = len(bond_types) if bond_types else 0

        masses = self.get_masses(all_types)
        charges = self.get_charges(all_types)

        with open(self.output_file, "w") as f:
            # Write header
            f.write("LAMMPS data file created by MLIP SR-LR converter\n\n")
            f.write(f"{total_atoms} atoms\n")
            if total_bonds > 0:
                f.write(f"{total_bonds} bonds\n")
            f.write(f"{total_atom_types} atom types\n")
            if total_bond_types > 0:
                f.write(f"{total_bond_types} bond types\n")
            f.write("\n")

            # Write box bounds
            f.write(f"{self.box_bounds['xlo']:.10f}   {self.box_bounds['xhi']:.10f} xlo xhi\n")
            f.write(f"{self.box_bounds['ylo']:.10f}   {self.box_bounds['yhi']:.10f} ylo yhi\n")
            f.write(f"{self.box_bounds['zlo']:.10f}   {self.box_bounds['zhi']:.10f} zlo zhi\n")
            if "xy" in self.box_bounds:
                f.write(
                    f"{self.box_bounds['xy']:.10f}   {self.box_bounds['xz']:.10f}   "
                    f"{self.box_bounds['yz']:.10f} xy xz yz\n"
                )
            f.write("\n")

            # Write masses
            f.write("Masses\n\n")
            for type_id in sorted(masses.keys()):
                f.write(f"{type_id} {masses[type_id]:.6f}\n")
            f.write("\n")

            # Write atoms (atom style full: atom-ID molecule-ID atom-type charge x y z)
            f.write("Atoms # full\n\n")
            for atom in all_atoms:
                atom_id, atom_type, x, y, z = atom
                molecule_id = atom_id  # Molecule ID same as atom ID
                charge = charges[atom_type]
                f.write(
                    f"{atom_id:8d} {molecule_id:8d} {atom_type:4d} {charge:12.6f} "
                    f"{x:16.10f} {y:16.10f} {z:16.10f}\n"
                )
            f.write("\n")

            # Write bonds
            if bonds:
                f.write("Bonds\n\n")
                for bond in bonds:
                    bond_id, bond_type, atom1, atom2 = bond
                    f.write(f"{bond_id:8d} {bond_type:4d} {atom1:8d} {atom2:8d}\n")
                f.write("\n")

    def process(self):
        """Main processing function."""
        self.logger.step(f"Reading input file: {self.input_file}")
        self.read_input_file()

        self.logger.info(f"Found {len(self.atoms)} atoms in input file")

        # Get atom type statistics
        type_counts = {}
        for atom in self.atoms:
            atom_type = atom[1]
            type_counts[atom_type] = type_counts.get(atom_type, 0) + 1

        self.logger.info("Atom type distribution:")
        for type_id in sorted(type_counts.keys()):
            element = ""
            if type_id <= len(self.type_map):
                element = f" ({self.type_map[type_id-1]})"
            self.logger.info(f"  Type {type_id}{element}: {type_counts[type_id]} atoms")

        # Validate configuration
        self.validate_configuration()

        if self.duplication_config:
            self.logger.info("Duplication configuration:")
            for orig, new, bond in self.duplication_config:
                orig_count = type_counts.get(orig, 0)
                self.logger.info(
                    f"  Type {orig} → Type {new} (bond type {bond}): {orig_count} atoms to duplicate"
                )

            duplicated_atoms, bonds = self.duplicate_atoms()

            self.logger.info(f"Created {len(duplicated_atoms)} duplicate atoms")
            self.logger.info(f"Created {len(bonds)} bonds")

            # Count bonds by type
            if bonds:
                bond_counts = {}
                for bond in bonds:
                    bond_type = bond[1]
                    bond_counts[bond_type] = bond_counts.get(bond_type, 0) + 1

                self.logger.info("Bond statistics:")
                for bond_type in sorted(bond_counts.keys()):
                    # Find which duplication created this bond type
                    for orig, new, bt in self.duplication_config:
                        if bt == bond_type:
                            self.logger.info(
                                f"  Bond type {bond_type} ({orig}-{new} pairs): {bond_counts[bond_type]} bonds"
                            )
                            break
        else:
            self.logger.info("No duplication requested. Converting to atom style full format.")
            duplicated_atoms = []
            bonds = []

        # Display charge assignments if any
        all_types = self.get_all_atom_types(duplicated_atoms)
        charges = self.get_charges(all_types)

        if any(charge != 0.0 for charge in charges.values()):
            self.logger.info("Charge assignments:")
            for type_id in sorted(charges.keys()):
                element = ""
                if type_id <= len(self.type_map):
                    element = f" ({self.type_map[type_id-1]})"
                elif type_id in [new for _, new, _ in self.duplication_config]:
                    # Find original type for this duplicate
                    for orig, new, _ in self.duplication_config:
                        if new == type_id and orig <= len(self.type_map):
                            element = f" (duplicate of {self.type_map[orig-1]})"
                            break
                self.logger.info(f"  Type {type_id}{element}: {charges[type_id]:+.3f}")

        self.logger.step(f"Writing output file: {self.output_file}")
        self.write_output_file(duplicated_atoms, bonds)

        self.logger.success("Processing complete!")
        self.logger.info(f"Total atoms: {len(self.atoms) + len(duplicated_atoms)}")
        self.logger.info(f"Total bonds: {len(bonds)}")
        self.logger.info(f"Total atom types: {len(all_types)}")
        if bonds:
            self.logger.info(f"Total bond types: {len({bond[1] for bond in bonds})}")


def main():
    parser = argparse.ArgumentParser(
        description="MLIP SR-LR Converter: Process LAMMPS data files for short-range/long-range splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Duplicate only type 2 as type 4 with bond type 1
  %(prog)s input.data output.data --type-map C H O --duplicate "2:4:1"

  # Multiple duplications with charges
  %(prog)s input.data output.data --type-map Pt O H Na Cl \\
    --duplicate "2:6:1" "4:7:2" "5:8:3" \\
    --charge-map "1:0" "2:-2" "3:1" "4:1" "5:-1"

  # No duplication, just reformat with charges
  %(prog)s input.data output.data --type-map C H O --charge-map 0.0 0.5 -0.5

  # Duplicate with bond displacement
  %(prog)s input.data output.data --type-map C H O \\
    --duplicate "2:4:1" --bond-length 1.5

Notes:
  - Duplication format: "original_type:new_type:bond_type"
  - Charge format: "type:charge" or positional values
  - Duplicated atoms inherit charges from originals unless overridden
  - Output is in LAMMPS atom style full format
        """,
    )
    parser.add_argument("input_file", help="Input LAMMPS data file")
    parser.add_argument("output_file", help="Output LAMMPS data file")
    parser.add_argument(
        "--type-map",
        nargs="+",
        required=True,
        help="Element names for atom types (e.g., Pt O H Na Cl)",
    )
    parser.add_argument(
        "--duplicate",
        nargs="*",
        default=[],
        help='Duplication specs: "orig:new:bond" (e.g., "2:6:1" "4:7:2")',
    )
    parser.add_argument(
        "--charge-map",
        nargs="*",
        default=[],
        help='Charge specs: "type:charge" or positional (e.g., "1:0" "2:-1" or 0 -1 1)',
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.0,
        help="Displacement for duplicated atoms in Angstroms (default: 0.0)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without processing",
    )

    args = parser.parse_args()

    try:
        # Parse duplication configuration
        duplication_config = parse_duplication_spec(args.duplicate)

        # Get max type from input for charge parsing
        # We'll read the file briefly just to get this info
        temp_processor = LAMMPSDataProcessor(
            args.input_file, args.output_file, args.type_map, [], {}
        )
        temp_processor.read_input_file()
        max_type = (
            max(atom[1] for atom in temp_processor.atoms)
            if temp_processor.atoms
            else len(args.type_map)
        )

        # Parse charge map
        charge_map = parse_charge_map(args.charge_map, duplication_config, max_type)

        # Create processor with full configuration
        processor = LAMMPSDataProcessor(
            args.input_file,
            args.output_file,
            args.type_map,
            duplication_config,
            charge_map,
            args.bond_length,
        )

        if args.validate_only:
            processor.read_input_file()
            processor.validate_configuration()
            logger = get_logger()
            logger.success("Configuration validated successfully!")
        else:
            processor.process()

    except Exception as e:
        logger = get_logger()
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
