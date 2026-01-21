"""Geometry modification functions for constrained water box generation."""

import numpy as np
from ase import Atoms


def find_water_molecules(atoms: Atoms, oh_cutoff: float = 1.3) -> list[list[int]]:
    """
    Identify water molecules and return atom indices.

    Args:
        atoms: ASE Atoms object
        oh_cutoff: Maximum O-H distance to consider as bonded (Angstroms)

    Returns:
        List of [O_idx, H1_idx, H2_idx] for each water molecule
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Find all O and H indices
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]

    molecules = []
    used_h = set()

    for o_idx in o_indices:
        o_pos = positions[o_idx]
        # Find the two closest H atoms
        h_distances = []
        for h_idx in h_indices:
            if h_idx in used_h:
                continue
            h_pos = positions[h_idx]
            dist = np.linalg.norm(h_pos - o_pos)
            if dist < oh_cutoff:
                h_distances.append((dist, h_idx))

        # Sort by distance and take the two closest
        h_distances.sort()
        if len(h_distances) >= 2:
            h1_idx = h_distances[0][1]
            h2_idx = h_distances[1][1]
            molecules.append([o_idx, h1_idx, h2_idx])
            used_h.add(h1_idx)
            used_h.add(h2_idx)

    return molecules


def find_oh_bonds(atoms: Atoms, molecules: list[list[int]]) -> list[tuple[int, int, int]]:
    """
    Find all O-H bonds in the structure.

    Args:
        atoms: ASE Atoms object
        molecules: List of molecule indices from find_water_molecules

    Returns:
        List of (O_idx, H_idx, mol_idx) tuples
    """
    bonds = []
    for mol_idx, mol in enumerate(molecules):
        o_idx, h1_idx, h2_idx = mol
        bonds.append((o_idx, h1_idx, mol_idx))
        bonds.append((o_idx, h2_idx, mol_idx))
    return bonds


def find_hoh_angles(atoms: Atoms, molecules: list[list[int]]) -> list[tuple[int, int, int, int]]:
    """
    Find all H-O-H angles in the structure.

    Args:
        atoms: ASE Atoms object
        molecules: List of molecule indices from find_water_molecules

    Returns:
        List of (H1_idx, O_idx, H2_idx, mol_idx) tuples
    """
    angles = []
    for mol_idx, mol in enumerate(molecules):
        o_idx, h1_idx, h2_idx = mol
        angles.append((h1_idx, o_idx, h2_idx, mol_idx))
    return angles


def find_nearest_oo_pairs(
    atoms: Atoms, molecules: list[list[int]], count: int | str
) -> list[tuple[int, int]]:
    """
    Find N nearest O-O pairs between different molecules.

    Args:
        atoms: ASE Atoms object
        molecules: List of molecule indices from find_water_molecules
        count: Number of pairs to find, or "all"

    Returns:
        List of (mol1_idx, mol2_idx) sorted by O-O distance
    """
    positions = atoms.get_positions()
    n_mol = len(molecules)

    # Calculate all O-O distances between different molecules
    pairs = []
    for i in range(n_mol):
        o1_idx = molecules[i][0]
        o1_pos = positions[o1_idx]
        for j in range(i + 1, n_mol):
            o2_idx = molecules[j][0]
            o2_pos = positions[o2_idx]
            dist = np.linalg.norm(o2_pos - o1_pos)
            pairs.append((dist, i, j))

    # Sort by distance
    pairs.sort()

    # Determine how many to return
    n_pairs = len(pairs) if count == "all" else min(int(count), len(pairs))

    return [(mol1, mol2) for _, mol1, mol2 in pairs[:n_pairs]]


def modify_bond_distance(
    atoms: Atoms, atom_idx1: int, atom_idx2: int, target_distance: float
) -> None:
    """
    Move atom2 along the bond vector to achieve target distance.

    Args:
        atoms: ASE Atoms object (modified in-place)
        atom_idx1: Index of first atom (stays fixed, e.g., O)
        atom_idx2: Index of second atom (moves, e.g., H)
        target_distance: Target distance in Angstroms
    """
    positions = atoms.get_positions()
    pos1 = positions[atom_idx1]
    pos2 = positions[atom_idx2]

    bond_vec = pos2 - pos1
    current_dist = np.linalg.norm(bond_vec)

    if current_dist < 1e-10:
        raise ValueError(f"Atoms {atom_idx1} and {atom_idx2} are at the same position")

    unit_vec = bond_vec / current_dist
    new_pos = pos1 + target_distance * unit_vec

    positions[atom_idx2] = new_pos
    atoms.set_positions(positions)


def modify_angle(
    atoms: Atoms,
    atom_idx1: int,
    atom_idx2: int,
    atom_idx3: int,
    target_angle: float,
) -> None:
    """
    Rotate atom3 around atom2 to achieve target angle.

    Keeps atom1 and atom2 fixed, rotates atom3.

    Args:
        atoms: ASE Atoms object (modified in-place)
        atom_idx1: Index of first terminal atom (H1, fixed)
        atom_idx2: Index of central atom (O, fixed)
        atom_idx3: Index of second terminal atom (H2, rotates)
        target_angle: Target angle in degrees
    """
    positions = atoms.get_positions()
    pos1 = positions[atom_idx1]
    pos2 = positions[atom_idx2]
    pos3 = positions[atom_idx3]

    # Vectors from central atom
    vec1 = pos1 - pos2
    vec3 = pos3 - pos2

    # Current angle
    len1 = np.linalg.norm(vec1)
    len3 = np.linalg.norm(vec3)

    if len1 < 1e-10 or len3 < 1e-10:
        raise ValueError("Degenerate angle: atoms too close")

    unit1 = vec1 / len1
    unit3 = vec3 / len3

    current_cos = np.clip(np.dot(unit1, unit3), -1.0, 1.0)
    current_angle = np.degrees(np.arccos(current_cos))

    # Angle difference needed
    angle_diff = target_angle - current_angle

    if abs(angle_diff) < 1e-6:
        return  # Already at target angle

    # Rotation axis: perpendicular to the H1-O-H2 plane
    axis = np.cross(unit1, unit3)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        # Vectors are parallel, create arbitrary perpendicular axis
        if abs(unit1[0]) < 0.9:
            axis = np.cross(unit1, np.array([1, 0, 0]))
        else:
            axis = np.cross(unit1, np.array([0, 1, 0]))

    axis = axis / np.linalg.norm(axis)

    # Rotation matrix using Rodrigues' formula
    angle_rad = np.radians(angle_diff)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    k_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot_matrix = np.eye(3) + s * k_matrix + (1 - c) * (k_matrix @ k_matrix)

    # Rotate vec3 around the axis
    new_vec3 = rot_matrix @ vec3
    new_pos3 = pos2 + new_vec3

    positions[atom_idx3] = new_pos3
    atoms.set_positions(positions)


def modify_intermolecular_distance(
    atoms: Atoms,
    mol1_indices: list[int],
    mol2_indices: list[int],
    target_distance: float,
) -> None:
    """
    Move molecule 2 as rigid body to achieve target O-O distance.

    Args:
        atoms: ASE Atoms object (modified in-place)
        mol1_indices: [O_idx, H1_idx, H2_idx] for molecule 1 (fixed)
        mol2_indices: [O_idx, H1_idx, H2_idx] for molecule 2 (moves)
        target_distance: Target O-O distance in Angstroms
    """
    positions = atoms.get_positions()

    # Reference atoms (O atoms)
    ref1_idx = mol1_indices[0]
    ref2_idx = mol2_indices[0]

    pos1 = positions[ref1_idx]
    pos2 = positions[ref2_idx]

    # Current distance and direction
    vec = pos2 - pos1
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        raise ValueError("O atoms are at the same position")

    unit_vec = vec / current_dist

    # Displacement needed for molecule 2
    displacement = (target_distance - current_dist) * unit_vec

    # Move all atoms of molecule 2
    for idx in mol2_indices:
        positions[idx] += displacement

    atoms.set_positions(positions)


def get_current_distance(atoms: Atoms, idx1: int, idx2: int) -> float:
    """Get current distance between two atoms."""
    positions = atoms.get_positions()
    return float(np.linalg.norm(positions[idx2] - positions[idx1]))


def get_current_angle(atoms: Atoms, idx1: int, idx2: int, idx3: int) -> float:
    """Get current angle in degrees (idx2 is central atom)."""
    positions = atoms.get_positions()
    vec1 = positions[idx1] - positions[idx2]
    vec3 = positions[idx3] - positions[idx2]

    len1 = np.linalg.norm(vec1)
    len3 = np.linalg.norm(vec3)

    if len1 < 1e-10 or len3 < 1e-10:
        return 0.0

    cos_angle = np.clip(np.dot(vec1, vec3) / (len1 * len3), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
