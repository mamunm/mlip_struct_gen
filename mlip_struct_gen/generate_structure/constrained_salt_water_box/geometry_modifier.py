"""Geometry modification functions for constrained salt water box generation."""

import numpy as np
from ase import Atoms

# Re-export water-specific functions from constrained_water_box
from ..constrained_water_box.geometry_modifier import (
    find_hoh_angles,
    find_nearest_oo_pairs,
    find_oh_bonds,
    find_water_molecules,
    get_current_angle,
    get_current_distance,
    modify_angle,
    modify_bond_distance,
    modify_intermolecular_distance,
)

__all__ = [
    # From constrained_water_box
    "find_water_molecules",
    "find_oh_bonds",
    "find_hoh_angles",
    "find_nearest_oo_pairs",
    "modify_bond_distance",
    "modify_angle",
    "modify_intermolecular_distance",
    "get_current_distance",
    "get_current_angle",
    # New for ions
    "find_ions",
    "find_ion_water_pairs",
    "find_ion_pairs",
    "move_ion_to_distance",
]


def find_ions(atoms: Atoms, ion_elements: list[str]) -> dict[str, list[int]]:
    """
    Find all ions in the structure.

    Args:
        atoms: ASE Atoms object
        ion_elements: List of ion element symbols (e.g., ["Na", "Cl"])

    Returns:
        Dict mapping element symbol to list of atom indices
    """
    symbols = atoms.get_chemical_symbols()
    ions = {elem: [] for elem in ion_elements}

    for i, symbol in enumerate(symbols):
        if symbol in ion_elements:
            ions[symbol].append(i)

    return ions


def find_ion_water_pairs(
    atoms: Atoms,
    _ion_element: str,
    water_element: str,
    molecules: list[list[int]],
    ion_indices: list[int],
    count: int | str,
) -> list[tuple[int, int, int | None]]:
    """
    Find N nearest ion-water atom pairs.

    Args:
        atoms: ASE Atoms object
        ion_element: Ion element symbol (e.g., "Na", "Cl")
        water_element: Water atom element (e.g., "O", "H")
        molecules: List of water molecule indices [O, H1, H2]
        ion_indices: List of ion atom indices
        count: Number of pairs to find, or "all"

    Returns:
        List of (ion_idx, water_atom_idx, mol_idx) tuples sorted by distance.
        mol_idx is the molecule index if water_element is "O", None for H.
    """
    positions = atoms.get_positions()

    # Build list of water atom indices with their info
    water_atoms = []
    for mol_idx, mol in enumerate(molecules):
        o_idx, h1_idx, h2_idx = mol
        if water_element == "O":
            water_atoms.append((o_idx, mol_idx))
        elif water_element == "H":
            water_atoms.append((h1_idx, None))
            water_atoms.append((h2_idx, None))

    # Calculate all ion-water distances
    pairs = []
    for ion_idx in ion_indices:
        ion_pos = positions[ion_idx]
        for water_idx, mol_idx in water_atoms:
            water_pos = positions[water_idx]
            dist = np.linalg.norm(water_pos - ion_pos)
            pairs.append((dist, ion_idx, water_idx, mol_idx))

    # Sort by distance
    pairs.sort()

    # Determine how many to return
    n_pairs = len(pairs) if count == "all" else min(int(count), len(pairs))

    return [(ion_idx, water_idx, mol_idx) for _, ion_idx, water_idx, mol_idx in pairs[:n_pairs]]


def find_ion_pairs(
    atoms: Atoms,
    element1: str,
    element2: str,
    indices1: list[int],
    indices2: list[int],
    count: int | str,
) -> list[tuple[int, int]]:
    """
    Find N nearest pairs between two types of ions.

    Args:
        atoms: ASE Atoms object
        element1: First ion element (e.g., "Na")
        element2: Second ion element (e.g., "Cl")
        indices1: List of first ion indices
        indices2: List of second ion indices
        count: Number of pairs to find, or "all"

    Returns:
        List of (idx1, idx2) tuples sorted by distance
    """
    positions = atoms.get_positions()

    # Calculate all pairwise distances
    pairs = []

    if element1 == element2:
        # Same element: avoid self-pairs and duplicates
        for i, idx1 in enumerate(indices1):
            pos1 = positions[idx1]
            for idx2 in indices1[i + 1 :]:
                pos2 = positions[idx2]
                dist = np.linalg.norm(pos2 - pos1)
                pairs.append((dist, idx1, idx2))
    else:
        # Different elements
        for idx1 in indices1:
            pos1 = positions[idx1]
            for idx2 in indices2:
                pos2 = positions[idx2]
                dist = np.linalg.norm(pos2 - pos1)
                pairs.append((dist, idx1, idx2))

    # Sort by distance
    pairs.sort()

    # Determine how many to return
    n_pairs = len(pairs) if count == "all" else min(int(count), len(pairs))

    return [(idx1, idx2) for _, idx1, idx2 in pairs[:n_pairs]]


def move_ion_to_distance(
    atoms: Atoms,
    fixed_idx: int,
    moving_idx: int,
    target_distance: float,
) -> None:
    """
    Move an ion (or any single atom) to achieve target distance from fixed atom.

    Args:
        atoms: ASE Atoms object (modified in-place)
        fixed_idx: Index of fixed atom (stays in place)
        moving_idx: Index of moving atom
        target_distance: Target distance in Angstroms
    """
    positions = atoms.get_positions()
    pos_fixed = positions[fixed_idx]
    pos_moving = positions[moving_idx]

    # Vector from fixed to moving
    vec = pos_moving - pos_fixed
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        raise ValueError(f"Atoms {fixed_idx} and {moving_idx} are at the same position")

    unit_vec = vec / current_dist
    new_pos = pos_fixed + target_distance * unit_vec

    positions[moving_idx] = new_pos
    atoms.set_positions(positions)
