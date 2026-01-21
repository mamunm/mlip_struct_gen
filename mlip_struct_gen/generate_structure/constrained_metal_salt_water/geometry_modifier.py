"""Geometry modification functions for constrained metal-salt-water interface generation."""

import numpy as np
from ase import Atoms

# Re-export metal-water functions from constrained_metal_water
from ..constrained_metal_water.geometry_modifier import (
    find_metal_water_angles,
    find_metal_water_pairs,
    find_surface_metal_atoms,
    modify_metal_water_angle,
    move_water_molecule_to_metal_distance,
)

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
    # From constrained_metal_water
    "find_surface_metal_atoms",
    "find_metal_water_pairs",
    "move_water_molecule_to_metal_distance",
    "find_metal_water_angles",
    "modify_metal_water_angle",
    # New for metal-salt-water
    "find_ions",
    "find_metal_ion_pairs",
    "move_ion_to_metal_distance",
    "find_ion_water_pairs",
    "move_water_to_ion_distance",
    "find_ion_ion_pairs",
    "move_ion_to_ion_distance",
]


def find_ions(atoms: Atoms, ion_element: str) -> list[int]:
    """
    Find all ions of a given element in the structure.

    Args:
        atoms: ASE Atoms object
        ion_element: Ion element symbol (e.g., "Na", "Cl")

    Returns:
        List of atom indices for the specified ion
    """
    symbols = atoms.get_chemical_symbols()
    return [i for i, s in enumerate(symbols) if s == ion_element]


def find_metal_ion_pairs(
    atoms: Atoms,
    metal_element: str,
    ion_element: str,
    metal_indices: list[int],
    count: int | str,
) -> list[tuple[int, int]]:
    """
    Find N nearest metal-ion pairs.

    Args:
        atoms: ASE Atoms object
        metal_element: Metal element symbol (e.g., "Pt")
        ion_element: Ion element symbol (e.g., "Na", "Cl")
        metal_indices: List of surface metal atom indices
        count: Number of pairs to find, or "all"

    Returns:
        List of (metal_idx, ion_idx) tuples sorted by distance.
    """
    positions = atoms.get_positions()

    # Find all ions of the specified element
    ion_indices = find_ions(atoms, ion_element)

    if not ion_indices:
        return []

    # Calculate all metal-ion distances
    pairs = []
    for metal_idx in metal_indices:
        metal_pos = positions[metal_idx]
        for ion_idx in ion_indices:
            ion_pos = positions[ion_idx]
            dist = np.linalg.norm(ion_pos - metal_pos)
            pairs.append((dist, metal_idx, ion_idx))

    # Sort by distance
    pairs.sort()

    # Determine how many to return
    n_pairs = len(pairs) if count == "all" else min(int(count), len(pairs))

    return [(metal_idx, ion_idx) for _, metal_idx, ion_idx in pairs[:n_pairs]]


def move_ion_to_metal_distance(
    atoms: Atoms,
    metal_idx: int,
    ion_idx: int,
    target_distance: float,
) -> None:
    """
    Move ion to achieve target metal-ion distance.

    Metal atom stays fixed, ion moves.

    Args:
        atoms: ASE Atoms object (modified in-place)
        metal_idx: Index of metal atom (stays fixed)
        ion_idx: Index of ion atom (moves)
        target_distance: Target distance in Angstroms
    """
    positions = atoms.get_positions()
    metal_pos = positions[metal_idx]
    ion_pos = positions[ion_idx]

    # Vector from metal to ion
    vec = ion_pos - metal_pos
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        raise ValueError(f"Metal atom {metal_idx} and ion {ion_idx} are at the same position")

    unit_vec = vec / current_dist

    # New position for ion
    new_ion_pos = metal_pos + target_distance * unit_vec

    positions[ion_idx] = new_ion_pos
    atoms.set_positions(positions)


def find_ion_water_pairs(
    atoms: Atoms,
    ion_element: str,
    water_element: str,
    molecules: list[list[int]],
    count: int | str,
) -> list[tuple[int, int, int | None]]:
    """
    Find N nearest ion-water atom pairs.

    Args:
        atoms: ASE Atoms object
        ion_element: Ion element symbol (e.g., "Na", "Cl")
        water_element: Water atom element ("O" or "H")
        molecules: List of water molecule indices [O, H1, H2]
        count: Number of pairs to find, or "all"

    Returns:
        List of (ion_idx, water_atom_idx, mol_idx) tuples sorted by distance.
    """
    positions = atoms.get_positions()

    # Find all ions
    ion_indices = find_ions(atoms, ion_element)

    if not ion_indices:
        return []

    # Build list of water atom indices with their molecule info
    water_atoms = []
    for mol_idx, mol in enumerate(molecules):
        o_idx, h1_idx, h2_idx = mol
        if water_element == "O":
            water_atoms.append((o_idx, mol_idx))
        elif water_element == "H":
            water_atoms.append((h1_idx, mol_idx))
            water_atoms.append((h2_idx, mol_idx))

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


def move_water_to_ion_distance(
    atoms: Atoms,
    ion_idx: int,
    water_idx: int,
    mol_indices: list[int],
    target_distance: float,
) -> None:
    """
    Move entire water molecule as rigid body to achieve target ion-water distance.

    Ion stays fixed, water molecule moves.

    Args:
        atoms: ASE Atoms object (modified in-place)
        ion_idx: Index of ion atom (stays fixed)
        water_idx: Index of water atom (O or H) used for distance calculation
        mol_indices: [O_idx, H1_idx, H2_idx] of the water molecule
        target_distance: Target distance in Angstroms
    """
    positions = atoms.get_positions()
    ion_pos = positions[ion_idx]
    water_pos = positions[water_idx]

    # Vector from ion to water atom
    vec = water_pos - ion_pos
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        raise ValueError(f"Ion {ion_idx} and water atom {water_idx} are at the same position")

    unit_vec = vec / current_dist

    # Displacement needed
    displacement = (target_distance - current_dist) * unit_vec

    # Move all atoms of the water molecule
    for idx in mol_indices:
        positions[idx] += displacement

    atoms.set_positions(positions)


def find_ion_ion_pairs(
    atoms: Atoms,
    ion_element1: str,
    ion_element2: str,
    count: int | str,
) -> list[tuple[int, int]]:
    """
    Find N nearest ion-ion pairs between two different ion types.

    Args:
        atoms: ASE Atoms object
        ion_element1: First ion element symbol (e.g., "Na")
        ion_element2: Second ion element symbol (e.g., "Cl")
        count: Number of pairs to find, or "all"

    Returns:
        List of (ion1_idx, ion2_idx) tuples sorted by distance.
    """
    positions = atoms.get_positions()

    # Find all ions of each type
    ion1_indices = find_ions(atoms, ion_element1)
    ion2_indices = find_ions(atoms, ion_element2)

    if not ion1_indices or not ion2_indices:
        return []

    # Calculate all ion1-ion2 distances
    pairs = []
    for ion1_idx in ion1_indices:
        ion1_pos = positions[ion1_idx]
        for ion2_idx in ion2_indices:
            # Skip if same index (when ion_element1 == ion_element2)
            if ion1_idx == ion2_idx:
                continue
            ion2_pos = positions[ion2_idx]
            dist = np.linalg.norm(ion2_pos - ion1_pos)
            pairs.append((dist, ion1_idx, ion2_idx))

    # Sort by distance
    pairs.sort()

    # Determine how many to return
    n_pairs = len(pairs) if count == "all" else min(int(count), len(pairs))

    return [(ion1_idx, ion2_idx) for _, ion1_idx, ion2_idx in pairs[:n_pairs]]


def move_ion_to_ion_distance(
    atoms: Atoms,
    ion1_idx: int,
    ion2_idx: int,
    target_distance: float,
) -> None:
    """
    Move ion2 to achieve target ion1-ion2 distance.

    Ion1 stays fixed, ion2 moves.

    Args:
        atoms: ASE Atoms object (modified in-place)
        ion1_idx: Index of first ion atom (stays fixed)
        ion2_idx: Index of second ion atom (moves)
        target_distance: Target distance in Angstroms
    """
    positions = atoms.get_positions()
    ion1_pos = positions[ion1_idx]
    ion2_pos = positions[ion2_idx]

    # Vector from ion1 to ion2
    vec = ion2_pos - ion1_pos
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        raise ValueError(f"Ions {ion1_idx} and {ion2_idx} are at the same position")

    unit_vec = vec / current_dist

    # New position for ion2
    new_ion2_pos = ion1_pos + target_distance * unit_vec

    positions[ion2_idx] = new_ion2_pos
    atoms.set_positions(positions)
