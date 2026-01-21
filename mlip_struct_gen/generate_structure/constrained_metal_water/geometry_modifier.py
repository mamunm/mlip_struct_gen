"""Geometry modification functions for constrained metal-water interface generation."""

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
    # New for metal-water
    "find_surface_metal_atoms",
    "find_metal_water_pairs",
    "move_water_molecule_to_metal_distance",
    "find_metal_water_angles",
    "modify_metal_water_angle",
]


def find_surface_metal_atoms(
    atoms: Atoms,
    metal_element: str,
    layer_tolerance: float = 0.5,
    n_layers: int = 1,
) -> list[int]:
    """
    Identify top-layer metal atoms by z-coordinate clustering.

    Args:
        atoms: ASE Atoms object
        metal_element: Metal element symbol (e.g., "Pt", "Cu")
        layer_tolerance: Tolerance for grouping atoms into layers (Angstroms)
        n_layers: Number of top layers to include (default: 1)

    Returns:
        List of atom indices in the top n layers
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Get all metal atom indices and z-coordinates
    metal_indices = [i for i, s in enumerate(symbols) if s == metal_element]

    if not metal_indices:
        return []

    metal_z = [positions[i][2] for i in metal_indices]

    # Cluster z-coordinates into layers
    sorted_z = sorted(set(metal_z))
    layers = []
    current_layer = [sorted_z[0]]

    for z in sorted_z[1:]:
        if z - current_layer[-1] < layer_tolerance:
            current_layer.append(z)
        else:
            layers.append(current_layer)
            current_layer = [z]
    layers.append(current_layer)

    # Get z-values for the top n layers
    top_layers = layers[-n_layers:]
    top_z_values = set()
    for layer in top_layers:
        top_z_values.update(layer)

    # Find atoms in top layers
    surface_atoms = []
    for idx in metal_indices:
        z = positions[idx][2]
        # Check if this z is in any top layer (within tolerance)
        for layer_z in top_z_values:
            if abs(z - layer_z) < layer_tolerance:
                surface_atoms.append(idx)
                break

    return surface_atoms


def find_metal_water_pairs(
    atoms: Atoms,
    metal_element: str,
    water_element: str,
    molecules: list[list[int]],
    metal_indices: list[int],
    count: int | str,
) -> list[tuple[int, int, int | None]]:
    """
    Find N nearest metal-water atom pairs.

    Args:
        atoms: ASE Atoms object
        metal_element: Metal element symbol (e.g., "Pt")
        water_element: Water atom element ("O" or "H")
        molecules: List of water molecule indices [O, H1, H2]
        metal_indices: List of surface metal atom indices
        count: Number of pairs to find, or "all"

    Returns:
        List of (metal_idx, water_atom_idx, mol_idx) tuples sorted by distance.
        mol_idx is the molecule index for reference.
    """
    positions = atoms.get_positions()

    # Build list of water atom indices with their molecule info
    water_atoms = []
    for mol_idx, mol in enumerate(molecules):
        o_idx, h1_idx, h2_idx = mol
        if water_element == "O":
            water_atoms.append((o_idx, mol_idx))
        elif water_element == "H":
            water_atoms.append((h1_idx, mol_idx))
            water_atoms.append((h2_idx, mol_idx))

    # Calculate all metal-water distances
    pairs = []
    for metal_idx in metal_indices:
        metal_pos = positions[metal_idx]
        for water_idx, mol_idx in water_atoms:
            water_pos = positions[water_idx]
            dist = np.linalg.norm(water_pos - metal_pos)
            pairs.append((dist, metal_idx, water_idx, mol_idx))

    # Sort by distance
    pairs.sort()

    # Determine how many to return
    n_pairs = len(pairs) if count == "all" else min(int(count), len(pairs))

    return [(metal_idx, water_idx, mol_idx) for _, metal_idx, water_idx, mol_idx in pairs[:n_pairs]]


def move_water_molecule_to_metal_distance(
    atoms: Atoms,
    metal_idx: int,
    water_idx: int,
    mol_indices: list[int],
    target_distance: float,
) -> None:
    """
    Move entire water molecule as rigid body to achieve target metal-water distance.

    Metal atom stays fixed, water molecule moves.

    Args:
        atoms: ASE Atoms object (modified in-place)
        metal_idx: Index of metal atom (stays fixed)
        water_idx: Index of water atom (O or H) used for distance calculation
        mol_indices: [O_idx, H1_idx, H2_idx] of the water molecule
        target_distance: Target distance in Angstroms
    """
    positions = atoms.get_positions()
    metal_pos = positions[metal_idx]
    water_pos = positions[water_idx]

    # Vector from metal to water atom
    vec = water_pos - metal_pos
    current_dist = np.linalg.norm(vec)

    if current_dist < 1e-10:
        raise ValueError(
            f"Metal atom {metal_idx} and water atom {water_idx} are at the same position"
        )

    unit_vec = vec / current_dist

    # Displacement needed
    displacement = (target_distance - current_dist) * unit_vec

    # Move all atoms of the water molecule
    for idx in mol_indices:
        positions[idx] += displacement

    atoms.set_positions(positions)


def find_metal_water_angles(
    atoms: Atoms,
    metal_element: str,
    molecules: list[list[int]],
    metal_indices: list[int],
    count: int | str,
) -> list[tuple[int, int, int, int, int]]:
    """
    Find Metal-O-H angles for constraint application.

    For each water molecule near the metal surface, finds the closest metal atom
    and returns angles for both M-O-H1 and M-O-H2.

    Args:
        atoms: ASE Atoms object
        metal_element: Metal element symbol
        molecules: List of water molecule indices [O, H1, H2]
        metal_indices: List of surface metal atom indices
        count: Number of angles to find, or "all"

    Returns:
        List of (metal_idx, o_idx, h_idx, mol_idx, h_num) tuples sorted by Metal-O distance.
        h_num is 1 or 2 indicating which hydrogen.
    """
    positions = atoms.get_positions()

    # For each water molecule, find the closest metal atom
    angles = []
    for mol_idx, mol in enumerate(molecules):
        o_idx, h1_idx, h2_idx = mol
        o_pos = positions[o_idx]

        # Find closest metal atom to this oxygen
        min_dist = float("inf")
        closest_metal = None
        for metal_idx in metal_indices:
            metal_pos = positions[metal_idx]
            dist = np.linalg.norm(o_pos - metal_pos)
            if dist < min_dist:
                min_dist = dist
                closest_metal = metal_idx

        if closest_metal is not None:
            # Add both M-O-H angles for this molecule
            angles.append((min_dist, closest_metal, o_idx, h1_idx, mol_idx, 1))
            angles.append((min_dist, closest_metal, o_idx, h2_idx, mol_idx, 2))

    # Sort by Metal-O distance
    angles.sort()

    # Determine how many to return
    n_angles = len(angles) if count == "all" else min(int(count), len(angles))

    return [
        (metal_idx, o_idx, h_idx, mol_idx, h_num)
        for _, metal_idx, o_idx, h_idx, mol_idx, h_num in angles[:n_angles]
    ]


def modify_metal_water_angle(
    atoms: Atoms,
    metal_idx: int,
    o_idx: int,
    h_idx: int,
    mol_indices: list[int],
    target_angle: float,
) -> None:
    """
    Rotate water molecule around Metal-O axis to achieve target Metal-O-H angle.

    The water molecule rotates as a rigid body around the Metal-O bond axis.
    The oxygen position stays fixed, only the hydrogens move.

    Args:
        atoms: ASE Atoms object (modified in-place)
        metal_idx: Index of metal atom
        o_idx: Index of oxygen atom (center of angle)
        h_idx: Index of hydrogen atom to achieve target angle with
        mol_indices: [O_idx, H1_idx, H2_idx] of the water molecule
        target_angle: Target Metal-O-H angle in degrees
    """
    positions = atoms.get_positions()
    metal_pos = positions[metal_idx]
    o_pos = positions[o_idx]
    h_pos = positions[h_idx]

    # Vectors from O
    vec_mo = metal_pos - o_pos  # O -> Metal
    vec_oh = h_pos - o_pos  # O -> H

    len_mo = np.linalg.norm(vec_mo)
    len_oh = np.linalg.norm(vec_oh)

    if len_mo < 1e-10 or len_oh < 1e-10:
        raise ValueError("Degenerate angle: atoms too close")

    unit_mo = vec_mo / len_mo
    unit_oh = vec_oh / len_oh

    # Current angle
    current_cos = np.clip(np.dot(unit_mo, unit_oh), -1.0, 1.0)
    current_angle = np.degrees(np.arccos(current_cos))

    # Angle difference needed
    angle_diff = target_angle - current_angle

    if abs(angle_diff) < 1e-6:
        return  # Already at target angle

    # Rotation axis: perpendicular to the M-O-H plane
    axis = np.cross(unit_mo, unit_oh)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        # Vectors are parallel/antiparallel, create arbitrary perpendicular axis
        if abs(unit_mo[0]) < 0.9:
            axis = np.cross(unit_mo, np.array([1, 0, 0]))
        else:
            axis = np.cross(unit_mo, np.array([0, 1, 0]))

    axis = axis / np.linalg.norm(axis)

    # Rotation matrix using Rodrigues' formula
    angle_rad = np.radians(angle_diff)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    k_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot_matrix = np.eye(3) + s * k_matrix + (1 - c) * (k_matrix @ k_matrix)

    # Rotate both hydrogens around O to maintain water geometry
    h1_idx, h2_idx = mol_indices[1], mol_indices[2]

    for hidx in [h1_idx, h2_idx]:
        h_rel = positions[hidx] - o_pos
        new_h_rel = rot_matrix @ h_rel
        positions[hidx] = o_pos + new_h_rel

    atoms.set_positions(positions)
