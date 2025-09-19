"""Utilities for handling periodic boundary conditions."""


import numpy as np


def minimum_image_distance(
    pos1: np.ndarray, pos2: np.ndarray, cell: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Calculate the minimum image distance between two positions under PBC.

    Args:
        pos1: Position of first point (3,)
        pos2: Position of second point (3,)
        cell: Cell matrix (3, 3)

    Returns:
        Tuple of (distance, vector from pos1 to pos2)
    """
    inv_cell = np.linalg.inv(cell)

    delta = pos2 - pos1

    delta_frac = np.dot(delta, inv_cell)

    delta_frac = delta_frac - np.round(delta_frac)

    delta_cart = np.dot(delta_frac, cell)

    distance = np.linalg.norm(delta_cart)

    return distance, delta_cart


def find_k_nearest_neighbors(
    central_pos: np.ndarray, neighbor_positions: np.ndarray, cell: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors under periodic boundary conditions.

    Args:
        central_pos: Position of central atom (3,)
        neighbor_positions: Array of neighbor positions (N, 3)
        cell: Cell matrix (3, 3)
        k: Number of nearest neighbors to find

    Returns:
        Tuple of (indices, distances, vectors) for k nearest neighbors
    """
    n_neighbors = len(neighbor_positions)
    distances = np.zeros(n_neighbors)
    vectors = np.zeros((n_neighbors, 3))

    for i, pos in enumerate(neighbor_positions):
        dist, vec = minimum_image_distance(central_pos, pos, cell)
        distances[i] = dist
        vectors[i] = vec

    indices = np.argsort(distances)[:k]

    return indices, distances[indices], vectors[indices]


def apply_pbc(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Apply periodic boundary conditions to positions.

    Args:
        positions: Atomic positions (N, 3)
        cell: Cell matrix (3, 3)

    Returns:
        Positions wrapped into the unit cell
    """
    inv_cell = np.linalg.inv(cell)

    frac_coords = np.dot(positions, inv_cell)

    frac_coords = frac_coords - np.floor(frac_coords)

    return np.dot(frac_coords, cell)
