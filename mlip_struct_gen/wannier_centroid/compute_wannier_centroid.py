"""Compute wannier centroids from wannier centers and atomic positions."""

from pathlib import Path

import numpy as np

from .pbc_utils import find_k_nearest_neighbors


def parse_poscar(poscar_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Parse POSCAR file to get cell dimensions and atomic positions.

    Args:
        poscar_path: Path to POSCAR file

    Returns:
        Tuple of (cell matrix, atom symbols, positions)
    """
    with open(poscar_path) as f:
        lines = f.readlines()

    scale = float(lines[1].strip())

    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[2 + i].split()]
    cell *= scale

    atom_types = lines[5].split()
    atom_counts = [int(x) for x in lines[6].split()]

    atom_symbols = []
    for atom_type, count in zip(atom_types, atom_counts, strict=False):
        atom_symbols.extend([atom_type] * count)

    coord_type = lines[7].strip()[0].upper()

    n_atoms = sum(atom_counts)
    positions = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        pos = [float(x) for x in lines[8 + i].split()[:3]]
        positions[i] = pos

    if coord_type in ["D", "F"]:
        positions = np.dot(positions, cell)

    return cell, atom_symbols, positions


def parse_wannier_centers(wannier_path: Path) -> np.ndarray:
    """
    Parse wannier90_centres.xyz file to get wannier center positions.

    Handles format with 'X' as the wannier center label.

    Args:
        wannier_path: Path to wannier90_centres.xyz file

    Returns:
        Array of wannier center positions (N, 3)
    """
    with open(wannier_path) as f:
        lines = f.readlines()

    n_centers = int(lines[0].strip())

    positions = []
    wannier_count = 0
    for i in range(2, len(lines)):
        parts = lines[i].split()
        if len(parts) >= 4 and parts[0] == "X":
            pos = [float(parts[1]), float(parts[2]), float(parts[3])]
            positions.append(pos)
            wannier_count += 1
            if wannier_count >= n_centers:
                break

    return np.array(positions)


def compute_wannier_centroid(folder_path: Path, verbose: bool = False) -> dict:
    """
    Compute wannier centroids for atoms in the system.

    Args:
        folder_path: Path to folder containing POSCAR and wannier90_centres.xyz
        verbose: Print detailed information during computation

    Returns:
        Dictionary containing results for each atom
    """
    poscar_path = folder_path / "POSCAR"
    wannier_path = folder_path / "wannier90_centres.xyz"

    if not poscar_path.exists():
        raise FileNotFoundError(f"POSCAR not found at {poscar_path}")
    if not wannier_path.exists():
        raise FileNotFoundError(f"wannier90_centres.xyz not found at {wannier_path}")

    cell, atom_symbols, atom_positions = parse_poscar(poscar_path)
    wannier_positions = parse_wannier_centers(wannier_path)

    if verbose:
        print(f"Cell dimensions:\n{cell}")
        print(f"Number of atoms: {len(atom_symbols)}")
        print(f"Number of wannier centers: {len(wannier_positions)}")

    target_atoms = ["O", "Na", "Cl", "K", "Li", "Cs"]
    n_neighbors = {"O": 4, "Na": 4, "Cl": 4, "K": 4, "Li": 1, "Cs": 4}

    results = []

    for i, (symbol, pos) in enumerate(zip(atom_symbols, atom_positions, strict=False)):
        if symbol not in target_atoms:
            continue

        k = n_neighbors[symbol]

        indices, distances, vectors = find_k_nearest_neighbors(pos, wannier_positions, cell, k)

        # Calculate PBC-corrected positions of wannier centers relative to the atom
        # This follows the logic: corrected_pos = atom_pos + vector_to_wannier
        pbc_corrected_wannier_positions = []
        for vec in vectors:
            corrected_pos = pos + vec
            pbc_corrected_wannier_positions.append(corrected_pos)

        # Calculate average position of wannier centers
        avg_wannier_pos = np.mean(pbc_corrected_wannier_positions, axis=0)

        # Calculate vector from atom to average wannier position
        wannier_centroid = avg_wannier_pos - pos

        # Apply minimum image convention to the final vector
        inv_cell = np.linalg.inv(cell)
        wannier_centroid = wannier_centroid - np.round(wannier_centroid @ inv_cell) @ cell

        centroid_norm = np.linalg.norm(wannier_centroid)

        atom_result = {
            "atom_index": i,
            "atom_symbol": symbol,
            "atom_position": pos,
            "wannier_centers": wannier_positions[indices],
            "wannier_vectors": vectors,
            "wannier_distances": distances,
            "average_wannier_position": avg_wannier_pos,
            "wannier_centroid": wannier_centroid,
            "wannier_centroid_norm": centroid_norm,
        }

        results.append(atom_result)

        if verbose:
            print(f"\nAtom {i} ({symbol}) at {pos}")
            print(f"  Nearest {k} wannier centers:")
            for j, (idx, dist, vec) in enumerate(zip(indices, distances, vectors, strict=False)):
                print(f"    WC{j+1}: index={idx}, dist={dist:.4f}, vec={vec}")
            print(f"  Average wannier position: {avg_wannier_pos}")
            print(f"  Wannier centroid (vector from atom to avg): {wannier_centroid}")
            print(f"  Centroid norm: {centroid_norm:.4f}")

    return results


def save_results(results: list[dict], output_dir: Path) -> None:
    """
    Save results to both .npy and .txt formats.

    Args:
        results: List of dictionaries containing results for each atom
        output_dir: Directory to save output files
    """
    npy_file = output_dir / "wc_out.npy"
    txt_file = output_dir / "wc_out.txt"

    np.save(npy_file, results, allow_pickle=True)

    with open(txt_file, "w") as f:
        f.write("# Wannier Centroid Computation Results\n")
        f.write("# Format: atom_index atom_symbol atom_position[x,y,z] ")
        f.write("wannier_centers[positions] wannier_vectors[x,y,z] wannier_distances ")
        f.write("average_wannier_position[x,y,z] wannier_centroid[x,y,z] wannier_centroid_norm\n")
        f.write("#" + "=" * 100 + "\n")

        for result in results:
            f.write(f"\nAtom {result['atom_index']} ({result['atom_symbol']})\n")
            f.write(
                f"  Position: {result['atom_position'][0]:.6f} {result['atom_position'][1]:.6f} {result['atom_position'][2]:.6f}\n"
            )

            f.write(f"  Wannier centers ({len(result['wannier_centers'])}):\n")
            for i, (wc_pos, wc_vec, wc_dist) in enumerate(
                zip(
                    result["wannier_centers"],
                    result["wannier_vectors"],
                    result["wannier_distances"],
                    strict=False,
                )
            ):
                f.write(f"    WC{i+1}: pos=({wc_pos[0]:.6f}, {wc_pos[1]:.6f}, {wc_pos[2]:.6f}), ")
                f.write(f"vec=({wc_vec[0]:.6f}, {wc_vec[1]:.6f}, {wc_vec[2]:.6f}), ")
                f.write(f"dist={wc_dist:.6f}\n")

            f.write(
                f"  Average wannier position: ({result['average_wannier_position'][0]:.6f}, {result['average_wannier_position'][1]:.6f}, {result['average_wannier_position'][2]:.6f})\n"
            )
            f.write(
                f"  Wannier centroid (atom->avg): ({result['wannier_centroid'][0]:.6f}, {result['wannier_centroid'][1]:.6f}, {result['wannier_centroid'][2]:.6f})\n"
            )
            f.write(f"  Centroid norm: {result['wannier_centroid_norm']:.6f}\n")
            f.write("-" * 50 + "\n")

    if npy_file.exists() and txt_file.exists():
        print("Results saved to:")
        print(f"  - {npy_file}")
        print(f"  - {txt_file}")


def main(folder_path: str, verbose: bool = False) -> None:
    """
    Main function to compute wannier centroids.

    Args:
        folder_path: Path to folder containing POSCAR and wannier90_centres.xyz
        verbose: Print detailed information during computation
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder {folder} does not exist")

    results = compute_wannier_centroid(folder, verbose)

    save_results(results, folder)

    print("Wannier centroid computation completed successfully!")
    print(f"Processed {len(results)} atoms")
