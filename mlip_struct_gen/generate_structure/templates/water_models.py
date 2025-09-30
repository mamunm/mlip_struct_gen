"""Water model templates for packmol structure generation."""

from typing import Any

# Water model definitions with coordinates and charges
WATER_MODELS = {
    "SPC/E": {
        "description": "Simple Point Charge/Extended water model",
        "atoms": [
            {"element": "O", "position": [0.0000, 0.0000, 0.0000], "charge": -0.8476},
            {"element": "H", "position": [0.8165, 0.5773, 0.0000], "charge": 0.4238},
            {"element": "H", "position": [-0.8165, 0.5773, 0.0000], "charge": 0.4238},
        ],
        "bonds": [{"atoms": [0, 1], "length": 1.0000}, {"atoms": [0, 2], "length": 1.0000}],
        "angles": [{"atoms": [1, 0, 2], "angle": 109.47}],
        "density": 0.997,  # g/cm³ at 25°C
    },
    "TIP3P": {
        "description": "Transferable Intermolecular Potential 3-Point water model",
        "atoms": [
            {"element": "O", "position": [0.0000, 0.0000, 0.0000], "charge": -0.834},
            {"element": "H", "position": [0.7569, 0.5858, 0.0000], "charge": 0.417},
            {"element": "H", "position": [-0.7569, 0.5858, 0.0000], "charge": 0.417},
        ],
        "bonds": [{"atoms": [0, 1], "length": 0.9572}, {"atoms": [0, 2], "length": 0.9572}],
        "angles": [{"atoms": [1, 0, 2], "angle": 104.52}],
        "density": 0.997,  # g/cm³ at 25°C
    },
    "TIP4P": {
        "description": "Transferable Intermolecular Potential 4-Point water model",
        "atoms": [
            {"element": "O", "position": [0.0000, 0.0000, 0.0000], "charge": 0.0},
            {"element": "H", "position": [0.7569, 0.5858, 0.0000], "charge": 0.520},
            {"element": "H", "position": [-0.7569, 0.5858, 0.0000], "charge": 0.520},
            {
                "element": "M",
                "position": [0.0000, -0.1500, 0.0000],
                "charge": -1.040,
            },  # Virtual site
        ],
        "bonds": [{"atoms": [0, 1], "length": 0.9572}, {"atoms": [0, 2], "length": 0.9572}],
        "angles": [{"atoms": [1, 0, 2], "angle": 104.52}],
        "density": 0.997,  # g/cm³ at 25°C
    },
    "SPC/Fw": {
        "description": "Flexible SPC water model",
        "atoms": [
            {"element": "O", "position": [0.0000, 0.0000, 0.0000], "charge": -0.82},
            {"element": "H", "position": [0.8660, 0.5574, 0.0000], "charge": 0.41},
            {"element": "H", "position": [-0.8660, 0.5574, 0.0000], "charge": 0.41},
        ],
        "bonds": [{"atoms": [0, 1], "length": 1.012}, {"atoms": [0, 2], "length": 1.012}],
        "angles": [{"atoms": [1, 0, 2], "angle": 113.24}],
        "density": 0.997,  # g/cm³ at 25°C
    },
}


def get_water_model(model_name: str) -> dict[str, Any]:
    """
    Get water model parameters.

    Args:
        model_name: Name of the water model ('SPC/E', 'TIP3P', 'TIP4P', 'SPC/Fw')

    Returns:
        Dictionary containing water model parameters

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in WATER_MODELS:
        available = ", ".join(WATER_MODELS.keys())
        raise ValueError(f"Water model '{model_name}' not supported. Available: {available}")

    return WATER_MODELS[model_name].copy()


def create_water_xyz(model_name: str, output_file: str) -> None:
    """
    Create XYZ file for a single water molecule.

    Args:
        model_name: Name of the water model
        output_file: Path to output XYZ file
    """
    model = get_water_model(model_name)

    with open(output_file, "w") as f:
        # Write number of atoms
        f.write(f"{len(model['atoms'])}\n")
        f.write(f"{model_name} water molecule\n")

        # Write atom coordinates
        for atom in model["atoms"]:
            f.write(
                f"{atom['element']} {atom['position'][0]:.6f} "
                f"{atom['position'][1]:.6f} {atom['position'][2]:.6f}\n"
            )


def create_water_pdb(model_name: str, output_file: str) -> None:
    """
    Create PDB file for a single water molecule.

    Args:
        model_name: Name of the water model
        output_file: Path to output PDB file
    """
    model = get_water_model(model_name)

    with open(output_file, "w") as f:
        # Write header
        f.write(f"TITLE     {model_name} water molecule\n")

        # Write atom records
        atom_num = 1
        for atom in model["atoms"]:
            # PDB ATOM record format
            f.write(
                f"ATOM  {atom_num:5d}  {atom['element']:<3s} HOH A   1    "
                f"{atom['position'][0]:8.3f}{atom['position'][1]:8.3f}{atom['position'][2]:8.3f}"
                f"  1.00 20.00           {atom['element']:>2s}\n"
            )
            atom_num += 1

        f.write("END\n")


def create_water_template(model_name: str, output_file: str, file_format: str = "xyz") -> None:
    """
    Create water molecule template file in specified format.

    Args:
        model_name: Name of the water model
        output_file: Path to output file
        file_format: Format to use ("xyz" or "pdb")
    """
    if file_format == "pdb":
        create_water_pdb(model_name, output_file)
    else:
        create_water_xyz(model_name, output_file)


def get_water_density(model_name: str) -> float:
    """
    Get water density for the specified model.

    Args:
        model_name: Name of the water model

    Returns:
        Density in g/cm³
    """
    model = get_water_model(model_name)
    return model["density"]


def calculate_water_molecules(
    box_size: tuple[float, float, float], model_name: str = "SPC/E"
) -> int:
    """
    Calculate number of water molecules to fill a box at standard density.

    Args:
        box_size: Box dimensions (x, y, z) in Angstroms
        model_name: Water model to use

    Returns:
        Number of water molecules
    """
    # Box volume in cm³
    volume_cm3 = (box_size[0] * box_size[1] * box_size[2]) * 1e-24

    # Water molar mass (g/mol)
    water_molar_mass = 18.015

    # Avogadro's number
    na = 6.022e23

    # Density in g/cm³
    density = get_water_density(model_name)

    # Calculate number of molecules
    mass_g = density * volume_cm3
    moles = mass_g / water_molar_mass
    n_molecules = int(moles * na)

    return n_molecules
