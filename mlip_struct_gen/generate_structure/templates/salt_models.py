"""Salt model templates for salt water box generation."""

from typing import Dict, List, Tuple, Any


SALT_MODELS = {
    "NaCl": {
        "name": "Sodium Chloride",
        "cation": {
            "element": "Na",
            "charge": 1.0,
            "mass": 22.98977,
            "ionic_radius": 1.02,  # Angstroms
            "vdw_radius": 2.27,    # Angstroms
            "lj_params": {
                "charmm": {"epsilon": 0.1301, "sigma": 2.4299},  # CHARMM27
                "amber": {"epsilon": 0.1684, "sigma": 2.4389}    # AMBER (Joung & Cheatham)
            }
        },
        "anion": {
            "element": "Cl", 
            "charge": -1.0,
            "mass": 35.453,
            "ionic_radius": 1.81,  # Angstroms
            "vdw_radius": 1.75,    # Angstroms
            "lj_params": {
                "charmm": {"epsilon": 0.1500, "sigma": 4.0400},  # CHARMM27
                "amber": {"epsilon": 0.0128, "sigma": 4.9237}    # AMBER (Joung & Cheatham)
            }
        },
        "dissociation": 1.0,  # Complete dissociation
        "default_concentration": 0.15  # M (physiological)
    },
    
    "NaBr": {
        "name": "Sodium Bromide",
        "cation": {
            "element": "Na",
            "charge": 1.0,
            "mass": 22.98977,
            "ionic_radius": 1.02,
            "vdw_radius": 2.27,
            "lj_params": {
                "charmm": {"epsilon": 0.1301, "sigma": 2.4299},
                "amber": {"epsilon": 0.1684, "sigma": 2.4389}
            }
        },
        "anion": {
            "element": "Br",
            "charge": -1.0,
            "mass": 79.904,
            "ionic_radius": 1.96,
            "vdw_radius": 1.85,
            "lj_params": {
                "charmm": {"epsilon": 0.2200, "sigma": 4.6200},
                "amber": {"epsilon": 0.0269, "sigma": 4.6237}
            }
        },
        "dissociation": 1.0,
        "default_concentration": 0.15  # M
    },
    
    "KCl": {
        "name": "Potassium Chloride",
        "cation": {
            "element": "K",
            "charge": 1.0,
            "mass": 39.0983,
            "ionic_radius": 1.38,
            "vdw_radius": 2.75,
            "lj_params": {
                "charmm": {"epsilon": 0.0870, "sigma": 3.3158},
                "amber": {"epsilon": 0.000328, "sigma": 3.3310}
            }
        },
        "anion": {
            "element": "Cl",
            "charge": -1.0,
            "mass": 35.453,
            "ionic_radius": 1.81,
            "vdw_radius": 1.75,
            "lj_params": {
                "charmm": {"epsilon": 0.1500, "sigma": 4.0400},
                "amber": {"epsilon": 0.0128, "sigma": 4.9237}
            }
        },
        "dissociation": 1.0,
        "default_concentration": 0.15  # M
    },
    
    "KBr": {
        "name": "Potassium Bromide",
        "cation": {
            "element": "K",
            "charge": 1.0,
            "mass": 39.0983,
            "ionic_radius": 1.38,
            "vdw_radius": 2.75,
            "lj_params": {
                "charmm": {"epsilon": 0.0870, "sigma": 3.3158},
                "amber": {"epsilon": 0.000328, "sigma": 3.3310}
            }
        },
        "anion": {
            "element": "Br",
            "charge": -1.0,
            "mass": 79.904,
            "ionic_radius": 1.96,
            "vdw_radius": 1.85,
            "lj_params": {
                "charmm": {"epsilon": 0.2200, "sigma": 4.6200},
                "amber": {"epsilon": 0.0269, "sigma": 4.6237}
            }
        },
        "dissociation": 1.0,
        "default_concentration": 0.15  # M
    },
    
    "LiCl": {
        "name": "Lithium Chloride",
        "cation": {
            "element": "Li",
            "charge": 1.0,
            "mass": 6.941,
            "ionic_radius": 0.76,
            "vdw_radius": 1.82,
            "lj_params": {
                "charmm": {"epsilon": 0.0279, "sigma": 2.1266},
                "amber": {"epsilon": 0.0183, "sigma": 1.4090}
            }
        },
        "anion": {
            "element": "Cl",
            "charge": -1.0,
            "mass": 35.453,
            "ionic_radius": 1.81,
            "vdw_radius": 1.75,
            "lj_params": {
                "charmm": {"epsilon": 0.1500, "sigma": 4.0400},
                "amber": {"epsilon": 0.0128, "sigma": 4.9237}
            }
        },
        "dissociation": 1.0,
        "default_concentration": 0.15  # M
    },
    
    "CaCl2": {
        "name": "Calcium Chloride",
        "cation": {
            "element": "Ca",
            "charge": 2.0,
            "mass": 40.078,
            "ionic_radius": 1.00,
            "vdw_radius": 2.31,
            "lj_params": {
                "charmm": {"epsilon": 0.1200, "sigma": 2.8700},
                "amber": {"epsilon": 0.4598, "sigma": 1.7131}
            }
        },
        "anion": {
            "element": "Cl",
            "charge": -1.0,
            "mass": 35.453,
            "ionic_radius": 1.81,
            "vdw_radius": 1.75,
            "lj_params": {
                "charmm": {"epsilon": 0.1500, "sigma": 4.0400},
                "amber": {"epsilon": 0.0128, "sigma": 4.9237}
            }
        },
        "dissociation": 1.0,
        "stoichiometry": {"cation": 1, "anion": 2},  # 1 Ca2+ : 2 Cl-
        "default_concentration": 0.1  # M
    },
    
    "MgCl2": {
        "name": "Magnesium Chloride",
        "cation": {
            "element": "Mg",
            "charge": 2.0,
            "mass": 24.305,
            "ionic_radius": 0.72,
            "vdw_radius": 1.73,
            "lj_params": {
                "charmm": {"epsilon": 0.0150, "sigma": 1.5850},
                "amber": {"epsilon": 0.8947, "sigma": 1.0240}
            }
        },
        "anion": {
            "element": "Cl",
            "charge": -1.0,
            "mass": 35.453,
            "ionic_radius": 1.81,
            "vdw_radius": 1.75,
            "lj_params": {
                "charmm": {"epsilon": 0.1500, "sigma": 4.0400},
                "amber": {"epsilon": 0.0128, "sigma": 4.9237}
            }
        },
        "dissociation": 1.0,
        "stoichiometry": {"cation": 1, "anion": 2},  # 1 Mg2+ : 2 Cl-
        "default_concentration": 0.1  # M
    }
}


def get_salt_model(salt_name: str) -> Dict[str, Any]:
    """
    Get salt model parameters.
    
    Args:
        salt_name: Name of the salt model (e.g., 'NaCl', 'KCl')
        
    Returns:
        Dictionary containing salt model parameters
        
    Raises:
        ValueError: If salt_name is not supported
    """
    if salt_name not in SALT_MODELS:
        available = ", ".join(SALT_MODELS.keys())
        raise ValueError(f"Salt model '{salt_name}' not supported. Available: {available}")
    
    return SALT_MODELS[salt_name].copy()


def get_salt_stoichiometry(salt_name: str) -> Tuple[int, int]:
    """
    Get the stoichiometric ratio of cations to anions.
    
    Args:
        salt_name: Name of the salt model
        
    Returns:
        Tuple of (n_cations, n_anions) per formula unit
    """
    salt_model = get_salt_model(salt_name)
    
    if "stoichiometry" in salt_model:
        return salt_model["stoichiometry"]["cation"], salt_model["stoichiometry"]["anion"]
    else:
        # Default 1:1 stoichiometry
        return 1, 1


def calculate_ion_numbers(
    salt_name: str,
    concentration: float,
    box_volume: float
) -> Tuple[int, int]:
    """
    Calculate number of cations and anions for given concentration.
    
    Args:
        salt_name: Name of the salt
        concentration: Molar concentration (M)
        box_volume: Box volume in Angstrom^3
        
    Returns:
        Tuple of (n_cations, n_anions)
    """
    # Avogadro's number
    na = 6.022e23
    
    # Convert volume from A^3 to L
    volume_liters = box_volume * 1e-27
    
    # Calculate number of formula units
    n_formula_units = int(concentration * volume_liters * na)
    
    # Get stoichiometry
    n_cat_per_formula, n_an_per_formula = get_salt_stoichiometry(salt_name)
    
    # Calculate total ions
    n_cations = n_formula_units * n_cat_per_formula
    n_anions = n_formula_units * n_an_per_formula
    
    return n_cations, n_anions


def get_available_salts() -> List[str]:
    """Get list of available salt models."""
    return list(SALT_MODELS.keys())


def get_ion_lj_params(salt_name: str, lj_type: str = "charmm") -> Dict[str, Dict[str, float]]:
    """
    Get LJ parameters for a salt's ions based on force field type.
    
    Args:
        salt_name: Name of the salt
        lj_type: Force field type ("charmm" or "amber")
        
    Returns:
        Dictionary with cation and anion LJ parameters
        
    Raises:
        ValueError: If salt_name or lj_type is not supported
    """
    salt_model = get_salt_model(salt_name)
    
    if lj_type not in ["charmm", "amber"]:
        raise ValueError(f"Unsupported LJ type '{lj_type}'. Available: charmm, amber")
    
    result = {}
    for ion_type in ["cation", "anion"]:
        ion_data = salt_model[ion_type]
        if "lj_params" in ion_data and lj_type in ion_data["lj_params"]:
            result[ion_type] = ion_data["lj_params"][lj_type]
        else:
            raise ValueError(f"LJ parameters for {lj_type} not available for {ion_data['element']}")
    
    return result


def create_ion_xyz(ion_data: Dict[str, Any], position: List[float], output_file: str) -> None:
    """
    Create XYZ file for a single ion.
    
    Args:
        ion_data: Ion parameters dictionary
        position: Ion position [x, y, z] in Angstroms
        output_file: Path to output XYZ file
    """
    with open(output_file, 'w') as f:
        f.write("1\n")
        f.write(f"{ion_data['element']} ion\n")
        f.write(f"{ion_data['element']} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}\n")