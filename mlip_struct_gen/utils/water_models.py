"""Water model parameters and geometries."""

# Water model parameters including geometry and charges
WATER_MODELS = {
    "SPCE": {
        "geometry": {
            "O": [0.0000, 0.0000, 0.0000],
            "H1": [0.8164, 0.0000, 0.5773],
            "H2": [-0.8164, 0.0000, 0.5773],
        },
        "charges": {
            "O": -0.8476,
            "H": 0.4238,
        },
        "OH_distance": 1.0,  # Angstroms
        "HOH_angle": 109.47,  # degrees
    },
    "TIP3P": {
        "geometry": {
            "O": [0.0000, 0.0000, 0.0000],
            "H1": [0.7569, 0.0000, 0.5858],
            "H2": [-0.7569, 0.0000, 0.5858],
        },
        "charges": {
            "O": -0.834,
            "H": 0.417,
        },
        "OH_distance": 0.9572,  # Angstroms
        "HOH_angle": 104.52,  # degrees
    },
    "TIP4P": {
        "geometry": {
            "O": [0.0000, 0.0000, 0.0000],
            "H1": [0.7569, 0.0000, 0.5858],
            "H2": [-0.7569, 0.0000, 0.5858],
        },
        "charges": {
            "O": 0.0,  # Charge is on M site
            "H": 0.52,
            "M": -1.04,  # Virtual site
        },
        "OH_distance": 0.9572,  # Angstroms
        "HOH_angle": 104.52,  # degrees
    },
}
