"""Metal composition parsing and alloy (HEA) utilities.

Single source of truth for supported metals, lattice constants, and element
masses used by the metal-surface, metal-water, and metal-salt-water
generators. Also provides the composition-string parser and random
solid-solution helpers that enable high-entropy alloy (HEA) slabs.

A "metal" specification is either a single element symbol ("Pt") or an
alloy composition string:

- "CoCrFeMnNi"        -> equimolar (Cantor alloy)
- "Cu0.5Ni0.3Co0.2"   -> explicit molar fractions (must sum to 1)

The order of elements in the composition string is canonical: it defines
the LAMMPS atom-type ordering (type_map) in generated data files. Use the
same composition string across a whole training dataset so type maps stay
consistent.
"""

import re

import numpy as np

# Supported FCC metals (plus common HEA constituents with effective FCC
# lattice constants)
SUPPORTED_METALS: set[str] = {
    "Al",
    "Au",
    "Ag",
    "Cu",
    "Ni",
    "Pd",
    "Pt",
    "Pb",
    "Rh",
    "Ir",
    "Ca",
    "Sr",
    "Yb",
    "Co",
    "Cr",
    "Fe",
    "Mn",
}

# Default lattice constants (Angstroms). The first 13 are experimental FCC
# values; Co/Cr/Fe/Mn are effective-FCC values (gamma phase where relevant)
# intended for building HEA solid solutions. For production HEA work prefer
# an explicitly relaxed value via the lattice_constant parameter.
DEFAULT_LATTICE_CONSTANTS: dict[str, float] = {
    "Al": 4.050,
    "Au": 4.078,
    "Ag": 4.085,
    "Cu": 3.615,
    "Ni": 3.524,
    "Pd": 3.890,
    "Pt": 3.924,
    "Pb": 4.950,
    "Rh": 3.803,
    "Ir": 3.839,
    "Ca": 5.588,
    "Sr": 6.085,
    "Yb": 5.485,
    "Co": 3.544,
    "Cr": 3.680,
    "Fe": 3.647,
    "Mn": 3.865,
}

# Element masses in g/mol
ELEMENT_MASSES: dict[str, float] = {
    "H": 1.008,
    "O": 15.9994,
    "Na": 22.98977,
    "Cl": 35.453,
    "K": 39.0983,
    "Li": 6.941,
    "Ca": 40.078,
    "Mg": 24.305,
    "Br": 79.904,
    "Cs": 132.905,
    "Pt": 195.078,
    "Au": 196.967,
    "Ag": 107.868,
    "Cu": 63.546,
    "Ni": 58.693,
    "Pd": 106.42,
    "Fe": 55.845,
    "Al": 26.982,
    "Pb": 207.2,
    "Rh": 102.906,
    "Ir": 192.217,
    "Sr": 87.62,
    "Yb": 173.04,
    "Co": 58.933,
    "Cr": 51.996,
    "Mn": 54.938,
}

# Matches one element token: capitalized symbol + optional molar fraction
_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d+)?")


def get_element_mass(element: str) -> float:
    """Get the atomic mass (g/mol) for an element, falling back to ASE data."""
    if element in ELEMENT_MASSES:
        return ELEMENT_MASSES[element]
    from ase.data import atomic_masses, atomic_numbers

    if element in atomic_numbers:
        return float(atomic_masses[atomic_numbers[element]])
    raise ValueError(f"Unknown element: '{element}'")


def parse_composition(metal: str) -> dict[str, float]:
    """
    Parse a metal specification into an ordered {element: fraction} mapping.

    Args:
        metal: Single element ("Pt") or composition string ("CoCrFeMnNi",
            "Cu0.5Ni0.3Co0.2"). Either all elements carry fractions or none
            (equimolar); fractions must sum to 1 within 1e-3.

    Returns:
        Insertion-ordered dict of element -> molar fraction. The order of
        appearance in the string is canonical and defines atom-type ordering
        in generated LAMMPS data files.

    Raises:
        ValueError: If the string is malformed, an element is duplicated, or
            fractions are inconsistent.
    """
    if not metal:
        raise ValueError("Metal element symbol is required")

    tokens = _TOKEN_RE.findall(metal)
    # Reject anything the tokenizer did not fully consume (e.g. "PtXy", "pt")
    if "".join(sym + frac for sym, frac in tokens) != metal:
        raise ValueError(
            f"Invalid metal specification: '{metal}'. "
            "Expected a single element (e.g. 'Pt') or a composition string "
            "(e.g. 'CoCrFeMnNi' or 'Cu0.5Ni0.3Co0.2')."
        )

    from ase.data import atomic_numbers

    composition: dict[str, float] = {}
    fractions: list[str] = []
    for symbol, fraction in tokens:
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element '{symbol}' in metal specification '{metal}'")
        if symbol in composition:
            raise ValueError(f"Duplicate element '{symbol}' in composition '{metal}'")
        composition[symbol] = float(fraction) if fraction else 0.0
        fractions.append(fraction)

    n_with_fraction = sum(1 for f in fractions if f)
    if n_with_fraction == 0:
        # Equimolar
        for symbol in composition:
            composition[symbol] = 1.0 / len(composition)
    elif n_with_fraction == len(composition):
        total = sum(composition.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Molar fractions in '{metal}' must sum to 1.0 (got {total:.4f})")
        # Normalize away the residual rounding error
        for symbol in composition:
            composition[symbol] /= total
    else:
        raise ValueError(
            f"Composition '{metal}' mixes elements with and without fractions. "
            "Specify fractions for all elements or none (equimolar)."
        )

    return composition


def vegard_lattice_constant(composition: dict[str, float], custom: float | None = None) -> float:
    """
    Lattice constant for a composition: custom value or Vegard's law average.

    Args:
        composition: Ordered {element: fraction} from parse_composition().
        custom: User-specified lattice constant; returned as-is if given.

    Returns:
        Lattice constant in Angstroms.

    Raises:
        ValueError: If no custom value is given and an element has no default
            lattice constant.
    """
    if custom is not None:
        return custom

    missing = [el for el in composition if el not in DEFAULT_LATTICE_CONSTANTS]
    if missing:
        raise ValueError(
            f"No default lattice constant for: {', '.join(missing)}. "
            "Specify one explicitly with the lattice_constant parameter."
        )
    return sum(frac * DEFAULT_LATTICE_CONSTANTS[el] for el, frac in composition.items())


def allocate_counts(composition: dict[str, float], n_sites: int) -> dict[str, int]:
    """
    Allocate lattice sites to elements matching the composition exactly.

    Uses largest-remainder (Hamilton) rounding so counts sum to n_sites;
    ties are broken by composition order for determinism.
    """
    counts = {el: int(frac * n_sites) for el, frac in composition.items()}
    remainders = {el: frac * n_sites - counts[el] for el, frac in composition.items()}
    leftover = n_sites - sum(counts.values())
    # Stable sort keeps composition order on remainder ties
    for el in sorted(composition, key=lambda e: -remainders[e])[:leftover]:
        counts[el] += 1
    return counts


def assign_random_symbols(
    slab: "object", composition: dict[str, float], rng: np.random.Generator
) -> None:
    """
    Randomly assign alloy elements to the sites of an ASE Atoms slab in place.

    The exact per-element counts come from allocate_counts(); the assignment
    is a seeded shuffle, so results are reproducible for a given rng seed.
    No-op for single-element compositions.
    """
    if len(composition) <= 1:
        return
    counts = allocate_counts(composition, len(slab))  # type: ignore[arg-type]
    symbols = [el for el, n in counts.items() for _ in range(n)]
    rng.shuffle(symbols)
    slab.set_chemical_symbols(symbols)  # type: ignore[attr-defined]
