"""Read and validate restricted single-point inputs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANGSTROM_PER_BOHR = 0.529177210903
ATOMIC_NUMBERS = {"H": 1, "He": 2, "O": 8}


@dataclass(frozen=True)
class Molecule:
    symbols: tuple[str, ...]
    centers: np.ndarray
    charge: int
    multiplicity: int
    method: str
    basis: str

    @property
    def charges(self):
        return np.array([ATOMIC_NUMBERS[symbol] for symbol in self.symbols])

    @property
    def electrons(self):
        return int(self.charges.sum()) - self.charge


def parse_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object.")
    unknown = data.keys() - {"atoms", "units", "charge", "multiplicity", "method", "basis"}
    if unknown:
        raise ValueError(f"Unknown input fields: {', '.join(sorted(unknown))}")
    method = str(data.get("method", "RHF")).upper()
    basis = str(data.get("basis", "STO-3G")).upper()
    units = str(data.get("units", "angstrom")).lower()
    charge = data.get("charge", 0)
    multiplicity = data.get("multiplicity", 1)
    if method not in {"RHF", "LDA"} or basis != "STO-3G":
        raise ValueError("Supported methods are RHF and LDA with STO-3G.")
    if units not in {"angstrom", "bohr"}:
        raise ValueError("Units must be angstrom or bohr.")
    if type(charge) is not int or type(multiplicity) is not int or multiplicity != 1:
        raise ValueError("Use an integer charge and singlet multiplicity 1.")
    atoms = data.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("Supply at least one atom as [symbol, x, y, z].")
    if any(not isinstance(atom, list) or len(atom) != 4 for atom in atoms):
        raise ValueError("Supply each atom as [symbol, x, y, z].")
    symbols = tuple(str(atom[0]).capitalize() for atom in atoms)
    if any(symbol not in ATOMIC_NUMBERS for symbol in symbols):
        raise ValueError("STO-3G is available for H, He and O.")
    try:
        centers = np.array([atom[1:] for atom in atoms], dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Coordinates must be finite numbers.") from error
    if centers.shape != (len(atoms), 3) or not np.isfinite(centers).all():
        raise ValueError("Coordinates must be finite numbers.")
    if units == "angstrom":
        centers /= ANGSTROM_PER_BOHR
    if not np.isfinite(centers).all():
        raise ValueError("Coordinates exceed the representable range in bohr.")
    for i, center in enumerate(centers):
        if np.any(np.linalg.norm(centers[:i] - center, axis=1) < 1e-7):
            raise ValueError("Nuclei must have distinct positions.")
    molecule = Molecule(symbols, centers, charge, multiplicity, method, basis)
    capacity = 2 * sum(5 if symbol == "O" else 1 for symbol in symbols)
    if molecule.electrons <= 0 or molecule.electrons % 2 or molecule.electrons > capacity:
        raise ValueError("A positive even electron count fitting the basis is required.")
    return molecule


def read_input(path):
    return parse_input(json.loads(Path(path).read_text(encoding="utf-8-sig")))
