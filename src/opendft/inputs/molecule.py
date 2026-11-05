"""Parse validated JSON, with optional spin-polarized DFT settings."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ANGSTROM_PER_BOHR = 0.529177210903
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr").split()
ATOMIC_NUMBERS = {symbol: i for i, symbol in enumerate(ELEMENTS, 1)}
SOLVENT_ALIASES = {"mecn": "acetonitrile", "acn": "acetonitrile", "meoh": "methanol",
                   "h2o": "water", "etoh": "ethanol", "dcm": "dichloromethane"}


@dataclass(frozen=True)
class Molecule:
    symbols: tuple[str, ...]
    centers: np.ndarray
    charge: int
    multiplicity: int
    method: str
    basis: str
    solvent: str | None = None
    dielectric: float | None = None
    dispersion: str | None = None
    grid_level: int = 3
    threads: int | None = None

    @property
    def charges(self):
        return np.array([ATOMIC_NUMBERS[s] for s in self.symbols])

    @property
    def electrons(self):
        return int(self.charges.sum()) - self.charge


def parse_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object.")
    unknown = data.keys() - {"atoms", "units", "charge", "multiplicity", "method", "basis",
                              "solvent", "dielectric", "dispersion", "grid_level", "threads"}
    if unknown:
        raise ValueError(f"Unknown input fields: {', '.join(sorted(unknown))}")
    method = str(data.get("method", "RHF")).upper()
    basis = str(data.get("basis", "STO-3G")).upper().replace("_", "-")
    basis = {"DEF2SVP": "DEF2-SVP", "DEF2TZVPP": "DEF2-TZVPP"}.get(basis, basis)
    advanced = method in {"UB3LYP", "B3LYP"}
    if not ((advanced and basis in {"DEF2-SVP", "DEF2-TZVPP"}) or
            (method in {"RHF", "LDA"} and basis == "STO-3G")):
        raise ValueError("Supported: RHF/LDA with STO-3G; UB3LYP/B3LYP with def2-SVP or def2-TZVPP.")
    units = str(data.get("units", "angstrom")).lower()
    if units not in {"angstrom", "bohr"}:
        raise ValueError("Units must be angstrom or bohr.")
    charge, multiplicity = data.get("charge", 0), data.get("multiplicity", 1)
    if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("Charge and multiplicity must be integers, multiplicity >= 1.")
    atoms = data.get("atoms")
    if not isinstance(atoms, list) or not atoms or any(
            not isinstance(a, list) or len(a) != 4 for a in atoms):
        raise ValueError("Each atom must be [symbol, x, y, z].")
    symbols = tuple(str(a[0]).capitalize() for a in atoms)
    allowed = ATOMIC_NUMBERS if advanced else {"H", "He", "O"}
    if any(s not in allowed for s in symbols):
        raise ValueError("Unsupported element for the selected method.")
    try:
        centers = np.array([a[1:] for a in atoms], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Coordinates must be finite numbers.") from exc
    if centers.shape != (len(atoms), 3) or not np.isfinite(centers).all():
        raise ValueError("Coordinates must be finite numbers.")
    if units == "angstrom":
        centers /= ANGSTROM_PER_BOHR
    if not np.isfinite(centers).all():
        raise ValueError("Coordinates overflow.")
    for i, center in enumerate(centers):
        if np.any(np.linalg.norm(centers[:i] - center, axis=1) < 1e-7):
            raise ValueError("Nuclei must have distinct positions.")
    solvent = data.get("solvent")
    if solvent is not None:
        if not advanced or not isinstance(solvent, str) or not solvent.strip():
            raise ValueError("C-PCM solvent requires a nonempty name and B3LYP.")
        solvent = SOLVENT_ALIASES.get(solvent.strip().lower(), solvent.strip())
    dispersion = data.get("dispersion")
    if dispersion is not None:
        if not advanced or str(dispersion).upper() not in {"GD3BJ", "D3BJ"}:
            raise ValueError("Only EmpiricalDispersion=GD3BJ is supported with B3LYP.")
        dispersion = "D3BJ"
    dielectric = data.get("dielectric")
    if dielectric is not None:
        if not advanced or isinstance(dielectric, bool):
            raise ValueError("Dielectric requires B3LYP.")
        try:
            dielectric = float(dielectric)
        except (ValueError, TypeError) as exc:
            raise ValueError("Dielectric must be finite and > 1.") from exc
        if not np.isfinite(dielectric) or dielectric <= 1:
            raise ValueError("Dielectric must be finite and > 1.")
    level = data.get("grid_level", 3)
    if type(level) is not int or not 0 <= level <= 9:
        raise ValueError("grid_level must be an integer 0 to 9.")
    threads = data.get("threads")
    if threads is not None and (type(threads) is not int or not 1 <= threads <= 256):
        raise ValueError("threads must be an integer between 1 and 256.")
    mol = Molecule(symbols, centers, charge, multiplicity, method, basis, solvent,
                   dielectric, dispersion, level, threads)
    spin = multiplicity - 1
    if advanced:
        if mol.electrons <= 0 or spin > mol.electrons or (mol.electrons - spin) % 2:
            raise ValueError("Electron number and spin multiplicity are inconsistent.")
    else:
        capacity = 2 * sum(5 if s == "O" else 1 for s in symbols)
        if multiplicity != 1 or mol.electrons <= 0 or mol.electrons % 2 or mol.electrons > capacity:
            raise ValueError("Restricted STO-3G requires valid positive even electron count and singlet.")
    return mol


def read_input(path):
    path = Path(path)
    if path.suffix.lower() in {".gjf", ".com"}:
        from .gaussian import parse_gaussian
        return parse_gaussian(path.read_text(encoding="utf-8-sig"))
    return parse_input(json.loads(path.read_text(encoding="utf-8-sig")))
