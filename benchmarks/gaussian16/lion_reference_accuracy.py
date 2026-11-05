"""Reproducible Gaussian 16 vs YoungCC fixed-geometry energy comparisons.

References: committed Lion Gaussian16 output files. This is not an Opt/Freq/IRC
validation: the final Gaussian SCF geometry is used for a new YoungCC SP job.
Requires optional PySCF backend only when --run is set.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from opendft.input import ATOMIC_NUMBERS, parse_input

HARTREE_TO_KCAL_MOL = 627.5094740631
ROOT = Path(__file__).resolve().parent / "lion-test"
ELEMENTS = {n: symbol for symbol, n in ATOMIC_NUMBERS.items()}
CASES = {
    "ts_svp": (
        "MalonicAcid-scan-freq-ts-freq.out", "def2-SVP", -345.554981665
    ),
    "ts_tzvpp": (
        "MalonicAcid-scan-freq-ts-freq-bb.out", "def2-TZVPP", -345.954367965
    ),
    "min_svp": (
        "MalonicAcid-scan-freq-ts-freq-ircfwd-opt-freq.out", "def2-SVP", -345.557581541
    ),
    "min_tzvpp": (
        "MalonicAcid-scan-freq-ts-freq-ircfwd-opt-freq-bb.out", "def2-TZVPP", -345.957709189
    ),
}
SCF_PATTERN = re.compile(
    r"SCF Done:\s+E\((?P<method>[UR]B3LYP)\)\s*=\s*(?P<energy>-?\d+\.\d+)"
)
ATOM_PATTERN = re.compile(
    r"^\s*\d+\s+(\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


def gaussian_orientations(log_text):
    """Return (character offset, Cartesian atom list) for Gaussian orientations."""
    lines = log_text.splitlines(keepends=True)
    positions = []
    offset = 0
    for i, line in enumerate(lines):
        if re.search(r"\b(?:Input|Standard) orientation:", line):
            atoms = []
            j = i + 5  # Center header and horizontal rules
            while j < len(lines):
                match = ATOM_PATTERN.match(lines[j])
                if not match:
                    break
                atomic_number = int(match.group(1))
                if atomic_number not in ELEMENTS:
                    raise ValueError(f"Unknown atomic number {atomic_number}")
                atoms.append([ELEMENTS[atomic_number], *[float(x) for x in match.groups()[1:]]])
                j += 1
            if atoms:
                positions.append((offset, atoms))
        offset += len(line)
    return positions


def extract_reference(case):
    filename, basis, expected_energy = CASES[case]
    path = ROOT / filename
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if "Normal termination of Gaussian 16" not in text:
        raise ValueError(f"Reference did not terminate normally: {filename}")
    if "Model                : C-PCM." not in text:
        raise ValueError("Expected Gaussian C-PCM solvation.")
    if "Atomic radii         : UFF" not in text:
        raise ValueError("Expected Gaussian UFF cavity radii.")
    if not re.search(r"Solvent\s*:\s*Acetonitrile", text):
        raise ValueError("Expected acetonitrile.")
    scf = list(SCF_PATTERN.finditer(text))
    if len(scf) != 1:
        raise ValueError(f"Fixed-geometry reference must have one SCF energy: {filename} ({len(scf)})")
    energy = float(scf[0].group("energy"))
    if abs(energy - expected_energy) > 5e-9:
        raise ValueError(f"Reference energy mismatch in {filename}: {energy}")
    if scf[0].group("method") != "UB3LYP":
        raise ValueError("Unexpected XC method in reference.")
    orientations = [(position, atoms) for position, atoms in gaussian_orientations(text)
                    if position < scf[0].start()]
    if not orientations:
        raise ValueError("No Cartesian geometry before SCF energy.")
    atoms = orientations[-1][1]
    if len(atoms) != 15:
        raise ValueError(f"Unexpected atom count {len(atoms)}")
    counts = {symbol: sum(a[0] == symbol for a in atoms) for symbol in ("C", "H", "O")}
    if counts != {"C": 5, "H": 8, "O": 2}:
        raise ValueError(f"Unexpected composition: {counts}")
    data = {
        "method": "UB3LYP", "basis": basis, "units": "angstrom",
        "charge": 0, "multiplicity": 1, "solvent": "acetonitrile",
        "atoms": atoms
    }
    # Input consistency is checked independently of the numerical DFT backend.
    parse_input(data)
    return {
        "case": case, "filename": filename, "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_energy_hartree": energy, "basis": basis,
        "molecule_formula": "C5H8O2", "atoms": atoms,
        "charge": 0, "multiplicity": 1, "solvent": "acetonitrile",
        "gaussian_cavity_radii": "UFF",
        "youngcc_cavity_radii": "PySCF default, not proven equivalent to Gaussian UFF",
        "method": "UB3LYP", "dispersion": None, "input": data,
    }


def compare(case, *, max_cycles=80):
    from opendft.advanced import run_advanced
    record = extract_reference(case)
    result = run_advanced(parse_input(record["input"]), max_cycles=max_cycles)
    reference = record["reference_energy_hartree"]
    delta = result.total_energy - reference
    record.update({
        "youngcc_total_energy_hartree": result.total_energy,
        "youngcc_converged": result.converged,
        "youngcc_engine": result.engine,
        "youngcc_iterations": len(result.iterations),
        "delta_hartree": delta,
        "delta_millihartree": 1000 * delta,
        "delta_kcal_per_mol": HARTREE_TO_KCAL_MOL * delta,
        "abs_delta_hartree": abs(delta),
        "youngcc_seconds": result.seconds,
        "interpretation": (
            "This is a raw discrepancy, not an isolated XC/integral error: "
            "Gaussian reports UFF PCM cavity while YoungCC uses PySCF defaults."
        ),
    })
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--run", action="store_true",
                        help="Run YoungCC SCF; without this flag extract references only")
    parser.add_argument("--output", type=Path, help="Write JSON accuracy report")
    parser.add_argument("--max-cycles", type=int, default=80)
    args = parser.parse_args()
    cases = list(CASES) if args.case == "all" else [args.case]
    results = []
    for case in cases:
        print(f"Checking {case}", flush=True)
        result = compare(case, max_cycles=args.max_cycles) if args.run else extract_reference(case)
        results.append(result)
        print(f"  Gaussian: {result['reference_energy_hartree']:.9f} Eh", flush=True)
        if args.run:
            print(f"  YoungCC:  {result['youngcc_total_energy_hartree']:.9f} Eh "
                  f"(delta={result['delta_millihartree']:+.6f} mEh)", flush=True)
    payload = {
        "reference_commit": "ae5afd19fbe8d30be712e161e7b3239488e5d8a9",
        "reference_method": "Gaussian 16 UB3LYP C-PCM(MeCN), UFF cavity radii",
        "comparison_scope": "fixed-geometry electronic single points",
        "caution": "Different PCM cavities and grids; raw discrepancy is not necessarily YoungCC error",
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.run and not all(r["youngcc_converged"] for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
