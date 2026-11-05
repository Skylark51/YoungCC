"""Generate matched Gaussian inputs or compare completed log and fchk files."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import compare

from opendft.basis import SHELLS, build_basis, evaluate_basis
from opendft.dft import density_at
from opendft.input import read_input
from opendft.scf import run_scf

ROOT = Path(__file__).resolve().parents[2]
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def deck(molecule, stem):
    method = "RHF" if molecule.method == "RHF" else "SVWN5"
    grid = " Int=SuperFineGrid" if molecule.method == "LDA" else ""
    lines = [f"%chk={stem}.chk", "%nprocshared=1",
             f"#p {method}/STO-3G SP NoSymm Units=Bohr Guess=Core SCF=(Conver=10,MaxCycle=80){grid}",
             "", f"YoungCC benchmark {stem}", "", f"{molecule.charge} {molecule.multiplicity}"]
    lines.extend(f"{symbol} {x:.14f} {y:.14f} {z:.14f}"
                 for symbol, (x, y, z) in zip(molecule.symbols, molecule.centers))
    return "\n".join([*lines, "", ""])


def number(value):
    return float(value.replace("D", "E").replace("d", "e"))


def fchk_field(text, label):
    match = re.search(rf"^{re.escape(label)}\s+[IR]\s+(?:(N=)\s+)?({NUMBER})[^\S\n]*$",
                      text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing fchk field: {label}")
    if match[1] is None:
        return number(match[2])
    count = int(match[2])
    values = []
    for line in text[match.end():].splitlines():
        values.extend(number(value) for value in line.split())
        if len(values) >= count:
            break
    if len(values) != count:
        raise ValueError(f"Unexpected fchk array size: {label}")
    return np.asarray(values)


def gaussian_reference(molecule, logfile, checkpoint):
    log = logfile.read_text(errors="replace")
    scf = re.findall(rf"SCF Done:\s+E\(([^)]+)\)\s*=\s*({NUMBER})\s+A\.U\. after\s+(\d+)\s+cycles?", log)
    termination = [line for line in log.splitlines() if "termination" in line.lower()]
    if len(scf) != 1 or not termination or "Normal termination of Gaussian 16" not in termination[-1]:
        raise ValueError(f"Expected one normally terminated Gaussian 16 single point in {logfile}.")
    method, energy, cycles = scf[0]
    expected = "RHF" if molecule.method == "RHF" else "RSVWN5"
    if method.upper().replace("-", "") != expected:
        raise ValueError(f"Expected Gaussian SCF method {expected}, found {method}.")
    elapsed = re.findall(r"Elapsed time:\s+(\d+) days\s+(\d+) hours\s+(\d+) minutes\s+([\d.]+) seconds", log)
    seconds = sum(float(value) * factor for value, factor in
                  zip(elapsed[-1], (86400, 3600, 60, 1))) if elapsed else None
    fchk = checkpoint.read_text()
    if not np.isclose(fchk_field(fchk, "SCF Energy"), number(energy), rtol=0, atol=1e-8):
        raise ValueError("Gaussian log and checkpoint energies disagree.")
    if (fchk_field(fchk, "Charge") != molecule.charge
            or fchk_field(fchk, "Multiplicity") != molecule.multiplicity):
        raise ValueError("Gaussian checkpoint charge or multiplicity differs from the input.")
    charges = fchk_field(fchk, "Atomic numbers")
    centers = fchk_field(fchk, "Current cartesian coordinates").reshape(-1, 3)
    if (not np.array_equal(charges, molecule.charges) or centers.shape != molecule.centers.shape
            or not np.allclose(centers, molecule.centers, atol=1e-7, rtol=0)):
        raise ValueError("Gaussian checkpoint atoms or coordinate frame differ from the input.")
    shells = [(i, shell) for i, symbol in enumerate(molecule.symbols, 1)
              for shell in ((0, -1) if symbol == "O" else (0,))]
    if (list(zip(fchk_field(fchk, "Shell to atom map"), fchk_field(fchk, "Shell types"))) != shells
            or not np.all(fchk_field(fchk, "Number of primitives per shell") == 3)):
        raise ValueError("Gaussian checkpoint shell layout differs from the expected STO-3G layout.")
    basis_shells = [shell for symbol in molecule.symbols for shell in SHELLS[symbol]]
    s_shells = [shell for shell in basis_shells if shell[2] == 0]
    fields = {
        "Primitive exponents": [x for exponents, _, _ in s_shells for x in exponents],
        "Contraction coefficients": [c for _, coefficients, _ in s_shells for c in coefficients],
    }
    if "O" in molecule.symbols:
        fields["P(S=P) Contraction coefficients"] = [
            c for symbol in molecule.symbols
            for shell in ([(0, 0, 0), SHELLS["O"][-1][1]] if symbol == "O" else [(0, 0, 0)])
            for c in shell
        ]
    for label, expected_values in fields.items():
        actual_values = fchk_field(fchk, label)
        if (actual_values.shape != (len(expected_values),)
                or not np.allclose(actual_values, expected_values, rtol=1e-7, atol=1e-10)):
            raise ValueError(f"Gaussian checkpoint differs from STO-3G in {label}.")
    orbitals = build_basis(molecule)
    size = len(orbitals)
    eigenvalues = fchk_field(fchk, "Alpha Orbital Energies")
    count = fchk_field(fchk, "Number of electrons")
    if eigenvalues.size != size or count != molecule.electrons:
        raise ValueError("Gaussian checkpoint orbital or electron count differs from the input.")
    triangle = np.tril_indices(size)
    packed = fchk_field(fchk, "Total SCF Density")
    if packed.size != triangle[0].size:
        raise ValueError("Gaussian checkpoint has the wrong AO density matrix size.")
    matrix = np.zeros((size, size))
    matrix[triangle] = matrix[triangle[::-1]] = packed
    return {"total_energy": number(energy), "cycles": int(cycles), "seconds": seconds,
            "converged": True, "orbital_energies": eigenvalues, "electron_count": count,
            "density": lambda points: density_at(evaluate_basis(orbitals, points), matrix)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "compare"])
    parser.add_argument("directory", type=Path, help="Directory for .gjf, .log, and .fchk files")
    args = parser.parse_args()
    results = {}
    args.directory.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "examples").glob("*.json")):
        molecule = read_input(path)
        base = args.directory / path.stem
        if args.action == "prepare":
            base.with_suffix(".gjf").write_text(deck(molecule, path.stem), encoding="ascii")
        else:
            reference = gaussian_reference(molecule, base.with_suffix(".log"), base.with_suffix(".fchk"))
            results[path.stem] = compare(molecule, run_scf(molecule), reference)
    if results:
        report = args.directory / "comparison.json"
        report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(report)
        return int(not all(case["energy_target_passed"] for case in results.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
