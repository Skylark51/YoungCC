"""Optional independent cross-check. PySCF is only used by this script."""

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import pyscf
from pyscf import dft, gto, lib, scf

from metrics import compare
from opendft.input import read_input
from opendft.scf import run_scf


def main():
    lib.num_threads(1)
    results = {"reference": f"PySCF {pyscf.__version__}", "platform": platform.platform(), "cases": {}}
    for path in sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.json")):
        molecule = read_input(path)
        start = perf_counter()
        reference_molecule = gto.M(atom=list(zip(molecule.symbols, molecule.centers)),
                                   unit="Bohr", basis="sto-3g", charge=molecule.charge,
                                   spin=0, verbose=0)
        if molecule.method == "LDA":
            solver = dft.RKS(reference_molecule, xc="LDA_X,LDA_C_VWN")
            solver.grids.level = 7
        else:
            solver = scf.RHF(reference_molecule)
        solver.conv_tol = 1e-12
        solver.conv_tol_grad = 1e-9
        energy = solver.kernel()
        elapsed = perf_counter() - start
        matrix = solver.make_rdm1()

        def density(points):
            values = dft.numint.eval_ao(reference_molecule, points)
            return dft.numint.eval_rho(reference_molecule, values, matrix)

        reference = {"total_energy": energy, "orbital_energies": solver.mo_energy,
                     "electron_count": float((matrix * solver.get_ovlp()).sum()),
                     "density": density, "converged": bool(solver.converged),
                     "seconds": elapsed, "cycles": solver.cycles}
        results["cases"][path.stem] = compare(molecule, run_scf(molecule), reference)
    Path(sys.argv[1]).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return int(not all(case["energy_target_passed"] for case in results["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
