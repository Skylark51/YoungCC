"""Shared log and JSON result output."""

import json
from dataclasses import asdict
from pathlib import Path

from .scf import run_scf


def format_cycle(row):
    delta = "--" if row["delta_energy"] is None else f'{row["delta_energy"]:+.3e}'
    return (f'{row["cycle"]:4d}  {row["total_energy"]:20.12f}  {delta:>11}  '
            f'{row["density_rms"]:11.3e}  {row["commutator_max"]:11.3e}')


def calculate(molecule, prefix, *, emit=print, **scf_options):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{prefix}.log").open("w", encoding="utf-8") as log:
        def write(line):
            print(line, file=log, flush=True)
            emit(line)

        write(f"YoungCC  {molecule.method}/{molecule.basis}  charge={molecule.charge}  "
              f"multiplicity={molecule.multiplicity}")
        write("Cycle        E(total) / Eh          Delta E    RMS density   Max [F,P]")
        try:
            result = run_scf(molecule, report=lambda row: write(format_cycle(row)), **scf_options)
        except Exception as error:
            write(f"Calculation failed: {error}")
            raise
        write("SCF converged" if result.converged else "SCF did not converge")
        write(f"Electronic energy  {result.electronic_energy: .12f} Eh")
        write(f"Nuclear repulsion  {result.nuclear_repulsion: .12f} Eh")
        write(f"Total energy       {result.total_energy: .12f} Eh")
        write(f"Electron count     {result.electron_count: .10f}")
        write(f"Wall time          {result.seconds: .3f} s")
    data = {
        "input": {
            "method": molecule.method, "basis": molecule.basis,
            "charge": molecule.charge, "multiplicity": molecule.multiplicity,
            "units": "bohr",
            "atoms": [[symbol, *center] for symbol, center in zip(molecule.symbols, molecule.centers)],
        },
        **asdict(result),
    }
    Path(f"{prefix}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    return result
