"""Shared log and JSON result output."""

import json
from dataclasses import asdict
from pathlib import Path

from .scf import run_scf
from .advanced import run_advanced


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
            runner = run_advanced if molecule.method in {"B3LYP", "UB3LYP"} else run_scf
            result = runner(molecule, report=lambda row: write(format_cycle(row)), **scf_options)
        except Exception as error:
            write(f"Calculation failed: {error}")
            raise
        write("SCF converged" if result.converged else "SCF did not converge")
        write(f"Electronic energy  {result.electronic_energy: .12f} Eh")
        write(f"Nuclear repulsion  {result.nuclear_repulsion: .12f} Eh")
        write(f"Total energy       {result.total_energy: .12f} Eh")
        write(f"Electron count     {result.electron_count: .10f}")
        write(f"Wall time          {result.seconds: .3f} s")
        write("")
        write("========== YOUNGCC FINAL SUMMARY ==========")
        write(f"Status            {'CONVERGED' if result.converged else 'NOT CONVERGED'}")
        write(f"Method / Basis    {molecule.method} / {molecule.basis}")
        write(f"Solvent           {molecule.solvent or 'gas phase'}")
        write(f"Dispersion        {molecule.dispersion or 'none'}")
        if hasattr(result, "dispersion_energy"):
            write(f"D3(BJ) correction {result.dispersion_energy: .12f} Eh")
        write(f"Final total E     {result.total_energy: .12f} Eh")
        write(f"SCF cycles        {len(result.iterations)}")
        write(f"Wall time         {result.seconds:.3f} s")
        write("===========================================")
    data = {
        "input": {
            "method": molecule.method, "basis": molecule.basis,
            "charge": molecule.charge, "multiplicity": molecule.multiplicity,
            "solvent": molecule.solvent, "dielectric": molecule.dielectric,
            "dispersion": molecule.dispersion, "grid_level": molecule.grid_level,
            "units": "bohr",
            "atoms": [[symbol, *center] for symbol, center in zip(molecule.symbols, molecule.centers)],
        },
        **asdict(result),
    }
    Path(f"{prefix}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    return result
