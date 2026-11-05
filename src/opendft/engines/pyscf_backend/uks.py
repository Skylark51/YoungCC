"""UB3LYP/def2 single points through PySCF with optional C-PCM and D3(BJ).

This is a PySCF backend integration, not an independent B3LYP implementation.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np


@dataclass
class AdvancedResult:
    converged: bool
    electronic_energy: float
    nuclear_repulsion: float
    total_energy: float
    orbital_energies: dict
    occupied_orbitals: dict
    ao_labels: list
    density_matrix: list
    electron_count: float
    grid_electron_count: None
    iterations: list
    seconds: float
    dispersion_energy: float
    solvent: str | None
    dispersion: str | None
    engine: str


def run_advanced(molecule, *, max_cycles=80, energy_tol=1e-9, report=None):
    if molecule.method not in {"B3LYP", "UB3LYP"}:
        raise ValueError("Advanced backend supports B3LYP and UB3LYP only.")
    if max_cycles < 1 or not (0 < energy_tol < float("inf")):
        raise ValueError("Invalid convergence settings.")
    try:
        from pyscf import dft, gto, lib
    except ImportError as exc:
        raise RuntimeError('Install advanced dependencies: pip install -e ".[advanced]"') from exc
    if molecule.dispersion:
        try:
            import pyscf.dispersion  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('D3(BJ) requires pyscf-dispersion: pip install -e ".[advanced]"') from exc
    if molecule.threads is not None:
        lib.num_threads(molecule.threads)
    start = perf_counter()
    mol = gto.Mole()
    mol.atom = [(symbol, tuple(center)) for symbol, center in zip(molecule.symbols, molecule.centers)]
    mol.unit = "Bohr"
    mol.basis = molecule.basis.lower()
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = 0
    mol.build()
    # Explicit UKS even for spin-zero configurations.
    mf = dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = molecule.grid_level
    mf.max_cycle = max_cycles
    mf.conv_tol = energy_tol
    if molecule.dispersion:
        mf.disp = "d3bj"
    if molecule.solvent is not None or molecule.dielectric is not None:
        mf = mf.PCM(molecule.solvent or "water")
        mf.with_solvent.method = "C-PCM"
        if molecule.dielectric is not None:
            mf.with_solvent.eps = molecule.dielectric
    rows = []
    def callback(env):
        energy = float(env.get("e_tot", np.nan))
        row = {
            "cycle": int(env.get("cycle", len(rows))) + 1,
            "total_energy": energy,
            "delta_energy": None if not rows else energy - rows[-1]["total_energy"],
            "density_rms": float(env.get("norm_ddm", np.nan)),
            "commutator_max": float(env.get("norm_gorb", np.nan)),
        }
        rows.append(row)
        if report is not None:
            report(row)
    mf.callback = callback
    total = float(mf.kernel())
    if not np.isfinite(total):
        raise ValueError("PySCF returned non-finite energy.")
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim != 3 or dm.shape[0] != 2:
        raise ValueError("Expected separate alpha/beta UKS density matrices.")
    overlap = mol.intor("int1e_ovlp")
    electrons = float(np.einsum("sij,ji->", dm, overlap))
    nuclear = float(mol.energy_nuc())
    summary = getattr(mf, "scf_summary", {})
    if molecule.dispersion and "dispersion" not in summary:
        raise RuntimeError("Requested D3(BJ) but PySCF did not record the correction.")
    d3 = float(summary.get("dispersion", 0.0))
    if not np.isfinite(d3):
        raise ValueError("Non-finite D3(BJ) correction.")
    return AdvancedResult(
        bool(mf.converged), total - nuclear, nuclear, total,
        {"alpha": np.asarray(mf.mo_energy[0]).tolist(),
         "beta": np.asarray(mf.mo_energy[1]).tolist()},
        {"alpha": int(mol.nelec[0]), "beta": int(mol.nelec[1])},
        [str(s) for s in mol.ao_labels()], dm.tolist(),
        electrons, None, rows, perf_counter() - start,
        d3, molecule.solvent, molecule.dispersion, "PySCF"
    )
