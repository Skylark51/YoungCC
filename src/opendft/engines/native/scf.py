"""Independent RHF and RKS single-point SCF driver."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import eigh

from ...chemistry.basis import build_basis, evaluate_basis
from ...chemistry.dft import density_at, xc_matrix
from ...chemistry.grid import molecular_grid
from ...chemistry.integrals import build_integrals, nuclear_repulsion


@dataclass
class Result:
    converged: bool
    electronic_energy: float
    nuclear_repulsion: float
    total_energy: float
    orbital_energies: list
    occupied_orbitals: int
    ao_labels: list
    density_matrix: list
    electron_count: float
    grid_electron_count: float | None
    iterations: list
    seconds: float


def _diis(history):
    errors = np.array([error.ravel() for _, error in history])
    size = len(errors)
    system = np.ones((size + 1, size + 1))
    system[:size, :size] = errors @ errors.T
    scale = np.max(np.abs(system[:size, :size]))
    if scale == 0:
        return history[-1][0]
    system[:size, :size] /= scale
    system[-1, -1] = 0
    target = np.zeros(size + 1)
    target[-1] = 1
    try:
        coefficients = np.linalg.solve(system, target)[:-1]
    except np.linalg.LinAlgError:
        return history[-1][0]
    if np.max(np.abs(coefficients)) > 100:
        return history[-1][0]
    return sum(coefficient * fock for coefficient, (fock, _) in zip(coefficients, history))


def run_scf(molecule, *, max_cycles=80, energy_tol=1e-10, density_tol=1e-8,
            grid_orders=(80, 24, 48), report=None):
    if max_cycles < 1 or not (0 < energy_tol < np.inf and 0 < density_tol < np.inf):
        raise ValueError("SCF needs positive finite tolerances and at least one cycle.")
    start = perf_counter()
    orbitals = build_basis(molecule)
    overlap, kinetic, attraction, eri = build_integrals(orbitals, molecule)
    if eigh(overlap, eigvals_only=True)[0] < 1e-8:
        raise ValueError("The AO basis is numerically linearly dependent.")
    core = kinetic + attraction
    nuclear = nuclear_repulsion(molecule)
    occupied = molecule.electrons // 2
    is_lda = molecule.method == "LDA"
    if is_lda:
        points, weights = molecular_grid(molecule.centers, *grid_orders)
        values = evaluate_basis(orbitals, points)

    def diagonalize(fock):
        eigenvalues, coefficients = eigh(fock, overlap)
        occupied_coefficients = coefficients[:, :occupied]
        return eigenvalues, 2 * occupied_coefficients @ occupied_coefficients.T

    _, density = diagonalize(core)
    previous_energy = None
    records, history = [], []
    converged = False
    for cycle in range(1, max_cycles + 1):
        coulomb = np.einsum("ijkl,kl->ij", eri, density)
        if is_lda:
            xc_energy, potential = xc_matrix(values, weights, density)
            fock = core + coulomb + potential
            energy = np.sum(density * (core + 0.5 * coulomb)) + xc_energy
        else:
            exchange = np.einsum("ikjl,kl->ij", eri, density)
            fock = core + coulomb - 0.5 * exchange
            energy = 0.5 * np.sum(density * (core + fock))
        if not np.isfinite(energy) or not np.isfinite(fock).all():
            raise ValueError("SCF produced non-finite values.")
        orbital_energies, next_density = diagonalize(fock)
        delta = None if previous_energy is None else float(energy - previous_energy)
        rms = float(np.sqrt(np.mean((density - next_density)**2)))
        error = fock @ density @ overlap - overlap @ density @ fock
        residual = float(np.max(np.abs(error)))
        row = {"cycle": cycle, "total_energy": float(energy + nuclear),
               "delta_energy": delta, "density_rms": rms, "commutator_max": residual}
        records.append(row)
        if report is not None:
            report(row)
        converged = (delta is not None and abs(delta) < energy_tol
                     and rms < density_tol and residual < density_tol)
        if converged or cycle == max_cycles:
            break
        history.append((fock, error))
        history = history[-6:]
        if len(history) > 1:
            _, next_density = diagonalize(_diis(history))
        density = next_density
        previous_energy = energy

    return Result(converged, float(energy), float(nuclear), float(energy + nuclear),
                  orbital_energies.tolist(), occupied, [ao.label for ao in orbitals],
                  density.tolist(), float(np.sum(density * overlap)),
                  float(weights @ density_at(values, density)) if is_lda else None,
                  records, perf_counter() - start)
