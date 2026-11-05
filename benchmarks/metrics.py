"""Compare energies, orbitals, density, convergence and timing."""

import numpy as np

from opendft.basis import build_basis, evaluate_basis
from opendft.dft import density_at


def errors(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    delta = np.abs(actual - reference)
    maximum = float(np.max(delta))
    scale = float(np.max(np.abs(reference)))
    return {"max_absolute": maximum, "relative_to_reference_max": maximum / scale if scale else None}


def compare(molecule, local, reference):
    orbitals = build_basis(molecule)
    # Fixed off-nuclear probes in bohr relative to each input nucleus.
    offsets = np.array([[0.17, 0.23, 0.31], [0.4, -0.6, 0.7], [-1.0, 0.8, -0.5]])
    points = (molecule.centers[:, None, :] + offsets).reshape(-1, 3)
    local_density = density_at(evaluate_basis(orbitals, points), local.density_matrix)
    energy = errors(local.total_energy, reference["total_energy"])
    return {
        "youngcc_total_energy_eh": local.total_energy,
        "reference_total_energy_eh": reference["total_energy"],
        "total_energy_error_eh": energy,
        "energy_target_eh": 1e-3,
        "energy_target_passed": (local.converged and reference["converged"]
                                 and energy["max_absolute"] <= 1e-3),
        "orbital_energy_error_eh": errors(local.orbital_energies, reference["orbital_energies"]),
        "electron_density_probe_error_e_per_bohr3": errors(local_density, reference["density"](points)),
        "density_probe_points_bohr": points.tolist(),
        "electron_count_error": errors(local.electron_count, reference["electron_count"]),
        "youngcc_analytic_electron_count": local.electron_count,
        "youngcc_grid_electron_count": local.grid_electron_count,
        "reference_electron_count": reference["electron_count"],
        "youngcc_scf": {"converged": local.converged, "cycles": len(local.iterations),
                        "seconds": local.seconds},
        "reference_scf": {key: reference[key] for key in ("converged", "cycles", "seconds")},
    }
