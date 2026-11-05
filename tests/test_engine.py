from pathlib import Path

import numpy as np
import pytest

from opendft.basis import build_basis
from opendft.dft import lda_xc
from opendft.input import parse_input, read_input
from opendft.integrals import build_integrals
from opendft.scf import run_scf

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(("case", "energy", "electrons"), [
    ("he_rhf", -2.8077839566, 2),
    ("h2_rhf", -1.1167593075, 2),
    ("h2o_rhf", -74.9630631541, 10),
    ("he_lda", -2.7718860436, 2),
    ("h2_lda", -1.1212061155, 2),
    ("h2o_lda", -74.7321048628, 10),
])
def test_single_point(case, energy, electrons):
    result = run_scf(read_input(EXAMPLES / f"{case}.json"))
    assert result.converged
    assert len(result.iterations) >= 2
    assert result.total_energy == pytest.approx(energy, abs=2e-7, rel=0)
    assert result.electronic_energy + result.nuclear_repulsion == result.total_energy
    assert result.electron_count == pytest.approx(electrons, abs=1e-10, rel=0)
    assert np.all(np.diff(result.orbital_energies) >= 0)
    if case.endswith("lda"):
        assert result.grid_electron_count == pytest.approx(electrons, abs=2e-5, rel=0)


def test_integral_symmetry_and_positive_kinetic():
    molecule = read_input(EXAMPLES / "h2o_rhf.json")
    overlap, kinetic, attraction, eri = build_integrals(build_basis(molecule), molecule)
    for matrix in (overlap, kinetic, attraction):
        np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(overlap), 1)
    assert np.min(np.linalg.eigvalsh(kinetic)) > 0
    assert np.max(np.linalg.eigvalsh(attraction)) < 0
    np.testing.assert_allclose(eri, eri.swapaxes(0, 1))
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1))


def test_xc_potential_is_energy_density_derivative():
    rho = np.geomspace(1e-5, 100, 25)
    epsilon, potential = lda_xc(rho)
    step = 1e-5 * rho
    finite_difference = ((rho + step) * lda_xc(rho + step)[0]
                         - (rho - step) * lda_xc(rho - step)[0]) / (2 * step)
    np.testing.assert_allclose(potential, finite_difference, rtol=1e-8)
    assert np.all(epsilon < 0)


@pytest.mark.parametrize("changes", [
    {"charge": 1}, {"charge": -2}, {"multiplicity": 3}, {"method": "PBE0"},
    {"basis": "6-31G"}, {"atoms": [["He", 0, 0, float("nan")]]},
    {"atoms": [["He", 0, 0, 0], ["H", 0, 0, 0]]},
])
def test_invalid_input(changes):
    with pytest.raises(ValueError):
        parse_input({"atoms": [["He", 0, 0, 0]], **changes})
