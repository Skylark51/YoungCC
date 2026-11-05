"""Verify advanced inputs, Gaussian routes and optional PySCF calculations."""
import json
from pathlib import Path
import pytest
from opendft.gaussian_input import parse_gaussian
from opendft.input import parse_input

OH = {
    "method": "UB3LYP", "basis": "def2-SVP", "charge": 0, "multiplicity": 2,
    "atoms": [["O", 0, 0, 0], ["H", 0, 0, 0.97]],
}

def test_advanced_input():
    mol = parse_input({**OH, "basis": "def2-TZVPP", "solvent": "MeCN",
                       "dispersion": "GD3BJ"})
    assert (mol.electrons, mol.multiplicity, mol.basis) == (9, 2, "DEF2-TZVPP")
    assert (mol.solvent, mol.dispersion) == ("acetonitrile", "D3BJ")

def test_gaussian_input():
    route = """%nprocshared=8
#p UB3LYP/def2SVP SCRF=(CPCM,Solvent=Acetonitrile) EmpiricalDispersion=GD3BJ

OH radical

0 2
O 0.0 0.0 0.0
H 0.0 0.0 0.97

"""
    mol = parse_gaussian(route)
    assert mol.electrons == 9
    assert mol.solvent == "Acetonitrile"
    assert mol.dispersion == "D3BJ"

@pytest.mark.parametrize("option", ["Opt", "Freq", "IRC", "Guess=Read",
                                    "EmpiricalDispersion=GD3"])
def test_unsupported_route(option):
    with pytest.raises(ValueError):
        parse_gaussian("#p UB3LYP/def2SVP " + option +
                       "\n\nTitle\n\n0 2\nO 0 0 0\nH 0 0 1\n")

@pytest.mark.parametrize("change", [
    {"multiplicity": 1}, {"multiplicity": 3}, {"dielectric": 1},
    {"solvent": ""}, {"dispersion": "GD3"}, {"grid_level": 12},
])
def test_bad_advanced_input(change):
    with pytest.raises(ValueError):
        parse_input({**OH, **change})

@pytest.mark.advanced
def test_uks_both_basis_sets():
    pytest.importorskip("pyscf")
    from opendft.advanced import run_advanced
    for basis in ("def2-SVP", "def2-TZVPP"):
        result = run_advanced(parse_input({**OH, "basis": basis}), max_cycles=60)
        assert result.converged
        assert result.occupied_orbitals == {"alpha": 5, "beta": 4}
        assert result.electron_count == pytest.approx(9, abs=1e-6)
        assert -100 < result.total_energy < -60

@pytest.mark.advanced
def test_pcm_and_d3bj_corrections():
    pytest.importorskip("pyscf")
    pytest.importorskip("pyscf.dispersion")
    from opendft.advanced import run_advanced
    gas = run_advanced(parse_input(OH), max_cycles=60)
    pcm = run_advanced(parse_input({**OH, "solvent": "MeCN"}), max_cycles=60)
    d3 = run_advanced(parse_input({**OH, "dispersion": "GD3BJ"}), max_cycles=60)
    assert all(r.converged for r in (gas, pcm, d3))
    assert abs(gas.total_energy - pcm.total_energy) > 1e-6
    assert d3.dispersion_energy < 0
    assert d3.total_energy - gas.total_energy == pytest.approx(d3.dispersion_energy, abs=2e-6)

@pytest.mark.advanced
def test_advanced_output(tmp_path):
    pytest.importorskip("pyscf")
    from opendft.output import calculate
    prefix = tmp_path / "oh"
    result = calculate(parse_input(OH), prefix, emit=lambda _: None, max_cycles=60)
    data = json.loads(Path(str(prefix) + ".json").read_text())
    log = Path(str(prefix) + ".log").read_text()
    assert result.converged
    assert data["occupied_orbitals"] == {"alpha": 5, "beta": 4}
    assert "YOUNGCC FINAL SUMMARY" in log
