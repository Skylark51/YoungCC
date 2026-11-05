"""Validate the archived Gaussian logs' energy, molecular formula and basis."""
import pytest
from benchmarks.gaussian16.lion_reference_accuracy import (
    CASES, extract_reference, gaussian_orientations,
)

@pytest.mark.parametrize("case", list(CASES))
def test_reference_energy_geometry_and_pcm(case):
    record = extract_reference(case)
    assert record["reference_energy_hartree"] == CASES[case][2]
    assert record["basis"] == CASES[case][1]
    assert len(record["atoms"]) == 15
    assert record["molecule_formula"] == "C5H8O2"
    assert record["gaussian_cavity_radii"] == "UFF"
    assert record["dispersion"] is None

def test_reference_is_not_malonic_acid():
    record = extract_reference("ts_svp")
    symbols = [x[0] for x in record["atoms"]]
    assert symbols.count("C") == 5
    assert symbols.count("H") == 8
    assert symbols.count("O") == 2

def test_garbage_has_no_gaussian_orientation():
    assert gaussian_orientations("no Cartesian coordinates") == []
