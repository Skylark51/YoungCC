"""Gaussian CLI workflow regressions against the committed Lion inputs."""

from pathlib import Path

import numpy as np
import pytest

from opendft.interfaces.gaussian_job import read_job
from opendft.interfaces.geometry_workflow import _dihedral, _write_checkpoint

REFERENCES = Path(__file__).resolve().parents[2] / "benchmarks" / "gaussian16" / "lion-test"


def test_lion_dihedral_scan_input():
    job = read_job((REFERENCES / "n-butane-dihedral-scan.inp").read_text(), set())
    assert job.opt == {"modredundant"}
    assert job.scan == ((0, 1, 2, 3), 12, 10.0)
    assert abs(np.rad2deg(_dihedral(job.molecule.centers, job.scan[0]))) == pytest.approx(60, abs=2)
    assert job.molecule.threads == 16


def test_lion_frequency_checkpoint_and_ts_seed(tmp_path):
    frequency = (REFERENCES / "n-butane-dihderal-scan-freq.inp").read_text()
    job = read_job(frequency, set())
    assert job.freq and not job.opt
    assert job.memory_mb == 60 * 1024
    assert job.conventional_cycles == 60

    # Use the real frequency input's geometry to test checkpoint handoff independently of SCF.
    seed = tmp_path / "seed.chk"
    _write_checkpoint(seed, symbols=job.molecule.symbols, centers=job.molecule.centers)
    ts = (REFERENCES / "n-butane-dihderal-scan-freq-ts.inp").read_text()
    ts = "\n".join(f"%oldchk={seed}" if line.startswith("%oldchk=") else line
                   for line in ts.splitlines())
    followup = read_job(ts, set())
    assert followup.opt == {"nofreeze", "ts", "rcfc", "noeigentest"}
    assert followup.guess_read
    assert followup.checkpoint.name == "n-butane-dihderal-scan-freq-ts.chk"
    np.testing.assert_allclose(followup.molecule.centers, job.molecule.centers)


def test_checkpoint_cannot_overwrite_input():
    frequency = (REFERENCES / "n-butane-dihderal-scan-freq.inp").read_text()
    job = read_job(frequency, set())
    with pytest.raises(ValueError, match="overlaps"):
        read_job(frequency, {job.checkpoint})


def test_opt_freq_cli_writes_checkpoint(tmp_path, monkeypatch):
    import json
    import sys

    pyscf = pytest.importorskip("pyscf")
    from opendft.interfaces.cli import main

    example = REFERENCES.parents[2] / "examples" / "h2o_rhf.json"
    water = json.loads(example.read_text())
    rows = [" ".join(map(str, atom)) for atom in water["atoms"]]
    inp = tmp_path / "water.inp"
    checkpoint = tmp_path / "water.chk"
    inp.write_text(f"%chk={checkpoint}\n%nprocshared=1\n#p b3lyp/def2svp opt freq\n\nWater\n\n0 1\n"
                   + "\n".join(rows) + "\n\n")
    monkeypatch.setattr(sys, "argv", ["youngcc", str(inp)])
    threads = pyscf.lib.num_threads()
    try:
        assert main() == 0
    finally:
        pyscf.lib.num_threads(threads)

    result = json.loads(inp.with_suffix(".youngcc.json").read_text())
    assert len(result["frequencies_cm-1"]) == 3
    assert min(result["frequencies_cm-1"]) > 0
    with np.load(checkpoint, allow_pickle=False) as seed:
        assert seed["youngcc_checkpoint"].item() == 1
        assert seed["hessian"].shape == (9, 9)
        np.testing.assert_allclose(seed["centers"], result["centers_bohr"])
