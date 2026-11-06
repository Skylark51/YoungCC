"""Integration boundaries for parallel scans, using immutable Lion inputs."""

import json
from pathlib import Path
import sys

import pytest

from opendft.interfaces import cli, parallel_scan
from opendft.interfaces.distributed_scan import _read_scan
from opendft.interfaces.scan_workers import _allocate_all_cores
from opendft.interfaces.worker_affinity import worker_hosts

REFERENCES = Path(__file__).resolve().parents[2] / "benchmarks/gaussian16/lion-test"
SCAN = REFERENCES / "MalonicAcid-scan.inp"


def test_cli_routes_parallel_scan_without_serial_calculation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(parallel_scan, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["youngcc", str(SCAN), "--scan-hosts", "lion38", "lion51",
                                    "--scan-directory", str(tmp_path / "results"), "--scan-threads", "8"])
    assert cli.main() == 0
    assert calls[0][0] == (SCAN, tmp_path / "results", ["lion38", "lion51"])
    assert calls[0][1]["threads"] == 8


def test_duplicate_host_aliases_and_overlapping_cpu_reservations_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        worker_hosts(["lion38", "lion38.example"])
    with pytest.raises(ValueError, match="duplicates"):
        worker_hosts(["lion38=0,0"])
    teams = _allocate_all_cores(["lion38"], [list(range(16))], 3, 4, 8)
    allocated = [cpu for _, cpus in teams for cpu in cpus]
    assert sorted(allocated) == list(range(16)) and len(set(allocated)) == len(allocated)


def test_invalid_scan_and_excess_hosts_fail_before_output_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(parallel_scan.sys, "platform", "linux")
    with pytest.raises(ValueError, match="relaxed bond scan"):
        _read_scan(REFERENCES / "n-butane-dihedral-scan.inp")
    directory = tmp_path / "scan"
    with pytest.raises(ValueError, match="more scan hosts"):
        parallel_scan.run(SCAN, directory, [f"lion{i}" for i in range(8)])
    assert not directory.exists()


def test_output_directory_is_exclusive_and_existing_models_skip_xtb(monkeypatch, tmp_path):
    monkeypatch.setattr(parallel_scan.sys, "platform", "linux")
    directory = tmp_path / "scan"
    models = tmp_path / "models"
    models.mkdir()
    def fake_scan(path, output, hosts, threads, optimizer, **kwargs):
        assert path == SCAN.resolve()
        assert optimizer == "projected-restricted-armijo"
        assert kwargs["model_directory"] == models
        assert kwargs["save_progress"] and kwargs["save_checkpoints"]
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"complete_scan": True}))
    monkeypatch.setattr(parallel_scan, "run_scan", fake_scan)
    monkeypatch.setattr(parallel_scan, "_available", lambda *_: pytest.fail("Unexpected seed preparation"))
    result = parallel_scan.run(SCAN, directory, ["lion51"], model_directory=models)
    assert result["complete_scan"] and result["input_to_result_s"] >= result["preparation_s"]
    with pytest.raises(FileExistsError):
        parallel_scan.run(SCAN, directory, ["lion51"], model_directory=models)


def test_worker_memory_limit_rejected_before_output_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(parallel_scan.sys, "platform", "linux")
    directory = tmp_path / "scan"
    with pytest.raises(ValueError, match="%Mem"):
        parallel_scan.run(SCAN, directory, ["lion51"], memory_mb=100000)
    assert not directory.exists()


def test_bond_scan_without_hosts_explains_required_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["youngcc", str(SCAN)])
    assert cli.main() == 2
    assert "Bond scans require --scan-hosts" in capsys.readouterr().err
