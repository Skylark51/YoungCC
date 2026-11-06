"""Production entry point for independently seeded, SSH-distributed bond scans."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from .distributed_scan import _publish_result, _read_scan, run as run_scan
from .scan_workers import _available, _ssh
from .worker_affinity import reserved_cpus, worker_hosts


def run(path, directory, hosts, *, threads=16, memory_mb=4000, xtb=None, model_directory=None):
    started = monotonic()
    path, directory = path.resolve(), directory.resolve()
    if sys.platform != "linux":
        raise ValueError("Run SSH parallel scans on the Linux cluster with a shared working directory.")
    job = _read_scan(path)
    assignments = worker_hosts(hosts)
    if job.guess_read or job.molecule.multiplicity != 1:
        raise ValueError("The verified parallel scan requires a fresh singlet input.")
    if len(assignments) > job.scan[1] + 1:
        raise ValueError("There are more scan hosts than scan points.")
    if threads < 1 or not math.isfinite(memory_mb) or memory_mb <= 0:
        raise ValueError("Scan threads and memory must be positive.")
    if job.memory_mb is not None and memory_mb > job.memory_mb:
        raise ValueError("Scan worker memory exceeds the input %Mem allocation.")
    directory.mkdir(parents=True)
    source = str(Path(__file__).resolve().parents[2])
    xtb = xtb or os.environ.get("YOUNGCC_XTB")
    if model_directory is None:
        host = next(reversed(assignments))
        cpus = reserved_cpus(host, _available(host), assignments[host])[:min(4, threads)]
        if not cpus:
            raise ValueError("Seed host has no available CPUs.")
        command = _ssh(host, "env", f"PYTHONPATH={source}", "PYTHONDONTWRITEBYTECODE=1",
                       "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OMP_NUM_THREADS=4",
                       sys.executable, "-u", "-m", "opendft.interfaces.scan_models", str(path),
                       str(directory / "seeds"), "--cpus", ",".join(map(str, cpus)),
                       *(["--xtb", str(xtb)] if xtb else []))
        with (directory / "prepare.stdout").open("x") as output, (directory / "prepare.stderr").open("x") as error:
            try:
                subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output, stderr=error, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Scan seed preparation failed on {host}; see {directory / 'prepare.stderr'}") from exc
        model_directory = directory / "seeds/models"
    prepared = monotonic()
    run_scan(path, directory / "points", hosts, threads, "projected-restricted-armijo",
             model_directory=Path(model_directory), team_memory_mb=memory_mb,
             save_checkpoints=True, save_progress=True)
    summary = json.loads((directory / "points/summary.json").read_text())
    summary.update(input_to_result_s=monotonic()-started, preparation_s=prepared-started,
                   requested_hosts=list(assignments), threads_per_worker=threads)
    _publish_result(directory / "summary.json", summary)
    print(f"Parallel scan complete: {directory / 'summary.json'}", flush=True)
    return summary
