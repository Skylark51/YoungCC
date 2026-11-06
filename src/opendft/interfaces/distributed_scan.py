"""Run independently seeded points of a relaxed bond scan on SSH nodes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
from queue import Empty, Queue
try:
    import resource
except ImportError:  # Worker execution is Linux-only; parsing/help remains portable.
    resource = None
import socket
import subprocess
import sys
from time import monotonic, process_time

import numpy as np

from ..inputs.molecule import ANGSTROM_PER_BOHR
from .scan_workers import _allocate_all_cores, _available, _available_memory_mb, _ssh
from .gaussian_job import read_job
from .geometry_workflow import _Surface, _checkpoint_metadata, _minimum, _write_checkpoint
from .pcm_profile import PCM_CAVITIES
from .worker_affinity import reserved_cpus, worker_hosts

_ARMIJO = ("projected-armijo", "projected-restricted-armijo")
_PROJECTED = ("projected", *_ARMIJO)


def _read_scan(path, pcm_cavity=None):
    job = read_job(path.read_text(encoding="utf-8"), {path}, base_directory=path.parent, pcm_cavity=pcm_cavity)
    if job is None or not job.scan or len(job.scan[0]) != 2 or job.freq:
        raise ValueError("Expected one YoungCC-supported relaxed bond scan without Freq.")
    if job.guess_read:
        for key, value in _checkpoint_metadata(job.molecule, job.pcm_cavity).items():
            if not np.array_equal(job.seed.get(key), value):
                raise ValueError(f"Checkpoint {key} differs from the scan input.")
    return job


def _publish_result(path, result):
    staging = path.with_suffix(".json.tmp")
    try:
        staging.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _point(path, index, cpus, memory, optimizer="slsqp", model_path=None, checkpoint_path=None,
           progress_path=None, resume_path=None, pcm_cavity=None, gradient_request=None, phase="", phase_path=""):
    if gradient_request:
        raise ValueError("Remote gradient offload is not part of the verified scan runner.")
    if resource is None or not hasattr(os, "sched_getaffinity"):
        raise ValueError("Scan workers require Linux CPU affinity support.")
    cpus = {int(cpu) for cpu in cpus.split(",")}
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise ValueError("Scan team affinity is outside the worker CPU set.")
    if (progress_path or resume_path) and optimizer not in _PROJECTED:
        raise ValueError("Scan progress checkpoints require the projected optimizer.")
    if optimizer in _ARMIJO and not model_path:
        raise ValueError("The projected Armijo optimizer requires an initial Hessian model.")
    os.sched_setaffinity(0, cpus)
    job = _read_scan(path, pcm_cavity or None)
    _, steps, _ = job.scan
    if not 0 <= index <= steps:
        raise ValueError("Scan point index is out of range.")
    with Path("/proc/meminfo").open(encoding="ascii") as info:
        available_kib = next(int(line.split()[1]) for line in info if line.startswith("MemAvailable:"))
    job = replace(job, molecule=replace(job.molecule, threads=len(cpus)),
                  memory_mb=min(memory, 0.75 * available_kib / 1024))
    if phase and (phase not in {"predict", "refine"} or not phase_path
                  or optimizer != "projected-restricted-armijo" or resume_path or gradient_request
                  or job.guess_read or job.molecule.multiplicity != 1):
        raise ValueError("Split phases require a fresh restricted-predictor scan point.")
    surface = _Surface(job, 80)
    if optimizer == "projected-restricted-armijo" and phase != "predict" and not gradient_request:
        from functools import partial

        from .symmetric_uks_gradient import gradient

        surface._gradient = partial(gradient, surface._gradient)
    _optimize_point(surface, path, index, cpus, optimizer, model_path, checkpoint_path, progress_path, resume_path, phase, phase_path)


def _optimize_point(surface, path, index, cpus, optimizer, model_path, checkpoint_path,
                    progress_path, resume_path, phase="", phase_path=""):
    job = surface.job
    (first, second), _, increment = job.scan
    xyz = job.molecule.centers.copy()
    delta = xyz[first] - xyz[second]
    distance = np.linalg.norm(delta)
    target = distance + index * increment / ANGSTROM_PER_BOHR
    if min(distance, target) <= 0:
        raise ValueError("Bond scan distances must be positive.")
    masses = surface.template.atom_mass_list()
    translation = delta * ((target - distance) / distance)
    xyz[first] += translation * masses[second] / (masses[first] + masses[second])
    xyz[second] -= translation * masses[first] / (masses[first] + masses[second])
    started = monotonic()
    if optimizer in _PROJECTED:
        from .bond_opt import minimize_bond, minimize_bond_armijo

        hessian = None
        if model_path:
            with np.load(model_path, allow_pickle=False) as model:
                if (not np.array_equal(model["symbols"], job.molecule.symbols)
                        or model["charge"].item() != job.molecule.charge
                        or model["multiplicity"].item() != job.molecule.multiplicity
                        or not np.isclose(model["target_bohr"].item(), target, atol=1e-8, rtol=0)):
                    raise ValueError("Initial scan model has the wrong atoms, state or target.")
                xyz = model["centers_bohr"]
                hessian = model["hessian"]
            if xyz.shape != job.molecule.centers.shape or not np.isfinite(xyz).all():
                raise ValueError("Initial scan model has invalid coordinates.")
        accepted = None
        if progress_path or resume_path:
            from functools import partial
            from .scan_progress import progress_metadata, restore_progress, write_progress

            metadata = progress_metadata(path, job.molecule, index, target, job.pcm_cavity)
            if resume_path:
                xyz = restore_progress(Path(resume_path), surface, metadata, (first, second))
            if progress_path:
                accepted = partial(write_progress, Path(progress_path), surface, metadata)
        if phase == "refine":
            with np.load(phase_path, allow_pickle=False) as seed:
                for key, value in _checkpoint_metadata(job.molecule, job.pcm_cavity).items():
                    if not np.array_equal(seed[key], value):
                        raise ValueError(f"Predictor {key} differs from the scan input.")
                xyz = seed["centers"]
                density = seed["density"]
                if (seed["index"].item() != index or seed["target"].item() != target
                        or xyz.size != job.molecule.centers.size or not np.isfinite(xyz).all()
                        or (density.size and (density.shape != (2, surface.template.nao, surface.template.nao)
                                              or not np.isfinite(density).all()))):
                    raise ValueError("Predictor state differs from the scan point.")
                surface.density = density if density.size else None
        if (phase != "refine" and optimizer == "projected-restricted-armijo" and not resume_path
                and job.molecule.multiplicity == 1 and not job.guess_read
                and surface.distributed_gradient is None):
            from .restricted_scan_seed import restricted_scan_seed

            xyz = restricted_scan_seed(surface, xyz, (first, second), target, hessian)
        if phase == "predict":
            _write_checkpoint(Path(phase_path), **_checkpoint_metadata(job.molecule, job.pcm_cavity),
                              index=index, target=target, centers=xyz,
                              density=surface.density if surface.density is not None else np.empty(0))
            print(json.dumps({"index": index, "host": socket.gethostname(), "cpus": sorted(cpus),
                              "optimization_s": monotonic() - started, "cpu_s": process_time()}), flush=True)
            return
        minimize_point = minimize_bond_armijo if optimizer in _ARMIJO else minimize_bond
        optimized = minimize_point(surface, xyz.ravel(), (first, second), target,
                                   initial_hessian=hessian, accepted=accepted)
    else:
        optimized = _minimum(surface, xyz.ravel(), ((first, second), target))
    energy, gradient, density = surface.evaluate(optimized)
    optimized = optimized.reshape(-1, 3)
    bond = optimized[first] - optimized[second]
    actual = np.linalg.norm(bond)
    constraint = np.zeros_like(optimized)
    constraint[first] = bond / actual
    constraint[second] = -constraint[first]
    normal = constraint.ravel()
    residual = gradient - normal * (gradient @ normal) / (normal @ normal)
    if checkpoint_path:
        _write_checkpoint(Path(checkpoint_path), **_checkpoint_metadata(job.molecule, job.pcm_cavity),
                          centers=optimized, density=density, energy=energy, gradient=gradient)
    row = {
        "index": index, "host": socket.gethostname(), "cpus": sorted(cpus),
        "target_angstrom": target * ANGSTROM_PER_BOHR,
        "actual_angstrom": actual * ANGSTROM_PER_BOHR,
        "total_energy": energy, "centers_bohr": optimized.tolist(),
        "max_projected_gradient": float(max(abs(residual))),
        "optimization_s": monotonic() - started, "cpu_s": process_time(),
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_mb": job.memory_mb,
        "resumed_progress": bool(resume_path),
        "pcm_cavity": job.pcm_cavity,
        "optimizer": optimizer,
    }
    if surface.distributed_gradient is not None:
        metrics = surface.distributed_gradient.gradient_metrics
        row["distributed_gradient_jk"] = metrics
        row["cpu_s"] += metrics["worker_cpu_s"]
    print(json.dumps(row, allow_nan=False), flush=True)


def run(path, output, hosts, threads, optimizer="slsqp", *, all_cores=False, order=None,
        model_directory=None, points=None, save_checkpoints=False, save_progress=False, resume_from=None,
        pcm_cavity=None, team_memory_mb=None, gradient_hosts=None, gradient_threads=4):
    started = monotonic()
    assignments = worker_hosts(hosts)
    if threads < 1:
        raise ValueError("Supply a positive team width.")
    if team_memory_mb is not None and (not np.isfinite(team_memory_mb) or team_memory_mb <= 0):
        raise ValueError("Team memory must be a finite positive number of MiB.")
    if (save_progress or resume_from is not None) and optimizer not in _PROJECTED:
        raise ValueError("Scan progress checkpoints require the projected optimizer.")
    if optimizer in _ARMIJO and model_directory is None:
        raise ValueError("The projected Armijo optimizer requires an initial Hessian model directory.")
    if resume_from is not None:
        resume_from = resume_from.resolve()
        if not resume_from.is_dir():
            raise ValueError("The scan resume directory does not exist.")
    if model_directory is not None:
        model_directory = model_directory.resolve()
    job = _read_scan(path, pcm_cavity)
    if team_memory_mb is not None and job.memory_mb is not None and team_memory_mb > job.memory_mb:
        raise ValueError("Team memory exceeds the scan input's %Mem allocation.")
    total = job.scan[1] + 1
    selected = list(range(total)) if points is None else list(points)
    if not selected or len(set(selected)) != len(selected) or not all(0 <= i < total for i in selected):
        raise ValueError("Scan points must be distinct indices within the input scan.")
    indices = selected if order is None else list(order)
    if sorted(indices) != sorted(selected):
        raise ValueError("Scan order must contain each selected target index exactly once.")
    count = len(indices)
    gradient_request = ""
    if gradient_hosts is not None:
        raise ValueError("Remote gradient offload is not part of the verified scan runner.")
    models = None
    if model_directory is not None:
        if optimizer not in _PROJECTED:
            raise ValueError("Initial models require the projected optimizer.")
        models = {index: model_directory / f"point-{index}.npz" for index in indices}
        for model in models.values():
            if not model.is_file():
                raise ValueError(f"Missing initial scan model: {model}")
    hosts = list(assignments)[:count]
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        affinities = [reserved_cpus(host, available, assignments[host])
                      for host, available in zip(hosts, pool.map(_available, hosts), strict=True)]
        budgets = dict(zip(hosts, (0.75 * available for available in pool.map(_available_memory_mb, hosts)),
                           strict=True))
    host_limits = None
    if team_memory_mb is not None:
        host_limits = [min(8, int(budgets[host] / team_memory_mb)) for host in hosts]
        for host, limit in zip(hosts, host_limits, strict=True):
            if limit < 1:
                raise ValueError(f"{host} has a {budgets[host]:.0f} MiB budget, below {team_memory_mb:g} MiB.")
    if all_cores:
        teams = _allocate_all_cores(hosts, affinities, count, threads, 8, host_limits=host_limits)
    else:
        if any(len(cpus) < threads for cpus in affinities):
            raise ValueError("A scan host has fewer available CPUs than the requested team width.")
        teams = [(host, cpus[:threads]) for host, cpus in zip(hosts, affinities, strict=True)]
    host_counts = Counter(host for host, _ in teams)
    pending = Queue()
    for index in indices[len(teams):]:
        pending.put(index)
    output.mkdir(parents=True)
    source = str(Path(__file__).resolve().parents[2])

    def team(item):
        slot, (host, cpus) = item
        index = indices[slot]
        memory = team_memory_mb or min(job.memory_mb or 4000, budgets[host] / host_counts[host])
        results = []
        while True:
            model = str(models[index]) if models is not None else ""
            checkpoint = str(output / f"point-{index}.npz") if save_checkpoints else ""
            name = f"point-{index}-progress.npz"
            progress = str(output / name) if save_progress or resume_from is not None else ""
            resume = str(resume_from / name) if resume_from is not None and (resume_from / name).is_file() else ""
            command = _ssh(
                host, "env", f"PYTHONPATH={source}", "PYTHONDONTWRITEBYTECODE=1",
                "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OMP_WAIT_POLICY=PASSIVE",
                f"OMP_NUM_THREADS={len(cpus)}", sys.executable, "-u", "-m",
                "opendft.interfaces.distributed_scan", "--worker", str(path),
                str(index), ",".join(map(str, cpus)), str(memory), optimizer,
                model, checkpoint, progress, resume, job.pcm_cavity, gradient_request,
            )
            launched = monotonic()
            with (output / f"point-{index}.stderr").open("x", encoding="utf-8") as err:
                response = subprocess.check_output(command, stdin=subprocess.DEVNULL, stderr=err, text=True)
            row = json.loads(response)
            if (row["index"] != index or row["host"].split(".")[0].lower() != host.split(".")[0].lower()
                    or row["cpus"] != cpus):
                raise RuntimeError(f"Unexpected scan worker response from {host}.")
            row["wall_s"] = monotonic() - launched
            _publish_result(output / f"point-{index}.json", row)
            results.append(row)
            try:
                index = pending.get_nowait()
            except Empty:
                return results

    with ThreadPoolExecutor(max_workers=len(teams)) as pool:
        rows = [row for result in pool.map(team, enumerate(teams)) for row in result]
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != sorted(selected):
        raise RuntimeError("Missing scan results.")
    payload = {
        "seed": ("Supplied initial geometries and Cartesian Hessians, refined on the original model."
                 if models is not None else
                 "Each point starts independently from the input geometry with the bond set to target."),
        "wall_s": monotonic() - started, "aggregate_cpu_s": sum(row["cpu_s"] for row in rows),
        "indices": list(job.scan[0]), "scan_points": rows,
        "pcm_cavity": job.pcm_cavity,
        "complete_scan": count == total,
    }
    _publish_result(output / "summary.json", payload)
    print(json.dumps({key: payload[key] for key in ("wall_s", "aggregate_cpu_s")}), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, path, index, cpus, memory, *options = sys.argv
        _point(Path(path).resolve(), int(index), cpus, float(memory), *options)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--hosts", nargs="+", required=True, metavar="HOST[=CPU,CPU]")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--optimizer", choices=("slsqp", *_PROJECTED), default="slsqp")
    parser.add_argument("--pcm-cavity", choices=PCM_CAVITIES, help="Override the C-PCM cavity profile")
    parser.add_argument("--all-cores", action="store_true",
                        help="Partition the participating hosts' available CPU affinities.")
    parser.add_argument("--team-memory-mb", type=float, metavar="MIB",
                        help="Reserve memory per scan team and limit concurrent teams by host memory.")
    parser.add_argument("--order", nargs="+", type=int, help="Optional order of all target indices.")
    parser.add_argument("--points", nargs="+", type=int, help="Run only the selected zero-based targets.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save the converged geometry, gradient and density for each target.")
    parser.add_argument("--save-progress", action="store_true",
                        help="Save the latest accepted exact evaluation for each projected scan point.")
    parser.add_argument("--resume-from", type=Path, metavar="DIRECTORY",
                        help="Resume selected points from saved progress in a previous results directory.")
    parser.add_argument("--model-directory", type=Path,
                        help="Directory of point-N.npz initial models for the projected optimizer.")
    args = parser.parse_args()
    run(args.input.resolve(), args.output_directory.resolve(), args.hosts, args.threads, args.optimizer,
        all_cores=args.all_cores, order=args.order,
        model_directory=args.model_directory, points=args.points, save_checkpoints=args.save_checkpoints,
        save_progress=args.save_progress, resume_from=args.resume_from, pcm_cavity=args.pcm_cavity,
        team_memory_mb=args.team_memory_mb)


if __name__ == "__main__":
    main()
