"""Generate independent xTB geometry/Hessian seeds for exact DFT scan refinement."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ..inputs.molecule import ANGSTROM_PER_BOHR
from .distributed_scan import _read_scan
from .geometry_workflow import _Surface


def find_xtb(executable=None):
    candidate = executable or os.environ.get("YOUNGCC_XTB") or shutil.which("xtb")
    if candidate is None:
        candidate = Path.home() / "software/xtb/xtb-dist/bin/xtb"
    candidate = str(candidate)
    if not Path(candidate).is_file():
        candidate = shutil.which(candidate)
    if not candidate:
        raise ValueError("xTB is required for automatic scan seeds; set YOUNGCC_XTB or --scan-xtb.")
    return str(Path(candidate).resolve())


def prepare(path, directory, *, executable=None, threads=4, cpus=None):
    job = _read_scan(path)
    molecule = job.molecule
    if molecule.multiplicity != 1 or job.guess_read:
        raise ValueError("Automatic restricted scan seeds require a fresh singlet input.")
    if threads < 1:
        raise ValueError("xTB thread count must be positive.")
    if cpus is not None:
        if not set(cpus) <= os.sched_getaffinity(0) or not cpus:
            raise ValueError("Seed CPU reservation is outside the worker affinity.")
        os.sched_setaffinity(0, set(cpus))
        threads = min(threads, len(cpus))
    xtb = find_xtb(executable)
    env = os.environ | {"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "256M"}
    share = Path(xtb).parent.parent / "share/xtb"
    if "XTBPATH" not in env and share.is_dir():
        env["XTBPATH"] = str(share)
    directory.mkdir(parents=True)
    models = directory / "models"
    models.mkdir()
    first, second = job.scan[0]
    masses = _Surface(job, 80).template.atom_mass_list()
    for index in range(job.scan[1] + 1):
        work = directory / f"seed-{index}"
        work.mkdir()
        xyz = molecule.centers.copy()
        delta = xyz[first] - xyz[second]
        distance = np.linalg.norm(delta)
        target = distance + index * job.scan[2] / ANGSTROM_PER_BOHR
        if target <= 0:
            raise ValueError("Scan target distance must be positive.")
        shift = delta * ((target - distance) / distance)
        xyz[first] += shift * masses[second] / (masses[first] + masses[second])
        xyz[second] -= shift * masses[first] / (masses[first] + masses[second])
        (work / "point.xyz").write_text(f"{len(xyz)}\nProjected input geometry\n" + "".join(
            f"{symbol} {x:.12f} {y:.12f} {z:.12f}\n"
            for symbol, (x, y, z) in zip(molecule.symbols, xyz * ANGSTROM_PER_BOHR)), encoding="utf-8")
        (work / "constraint").write_text(
            f"$constrain\n  force constant=1.0\n  distance: {first+1}, {second+1}, "
            f"{target*ANGSTROM_PER_BOHR:.12f}\n$end\n", encoding="utf-8")
        common = ["--gfn", "2", "--chrg", str(molecule.charge), "--uhf", "0", "--parallel", str(threads)]
        if molecule.solvent:
            common += ["--alpb", molecule.solvent]
        for stage, geometry, options in [("opt", "point.xyz", ["--opt", "--input", "constraint"]),
                                          ("hess", "xtbopt.xyz", ["--hess"])]:
            with (work / f"{stage}.out").open("x") as output, (work / f"{stage}.stderr").open("x") as error:
                subprocess.run([xtb, geometry, *common, *options], cwd=work, env=env,
                               stdin=subprocess.DEVNULL, stdout=output, stderr=error, check=True)
            if (work / "NOT_CONVERGED").exists() or not (work / "xtbopt.xyz").is_file():
                raise RuntimeError(f"xTB did not converge for scan point {index}; see {work}.")
        rows = [line.split() for line in (work / "xtbopt.xyz").read_text().splitlines()[2:2+len(xyz)]]
        if [line[0].capitalize() for line in rows] != list(molecule.symbols):
            raise ValueError("xTB output atom order differs from the scan input.")
        actual = np.asarray([[float(v) for v in line[1:4]] for line in rows]) / ANGSTROM_PER_BOHR
        # Preserve the input orientation used by the verified benchmark. The
        # projected seed is only the xTB starting geometry, not the alignment
        # reference: rotating the DFT grid can otherwise change finite-grid results.
        reference = molecule.centers
        a, b = actual - actual.mean(0), reference - reference.mean(0)
        u, _, v = np.linalg.svd(a.T @ b)
        rotation = (u * np.array([1., 1., np.linalg.det(u @ v)])) @ v
        aligned = a @ rotation + reference.mean(0)
        rmsd = np.sqrt(np.mean(np.sum((aligned - reference)**2, axis=1))) * ANGSTROM_PER_BOHR
        if not np.isfinite(rmsd) or rmsd >= .25:
            raise ValueError(f"xTB seed moved too far from scan point {index}: {rmsd:g} Angstrom.")
        values = [float(value.replace("D", "E")) for line in (work / "hessian").read_text().splitlines()
                  if not line.lstrip().startswith("$") for value in line.split()]
        hessian = np.asarray(values).reshape(xyz.size, xyz.size)
        if not np.isfinite(hessian).all() or not np.allclose(hessian, hessian.T, atol=1e-6):
            raise ValueError("Invalid xTB Hessian.")
        transform = np.kron(np.eye(len(xyz)), rotation)
        np.savez(models / f"point-{index}.npz", symbols=molecule.symbols, charge=molecule.charge,
                 multiplicity=molecule.multiplicity, target_bohr=target, centers_bohr=aligned,
                 hessian=transform.T @ hessian @ transform)
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--xtb")
    parser.add_argument("--cpus", required=True)
    args = parser.parse_args()
    prepare(args.input.resolve(), args.directory.resolve(), executable=args.xtb,
            cpus=[int(cpu) for cpu in args.cpus.split(",")])
