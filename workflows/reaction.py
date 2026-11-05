"""Run a checkpointed Gaussian Opt/Freq/Scan/TS/IRC reaction path."""

import argparse
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np

from .gaussian import GaussianLog, Geometry, diagnose, distance

METHOD = re.compile(r"(?:[RU]?)(?:HF|B3LYP|PBE1PBE|M06-2X|wB97XD)/[A-Za-z0-9+*(),-]+", re.I)
OPTION = re.compile(r"(?:scrf|int|empiricaldispersion)=[A-Za-z0-9(),=+-]+", re.I)


def xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line.split() for line in lines[2:] if line.strip()]
    if len(lines) < 2 or int(lines[0]) != len(rows) or len(rows) < 2:
        raise ValueError(f"{path}: invalid XYZ atom count")
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"{path}: expected an element and three coordinates per atom")
    # Gaussian accepts atomic numbers in place of element symbols.
    from opendft.inputs.molecule import ATOMIC_NUMBERS

    atoms = tuple(int(row[0]) if row[0].isdigit() else ATOMIC_NUMBERS[row[0].capitalize()] for row in rows)
    coordinates = np.array([[float(value) for value in row[1:]] for row in rows])
    if any(not 1 <= atom <= 118 for atom in atoms) or not np.isfinite(coordinates).all():
        raise ValueError(f"{path}: invalid atom or coordinate")
    if np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0), tol=1e-5) == 0:
        raise ValueError(f"{path}: coincident geometry")
    return Geometry(atoms, coordinates)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"invalid {name}")
    return value


@dataclass
class Result:
    checkpoint: Path
    value: object


class RetryableStageError(ValueError):
    """A calculation failed after its configured attempts."""


class WrongHessianIndex(ValueError):
    """A complete frequency calculation found the wrong stationary point."""


class Workflow:
    def __init__(self, config_path, output, executable):
        settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
        self.start, self.reactant, self.product = [
            xyz(config_path.parent / settings[key]) for key in ("start", "reactant", "product")
        ]
        require(self.start.atoms == self.reactant.atoms == self.product.atoms, "XYZ atom order differs")
        self.method = settings["method"]
        require(METHOD.fullmatch(self.method), "unsupported Gaussian method/basis")
        self.options = settings.get("options", [])
        require(isinstance(self.options, list) and all(OPTION.fullmatch(option) for option in self.options),
                "options must be SCRF, Int or EmpiricalDispersion route tokens")
        self.charge, self.multiplicity = settings["charge"], settings["multiplicity"]
        require(type(self.charge) is int and type(self.multiplicity) is int and self.multiplicity > 0,
                "invalid charge or multiplicity")
        electrons = sum(self.start.atoms) - self.charge
        require(electrons >= self.multiplicity - 1 and (electrons - self.multiplicity + 1) % 2 == 0,
                "charge and multiplicity are incompatible with electron count")
        self.cores, self.memory = settings.get("cores", 1), settings.get("memory", "2GB")
        require(type(self.cores) is int and self.cores > 0 and re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", self.memory, re.I),
                "invalid cores or memory")
        scan = settings["scan"]
        atoms, steps, step = scan["atoms"], scan["steps"], scan["step"]
        require(len(atoms) in (2, 3, 4) and all(type(i) is int and 1 <= i <= len(self.start.atoms) for i in atoms)
                and len(set(atoms)) == len(atoms), "scan requires 2-4 distinct, one-based atom indices")
        require(type(steps) is int and steps >= 2 and type(step) in (int, float) and math.isfinite(step) and step != 0,
                "invalid scan steps or increment")
        coordinate = {2: "B", 3: "A", 4: "D"}[len(atoms)]
        self.scan = f"{coordinate} {' '.join(map(str, atoms))} S {steps} {step:.12g}"
        self.imaginary = positive(settings.get("imaginary_cm", 50), "imaginary_cm")
        self.rmsd = positive(settings.get("rmsd_angstrom", 0.15), "rmsd_angstrom")
        self.pair = positive(settings.get("pair_angstrom", 0.35), "pair_angstrom")
        require(not self.matches(distance(self.reactant, self.product)), "reference endpoints overlap within tolerances")
        self.output = output
        self.executable = executable
        self.spec = {
            "schema": 1, "config": settings, "executable": executable,
            "geometries": [g.gaussian() for g in (self.start, self.reactant, self.product)],
        }

    def matches(self, metrics):
        return metrics[0] <= self.rmsd and metrics[1] <= self.pair

    def stage(self, name, geometry, routes, validate, *, parent=None, extra="", recovery=False):
        directory = self.output / name
        directory.mkdir(exist_ok=True)
        provenance = {
            "geometry": geometry.gaussian(), "routes": routes, "extra": extra,
            "parent": str(parent.checkpoint.relative_to(self.output)) if parent else None,
            "recovery": recovery,
        }
        key = sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:24]
        success = directory / f"success-{key}.json"
        if success.exists():
            job = directory / json.loads(success.read_text())["attempt"]
            checkpoint = job / "job.chk"
            log = GaussianLog((job / "job.out").read_text(errors="replace"), geometry.atoms)
            require(log.normal and checkpoint.stat().st_size > 0, f"{name}: saved attempt is incomplete")
            return Result(checkpoint, validate(log))

        failure = None
        for index, route in enumerate(routes):
            job = Path(tempfile.mkdtemp(prefix="attempt-", dir=directory))
            checkpoint = job / "job.chk"
            if parent is not None:
                shutil.copyfile(parent.checkpoint, job / "seed.chk")
            scf = "scf=(xqc,maxcycle=512)" if index or recovery else "scf=(xqc,maxcycle=128)"
            keywords = " ".join([self.method, *self.options, "nosymm", scf, route])
            link = "%oldchk=seed.chk\n" if parent is not None else ""
            coords = geometry.gaussian() + "\n" if parent is None else ""
            deck = (
                f"{link}%chk=job.chk\n%nprocshared={self.cores}\n%mem={self.memory}\n"
                f"#p {keywords}\n\n{name}\n\n{self.charge} {self.multiplicity}\n{coords}\n"
            )
            if extra:
                deck += f"{extra}\n\n"
            (job / "job.inp").write_text(deck, encoding="ascii")
            print(f"{name}: running {job.name}", file=sys.stderr, flush=True)
            with (job / "job.inp").open("rb") as stdin, (job / "job.out").open("wb") as stdout:
                process = subprocess.run([self.executable], cwd=job, stdin=stdin, stdout=stdout,
                                         stderr=subprocess.STDOUT, check=False)
            log = GaussianLog((job / "job.out").read_text(errors="replace"), geometry.atoms)
            try:
                if process.returncode != 0 or not log.normal:
                    raise ValueError(diagnose(log, process.returncode))
                require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "missing checkpoint")
                value = validate(log)
            except ValueError as error:
                failure = str(error)
                atomic_json(job / "failure.json", {"reason": failure, "exit_code": process.returncode})
                print(f"{name}: {failure} ({job.name})", file=sys.stderr, flush=True)
                if isinstance(error, WrongHessianIndex):
                    raise
                if failure in {"memory allocation", "scratch/checkpoint I/O"}:
                    raise ValueError(f"{name}: {failure}; inspect {directory}") from error
            else:
                atomic_json(success, {"attempt": job.name, **provenance})
                return Result(checkpoint, value)
        raise RetryableStageError(f"{name}: {failure}; inspect {directory}")

    @staticmethod
    def optimized(log):
        require(log.points, "optimization did not converge")
        return log.points[-1][1]

    def frequency(self, name, geometry, expected, parent=None):
        def check(log):
            values = log.frequencies(geometry)
            negatives = [value for value in values if value < 0]
            if len(negatives) != expected or (negatives and negatives[0] > -self.imaginary):
                raise WrongHessianIndex(f"expected {expected} imaginary mode(s), found {negatives}")
            return values

        route = "freq" if parent is None else "freq geom=check guess=read"
        return self.stage(name, geometry, [route, route], check, parent=parent)

    def minimum(self, name, geometry):
        for repair in (False, True):
            stem = f"{name}-repair" if repair else name
            routes = ["opt=(calcfc,tight,maxstep=5,maxcycles=200)"]
            if repair:
                routes = ["opt=(calcall,verytight,maxstep=3,maxcycles=200)"]
            else:
                routes.insert(0, "opt=(calcfc,tight,maxcycles=100)")
            optimized = self.stage(f"{stem}-opt", geometry, routes, self.optimized)
            geometry = optimized.value
            try:
                frequency = self.frequency(f"{stem}-freq", geometry, 0, optimized)
            except WrongHessianIndex:
                if repair:
                    raise
            else:
                return geometry, frequency.value

    def transition_state(self, seed, seed_freq, validate):
        routes = [
            "opt=(ts,rcfc,noeigentest,maxcycles=100) geom=check guess=read",
            "opt=(ts,calcfc,noeigentest,maxstep=5,maxcycles=200) geom=check guess=read",
        ]
        for attempt, route in enumerate(routes):
            stem = "ts-repair" if attempt else "ts"
            try:
                ts = self.stage(f"{stem}-opt", seed, [route], validate,
                                parent=seed_freq, recovery=bool(attempt))
                frequency = self.frequency(f"{stem}-freq", ts.value, 1, ts)
            except (RetryableStageError, WrongHessianIndex):
                if attempt:
                    raise
            else:
                return ts, frequency

    def run(self):
        start, _ = self.minimum("initial", self.start)

        def scan_peak(log):
            peak = log.peak()
            require(peak is not None, "scan has no bracketed energy maximum")
            return peak

        scan = self.stage(
            "scan", start,
            ["opt=(modredundant,calcfc,maxcycles=100)", "opt=(modredundant,calcfc,maxstep=5,maxcycles=200)"],
            scan_peak, extra=self.scan,
        )
        scan_energy, seed = scan.value
        seed_freq = self.frequency("scan-freq", seed, 1)

        def ts_geometry(log):
            geometry = self.optimized(log)
            drop = (log.energies[0] - min(log.energies)) * 627.509474
            require(drop < 15, f"TS search descended {drop:.2f} kcal/mol from its seed")
            return geometry

        ts, ts_freq = self.transition_state(seed, seed_freq, ts_geometry)
        endpoints = {}
        for direction in ("forward", "reverse"):
            irc = self.stage(
                f"irc-{direction}", ts.value,
                [f"irc=(rcfc,{direction},lqa,stepsize={step},maxpoints={points},maxcycle={cycles}) geom=check guess=read"
                 for step, points, cycles in [(25, 60, 30), (10, 150, 60)]],
                lambda log: log.irc_endpoint(direction), parent=ts_freq,
            )
            geometry, frequencies = self.minimum(direction, irc.value)
            endpoints[direction] = {
                "frequencies_cm-1": frequencies,
                "reactant": distance(geometry, self.reactant),
                "product": distance(geometry, self.product),
            }
        assignments = [
            {"forward": first, "reverse": second}
            for first, second in (("reactant", "product"), ("product", "reactant"))
            if self.matches(endpoints["forward"][first]) and self.matches(endpoints["reverse"][second])
            and not self.matches(endpoints["forward"][second]) and not self.matches(endpoints["reverse"][first])
        ]
        atomic_json(self.output / "validation.json", {
            "verified": len(assignments) == 1, "assignments": assignments,
            "scan_peak_hartree": scan_energy, "ts_frequencies_cm-1": ts_freq.value,
            "endpoint_distances_and_frequencies": endpoints,
        })
        require(len(assignments) == 1, "IRC endpoints do not uniquely match both references; see validation.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--g16", default="g16", help="Gaussian executable or local launcher")
    args = parser.parse_args()
    try:
        executable = shutil.which(args.g16)
        require(executable is not None, f"Gaussian executable not found: {args.g16}")
        workflow = Workflow(args.config, args.output.resolve(), str(Path(executable).resolve()))
        workflow.output.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(workflow.output / "lock.sqlite", timeout=0)) as lock:
            lock.execute("BEGIN EXCLUSIVE")
            spec = workflow.output / "spec.json"
            if spec.exists():
                require(json.loads(spec.read_text()) == workflow.spec,
                        "configuration changed; use a new output directory")
            else:
                require(all(path.name.startswith("lock.sqlite") for path in workflow.output.iterdir()),
                        "output directory is not empty")
                atomic_json(spec, workflow.spec)
            workflow.run()
    except (ValueError, TypeError, OSError, KeyError, sqlite3.Error) as error:
        parser.exit(1, f"workflow: {error}\n")


if __name__ == "__main__":
    main()
