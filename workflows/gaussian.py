"""Extract only the Gaussian 16 records needed by the reaction workflow."""

import re
from dataclasses import dataclass

import numpy as np

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?"
TABLE = re.compile(
    r"(?P<kind>(?:Input|Standard|Z-Matrix) orientation|Cartesian Coordinates \(Ang\)):"
    r"\s*^[^\S\n]*-+[^\S\n]*\n(?:[^\n]*\n){2}\s*^[^\S\n]*-+[^\S\n]*\n"
    r"(?P<rows>(?:[^\n]*\n)+?)\s*^[^\S\n]*-+[^\S\n]*$",
    re.M,
)
EVENT = re.compile(
    rf"SCF Done:[^\n=]*=\s*(?P<energy>{NUMBER})|(?P<converged>Optimization completed(?: on the basis of negligible forces)?\.)"
)
TERMINATION = re.compile(r"^\s*(Normal termination of Gaussian|Error termination)\b", re.M)


def number(value):
    return float(value.upper().replace("D", "E"))


@dataclass
class Geometry:
    atoms: tuple[int, ...]
    xyz: np.ndarray

    def gaussian(self):
        return "\n".join(
            f"{atom:3d} {x: .10f} {y: .10f} {z: .10f}"
            for atom, (x, y, z) in zip(self.atoms, self.xyz, strict=True)
        )

    @property
    def modes(self):
        rank = np.linalg.matrix_rank(self.xyz - self.xyz.mean(axis=0), tol=1e-5)
        return 3 * len(self.atoms) - (5 if rank == 1 else 6)


class GaussianLog:
    def __init__(self, text, atoms):
        self.text = text
        self.tables = []
        self.points = []
        self.energies = []
        energy = None
        geometry = None
        events = [(m.start(), m) for m in TABLE.finditer(text)]
        events.extend((m.start(), m) for m in EVENT.finditer(text))
        for offset, match in sorted(events, key=lambda event: event[0]):
            if match.re is TABLE:
                rows = [line.split() for line in match["rows"].splitlines()]
                width = 5 if match["kind"].startswith("Cartesian") else 6
                if not rows or any(len(row) != width for row in rows):
                    continue
                if [int(row[0]) for row in rows] != list(range(1, len(atoms) + 1)):
                    continue
                if tuple(int(row[1]) for row in rows) != atoms:
                    continue
                xyz = np.array([[number(value) for value in row[-3:]] for row in rows])
                if np.isfinite(xyz).all():
                    geometry = Geometry(atoms, xyz)
                    self.tables.append((offset, match["kind"], geometry))
            elif match["energy"]:
                energy = number(match["energy"])
                self.energies.append(energy)
            elif energy is not None and geometry is not None:
                self.points.append((energy, geometry))
                energy = geometry = None

    @property
    def normal(self):
        endings = TERMINATION.findall(self.text)
        return bool(endings and endings[-1].startswith("Normal"))

    def peak(self, epsilon=1e-6):
        peaks = [
            point for left, point, right in zip(self.points, self.points[1:], self.points[2:])
            if point[0] - max(left[0], right[0]) > epsilon
        ]
        return max(peaks, key=lambda point: point[0]) if peaks else None

    def frequencies(self, geometry):
        values = [
            number(value)
            for row in re.findall(r"^\s*Frequencies --\s*([^\n]+)", self.text, re.M)
            for value in row.split()
        ]
        if len(values) != geometry.modes or not values or not np.isfinite(values).all():
            raise ValueError(f"incomplete frequency result ({len(values)}/{geometry.modes})")
        return values

    def irc_endpoint(self, direction):
        completion = self.text.find(f"Calculation of {direction.upper()} path complete.")
        if completion < 0 or "Reaction path calculation complete." not in self.text[completion:]:
            raise ValueError(f"{direction} IRC did not complete")
        prefix = self.text[:completion]
        if "PES minimum detected" not in prefix:
            raise ValueError(f"{direction} IRC did not reach an LQA PES minimum")
        accepted = list(re.finditer(r"NET REACTION COORDINATE UP TO THIS POINT\s*=\s*(" + NUMBER + ")", prefix))
        if not accepted or abs(number(accepted[-1][1])) <= 1e-6:
            raise ValueError(f"{direction} IRC did not leave the TS")
        start = accepted[-2].end() if len(accepted) > 1 else 0
        for offset, kind, geometry in reversed(self.tables):
            if start < offset < accepted[-1].start() and kind == "Cartesian Coordinates (Ang)":
                return geometry
        raise ValueError(f"{direction} IRC has no geometry for the last accepted point")


def diagnose(log, returncode):
    patterns = (
        (r"Convergence failure|SCF has not converged", "SCF convergence"),
        (r"Optimization stopped|Number of steps exceeded", "optimization step limit"),
        (r"Wrong number of Negative eigenvalues", "wrong Hessian index"),
        (r"No space left|Erroneous write|FileIO operation", "scratch/checkpoint I/O"),
        (r"out of memory|not enough memory|galloc", "memory allocation"),
        (r"Error termination request processed by link 9999", "Gaussian link 9999"),
    )
    for pattern, cause in patterns:
        if re.search(pattern, log.text, re.I):
            return cause
    return f"missing normal termination or nonzero exit (exit={returncode})"


def distance(a, b):
    """Atom-mapped, proper-rotation RMSD and largest interatomic distance error."""
    x, y = a.xyz - a.xyz.mean(axis=0), b.xyz - b.xyz.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ vt)
    rmsd = np.sqrt(np.mean(np.sum((x @ u @ vt - y) ** 2, axis=1)))
    dx = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    dy = np.linalg.norm(y[:, None] - y[None, :], axis=2)
    return float(rmsd), float(np.max(np.abs(dx - dy)))
