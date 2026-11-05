"""Normalized Cartesian STO-3G contractions for H, He and O."""

from dataclasses import dataclass
from math import pi

import numpy as np

CORE = (0.1543289673, 0.5353281423, 0.4446345422)
O_VALENCE = (5.033151319, 1.169596125, 0.3803889600)
SHELLS = {
    "H": [((3.425250914, 0.6239137298, 0.1688554040), CORE, 0)],
    "He": [((6.362421394, 1.158922999, 0.3136497915), CORE, 0)],
    "O": [
        ((130.7093214, 23.80886605, 6.443608313), CORE, 0),
        (O_VALENCE, (-0.09996722919, 0.3995128261, 0.7001154689), 0),
        (O_VALENCE, (0.1559162750, 0.6076837186, 0.3919573931), 1),
    ],
}


@dataclass(frozen=True)
class Orbital:
    center: tuple[float, float, float]
    powers: tuple[int, int, int]
    primitives: tuple[tuple[float, float], ...]
    label: str


def build_basis(molecule):
    orbitals = []
    for index, (symbol, center) in enumerate(zip(molecule.symbols, molecule.centers), 1):
        for shell, (exponents, coefficients, angular) in enumerate(SHELLS[symbol], 1):
            a = np.array(exponents)
            weights = np.array(coefficients) * (2 * a / pi)**0.75 * (4 * a)**(angular / 2)
            pair_exponents = a[:, None] + a
            overlap = (pi / pair_exponents)**1.5 / (2 * pair_exponents)**angular
            weights /= np.sqrt(weights @ overlap @ weights)
            components = [((0, 0, 0), f"{shell}s")]
            if angular:
                components = [((1, 0, 0), "2px"), ((0, 1, 0), "2py"), ((0, 0, 1), "2pz")]
            for powers, name in components:
                orbitals.append(Orbital(tuple(center), powers, tuple(zip(a, weights)),
                                        f"{symbol}{index}:{name}"))
    return orbitals


def evaluate_basis(orbitals, points):
    values = []
    for orbital in orbitals:
        displacement = points - orbital.center
        radius_squared = np.einsum("gi,gi->g", displacement, displacement)
        radial = sum(weight * np.exp(-exponent * radius_squared)
                     for exponent, weight in orbital.primitives)
        values.append(radial * np.prod(displacement**orbital.powers, axis=1))
    return np.column_stack(values)
