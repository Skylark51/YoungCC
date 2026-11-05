"""Analytic molecular integrals via Hermite Gaussian expansions."""

from functools import cache
from itertools import product
from math import pi, sqrt

import numpy as np
from scipy.special import hyp1f1


def _hermite(i, j, a, b, displacement):
    p = a + b
    q = a * b / p

    @cache
    def coefficient(i, j, t):
        if t < 0 or t > i + j:
            return 0.0
        if i == j == 0:
            return np.exp(-q * displacement**2)
        if j:
            return (coefficient(i, j - 1, t - 1) / (2 * p)
                    + q * displacement / b * coefficient(i, j - 1, t)
                    + (t + 1) * coefficient(i, j - 1, t + 1))
        return (coefficient(i - 1, j, t - 1) / (2 * p)
                - q * displacement / a * coefficient(i - 1, j, t)
                + (t + 1) * coefficient(i - 1, j, t + 1))

    return [coefficient(i, j, t) for t in range(i + j + 1)]


def _coulomb(exponent, displacement, order):
    argument = exponent * sum(value**2 for value in displacement)
    n = np.arange(order + 1)
    base = (-2 * exponent)**n * hyp1f1(n + 0.5, n + 1.5, -argument) / (2 * n + 1)

    @cache
    def auxiliary(t, u, v, n=0):
        if t:
            return (displacement[0] * auxiliary(t - 1, u, v, n + 1)
                    + ((t - 1) * auxiliary(t - 2, u, v, n + 1) if t > 1 else 0))
        if u:
            return (displacement[1] * auxiliary(t, u - 1, v, n + 1)
                    + ((u - 1) * auxiliary(t, u - 2, v, n + 1) if u > 1 else 0))
        if v:
            return (displacement[2] * auxiliary(t, u, v - 1, n + 1)
                    + ((v - 1) * auxiliary(t, u, v - 2, n + 1) if v > 1 else 0))
        return base[n]

    return auxiliary


def _primitive_pairs(left, right):
    displacement = np.subtract(left.center, right.center)
    for (a, wa), (b, wb) in product(left.primitives, right.primitives):
        p = a + b
        center = (a * np.asarray(left.center) + b * np.asarray(right.center)) / p
        axes = [_hermite(i, j, a, b, dx)
                for i, j, dx in zip(left.powers, right.powers, displacement)]
        expansion = [(t, u, v, x * y * z * wa * wb)
                     for (t, x), (u, y), (v, z) in product(*(enumerate(axis) for axis in axes))
                     if x * y * z != 0]
        zero = [axis[0] for axis in axes]
        scale = wa * wb * (pi / p)**1.5
        overlap = scale * np.prod(zero)
        kinetic = b * (2 * sum(right.powers) + 3) * overlap
        for axis, (i, j, dx) in enumerate(zip(left.powers, right.powers, displacement)):
            factor = scale * zero[(axis + 1) % 3] * zero[(axis + 2) % 3]
            kinetic -= 2 * b**2 * factor * _hermite(i, j + 2, a, b, dx)[0]
            if j >= 2:
                kinetic -= 0.5 * j * (j - 1) * factor * _hermite(i, j - 2, a, b, dx)[0]
        yield p, center, expansion, sum(left.powers) + sum(right.powers), overlap, kinetic


def _attraction(pairs, molecule):
    value = 0.0
    for p, center, expansion, order, _, _ in pairs:
        for charge, nucleus in zip(molecule.charges, molecule.centers):
            aux = _coulomb(p, center - nucleus, order)
            value -= (2 * pi / p) * charge * sum(c * aux(t, u, v) for t, u, v, c in expansion)
    return value


def _repulsion(left, right):
    value = 0.0
    for (p, a, first, n, _, _), (q, b, second, m, _, _) in product(left, right):
        aux = _coulomb(p * q / (p + q), a - b, n + m)
        term = sum(c * d * (-1)**(x + y + z) * aux(t + x, u + y, v + z)
                   for t, u, v, c in first for x, y, z, d in second)
        value += 2 * pi**2.5 / (p * q * sqrt(p + q)) * term
    return value


def build_integrals(orbitals, molecule):
    size = len(orbitals)
    overlap, kinetic, attraction = np.zeros((3, size, size))
    repulsion = np.zeros((size,) * 4)
    pairs = []
    for i in range(size):
        for j in range(i + 1):
            primitives = list(_primitive_pairs(orbitals[i], orbitals[j]))
            overlap[i, j] = overlap[j, i] = sum(pair[4] for pair in primitives)
            kinetic[i, j] = kinetic[j, i] = sum(pair[5] for pair in primitives)
            attraction[i, j] = attraction[j, i] = _attraction(primitives, molecule)
            pairs.append((i, j, primitives))
    for index, (i, j, left) in enumerate(pairs):
        for k, l, right in pairs[:index + 1]:
            value = _repulsion(left, right)
            for (a, b), (c, d) in product(((i, j), (j, i)), ((k, l), (l, k))):
                repulsion[a, b, c, d] = repulsion[c, d, a, b] = value
    return overlap, kinetic, attraction, repulsion


def nuclear_repulsion(molecule):
    charges = molecule.charges
    return sum(charges[i] * charges[j] / np.linalg.norm(center - molecule.centers[j])
               for i, center in enumerate(molecule.centers) for j in range(i))
