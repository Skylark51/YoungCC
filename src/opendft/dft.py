"""Spin-unpolarized Slater exchange and VWN5 correlation."""

import numpy as np


def lda_xc(density):
    """Return XC energy per electron and the derivative of rho * epsilon_xc."""
    rho = np.maximum(density, 1e-30)
    exchange = -0.75 * np.cbrt(3 * rho / np.pi)
    x = (3 / (4 * np.pi * rho))**(1 / 6)
    a, b, c, x0 = 0.0310907, 3.72744, 12.9352, -0.10498
    quadratic = x*x + b*x + c
    q = np.sqrt(4*c - b*b)
    angle = np.arctan2(q, 2*x + b)
    factor = b*x0 / (x0*x0 + b*x0 + c)
    correlation = a * (
        np.log(x*x / quadratic) + 2*b/q * angle
        - factor * (np.log((x - x0)**2 / quadratic) + 2*(b + 2*x0)/q * angle)
    )
    # d(angle)/dx = -q/(2*quadratic), and rho*dx/drho = -x/6.
    derivative = a * (2/x - 2*(x + b)/quadratic
                      - factor * (2/(x - x0) - 2*(x + b + x0)/quadratic))
    return exchange + correlation, 4/3 * exchange + correlation - x/6 * derivative


def density_at(values, density_matrix):
    return np.einsum("gi,gi->g", values @ density_matrix, values)


def xc_matrix(values, weights, density_matrix):
    density = np.maximum(density_at(values, density_matrix), 0)
    energy, potential = lda_xc(density)
    return float(weights @ (density * energy)), values.T @ ((weights * potential)[:, None] * values)
