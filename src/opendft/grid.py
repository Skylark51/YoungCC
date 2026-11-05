"""Atom-centered radial and angular quadrature with Becke partitioning."""

import numpy as np
from scipy.special import roots_legendre


def molecular_grid(centers, radial_order=80, polar_order=24, azimuth_order=48):
    nodes, radial_weights = roots_legendre(radial_order)
    t = (nodes + 1) / 2
    radius = t / (1 - t)
    radial_weights *= radius**2 / (2 * (1 - t)**2)
    cos_theta, polar_weights = roots_legendre(polar_order)
    phi = 2 * np.pi * np.arange(azimuth_order) / azimuth_order
    sin_theta = np.sqrt(1 - cos_theta**2)
    directions = np.stack(np.broadcast_arrays(
        sin_theta[:, None] * np.cos(phi),
        sin_theta[:, None] * np.sin(phi),
        cos_theta[:, None],
    ), axis=-1).reshape(-1, 3)
    angular_weights = np.repeat(polar_weights, azimuth_order) * (2 * np.pi / azimuth_order)
    offsets = (radius[:, None, None] * directions).reshape(-1, 3)
    weights = np.outer(radial_weights, angular_weights).ravel()

    all_points, all_weights = [], []
    for owner, center in enumerate(centers):
        points = center + offsets
        distances = np.linalg.norm(points[:, None, :] - centers, axis=2)
        partition = np.ones_like(distances)
        for i, left in enumerate(centers):
            for j in range(i):
                mu = (distances[:, i] - distances[:, j]) / np.linalg.norm(left - centers[j])
                switch = np.clip(mu, -1, 1)
                for _ in range(3):
                    switch *= 1.5 - 0.5 * switch**2
                partition[:, i] *= (1 - switch) / 2
                partition[:, j] *= (1 + switch) / 2
        all_points.append(points)
        all_weights.append(weights * partition[:, owner] / partition.sum(axis=1))
    return np.concatenate(all_points), np.concatenate(all_weights)
