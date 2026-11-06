"""Constrained bond optimization using a differentiable Cartesian retraction."""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


def retract_bond(trial, indices, target):
    first, second = indices
    xyz = trial.reshape(-1, 3).copy()
    delta = xyz[first] - xyz[second]
    length = np.linalg.norm(delta)
    if length < 1e-12:
        raise ValueError("Bond optimizer encountered coincident atoms.")
    axis = delta / length
    midpoint = (xyz[first] + xyz[second]) / 2
    xyz[first] = midpoint + (target / 2) * axis
    xyz[second] = midpoint - (target / 2) * axis
    return xyz.ravel(), axis, length


def _model_hessian(initial_hessian, dimension):
    model = np.asarray(initial_hessian, dtype=float)
    if model.shape != (dimension, dimension) or not np.isfinite(model).all():
        raise ValueError("Invalid initial Cartesian Hessian for bond optimization.")
    return (model + model.T) / 2


def _projected_max(gradient, indices, axis):
    first, second = indices
    grad = gradient.reshape(-1, 3).copy()
    normal_component = ((grad[first] - grad[second]) @ axis) / 2
    grad[first] -= normal_component * axis
    grad[second] += normal_component * axis
    return np.max(np.abs(grad))


def _chart_gradient(gradient, indices, axis, scale):
    first, second = indices
    grad = gradient.reshape(-1, 3).copy()
    midpoint = (grad[first] + grad[second]) / 2
    difference = grad[first] - grad[second]
    tangential = difference - axis * (difference @ axis)
    half = (scale / 2) * tangential
    grad[first] = midpoint + half
    grad[second] = midpoint - half
    return grad.ravel()


def minimize_bond(surface, coordinates, indices, target, *, initial_hessian=None, accepted=None):
    first, second = indices
    if target <= 0:
        raise ValueError("Bond length must be positive.")

    transform = None
    if initial_hessian is not None:
        model = _model_hessian(initial_hessian, coordinates.size)
        _, axis, length = retract_bond(coordinates, indices, target)
        tangent = (np.eye(3) - np.outer(axis, axis)) * (target / length)
        first_rows = slice(3 * first, 3 * first + 3)
        second_rows = slice(3 * second, 3 * second + 3)
        jacobian = np.eye(coordinates.size)
        jacobian[first_rows, first_rows] = jacobian[second_rows, second_rows] = (np.eye(3) + tangent) / 2
        jacobian[first_rows, second_rows] = jacobian[second_rows, first_rows] = (np.eye(3) - tangent) / 2
        curvature = jacobian.T @ model @ jacobian
        values, vectors = np.linalg.eigh((curvature + curvature.T) / 2)
        # BFGS with an identity inverse Hessian in these coordinates starts
        # with the regularized model inverse Hessian in the Cartesian trial.
        values = np.maximum(values, 0.02)
        transform = vectors / np.sqrt(values)
        gradient_scale = np.linalg.norm(vectors * np.sqrt(values), ord=np.inf)

    last_trial = None
    last_force = None
    last_positions = None

    def objective(trial):
        nonlocal last_trial, last_force, last_positions
        cartesian = trial if transform is None else coordinates + transform @ trial
        actual, axis, length = retract_bond(cartesian, indices, target)
        energy, gradient = surface.objective(actual)
        if accepted is not None:
            if last_positions is None:
                accepted(actual, cartesian)
            last_positions = actual, cartesian
        if transform is not None:
            last_force = _projected_max(gradient, indices, axis)
        if transform is not None or accepted is not None:
            last_trial = trial.copy()
        gradient = _chart_gradient(gradient, indices, axis, target / length)
        return energy, gradient if transform is None else transform.T @ gradient

    def stop_when_converged(intermediate_result):
        if last_trial is not None and np.array_equal(last_trial, intermediate_result.x):
            if accepted is not None:
                accepted(*last_positions)
            if last_force is not None and last_force <= 8e-5:
                raise StopIteration

    start = coordinates if transform is None else np.zeros_like(coordinates)
    tolerance = 8e-5 if transform is None else 8e-5 / max(1.0, gradient_scale)
    if transform is not None and accepted is not None:
        objective(start)
        if last_force <= 8e-5:
            return last_positions[0]
    result = minimize(objective, start, jac=True, method="BFGS",
                      callback=stop_when_converged if transform is not None or accepted is not None else None,
                      options={"gtol": tolerance, "maxiter": 200})
    trial = result.x if transform is None else coordinates + transform @ result.x
    actual, axis, _ = retract_bond(trial, indices, target)
    _, gradient = surface.objective(actual)
    if _projected_max(gradient, indices, axis) > 1e-4:
        raise RuntimeError(f"Bond optimization did not converge: {result.message}")
    return actual


def minimize_bond_armijo(surface, coordinates, indices, target, *, initial_hessian, accepted=None):
    """Use a vibrational model and compute exact gradients only at accepted trial energies."""
    from .geometry_workflow import _vibrations

    if not np.isfinite(target) or target <= 0:
        raise ValueError("Bond length must be positive and finite.")
    model = _model_hessian(initial_hessian, coordinates.size)
    current, axis, length = retract_bond(coordinates, indices, target)
    if length == target and surface.cache is not None and np.array_equal(surface.cache[0], coordinates):
        # Retraction can round a bond that is already exactly at its target.
        current = coordinates
    normal = np.zeros_like(current).reshape(-1, 3)
    first, second = indices
    normal[first], normal[second] = axis, -axis
    masses = surface.template.atom_mass_list()
    vibrational = _vibrations(current.reshape(-1, 3), masses) / np.repeat(np.sqrt(masses), 3)[:, None]
    directions, _ = np.linalg.qr(vibrational @ null_space(normal.reshape(1, -1) @ vibrational))
    values, vectors = np.linalg.eigh(directions.T @ model @ directions)
    transform = (directions @ vectors) / np.sqrt(np.maximum(values, 0.002))

    position = np.zeros(transform.shape[1])
    identity = np.eye(position.size)
    inverse = identity.copy()
    radius = 0.10
    energy, gradient, _ = surface.evaluate(current)
    if accepted is not None:
        accepted(current, coordinates)
    reduced = transform.T @ _chart_gradient(gradient, indices, axis, target / length)
    for _ in range(60):
        if _projected_max(gradient, indices, axis) <= 8e-5:
            return current
        direction = -inverse @ reduced
        longest = np.linalg.norm((transform @ direction).reshape(-1, 3), axis=1).max()
        direction *= min(1.0, radius / max(longest, 1e-15))
        if reduced @ direction >= 0:
            raise RuntimeError("Bond model did not produce a descent step.")

        cache, density, scf = surface.cache, surface.density, surface.scf
        for backtrack in range(12):
            step = direction * (0.5 ** backtrack)
            trial = coordinates + transform @ (position + step)
            candidate, candidate_axis, candidate_length = retract_bond(trial, indices, target)
            keep = False
            try:
                trial_energy = surface.evaluate(candidate, gradient=False)[0]
                if trial_energy <= energy + 1e-4 * (reduced @ step):
                    energy, gradient, _ = surface.evaluate(candidate)
                    keep = True
                    break
            finally:
                if not keep:
                    surface.cache, surface.density, surface.scf = cache, density, scf
        else:
            raise RuntimeError("Bond optimizer could not find an energy-decreasing step.")

        candidate_reduced = transform.T @ _chart_gradient(gradient, indices, candidate_axis, target / candidate_length)
        change = candidate_reduced - reduced
        curvature = step @ change
        if curvature > 1e-12:
            factor = identity - np.outer(step, change) / curvature
            inverse = factor @ inverse @ factor.T + np.outer(step, step) / curvature
        position += step
        current, axis, reduced = candidate, candidate_axis, candidate_reduced
        if accepted is not None:
            accepted(current, trial)
        radius = max(0.015, radius * 0.5) if backtrack else min(0.18, radius * 1.2)
    if _projected_max(gradient, indices, axis) <= 8e-5:
        return current
    raise RuntimeError("Bond optimization exceeded 60 steps.")
