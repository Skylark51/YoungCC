"""Build a restricted DFT geometry predictor for unrestricted scan refinement."""

from types import SimpleNamespace

import numpy as np

from .bond_opt import _projected_max, minimize_bond_armijo
from .geometry_workflow import _Surface


class _RestrictedSurface(_Surface):
    def __init__(self, job, max_cycles, indices):
        super().__init__(job, max_cycles)
        self.indices = indices
        self.grid_response = False
        rks = self.dft.RKS

        def fitted(mol):
            return rks(mol).density_fit(auxbasis="def2-universal-jkfit")

        self.dft = SimpleNamespace(UKS=fitted)

    def _gradient(self, mf):
        if self.grid_response:
            return super()._gradient(mf)

        derivative = mf.nuc_grad_method()
        derivative.grid_response = False
        gradient = derivative.kernel().ravel()
        if not np.isfinite(gradient).all():
            raise RuntimeError("Non-finite restricted predictor gradient.")
        centers = mf.mol.atom_coords()
        delta = centers[self.indices[0]] - centers[self.indices[1]]
        projected_force = _projected_max(gradient, self.indices, delta / np.linalg.norm(delta))
        self.grid_response = projected_force < 1.5e-3
        return gradient

    def evaluate(self, coordinates, *, gradient=True):
        if self.cache is None or not np.array_equal(self.cache[0], coordinates):
            mf = self.new_scf(coordinates, min(2, self.job.molecule.grid_level))
            mf.conv_tol = 1e-8
            energy = mf.kernel(dm0=self.density)
            if not mf.converged or not np.isfinite(energy):
                raise RuntimeError("Restricted scan predictor did not converge.")
            self.density = np.asarray(mf.make_rdm1())
            self.cache = coordinates.copy(), float(energy), None, self.density
            self.scf = mf
        if gradient and self.cache[2] is None:
            self.cache = self.cache[:2] + (self._gradient(self.scf), self.density)
        return self.cache[1:]


def restricted_scan_seed(surface, xyz, indices, target, hessian):
    predictor = _RestrictedSurface(surface.job, surface.max_cycles, indices)
    try:
        seed = minimize_bond_armijo(predictor, xyz.ravel(), indices, target, initial_hessian=hessian)
    except RuntimeError:
        return xyz
    surface.density = np.stack((predictor.density * 0.5,) * 2)
    return seed
