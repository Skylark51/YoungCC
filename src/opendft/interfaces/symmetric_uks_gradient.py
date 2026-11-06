"""Use the restricted derivative representation of a verified closed-shell UKS state."""

import numpy as np
from pyscf.grad import rhf, uhf


def gradient(fallback, mf):
    if not mf.converged or mf.mol.spin:
        return fallback(mf)
    density = np.asarray(mf.make_rdm1())
    if density.ndim != 3 or density.shape[0] != 2:
        return fallback(mf)
    if not np.allclose(density[0], density[1], rtol=0, atol=1e-10):
        return fallback(mf)

    rks = mf.to_rks(xc=mf.xc)
    if type(getattr(rks, "with_solvent", None)) is not type(getattr(mf, "with_solvent", None)):
        return fallback(mf)
    energy_weighted = uhf.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ).sum(axis=0)
    restricted_weighted = rhf.make_rdm1e(rks.mo_energy, rks.mo_coeff, rks.mo_occ)
    if (not np.allclose(rks.make_rdm1(), density.sum(axis=0), rtol=0, atol=1e-10)
            or not np.allclose(restricted_weighted, energy_weighted, rtol=0, atol=1e-10)):
        return fallback(mf)
    derivative = rks.nuc_grad_method()
    derivative.grid_response = True
    result = derivative.kernel().ravel()
    return result if np.isfinite(result).all() else fallback(mf)
