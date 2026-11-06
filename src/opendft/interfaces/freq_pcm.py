"""Reuse fixed-geometry C-PCM integrals and the surface matrix factorization."""

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, lu_factor, lu_solve
from scipy.special import erf

from pyscf import df, gto, lib
from pyscf.solvent.pcm import PCM


class FrequencyCPCM(PCM):
    """C-PCM with caches scoped to one PySCF cavity and molecular geometry."""

    def reset(self, mol=None):
        self._factor = None
        self._integrals = None
        return super().reset(mol)

    def build(self, ng=None):
        super().build(ng)
        matrix = self._intermediates["K"]
        try:
            self._factor = cho_solve, cho_factor(matrix, check_finite=False)
        except LinAlgError:
            self._factor = lu_solve, lu_factor(matrix, check_finite=False)
        self._integrals = None
        grid = self.surface["grid_coords"]
        nao = self.mol.nao
        self._integral_point_mb = nao * (nao + 1) // 2 * 8 / 1e6
        available_mb = max(0, self.max_memory - lib.current_memory()[0])
        budget = min(12 * 1024, available_mb / 4)
        count = min(len(grid), int(budget / self._integral_point_mb))
        if count:
            self._integrals = self._build_integrals(0, count)

    def _build_integrals(self, start, stop, cintopt=None):
        coords = self.surface["grid_coords"][start:stop]
        exponents = self.surface["charge_exp"][start:stop]
        aux = gto.fakemol_for_charges(coords, expnt=exponents ** 2)
        aux.cart = self.mol.cart
        return df.incore.aux_e2(
            self.mol, aux, intor="int3c2e", aosym="s2ij", cintopt=cintopt,
        )

    def _integral_blocks(self):
        cached_points = self._integrals.shape[-1]
        yield slice(0, cached_points), self._integrals
        points = len(self.surface["grid_coords"])
        if cached_points < points:
            mol = self.mol
            cintopt = gto.moleintor.make_cintopt(
                mol._atm, mol._bas, mol._env, mol._add_suffix("int3c2e"),
            )
            available = self.max_memory - lib.current_memory()[0]
            block_size = max(1, int(min(1024, 0.9 * available) / self._integral_point_mb))
            for start, stop in lib.prange(cached_points, points, block_size):
                yield slice(start, stop), self._build_integrals(start, stop, cintopt)

    def _get_v(self, dms):
        if self._integrals is None:
            return super()._get_v(dms)
        packed = lib.pack_tril(dms + dms.swapaxes(-1, -2))
        nao = self.mol.nao
        packed[:, np.cumsum(np.arange(1, nao + 1)) - 1] *= 0.5
        potential = np.empty(
            (len(dms), len(self.surface["grid_coords"])), dtype=np.result_type(packed, self._integrals),
        )
        for points, integrals in self._integral_blocks():
            potential[:, points] = packed @ integrals
        return potential

    def _get_vmat(self, charges):
        if self._integrals is None:
            return super()._get_vmat(charges)
        charges = charges.reshape(-1, len(self.surface["grid_coords"]))
        packed = np.zeros(
            (len(charges), self._integrals.shape[0]), dtype=np.result_type(charges, self._integrals),
        )
        for points, integrals in self._integral_blocks():
            packed -= charges[:, points] @ integrals.T
        return lib.unpack_tril(packed)

    def _solve_surface(self, rhs):
        solve, factor = self._factor
        return solve(factor, rhs.T, check_finite=False).T

    def _charge(self, potential):
        # C-PCM has symmetric K and scalar R, so q and q_sym coincide.
        return -self._intermediates["f_epsilon"] * self._solve_surface(potential)

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()
        nao = dms.shape[-1]
        dms = dms.reshape(-1, nao, nao)
        if len(dms) == 2:
            dms = dms.sum(axis=0, keepdims=True)
        potential = self.v_grids_n - self._get_v(dms)
        charge = self._charge(potential)
        self._intermediates.update(q=charge[0], q_sym=charge[0], v_grids=potential[0], dm=dms)
        return 0.5 * np.dot(charge[0], potential[0]), self._get_vmat(charge)[0]

    def grad(self, dm):
        from pyscf.solvent.grad.pcm import get_dF_dA, grad_nuc

        from .pcm_hessian_qv import electronic_potential_derivatives

        if not self._intermediates:
            self.build()
        cached = self._intermediates.get("dm")
        if cached is None or np.linalg.norm(cached - dm) >= 1e-10:
            self._get_vind(dm)
        charge = self._intermediates["q"]
        factor = self._intermediates["f_epsilon"]
        if factor == 0:
            return np.zeros((self.mol.natm, 3))

        # For C-PCM, R is a scalar times identity and K is symmetric.
        # The left solution is therefore -q/f, and the off-diagonal
        # derivative is antisymmetric in the two surface-point indices.
        left = -charge / factor
        coords = self.surface["grid_coords"]
        exponents = self.surface["charge_exp"]
        per_point = np.empty_like(coords)
        for start in range(0, len(coords), 128):
            stop = min(start + 128, len(coords))
            delta = coords[start:stop, None, :] - coords
            radius = np.linalg.norm(delta, axis=2)
            exponent = exponents[start:stop, None] * exponents / np.hypot(
                exponents[start:stop, None], exponents
            )
            scaled = exponent * radius
            radius[np.arange(stop - start), np.arange(start, stop)] = 1.0
            radial = (2 * scaled / np.sqrt(np.pi) * np.exp(-scaled ** 2) - erf(scaled)) / radius ** 3
            per_point[start:stop] = left[start:stop, None] * np.einsum(
                "ij,ijx,j->ix", radial, delta, charge
            )
        interaction = -np.array([
            per_point[start:stop].sum(axis=0) for start, stop in self.surface["gslice_by_atom"]
        ])
        d_switch, _ = get_dF_dA(self.surface, self.surface_discretization_method)
        diagonal = -exponents * np.sqrt(2 / np.pi) / self.surface["switch_fun"] ** 2
        interaction -= 0.5 * np.einsum("i,inx->nx", left * charge * diagonal, d_switch)
        # Bound derivative integral buffers separately from the persistent cavity cache.
        memory = self.max_memory
        self.max_memory = min(memory, lib.current_memory()[0] + 2048)
        try:
            electronic = electronic_potential_derivatives(self, dm) @ charge
            return electronic + grad_nuc(self, dm) + interaction
        finally:
            self.max_memory = memory

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        nao = dms.shape[-1]
        potential = -self._get_v(dms.reshape(-1, nao, nao))
        return self._get_vmat(self._charge(potential)).reshape(dms.shape)

    def hess(self, dm):
        from .pcm_hessian_qv import solvent_hessian

        return solvent_hessian(self, dm)
