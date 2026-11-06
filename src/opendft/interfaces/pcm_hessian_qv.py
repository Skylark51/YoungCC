"""Contract the C-PCM charge–potential Hessian without repeating integrals."""

import numpy as np
from pyscf import df, gto, lib
from pyscf.solvent.grad.pcm import get_dD_dS, get_dF_dA
from pyscf.solvent.hessian.pcm import (
    get_d2D_d2S, get_d2F_d2A, get_d2Sii, get_dS_dot_q, get_dqsym_dx, get_v_dot_d2S_dot_q,
)


def _prepare_cavity(cavity, density):
    if not cavity._intermediates:
        cavity.build()
    cached = cavity._intermediates.get("dm")
    if cached is None or np.linalg.norm(cached - density) >= 1e-10:
        cavity._get_vind(density)


def electronic_potential_derivatives(cavity, density):
    """Derivative of the electronic potential at each moving cavity point."""
    molecule = cavity.mol
    nao = molecule.nao
    coords = cavity.surface["grid_coords"]
    exponents = cavity.surface["charge_exp"]
    ao_slices = molecule.aoslice_by_atom()[:, 2:]
    derivative = np.empty((molecule.natm, 3, len(coords)))
    symmetric_density = density + density.T
    operator = molecule._add_suffix("int3c2e_ip1")
    cintopt = gto.moleintor.make_cintopt(molecule._atm, molecule._bas, molecule._env, operator)
    memory = max(0, cavity.max_memory - lib.current_memory()[0])
    width = max(1, min(128, int(0.9 * memory * 1e6 / (8 * nao**2 * 3))))
    for first, last in lib.prange(0, len(coords), width):
        auxiliary = gto.fakemol_for_charges(coords[first:last], expnt=exponents[first:last] ** 2)
        auxiliary.cart = molecule.cart
        integrals = df.incore.aux_e2(molecule, auxiliary, intor=operator, aosym="s1", cintopt=cintopt)
        contracted = np.einsum("xijq,ij->xiq", integrals, symmetric_density)
        del integrals
        for atom, (start, stop) in enumerate(ao_slices):
            derivative[atom, :, first:last] = contracted[:, start:stop].sum(axis=1)
    grid_derivative = -derivative.sum(axis=0)
    for atom, (start, stop) in enumerate(cavity.surface["gslice_by_atom"]):
        derivative[atom, :, start:stop] += grid_derivative[:, start:stop]
    return derivative


def _charge_potential_terms(cavity, density):
    molecule = cavity.mol
    nao = molecule.nao
    atoms = molecule.natm
    coordinates = cavity.surface["grid_coords"]
    exponents = cavity.surface["charge_exp"]
    grid_slices = cavity.surface["gslice_by_atom"]
    ao_slices = molecule.aoslice_by_atom()[:, 2:]
    charges = cavity._intermediates["q_sym"]
    point_count = len(charges)
    symmetric_density = density + density.T
    memory = max(0, cavity.max_memory - lib.current_memory()[0])
    width = max(1, min(64, int(0.9 * memory * 1e6 / (8 * nao**2 * 9))))
    integral_options = {}

    def integrals(operator, first, last):
        operator = molecule._add_suffix(operator)
        if operator not in integral_options:
            integral_options[operator] = gto.moleintor.make_cintopt(
                molecule._atm, molecule._bas, molecule._env, operator,
            )
        cintopt = integral_options[operator]
        for start, stop in lib.prange(first, last, width):
            auxiliary = gto.fakemol_for_charges(coordinates[start:stop], expnt=exponents[start:stop] ** 2)
            auxiliary.cart = molecule.cart
            block = df.incore.aux_e2(molecule, auxiliary, intor=operator, aosym="s1", cintopt=cintopt)
            yield slice(start, stop), block.reshape(9, nao, nao, stop - start)

    def charge_weighted(operator, first, last):
        weighted = np.zeros((9, nao, nao))
        for points, block in integrals(operator, first, last):
            weighted += np.einsum("dijq,q->dij", block, charges[points])
        return weighted

    result = np.zeros((atoms, atoms, 3, 3))
    diagonal_ao = np.zeros((9, nao, nao))
    mixed_ao = np.zeros_like(diagonal_ao)
    for column, (first, last) in enumerate(grid_slices):
        diagonal = charge_weighted("int3c2e_ipip1", first, last)
        mixed = charge_weighted("int3c2e_ipvip1", first, last)
        diagonal_ao += diagonal
        mixed_ao += mixed

        # Translating both AO centers and the grid center leaves the integral invariant.
        mixed_grid = -(diagonal + mixed).reshape(3, 3, nao, nao)
        for row, (start, stop) in enumerate(ao_slices):
            block = np.einsum("xyij,ij->xy", mixed_grid[:, :, start:stop], symmetric_density[start:stop])
            result[row, column] += block
            result[column, row] += block.T

        grid_diagonal = np.einsum("dij,ij->d", diagonal, symmetric_density).reshape(3, 3)
        grid_mixed = np.einsum("dij,ij->d", mixed, density).reshape(3, 3)
        result[column, column] += grid_diagonal + grid_mixed + grid_mixed.T

    for atom, (start, stop) in enumerate(ao_slices):
        diagonal = np.einsum("dij,ij->d", diagonal_ao[:, start:stop], symmetric_density[start:stop])
        result[atom, atom] += diagonal.reshape(3, 3)

    for row, (start, stop) in enumerate(ao_slices):
        for column, (first, last) in enumerate(ao_slices):
            mixed = np.einsum(
                "dij,ij->d", mixed_ao[:, start:stop, first:last], symmetric_density[start:stop, first:last],
            )
            result[row, column] += mixed.reshape(3, 3)

    gradient = electronic_potential_derivatives(cavity, density)
    return -result, gradient.reshape(atoms * 3, point_count)


def charge_potential_hessian(cavity, density):
    _prepare_cavity(cavity, density)
    fixed, gradient = _charge_potential_terms(cavity, density)
    atoms = cavity.mol.natm
    charge_response = get_dqsym_dx(cavity, density, range(atoms)).reshape(gradient.shape)
    response = charge_response @ gradient.T
    return fixed - response.reshape(atoms, 3, atoms, 3).transpose(0, 2, 1, 3)


def _nuclear_terms(cavity):
    molecule = cavity.mol
    atoms = molecule.natm
    charge = cavity._intermediates["q_sym"]
    atomic_charges = molecule.atom_charges()
    slices = cavity.surface["gslice_by_atom"]
    nuclei = gto.fakemol_for_charges(molecule.atom_coords())
    points = gto.fakemol_for_charges(
        cavity.surface["grid_coords"], expnt=cavity.surface["charge_exp"] ** 2,
    )

    def integral(operator, first, second):
        return gto.mole.intor_cross(molecule._add_suffix(operator), first, second)

    gradient = integral("int2c2e_ip1", nuclei, points).transpose(1, 0, 2) * atomic_charges[:, None, None]
    grid_gradient = integral("int2c2e_ip1", points, nuclei) @ atomic_charges
    mixed = integral("int2c2e_ip1ip2", nuclei, points).reshape(3, 3, atoms, len(charge))
    diagonal_ao = -(mixed @ charge) * atomic_charges
    diagonal_grid = (integral("int2c2e_ipip1", points, nuclei) @ atomic_charges).reshape(3, 3, -1)
    result = np.zeros((atoms, atoms, 3, 3))
    for atom, (first, last) in enumerate(slices):
        gradient[atom, :, first:last] += grid_gradient[:, first:last]
        block = ((mixed[:, :, :, first:last] @ charge[first:last]) * atomic_charges).transpose(2, 0, 1)
        result[:, atom] += block
        result[atom, :] += block
        result[atom, atom] += diagonal_ao[:, :, atom] + diagonal_grid[:, :, first:last] @ charge[first:last]
    return result, gradient.reshape(atoms * 3, len(charge))


def solvent_hessian(cavity, density):
    """Compute all C-PCM Hessian terms using one potential and charge response."""
    _prepare_cavity(cavity, density)
    atoms = cavity.mol.natm
    factor = cavity._intermediates["f_epsilon"]
    if factor == 0:
        return np.zeros((atoms, atoms, 3, 3))

    nuclear, nuclear_gradient = _nuclear_terms(cavity)
    electronic, electronic_gradient = _charge_potential_terms(cavity, density)
    potential_gradient = nuclear_gradient - electronic_gradient
    charge = cavity._intermediates["q_sym"]
    # K is symmetric and R is -factor times the identity for C-PCM.
    left = -charge / factor
    surface = cavity.surface
    grid_slices = surface["gslice_by_atom"]
    method = cavity.surface_discretization_method
    switch_first, _ = get_dF_dA(surface, surface_discretization_method=method)
    _, first, first_diagonal = get_dD_dS(surface, switch_first, with_D=False, with_S=True)
    surface_gradient = get_dS_dot_q(first, first_diagonal, charge, range(atoms), grid_slices)
    del first, first_diagonal
    surface_gradient = surface_gradient.reshape(atoms * 3, len(charge))
    surface_response = cavity._solve_surface(surface_gradient)
    charge_response = surface_response + cavity._charge(potential_gradient)

    # Combine the nuclear, electronic and surface-response contractions.
    response = charge_response @ potential_gradient.T
    response += potential_gradient @ surface_response.T
    response -= surface_gradient @ surface_response.T / factor

    _, second = get_d2D_d2S(surface, with_D=False, with_S=True)
    switch_second, _ = get_d2F_d2A(surface, surface_discretization_method=method)
    second_diagonal = get_d2Sii(surface, switch_first, switch_second)
    curvature = get_v_dot_d2S_dot_q(second, second_diagonal, left, charge, atoms, grid_slices)
    return nuclear + electronic - 0.5 * curvature + response.reshape(atoms, 3, atoms, 3).transpose(0, 2, 1, 3)
