"""Check the old public API remains identical after module boundary refactor."""
from opendft import advanced, basis, cli, dft, gaussian_input, grid, input, integrals, output, scf
from opendft.chemistry import basis as chem_basis, dft as chem_dft, grid as chem_grid
from opendft.chemistry import integrals as chem_integrals
from opendft.engines.native import scf as native
from opendft.engines.pyscf_backend import uks
from opendft.inputs import gaussian, molecule
from opendft.interfaces import cli as cli_impl
from opendft.reporting import output as reporting

def test_backwards_compatible_public_function_identity():
    assert input.parse_input is molecule.parse_input
    assert input.read_input is molecule.read_input
    assert gaussian_input.parse_gaussian is gaussian.parse_gaussian
    assert basis.build_basis is chem_basis.build_basis
    assert dft.lda_xc is chem_dft.lda_xc
    assert grid.molecular_grid is chem_grid.molecular_grid
    assert integrals.build_integrals is chem_integrals.build_integrals
    assert scf.run_scf is native.run_scf
    assert advanced.run_advanced is uks.run_advanced
    assert output.calculate is reporting.calculate
    assert cli.main is cli_impl.main
