"""An opt-in C-PCM profile using verified Gaussian UFF sphere radii."""

from ..inputs.molecule import ANGSTROM_PER_BOHR, ATOMIC_NUMBERS


PCM_CAVITIES = ("native", "gaussian-uff")
# Scaled radii in Angstrom, verified in Gaussian's PCM-SphereRadii records.
_SCALED_UFF = {"H": 1.58730, "C": 2.11805, "N": 2.01300, "O": 1.92500, "Co": 1.57960}


def validate_cavity(profile, molecule):
    if profile not in PCM_CAVITIES:
        raise ValueError(f"Unknown C-PCM cavity profile {profile}.")
    if profile == "gaussian-uff":
        if (molecule.solvent or "").lower() != "acetonitrile":
            raise ValueError("The Gaussian UFF cavity profile requires acetonitrile.")
        unknown = set(molecule.symbols) - _SCALED_UFF.keys()
        if unknown:
            raise ValueError(f"Gaussian UFF radii are unverified for {', '.join(sorted(unknown))}.")


def configure_cavity(cavity, molecule):
    from pyscf.solvent.pcm import modified_Bondi

    validate_cavity("gaussian-uff", molecule)
    radii = modified_Bondi.copy()
    for symbol, radius in _SCALED_UFF.items():
        radii[ATOMIC_NUMBERS[symbol]] = radius / ANGSTROM_PER_BOHR
    cavity.radii_table = radii
    cavity.eps = 35.688
    cavity.lebedev_order = 21
