# Advanced B3LYP integration

## Scope

YoungCC legacy RHF/LDA/STO-3G remains native. New UB3LYP and B3LYP
calculations are executed through **PySCF UKS** (unrestricted Kohn-Sham)
with def2-SVP/def2-TZVPP, C-PCM and D3(BJ) as optional components.
This is an integration, not an independent YoungCC B3LYP engine.

## Primary Gaussian 16 syntax reference

- https://gaussian.com/
- https://gaussian.com/scrf/
- https://gaussian.com/empiricaldispersion/
- https://gaussian.com/basissets/

This first-stage parser accepts the documented shape of a Gaussian route,
but **does not claim full Gaussian compatibility**. It rejects unsupported
Opt/Freq/IRC/TS/Geom=Check/Guess=Read rather than silently treating jobs as SP.

## Install and run

```sh
python -m pip install -e ".[advanced,test]"
youngcc examples/oh_ub3lyp.gjf -o build/oh_ub3lyp
python -m pytest -q
```

C-PCM is self-consistent. D3(BJ) uses the PySCF dispersion extension.
Cavity, integration grid, SCF thresholds and D3 damping conventions must
be matched before comparing absolute energies against Gaussian 16.
The DFT backend is limited to atoms H through Kr in this milestone.
The Windows standalone exe still needs a separate advanced-backend
packaging check.

## Test coverage

- Input validation: spin parity, solvent, dispersion, basis and unsupported routes
- Numerical integration (GitHub Actions ubuntu-latest/Python 3.11):
  unrestricted OH doublet, both def2 bases, C-PCM effect,
  D3(BJ) energy effect, and machine-readable output
- **Gaussian16 numerical cross-validation is not yet completed**.
  No assertion of Gaussian-equivalent energies is made.
