# Gaussian 16 Lion accuracy validation

## Source records

This archive contains 31 Gaussian .inp and 17 .out files (13 normal and
4 abnormal terminations). Some files require omitted checkpoint files.
The records named MalonicAcid contain **C5H8O2**, not malonic acid
(C3H4O4), according to their coordinates and Gaussian formula.

## Fixed-geometry single-point comparison

The Python script at ../lion_reference_accuracy.py extracts the last
Gaussian orientation before a final SCF energy, validates the reference,
and launches a fresh YoungCC/PySCF UB3LYP single-point at identical
Cartesian coordinates. It does not claim to reproduce Opt/Freq/TS/IRC.

Commands (from repository root):

    python -m pip install -e ".[advanced,test]"
    python benchmarks/gaussian16/lion_reference_accuracy.py
    python benchmarks/gaussian16/lion_reference_accuracy.py --run --output build/accuracy.json

| Reference case | Gaussian UB3LYP C-PCM energy (Eh) |
|---|---:|
| TS at def2-SVP | -345.554981665 |
| TS at def2-TZVPP | -345.954367965 |
| IRC forward minimum at def2-SVP | -345.557581541 |
| IRC forward minimum at def2-TZVPP | -345.957709189 |

These use C-PCM(MeCN) and **no D3(BJ)**. Gaussian output specifies UFF
radii for its cavity. PySCF C-PCM defaults to different radii, cavity
surface discretization and grid; unadjusted total-energy differences
must not be called isolated basis/DFT errors.

There are n-butane D3(BJ) inputs but no matching n-butane Gaussian
outputs here. A separate Gaussian D3(BJ) reference remains necessary.

## Recommended isolation strategy

1. Match fixed Cartesian coordinates, electron/spin count, B3LYP
   functional, basis set, integral grid and SCF convergence.
2. Obtain matched **gas-phase Gaussian single points** to isolate
   XC/basis deviations independently of PCM.
3. Match Gaussian's UFF cavity radii and surface discretization as
   closely as supported, then repeat C-PCM comparisons.
4. For D3(BJ), compare dispersion contributions and total energies
   from corresponding Gaussian and YoungCC single points.
5. Compare same-method reaction **relative** electronic energies before
   interpreting barrier differences; distinguish SCF energies from free
   energies including thermal corrections.

Gaussian primary input documentation: https://gaussian.com/


## Observed YoungCC/PySCF results (GitHub Actions, 2026-11-05)

CI run: https://github.com/Skylark51/YoungCC/actions/runs/41988473217

| Case | Gaussian (Eh) | YoungCC (Eh) | YoungCC - Gaussian (mEh) |
|---|---:|---:|---:|
| TS def2-SVP | -345.554981665 | -345.557132609 | -2.150944 |
| Min def2-SVP | -345.557581541 | -345.559788712 | -2.207171 |
| TS def2-TZVPP | -345.954367965 | -345.956904682 | -2.536717 |
| Min def2-TZVPP | -345.957709189 | -345.960280327 | -2.571138 |

| Basis | Gaussian TS - Min (kcal/mol) | YoungCC TS - Min (kcal/mol) | Difference |
|---|---:|---:|---:|
| def2-SVP | 1.6314 | 1.6667 | +0.0353 |
| def2-TZVPP | 2.0966 | 2.1182 | +0.0216 |

All four standalone UKS single-point jobs converged. These are observed
numerical comparisons, **not matched-cavity or matched-grid accuracy claims**.
The reference outputs use UFF PCM cavity radii while the PySCF backend used
its default cavity; no Opt/TS/IRC or D3(BJ) reference equivalence was tested.
