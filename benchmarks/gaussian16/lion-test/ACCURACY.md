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
