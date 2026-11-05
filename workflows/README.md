# Gaussian reaction-path workflow

From the repository root, run python -m pip install -e . and activate a
licensed Gaussian 16 environment on a compute host. The executable or
launcher must read Gaussian input from stdin and write the log to stdout.
On Lion, run in an allocated compute environment with the Gaussian profile.

    python -m workflows.reaction reaction.json build/reaction --g16 /path/to/g16

Example reaction.json

    {
      "start": "start.xyz",
      "reactant": "reactant.xyz",
      "product": "product.xyz",
      "method": "UB3LYP/def2SVP",
      "options": ["scrf=(cpcm,solvent=acetonitrile)", "int=ultrafine"],
      "charge": 0,
      "multiplicity": 1,
      "cores": 4,
      "memory": "8GB",
      "scan": {"atoms": [6, 11], "steps": 10, "step": -0.1},
      "imaginary_cm": 50,
      "rmsd_angstrom": 0.15,
      "pair_angstrom": 0.35
    }

XYZ paths are relative to the JSON file or absolute. Use angstrom coordinates
and identical atom order, including hydrogens. Supply independently
identified reactant and product minima. Forward and Reverse can map to either
reference. The scan uses one relaxed bond, angle or dihedral with one-based
atom indices and increments in angstroms or degrees. Choose a range that
brackets an interior maximum starting from the optimized input. Supported
methods are HF, B3LYP, PBE1PBE, M06-2X and wB97XD with built-in basis sets.
Additional route tokens are limited to SCRF, Int and EmpiricalDispersion.

The runner checks initial Opt/Freq and the highest bracketed scan peak.
TS optimization starts from the peak Hessian. The converged TS must have
exactly one negative frequency, with magnitude at least imaginary_cm.
Both LQA IRC jobs start from the verified TS Hessian and must report
direction completion and detection of a PES minimum. Each last accepted
IRC geometry goes through Opt/Freq and must have zero negative frequencies.
The atom-mapped geometries must exclusively match distinct references under
both proper-rotation RMSD and maximum pair-distance error. validation.json
records the distance metrics, frequencies and verified assignment.

Run the same command to resume validated stages with matching inputs and
parent checkpoints. Each attempt keeps its input, output and checkpoint
separate. failure.json records diagnosed Gaussian or validation failures.
Retries increase the SCF or optimization budget, reduce the step size or
recompute the Hessian. An imaginary mode at a minimum triggers one tighter
optimization and frequency check. IRC retries restart from the verified TS
with a smaller step and larger point budget. Scratch and memory failures
stop for diagnosis. Use a fresh output directory for changed settings or
reference structures.

## Lion comparison

Lion's ~/skylark/aes.py wf resolves a working directory for AES sync.
The Gaussian calculation controller is ~/script/wf with wf-core.
Both parsers selected the same archived MalonicAcid scan peak
(step 3, -345.555116933 Eh) and TS imaginary mode (-1015.0054 cm^-1).
This runner waits for normal scan termination. The archive records Lion's
intentional early stop after finding a bracket. Cluster scheduling, AES sync
and the methyl-specific heuristic remain features of Lion's controller.
This runner checks endpoint identity against supplied reference geometries.

Run python -m pytest -q workflows to check the archived scan, frequencies,
IRC directions and endpoints, convergence and atom-mapped distances.
A fresh Gaussian calculation is needed to validate the generated decks
and end-to-end execution on the target installation.

Gaussian keyword references

- [Opt](https://gaussian.com/opt/)
- [Freq](https://gaussian.com/freq/)
- [IRC](https://gaussian.com/irc/)
