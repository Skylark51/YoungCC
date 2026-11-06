# Parallel relaxed bond scans

On Lion, from a shared working directory:

```bash
yc MalonicAcid-scan.inp --scan-hosts lion38 lion39 lion40 lion48 lion49 lion50 lion51
```

The same options work with `youngcc` or `python -m opendft.interfaces.cli`.
Use a fresh singlet input containing one bond scan without `Freq` or `Guess=Read`.
The verified implementation uses 16 CPUs and 4000 MiB per worker by default.
All hosts must see the same input, output, Python environment and source paths,
and support noninteractive SSH and Linux CPU affinity.

Initial xTB geometry/Hessian models are generated automatically on the last host.
Install xTB on that host and put it on PATH, set `YOUNGCC_XTB`, or pass
`--scan-xtb /absolute/path/to/xtb`. The standard Lion installation is also detected.
These models only initialize the scan; every point is refined with the input's
unrestricted DFT model, solvent and final convergence criteria.

Options:

- `--scan-directory PATH`: a new, nonexisting results directory; default is
  `INPUT.youngcc.scan`. Existing results are never reused or overwritten silently.
- `--scan-threads N`: CPUs per worker.
- `--scan-memory-mb MIB`: memory per worker, bounded by `%Mem`.
- `--scan-model-directory PATH`: explicitly reuse supplied `point-N.npz` initial
  models instead of generating xTB seeds.
- Host specifications such as `lion38=0,1,2,3` restrict the worker's CPU affinity.

The top-level `summary.json` includes total input-to-result time, preparation
time, per-point energies, convergence residuals, actual hosts and CPU assignments.
`points/` contains checkpoints, progress and worker error logs; `seeds/` contains
the xTB calculations. `yc` writes its console log to `INPUT.out` as before.
Failures return nonzero and retain diagnostic files. After correcting a failure,
choose a new output directory or explicitly use the lower-level resume command:

```bash
python -m opendft.interfaces.distributed_scan INPUT.inp NEW_RESULTS \
  --hosts lion38 lion39 --threads 16 --optimizer projected-restricted-armijo \
  --model-directory PREVIOUS/seeds/models --resume-from PREVIOUS/points \
  --team-memory-mb 4000 --save-checkpoints --save-progress
```

The measured seven-node speedup was 5.01–5.05x including fresh model preparation,
and 5.29–5.32x for the scan alone, with all seven nodes available. Competing jobs
reduced the earlier best result to 4.31x. This command never suspends other jobs;
allocate sufficient CPU capacity before expecting the uncontended benchmark rate.
Test completion and agreement between single/multiple nodes do not establish
exact Gaussian numerical equivalence.
