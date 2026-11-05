# YoungCC: independent multi-agent development workspaces

## Rule: separate code ownership, shared contracts
A Git folder alone does not prevent semantic conflicts. Independent agents
must use separate **Git worktrees and feature branches**, edit one module
folder each, and open separate PRs; the integration owner merges them in order.
Root opendft compatibility shims, pyproject.toml, README, .github/workflows
and existing root tests are integrator-owned.

## Current active workspaces / suggested agent instructions

| Scope (single agent-owned folder) | Example task to assign | Module tests |
|---|---|---|
| src/opendft/chemistry/ | "Improve AO integrals, def2 basis fidelity or XC grids, prove against archived Gaussian reference values; do not edit SCF." | tests/chemistry/ |
| src/opendft/inputs/ | "Extend exact Gaussian route syntax for %CPU, ModRedundant, Opt and checkpoint contracts following gaussian.com; reject unsupported routes." | tests/inputs/ |
| src/opendft/engines/native/ | "Improve native SCF convergence and mathematical validation without touching external PySCF or basis modules." | tests/engines/native/ |
| src/opendft/engines/pyscf_backend/ | "Improve unrestricted B3LYP, match C-PCM radii/grid with Gaussian and verify D3(BJ), using real reference logs." | tests/engines/pyscf_backend/ |
| src/opendft/reporting/ | "Extend raw Gaussian-style logs, concise final summary, errors and structured provenance; do not change engine energies." | tests/reporting/ |
| src/opendft/interfaces/ | "Improve GUI input editor, task progress, molecular structure display and CLI discoverability; no SCF changes." | tests/interfaces/ |
| benchmarks/gaussian16/ | "Use immutable Lion Gaussian logs to compute absolute/relative energy and convergence differences; never edit .inp/.out originals." | tests/benchmarks/ |

The first six folders contain real, moved implementation code.
The benchmarks folder already contains real Gaussian inputs/outputs and
comparison scripts. No empty stub directories were created.

## Windows PowerShell workflow per agent

From the existing local repository:

~~~powershell
git switch main
git pull --ff-only origin main
git worktree add -b agent/chemistry ../YoungCC-chemistry main
git worktree add -b agent/inputs ../YoungCC-inputs main
git worktree add -b agent/pyscf ../YoungCC-pyscf main
~~~

Assign each agent its matching folder and dedicated worktree.
Different agents should not share working trees. Open separate pull
requests and require an integration agent to review cross-module impacts.

Run tests from each worktree root:

~~~bash
python -m pip install -e ".[advanced,test]"
python -m pytest -q
python benchmarks/gaussian16/lion_reference_accuracy.py
~~~

Full Gaussian energy comparisons are more expensive and are run in CI
or in a controlled Linux environment. Do not automatically multiply
identical expensive benchmarks across all agent worktrees.

## Reference data and compatibility

- Gaussian keyword source of truth: https://gaussian.com/
- Real Lion archive: benchmarks/gaussian16/lion-test/ (read-only)
- Reproducible SP comparisons: benchmarks/gaussian16/lion_reference_accuracy.py
- Keep opendft.* legacy import paths, CLI module and GUI entry point working.
- New cross-folder interfaces require explicit coordinator approval first.

## Future workspaces (create only when real implementation begins)

- Workflow automation: Opt -> Freq -> Scan -> TS -> IRC -> endpoint validation.
- Distributed single-job execution: MPI/SCF integral parallelism across nodes.
- Job persistence, checkpoint/restart and recovery orchestration.

Do not precreate empty packages, fake Gaussian outputs or dummy job records
for these future workstreams.
