# YoungCC parallel-agent working rules

## Objective
Improve Gaussian-compatible computational chemistry, full TS/IRC automation,
single-job distributed compute, reliable error recovery, and research usability.
Gaussian official documentation (https://gaussian.com/) is the primary syntax
reference. Numerical results still require independent reference validation.

## Ownership / merge discipline
One agent = one feature branch + one worktree + **one source module folder**.
Read other folders if needed, but do not edit them in the same agent PR.
See docs/multi-agent-workspaces.md for paths, allowed changes and task prompts.
Root opendft/*.py files are compatibility shims: only the integration owner edits them.
Shared README, pyproject.toml, .github/workflows, docs/design.md and cross-module
API changes are coordinator-owned. Do not run agents concurrently on shared files.

## Public contracts
- Molecule and parse_input: opendft.inputs.molecule
- Gaussian .gjf/.com parsing: opendft.inputs.gaussian
- Legacy electronic-structure computations: opendft.engines.native.scf
- Advanced B3LYP: opendft.engines.pyscf_backend.uks
- Saved results and logs: opendft.reporting.output.calculate
- Entry points: opendft.interfaces.cli and opendft.interfaces.gui

Keep existing import paths functional via the root shims. Propose contract changes
as a separate integration PR; don't silently modify adjacent modules.

## Reference and data discipline
Do not edit, rewrite, or duplicate committed Gaussian 16 .inp/.out reference
records under benchmarks/gaussian16/lion-test. Treat them as immutable facts.
Do not commit generated checkpoints, logs, artificial Gaussian outputs,
random geometry datasets or placeholder fixtures. Real calculations and
reproducible test specifications are preferred. Generated files belong in build/
(already gitignored). Only add minimal synthetic unit cases for a clearly stated
edge condition that cannot be tested using real references.

## Acceptance
Run python -m pytest -q and relevant Gaussian reference checks.
Distinguish test completion from Gaussian numerical equivalence. Always report
what was executed and what remains unverified.
