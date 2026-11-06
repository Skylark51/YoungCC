"""PySCF geometry workflows for Gaussian-style YoungCC command-line jobs."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


def _write_checkpoint(path, **data):
    # Publish only complete calculations, including when reading and writing the same path.
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with staging.open("wb") as handle:
            np.savez_compressed(handle, youngcc_checkpoint=1, **data)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _dihedral(centers, indices):
    a, b, c, d = centers[list(indices)]
    axis = c - b
    axis /= np.linalg.norm(axis)
    left = a - b
    right = d - c
    left -= (left @ axis) * axis
    right -= (right @ axis) * axis
    if min(np.linalg.norm(left), np.linalg.norm(right)) < 1e-9:
        raise ValueError("Dihedral scan encountered collinear atoms.")
    return np.arctan2(np.cross(axis, left) @ right, left @ right)


def _vibrations(centers, masses):
    centered = centers - np.average(centers, axis=0, weights=masses)
    translations = np.broadcast_to(np.eye(3)[:, None, :], (3, len(masses), 3))
    rotations = np.cross(np.eye(3)[:, None, :], centered)
    rigid = np.concatenate((translations, rotations)) * np.sqrt(masses)[None, :, None]
    return null_space(rigid.reshape(6, -1), rcond=1e-8)


class _Surface:
    """Evaluate energy and derivatives with the same model at every geometry."""

    def __init__(self, job, max_cycles):
        try:
            from pyscf import dft, gto, lib
            if job.molecule.dispersion:
                import pyscf.dispersion  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Install workflow dependencies with pip install -e ".[advanced]"') from exc
        self.job = job
        self.dft = dft
        self.max_cycles = max_cycles
        molecule = job.molecule
        if molecule.threads is not None:
            lib.num_threads(molecule.threads)
        self.template = gto.M(
            atom=list(zip(molecule.symbols, molecule.centers, strict=True)),
            unit="Bohr", basis=molecule.basis.lower(), charge=molecule.charge,
            spin=molecule.multiplicity - 1, verbose=0,
            **({"max_memory": job.memory_mb} if job.memory_mb is not None else {}),
        )
        self.cache = None
        self.cache_grid = molecule.grid_level
        self.scf = None
        self.opt_grid = None
        self.distributed_xc = None
        self.distributed_gradient = None
        self.analytic_freq = False
        self.density = job.seed.get("density") if job.guess_read else None
        if job.guess_read and self.density is None:
            raise ValueError("Guess=Read requires a saved density matrix.")

    def evaluate(self, coordinates, *, gradient=True, reuse=True, grid_level=None):
        coordinates = np.asarray(coordinates)
        molecule = self.job.molecule
        grid_level = molecule.grid_level if grid_level is None else grid_level
        if (reuse and self.cache is not None and grid_level == self.cache_grid
                and np.array_equal(coordinates, self.cache[0])):
            if not gradient or self.cache[2] is not None:
                return self.cache[1:]
            if self.scf is not None:
                _, energy, _, density = self.cache
                nuclear_gradient = self._gradient(self.scf)
                self.cache = self.cache[0], energy, nuclear_gradient, density
                if not self.analytic_freq:
                    self.scf = None
                return self.cache[1:]
        if reuse:
            self.scf = None
        mf = self.new_scf(coordinates, grid_level)
        mol = mf.mol
        density_shape = (2, mol.nao, mol.nao)
        if self.density is not None and (
            self.density.shape != density_shape or not np.isfinite(self.density).all()
        ):
            raise ValueError("Invalid checkpoint density for the requested basis.")
        energy = mf.kernel(dm0=self.density)
        if not mf.converged and self.job.conventional_cycles:
            newton = mf.newton()
            newton.max_cycle = self.max_cycles
            energy = newton.kernel(mf.mo_coeff, mf.mo_occ)
            mf.mo_coeff, mf.mo_occ, mf.mo_energy = newton.mo_coeff, newton.mo_occ, newton.mo_energy
            mf.converged, mf.e_tot = newton.converged, newton.e_tot
        if not mf.converged or not np.isfinite(energy):
            raise RuntimeError("SCF did not converge during the geometry workflow.")
        nuclear_gradient = self._gradient(mf) if gradient else None
        density = np.asarray(mf.make_rdm1())
        if reuse:
            self.cache = coordinates.copy(), float(energy), nuclear_gradient, density
            self.cache_grid = grid_level
            self.density = density
            if self.analytic_freq or not gradient:
                self.scf = mf
        return float(energy), nuclear_gradient, density

    def new_scf(self, coordinates, grid_level):
        molecule = self.job.molecule
        mol = self.template.set_geom_(coordinates.reshape(-1, 3), unit="Bohr", inplace=False)
        mf = self.dft.UKS(mol)
        mf.xc = "b3lyp"
        if self.opt_grid is not None:
            self.opt_grid.attach_scf(mf)
        if self.distributed_xc is not None:
            mf._numint = self.distributed_xc
        mf.chkfile = None
        mf.grids.level = grid_level
        mf.conv_tol = 1e-11
        mf.max_cycle = self.job.conventional_cycles or self.max_cycles
        if molecule.dispersion:
            mf.disp = "d3bj"
        if molecule.solvent:
            from .freq_pcm import FrequencyCPCM

            cavity = FrequencyCPCM(mol, molecule.solvent)
            if self.job.pcm_cavity == "gaussian-uff":
                from .pcm_profile import configure_cavity

                configure_cavity(cavity, molecule)
            mf = mf.PCM(cavity)
        return mf

    def _gradient(self, mf):
        derivative = mf.nuc_grad_method()
        derivative.grid_response = True
        if self.opt_grid is not None:
            self.opt_grid.attach_gradient(derivative)
        if self.distributed_gradient is None:
            gradient = derivative.kernel().ravel()
        else:
            gradient = self.distributed_gradient.gradient(derivative).ravel()
        if not np.isfinite(gradient).all():
            raise RuntimeError("Non-finite nuclear gradient.")
        return gradient

    def objective(self, coordinates):
        energy, gradient, _ = self.evaluate(coordinates)
        return energy, gradient

    def hessian(self, coordinates, report=None):
        # Central differences of total analytic gradients include PCM and D3(BJ).
        self.evaluate(coordinates)
        step = 0.001
        hessian = np.empty((coordinates.size, coordinates.size))
        displaced = coordinates.copy()
        for i in range(coordinates.size):
            displaced[i] += step
            plus = self.evaluate(displaced, reuse=False)[1]
            displaced[i] -= 2 * step
            minus = self.evaluate(displaced, reuse=False)[1]
            displaced[i] = coordinates[i]
            hessian[:, i] = (plus - minus) / (2 * step)
            if report is not None and (i + 1) % 3 == 0:
                report(f"Cartesian Hessian  atom {(i + 1) // 3}/{coordinates.size // 3}")
        return (hessian + hessian.T) / 2


def _minimum(surface, coordinates, scan=None):
    if scan is None:
        result = minimize(surface.objective, coordinates, jac=True, method="BFGS",
                          options={"gtol": 1e-4, "maxiter": 200})
    else:
        indices, target = scan

        def residual(x):
            angle = _dihedral(x.reshape(-1, 3), indices) - target
            return np.arctan2(np.sin(angle), np.cos(angle))

        result = minimize(surface.objective, coordinates, jac=True, method="SLSQP",
                          constraints=[{"type": "eq", "fun": residual}],
                          options={"ftol": 1e-10, "maxiter": 200})
        if abs(residual(result.x)) > 1e-6:
            raise RuntimeError("Dihedral scan failed to reach its target angle.")
    if not result.success:
        raise RuntimeError(f"Geometry optimization failed: {result.message}")
    return result.x


def _transition_state(surface, coordinates, masses, hessian, check_index):
    # Follow one mode uphill and minimize all orthogonal modes in the vibrational space.
    weight = np.repeat(np.sqrt(masses), 3)
    mode = None
    for _ in range(100):
        basis = _vibrations(coordinates.reshape(-1, 3), masses)
        values, vectors = np.linalg.eigh(basis.T @ (hessian / np.outer(weight, weight)) @ basis)
        directions = basis @ vectors
        if not len(values):
            raise ValueError("TS optimization requires a vibrational mode.")
        index = 0 if mode is None else np.argmax(np.abs(directions.T @ mode))
        mode = directions[:, index]
        gradient = surface.objective(coordinates)[1]
        components = directions.T @ (gradient / weight)
        if np.max(np.abs(gradient)) < 1e-4:
            if check_index:
                exact = surface.hessian(coordinates)
                eigenvalues = np.linalg.eigvalsh(basis.T @ (exact / np.outer(weight, weight)) @ basis)
                if np.count_nonzero(eigenvalues < 0) != 1:
                    raise RuntimeError("Optimized geometry does not have exactly one imaginary mode.")
            return coordinates
        scaled = -components / np.maximum(np.abs(values), 1e-3)
        scaled[index] *= -1
        step = (directions @ scaled) / weight
        largest = np.max(np.abs(step))
        if largest < 1e-12:
            raise RuntimeError("TS optimization stalled before the gradient converged.")
        step *= min(1.0, 0.12 / largest)
        candidate = coordinates + step
        change = surface.objective(candidate)[1] - gradient - hessian @ step
        # Symmetric Powell secant update accommodates an indefinite Hessian.
        step_squared = step @ step
        hessian = (hessian + (np.outer(change, step) + np.outer(step, change)) / step_squared
                   - (change @ step) * np.outer(step, step) / step_squared**2)
        coordinates = candidate
    raise RuntimeError("TS optimization exceeded 100 steps.")


def run_job(job, prefix, *, max_cycles):
    if max_cycles < 1:
        raise ValueError("max-cycles must be positive.")
    if job.scan and len(job.scan[0]) == 2:
        raise ValueError("Bond scans require --scan-hosts; supply one host for a single-node run.")
    molecule = job.molecule
    coordinates = molecule.centers.ravel().copy()
    rcfc = "rcfc" in job.opt
    metadata = {
        "symbols": list(molecule.symbols), "charge": molecule.charge, "multiplicity": molecule.multiplicity,
        "method": molecule.method, "basis": molecule.basis, "solvent": molecule.solvent or "",
        "dispersion": molecule.dispersion or "", "grid_level": molecule.grid_level,
    }
    if job.seed is not None:
        keys = ("symbols", "charge", "multiplicity")
        if job.guess_read or rcfc:
            keys = tuple(metadata)
        for key in keys:
            if not np.array_equal(job.seed.get(key), metadata[key]):
                raise ValueError(f"Checkpoint {key} differs from the input.")
    initial_hessian = None
    if rcfc:
        initial_hessian = job.seed.get("hessian")
        if (initial_hessian is None or initial_hessian.shape != (coordinates.size, coordinates.size)
                or not np.isfinite(initial_hessian).all()
                or not np.allclose(job.seed["centers"], molecule.centers, atol=1e-9, rtol=0)):
            raise ValueError("RCFC requires a saved Cartesian Hessian at the starting geometry.")

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    surface = _Surface(job, max_cycles)
    masses = surface.template.atom_mass_list()
    with Path(f"{prefix}.log").open("w", encoding="utf-8") as log:
        def emit(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        emit(f"YoungCC PySCF  {molecule.method}/{molecule.basis}")
        scan_points = []
        if job.scan:
            indices, steps, increment = job.scan
            start = _dihedral(coordinates.reshape(-1, 3), indices)
            for i in range(steps + 1):
                target = start + np.deg2rad(i * increment)
                coordinates = _minimum(surface, coordinates, (indices, target))
                energy = surface.objective(coordinates)[0]
                emit(f"Scan point {i + 1}/{steps + 1}  {np.rad2deg(target):.6f} degrees  {energy:.12f} Eh")
                scan_points.append({"target_degrees": float(np.rad2deg(target)), "total_energy": energy,
                                    "centers_bohr": coordinates.reshape(-1, 3).tolist()})
        elif "ts" in job.opt:
            if initial_hessian is None:
                initial_hessian = surface.hessian(coordinates)
            coordinates = _transition_state(surface, coordinates, masses, initial_hessian,
                                            "noeigentest" not in job.opt)
            emit("TS optimization converged.")
        elif job.opt:
            coordinates = _minimum(surface, coordinates)
            emit("Geometry optimization converged.")
        energy, _, density = surface.evaluate(coordinates, gradient=bool(job.opt or job.freq))
        centers = coordinates.reshape(-1, 3)
        result = {**metadata, "engine": "PySCF", "converged": True, "total_energy": energy,
                  "centers_bohr": centers.tolist()}
        checkpoint = {**metadata, "centers": centers, "density": density}
        if scan_points:
            result["scan_points"] = scan_points
        if job.freq:
            from pyscf.hessian.thermo import harmonic_analysis

            hessian = surface.hessian(coordinates, report=emit)
            molecular = surface.template.set_geom_(centers, unit="Bohr", inplace=False)
            analysis = harmonic_analysis(
                molecular, hessian.reshape(len(masses), 3, len(masses), 3).transpose(0, 2, 1, 3),
                imaginary_freq=False, mass=masses,
            )
            frequencies = analysis["freq_wavenumber"].tolist()
            result["frequencies_cm-1"] = frequencies
            result["normal_modes"] = analysis["norm_mode"].tolist()
            result["reduced_masses_amu"] = analysis["reduced_mass"].tolist()
            checkpoint["hessian"] = hessian
            for i in range(0, len(frequencies), 3):
                emit(" Frequencies -- " + " ".join(f"{value:10.4f}" for value in frequencies[i:i + 3]))
        emit(f"Total energy  {energy:.12f} Eh")
        if job.checkpoint is not None:
            _write_checkpoint(job.checkpoint, **checkpoint)
            emit(f"YoungCC checkpoint written to {job.checkpoint}")
        Path(f"{prefix}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        emit("YoungCC calculation completed.")


def _checkpoint_metadata(molecule, pcm_cavity="native"):
    metadata = {
        "symbols": list(molecule.symbols), "charge": molecule.charge, "multiplicity": molecule.multiplicity,
        "method": molecule.method, "basis": molecule.basis, "solvent": molecule.solvent or "",
        "dispersion": molecule.dispersion or "", "grid_level": molecule.grid_level,
    }
    if pcm_cavity != "native":
        metadata["pcm_cavity"] = pcm_cavity
    return metadata
