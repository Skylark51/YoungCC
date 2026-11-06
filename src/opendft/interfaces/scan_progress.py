"""Save accepted exact scan evaluations and restore them without a new SCF."""

from hashlib import sha256

import numpy as np

from .bond_opt import retract_bond
from .geometry_workflow import _checkpoint_metadata, _write_checkpoint


def progress_metadata(path, molecule, index, target, pcm_cavity="native"):
    return dict(**_checkpoint_metadata(molecule, pcm_cavity), scan_progress=1, point_index=index,
                input_sha256=sha256(path.read_bytes()).hexdigest(), target_bohr=target)


def write_progress(path, surface, metadata, positions, trial):
    coords, energy, gradient, density = surface.cache
    if gradient is None or not np.array_equal(coords, positions):
        raise RuntimeError("An accepted scan position has no exact cached gradient.")
    _write_checkpoint(path, **metadata, centers=positions.reshape(-1, 3), trial=trial,
                      density=density, energy=energy, gradient=gradient)


def restore_progress(path, surface, metadata, indices):
    with np.load(path, allow_pickle=False) as data:
        if data["youngcc_checkpoint"].item() != 1:
            raise ValueError("Unsupported scan progress checkpoint.")
        if data.get("pcm_cavity", "native") != metadata.get("pcm_cavity", "native"):
            raise ValueError("Scan progress C-PCM cavity differs from the requested profile.")
        for key, value in metadata.items():
            matches = (np.isclose(data[key].item(), value, atol=1e-12, rtol=0) if key == "target_bohr"
                       else np.array_equal(data[key], value))
            if not matches:
                raise ValueError(f"Scan progress {key} differs from the requested point.")
        centers = data["centers"]
        trial, energy, gradient, density = (data[key] for key in ("trial", "energy", "gradient", "density"))
    molecule = surface.job.molecule
    n = molecule.centers.size
    nao = surface.template.nao
    if (centers.shape != molecule.centers.shape or trial.shape != (n,)
            or gradient.shape != (n,) or density.shape != (2, nao, nao) or energy.ndim != 0
            or not all(np.isfinite(value).all() for value in (centers, trial, energy, gradient, density))):
        raise ValueError("Invalid geometry or electronic state in scan progress.")
    actual, _, _ = retract_bond(trial, indices, metadata["target_bohr"])
    if not np.allclose(actual, centers.ravel(), atol=1e-12, rtol=0):
        raise ValueError("Scan progress geometry differs from the retracted trial.")
    surface.density = density
    surface.cache = centers.ravel(), energy.item(), gradient, density
    surface.cache_grid = molecule.grid_level
    surface.scf = None
    return trial
