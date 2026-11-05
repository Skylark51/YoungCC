"""Read Gaussian workflow keywords at the CLI boundary."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ..inputs.gaussian import parse_gaussian
from ..inputs.molecule import ANGSTROM_PER_BOHR


@dataclass(frozen=True)
class Job:
    molecule: object
    opt: frozenset
    freq: bool
    scan: tuple | None
    checkpoint: Path | None
    seed: dict | None
    guess_read: bool
    memory_mb: float | None
    conventional_cycles: int | None


def read_job(text, protected_paths):
    """Return a workflow job or let the original single-point parser handle it."""
    lines = text.splitlines()
    route_index = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if route_index is None:
        return None
    sections = re.split(r"\n[ \t]*\n", "\n".join(lines[route_index:]).strip(), maxsplit=3)
    route = " ".join(sections[0].lower().splitlines())
    token_pattern = r"(?:[^\s()]+|\([^()]*\))+"
    tokens = re.findall(token_pattern, route)
    workflow = any(re.fullmatch(r"(?:opt|freq)(?:=.*)?", token) for token in tokens)
    checkpointed = any(re.match(r"\s*%(?:old)?chk\s*=", line, re.I) for line in lines[:route_index])
    if not (workflow or checkpointed):
        return None
    if re.sub(token_pattern, "", route).strip():
        raise ValueError("Malformed Gaussian route options.")
    if len(sections) < 3:
        raise ValueError("Expected Gaussian route, title and molecule sections.")

    link = []
    paths = {}
    memory_mb = None
    for line in lines[:route_index]:
        match = re.fullmatch(r"\s*%(chk|oldchk)\s*=\s*(\S(?:.*\S)?)\s*", line, re.I)
        if match:
            name, value = match.groups()
            name = name.lower()
            if name in paths:
                raise ValueError(f"Repeated %{name} directive.")
            paths[name] = Path(value).resolve()
            continue
        match = re.fullmatch(r"\s*%mem\s*=\s*([1-9]\d*)\s*(mb|gb)\s*", line, re.I)
        if match:
            memory_mb = int(match[1]) * (1024 if match[2].lower() == "gb" else 1)
            continue
        match = re.fullmatch(r"\s*%cpu\s*=\s*(\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)*)\s*", line, re.I)
        if match:
            cores = set()
            for part in match[1].split(","):
                first, _, last = part.partition("-")
                low, high = int(first), int(last or first)
                if not 0 <= high - low < 256:
                    raise ValueError("Invalid %CPU range.")
                cores.update(range(low, high + 1))
            link.append(f"%NProcShared={len(cores)}")
            continue
        link.append(line)

    checkpoint = paths.get("chk")
    source = paths.get("oldchk", checkpoint)
    if any(path in protected_paths for path in paths.values()):
        raise ValueError("Checkpoint path overlaps an input or output file.")
    opt = frozenset()
    freq = geom_check = guess_read = False
    conventional_cycles = None
    remaining = []
    for token in tokens:
        if token == "opt" or token.startswith("opt="):
            value = token.partition("=")[2]
            options = value.removeprefix("(").removesuffix(")").split(",") if value else ["minimum"]
            opt = frozenset(option.strip() for option in options)
            if opt - {"minimum", "modredundant", "nofreeze", "ts", "rcfc", "noeigentest"}:
                raise ValueError(f"Unsupported optimization options: {token}")
        elif token == "freq":
            freq = True
        elif token == "geom=check":
            geom_check = True
        elif token == "guess=read":
            guess_read = True
        elif token == "nosymm":
            continue  # The PySCF workflow always runs with symmetry disabled.
        elif match := re.fullmatch(r"scf=\(xqc(?:,\s*maxconventionalcycles=([1-9]\d*))?\)|scf=xqc", token):
            conventional_cycles = int(match[1] or 80)
        else:
            remaining.append(token)

    if "modredundant" in opt and opt & {"ts", "nofreeze", "rcfc"}:
        raise ValueError("This dihedral scan requires Opt=ModRedundant without TS, RCFC or NoFreeze.")
    if opt & {"rcfc", "noeigentest"} and "ts" not in opt:
        raise ValueError("RCFC and NoEigenTest are supported with Opt=TS.")
    seed = None
    if geom_check or guess_read or "rcfc" in opt:
        if source is None:
            raise ValueError("Geom=Check, Guess=Read and RCFC require %Chk or %OldChk.")
        try:
            with np.load(source, allow_pickle=False) as data:
                if data["youngcc_checkpoint"].item() != 1:
                    raise ValueError("Unsupported checkpoint version.")
                seed = dict(data)
                centers, symbols = seed["centers"], seed["symbols"]
                if (symbols.ndim != 1 or centers.shape != (symbols.size, 3)
                        or not np.isfinite(centers).all()):
                    raise ValueError("Invalid checkpoint geometry.")
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Cannot read YoungCC checkpoint {source}: {exc}") from exc
    elif "oldchk" in paths:
        raise ValueError("%OldChk requires Geom=Check, Guess=Read or RCFC in YoungCC.")

    geometry = sections[2]
    if geom_check:
        if len(geometry.splitlines()) != 1:
            raise ValueError("Geom=Check expects charge and multiplicity followed by a blank line.")
        geometry += "\n" + "\n".join(
            f"{symbol} " + " ".join(map(str, center * ANGSTROM_PER_BOHR))
            for symbol, center in zip(seed["symbols"], seed["centers"], strict=True)
        )
    molecule = parse_gaussian("\n".join([*link, " ".join(remaining), "", sections[1], "", geometry]))
    extra = sections[3].strip() if len(sections) == 4 else ""
    scan = None
    if "modredundant" in opt:
        match = re.fullmatch(
            r"(?:D\s+)?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+S\s+(\d+)\s+([+-]?[\d.eE+-]+)",
            extra, re.I,
        )
        if not match:
            raise ValueError("Expected one dihedral scan line: [D] i j k l S steps increment.")
        indices = tuple(int(index) - 1 for index in match.groups()[:4])
        steps, increment = int(match[5]), float(match[6])
        if (len(set(indices)) != 4 or min(indices) < 0 or max(indices) >= len(molecule.symbols)
                or steps < 1 or not np.isfinite(increment) or not 0 < abs(increment) < 180):
            raise ValueError("Invalid dihedral scan indices, steps or increment.")
        scan = indices, steps, increment
    elif extra:
        raise ValueError("Unexpected section after Gaussian coordinates.")
    return Job(molecule, opt, freq, scan, checkpoint, seed, guess_read, memory_mb, conventional_cycles)
