"""Strict Gaussian-16-style single-point input subset.

Syntax reference: https://gaussian.com/ (route, SCRF, empirical dispersion).
Unsupported directives raise errors instead of silently being ignored.
"""
import re
from .input import parse_input


def parse_gaussian(text):
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("%")):
        i += 1
    if i == len(lines) or not lines[i].lstrip().startswith("#"):
        raise ValueError("Expected Gaussian route section starting with #.")
    route_lines = []
    while i < len(lines) and lines[i].strip():
        route_lines.append(lines[i].strip())
        i += 1
    route = " ".join(route_lines).lower()
    route = re.sub(r"^#\s*(?:p|n|t)?\s*", "", route)
    method_basis = re.search(r"\b(u?b3lyp)\s*/\s*(def2-?svp|def2-?tzvpp)\b", route)
    if not method_basis:
        raise ValueError("Only B3LYP/def2SVP and UB3LYP/def2TZVPP single-point routes are supported.")
    data = {"method": method_basis.group(1).upper(), "basis": method_basis.group(2).upper(),
            "units": "angstrom"}
    route = route[:method_basis.start()] + " " + route[method_basis.end():]
    pcm = re.search(r"\bscrf\s*=\s*\(([^)]*)\)", route)
    if pcm:
        args = [x.strip() for x in pcm.group(1).split(",")]
        if not args or args[0] != "cpcm":
            raise ValueError("Only SCRF=(CPCM,Solvent=...) is supported.")
        solvents = [x.partition("=")[2] for x in args[1:] if x.startswith("solvent=")]
        if len(solvents) != 1 or len(args) != 2:
            raise ValueError("Specify exactly one C-PCM solvent.")
        data["solvent"] = solvents[0]
        route = route.replace(pcm.group(0), " ")
    disp = re.search(r"\bempiricaldispersion\s*=\s*(\w+)", route)
    if disp:
        if disp.group(1) != "gd3bj":
            raise ValueError("Only EmpiricalDispersion=GD3BJ is supported.")
        data["dispersion"] = "GD3BJ"
        route = route.replace(disp.group(0), " ")
    grid = re.search(r"\bint\s*=\s*(\w+)", route)
    if grid:
        if grid.group(1) not in {"ultrafine", "superfine"}:
            raise ValueError("Only Int=UltraFine and Int=SuperFine are supported.")
        data["grid_level"] = {"ultrafine": 5, "superfine": 7}[grid.group(1)]
        route = route.replace(grid.group(0), " ")
    if route.strip():
        raise ValueError(f"Unsupported Gaussian route options: {route.strip()}")
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip():
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines):
        raise ValueError("Missing charge and multiplicity; Geom=Check is not supported yet.")
    charge_spin = lines[i].split()
    if len(charge_spin) != 2:
        raise ValueError("Expected integer charge and multiplicity.")
    try:
        data["charge"], data["multiplicity"] = map(int, charge_spin)
    except ValueError as exc:
        raise ValueError("Expected integer charge and multiplicity.") from exc
    atoms = []
    for line in lines[i + 1:]:
        if not line.strip():
            break
        atom = line.split()
        if len(atom) != 4:
            raise ValueError("Only Cartesian atom x y z coordinates are supported.")
        atoms.append(atom)
    data["atoms"] = atoms
    return parse_input(data)
