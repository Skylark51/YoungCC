"""Command-line interface for YoungCC."""

import argparse
import sys
from pathlib import Path

from ..inputs.gaussian import parse_gaussian
from ..inputs.molecule import read_input
from ..reporting.output import calculate
from .gaussian_job import read_job


def main():
    parser = argparse.ArgumentParser(description="YoungCC calculations, JSON or Gaussian .inp/.gjf/.com input")
    parser.add_argument("input", type=Path, help="JSON or Gaussian input")
    parser.add_argument("-o", "--output", type=Path, help="Output prefix (default: INPUT.youngcc)")
    parser.add_argument("--max-cycles", type=int, default=80)
    args = parser.parse_args()
    prefix = args.output or args.input.with_suffix(".youngcc")
    if Path(f"{prefix}.json").resolve() == args.input.resolve():
        parser.error("The output JSON must differ from the input file.")
    try:
        if args.input.suffix.lower() in {".inp", ".gjf", ".com"}:
            text = args.input.read_text(encoding="utf-8-sig")
            protected = {path.resolve() for path in (
                args.input, args.input.with_suffix(".out"), Path(f"{prefix}.log"), Path(f"{prefix}.json"),
            )}
            job = read_job(text, protected)
            if job is not None:
                from .geometry_workflow import run_job

                run_job(job, prefix, max_cycles=args.max_cycles)
                return 0
            molecule = parse_gaussian(text)
        else:
            molecule = read_input(args.input)
        result = calculate(molecule, prefix, max_cycles=args.max_cycles)
    except (ValueError, OSError, ArithmeticError, RuntimeError) as error:
        print(f"YoungCC: {error}", file=sys.stderr)
        return 2
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
