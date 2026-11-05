"""Command-line interface for YoungCC."""

import argparse
import sys
from pathlib import Path

from .input import read_input
from .output import calculate


def main():
    parser = argparse.ArgumentParser(description="YoungCC restricted single-point calculation")
    parser.add_argument("input", type=Path, help="JSON input")
    parser.add_argument("-o", "--output", type=Path, help="Output prefix (default: INPUT.youngcc)")
    parser.add_argument("--max-cycles", type=int, default=80)
    args = parser.parse_args()
    prefix = args.output or args.input.with_suffix(".youngcc")
    if Path(f"{prefix}.json").resolve() == args.input.resolve():
        parser.error("The output JSON must differ from the input file.")
    try:
        result = calculate(read_input(args.input), prefix, max_cycles=args.max_cycles)
    except (ValueError, OSError, ArithmeticError) as error:
        print(f"YoungCC: {error}", file=sys.stderr)
        return 2
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
