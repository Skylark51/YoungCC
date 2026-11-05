"""Compatibility entry point; implementation lives in interfaces/cli.py."""
from .interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
