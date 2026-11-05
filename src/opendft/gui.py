"""Compatibility entry point; implementation lives in interfaces/gui.py."""
from .interfaces.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
