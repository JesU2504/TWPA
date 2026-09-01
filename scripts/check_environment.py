"""Check the Python runtime and imports needed by the calculation pipeline."""

import importlib
import os
import platform

from _bootstrap import ROOT

MODULES = ("CyRK", "h5py", "matplotlib", "numba", "numpy", "pandas", "scipy", "skrf", "twpasolver")


def main():
    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as error:
            if os.environ.get("GITHUB_ACTIONS"):
                print(f"::error title=Import failed::{module}: {error}")
            if module == "CyRK" and platform.system() == "Darwin":
                print(
                    "CyRK could not load its OpenMP runtime. Run "
                    "`.venv/bin/python scripts/fix_cyrk_libomp.py`, then retry."
                )
            raise
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"Loaded {len(MODULES)} numerical packages and the solver from {ROOT / 'vendor/twpasolver'}"
    )


if __name__ == "__main__":
    main()
