"""Check the Python runtime and imports needed by the calculation pipeline."""

import importlib
import platform

from _bootstrap import ROOT

MODULES = ("CyRK", "h5py", "matplotlib", "numba", "numpy", "pandas", "scipy", "skrf", "twpasolver")


def main():
    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as error:
            print(f"::error title=Import failed::{module}: {error}")
            raise
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"Loaded {len(MODULES)} numerical packages and the solver from {ROOT / 'vendor/twpasolver'}"
    )


if __name__ == "__main__":
    main()
