"""Repository paths shared by command-line scripts."""

import os
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Python 3.12 is required; found {sys.version.split()[0]}")

ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / ".cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
for directory in (ROOT / "vendor/twpasolver", ROOT / "code", ROOT / "scripts"):
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
