"""Add Homebrew's libomp directory to the installed CyRK extensions on macOS."""

import importlib.util
import platform
import subprocess
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


def libomp_directory() -> Path:
    result = subprocess.run(
        ["brew", "--prefix", "libomp"],
        check=True,
        capture_output=True,
        text=True,
    )
    directory = Path(result.stdout.strip()) / "lib"
    if not directory.is_dir():
        raise RuntimeError(f"Homebrew libomp directory not found: {directory}")
    return directory


def cyrk_extensions() -> list[Path]:
    spec = importlib.util.find_spec("CyRK")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("CyRK is not installed in this Python environment")
    package = Path(next(iter(spec.submodule_search_locations)))
    extensions = sorted(package.rglob("*.so"))
    if not extensions:
        raise RuntimeError(f"No CyRK extension modules found under {package}")
    return extensions


def main() -> None:
    if platform.system() != "Darwin":
        print("CyRK OpenMP repair skipped: only needed on macOS")
        return

    libomp = libomp_directory()
    changed = 0
    extensions = cyrk_extensions()
    for extension in extensions:
        load_commands = subprocess.run(
            ["otool", "-l", str(extension)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if str(libomp) in load_commands:
            continue
        subprocess.run(
            ["install_name_tool", "-add_rpath", str(libomp), str(extension)],
            check=True,
        )
        changed += 1

    print(f"CyRK OpenMP path ready: {len(extensions)} extensions checked, {changed} updated")


if __name__ == "__main__":
    main()
