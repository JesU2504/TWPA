# Solver provenance

The Python source matches the `twpasolver 0.0.1` installation used for the
2026-08-31 validation. Compiled caches are excluded. The original
`installation/twpasolver` directory is unavailable; the upstream commit is unknown.

Upstream: [twpalab/twpasolver](https://github.com/twpalab/twpasolver), Apache-2.0.
The license and installed metadata are included. The local `pyproject.toml`
provides packaging. The package initializer falls back to version `0.0.1` when
distribution metadata is unavailable, which permits direct use of the vendored
source. The reproduction scripts add this source directory to Python's import
path through `scripts/_bootstrap.py`.
