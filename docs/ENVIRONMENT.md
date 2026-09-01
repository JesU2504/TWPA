# Reproduction environment

The validated environment uses Python 3.12.12 on macOS 14.5, Apple silicon.
`requirements.txt` pins the numerical dependencies, including NumPy 1.26.4,
SciPy 1.17.1, Matplotlib 3.11.1, scikit-rf 2.0.1, Numba 0.66.0, Pydantic 2.13.4,
CyRK 0.8.8, and Pillow 12.3.0. Fresh dependency installation remains untested.

The CyRK extensions required a runtime search path for
`/opt/homebrew/opt/libomp/lib`. Reinstalling CyRK can overwrite that local binary
repair; import and full-pipeline checks are required afterward.

The `twpasolver 0.0.1` Python source and license are included under
`vendor/twpasolver/`. The original installation source is unavailable and its
upstream commit is unknown. The runner selects the included copy through
`PYTHONPATH`; provenance is recorded in `vendor/twpasolver/PROVENANCE.md`.

The runner uses Matplotlib's Agg backend and stores Matplotlib and Numba caches
under `.cache/`. `.venv/` and `.cache/` are ignored by Git. HFSS is required to
regenerate electromagnetic exports, but not to calculate figures from saved exports.
