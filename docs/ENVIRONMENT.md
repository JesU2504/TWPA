# Environment

The reference environment is Python 3.12.12 on macOS 14.5, Apple silicon. The
pinned dependencies also resolve in a clean Python 3.12 environment. GitHub Actions
installs them on Linux and runs the complete recalculation.

CyRK requires an OpenMP runtime. On Apple silicon, `brew bundle` installs `libomp`
under `/opt/homebrew/opt/libomp/lib`, but the CyRK 0.8.8 wheels do not search that
directory. Repair every installed CyRK extension and check the environment with:

```bash
.venv/bin/python scripts/fix_cyrk_libomp.py
.venv/bin/python scripts/check_environment.py
```

The repair script obtains the Homebrew prefix, finds all CyRK `.so` files, and adds
the missing runtime path with `install_name_tool`. It is safe to run again after
reinstalling or upgrading CyRK. If an import still fails, inspect each extension
with `otool -L` and `otool -l`.

The `twpasolver 0.0.1` source and license are included under `vendor/twpasolver/`.
`scripts/_bootstrap.py` loads that copy for both the main runner and individual
scripts. Its original upstream commit is unknown; the available provenance is in
`vendor/twpasolver/PROVENANCE.md`.

The runner uses Matplotlib's Agg backend and keeps Matplotlib and Numba caches under
`.cache/`. Linux is tested in CI. Windows has not been tested. HFSS is needed to
regenerate electromagnetic exports, but not to calculate figures from the saved
Touchstone files.
