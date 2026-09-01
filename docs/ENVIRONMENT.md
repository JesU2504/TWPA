# Environment

The reference environment is Python 3.12.12 on macOS 14.5, Apple silicon. The
pinned dependencies also resolve in a clean Python 3.12 environment. GitHub Actions
installs them on Linux, macOS, and Windows and runs the complete `--recompute
--verify` recalculation on each. `--verify` checks the recalculated values against
the archived reference chart data within tolerance; cross-platform BLAS and libm
differences move results at roughly the tenth significant figure, so regenerated
tables are not expected to be byte-identical to the tracked copies.

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

On Linux, install `libomp` from the system package manager (the CI job uses
`apt-get install libomp-dev`); no post-install repair is needed. On Windows, the
CyRK wheels bundle their OpenMP runtime, so `fix_cyrk_libomp.py` is not required
and exits without changes if run. `scripts/check_environment.py` reports the
platform and confirms every numerical import, including the vendored solver.

All tracked text files are normalized through `.gitattributes`: source and
metadata as LF, and the generated `.csv` tables and `.s2p` exports byte-preserved
(the `csv` module writes CRLF on every platform). Set `git config --global
core.autocrlf false` before cloning on Windows so a checkout does not rewrite line
endings before the attributes apply.

The `twpasolver 0.0.1` source and license are included under `vendor/twpasolver/`.
`scripts/_bootstrap.py` loads that copy for both the main runner and individual
scripts. Its original upstream commit is unknown; the available provenance is in
`vendor/twpasolver/PROVENANCE.md`.

The runner uses Matplotlib's Agg backend and keeps Matplotlib and Numba caches under
`.cache/`. Linux, macOS, and Windows are exercised in CI. HFSS is needed to
regenerate electromagnetic exports, but not to calculate figures from the saved
Touchstone files.
