# Environment

The reference environment is Python 3.12.12 on macOS 14.5, Apple silicon. The
pinned dependencies also resolve in a clean Python 3.12 environment. GitHub Actions
installs them on Linux and runs the complete recalculation.

CyRK requires an OpenMP runtime. On Apple silicon, `brew bundle` installs `libomp`
under `/opt/homebrew/opt/libomp/lib`. Check the installation with:

```bash
.venv/bin/python scripts/check_environment.py
```

If CyRK still cannot resolve `@rpath/libomp.dylib`, inspect its extension with
`otool -L`. The Homebrew path can be added with:

```bash
install_name_tool -add_rpath /opt/homebrew/opt/libomp/lib /path/to/cyrk-extension.so
```

The `twpasolver 0.0.1` source and license are included under `vendor/twpasolver/`.
`scripts/_bootstrap.py` loads that copy for both the main runner and individual
scripts. Its original upstream commit is unknown; the available provenance is in
`vendor/twpasolver/PROVENANCE.md`.

The runner uses Matplotlib's Agg backend and keeps Matplotlib and Numba caches under
`.cache/`. Linux is tested in CI. Windows has not been tested. HFSS is needed to
regenerate electromagnetic exports, but not to calculate figures from the saved
Touchstone files.
