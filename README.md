# NbTiN kinetic-inductance TWPA

This repository contains the HFSS-to-Python workflow used to design and audit
an NbTiN kinetic-inductance travelling-wave parametric amplifier (KI-TWPA).

The project combines:

- HFSS two-port simulations of unloaded, loaded, and patterned supercells;
- Touchstone-based extraction of passive dispersion and effective cell models;
- nonlinear coupled-mode simulations of pump, signal, idler, harmonic, and
  parasitic mixing modes;
- operating-point, loading-pattern, gain-ripple, and mode-ablation studies;
- reproducible figures and numerical result tables.

## Current status

The passive HFSS data supports comparisons among 13-3-13, 15-4-15, and
16-2-16 loading patterns. Periodic propagation is now extracted from the
reciprocal-normalized ABCD eigenvalues of each physical HFSS supercell. The
nonlinear workflows replace the solver's finite-network transmission phase,
attenuation, and reflection with Bloch phase, evanescent attenuation, and
Bloch-impedance mismatch throughout the measured 3--27 GHz band.

The former 15-4-15 operating point remains highly rippled after this correction:
the corrected eight-mode check gives 25.20--41.32 dB, or 16.11 dB ripple. This
point has not yet been reoptimized under the corrected dispersion model. New
results are written below `results/bloch_corrected/`; historical result folders
are preserved for comparison. Port-launch de-embedding is still unavailable
in the supplied 50-ohm-renormalized exports. See
[`CLAUDE_HANDOFF.md`](CLAUDE_HANDOFF.md) for the detailed audit brief,
assumptions, current problems, and results.

## Repository map

- `scripts/`: fitting, validation, optimization, and nonlinear diagnostics;
- `code/`: analysis notebook and shared project utilities;
- `hfss_inputs/`: compact HFSS exports, configurations, and fitted parameters;
- `results/`: generated figures and numerical outputs for each analysis step;
- `tests/`: consistency and regression tests;
- `models/`: serialized imported-amplifier model data;
- `tutorials/`: reference notebooks used by the workflow.

The Bloch extraction implementation is in `code/hfss_bloch.py`, with matched
and mismatched transmission-line regression tests in `tests/test_twpa_project.py`.

## Environment

The local virtual environment is intentionally not committed. Environment and
dependency notes are recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md). Large
AEDT project/result databases are also excluded; the compact Touchstone/CSV
exports needed for the offline analysis are versioned under `hfss_inputs/`.
