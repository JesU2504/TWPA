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
16-2-16 loading patterns. The current 15-4-15 nonlinear candidate produces
useful gain but excessive ripple. Mode-ablation identifies the second signal
harmonic (`s2`) interaction as the strongest contributor to the principal
gain dip, while the `ps/pi` modes are essential for the useful gain. See
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

## Environment

The local virtual environment is intentionally not committed. Environment and
dependency notes are recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md). Large
AEDT project/result databases are also excluded; the compact Touchstone/CSV
exports needed for the offline analysis are versioned under `hfss_inputs/`.
