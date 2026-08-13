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
and reflection with Bloch phase and Bloch-impedance mismatch throughout the
measured 3--27 GHz band. Extracted stopband evanescence is retained separately
for diagnostics. It is not treated as dissipative material loss by default,
because the supplied HFSS models are lossless.

The corrected physical-pattern comparison selected 16-2-16. A staged
eight-mode refinement followed by validation on a 0.01 GHz signal grid found a
64.6 mm candidate at a 12.66 GHz pump, 175 µA DC bias, and 150 µA pump current.
It gives 24.50--25.87 dB over 4--8 GHz, corresponding to 1.37 dB peak-to-peak
ripple. All ten dense finalists pass the 17 dB minimum and 3 dB ripple targets.
The selected point also passes ideal, loss-only, reflection-enabled, and
leave-one-mode-family-out checks.

This is a model-level engineering candidate, not yet a hardware prediction.
The nonlinear calculation uses a scalar `Istar = 2 mA`, whose relation to the
fabricated film and line cross-section has not been independently validated.
The peak/RMS convention of the solver current amplitudes also needs an explicit
source-level or analytic calibration. The HFSS exports provide passive Bloch
dispersion and impedance, but not nonlinear Bloch-mode overlap factors. The
coupled-envelope model propagates forward nonlinear modes only; its optional
reflection term is not an independent backward-wave nonlinear solution.

Corrected results are written below `results/bloch_corrected/`; historical
result folders are preserved for comparison. Port-launch de-embedding is still
unavailable in the supplied 50-ohm-renormalized exports. See
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

The Bloch extraction implementation is in `code/hfss_bloch.py`, with matched,
mismatched, and periodic-stopband regression tests in
`tests/test_twpa_project.py`.
The staged 16-2-16 engineering refinement and 0.01 GHz final validation are in
`scripts/refine_bloch_16_2_16.py`.

## Environment

The local virtual environment is intentionally not committed. Environment and
dependency notes are recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md). Large
AEDT project/result databases are also excluded; the compact Touchstone/CSV
exports needed for the offline analysis are versioned under `hfss_inputs/`.
