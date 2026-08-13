# Presentation results

Only the corrected Bloch-based result chain is retained in the current
repository. Superseded pre-Bloch generated outputs remain available through
Git history.

Use the folders in this order when presenting the project:

1. `bloch_corrected/step_13_hfss_supercell_validation/` — validates passive
   supercell extraction and the fitted-cell comparison.
2. `bloch_corrected/step_21_hfss_pattern_comparison/` — compares the physical
   13-3-13, 15-4-15, and 16-2-16 HFSS structures.
3. `bloch_corrected/step_22_*_physical_nonlinear_scan/` — records the controlled
   nonlinear comparison of all three patterns.
4. `bloch_corrected/step_23_15_4_15_mode_ablation/` and
   `step_24_15_4_15_harmonic_diagnosis/` — explain why the earlier 15-4-15
   candidate was rejected.
5. `bloch_corrected/step_25_16_2_16_refinement/` — contains the final selected
   design, dense gain curves, sensitivity checks, mode ablations, and solver
   failure audit.
6. `bloch_corrected/step_26_periodic_eigenmode_verification/` — records the
   preliminary independent periodic-eigenmode comparison with the ABCD Bloch
   dispersion, including its convergence and mode-tracking limitations.

The authoritative final numerical summary is
`bloch_corrected/step_25_16_2_16_refinement/06_assessment.json`.
