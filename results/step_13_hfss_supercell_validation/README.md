# Step 13: HFSS physical-supercell validation

This directory compares the physical 13/3/13 HFSS supercell with the separately simulated unit-cell cascade and the fitted LCLf model. The passive match is validated, but the accumulated dispersion correction is large enough that the nonlinear operating point must be retuned after a finer 3-14 GHz HFSS sweep.

`05_corrected_gain_assessment.json` and its companion CSV/plots apply the
coarse HFSS phase correction to the minimal-3WM model.  They are explicitly
provisional because the available supercell phase samples are spaced by
0.5 GHz.
