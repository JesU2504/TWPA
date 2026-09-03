# Figure workflow and provenance

## Scope

The final 25-slide presentation is included as `KI-TWPA_3WM.pptx`, SHA-256
`e5108f274a1308f6e1dfb8473b56a8ec6b5788b2572eaf8dfda35786b1743d0d`.
It contains hand-authored material and is not generated or modified by the pipeline.
The numerical verification baseline was extracted from its charts. Cell fitting and
gain calculations start from saved electromagnetic exports; HFSS execution is a separate step.

## Execution order

`scripts/reproduce_figures.py --recompute --verify` runs these stages in order:

1. `fit_hfss_unit_cells.py`: refit the loaded/unloaded structural cells from
   broadband and low-frequency two-port exports; write the two-row parameter CSV
   and its fit report under `hfss_inputs/`.
2. `prepare_reference_curves.py`: copy archived digitized impedance and gain
   CSVs from `data/reference/` into step 32 and `results/paper_curves/`.
3. `fit_stub_length_sweep.py`: fit the nine available 2.5–12 µm stub lengths;
   write step-39 tables. These fits give approximately 43.09 Ω and 70.20 Ω for
   the unloaded/loaded structural cells.
4. `fit_extended_stubs.py`: fit restored 40/50/60 µm exports when present;
   otherwise use the archived fitted values. Write `results/extended_stubs.csv`.
5. `reproduce_howe_fig3_hfss_bloch.py`: stitch the three baseline supercell bands,
   calculate Bloch parameters, recompute k*, and solve the minimal-3WM pump/signal
   grid. Outputs are under `results/bloch_corrected/step_34_.../`.
6. `refine_pump_bandwidth.py`: sweep 13.80–13.92 GHz in 0.005 GHz steps, using
   0.01 GHz signal spacing; select the result nearest the target 3.15 GHz bandwidth.
   Saved results select 13.865 GHz, 3.16 GHz bandwidth, and 33.74 dB peak gain.
7. `reproduce_howe_fig_s5_gain_ripple.py`: calculate the baseline eight-mode
   reflection ripple, plus its analytic diagnostic, under step 37.
8. `reproduce_fig3_zmatched_supercell.py`: calculate the stub-only redesigned
   device gain grid and profiles under step 40, using the corresponding exports.
9. `compare_zmatched_gain_ripple.py`: calculate the redesigned eight-mode ripple
    and compare its metrics with the baseline under step 41.
10. `plot_slide_figures.py`: create the five standalone plot groups from saved
    numerical data and run the optional numerical checks.

Without `--recompute`, the pipeline plots saved numerical results. Individual
calculation scripts also produce supporting diagnostic panels.

## Source details

### Slide 13

Nine current two-port fits come from the nine raw length pairs under `hfss_inputs/`.
The three additional points come from `data/reference/hfss_40_50_60um_fits.csv`.
The fit report accompanies the CSV. The source exports
`kinetic_{40,50,60}um_{debug,lowfreq}.s2p` are unavailable.

The empirical trend is `Z0(l) = A / sqrt(l + l0) + Zinf`. A two-stage grid search
gives A ≈ 146.85129118, l0 ≈ 0.7406 µm, Zinf ≈ 1.61367272 Ω, and RMSE ≈ 0.5290 Ω.
This trend is fitted to the extracted impedances, separately from the two-port
L/C/Lf calculation. The paper curve was digitized from the Fig. 2 raster crop and
is retained as `data/reference/paper_impedance_fig2.csv`.

### Slides 16–18

The baseline supercell is 15 unloaded + 4 loaded + 15 unloaded cells (34 total),
with corrected 12.1 µm long stubs in the supercell exports. The structural
single-cell fit is at 12.0 µm. The baseline exports cover 1–3, 3–14, and 14–27 GHz.
The operating point is retained in `hfss_inputs/amplifier_config.json`:
1,200 supercells, I* = 2 mA, Idc = 220 µA, Ip = 100 µA, and Is = 1.4 µA.

The slide-16 transmission trace is the existing model's accumulated Bloch-
evanescence estimate, `-20 log10(e) N_cells alpha_B`. It is not a direct HFSS
simulation of full-device S21 including all finite-length boundary effects.
The bare-line reference is an analytic cell model, not another full-wave dataset.

The slide-17 grid extends the signal range to 14 GHz. Samples with idler
`fp - fs < 1 GHz` are masked rather than extrapolated. The zero contour is the
calculated effective phase mismatch from the same saved grid.

For slide 18, the magenta experimental vector is the selected Fig. 4(a) path on
page 6 of Howe et al., arXiv:2507.07706v1. The gray simulation is the red maximum-GBP
path in Fig. 3(d), page 5. The digitized CSVs are retained in `data/reference/`;
the full paper and its figure images are not bundled. These are digitized figure
traces, not author-provided raw measurement data. The paper's two traces represent
different cases; the model uses a different pump frequency from the experiment.
The deck samples the experimental path every fourth point and the refined model
every fifth point, retaining the final endpoint; the plotter does the same.

### Slide 25 and appendices

The baseline ripple uses pump 14.10 GHz; the redesign uses pump 16 GHz.
The displayed peak-to-peak values are 4.09535 dB and 2.74155 dB. Their mean gains
and reflection coefficients refer to different mid-band signal windows, not an
identical-frequency comparison. The two ripple values should be compared together
with gain and bandwidth.

The model treats dissipative loss as zero; Bloch evanescence is retained separately.
The extended mode calculations may sample frequencies above the 27 GHz measured
band, where the analytic cell fallback is used.
Scalar nonlinear parameters, forward coupled envelopes, and reflection treatment
remain model assumptions. The verifier tests the calculation, not a fabricated device.

## Presentation

The presentation combines regenerated numerical plots with hand-authored text,
diagrams, screenshots, and other visual material. The repository does not attempt
to rebuild the PowerPoint file. Running the figure workflow writes PNG and PDF plots
to `figures/`; updating the corresponding slides remains a manual step.

## Reference data

Full citations, source links, and attribution for the retained traces are listed in
[`papers_and_others/README.md`](../papers_and_others/README.md). The digitized
comparison curves are committed under `data/reference/`; the pipeline copies them
into `results/` via `prepare_reference_curves.py` and treats them as fixed inputs.
`--recompute` recalculates the device model only and does not redigitize the
literature. The digitization method for each curve is recorded in
`papers_and_others/README.md`; the repository does not ship figure-extraction tools
or the source figure images.

## HFSS source availability

The Python workflow uses the saved Touchstone exports. The available AEDT project,
manual setup, and cache handling are covered by `HFSS_MODEL_SPEC.md` and
`HFSS_STUB_LENGTH_SWEEP_SPEC.md` in this directory.

## Verification

`--verify` compares recalculated/replotted chart values against a frozen extraction
of the presentation's native charts. It checks the 12 impedance points and fitted
trend, the three gain series, peak gain, and the two appendix ripple values.
The presentation is not opened or changed during verification. These checks catch a
wrong source curve, wrong pump, missing dependency, or numerical drift.
