# KI-TWPA modelling with HFSS and TWPAlab

This project develops a simulation workflow for a three-wave-mixing kinetic-inductance
travelling-wave parametric amplifier (KI-TWPA). It connects electromagnetic simulations
in Ansys HFSS to the nonlinear coupled-mode solver in
[TWPAlab's twpasolver](https://github.com/twpalab/twpasolver).

The goal is to predict amplifier performance from the transmission-line geometry:
how periodic loading changes dispersion, where phase matching occurs, and how the
pump and DC bias affect gain and bandwidth. The current model studies a NbTiN
microstrip line with capacitive stubs and a 15–4–15 loading pattern.

## From geometry to gain

HFSS provides the passive microwave response. Single-cell S-parameters are converted
to ABCD matrices and fitted to an equivalent circuit with inductance `L`, capacitance
`C`, and stub inductance `Lf`. These fits connect stub length to line impedance.

A separate HFSS simulation describes one complete periodic supercell. Bloch analysis
of its ABCD matrix gives the propagation constant and Bloch impedance. The impedance
sets the boundary reflection coefficient relative to the 50-ohm ports.

These quantities are passed to `twpasolver`, together with the line length, DC bias,
pump current, and nonlinear current scale. The solver calculates pump–signal–idler
coupling along the line. The analyses cover gain spectra, phase matching, pump-frequency
selection, and reflection-induced ripple, including a comparison with a stub-only
redesign.

The HFSS extraction and solver interface are implemented in
[`code/hfss_bloch.py`](code/hfss_bloch.py) and
[`code/device_models.py`](code/device_models.py).

## Running the model

The Python calculations start from the Touchstone exports in `hfss_inputs/`.
Changing the electromagnetic geometry requires new HFSS exports. The available
project and setup specifications are in `hfss_projects/` and
[`docs/HFSS_MODEL_SPEC.md`](docs/HFSS_MODEL_SPEC.md).

With Python 3.12, install the dependencies from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Fit the baseline cells and calculate the HFSS–Bloch amplifier response:

```bash
.venv/bin/python scripts/fit_hfss_unit_cells.py
.venv/bin/python scripts/reproduce_howe_fig3_hfss_bloch.py
```

The baseline calculation writes dispersion, gain maps, bandwidth scans, and gain
profiles to [`results/bloch_corrected/step_34_howe_fig3_hfss_bloch_reproduction/`](results/bloch_corrected/step_34_howe_fig3_hfss_bloch_reproduction/).

Operating currents, the nonlinear current scale, and cell counts are defined in
[`hfss_inputs/amplifier_config.json`](hfss_inputs/amplifier_config.json). Pump and
signal sweep ranges are defined in the analysis scripts. The supercell geometry
must remain consistent with the supplied HFSS exports.

Further analyses use the same extraction and solver:

- `scripts/fit_stub_length_sweep.py`: impedance versus stub length and the 50/80-ohm crossings.
- `scripts/refine_pump_bandwidth.py`: pump selection for a target bandwidth.
- `scripts/reproduce_howe_fig_s5_gain_ripple.py`: baseline reflection-ripple calculation.
- `scripts/reproduce_fig3_zmatched_supercell.py` followed by
  `scripts/compare_zmatched_gain_ripple.py`: gain and ripple of the redesigned line.

[Environment notes](docs/ENVIRONMENT.md) record the tested dependencies and the
macOS CyRK/OpenMP setup. A copy of the solver source used for the calculations is
included in `vendor/twpasolver/`, with its license and provenance.

## Model scope

The current HFSS exports cover 1–27 GHz. The model treats dissipative loss as zero
and keeps stopband evanescence separate from material attenuation. Extended-mode
calculations use an analytic cell model outside the exported frequency range.
The nonlinear current scale is an input parameter; agreement between reruns does
not establish agreement with a fabricated device.

The 40/50/60 µm points in the extended stub study currently use archived fits because
their raw exports are unavailable. This does not affect the baseline supercell inputs.
The available AEDT project requires a fresh solve before its cached results can be used.

## Results and references

Full citations and source links are in [`papers_and_others/README.md`](papers_and_others/README.md).
The repository does not bundle full-text papers or standalone copies of their figures.
Digitized curves in `data/reference/` support comparisons
with published impedance and gain results. Calculation details, data sources, and
verification procedures are documented in [`docs/FIGURE_WORKFLOW.md`](docs/FIGURE_WORKFLOW.md).

Figures for reports and presentations can be regenerated with
`scripts/reproduce_figures.py`; they are one application of the modelling workflow.
Presentation files are distributed separately from the source repository.

## License

Original code and documentation in this repository are available under the
[MIT License](LICENSE). The vendored `twpasolver` source remains under its bundled
Apache-2.0 license. Digitized literature data retain the attribution described in
[`papers_and_others/README.md`](papers_and_others/README.md).
