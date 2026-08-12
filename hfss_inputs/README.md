# HFSS-to-Python handoff

This directory is the boundary between the future HFSS work and the existing
Python analysis.

1. Run `scripts/fit_hfss_unit_cells.py` after replacing any Touchstone export.
2. Review `hfss_cell_parameters.fit.json` and its acceptance gates.
3. Keep `source` set to exactly `HFSS` only for fitted electromagnetic data.
4. Adjust pump, bias, and cell counts in `amplifier_config.json` if the design
   changes.
5. Run `code/step_11_HFSS_Imported_Amplifier_Analysis.ipynb` from a clean
   kernel.

The notebook refuses to call placeholder data validated HFSS data. The template
can still be used with `ALLOW_PLACEHOLDER = True` to test the workflow while
HFSS is unavailable.

## CSV columns

- `cell_type`: exactly `unloaded` or `loaded`.
- `stub_length_um`: physical stub length.
- `L_pH`, `C_fF`, `Lf_pH`: fitted `LCLfBaseCell` values.
- `loss_tangent`: fitted/effective dielectric-loss term used by the circuit
  model; use zero only for a deliberately lossless study.
- `Z0_ohm`: Bloch/characteristic impedance at the extraction frequency.
- `fit_s21_rmse_db`: S21 magnitude fit error outside stopbands.
- `fit_phase_rmse_deg`: phase-per-cell fit error over 4–14 GHz.
- `source`: `PLACEHOLDER` or `HFSS`.

Keep the full Touchstone sweep separately. This two-row file contains only the
two selected cells used to build the complete periodic amplifier.

`amplifier_config_candidate_13_3_13.json` records the loading-pattern candidate
found during operating-point optimization. It intentionally does not replace
`amplifier_config.json`: the candidate remains pending a full 58 um HFSS
supercell validation because the reflection-aware extended-mode model did not
confirm the flat minimal-model gain prediction.

## 15-4-15 export provenance (2026-08-12)

The files named `supercell_15_4_15_native_*.s2p` and
`supercell_15_4_15_port_Zo_*.csv` were imported from the updated HFSS archive.
They do **not** constitute an independent, unrenormalized modal-impedance data
set:

- the 3--14 GHz native export and the earlier fine export have identical
  frequency grids and differ by at most `8.69e-15` in any complex S entry;
- the 14--27 GHz native and earlier auxiliary exports are exactly identical;
- both native Touchstone files declare a 50 ohm reference and list port
  impedances of `50 + j0` ohm at every frequency;
- both Port Zo CSV reports likewise contain `50 + j0` ohm for both ports at
  every frequency.

The low- and high-band files carry the copied HFSS solution labels
`Sweep_16_2_16_fine` and `Sweep_16_2_16_aux`, respectively, even though their
project and design metadata identify the 15-4-15 design. Treat these labels as
provenance warnings and verify the active design/sweep before any later export.
The complete AEDT project and result tree remain preserved in the source
`Desktop/hffs_inputs.zip`; only compact numerical exports are copied here.
