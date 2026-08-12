# HFSS unit-cell model specification and selected baseline

Status: **executed and fitted** using Ansys HFSS 2026 R1. The selected baseline
preserves the 12 µm and 3.9 µm arm lengths and uses the extracted impedances
43.09 Ω and 70.20 Ω. The nominal 48 Ω and 78 Ω values remain comparison values.

## Purpose

Extract the linear unit-cell S-matrix and fitted `L`, `C`, and `Lf` values used
by `twpasolver`. Do not model the complete 40,800-cell amplifier in HFSS.

## Stack and geometry

- 10 nm NbTiN signal sheet with a 30 pH/square kinetic-inductance boundary.
- 100 nm amorphous-silicon dielectric with relative permittivity 9.1.
- 100 nm Nb top ground plane.
- 1 µm center-line width, 1 µm stub width, and 1 µm spacing.
- 2 µm cell pitch.
- Parameter `stub_length_um`, swept from 2 to 80 µm.
- Required verification points: 3.9 µm and 12 µm.

## Solves and exports

1. Driven-modal two-port solve from 0.2 to 30 GHz.
2. Low-frequency solve below 100 MHz for quasistatic parameter extraction.
3. Export one Touchstone file per stub length.
4. Export CSV columns: stub length, Bloch impedance, phase per cell, fitted L,
   fitted C, fitted Lf, and fit residual.
5. Fit the `LCLfBaseCell` response only outside stopbands.

## Acceptance gates

- 43.09 Ω ±5% at 12 µm (selected HFSS baseline; nominal reference 48 Ω).
- 70.20 Ω ±5% at 3.9 µm (selected HFSS baseline; nominal reference 78 Ω).
- Loaded-to-unloaded impedance ratio 1.629 ±5% (nominal ratio 1.625).
- Lumped-model S21 error below 0.5 dB outside stopbands.
- Phase error below 5 degrees per cell from 4 to 14 GHz.
- Full-line stopband shift below 5% after replacing published surrogate values.

The selected values and fit residuals are stored in
`hfss_inputs/hfss_cell_parameters.csv` and its adjacent fit report.
