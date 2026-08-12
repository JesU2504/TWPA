# HFSS-derived operating-point optimization

## Outcome

The original `15-4-15` loading pattern was optimized first. Its best minimal
three-mode point was 11.22 GHz pump, 300 uA DC bias, and 360 uA pump current.
It predicted 23.42 dB mean gain over 4-8 GHz with 4.52 dB peak-to-peak ripple,
so it did not meet the 3 dB ripple goal.

Keeping the total amplifier length near 81.6 mm, a scan of nearby loading
patterns selected `13-3-13` (13 unloaded cells, 3 loaded cells, 13 unloaded
cells), repeated 1407 times. The total is 40,803 cells or 81.606 mm at the
2 um cell pitch.

Under the minimal three-mode model, the candidate point is:

- Pump frequency: 13.13 GHz
- DC bias: 350 uA
- Pump current: 340 uA
- Mean 4-8 GHz gain: 34.24 dB
- Peak-to-peak ripple: 2.66 dB
- Fraction of 4-8 GHz above 17 dB: 100%

This is **not yet a final accepted operating point**. The reflection-aware
extended-mode calculation did not reproduce the minimal-model flat response.
It showed a sharp transition between a smooth low-gain regime and a high-gain,
strongly rippled regime; none of the 60 targeted full-model points met all of
the gain, coverage, and ripple gates.

## Required next HFSS validation

Build and solve one complete `13-3-13` physical supercell before promoting the
candidate configuration:

1. Make a 58 um long structure: 29 cells times the 2 um pitch.
2. Use 13 cells with the 12 um stub, then 3 cells with the 3.9 um stub, then
   another 13 cells with the 12 um stub.
3. Keep all stackup, NbTiN sheet impedance, widths, and port construction
   identical to the validated single-cell simulations.
4. Put one single-mode wave port at each longitudinal end and run a converged
   discrete sweep over 0.2-30 GHz.
5. Export the two-port Touchstone file as `supercell_13_3_13.s2p`.

The supercell S-parameters will capture interfaces between loaded and unloaded
cells directly. They can then replace the idealized lumped-cell cascade for the
final dispersion, reflection, and gain verification.

## Main artifacts

- `fine_operating_point_scan.json`: fine current/pump scan for `15-4-15`
- `loading_pattern_scan.json`: minimal-model loading-pattern scan
- `selected_pattern_operating_scan.json`: conservative `13-3-13` current scan
- `extended_mode_validation.json`: minimal versus extended-model comparison
- `full_model_targeted_scan.json`: targeted reflection-aware scan
- `extended_mode_validation.png`: model-comparison plot
- `full_model_selected_gain.png`: best result according to the targeted scan
