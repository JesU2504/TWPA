# HFSS-corrected extended nonlinear validation

## Outcome

The provisional setting (pump 12.94 GHz, DC bias 315 uA, pump current 310 uA)
passes only in the three-frequency model.  That model predicts 35.90 dB
minimum gain and 2.38 dB ripple over 4-8 GHz.

When eight nonlinear modes are included, the smooth response is replaced by
strong notches.  The most complete available model predicts -11.18 dB minimum
gain and 54.24 dB ripple, so the operating point does not pass the final gate.

## Mode-ablation result

- Basic pump/signal/idler: 34.06 dB minimum, 3.56 dB ripple.
- Add second pump harmonic: 33.96 dB minimum, 3.99 dB ripple.
- Add pump-plus-signal/idler conversions: -2.82 dB minimum, 47.46 dB ripple.
- Add second signal/idler harmonics: 7.74 dB minimum, 34.08 dB ripple.
- All eight modes: -5.78 dB minimum, 50.39 dB ripple in the ideal model.

The pump-plus-signal/idler conversion channels are the largest warning, with
signal/idler harmonics also important.

## Interpretation and next HFSS data

This is a warning, not yet proof that the geometry fails.  The HFSS phase
correction is sampled every 0.5 GHz and stops at 20 GHz, while the added modes
reach approximately 22-26 GHz.  The next physical-supercell export should
therefore include:

1. A fine discrete sweep over 3-14 GHz (0.05 or 0.1 GHz spacing).
2. An auxiliary sweep over 14-27 GHz (0.1 or 0.2 GHz spacing).
3. RI Touchstone data with 15 digits and 50-ohm normalization.

The current geometry should not be changed until those data are used to repeat
the phase correction and extended-mode optimization.
