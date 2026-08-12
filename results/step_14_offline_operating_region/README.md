# Offline HFSS-corrected operating-region scan

This scan uses the measured physical-supercell phase correction and the minimal
3WM gain model.  It is provisional because the HFSS phase data are spaced by
0.5 GHz.

## Selected provisional setting

- Pump: 12.940 GHz
- DC bias: 315 uA
- Pump current: 310 uA
- Minimum 4-8 GHz gain: 35.94 dB
- Mean 4-8 GHz gain: 37.57 dB
- Ripple: 2.37 dB
- Strict target passed: True

## Local tolerance check

The 27-point cube varies pump frequency by +/-0.01 GHz and both currents by
+/-10 uA.  9 of 27 points pass both the
17 dB minimum-gain and 3 dB ripple requirements.

This result identifies where to operate and what to verify with a fine HFSS
sweep.  It is not yet a final device guarantee.
