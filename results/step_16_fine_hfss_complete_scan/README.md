# Fine-HFSS complete-model operating scan

The joined physical-supercell datasets contain 111 complex S-parameter points
from 3--14 GHz and 66 points from 14--27 GHz.  Their shared 14 GHz samples are
identical, so the S21 phase can be unwrapped continuously across both sweeps.

The former operating point (pump 12.94 GHz, Idc 315 uA, Ip 310 uA) remains
strongly rippled in the complete eight-mode calculation.  A broader scan of 144
operating points found a more promising region at pump 12.80 GHz, Idc 400 uA,
and Ip 280 uA.  On the 0.1 GHz signal grid this point predicts:

- minimum gain: 24.00 dB;
- maximum gain: 41.85 dB;
- mean gain: 32.71 dB;
- 4--8 GHz coverage above 17 dB: 100%; and
- peak-to-peak ripple: 17.86 dB.

This is not a final pass because the ripple target is 3 dB.  It does show that
the physical 13-3-13 geometry can support broadband gain in the current model;
the next offline task is a finer local optimization around this operating
region, including lower-gain/current settings and possibly device length, to
trade excess gain for lower ripple.

The complete model includes pump, signal, idler, pump-plus-signal,
pump-plus-idler, second pump harmonic, second signal harmonic, and second idler
harmonic modes.  It still represents the full amplifier as repeats of the
simulated physical supercell and solves each signal frequency separately.
