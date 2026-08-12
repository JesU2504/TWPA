# KI-TWPA project — independent technical review brief

> **Update after review (2026-08-12):** The production scripts no longer add an
> HFSS-minus-lumped correction derived from finite-network S21 phase. They now
> extract branch-continuous complex Bloch eigenvalues and Bloch impedance from
> each measured supercell ABCD matrix and replace solver `k`, `alpha`, and
> `gammas` within 3--27 GHz. Historical results described below are preserved;
> corrected outputs are under `results/bloch_corrected/`. At the former
> 15-4-15 operating point, the corrected eight-mode result is 25.20--41.32 dB
> (16.11 dB ripple), so the point still fails the 3 dB flatness target. Port
> launch de-embedding still requires a new HFSS export.
>
> **16-2-16 refinement update:** A later Bloch-corrected eight-mode refinement
> selected 950 supercells (64.6 mm), a 12.61 GHz pump, 200 uA DC bias, and
> 120 uA pump current. On a 0.01 GHz signal grid it predicts 23.61--24.94 dB
> gain over 4--8 GHz, or 1.33 dB ripple. All ten dense finalists satisfy the
> 17 dB minimum and 3 dB ripple requirements. The selected point also passes
> the ideal, loss-only, reflection-enabled, and controlled mode-ablation checks.
> Results are in `results/bloch_corrected/step_25_16_2_16_refinement/`.

## Purpose of this review

We are trying to design and model a kinetic-inductance travelling-wave
parametric amplifier (KI-TWPA) based on a thin NbTiN transmission line. The
device is intended to use three-wave mixing under DC bias and microwave-pump
excitation to amplify weak microwave signals.

We are requesting an independent check of the work completed so far. Please
look for possible mistakes in the electromagnetic model, geometry, boundary
conditions, port definitions, parameter extraction, conversion of HFSS data
into the nonlinear model, mathematical assumptions, numerical implementation,
and interpretation of the results.

This document is intentionally not a plan for future work. Its purpose is to
describe what we attempted, what we obtained, and where an error may be hiding.

## Intended amplifier performance

The provisional engineering target is:

- signal band: 4–8 GHz;
- gain: at least 17 dB everywhere in that band;
- gain ripple: no more than 3 dB peak-to-peak;
- passive dispersion taken from physical Ansys HFSS simulations;
- nonlinear validation that includes the wanted signal/idler interaction and
  important unwanted mixing products.

HFSS models only the passive, linear electromagnetic structure. The nonlinear
gain is calculated offline in Python with `twpasolver`, using the propagation
information extracted from HFSS.

## Physical structure represented in HFSS

The simulated structure is a periodically loaded NbTiN centre line with open
transverse stubs. Two stub lengths are used:

- long or “unloaded” stub extension: 12 µm on each side of the centre line;
- short or “loaded” stub extension: 3.9 µm on each side;
- centre-line width: 1 µm;
- stub width: 1 µm;
- longitudinal cell pitch: 2 µm.

Layer and material parameters currently used are:

- NbTiN thickness represented geometrically at 10 nm;
- kinetic sheet inductance: 30 pH/square;
- dielectric: 100 nm amorphous silicon, relative permittivity 9.1;
- silicon substrate: 500 µm, relative permittivity 11.7;
- ground thickness: 100 nm;
- dielectric and substrate loss tangents presently set to zero.

The patterned NbTiN conductor is implemented as a zero-thickness sheet located
at the NbTiN plane. It does not have a bulk material assigned. Instead, HFSS
uses an impedance boundary named `NbTiN_Kinetic` with:

- sheet resistance = 0 ohm/square;
- sheet reactance = `2*pi*Freq*Lk_square` ohm/square;
- `Lk_square = 30 pH`;
- Infinite Ground Plane disabled.

The centre line and all stubs are united into one patterned sheet before this
boundary is assigned. The ground is modelled separately as PEC.

The model uses an air/vacuum box and two wave ports at the longitudinal ends.
The final supercell models were intended to use one propagating port mode,
50-ohm renormalization, no de-embedding, and integration lines drawn between
the signal conductor and ground with consistent polarity. Earlier debugging
runs temporarily used two port modes after HFSS reported an additional
propagating or slowly decaying mode. This history should be checked carefully
because inconsistent port-mode choices could contaminate comparisons.

## Loading patterns physically simulated

Three symmetric supercells were simulated:

| Pattern | Long cells left | Short cells | Long cells right | Total cells | Length |
|---|---:|---:|---:|---:|---:|
| 13-3-13 | 13 | 3 | 13 | 29 | 58 µm |
| 16-2-16 | 16 | 2 | 16 | 34 | 68 µm |
| 15-4-15 | 15 | 4 | 15 | 34 | 68 µm |

The 15-4-15 pattern was selected because it matches the reported loading count
in the reference paper *Kinetic Inductance Traveling-Wave Parametric
Amplifiers Near the Quantum Limit: Methodology and Characterization*.

For every physical pattern, two discrete two-port sweeps were exported and
joined:

- 3–14 GHz with 0.1 GHz spacing;
- 14–27 GHz with 0.2 GHz spacing.

The duplicate 14 GHz sample agrees between the two sweeps and is retained only
once. The Touchstone data were exported as 50-ohm-renormalized S parameters.

## Unit-cell fitting and HFSS-to-Python conversion

Separate long-stub and short-stub unit cells were simulated with the kinetic
sheet boundary. Corresponding PEC reference cells were also simulated. Python
fits each kinetic unit cell to an `L-C-Lf` equivalent circuit used by
`twpasolver`.

The fitted effective characteristic impedances are approximately:

- long/unloaded cell: 43.1 ohm;
- short/loaded cell: 70.2 ohm.

The intended reference values discussed earlier were approximately 48 and
78 ohm. Their ratio is similar, but the absolute fitted values differ.

For each physical supercell, the Python model constructs a cascade of fitted
long and short cells with the same loading count. It compares the phase of
that cascade with the unwrapped measured HFSS S21 phase. The difference is
divided by the number of cells in the supercell and added as a phase/propagation
correction to the periodic nonlinear model.

This conversion is a major point requiring review. In particular, please
check whether it is physically and mathematically valid to:

- infer propagation constant directly from the unwrapped phase of S21;
- treat a finite, mismatched supercell as an equivalent uniform propagation
  section;
- subtract the fitted-cell cascade phase from the physical-supercell phase;
- divide that difference uniformly by the number of cells;
- add the result directly to the `k` array used by `twpasolver`;
- repeat that per-cell correction over a many-centimetre device;
- ignore the measured correction outside the 3–27 GHz HFSS range;
- use renormalized S parameters without an explicit de-embedding or Bloch-wave
  extraction step.

## Nonlinear model

The most complete calculation uses eight forward-propagating modes:

- `p`: pump;
- `s`: signal;
- `i`: idler;
- `p2`: second pump harmonic;
- `ps`: pump-plus-signal product;
- `pi`: pump-plus-idler product;
- `s2`: second signal harmonic;
- `i2`: second idler harmonic.

The nonlinear model currently uses:

- `Istar = 2 mA`;
- signal input current `Is0 = 1.4 µA`;
- zero dielectric and substrate loss in HFSS;
- only forward modes;
- one order of the listed harmonics/conversion products;
- a periodically repeated physical supercell represented through the fitted
  cells plus the HFSS phase correction.

Pump frequency, DC current, pump current, and the number of repeated
supercells were scanned. Sparse signal-frequency scans were used only to
nominate candidates; reported candidates were re-evaluated on a dense 4–8 GHz
grid because coarse sampling can miss narrow peaks and dips.

## Passive HFSS results

The HFSS files pass the numerical checks we performed:

- the fine and auxiliary sweeps agree at 14 GHz;
- reciprocity errors are small;
- passivity excess is numerically very small;
- HFSS validation checks passed for the final geometries.

At a 12.8 GHz pump, the linear three-wave-mixing mismatch calculated as
`k_p - k_s - k_i`, with `f_i = f_p - f_s`, was:

| Pattern | Mean absolute mismatch over 4–8 GHz | Peak-to-peak variation |
|---|---:|---:|
| 13-3-13 | 0.3357 rad/mm | 0.0499 rad/mm |
| 16-2-16 | 0.3481 rad/mm | 0.0675 rad/mm |
| 15-4-15 | 0.5047 rad/mm | 0.1078 rad/mm |

The 15-4-15 structure creates the strongest dispersion feature, but it is not
aligned with the previously assumed 12.8 GHz pump. Within the pump range we
examined, its lowest measured mean absolute linear mismatch occurred at
11.0 GHz and was approximately 0.3010 rad/mm.

The 16-2-16 structure was almost transparent near 25–26 GHz. It therefore did
not create the hoped-for suppression near the second-pump-harmonic region.

## Nonlinear results obtained so far

### 13-3-13

The physical-HFSS-corrected eight-mode model can produce more than enough
gain, but it is highly uneven. Representative results included:

- 81.606 mm line, 12.8 GHz pump, 400 µA DC and 280 µA pump current:
  24.00–41.85 dB gain, or 17.86 dB ripple;
- 60.9 mm line, 12.7 GHz pump, 360 µA DC and 280 µA pump current:
  22.82–42.33 dB gain, or 19.51 dB ripple;
- 150.8 mm lower-drive line, 12.7 GHz pump, 250 µA DC and 250 µA pump
  current: 17.15–38.79 dB gain, or 21.64 dB ripple.

None satisfies the 3 dB ripple requirement.

At the 60.9 mm candidate, the model comparison was:

| Model | Minimum gain | Maximum gain | Ripple |
|---|---:|---:|---:|
| Minimal three-mode 3WM | 15.67 dB | 21.26 dB | 5.59 dB |
| General eight-mode, ideal propagation | 20.49 dB | 43.23 dB | 22.74 dB |
| General eight-mode with HFSS correction | 22.82 dB | 42.33 dB | 19.51 dB |

The large ripple is therefore present even without the physical HFSS phase
correction. We interpreted this as evidence that unwanted nonlinear mode
interactions, rather than passive port reflection alone, are an important
source of the gain structure. That interpretation should be independently
verified.

### 16-2-16

The physical passive HFSS result did not improve the desired phase matching
relative to 13-3-13 and did not suppress the 25–26 GHz region. It was not
selected as an improvement.

### 15-4-15

A pattern-aware scan used the actual 15-4-15 counts, its 68 µm measured
supercell, and its own HFSS-minus-lumped phase correction. It evaluated 468
coarse eight-mode operating points and densely checked the 16 best nominated
settings.

The selected result was:

- 900 supercells;
- physical length: 61.2 mm;
- pump: 11.2 GHz;
- DC current: 350 µA;
- pump current: 290 µA;
- minimum gain over 4–8 GHz: 19.71 dB;
- maximum gain: 33.22 dB;
- ripple: 13.51 dB;
- every densely sampled point above 17 dB;
- target still failed because ripple exceeds 3 dB.

This is less rippled than the earlier physical 13-3-13 complete-model
candidates, but it remains far from the required flatness.

## Current problems and uncertainties

The main observed problem is not insufficient gain. All three physical loading
patterns can support substantial transmission, and the nonlinear calculations
can produce gain above 17 dB. The unresolved problem is very large gain ripple.

Our present interpretation is that energy is exchanged among the desired
signal/idler pair and unwanted `p2`, `ps`, `pi`, `s2`, and `i2` channels,
creating frequency-dependent peaks and dips. However, the magnitude and
irregularity of the calculated ripple could also indicate an error or unstable
assumption in the modelling pipeline.

Specific uncertainties that should be audited include:

- whether the NbTiN sheet location, thickness convention, and impedance
  boundary together double-count or misplace the film;
- whether the impedance boundary covers the entire united conductor and no
  unintended sheets;
- whether using zero sheet resistance and zero dielectric/substrate loss makes
  the model unrealistically resonant;
- whether the wave-port sheets, integration lines, mode count, polarity,
  renormalization, and lack of de-embedding are consistent across every export;
- whether outer stubs or the centre conductor touch the port plane in a way
  that changes the intended reference plane;
- whether the 100 nm dielectric and 500 µm substrate are located on the correct
  sides of the NbTiN sheet and ground;
- whether S21 phase is unwrapped with the correct sign and branch;
- whether S21 phase from a finite periodic network is being confused with a
  Bloch propagation constant;
- whether distributing the physical-supercell correction uniformly per cell
  is justified;
- whether the correction should be per cell, per supercell, or per unit length;
- whether the fitted `L-C-Lf` circuit and extracted 43.1/70.2-ohm impedances
  reproduce the relevant phase and impedance accurately enough;
- whether `Istar = 2 mA`, `Is0 = 1.4 µA`, DC bias, and pump-current conventions
  are physically correct for the assumed NbTiN film and line cross-section;
- whether current amplitudes in `twpasolver` are peak, RMS, or another
  normalization and whether they were interpreted consistently;
- whether the eight-mode configuration contains duplicate, missing, or
  incorrectly coupled processes;
- whether using only forward modes omits reflections/backward waves that could
  materially alter the solution;
- whether the unexpectedly jagged dense gain curves reflect real multi-mode
  energy exchange, insufficient ODE accuracy, sampling artefacts, or numerical
  instability;
- whether the nonlinear solver conserves the appropriate power/Manley–Rowe
  quantities and remains converged at the reported high-gain points;
- whether comparing different supercell patterns through the same isolated
  unit-cell fits is self-consistent;
- whether the reference paper's geometry, material values, bias conditions,
  pump frequency, and impedance definitions have been interpreted correctly.

## Selected-point mode-ablation result

A controlled mode-ablation calculation was performed for the current 15-4-15
candidate at 11.2 GHz, 61.2 mm (900 physical supercells), 350 uA DC bias, and
290 uA pump current. Geometry, HFSS phase correction, currents, signal grid,
and solver settings were held fixed while nonlinear mode families were added
or removed.

With all eight forward modes (`p`, `s`, `i`, `ps`, `pi`, `p2`, `s2`, `i2`),
the calculation reproduced the previous result: 19.714-33.223 dB gain over
4-8 GHz, corresponding to 13.509 dB peak-to-peak ripple.

The complete-model leave-one-family-out results were:

- without `p2`: 24.169-37.612 dB, 13.444 dB ripple;
- without `ps/pi`: -0.234-2.537 dB, 2.772 dB ripple;
- without `s2/i2`: 24.251-32.839 dB, 8.588 dB ripple.

These results suggest that `ps/pi` supplies nearly all of the useful gain and
therefore cannot be interpreted simply as an unwanted ripple source. Removing
`s2/i2` preserves high gain while reducing ripple by about 4.92 dB, making
that family the clearest identified contributor to the remaining ripple.
Removing `p2` changes the peak-to-peak ripple by only about 0.07 dB, although
it changes the absolute gain level. Because the nonlinear interactions are
not additive, this attribution is provisional and should itself be audited
for mode definitions, coupling terms, normalization, and numerical stability.

A subsequent split-mode calculation separated `s2` from `i2`. In the presence
of `p`, `s`, `i`, `ps`, `pi`, and `p2`, adding `s2` changed the gain by as much
as 6.69 dB and lowered the full-model 5.45 GHz dip by 5.41 dB. Adding only
`i2` changed the gain by at most 2.51 dB and lowered that dip by 0.50 dB. The
stronger association is therefore with `s2`, although this is still an
interaction-dependent attribution rather than an additive decomposition.

The closest linear phase matching of the direct `2s -> s2` and cross-coupled
`ps -> i+s2` processes occurs near 5.9 GHz, whereas the calculated full-model
minimum occurs at 5.45 GHz. Pump-induced nonlinear phase shifts and simultaneous
couplings could move the effective condition, but the offset should be audited.
Repeating the calculation with three different returned position resolutions
changed the gain by 0 dB at stored precision. The installed solver already uses
relative and absolute ODE tolerances of 1e-10 and 1e-14, respectively; this
reduces, but does not eliminate, concern about numerical or implementation
artefacts.

Additional 15-4-15 exports labelled `native` were subsequently audited. They
contain no new unrenormalized port information: their S matrices are identical
to the previously used 50-ohm exports (maximum complex-entry difference
`8.69e-15` in the low band and exactly zero in the high band), and their Port
Zo reports are exactly `50 + j0` ohm for both ports at every frequency. The
files also retain copied sweep names containing `16_2_16` despite identifying
the 15-4-15 project/design in their headers. This should be treated as a
provenance and HFSS port-normalization issue to audit, not as evidence that the
physical modal impedance is frequency-independent and exactly 50 ohm.

## Requested review

Please audit the supplied HFSS assumptions, Touchstone data, Python scripts,
equations, unit conventions, and numerical results. Identify any likely or
possible mistakes, unsupported assumptions, internal inconsistencies, or
places where the reported conclusions do not follow from the available data.

Please distinguish clearly between:

1. definite errors;
2. likely errors;
3. modelling approximations that may be acceptable but need justification;
4. results that appear internally consistent.

Do not assume that the present diagnosis of “parasitic nonlinear modes cause
the ripple” is correct. Treat it as a hypothesis to be tested against the code
and data.

## Relevant project material

The principal analysis scripts are in `scripts/`, particularly:

- `fit_hfss_unit_cells.py`;
- `analyze_hfss_supercell.py`;
- `compare_hfss_supercells.py`;
- `validate_hfss_corrected_extended_models.py`;
- `diagnose_refined_candidate_physics.py`;
- `scan_fine_hfss_complete_model.py`;
- `refine_fine_hfss_length_and_bias.py`;
- `scan_low_drive_long_line.py`;
- `scan_physical_hfss_pattern.py`;
- `ablate_15_4_15_modes.py`;
- `diagnose_15_4_15_harmonic_ripple.py`.

The physical Touchstone exports and fitted parameters are in `hfss_inputs/`.
The latest three-pattern passive comparison is in
`results/step_21_hfss_pattern_comparison`, and the physical 15-4-15 nonlinear
result is in `results/step_22_15_4_15_physical_nonlinear_scan`.
The controlled mode-ablation data and figure are in
`results/step_23_15_4_15_mode_ablation`.
The split `s2`/`i2` curves and linear phase-mismatch diagnosis are in
`results/step_24_15_4_15_harmonic_diagnosis`.
