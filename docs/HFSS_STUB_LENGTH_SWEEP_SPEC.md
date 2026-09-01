# HFSS single-cell stub-length sweep: specification

This sweep extends the single-cell model in `HFSS_MODEL_SPEC.md` to locate
50-ohm unloaded and 80-ohm loaded cells. Only `stub_length` varies; geometry,
materials, ports, and solver settings remain fixed.

The settings below describe the source model used for the existing single-cell
exports. The original parametric project is required to repeat that sweep.

## Purpose

The model contains one elementary cell with one stub pair. Two-port extraction
and interpolation determine the lengths at the target impedances. The baseline
single-cell fits use 12.0 and 3.9 µm stubs; the supercell uses 12.1 µm long stubs.

## Geometry and stack (fixed -- do not change)

| Parameter | Value | Role |
|---|---|---|
| `cell_pitch` | 2 um | Unit cell length (2 squares of center line) |
| `line_width` | 1 um | Center conductor width |
| `stub_width` | 1 um | Stub width |
| `stub_side` | 1 | Stubs on both sides of the center line per cell |
| `nbtin_thickness` | 10 nm | NbTiN signal layer thickness |
| `dielectric_thick` | 100 nm | a-Si dielectric thickness |
| `eps_dielectric` | 9.1 | a-Si relative permittivity |
| `tanD_dielectric` | 0 | a-Si loss tangent (lossless model) |
| `Lk_square` | 30 pH | NbTiN kinetic inductance, per square |
| `ground_thickness` | 100 nm | Nb ground plane thickness |
| `ground_width` | 100 um | Nb ground plane width |
| `substrate_thick` | 500 um | Si substrate thickness |
| `substrate_width` | 100 um | Si substrate width |
| `eps_substrate` | 11.7 | Si substrate relative permittivity |
| `tanD_substrate` | 0 | Si substrate loss tangent (lossless model) |
| `air_side` | 50 um | Airbox padding, lateral |
| `air_below` | 50 um | Airbox padding, below substrate |
| `air_above` | 20 um | Airbox padding, above stack |
| `port_width` | 30 um | Wave port width |
| `port_below` | 1 um | Port extension below the line |

## Variable to sweep

| Variable | Existing values already in the project | **New values to add** |
|---|---|---|
| `stub_length` | `12.1um` (unloaded baseline), `3.9um` (loaded baseline) | `7um, 8um, 9um, 10um, 11um` (targeting Z0~50 ohm) and `2.5um, 3um, 3.5um` (targeting Z0~80 ohm) |

Nothing else changes per variation -- same pitch, widths, materials, ports,
mesh settings.

## Boundary conditions

- **NbTiN kinetic-inductance boundary** (`NbTiN_Kinetic`): Impedance
  boundary, `Resistance = 0`, `Reactance = 2*pi*Freq*Lk_square` (a purely
  reactive, frequency-proportional surface impedance -- the standard
  kinetic-inductance surface model). Applied to the NbTiN signal-layer
  objects.
- **Nb ground plane**: modeled as a solid conductor object (not a distinct
  boundary condition entry in the source project).
- **Ports** (`Port1`, `Port2`): Wave Ports, 1 mode each, `RenormImp = 50ohm`,
  characteristic impedance definition `CharImp = 'Zpi'` (power-current).
  Integration line placed at each port's edge center, spanning the
  dielectric thickness (z=0.01 to z=0.11 in the source project's units).

## Solve setup

Single HFSS Driven Modal setup:

- Adaptive mesh refinement frequency: 15 GHz
- `MaxDeltaS = 0.02`, `MaximumPasses = 15`, `MinimumPasses = 2`,
  `MinimumConvergedPasses = 2`, `PercentRefinement = 30`
- `PortAccuracy = 2`, `DrivenSolverType = 'Auto Select Direct/Iterative'`

Two frequency sweeps (both `Discrete`, i.e. solved at every listed
frequency, not interpolated):

1. **Broadband** (-> `kinetic_{stub_length}um_debug.s2p`): `LinearCount`,
   0.2 GHz to 30 GHz, 61 points.
2. **Low-frequency** (-> `kinetic_{stub_length}um_lowfreq.s2p`):
   `LinearCount`, 0.01 GHz to 0.1 GHz, 10 points.

(A third, `Interpolating` 0.2-30 GHz sweep with a 0.5 tolerance also exists
in the source project as `Sweep_test`, but it is disabled -- not required.)

## Export

For each `stub_length` variation, export both sweeps' S-parameters as
Touchstone (.s2p, 2-port, `# GHz S MA R 50` or `RI` format), named:

```
hfss_inputs/kinetic_{stub_length}um_debug.s2p
hfss_inputs/kinetic_{stub_length}um_lowfreq.s2p
```

e.g. `kinetic_8um_debug.s2p`, `kinetic_2p5um_lowfreq.s2p` (use `p` for the
decimal point, matching the existing `kinetic_3p9um_*` files).

## Downstream use

`scripts/fit_stub_length_sweep.py` uses the two-port extraction from
`scripts/fit_hfss_unit_cells.py`: a 4–14 GHz ABCD fit and the low-frequency
inductance limit. The one-port open/short method is excluded because its
capacitance estimates disagreed with the two-port fits.
