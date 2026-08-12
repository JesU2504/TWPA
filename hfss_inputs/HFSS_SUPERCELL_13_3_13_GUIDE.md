# HFSS guide: 13-3-13 physical supercell

## Purpose

This simulation is the remaining validation required before replacing the
current `15-4-15` amplifier configuration with the `13-3-13` candidate.
Unlike cascading separately fitted cells, the physical supercell includes the
interfaces between 12 um and 3.9 um stubs directly.

## Fixed dimensions

| Item | Value |
|---|---:|
| Cell pitch | 2 um |
| Cells per supercell | 29 |
| Supercell length | 58 um |
| Center-line width | 1 um |
| Transverse-stub width | 1 um |
| Long stub arm | 12 um per side |
| Long cross total Y span | 25 um |
| Short stub arm | 3.9 um per side |
| Short cross total Y span | 8.8 um |
| NbTiN sheet plane | z = 0.01 um |
| Port width | 30 um |
| Port vertical span | z = -1 to +0.2 um |

All dielectric, substrate, ground-plane, airbox, material, and impedance
properties must remain identical to the converged single-cell kinetic design.

## 1. Duplicate the validated design

1. Duplicate `HFSSDesign_Kinetic_debug`.
2. Rename the copy `HFSSDesign_Supercell_13_3_13`.
3. Keep the validated setup as a reference, but delete or suppress the old
   one-cell NbTiN sheet and its two port sheets before creating the new objects.
4. Do not change `Lk_square = 30 pH` or the impedance expression
   `2*pi*Freq*Lk_square`.

## 2. Create the NbTiN sheet in the XY plane

Use rectangles at `z = 0.01 um`.

### Continuous center line

- Position: `(-29 um, -0.5 um, 0.01 um)`
- X size: `58 um`
- Y size: `1 um`

### Long transverse stubs

For each center coordinate below, create a rectangle with:

- Position: `(xc - 0.5 um, -12.5 um, 0.01 um)`
- X size: `1 um`
- Y size: `25 um`

Long-stub center coordinates:

```text
-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4,
  4,   6,   8,  10,  12,  14,  16,  18,  20, 22, 24, 26, 28  um
```

### Short transverse stubs

At `xc = -2, 0, +2 um`, create rectangles with:

- Position: `(xc - 0.5 um, -4.4 um, 0.01 um)`
- X size: `1 um`
- Y size: `8.8 um`

### Finish the conductor

1. Select the center line and all 29 transverse rectangles.
2. Use **Modeler > Boolean > Unite**.
3. Rename the united sheet `NbTiN_supercell_13_3_13`.
4. Assign the existing `NbTiN_Kinetic` impedance boundary to the united sheet:
   resistance `0`, reactance `2*pi*Freq*Lk_square` ohm/square.
5. Confirm the old `NbTiN_sheet` is suppressed or has no boundary assignment.

## 3. Extend the stack and airbox

Extend the existing substrate, a-Si dielectric, Nb ground, and airbox to cover
`x = -29 to +29 um`. Preserve their existing Y and Z dimensions and materials.
The NbTiN sheet must remain exactly on the dielectric surface without a vacuum
gap or overlapping 10 nm conductor solid.

## 4. Recreate the wave ports

Create one YZ-plane sheet at each longitudinal end:

- Port 1: `x = -29 um`
- Port 2: `x = +29 um`
- Y span: `-15 to +15 um`
- Z span: `-1 to +0.2 um`

For both ports:

1. Assign a wave port with **one mode**.
2. Keep 50 ohm renormalization.
3. Draw the integration line from the NbTiN center conductor toward the Nb
   ground plane.
4. Make sure each port touches the end of the NbTiN center strip and intersects
   the dielectric/ground stack exactly as in the converged unit-cell design.

## 5. Solve in two stages

### Stage A: geometry and convergence check

1. Validate the design.
2. Use a single-frequency adaptive solution at 15 GHz.
3. Use maximum Delta S = 0.02, at least 2 consecutive passes, and maximum 15
   passes.
4. Do not proceed unless the adaptive solution converges.

### Stage B: discrete network sweep

Start with a manageable coarse discrete sweep:

- Start: 0.2 GHz
- Stop: 30 GHz
- Step: 0.5 GHz (61 points)
- Sweep type: **Discrete**
- Save fields: off

After inspecting the coarse response, add a second discrete sweep from
10 to 14 GHz with 0.05 GHz spacing to resolve the first loading stopband and
the pump region.

## 6. Plot and export

Plot these traces:

- `dB(S(Port1,Port1))`
- `dB(S(Port2,Port1))`
- `ang_deg(S(Port2,Port1))`

Check for passivity (`S21` must not exceed 0 dB), reciprocity, smooth phase, and
the expected stopband near the proposed 13 GHz pump region.

Export an S-matrix Touchstone file with 50 ohm renormalization as:

```text
supercell_13_3_13.s2p
```

Copy it into `/Users/lgsus/Desktop/TWPA/hfss_inputs/`. The next Python step is
to extract the supercell ABCD/Bloch response and rerun the reflection-aware gain
verification.
