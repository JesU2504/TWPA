# Preliminary periodic-eigenmode verification

This folder compares a periodic HFSS eigenmode calculation of the 68 µm
16-2-16 supercell with the independently extracted ABCD Bloch dispersion.

The HFSS model used separate matching lattice pairs on the vacuum, silicon,
and a-Si end faces. The NbTiN sheet retained its kinetic-impedance boundary.
Because HFSS Eigenmode does not accept the original frequency-dependent sheet
expression, its reactance was evaluated at `KineticFreq`; that value was
manually iterated toward the resulting eigenfrequency.

The 90° and 120° points reproduce the increasing dispersion trend and lie
about 4–5% below the ABCD prediction. They are provisional because their final
adaptive frequency changes were 1.0309% and 1.2426%, above the requested 0.5%
criterion. The 75° and 150° trials suffered severe mode switching and are
shown only as instability evidence.

**Conclusion:** this is preliminary qualitative support for the ABCD Bloch
extraction, not quantitative closure and not experimental validation. Mesh
convergence and robust eigenmode tracking remain deferred.

The source AEDT project is kept locally as
`hfss_inputs/16-2-16_periodic_eigenmode_verification.aedt`; AEDT files are
excluded from Git because they are binary simulation artifacts.
