# References

1. L. Howe, A. Giachero, M. Vissers, P. Campana, J. Wheeler, J. Gao,
   J. Austermann, J. Hubmayr, A. Nucciotti, and J. Ullom (2025).
   *Kinetic Inductance Traveling Wave Parametric Amplifiers Near the Quantum Limit:
   Methodology and Characterization.*
   [arXiv:2507.07706v1](https://arxiv.org/abs/2507.07706v1).
   Source of the impedance and gain comparison curves and the reflection model.
   The arXiv record links to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

2. Boon-Kok Tan, Faouzi Boussaha, Christine Chaumont, Joseph Longden, and
   Javier Navarro Montilla (2023).
   *Engineering the thin film characteristics for optimal performance of
   superconducting kinetic inductance amplifiers using a rigorous modelling technique.*
   Open Research Europe **2**, 88, version 2.
   [DOI: 10.12688/openreseurope.14860.2](https://doi.org/10.12688/openreseurope.14860.2).
   Thin-film properties and electromagnetic modelling of kinetic-inductance amplifiers.
   The article carries a Creative Commons Attribution license notice.

3. Joseph Christopher Longden (2023).
   *Development of Superconducting Thin Film Travelling Wave Parametric Amplifiers.*
   DPhil thesis, University of Oxford.
   [DOI: 10.5287/ora-kz56ryqy8](https://doi.org/10.5287/ora-kz56ryqy8).
   Device design, modelling, and experimental background.

4. A. R. Kerr (1999, revised August 9).
   *Surface Impedance of Superconductors and Normal Conductors in EM Simulators.*
   NRAO, MMA Memo No. 245.
   [NRAO archive](https://library.nrao.edu/public/memos/alma/memo245.pdf).
   Surface-impedance boundary conditions for electromagnetic simulation.

## Digitized comparison data

The following files contain numerical traces digitized from Howe et al., version 1:

| File in `data/reference/` | Source | Processing |
|---|---|---|
| `paper_impedance_fig2.csv` | Fig. 2, page 4 | Pixel selection and axis calibration from a 400 dpi crop |
| `paper_simulation_fig3d.csv` | Fig. 3(d), page 5 | Red maximum-GBP vector, calibrated to GHz and dB, restricted to 2–11 GHz |
| `paper_experiment_fig4a.csv` | Fig. 4(a), page 6 | Magenta vector calibrated to GHz and dB |

These are digitized published curves, not author-provided raw measurements.
Credit for the source figures belongs to Howe et al.; the traces are supplied with
the source's [CC BY 4.0 attribution](https://creativecommons.org/licenses/by/4.0/).
The transformations are recorded in the table above. Experimental and simulated
curves represent different operating cases.

The calculation pipeline copies these CSVs into `results/` and does not require
paper PDFs or figure images. Full-text papers and standalone figure reproductions
are not bundled. Consult each source's terms before reusing other material.
