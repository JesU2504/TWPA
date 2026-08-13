#!/usr/bin/env python3
"""Report the preliminary periodic-eigenmode check of the 16-2-16 supercell."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "scripts")]

from analyze_hfss_supercell import read_touchstone  # noqa: E402
from hfss_bloch import extract_bloch_parameters  # noqa: E402


MEASUREMENTS = [
    # phase, kinetic input, eigenfrequency, final mesh delta, classification
    (75.0, 4.84460, 4.81245, 263.2200, "mode_tracking_unstable"),
    (90.0, 5.76610, 5.77295, 1.0309, "provisional_usable"),
    (120.0, 7.70000, 7.65937, 1.2426, "provisional_usable"),
    (150.0, 9.64500, 9.62298, 36.7710, "mode_tracking_unstable"),
]


def main() -> None:
    output = (
        ROOT
        / "results"
        / "bloch_corrected"
        / "step_26_periodic_eigenmode_verification"
    )
    output.mkdir(parents=True, exist_ok=True)

    frequencies, s = read_touchstone(
        ROOT / "hfss_inputs" / "supercell_16_2_16_fine_3_14GHz.s2p"
    )
    phases = np.rad2deg(extract_bloch_parameters(s).phase)

    rows: list[dict[str, float | str]] = []
    for phase, kinetic, eigenfrequency, mesh_delta, status in MEASUREMENTS:
        abcd_frequency = float(np.interp(phase, phases, frequencies))
        rows.append(
            {
                "bloch_phase_deg": phase,
                "kinetic_input_GHz": kinetic,
                "eigenfrequency_GHz": eigenfrequency,
                "abcd_frequency_GHz": abcd_frequency,
                "eigenmode_minus_abcd_percent": 100
                * (eigenfrequency - abcd_frequency)
                / abcd_frequency,
                "max_delta_freq_percent": mesh_delta,
                "status": status,
            }
        )

    with (output / "01_eigenmode_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    usable = [row for row in rows if row["status"] == "provisional_usable"]
    mean_offset = float(
        np.mean([float(row["eigenmode_minus_abcd_percent"]) for row in usable])
    )
    assessment = {
        "status": "preliminary_qualitative_support",
        "geometry": "16-2-16",
        "supercell_length_um": 68.0,
        "periodic_boundary": "partitioned matching end-face lattice pairs",
        "kinetic_boundary_workaround": (
            "The frequency-dependent sheet reactance was evaluated at a discrete "
            "KineticFreq and iterated toward the eigenfrequency."
        ),
        "usable_phases_deg": [90.0, 120.0],
        "unstable_phases_deg": [75.0, 150.0],
        "mean_usable_eigenmode_minus_abcd_percent": mean_offset,
        "conclusion": (
            "The independent periodic eigenmode calculation follows the ABCD Bloch "
            "dispersion trend, with the two provisionally usable points about 4--5% "
            "below the ABCD prediction."
        ),
        "limitation": (
            "No point met the requested 0.5% adaptive convergence criterion; 75 and "
            "150 degrees also showed severe mode-tracking instability. This is not a "
            "quantitative validation or experimental validation."
        ),
    }
    (output / "02_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    mask = (phases >= 60) & (phases <= 165)
    order = np.argsort(phases[mask])
    ax.plot(
        phases[mask][order],
        frequencies[mask][order],
        color="#1f77b4",
        linewidth=2,
        label="ABCD Bloch extraction",
    )
    for status, marker, color, label in (
        ("provisional_usable", "o", "#2ca02c", "Eigenmode (provisional)"),
        ("mode_tracking_unstable", "x", "#d62728", "Eigenmode (unstable)"),
    ):
        selected = [row for row in rows if row["status"] == status]
        ax.scatter(
            [float(row["bloch_phase_deg"]) for row in selected],
            [float(row["eigenfrequency_GHz"]) for row in selected],
            marker=marker,
            color=color,
            s=70,
            linewidths=2,
            label=label,
            zorder=3,
        )
    ax.set_xlabel("Bloch phase per 68 µm supercell (degrees)")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title("16-2-16 periodic eigenmode cross-check")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output / "03_eigenmode_comparison.png", dpi=200)
    fig.savefig(output / "03_eigenmode_comparison.pdf")
    plt.close(fig)

    readme = """# Preliminary periodic-eigenmode verification

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
"""
    (output / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
