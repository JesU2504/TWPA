#!/usr/bin/env python3
"""Two-port HFSS stub-length fits and 50/80-ohm impedance crossings.

The extraction uses the LCLfBaseCell model over 4–14 GHz and the low-frequency
inductance limit. The 3.5 µm point is excluded because its broadband export
duplicated the 3.0 µm variation. Outputs are stored in result step 39."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code"))

from fit_hfss_unit_cells import fit_cell  # noqa: E402
from io_utils import write_csv_rows  # noqa: E402
from plotting import save_figure  # noqa: E402

INPUTS = ROOT / "hfss_inputs"
OUTPUT = ROOT / "results" / "bloch_corrected" / "step_39_stub_length_50_80ohm_sweep"
PAPER_DIGITIZED_CURVE = (
    ROOT
    / "results"
    / "bloch_corrected"
    / "step_32_fig2_z0_comparison"
    / "01b_paper_digitized_Z0_curve.csv"
)

# Exclude the 3.5 µm export: it duplicated the 3.0 µm variation.
CELL_DEFINITIONS: dict[str, dict[str, float | str]] = {
    "2.5um": {
        "stub_length_um": 2.5,
        "broadband_file": "kinetic_2p5um_debug.s2p",
        "lowfreq_file": "kinetic_2p5um_lowfreq.s2p",
    },
    "3um": {
        "stub_length_um": 3.0,
        "broadband_file": "kinetic_3um_debug.s2p",
        "lowfreq_file": "kinetic_3um_lowfreq.s2p",
    },
    "3.9um_baseline": {
        "stub_length_um": 3.9,
        "broadband_file": "kinetic_3p9um_debug.s2p",
        "lowfreq_file": "kinetic_3p9um_lowfreq.s2p",
    },
    "7um": {
        "stub_length_um": 7.0,
        "broadband_file": "kinetic_7um_debug.s2p",
        "lowfreq_file": "kinetic_7um_lowfreq.s2p",
    },
    "8um": {
        "stub_length_um": 8.0,
        "broadband_file": "kinetic_8um_debug.s2p",
        "lowfreq_file": "kinetic_8um_lowfreq.s2p",
    },
    "9um": {
        "stub_length_um": 9.0,
        "broadband_file": "kinetic_9um_debug.s2p",
        "lowfreq_file": "kinetic_9um_lowfreq.s2p",
    },
    "10um": {
        "stub_length_um": 10.0,
        "broadband_file": "kinetic_10um_debug.s2p",
        "lowfreq_file": "kinetic_10um_lowfreq.s2p",
    },
    "11um": {
        "stub_length_um": 11.0,
        "broadband_file": "kinetic_11um_debug.s2p",
        "lowfreq_file": "kinetic_11um_lowfreq.s2p",
    },
    "12um_baseline": {
        "stub_length_um": 12.0,
        "broadband_file": "kinetic_12um_debug.s2p",
        "lowfreq_file": "kinetic_12um_lowfreq.s2p",
    },
}


def find_crossing_length(
    lengths_um: np.ndarray, z0_ohm: np.ndarray, target_ohm: float
) -> float | None:
    """Interpolated length at the target impedance, or None outside the measured range.

    Lengths must increase and impedance must decrease across the input arrays."""
    if not (z0_ohm.min() <= target_ohm <= z0_ohm.max()):
        return None
    # np.interp requires an increasing impedance axis.
    return float(np.interp(target_ohm, z0_ohm[::-1], lengths_um[::-1]))


def load_paper_digitized_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Length and impedance arrays from the digitized Fig. 2 CSV."""
    rows = list(csv.DictReader(open(path)))
    lengths_um = np.array([float(row["finger_length_um"]) for row in rows])
    z0_ohm = np.array([float(row["Z0_ohm"]) for row in rows])
    return lengths_um, z0_ohm


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fitted_rows = []
    for label, definition in CELL_DEFINITIONS.items():
        fit = fit_cell(INPUTS, definition)
        fitted_rows.append({"label": label, **fit})
        print(
            f"{label:16s} l={fit['stub_length_um']:5.2f} um  L={fit['L_pH']:6.2f} pH  C={fit['C_fF']:6.2f} fF  Z0={fit['Z0_ohm']:6.2f} ohm  "
            f"(S21 rmse={fit['fit_s21_rmse_db']:.4f} dB, phase rmse={fit['fit_phase_rmse_deg']:.3f} deg)"
        )

    fitted_rows.sort(key=lambda row: row["stub_length_um"])
    lengths_um = np.array([row["stub_length_um"] for row in fitted_rows])
    inductance_ph = np.array([row["L_pH"] for row in fitted_rows])
    capacitance_ff = np.array([row["C_fF"] for row in fitted_rows])
    z0_ohm = np.array([row["Z0_ohm"] for row in fitted_rows])

    write_csv_rows(
        OUTPUT / "01_fitted_lengths.csv",
        [
            "label",
            "stub_length_um",
            "L_pH",
            "C_fF",
            "Z0_ohm",
            "fit_s21_rmse_db",
            "fit_phase_rmse_deg",
        ],
        [
            [row["label"] for row in fitted_rows],
            lengths_um,
            inductance_ph,
            capacitance_ff,
            z0_ohm,
            [row["fit_s21_rmse_db"] for row in fitted_rows],
            [row["fit_phase_rmse_deg"] for row in fitted_rows],
        ],
    )

    length_for_50ohm = find_crossing_length(lengths_um, z0_ohm, 50.0)
    length_for_80ohm = find_crossing_length(lengths_um, z0_ohm, 80.0)

    paper_lengths_um, paper_z0_ohm = load_paper_digitized_curve(PAPER_DIGITIZED_CURVE)
    paper_length_for_50ohm = find_crossing_length(paper_lengths_um, paper_z0_ohm, 50.0)
    paper_length_for_80ohm = find_crossing_length(paper_lengths_um, paper_z0_ohm, 80.0)

    fig, (ax_z0, ax_lc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_z0.plot(
        paper_lengths_um,
        paper_z0_ohm,
        "-",
        color="#2a78d6",
        lw=1.5,
        alpha=0.7,
        label="Paper (digitized Fig. 2)",
    )
    ax_z0.plot(
        lengths_um,
        z0_ohm,
        "o-",
        color="tab:blue",
        label="This work (fitted from HFSS, 2-port ABCD)",
    )
    ax_z0.axhline(50, color="gray", ls=":", lw=1)
    ax_z0.axhline(80, color="gray", ls=":", lw=1)
    if length_for_50ohm is not None:
        ax_z0.axvline(
            length_for_50ohm,
            color="green",
            ls="--",
            lw=1,
            label=f"Our Z0=50 ohm at l={length_for_50ohm:.2f} um",
        )
    if length_for_80ohm is not None:
        ax_z0.axvline(
            length_for_80ohm,
            color="purple",
            ls="--",
            lw=1,
            label=f"Our Z0=80 ohm at l={length_for_80ohm:.2f} um",
        )
    if paper_length_for_50ohm is not None:
        ax_z0.axvline(
            paper_length_for_50ohm,
            color="green",
            ls=":",
            lw=1,
            alpha=0.6,
            label=f"Paper's Z0=50 ohm at l={paper_length_for_50ohm:.2f} um",
        )
    if paper_length_for_80ohm is not None:
        ax_z0.axvline(
            paper_length_for_80ohm,
            color="purple",
            ls=":",
            lw=1,
            alpha=0.6,
            label=f"Paper's Z0=80 ohm at l={paper_length_for_80ohm:.2f} um",
        )
    ax_z0.set_xlabel("Finger length [um]")
    ax_z0.set_ylabel("Z0 [ohm]")
    ax_z0.set_xlim(0, 13)
    ax_z0.set_title("Our HFSS stack vs. the paper's own curve")
    ax_z0.legend(fontsize=7.5)
    ax_z0.grid(alpha=0.3)

    ax_lc.plot(lengths_um, inductance_ph, "o-", color="tab:orange", label="L [pH/cell]")
    ax_lc_twin = ax_lc.twinx()
    ax_lc_twin.plot(lengths_um, capacitance_ff, "s-", color="tab:red", label="C [fF/cell]")
    ax_lc.set_xlabel("Finger length [um]")
    ax_lc.set_ylabel("L [pH/cell]", color="tab:orange")
    ax_lc_twin.set_ylabel("C [fF/cell]", color="tab:red")
    ax_lc.set_title("L is ~constant; C rises with length")
    ax_lc.grid(alpha=0.3)

    fig.suptitle("Stub-length sweep: locating Z0=50/80 ohm from HFSS-simulated data")
    fig.tight_layout()
    save_figure(fig, OUTPUT, "02_z0_vs_length")
    plt.close(fig)

    fig_slide, ax_slide = plt.subplots(figsize=(9.0, 5.2))
    ax_slide.plot(
        paper_lengths_um,
        paper_z0_ohm,
        "-",
        color="#2a78d6",
        lw=2.6,
        alpha=0.78,
        label="Paper (digitized Fig. 2)",
    )
    ax_slide.plot(
        lengths_um,
        z0_ohm,
        "o-",
        color="#eb6834",
        lw=2.6,
        ms=7,
        label="This work (HFSS-derived, 2-port fit)",
    )
    for length_um, impedance_ohm, label in (
        (3.9, float(z0_ohm[np.argmin(np.abs(lengths_um - 3.9))]), "short / loaded"),
        (12.0, float(z0_ohm[np.argmin(np.abs(lengths_um - 12.0))]), "long / unloaded"),
    ):
        ax_slide.scatter(
            [length_um],
            [impedance_ohm],
            s=100,
            color="#eb6834",
            edgecolor="white",
            linewidth=1.3,
            zorder=5,
        )
        offset = (12, 12) if length_um < 6 else (-170, 14)
        ax_slide.annotate(
            f"{label}\n{length_um:.1f} µm → {impedance_ohm:.1f} Ω",
            xy=(length_um, impedance_ohm),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color="#343a40",
            arrowprops={"arrowstyle": "-", "color": "#8a8f98", "lw": 1.0},
        )
    ax_slide.set_xlabel("Stub length (µm)")
    ax_slide.set_ylabel("Characteristic impedance, Z₀ (Ω)")
    ax_slide.set_xlim(0, 13)
    ax_slide.set_ylim(35, 125)
    ax_slide.grid(alpha=0.24)
    ax_slide.legend(loc="upper right", frameon=False, fontsize=10)
    fig_slide.tight_layout()
    save_figure(fig_slide, OUTPUT, "02a_z0_vs_length_slide")
    plt.close(fig_slide)

    provenance = {
        "purpose": "Locate finger lengths giving Z0=50/80 ohm for this project's own HFSS stack",
        "method": "fit_hfss_unit_cells.fit_cell() (2-port ABCD fit, same as the main pipeline validated in step_38)",
        "excluded_point": {
            "stub_length_um": 3.5,
            "reason": "kinetic_3p5um_debug.s2p was found byte-identical to kinetic_3um_debug.s2p -- a mis-exported HFSS variation, not real 3.5um data. Excluded until re-exported.",
        },
        "inductance_range_pH": [float(inductance_ph.min()), float(inductance_ph.max())],
        "inductance_approximately_constant": bool(
            np.ptp(inductance_ph) < 0.05 * np.mean(inductance_ph)
        ),
        "length_for_Z0_50ohm_um": length_for_50ohm,
        "length_for_Z0_80ohm_um": length_for_80ohm,
        "paper_digitized_curve_source": str(PAPER_DIGITIZED_CURVE.relative_to(ROOT)),
        "paper_length_for_Z0_50ohm_um": paper_length_for_50ohm,
        "paper_length_for_Z0_80ohm_um": paper_length_for_80ohm,
        "prior_rough_estimate_um": {
            "Z0_50ohm": [8.38, 9.45],
            "Z0_80ohm": [2.97, 3.39],
            "note": "From linearly extrapolating the 7-point 01_our_data.csv fit before this HFSS sweep existed",
        },
    }
    (OUTPUT / "03_provenance.json").write_text(json.dumps(provenance, indent=2))

    print()
    print(
        f"L is approximately constant: {inductance_ph.min():.2f}-{inductance_ph.max():.2f} pH across all lengths"
    )
    if length_for_50ohm is not None:
        print(f"Z0 = 50 ohm at finger length = {length_for_50ohm:.2f} um")
    else:
        print("Z0 = 50 ohm is NOT bracketed by the simulated data")
    if length_for_80ohm is not None:
        print(f"Z0 = 80 ohm at finger length = {length_for_80ohm:.2f} um")
    else:
        print("Z0 = 80 ohm is NOT bracketed by the simulated data")
    print(
        f"Paper's own curve: Z0=50 ohm at l={paper_length_for_50ohm:.2f} um, Z0=80 ohm at l={paper_length_for_80ohm:.2f} um"
    )
    print(f"Wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
