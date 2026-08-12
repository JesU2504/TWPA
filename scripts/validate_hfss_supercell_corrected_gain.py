#!/usr/bin/env python3
"""Provisional minimal-3WM gain validation using HFSS supercell dispersion."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "scripts")]

from analyze_hfss_supercell import (  # noqa: E402
    SUPERCELL_CELLS,
    cascade_pattern,
    phase_delay,
    read_touchstone,
)
from fit_hfss_unit_cells import lclf_s  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402


PUMP_CANDIDATE_GHZ = 13.13
IDC_A = 350e-6
IPUMP_A = 340e-6


def qualified(metrics: dict[str, float]) -> bool:
    return (
        metrics["mean_in_band_gain_db"] >= 15.0
        and metrics["fraction_above_17db"] >= 0.8
        and metrics["ripple_peak_to_peak_db"] <= 3.0
    )


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "step_13_hfss_supercell_validation"
    output.mkdir(parents=True, exist_ok=True)

    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    measured_f, measured_s = read_touchstone(inputs / "supercell_13_3_13_coarse.s2p")

    def fitted_cell(role: str) -> np.ndarray:
        row = cells[role]
        return lclf_s(
            measured_f * 1e9,
            row["L_pH"] * 1e-12,
            row["C_fF"] * 1e-15,
            row["Lf_pH"] * 1e-12,
        )

    lumped_supercell = cascade_pattern(fitted_cell("unloaded"), fitted_cell("loaded"))
    measured_phase, _ = phase_delay(measured_s[:, 1, 0], measured_f)
    lumped_phase, _ = phase_delay(lumped_supercell[:, 1, 0], measured_f)
    # TWPAnalysis k is positive propagation phase per physical 2 um cell.
    k_correction = ((-measured_phase) - (-lumped_phase)) / SUPERCELL_CELLS

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 13),
            make_cell(cells["loaded"], 3),
            make_cell(cells["unloaded"], 13),
        ],
        N=1407,
        Istar=2e-3,
        Idc=IDC_A,
        Ip0=IPUMP_A,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 20.0, 0.002), unit="GHz")
    analysis.update_base_data()
    signals = np.arange(4.0, 8.0001, 0.1)

    baseline_gain = np.asarray(
        analysis.gain(
            signals,
            pump=PUMP_CANDIDATE_GHZ,
            Is0=1.4e-6,
            Ip0=None,
            model="minimal_3wm",
            mode_array_config="basic_3wm",
            thin=300,
            save=False,
        )["gain_db"]
    )
    baseline_metrics = gain_metrics(signals, baseline_gain)

    dense_f = np.asarray(analysis.data["freqs"])
    correction_dense = np.zeros_like(dense_f)
    valid = (dense_f >= measured_f[0]) & (dense_f <= measured_f[-1])
    correction_dense[valid] = np.interp(
        dense_f[valid], measured_f, k_correction
    )
    analysis.data["k"] = np.asarray(analysis.data["k"]) + correction_dense

    corrected_candidate_gain = np.asarray(
        analysis.gain(
            signals,
            pump=PUMP_CANDIDATE_GHZ,
            Is0=1.4e-6,
            Ip0=None,
            model="minimal_3wm",
            mode_array_config="basic_3wm",
            thin=300,
            save=False,
        )["gain_db"]
    )
    corrected_candidate_metrics = gain_metrics(signals, corrected_candidate_gain)

    rows = []
    curves: dict[float, np.ndarray] = {}
    for pump in np.arange(13.05, 13.2501, 0.01):
        gain = np.asarray(
            analysis.gain(
                signals,
                pump=float(pump),
                Is0=1.4e-6,
                Ip0=None,
                model="minimal_3wm",
                mode_array_config="basic_3wm",
                thin=300,
                save=False,
            )["gain_db"]
        )
        metrics = gain_metrics(signals, gain)
        row = {
            "pump_GHz": float(pump),
            **metrics,
            "qualified": qualified(metrics),
        }
        rows.append(row)
        curves[float(pump)] = gain

    best = max(
        rows,
        key=lambda row: (
            float(row["qualified"]),
            row["fraction_above_17db"],
            -max(0.0, row["ripple_peak_to_peak_db"] - 3.0),
            -abs(row["mean_in_band_gain_db"] - 20.0),
        ),
    )
    best_gain = curves[best["pump_GHz"]]

    with (output / "05_corrected_pump_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "05_corrected_gain_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "signal_GHz",
                "uncorrected_13p13GHz_gain_dB",
                "HFSS_corrected_13p13GHz_gain_dB",
                "HFSS_corrected_best_pump_gain_dB",
            ]
        )
        writer.writerows(
            zip(signals, baseline_gain, corrected_candidate_gain, best_gain)
        )

    report = {
        "status": "provisional_from_0p5GHz_spaced_HFSS_phase_data",
        "fixed_bias": {"Idc_A": IDC_A, "Ip0_A": IPUMP_A},
        "candidate_pump_GHz": PUMP_CANDIDATE_GHZ,
        "uncorrected_candidate": baseline_metrics,
        "HFSS_corrected_candidate": corrected_candidate_metrics,
        "HFSS_corrected_pump_scan": {
            "range_GHz": [13.05, 13.25],
            "step_GHz": 0.01,
            "qualified_count": sum(bool(row["qualified"]) for row in rows),
            "selected": best,
        },
        "interpretation": (
            "The physical-supercell phase correction changes the predicted gain "
            "and removes the previous <=3 dB-ripple qualification. A finer HFSS "
            "dispersion sweep is required before a final current/pump optimization."
        ),
        "hfss_cell_provenance": provenance,
    }
    (output / "05_corrected_gain_assessment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(signals, baseline_gain, label="lumped model, pump 13.13 GHz")
    ax.plot(signals, corrected_candidate_gain, label="HFSS-corrected, pump 13.13 GHz")
    ax.plot(
        signals,
        best_gain,
        "--",
        label=f"HFSS-corrected scan choice, pump {best['pump_GHz']:.2f} GHz",
    )
    ax.axhline(17, color="k", linestyle=":", label="17 dB target")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Minimal-3WM gain (dB)",
        title="Provisional gain impact of physical-supercell dispersion",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "05_corrected_gain_validation.png", dpi=180)
    fig.savefig(output / "05_corrected_gain_validation.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
