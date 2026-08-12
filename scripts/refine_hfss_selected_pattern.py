#!/usr/bin/env python3
"""Choose a conservative operating point for the selected 13/3/13 pattern."""

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
sys.path.insert(0, str(ROOT / "code"))

from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402


def qualified(row: dict[str, float]) -> bool:
    return (
        row["mean_in_band_gain_db"] >= 15.0
        and row["fraction_above_17db"] >= 0.8
        and row["ripple_peak_to_peak_db"] <= 3.0
    )


def score(row: dict[str, float]) -> tuple[float, ...]:
    # Once the specifications pass, prefer a moderate 20 dB mean over extreme
    # undepleted-pump gain, then prefer lower ripple and lower total current.
    return (
        float(qualified(row)),
        -abs(row["mean_in_band_gain_db"] - 20.0),
        -row["ripple_peak_to_peak_db"],
        -(row["Idc_A"] + row["Ip0_A"]),
    )


def main() -> None:
    cells, provenance = load_hfss_cell_parameters(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv"
    )
    side, loaded, repeats = 13, 3, 1407
    signals = np.arange(4.0, 8.0001, 0.05)
    pumps = np.arange(12.85, 13.2001, 0.01)
    dc_biases = np.arange(200, 350.1, 25.0) * 1e-6
    pump_currents = np.arange(180, 340.1, 20.0) * 1e-6
    rows: list[dict[str, float | bool]] = []
    best_row = None
    best_gain = None

    for dc_bias in dc_biases:
        for pump_current in pump_currents:
            twpa = TWPA(
                cells=[
                    make_cell(cells["unloaded"], side),
                    make_cell(cells["loaded"], loaded),
                    make_cell(cells["unloaded"], side),
                ],
                N=repeats,
                Istar=2e-3,
                Idc=float(dc_bias),
                Ip0=float(pump_current),
            )
            analysis = TWPAnalysis(
                twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz"
            )
            for pump in pumps:
                try:
                    result = analysis.gain(
                        signals,
                        pump=float(pump),
                        Is0=1.4e-6,
                        Ip0=None,
                        model="minimal_3wm",
                        mode_array_config="basic_3wm",
                        thin=300,
                        save=False,
                    )
                    gain = np.asarray(result["gain_db"], dtype=float)
                except Exception:
                    continue
                if not np.all(np.isfinite(gain)):
                    continue
                row = {
                    "unloaded_cells_each_side": side,
                    "loaded_cells": loaded,
                    "supercells": repeats,
                    "pump_GHz": float(pump),
                    "Idc_A": float(dc_bias),
                    "Ip0_A": float(pump_current),
                    **gain_metrics(signals, gain, band=(4.0, 8.0)),
                }
                rows.append(row)
                if best_row is None or score(row) > score(best_row):
                    best_row = row
                    best_gain = gain.copy()

    if best_row is None or best_gain is None:
        raise RuntimeError("No finite operating point")

    output = ROOT / "results" / "step_12_hfss_operating_point_optimization"
    fieldnames = list(rows[0])
    with (output / "selected_pattern_operating_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passing = [row for row in rows if qualified(row)]
    report = {
        "selection_policy": (
            "Meet mean>=15 dB, >=80% of 4-8 GHz above 17 dB, and ripple<=3 dB; "
            "then select mean gain nearest 20 dB, lower ripple, and lower total current."
        ),
        "evaluated_points": len(rows),
        "qualified_count": len(passing),
        "selected": best_row,
        "top_qualified": sorted(passing, key=score, reverse=True)[:20],
        "hfss_provenance": provenance,
    }
    (output / "selected_pattern_operating_scan.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output / "selected_operating_gain.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_GHz", "gain_db"])
        writer.writerows(zip(signals, best_gain))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(signals, best_gain, linewidth=2, label="Predicted gain")
    ax.axhline(17, color="C3", linestyle="--", label="17 dB coverage target")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Gain (dB)",
        title=(
            f"Selected 13/3/13 point: pump={best_row['pump_GHz']:.2f} GHz, "
            f"Idc={best_row['Idc_A']*1e6:.0f} µA, Ip={best_row['Ip0_A']*1e6:.0f} µA"
        ),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "selected_operating_gain.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"selected": best_row, "qualified_count": len(passing)}, indent=2))


if __name__ == "__main__":
    main()
