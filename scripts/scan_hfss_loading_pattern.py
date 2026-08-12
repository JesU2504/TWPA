#!/usr/bin/env python3
"""Scan nearby periodic-loading patterns at fixed total amplifier length."""

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
from twpasolver.models import LCLfBaseCell, TWPA  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402


TARGET_CELLS = 40_800
ISTAR = 2.0e-3
IDC = 300e-6
IPUMP = 360e-6


def score(row: dict[str, float]) -> tuple[float, ...]:
    qualified = (
        row["mean_in_band_gain_db"] >= 15.0
        and row["fraction_above_17db"] >= 0.8
        and row["ripple_peak_to_peak_db"] <= 3.0
    )
    return (
        float(qualified),
        row["fraction_above_17db"],
        -row["ripple_peak_to_peak_db"],
        row["mean_in_band_gain_db"],
    )


def make_cell(row: dict[str, float], count: int) -> LCLfBaseCell:
    return LCLfBaseCell(
        L=row["L_pH"] * 1e-12,
        C=row["C_fF"] * 1e-15,
        Lf=row["Lf_pH"] * 1e-12,
        delta=row["loss_tangent"],
        N=count,
        centered=False,
    )


def main() -> None:
    cells, provenance = load_hfss_cell_parameters(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv"
    )
    signals = np.arange(4.0, 8.0001, 0.05)
    rows: list[dict[str, float | bool]] = []
    best_row = None
    best_gain = None

    for side in range(13, 18):
        for loaded in range(3, 6):
            period = 2 * side + loaded
            repeats = round(TARGET_CELLS / period)
            total_cells = period * repeats
            twpa = TWPA(
                cells=[
                    make_cell(cells["unloaded"], side),
                    make_cell(cells["loaded"], loaded),
                    make_cell(cells["unloaded"], side),
                ],
                N=repeats,
                Istar=ISTAR,
                Idc=IDC,
                Ip0=IPUMP,
            )
            analysis = TWPAnalysis(
                twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz"
            )
            # The first loading stopband scales approximately as 1/period.
            scale = 34.0 / period
            pumps = np.arange(11.05 * scale, 11.3501 * scale, 0.01 * scale)
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
                    gain = np.full_like(signals, np.nan)
                if not np.all(np.isfinite(gain)):
                    continue
                metrics = gain_metrics(signals, gain, band=(4.0, 8.0))
                row = {
                    "unloaded_cells_each_side": side,
                    "loaded_cells": loaded,
                    "period_cells": period,
                    "supercells": repeats,
                    "total_cells": total_cells,
                    "length_mm": total_cells * 2e-3,
                    "pump_GHz": float(pump),
                    "Idc_A": IDC,
                    "Ip0_A": IPUMP,
                    **metrics,
                }
                rows.append(row)
                if best_row is None or score(row) > score(best_row):
                    best_row = row
                    best_gain = gain.copy()

    if best_row is None or best_gain is None:
        raise RuntimeError("No finite pattern result")

    output = ROOT / "results" / "step_12_hfss_operating_point_optimization"
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output / "loading_pattern_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    qualified = [
        row for row in rows
        if row["mean_in_band_gain_db"] >= 15.0
        and row["fraction_above_17db"] >= 0.8
        and row["ripple_peak_to_peak_db"] <= 3.0
    ]
    report = {
        "fixed_operating_point": {"Idc_A": IDC, "Ip0_A": IPUMP},
        "target_total_cells": TARGET_CELLS,
        "evaluated_points": len(rows),
        "best": best_row,
        "qualified_count": len(qualified),
        "qualified": sorted(qualified, key=score, reverse=True)[:20],
        "hfss_provenance": provenance,
    }
    (output / "loading_pattern_scan.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output / "loading_pattern_best_gain.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_GHz", "gain_db"])
        writer.writerows(zip(signals, best_gain))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(signals, best_gain, linewidth=2)
    ax.axhline(17, color="C3", linestyle="--", label="17 dB coverage target")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Gain (dB)",
        title=(
            f"Pattern {best_row['unloaded_cells_each_side']}/"
            f"{best_row['loaded_cells']}/"
            f"{best_row['unloaded_cells_each_side']}, "
            f"pump={best_row['pump_GHz']:.3f} GHz"
        ),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loading_pattern_best_gain.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"best": best_row, "qualified_count": len(qualified)}, indent=2))


if __name__ == "__main__":
    main()
