#!/usr/bin/env python3
"""Targeted full coupled-mode scan for the selected 13/3/13 pattern."""

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
sys.path.insert(0, str(ROOT / "code"))

from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402


log.setLevel(logging.WARNING)


def qualified(row: dict[str, float]) -> bool:
    return (
        row["mean_in_band_gain_db"] >= 15.0
        and row["fraction_above_17db"] >= 0.8
        and row["ripple_peak_to_peak_db"] <= 3.0
    )


def score(row: dict[str, float]) -> tuple[float, ...]:
    return (
        float(qualified(row)),
        row["fraction_above_17db"],
        -max(0.0, row["ripple_peak_to_peak_db"] - 3.0),
        -abs(row["mean_in_band_gain_db"] - 20.0),
    )


def main() -> None:
    cells, provenance = load_hfss_cell_parameters(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv"
    )
    signals = np.arange(4.0, 8.0001, 0.1)
    pumps = np.array([13.10, 13.12, 13.14, 13.16])
    dc_biases = np.array([340, 350, 360, 375, 400]) * 1e-6
    pump_currents = np.array([330, 340, 350]) * 1e-6
    rows: list[dict[str, float | bool]] = []
    best_row = None
    best_gain = None

    for dc_bias in dc_biases:
        for pump_current in pump_currents:
            twpa = TWPA(
                cells=[
                    make_cell(cells["unloaded"], 13),
                    make_cell(cells["loaded"], 3),
                    make_cell(cells["unloaded"], 13),
                ],
                N=1407,
                Istar=2e-3,
                Idc=float(dc_bias),
                Ip0=float(pump_current),
            )
            analysis = TWPAnalysis(
                twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz"
            )
            analysis.update_base_data()
            extended = ModeArrayFactory.create_extended_3wm(
                analysis.data,
                n_pump_harmonics=1,
                n_frequency_conversion=1,
                n_signal_harmonics=1,
            )
            analysis.add_mode_array("hfss_extended", extended)
            for pump in pumps:
                try:
                    result = analysis.gain(
                        signals,
                        pump=float(pump),
                        Is0=1.4e-6,
                        Ip0=None,
                        model="general",
                        mode_array_config="hfss_extended",
                        thin=300,
                        save=False,
                    )
                    gain = np.asarray(result["gain_db"], dtype=float)
                except Exception:
                    continue
                if not np.all(np.isfinite(gain)):
                    continue
                row = {
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
        raise RuntimeError("No finite full-model result")

    output = ROOT / "results" / "step_12_hfss_operating_point_optimization"
    with (output / "full_model_targeted_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passing = [row for row in rows if qualified(row)]
    report = {
        "model": "general",
        "mode_array": {
            "n_pump_harmonics": 1,
            "n_frequency_conversion": 1,
            "n_signal_harmonics": 1,
        },
        "evaluated_points": len(rows),
        "qualified_count": len(passing),
        "selected": best_row,
        "top_qualified": sorted(passing, key=score, reverse=True)[:20],
        "hfss_provenance": provenance,
    }
    (output / "full_model_targeted_scan.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output / "full_model_selected_gain.csv").open("w", newline="") as handle:
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
            f"Full model: pump={best_row['pump_GHz']:.2f} GHz, "
            f"Idc={best_row['Idc_A']*1e6:.0f} µA, Ip={best_row['Ip0_A']*1e6:.0f} µA"
        ),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "full_model_selected_gain.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"selected": best_row, "qualified_count": len(passing)}, indent=2))


if __name__ == "__main__":
    main()
