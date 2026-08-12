#!/usr/bin/env python3
"""Fine operating-point scan around the best coarse HFSS-derived solution."""

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
from twpa_project_utils import build_twpa_from_hfss, gain_metrics  # noqa: E402


def candidate_score(row: dict[str, float]) -> tuple[float, ...]:
    """Prefer target-compliant points, then coverage, ripple, and mean gain."""
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


def main() -> None:
    twpa, provenance = build_twpa_from_hfss(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv",
        ROOT / "hfss_inputs" / "amplifier_config.json",
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")

    signals = np.arange(4.0, 8.0001, 0.05)
    pumps = np.arange(11.05, 11.3501, 0.01)
    dc_biases = np.arange(250, 375.1, 25.0) * 1e-6
    pump_currents = np.arange(300, 400.1, 10.0) * 1e-6

    rows: list[dict[str, float | bool]] = []
    best_row = None
    best_gain = None

    for dc_bias in dc_biases:
        for pump_current in pump_currents:
            if dc_bias + pump_current >= analysis.twpa.Istar:
                continue
            analysis.twpa.Idc = float(dc_bias)
            analysis.twpa.Ip0 = float(pump_current)
            analysis.update_base_data()
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

                if np.all(np.isfinite(gain)):
                    metrics = gain_metrics(signals, gain, band=(4.0, 8.0))
                else:
                    metrics = {
                        "finite": False,
                        "mean_in_band_gain_db": float("-inf"),
                        "median_in_band_gain_db": float("-inf"),
                        "ripple_peak_to_peak_db": float("inf"),
                        "ripple_std_db": float("inf"),
                        "fraction_above_15db": 0.0,
                        "fraction_above_17db": 0.0,
                        "fraction_above_20db": 0.0,
                        "peak_gain_db": float("nan"),
                        "peak_frequency_GHz": float("nan"),
                    }
                row = {
                    "pump_GHz": float(pump),
                    "Idc_A": float(dc_bias),
                    "Ip0_A": float(pump_current),
                    **metrics,
                }
                rows.append(row)
                if row["finite"] and (
                    best_row is None or candidate_score(row) > candidate_score(best_row)
                ):
                    best_row = row
                    best_gain = gain.copy()

    if best_row is None or best_gain is None:
        raise RuntimeError("No finite operating point found")

    output = ROOT / "results" / "step_12_hfss_operating_point_optimization"
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pump_GHz", "Idc_A", "Ip0_A", "finite", "mean_in_band_gain_db",
        "median_in_band_gain_db", "ripple_peak_to_peak_db", "ripple_std_db",
        "fraction_above_15db", "fraction_above_17db", "fraction_above_20db",
        "peak_gain_db", "peak_frequency_GHz",
    ]
    with (output / "fine_operating_point_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    qualified = [
        row for row in rows
        if row["finite"]
        and row["mean_in_band_gain_db"] >= 15.0
        and row["fraction_above_17db"] >= 0.8
        and row["ripple_peak_to_peak_db"] <= 3.0
    ]
    report = {
        "search": {
            "signal_band_GHz": [4.0, 8.0],
            "signal_step_GHz": 0.05,
            "pump_band_GHz": [11.05, 11.35],
            "pump_step_GHz": 0.01,
            "Idc_uA": list(dc_biases * 1e6),
            "Ip0_uA": list(pump_currents * 1e6),
            "evaluated_points": len(rows),
        },
        "best": best_row,
        "qualified_count": len(qualified),
        "qualified": sorted(qualified, key=candidate_score, reverse=True)[:20],
        "hfss_provenance": provenance["cell_input"],
    }
    (output / "fine_operating_point_scan.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output / "fine_best_gain.csv").open("w", newline="") as handle:
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
            f"Fine scan: pump={best_row['pump_GHz']:.2f} GHz, "
            f"Idc={best_row['Idc_A']*1e6:.0f} µA, Ip={best_row['Ip0_A']*1e6:.0f} µA"
        ),
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fine_best_gain.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"best": best_row, "qualified_count": len(qualified)}, indent=2))


if __name__ == "__main__":
    main()
