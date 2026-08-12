#!/usr/bin/env python3
"""Scan pump frequency on both sides of the first HFSS-derived stopband."""

from __future__ import annotations

import argparse
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


def pump_grid() -> np.ndarray:
    below = np.arange(10.3, 11.31, 0.1)
    above = np.arange(12.4, 14.51, 0.1)
    return np.concatenate([below, above])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip0-ua", type=float, default=100.0)
    parser.add_argument("--idc-ua", type=float, default=220.0)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    twpa, provenance = build_twpa_from_hfss(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv",
        ROOT / "hfss_inputs" / "amplifier_config.json",
    )
    twpa.Idc = args.idc_ua * 1e-6
    twpa.Ip0 = args.ip0_ua * 1e-6
    if twpa.Idc + twpa.Ip0 >= twpa.Istar:
        raise ValueError("Idc + Ip0 must remain below Istar for this scan")

    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
    analysis.update_base_data()
    signals = np.arange(2.0, 10.01, 0.1)
    pumps = pump_grid()
    gain_rows = []
    summaries = []
    for pump in pumps:
        result = analysis.gain(
            signals,
            pump=float(pump),
            Is0=1.4e-6,
            Ip0=twpa.Ip0,
            model="minimal_3wm",
            mode_array_config="basic_3wm",
            thin=300,
            save=False,
        )
        gain = np.asarray(result["gain_db"], dtype=float)
        gain_rows.append(gain)
        if np.all(np.isfinite(gain)):
            metrics = gain_metrics(signals, gain, band=(4.0, 8.0))
        else:
            metrics = {
                "finite": False,
                "mean_in_band_gain_db": float("-inf"),
                "fraction_above_17db": 0.0,
                "ripple_peak_to_peak_db": float("inf"),
                "peak_gain_db": float("nan"),
                "peak_frequency_GHz": float("nan"),
            }
        summaries.append({"pump_GHz": float(pump), **metrics})

    gain_map = np.asarray(gain_rows)
    finite = [row for row in summaries if row["finite"]]
    best = max(
        finite,
        key=lambda row: (
            row["fraction_above_17db"],
            row["mean_in_band_gain_db"],
            -row["ripple_peak_to_peak_db"],
        ),
    )

    output = ROOT / "results" / "step_12_hfss_operating_point_optimization"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"pump_scan_{args.label}_idc{args.idc_ua:g}uA_ip{args.ip0_ua:g}uA"
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pump_GHz", "signal_GHz", "gain_db"])
        for pump, row in zip(pumps, gain_map):
            for signal, gain in zip(signals, row):
                writer.writerow([pump, signal, gain])

    report = {
        "label": args.label,
        "Idc_A": twpa.Idc,
        "Ip0_A": twpa.Ip0,
        "Istar_A": twpa.Istar,
        "pump_ranges_GHz": [[10.3, 11.3], [12.4, 14.5]],
        "excluded_first_stopband_GHz": [11.44, 12.31],
        "best": best,
        "all_pumps": summaries,
        "hfss_provenance": provenance["cell_input"],
    }
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.pcolormesh(signals, pumps, gain_map, shading="auto", cmap="magma")
    fig.colorbar(image, ax=ax, label="Gain (dB)")
    ax.axvspan(4, 8, color="cyan", alpha=0.08)
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Pump frequency (GHz)",
        title=f"HFSS pump scan: Idc={args.idc_ua:g} µA, Ip={args.ip0_ua:g} µA",
    )
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(output), "best": best}, indent=2))


if __name__ == "__main__":
    main()
