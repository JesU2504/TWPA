#!/usr/bin/env python3
"""Provisional eight-mode screen of loading patterns using HFSS-fitted cells."""

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

from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


TARGET_CELLS = 30_450  # approximately 60.9 mm at 2 um per cell


def metrics(freqs: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    result = gain_metrics(freqs, gain, band=(4.0, 8.0))
    result["minimum_in_band_gain_db"] = float(np.min(gain))
    result["maximum_in_band_gain_db"] = float(np.max(gain))
    result["strict_pass"] = bool(
        result["minimum_in_band_gain_db"] >= 17.0
        and result["ripple_peak_to_peak_db"] <= 3.0
    )
    return result


def score(row: dict[str, object]) -> tuple[float, ...]:
    minimum = float(row["minimum_in_band_gain_db"])
    ripple = float(row["ripple_peak_to_peak_db"])
    return (
        float(bool(row["strict_pass"])),
        float(row["fraction_above_17db"]),
        -max(0.0, 17.0 - minimum),
        -ripple,
        -abs(float(row["mean_in_band_gain_db"]) - 22.0),
    )


def main() -> None:
    log.setLevel(logging.WARNING)
    cells, provenance = load_hfss_cell_parameters(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv"
    )
    output = ROOT / "results" / "step_19_complete_model_pattern_screen"
    output.mkdir(parents=True, exist_ok=True)
    signals = np.arange(4.0, 8.0001, 0.5)
    rows: list[dict[str, object]] = []

    for side in range(10, 18):
        for loaded in range(2, 6):
            period = 2 * side + loaded
            repeats = round(TARGET_CELLS / period)
            twpa = TWPA(
                cells=[
                    make_cell(cells["unloaded"], side),
                    make_cell(cells["loaded"], loaded),
                    make_cell(cells["unloaded"], side),
                ],
                N=repeats,
                Istar=2e-3,
                Idc=360e-6,
                Ip0=280e-6,
            )
            analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
            analysis.update_base_data()
            mode_array = ModeArrayFactory.create_extended_3wm(
                analysis.data,
                n_pump_harmonics=1,
                n_frequency_conversion=1,
                n_signal_harmonics=1,
            )
            analysis.add_mode_array("pattern_extended", mode_array)
            # First-stopband frequency scales approximately inversely with period.
            center = 12.70 * 29.0 / period
            for pump in np.arange(center - 0.15, center + 0.1501, 0.05):
                try:
                    result = analysis.gain(
                        signals,
                        pump=float(pump),
                        Is0=1.4e-6,
                        Ip0=None,
                        model="general_ideal",
                        mode_array_config="pattern_extended",
                        thin=300,
                        save=False,
                    )
                    gain = np.asarray(result["gain_db"], dtype=float)
                except Exception:
                    continue
                if not np.all(np.isfinite(gain)):
                    continue
                rows.append(
                    {
                        "unloaded_cells_each_side": side,
                        "loaded_cells": loaded,
                        "period_cells": period,
                        "supercell_repeats": repeats,
                        "device_length_mm": repeats * period * 2e-3,
                        "pump_GHz": float(pump),
                        "Idc_A": 360e-6,
                        "Ip0_A": 280e-6,
                        **metrics(signals, gain),
                    }
                )

    rows.sort(key=score, reverse=True)
    with (output / "01_pattern_screen.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Dense verification of the best pump for each of the top five distinct patterns.
    pattern_best: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["unloaded_cells_each_side"]), int(row["loaded_cells"]))
        if key not in seen:
            seen.add(key)
            pattern_best.append(row)
        if len(pattern_best) == 5:
            break

    dense_signals = np.arange(4.0, 8.0001, 0.1)
    dense_rows: list[dict[str, object]] = []
    curves: list[np.ndarray] = []
    for candidate in pattern_best:
        side = int(candidate["unloaded_cells_each_side"])
        loaded = int(candidate["loaded_cells"])
        repeats = int(candidate["supercell_repeats"])
        twpa = TWPA(
            cells=[
                make_cell(cells["unloaded"], side),
                make_cell(cells["loaded"], loaded),
                make_cell(cells["unloaded"], side),
            ],
            N=repeats,
            Istar=2e-3,
            Idc=360e-6,
            Ip0=280e-6,
        )
        analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
        analysis.update_base_data()
        mode_array = ModeArrayFactory.create_extended_3wm(
            analysis.data, n_pump_harmonics=1, n_frequency_conversion=1,
            n_signal_harmonics=1,
        )
        analysis.add_mode_array("pattern_extended", mode_array)
        result = analysis.gain(
            dense_signals,
            pump=float(candidate["pump_GHz"]),
            Is0=1.4e-6,
            Ip0=None,
            model="general_ideal",
            mode_array_config="pattern_extended",
            thin=300,
            save=False,
        )
        gain = np.asarray(result["gain_db"], dtype=float)
        dense_rows.append(
            {
                "unloaded_cells_each_side": side,
                "loaded_cells": loaded,
                "period_cells": 2 * side + loaded,
                "supercell_repeats": repeats,
                "device_length_mm": repeats * (2 * side + loaded) * 2e-3,
                "pump_GHz": float(candidate["pump_GHz"]),
                "Idc_A": 360e-6,
                "Ip0_A": 280e-6,
                **metrics(dense_signals, gain),
            }
        )
        curves.append(gain)
    order = sorted(range(len(dense_rows)), key=lambda i: score(dense_rows[i]), reverse=True)
    dense_rows = [dense_rows[i] for i in order]
    curves = [curves[i] for i in order]

    with (output / "02_dense_pattern_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dense_rows[0]))
        writer.writeheader()
        writer.writerows(dense_rows)
    with (output / "03_dense_pattern_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *[f"candidate_{i+1}_gain_dB" for i in range(len(curves))]])
        writer.writerows(zip(dense_signals, *curves))

    best = dense_rows[0]
    report = {
        "status": "provisional_pattern_screen_using_HFSS_fitted_unit_cells",
        "important_limitation": (
            "Alternative patterns do not yet include a physical-supercell HFSS phase correction. "
            "This screen ranks candidates for a future HFSS simulation; it is not final validation."
        ),
        "evaluated_points": len(rows),
        "selected_for_next_HFSS": best,
        "top_dense_patterns": dense_rows,
        "hfss_cell_provenance": provenance,
    }
    (output / "04_pattern_screen_assessment.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row, gain in zip(dense_rows, curves):
        ax.plot(
            dense_signals,
            gain,
            label=(f"{int(row['unloaded_cells_each_side'])}-{int(row['loaded_cells'])}-"
                   f"{int(row['unloaded_cells_each_side'])}, fp={row['pump_GHz']:.2f} GHz"),
        )
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(
        xlabel="Signal frequency (GHz)", ylabel="Ideal eight-mode gain (dB)",
        title="Provisional loading-pattern screen",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "05_pattern_candidates.png", dpi=180)
    fig.savefig(output / "05_pattern_candidates.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
