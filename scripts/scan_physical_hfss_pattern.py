#!/usr/bin/env python3
"""Scan a physically simulated HFSS loading pattern with the eight-mode model.

Unlike the older one-off scripts, this script keeps the loading counts,
supercell length, Touchstone files, and HFSS Bloch parameters consistent for
each pattern.  It is intended for comparisons between 13-3-13, 16-2-16, and
the paper-baseline 15-4-15 supercells.
"""

from __future__ import annotations

import argparse
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

from analyze_hfss_supercell import read_touchstone  # noqa: E402
from hfss_bloch import apply_bloch_parameters, extract_bloch_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


PATTERNS = {
    "13-3-13": {"left": 13, "loaded": 3, "right": 13, "length_um": 58.0},
    "16-2-16": {"left": 16, "loaded": 2, "right": 16, "length_um": 68.0},
    "15-4-15": {"left": 15, "loaded": 4, "right": 15, "length_um": 68.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=PATTERNS, default="15-4-15")
    return parser.parse_args()


def measure(frequency: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    result = gain_metrics(frequency, gain, band=(4.0, 8.0))
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
    args = parse_args()
    config = PATTERNS[args.pattern]
    left, loaded, right = (int(config[key]) for key in ("left", "loaded", "right"))
    cell_count = left + loaded + right
    length_um = float(config["length_um"])
    slug = args.pattern.replace("-", "_")

    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "bloch_corrected" / f"step_22_{slug}_physical_nonlinear_scan"
    output.mkdir(parents=True, exist_ok=True)

    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    f1, s1 = read_touchstone(inputs / f"supercell_{slug}_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / f"supercell_{slug}_aux_14_27GHz.s2p")
    if not np.isclose(f1[-1], f2[0]) or not np.allclose(s1[-1], s2[0], atol=1e-12):
        raise ValueError("HFSS fine and auxiliary sweeps do not agree at 14 GHz")
    measured_f = np.concatenate((f1, f2[1:]))
    measured_s = np.concatenate((s1, s2[1:]), axis=0)

    measured_bloch = extract_bloch_parameters(measured_s)

    # Passive scan: useful for seeing where the physical structure most nearly
    # phase matches before nonlinear phase shifts are included.
    passive_signals = np.arange(4.0, 8.0001, 0.05)
    passive_pumps = np.arange(11.0, 13.4001, 0.02)
    k_per_mm = measured_bloch.phase / (length_um / 1000.0)
    passive_rows: list[dict[str, float]] = []
    passive_curves: list[np.ndarray] = []
    for pump in passive_pumps:
        mismatch = (
            np.interp(pump, measured_f, k_per_mm)
            - np.interp(passive_signals, measured_f, k_per_mm)
            - np.interp(pump - passive_signals, measured_f, k_per_mm)
        )
        passive_curves.append(mismatch)
        passive_rows.append(
            {
                "pump_GHz": float(pump),
                "mean_abs_mismatch_rad_per_mm": float(np.mean(np.abs(mismatch))),
                "minimum_mismatch_rad_per_mm": float(np.min(mismatch)),
                "maximum_mismatch_rad_per_mm": float(np.max(mismatch)),
                "peak_to_peak_mismatch_rad_per_mm": float(np.ptp(mismatch)),
            }
        )
    with (output / "01_passive_pump_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(passive_rows[0]))
        writer.writeheader()
        writer.writerows(passive_rows)

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], left),
            make_cell(cells["loaded"], loaded),
            make_cell(cells["unloaded"], right),
        ],
        N=1200,
        Istar=2e-3,
        Idc=350e-6,
        Ip0=220e-6,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")

    def prepare(repeats: int, idc: float, ip: float) -> None:
        analysis.twpa.N = int(repeats)
        analysis.twpa.Idc = float(idc)
        analysis.twpa.Ip0 = float(ip)
        analysis.update_base_data()
        apply_bloch_parameters(analysis.data, measured_f, measured_s, cell_count)
        modes = ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=1,
            n_frequency_conversion=1,
            n_signal_harmonics=1,
        )
        analysis.add_mode_array("physical_hfss_extended", modes)

    def solve(
        repeats: int, idc: float, ip: float, pump: float, signals: np.ndarray
    ) -> tuple[dict[str, object], np.ndarray] | None:
        prepare(repeats, idc, ip)
        try:
            solved = analysis.gain(
                signals,
                pump=float(pump),
                Is0=1.4e-6,
                Ip0=None,
                model="general",
                mode_array_config="physical_hfss_extended",
                thin=300,
                save=False,
            )
            gain = np.asarray(solved["gain_db"], dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(gain)):
            return None
        row = {
            "pattern": args.pattern,
            "supercell_repeats": int(repeats),
            "device_length_mm": float(repeats * length_um / 1000.0),
            "pump_GHz": float(pump),
            "Idc_A": float(idc),
            "Ip0_A": float(ip),
            **measure(signals, gain),
        }
        return row, gain

    coarse_signals = np.arange(4.0, 8.0001, 0.5)
    rows: list[dict[str, object]] = []
    for repeats in (900, 1200, 1600, 2200):
        for idc_uA in (250, 350, 450):
            for ip_uA in (150, 220, 290):
                for pump in np.arange(11.0, 13.4001, 0.2):
                    result = solve(
                        repeats,
                        idc_uA * 1e-6,
                        ip_uA * 1e-6,
                        float(pump),
                        coarse_signals,
                    )
                    if result is not None:
                        rows.append(result[0])
    if not rows:
        raise RuntimeError("No finite nonlinear scan points")
    rows.sort(key=score, reverse=True)
    with (output / "02_coarse_nonlinear_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dense_signals = np.arange(4.0, 8.0001, 0.05)
    dense_rows: list[dict[str, object]] = []
    dense_curves: list[np.ndarray] = []
    # Verify more candidates than the older scripts because sparse sampling can
    # hide narrow gain peaks and dips.
    for candidate in rows[:16]:
        result = solve(
            int(candidate["supercell_repeats"]),
            float(candidate["Idc_A"]),
            float(candidate["Ip0_A"]),
            float(candidate["pump_GHz"]),
            dense_signals,
        )
        if result is not None:
            dense_rows.append(result[0])
            dense_curves.append(result[1])
    order = sorted(range(len(dense_rows)), key=lambda index: score(dense_rows[index]), reverse=True)
    dense_rows = [dense_rows[index] for index in order]
    dense_curves = [dense_curves[index] for index in order]

    with (output / "03_dense_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dense_rows[0]))
        writer.writeheader()
        writer.writerows(dense_rows)
    with (output / "04_dense_gain_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["signal_GHz", *[f"candidate_{index + 1}_gain_dB" for index in range(len(dense_curves))]]
        )
        writer.writerows(zip(dense_signals, *dense_curves))

    best_passive = min(passive_rows, key=lambda row: row["mean_abs_mismatch_rad_per_mm"])
    best = dense_rows[0]
    report = {
        "status": "physical_HFSS_eight_mode_pattern_scan",
        "pattern": args.pattern,
        "geometry": {
            "unloaded_cells_left": left,
            "loaded_cells": loaded,
            "unloaded_cells_right": right,
            "cells_per_supercell": cell_count,
            "supercell_length_um": length_um,
        },
        "hfss_files": [
            f"supercell_{slug}_fine_3_14GHz.s2p",
            f"supercell_{slug}_aux_14_27GHz.s2p",
        ],
        "dispersion_extraction": apply_bloch_parameters(
            {
                "freqs": measured_f.copy(),
                "k": np.zeros_like(measured_f),
                "alpha": np.zeros_like(measured_f),
                "gammas": np.zeros_like(measured_f, dtype=complex),
            },
            measured_f,
            measured_s,
            cell_count,
        ),
        "best_passive_linear_mismatch": best_passive,
        "evaluated_coarse_points": len(rows),
        "coarse_grid": {
            "supercell_repeats": [900, 1200, 1600, 2200],
            "pump_GHz": [11.0, 13.4, 0.2],
            "Idc_uA": [250, 350, 450],
            "Ip0_uA": [150, 220, 290],
            "signal_GHz": [4.0, 8.0, 0.5],
        },
        "selected_dense_candidate": best,
        "strict_pass": bool(best["strict_pass"]),
        "top_dense_candidates": dense_rows,
        "hfss_cell_fit_provenance": provenance,
        "cautions": [
            "Measured Bloch parameters replace solver k, alpha, and gamma only inside 3-27 GHz.",
            "Port-launch de-embedding remains unavailable in the supplied HFSS exports.",
            "The nonlinear model includes p, s, i, p2, ps, pi, s2, and i2 modes.",
            "The coarse scan nominates candidates; reported selection comes from 0.05 GHz dense verification.",
        ],
    }
    (output / "05_assessment.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    mismatch_map = np.asarray(passive_curves)
    image = axes[0].pcolormesh(
        passive_signals, passive_pumps, mismatch_map, shading="auto", cmap="viridis"
    )
    fig.colorbar(image, ax=axes[0], label="Linear mismatch (rad/mm)")
    axes[0].set(
        xlabel="Signal frequency (GHz)",
        ylabel="Pump frequency (GHz)",
        title=f"{args.pattern}: measured HFSS linear 3WM mismatch",
    )
    for index, (row, gain) in enumerate(zip(dense_rows[:5], dense_curves[:5]), 1):
        axes[1].plot(
            dense_signals,
            gain,
            label=(
                f"{index}: {row['device_length_mm']:.1f} mm, fp {row['pump_GHz']:.1f}, "
                f"Idc {row['Idc_A'] * 1e6:.0f}, Ip {row['Ip0_A'] * 1e6:.0f} µA"
            ),
        )
    axes[1].axhline(17.0, color="black", linestyle=":", label="17 dB minimum")
    axes[1].set(
        xlabel="Signal frequency (GHz)",
        ylabel="Eight-mode gain (dB)",
        title=f"{args.pattern}: densely verified nonlinear candidates",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "06_physical_pattern_scan.png", dpi=180)
    fig.savefig(output / "06_physical_pattern_scan.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
