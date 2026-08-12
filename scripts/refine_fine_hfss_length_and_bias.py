#!/usr/bin/env python3
"""Refine length and bias using the fine-HFSS eight-mode model."""

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

from analyze_hfss_supercell import SUPERCELL_CELLS, read_touchstone  # noqa: E402
from hfss_bloch import apply_bloch_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


def add_metrics(f: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    result = gain_metrics(f, gain, band=(4.0, 8.0))
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
    fraction = float(row["fraction_above_17db"])
    mean = float(row["mean_in_band_gain_db"])
    return (
        float(bool(row["strict_pass"])),
        fraction,
        -max(0.0, 17.0 - minimum),
        -max(0.0, ripple - 3.0),
        -ripple,
        -abs(mean - 22.0),
    )


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "bloch_corrected" / "step_17_length_and_bias_refinement"
    output.mkdir(parents=True, exist_ok=True)
    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")

    f1, s1 = read_touchstone(inputs / "supercell_13_3_13_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / "supercell_13_3_13_aux_14_27GHz.s2p")
    if not np.allclose(s1[-1], s2[0], rtol=1e-10, atol=1e-12):
        raise ValueError("HFSS sweeps disagree at 14 GHz")
    measured_f = np.concatenate((f1, f2[1:]))
    measured_s = np.concatenate((s1, s2[1:]), axis=0)

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 13),
            make_cell(cells["loaded"], 3),
            make_cell(cells["unloaded"], 13),
        ],
        N=1407,
        Istar=2e-3,
        Idc=400e-6,
        Ip0=280e-6,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")

    def prepare(n: int, idc: float, ip: float) -> None:
        analysis.twpa.N = int(n)
        analysis.twpa.Idc = float(idc)
        analysis.twpa.Ip0 = float(ip)
        analysis.update_base_data()
        apply_bloch_parameters(
            analysis.data, measured_f, measured_s, SUPERCELL_CELLS
        )
        modes = ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=1,
            n_frequency_conversion=1,
            n_signal_harmonics=1,
        )
        analysis.add_mode_array("fine_hfss_extended", modes)

    def evaluate(n: int, idc: float, ip: float, pump: float, signals: np.ndarray):
        prepare(n, idc, ip)
        try:
            result = analysis.gain(
                signals,
                pump=float(pump),
                Is0=1.4e-6,
                Ip0=None,
                model="general",
                mode_array_config="fine_hfss_extended",
                thin=300,
                save=False,
            )
            gain = np.asarray(result["gain_db"], dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(gain)):
            return None
        return {
            "supercell_repeats": int(n),
            "device_length_mm": float(n * 58e-3),
            "pump_GHz": float(pump),
            "Idc_A": float(idc),
            "Ip0_A": float(ip),
            **add_metrics(signals, gain),
        }, gain

    # Stage 1: isolate the length tradeoff at the best point found previously.
    coarse_signals = np.arange(4.0, 8.0001, 0.25)
    length_rows: list[dict[str, object]] = []
    for n in (500, 650, 800, 950, 1100, 1250, 1407):
        result = evaluate(n, 400e-6, 280e-6, 12.8, coarse_signals)
        if result is not None:
            length_rows.append(result[0])

    # Stage 2: local joint scan around the most promising length.
    best_length = max(length_rows, key=score)
    center_n = int(best_length["supercell_repeats"])
    n_values = sorted(set(max(350, center_n + dn) for dn in (-200, -100, 0, 100, 200)))
    rows: list[dict[str, object]] = []
    for n in n_values:
        for idc_uA in (360, 400, 440):
            for ip_uA in (240, 280, 320):
                for pump in (12.70, 12.75, 12.80, 12.85, 12.90):
                    result = evaluate(
                        n, idc_uA * 1e-6, ip_uA * 1e-6, pump, coarse_signals
                    )
                    if result is not None:
                        rows.append(result[0])
    rows.sort(key=score, reverse=True)

    with (output / "01_length_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(length_rows[0]))
        writer.writeheader()
        writer.writerows(length_rows)
    with (output / "02_joint_coarse_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Dense verification of the ten best coarse candidates.
    signals = np.arange(4.0, 8.0001, 0.05)
    refined: list[dict[str, object]] = []
    curves: list[np.ndarray] = []
    for candidate in rows[:10]:
        result = evaluate(
            int(candidate["supercell_repeats"]),
            float(candidate["Idc_A"]),
            float(candidate["Ip0_A"]),
            float(candidate["pump_GHz"]),
            signals,
        )
        if result is not None:
            refined.append(result[0])
            curves.append(result[1])
    order = sorted(range(len(refined)), key=lambda i: score(refined[i]), reverse=True)
    refined = [refined[i] for i in order]
    curves = [curves[i] for i in order]

    with (output / "03_dense_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(refined[0]))
        writer.writeheader()
        writer.writerows(refined)
    with (output / "04_dense_gain_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *[f"candidate_{i+1}_gain_dB" for i in range(len(curves))]])
        writer.writerows(zip(signals, *curves))

    best = refined[0]
    report = {
        "status": "fine_HFSS_complete_model_length_and_bias_refinement",
        "evaluated_joint_points": len(rows),
        "selected": best,
        "strict_pass": bool(best["strict_pass"]),
        "top_dense_candidates": refined,
        "interpretation": (
            "Length was varied to trade excess nonlinear gain for flatness. "
            "A strict pass requires minimum gain >=17 dB and ripple <=3 dB."
        ),
        "hfss_provenance": provenance,
    }
    (output / "05_assessment.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    lengths = np.array([float(row["device_length_mm"]) for row in length_rows])
    ax.plot(lengths, [float(row["minimum_in_band_gain_db"]) for row in length_rows], "o-", label="minimum gain")
    ax.plot(lengths, [float(row["maximum_in_band_gain_db"]) for row in length_rows], "o-", label="maximum gain")
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(xlabel="Active device length (mm)", ylabel="Gain (dB)", title="Length tradeoff at the initial candidate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for index, (row, gain) in enumerate(zip(refined[:5], curves[:5]), start=1):
        ax.plot(
            signals,
            gain,
            label=(f"{index}: {row['device_length_mm']:.1f} mm, "
                   f"fp {row['pump_GHz']:.2f}, Idc {row['Idc_A']*1e6:.0f}, Ip {row['Ip0_A']*1e6:.0f} µA"),
        )
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(xlabel="Signal frequency (GHz)", ylabel="Complete-model gain (dB)", title="Best jointly refined candidates")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "06_length_and_bias_refinement.png", dpi=180)
    fig.savefig(output / "06_length_and_bias_refinement.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
