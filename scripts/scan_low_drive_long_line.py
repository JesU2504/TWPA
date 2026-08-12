#!/usr/bin/env python3
"""Test whether lower drive and a longer 13-3-13 line suppress gain ripple."""

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

from analyze_hfss_supercell import SUPERCELL_CELLS, cascade_pattern, phase_delay, read_touchstone  # noqa: E402
from fit_hfss_unit_cells import lclf_s  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


def measure(f: np.ndarray, g: np.ndarray) -> dict[str, object]:
    m = gain_metrics(f, g, band=(4.0, 8.0))
    m["minimum_in_band_gain_db"] = float(np.min(g))
    m["maximum_in_band_gain_db"] = float(np.max(g))
    m["strict_pass"] = bool(m["minimum_in_band_gain_db"] >= 17 and m["ripple_peak_to_peak_db"] <= 3)
    return m


def score(r: dict[str, object]) -> tuple[float, ...]:
    min_gain = float(r["minimum_in_band_gain_db"])
    ripple = float(r["ripple_peak_to_peak_db"])
    return (
        float(bool(r["strict_pass"])),
        float(r["fraction_above_17db"]),
        -max(0.0, 17.0 - min_gain),
        -ripple,
        -abs(float(r["mean_in_band_gain_db"]) - 22.0),
    )


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "step_20_low_drive_long_line"
    output.mkdir(parents=True, exist_ok=True)
    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    f1, s1 = read_touchstone(inputs / "supercell_13_3_13_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / "supercell_13_3_13_aux_14_27GHz.s2p")
    measured_f = np.concatenate((f1, f2[1:]))
    measured_s = np.concatenate((s1, s2[1:]), axis=0)

    def cell_s(role: str) -> np.ndarray:
        row = cells[role]
        return lclf_s(measured_f * 1e9, row["L_pH"] * 1e-12, row["C_fF"] * 1e-15, row["Lf_pH"] * 1e-12)

    lumped = cascade_pattern(cell_s("unloaded"), cell_s("loaded"))
    measured_phase, _ = phase_delay(measured_s[:, 1, 0], measured_f)
    lumped_phase, _ = phase_delay(lumped[:, 1, 0], measured_f)
    correction_samples = ((-measured_phase) - (-lumped_phase)) / SUPERCELL_CELLS

    twpa = TWPA(
        cells=[make_cell(cells["unloaded"], 13), make_cell(cells["loaded"], 3), make_cell(cells["unloaded"], 13)],
        N=1400, Istar=2e-3, Idc=300e-6, Ip0=150e-6,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")

    def prepare(n: int, idc: float, ip: float) -> None:
        analysis.twpa.N, analysis.twpa.Idc, analysis.twpa.Ip0 = int(n), float(idc), float(ip)
        analysis.update_base_data()
        dense_f = np.asarray(analysis.data["freqs"])
        correction = np.zeros_like(dense_f)
        mask = (dense_f >= measured_f[0]) & (dense_f <= measured_f[-1])
        correction[mask] = np.interp(dense_f[mask], measured_f, correction_samples)
        analysis.data["k"] = np.asarray(analysis.data["k"]) + correction
        modes = ModeArrayFactory.create_extended_3wm(
            analysis.data, n_pump_harmonics=1, n_frequency_conversion=1, n_signal_harmonics=1
        )
        analysis.add_mode_array("extended", modes)

    def solve(n: int, idc: float, ip: float, pump: float, signals: np.ndarray):
        prepare(n, idc, ip)
        try:
            result = analysis.gain(signals, pump=pump, Is0=1.4e-6, Ip0=None, model="general", mode_array_config="extended", thin=300, save=False)
            gain = np.asarray(result["gain_db"], dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(gain)):
            return None
        return {
            "supercell_repeats": n, "device_length_mm": n * 0.058,
            "pump_GHz": pump, "Idc_A": idc, "Ip0_A": ip, **measure(signals, gain)
        }, gain

    # Sparse signals are used only to nominate candidates; dense verification follows.
    signals = np.arange(4.0, 8.0001, 0.5)
    rows: list[dict[str, object]] = []
    for n in (1400, 1800, 2200, 2600):
        for idc_uA in (250, 350, 450):
            for ip_uA in (100, 150, 200, 250):
                for pump in (12.6, 12.7, 12.8, 12.9, 13.0):
                    result = solve(n, idc_uA * 1e-6, ip_uA * 1e-6, pump, signals)
                    if result is not None:
                        rows.append(result[0])
    rows.sort(key=score, reverse=True)
    with (output / "01_coarse_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    dense_f = np.arange(4.0, 8.0001, 0.05)
    dense_rows: list[dict[str, object]] = []
    curves: list[np.ndarray] = []
    for row in rows[:12]:
        result = solve(int(row["supercell_repeats"]), float(row["Idc_A"]), float(row["Ip0_A"]), float(row["pump_GHz"]), dense_f)
        if result is not None:
            dense_rows.append(result[0]); curves.append(result[1])
    order = sorted(range(len(dense_rows)), key=lambda i: score(dense_rows[i]), reverse=True)
    dense_rows = [dense_rows[i] for i in order]; curves = [curves[i] for i in order]
    with (output / "02_dense_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dense_rows[0])); writer.writeheader(); writer.writerows(dense_rows)
    with (output / "03_dense_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["signal_GHz", *[f"candidate_{i+1}_gain_dB" for i in range(len(curves))]]); writer.writerows(zip(dense_f, *curves))

    best = dense_rows[0]
    report = {
        "status": "fine_HFSS_complete_model_low_drive_long_line_scan",
        "evaluated_points": len(rows), "selected": best,
        "strict_pass": bool(best["strict_pass"]), "top_dense_candidates": dense_rows,
        "hfss_provenance": provenance,
    }
    (output / "04_assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, (row, gain) in enumerate(zip(dense_rows[:5], curves[:5]), 1):
        ax.plot(dense_f, gain, label=f"{idx}: {row['device_length_mm']:.0f} mm, fp {row['pump_GHz']:.2f}, Idc {row['Idc_A']*1e6:.0f}, Ip {row['Ip0_A']*1e6:.0f} µA")
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(xlabel="Signal frequency (GHz)", ylabel="Complete-model gain (dB)", title="Lower-drive, longer-line candidates")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output / "05_low_drive_long_line.png", dpi=180); fig.savefig(output / "05_low_drive_long_line.pdf"); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
