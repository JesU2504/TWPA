#!/usr/bin/env python3
"""Coarse complete-model operating scan using joined 3--27 GHz HFSS data."""

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
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


def score(row: dict[str, object]) -> tuple[float, ...]:
    minimum = float(row["minimum_in_band_gain_db"])
    ripple = float(row["ripple_peak_to_peak_db"])
    fraction = float(row["fraction_above_17db"])
    return (
        float(minimum >= 17.0 and ripple <= 3.0),
        fraction,
        -max(0.0, 17.0 - minimum),
        -ripple,
        -abs(float(row["mean_in_band_gain_db"]) - 22.0),
    )


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "step_16_fine_hfss_complete_scan"
    output.mkdir(parents=True, exist_ok=True)

    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    f1, s1 = read_touchstone(inputs / "supercell_13_3_13_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / "supercell_13_3_13_aux_14_27GHz.s2p")
    if not np.allclose(s1[-1], s2[0], rtol=1e-10, atol=1e-12):
        raise ValueError("HFSS sweeps disagree at their 14 GHz overlap")
    measured_f = np.concatenate((f1, f2[1:]))
    measured_s = np.concatenate((s1, s2[1:]), axis=0)

    def fitted_cell(role: str) -> np.ndarray:
        row = cells[role]
        return lclf_s(
            measured_f * 1e9,
            row["L_pH"] * 1e-12,
            row["C_fF"] * 1e-15,
            row["Lf_pH"] * 1e-12,
        )

    lumped = cascade_pattern(fitted_cell("unloaded"), fitted_cell("loaded"))
    measured_phase, _ = phase_delay(measured_s[:, 1, 0], measured_f)
    lumped_phase, _ = phase_delay(lumped[:, 1, 0], measured_f)
    correction_samples = ((-measured_phase) - (-lumped_phase)) / SUPERCELL_CELLS

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 13),
            make_cell(cells["loaded"], 3),
            make_cell(cells["unloaded"], 13),
        ],
        N=1407,
        Istar=2e-3,
        Idc=315e-6,
        Ip0=310e-6,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")

    def prepare(idc: float, ip: float) -> None:
        analysis.twpa.Idc = idc
        analysis.twpa.Ip0 = ip
        analysis.update_base_data()
        dense_f = np.asarray(analysis.data["freqs"])
        correction = np.zeros_like(dense_f)
        valid = (dense_f >= measured_f[0]) & (dense_f <= measured_f[-1])
        correction[valid] = np.interp(dense_f[valid], measured_f, correction_samples)
        analysis.data["k"] = np.asarray(analysis.data["k"]) + correction
        modes = ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=1,
            n_frequency_conversion=1,
            n_signal_harmonics=1,
        )
        analysis.add_mode_array("fine_hfss_extended", modes)

    coarse_signals = np.arange(4.0, 8.0001, 0.5)
    pumps = np.arange(12.6, 13.4001, 0.1)
    idcs = np.array([250, 300, 350, 400]) * 1e-6
    ips = np.array([180, 230, 280, 330]) * 1e-6
    rows: list[dict[str, object]] = []
    for idc in idcs:
        for ip in ips:
            prepare(float(idc), float(ip))
            for pump in pumps:
                try:
                    result = analysis.gain(
                        coarse_signals,
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
                    continue
                if not np.all(np.isfinite(gain)):
                    continue
                metrics = gain_metrics(coarse_signals, gain, band=(4.0, 8.0))
                rows.append(
                    {
                        "pump_GHz": float(pump),
                        "Idc_A": float(idc),
                        "Ip0_A": float(ip),
                        **metrics,
                        "minimum_in_band_gain_db": float(np.min(gain)),
                        "maximum_in_band_gain_db": float(np.max(gain)),
                    }
                )

    rows.sort(key=score, reverse=True)
    if not rows:
        raise RuntimeError("No finite complete-model points")
    with (output / "01_coarse_complete_model_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Re-evaluate the five best distinct settings on the dense signal grid.
    signals = np.arange(4.0, 8.0001, 0.1)
    refined: list[dict[str, object]] = []
    curves: list[np.ndarray] = []
    for candidate in rows[:5]:
        prepare(float(candidate["Idc_A"]), float(candidate["Ip0_A"]))
        result = analysis.gain(
            signals,
            pump=float(candidate["pump_GHz"]),
            Is0=1.4e-6,
            Ip0=None,
            model="general",
            mode_array_config="fine_hfss_extended",
            thin=300,
            save=False,
        )
        gain = np.asarray(result["gain_db"], dtype=float)
        metrics = gain_metrics(signals, gain, band=(4.0, 8.0))
        refined.append(
            {
                "pump_GHz": float(candidate["pump_GHz"]),
                "Idc_A": float(candidate["Idc_A"]),
                "Ip0_A": float(candidate["Ip0_A"]),
                **metrics,
                "minimum_in_band_gain_db": float(np.min(gain)),
                "maximum_in_band_gain_db": float(np.max(gain)),
            }
        )
        curves.append(gain)
    order = sorted(range(len(refined)), key=lambda i: score(refined[i]), reverse=True)
    refined = [refined[i] for i in order]
    curves = [curves[i] for i in order]

    with (output / "02_refined_complete_model_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(refined[0]))
        writer.writeheader()
        writer.writerows(refined)
    with (output / "03_refined_gain_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *[f"candidate_{i+1}_gain_dB" for i in range(len(curves))]])
        writer.writerows(zip(signals, *curves))

    best = refined[0]
    report = {
        "status": "complete_model_scan_with_fine_3_to_27_GHz_HFSS_data",
        "evaluated_coarse_points": len(rows),
        "coarse_grid": {
            "signals_GHz": [4.0, 8.0, 0.5],
            "pumps_GHz": [12.6, 13.4, 0.1],
            "Idc_uA": [250, 300, 350, 400],
            "Ip0_uA": [180, 230, 280, 330],
        },
        "selected": best,
        "strict_pass": bool(
            float(best["minimum_in_band_gain_db"]) >= 17.0
            and float(best["ripple_peak_to_peak_db"]) <= 3.0
        ),
        "top_refined": refined,
        "hfss_provenance": provenance,
    }
    (output / "04_complete_model_scan_assessment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, (row, gain) in enumerate(zip(refined, curves), start=1):
        ax.plot(
            signals,
            gain,
            label=(
                f"{index}: fp={row['pump_GHz']:.2f} GHz, "
                f"Idc={row['Idc_A']*1e6:.0f} µA, Ip={row['Ip0_A']*1e6:.0f} µA"
            ),
        )
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Complete-model gain (dB)",
        title="Fine-HFSS complete-model operating scan",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "05_complete_model_candidates.png", dpi=180)
    fig.savefig(output / "05_complete_model_candidates.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
