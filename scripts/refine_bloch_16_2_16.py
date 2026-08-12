#!/usr/bin/env python3
"""Refine and validate the Bloch-corrected 16-2-16 operating point.

The search is deliberately staged:

1. map the two passing pump regions found by the physical-pattern scan;
2. refine length, bias, pump current, and pump frequency around the best seeds;
3. verify finalists on a 0.01 GHz signal grid; and
4. compare propagation models and mode-family ablations at the winner.

Pumps below 11 GHz are excluded because an 8 GHz signal would then produce an
idler below the 3 GHz lower bound of the available physical-supercell HFSS
data. No Bloch data are extrapolated for candidate selection.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "scripts")]

from analyze_hfss_supercell import read_touchstone  # noqa: E402
from hfss_bloch import apply_bloch_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


LEFT = 16
LOADED = 2
RIGHT = 16
CELLS_PER_SUPERCELL = 34
SUPERCELL_LENGTH_UM = 68.0
IS0 = 1.4e-6


def summarize(signals: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    result = gain_metrics(signals, gain, band=(4.0, 8.0))
    result["minimum_in_band_gain_db"] = float(np.min(gain))
    result["maximum_in_band_gain_db"] = float(np.max(gain))
    result["strict_pass"] = bool(
        result["minimum_in_band_gain_db"] >= 17.0
        and result["ripple_peak_to_peak_db"] <= 3.0
    )
    return result


def score(row: dict[str, object]) -> tuple[float, ...]:
    """Prefer passing, flat, moderate-gain points with operating margin."""
    return (
        float(bool(row["strict_pass"])),
        -float(row["ripple_peak_to_peak_db"]),
        -abs(float(row["mean_in_band_gain_db"]) - 25.0),
        float(row["minimum_in_band_gain_db"]),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "bloch_corrected" / "step_25_16_2_16_refinement"
    output.mkdir(parents=True, exist_ok=True)

    cells, cell_provenance = load_hfss_cell_parameters(
        inputs / "hfss_cell_parameters.csv"
    )
    f1, s1 = read_touchstone(inputs / "supercell_16_2_16_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / "supercell_16_2_16_aux_14_27GHz.s2p")
    if not np.isclose(f1[-1], f2[0]) or not np.allclose(s1[-1], s2[0], atol=1e-12):
        raise ValueError("The 16-2-16 HFSS sweeps disagree at 14 GHz")
    measured_f = np.concatenate((f1, f2[1:]))
    measured_s = np.concatenate((s1, s2[1:]), axis=0)

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], LEFT),
            make_cell(cells["loaded"], LOADED),
            make_cell(cells["unloaded"], RIGHT),
        ],
        N=900,
        Istar=2e-3,
        Idc=250e-6,
        Ip0=290e-6,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
    extraction_provenance: dict[str, object] | None = None

    def prepare(repeats: int, idc: float, ip: float) -> None:
        nonlocal extraction_provenance
        analysis.twpa.N = int(repeats)
        analysis.twpa.Idc = float(idc)
        analysis.twpa.Ip0 = float(ip)
        analysis.update_base_data()
        extraction_provenance = apply_bloch_parameters(
            analysis.data, measured_f, measured_s, CELLS_PER_SUPERCELL
        )
        modes = ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=1,
            n_frequency_conversion=1,
            n_signal_harmonics=1,
        )
        analysis.add_mode_array("all_8_modes", modes)

    def solve_prepared(
        repeats: int,
        idc: float,
        ip: float,
        pump: float,
        signals: np.ndarray,
        *,
        model: str = "general",
        mode_config: str = "all_8_modes",
    ) -> tuple[dict[str, object], np.ndarray] | None:
        try:
            result = analysis.gain(
                signals,
                pump=float(pump),
                Is0=IS0,
                model=model,
                mode_array_config=mode_config,
                thin=300,
                save=False,
            )
            gain = np.asarray(result["gain_db"], dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(gain)):
            return None
        row = {
            "supercell_repeats": int(repeats),
            "device_length_mm": float(repeats * SUPERCELL_LENGTH_UM / 1000.0),
            "pump_GHz": float(pump),
            "Idc_A": float(idc),
            "Ip0_A": float(ip),
            **summarize(signals, gain),
        }
        return row, gain

    def evaluate(
        configurations: set[tuple[int, float, float, float]],
        signals: np.ndarray,
    ) -> list[dict[str, object]]:
        grouped: dict[tuple[int, float, float], list[float]] = defaultdict(list)
        for repeats, idc, ip, pump in configurations:
            grouped[(repeats, idc, ip)].append(pump)
        rows: list[dict[str, object]] = []
        for (repeats, idc, ip), pumps in sorted(grouped.items()):
            prepare(repeats, idc, ip)
            for pump in sorted(set(pumps)):
                solved = solve_prepared(repeats, idc, ip, pump, signals)
                if solved is not None:
                    rows.append(solved[0])
        rows.sort(key=score, reverse=True)
        return rows

    # Stage 1: cover both passing islands without mixing unmeasured idler data.
    stage1_signals = np.arange(4.0, 8.0001, 0.1)
    pumps = np.concatenate(
        (np.arange(11.0, 11.3001, 0.05), np.arange(12.7, 13.3001, 0.05))
    )
    stage1 = {
        (900, idc_uA * 1e-6, ip_uA * 1e-6, float(pump))
        for idc_uA in (200, 225, 250, 275)
        for ip_uA in (120, 150, 180, 220, 250, 290, 320)
        for pump in pumps
    }
    print(f"Stage 1: evaluating {len(stage1)} operating points", flush=True)
    stage1_rows = evaluate(stage1, stage1_signals)
    if not stage1_rows:
        raise RuntimeError("No finite stage-1 points")
    write_rows(output / "01_regional_scan.csv", stage1_rows)

    # Stage 2: refine length and all controls around the best six seeds.
    stage2: set[tuple[int, float, float, float]] = set()
    for seed in stage1_rows[:6]:
        for repeats in (800, 850, 900, 950, 1000):
            for idc_offset in (-25e-6, 0.0, 25e-6):
                for ip_offset in (-20e-6, 0.0, 20e-6):
                    for pump_offset in (-0.05, 0.0, 0.05):
                        idc = float(seed["Idc_A"]) + idc_offset
                        ip = float(seed["Ip0_A"]) + ip_offset
                        pump = float(seed["pump_GHz"]) + pump_offset
                        if 11.0 <= pump <= 13.4 and idc > 0 and ip > 0:
                            stage2.add((repeats, idc, ip, pump))
    stage2_signals = np.arange(4.0, 8.0001, 0.05)
    print(f"Stage 2: evaluating {len(stage2)} refined points", flush=True)
    stage2_rows = evaluate(stage2, stage2_signals)
    if not stage2_rows:
        raise RuntimeError("No finite stage-2 points")
    write_rows(output / "02_local_refinement.csv", stage2_rows)

    # Stage 3: refine pump at 0.01 GHz around the best six local candidates.
    stage3: set[tuple[int, float, float, float]] = set()
    for seed in stage2_rows[:6]:
        for pump in np.arange(
            float(seed["pump_GHz"]) - 0.04,
            float(seed["pump_GHz"]) + 0.0401,
            0.01,
        ):
            if 11.0 <= pump <= 13.4:
                stage3.add(
                    (
                        int(seed["supercell_repeats"]),
                        float(seed["Idc_A"]),
                        float(seed["Ip0_A"]),
                        float(pump),
                    )
                )
    print(f"Stage 3: evaluating {len(stage3)} fine-pump points", flush=True)
    stage3_rows = evaluate(stage3, stage2_signals)
    if not stage3_rows:
        raise RuntimeError("No finite stage-3 points")
    write_rows(output / "03_fine_pump_scan.csv", stage3_rows)

    # Finalists: the specification is checked on a 0.01 GHz signal grid.
    final_signals = np.arange(4.0, 8.0001, 0.01)
    finalists: list[dict[str, object]] = []
    final_curves: list[np.ndarray] = []
    for seed in stage3_rows[:10]:
        repeats = int(seed["supercell_repeats"])
        idc = float(seed["Idc_A"])
        ip = float(seed["Ip0_A"])
        pump = float(seed["pump_GHz"])
        prepare(repeats, idc, ip)
        solved = solve_prepared(repeats, idc, ip, pump, final_signals)
        if solved is not None:
            finalists.append(solved[0])
            final_curves.append(solved[1])
    order = sorted(range(len(finalists)), key=lambda index: score(finalists[index]), reverse=True)
    finalists = [finalists[index] for index in order]
    final_curves = [final_curves[index] for index in order]
    if not finalists:
        raise RuntimeError("No finite finalists")
    write_rows(output / "04_dense_finalists.csv", finalists)
    with (output / "05_dense_gain_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["signal_GHz", *[f"finalist_{i + 1}_gain_dB" for i in range(len(finalists))]]
        )
        writer.writerows(zip(final_signals, *final_curves))

    best = finalists[0]
    repeats = int(best["supercell_repeats"])
    idc = float(best["Idc_A"])
    ip = float(best["Ip0_A"])
    pump = float(best["pump_GHz"])
    prepare(repeats, idc, ip)

    # Propagation-model sensitivity at the same physical operating point.
    propagation_models: dict[str, dict[str, object]] = {}
    for model in ("general_ideal", "general_loss_only", "general"):
        solved = solve_prepared(
            repeats, idc, ip, pump, final_signals, model=model
        )
        if solved is not None:
            propagation_models[model] = solved[0]

    # Controlled mode-family ablation at the refined winner.
    mode_specs = {
        "basic_p_s_i": (0, 0, 0),
        "all_except_p2": (0, 1, 1),
        "all_except_ps_pi": (1, 0, 1),
        "all_except_s2_i2": (1, 1, 0),
        "all_8_modes": (1, 1, 1),
    }
    ablation: dict[str, dict[str, object]] = {}
    for label, (pump_harmonics, conversions, signal_harmonics) in mode_specs.items():
        if label == "basic_p_s_i":
            modes = ModeArrayFactory.create_basic(analysis.data)
        else:
            modes = ModeArrayFactory.create_extended_3wm(
                analysis.data,
                n_pump_harmonics=pump_harmonics,
                n_frequency_conversion=conversions,
                n_signal_harmonics=signal_harmonics,
            )
        analysis.add_mode_array(label, modes)
        solved = solve_prepared(
            repeats,
            idc,
            ip,
            pump,
            final_signals,
            mode_config=label,
        )
        if solved is not None:
            ablation[label] = solved[0]

    report = {
        "status": "16_2_16_Bloch_corrected_refinement",
        "geometry": {
            "pattern": "16-2-16",
            "cells_per_supercell": CELLS_PER_SUPERCELL,
            "supercell_length_um": SUPERCELL_LENGTH_UM,
        },
        "search_limits": {
            "minimum_pump_GHz": 11.0,
            "reason": "For an 8 GHz signal, lower pumps require idler Bloch data below the measured 3 GHz HFSS floor.",
            "final_signal_step_GHz": 0.01,
        },
        "evaluated_points": {
            "regional": len(stage1_rows),
            "local_refinement": len(stage2_rows),
            "fine_pump": len(stage3_rows),
            "dense_finalists": len(finalists),
        },
        "selected": best,
        "strict_pass": bool(best["strict_pass"]),
        "top_dense_finalists": finalists,
        "propagation_model_sensitivity": propagation_models,
        "mode_ablation": ablation,
        "bloch_extraction": extraction_provenance,
        "hfss_cell_fit_provenance": cell_provenance,
        "cautions": [
            "A result at 11.0 GHz is on the lowest pump boundary allowed by the measured idler band.",
            "Bloch-mode nonlinear overlap factors are not available from the current HFSS exports.",
            "The result remains a forward-mode coupled-envelope prediction, not a full nonlinear electromagnetic solve.",
        ],
    }
    (output / "06_assessment.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (row, gain) in enumerate(zip(finalists[:5], final_curves[:5]), 1):
        ax.plot(
            final_signals,
            gain,
            label=(
                f"{index}: {row['device_length_mm']:.1f} mm, "
                f"fp={row['pump_GHz']:.2f}, Idc={row['Idc_A'] * 1e6:.0f}, "
                f"Ip={row['Ip0_A'] * 1e6:.0f} µA"
            ),
        )
    ax.axhline(17.0, color="black", linestyle=":", label="17 dB minimum")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Eight-mode gain (dB)",
        title="Bloch-corrected 16-2-16 dense finalists",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "07_dense_finalists.png", dpi=180)
    fig.savefig(output / "07_dense_finalists.pdf")
    plt.close(fig)

    print(json.dumps({
        "selected": best,
        "strict_pass": bool(best["strict_pass"]),
        "propagation_models": propagation_models,
        "mode_ablation": ablation,
    }, indent=2))


if __name__ == "__main__":
    main()
