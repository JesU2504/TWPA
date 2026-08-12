#!/usr/bin/env python3
"""Diagnose the s2/i2 contribution to ripple at the selected 15-4-15 point.

This is a controlled follow-up to ``ablate_15_4_15_modes.py``.  It separates
the signal and idler second harmonics, computes the linear phase mismatch of
the direct and cross-coupling processes present in the eight-mode model, and
checks that the reported gain does not depend on the requested output-position
resolution (``thin``).  The ODE solver itself already uses rtol=1e-10 and
atol=1e-14 in twpasolver.
"""

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

from analyze_hfss_supercell import phase_delay, read_touchstone  # noqa: E402
from fit_hfss_unit_cells import lclf_s  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from scan_physical_hfss_pattern import cascade_cells  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


PUMP = 11.2
IDC = 350e-6
IP0 = 290e-6
IS0 = 1.4e-6
REPEATS = 900
CELL_LENGTH_MM = 0.002


MODE_CONFIGS = {
    "basic": (
        ["p", "s", "i"],
        [["i", "p-s"]],
    ),
    "basic_plus_s2": (
        ["p", "s", "i", "s2"],
        [["i", "p-s"], ["s2", "s+s"]],
    ),
    "basic_plus_i2": (
        ["p", "s", "i", "i2"],
        [["i", "p-s"], ["i2", "i+i"]],
    ),
    "all_except_s2_i2": (
        ["p", "s", "i", "ps", "pi", "p2"],
        [["i", "p-s"], ["ps", "p+s"], ["pi", "p+i"], ["p2", "p+p"]],
    ),
    "all_except_i2": (
        ["p", "s", "i", "ps", "pi", "p2", "s2"],
        [
            ["i", "p-s"],
            ["ps", "p+s"],
            ["pi", "p+i"],
            ["p2", "p+p"],
            ["s2", "s+s"],
        ],
    ),
    "all_except_s2": (
        ["p", "s", "i", "ps", "pi", "p2", "i2"],
        [
            ["i", "p-s"],
            ["ps", "p+s"],
            ["pi", "p+i"],
            ["p2", "p+p"],
            ["i2", "i+i"],
        ],
    ),
    "all_8_modes": (
        ["p", "s", "i", "ps", "pi", "p2", "s2", "i2"],
        [
            ["i", "p-s"],
            ["ps", "p+s"],
            ["pi", "p+i"],
            ["p2", "p+p"],
            ["s2", "s+s"],
            ["i2", "i+i"],
        ],
    ),
}


def build_analysis() -> tuple[TWPAnalysis, dict[str, object]]:
    inputs = ROOT / "hfss_inputs"
    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    f1, s1 = read_touchstone(inputs / "supercell_15_4_15_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / "supercell_15_4_15_aux_14_27GHz.s2p")
    if not np.isclose(f1[-1], f2[0]) or not np.allclose(s1[-1], s2[0], atol=1e-12):
        raise ValueError("The 15-4-15 HFSS sweeps disagree at 14 GHz")
    measured_f = np.concatenate((f1, f2[1:]))
    measured_s = np.concatenate((s1, s2[1:]), axis=0)

    def fitted(role: str) -> np.ndarray:
        row = cells[role]
        return lclf_s(
            measured_f * 1e9,
            row["L_pH"] * 1e-12,
            row["C_fF"] * 1e-15,
            row["Lf_pH"] * 1e-12,
        )

    lumped = cascade_cells(fitted("unloaded"), fitted("loaded"), 15, 4, 15)
    measured_phase, _ = phase_delay(measured_s[:, 1, 0], measured_f)
    lumped_phase, _ = phase_delay(lumped[:, 1, 0], measured_f)
    correction_samples = ((-measured_phase) - (-lumped_phase)) / 34

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 15),
            make_cell(cells["loaded"], 4),
            make_cell(cells["unloaded"], 15),
        ],
        N=REPEATS,
        Istar=2e-3,
        Idc=IDC,
        Ip0=IP0,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
    analysis.update_base_data()
    dense_f = np.asarray(analysis.data["freqs"])
    correction = np.zeros_like(dense_f)
    valid = (dense_f >= measured_f[0]) & (dense_f <= measured_f[-1])
    correction[valid] = np.interp(dense_f[valid], measured_f, correction_samples)
    analysis.data["k"] = np.asarray(analysis.data["k"]) + correction
    return analysis, provenance


def phase_mismatches(analysis: TWPAnalysis, signal: np.ndarray) -> dict[str, np.ndarray]:
    base_f = np.asarray(analysis.data["freqs"], dtype=float)
    base_k = np.asarray(analysis.data["k"], dtype=float)

    def k(freq: np.ndarray | float) -> np.ndarray:
        return np.interp(freq, base_f, base_k)

    s = np.asarray(signal)
    i = PUMP - s
    ps = PUMP + s
    pi = PUMP + i
    p2 = 2 * PUMP
    s2 = 2 * s
    i2 = 2 * i
    # Values are converted from rad per 2 um physical cell to rad/mm.
    scale = 1.0 / CELL_LENGTH_MM
    return {
        "p_minus_s_i": (k(PUMP) - k(s) - k(i)) * scale,
        "p_s_minus_ps": (k(PUMP) + k(s) - k(ps)) * scale,
        "p_i_minus_pi": (k(PUMP) + k(i) - k(pi)) * scale,
        "2p_minus_p2": np.full_like(s, (2 * k(PUMP) - k(p2)) * scale),
        "2s_minus_s2": (2 * k(s) - k(s2)) * scale,
        "2i_minus_i2": (2 * k(i) - k(i2)) * scale,
        "ps_minus_i_s2": (k(ps) - k(i) - k(s2)) * scale,
        "pi_minus_s_i2": (k(pi) - k(s) - k(i2)) * scale,
        "p2_minus_s2_i2": (k(p2) - k(s2) - k(i2)) * scale,
    }


def metrics(signal: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    row = gain_metrics(signal, gain, band=(4.0, 8.0))
    row["minimum_in_band_gain_db"] = float(np.min(gain))
    row["maximum_in_band_gain_db"] = float(np.max(gain))
    row["minimum_gain_frequency_GHz"] = float(signal[np.argmin(gain)])
    return row


def main() -> None:
    log.setLevel(logging.WARNING)
    output = ROOT / "results" / "step_24_15_4_15_harmonic_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    analysis, provenance = build_analysis()

    for name, (labels, relations) in MODE_CONFIGS.items():
        modes = ModeArrayFactory.create_custom(
            analysis.data,
            mode_labels=labels,
            mode_directions=[1] * len(labels),
            relations=relations,
        )
        analysis.add_mode_array(name, modes)

    signal = np.arange(4.0, 8.0001, 0.05)
    curves: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for name, (labels, _) in MODE_CONFIGS.items():
        result = analysis.gain(
            signal,
            pump=PUMP,
            Is0=IS0,
            model="general",
            mode_array_config=name,
            thin=300,
            save=False,
        )
        gain = np.asarray(result["gain_db"], dtype=float)
        if not np.all(np.isfinite(gain)):
            raise RuntimeError(f"Non-finite result for {name}")
        curves[name] = gain
        rows.append({"configuration": name, "included_modes": labels, **metrics(signal, gain)})

    # ``thin`` changes only returned evaluation positions. Agreement confirms
    # that the final endpoint and reported gain are independent of that output grid.
    resolution_curves: dict[str, np.ndarray] = {}
    for thin in (150, 300, 600):
        result = analysis.gain(
            signal,
            pump=PUMP,
            Is0=IS0,
            model="general",
            mode_array_config="all_8_modes",
            thin=thin,
            save=False,
        )
        resolution_curves[str(thin)] = np.asarray(result["gain_db"], dtype=float)
    reference = resolution_curves["300"]
    resolution_differences = {
        key: float(np.max(np.abs(value - reference)))
        for key, value in resolution_curves.items()
    }

    mismatch = phase_mismatches(analysis, signal)
    full = next(row for row in rows if row["configuration"] == "all_8_modes")
    min_index = int(np.argmin(curves["all_8_modes"]))
    at_main_dip = {key: float(value[min_index]) for key, value in mismatch.items()}
    nearest_phase_match = {
        key: {
            "signal_GHz": float(signal[np.argmin(np.abs(value))]),
            "absolute_mismatch_rad_per_mm": float(np.min(np.abs(value))),
        }
        for key, value in mismatch.items()
    }

    baseline = curves["all_except_s2_i2"]
    family_effects = {
        "add_s2_only_to_complete_without_harmonics": {
            "maximum_absolute_gain_change_db": float(
                np.max(np.abs(curves["all_except_i2"] - baseline))
            ),
            "gain_change_at_full_model_dip_db": float(
                curves["all_except_i2"][min_index] - baseline[min_index]
            ),
        },
        "add_i2_only_to_complete_without_harmonics": {
            "maximum_absolute_gain_change_db": float(
                np.max(np.abs(curves["all_except_s2"] - baseline))
            ),
            "gain_change_at_full_model_dip_db": float(
                curves["all_except_s2"][min_index] - baseline[min_index]
            ),
        },
    }

    with (output / "01_split_mode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "02_split_mode_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *curves])
        writer.writerows(zip(signal, *(curves[key] for key in curves)))
    with (output / "03_phase_mismatch.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *mismatch])
        writer.writerows(zip(signal, *(mismatch[key] for key in mismatch)))

    assessment = {
        "status": "15_4_15_s2_i2_split_and_phase_mismatch_diagnosis",
        "operating_point": {
            "pump_GHz": PUMP,
            "device_length_mm": 61.2,
            "supercell_repeats": REPEATS,
            "Idc_A": IDC,
            "Ip0_A": IP0,
            "Is0_A": IS0,
        },
        "full_model_metrics": full,
        "split_mode_metrics": {row["configuration"]: row for row in rows},
        "individual_harmonic_effects_relative_to_no_s2_i2": family_effects,
        "linear_phase_mismatch_at_full_model_main_dip_rad_per_mm": at_main_dip,
        "nearest_linear_phase_match_in_4_to_8_GHz": nearest_phase_match,
        "output_resolution_max_gain_difference_db": resolution_differences,
        "solver_tolerances_from_installed_twpasolver": {"rtol": 1e-10, "atol": 1e-14},
        "hfss_cell_fit_provenance": provenance,
        "cautions": [
            "Linear phase mismatch alone does not include pump-induced nonlinear phase shifts.",
            "Mode-family effects are nonlinear and therefore are not additive.",
            "The output-resolution check is not an independent ODE-tolerance implementation.",
        ],
    }
    (output / "04_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    ax = axes[0, 0]
    for name in ("all_8_modes", "all_except_s2_i2", "all_except_i2", "all_except_s2"):
        ax.plot(signal, curves[name], label=name)
    ax.axhline(17, color="black", linestyle=":", linewidth=1)
    ax.set(title="Separate s2 and i2 in the complete model", ylabel="Gain (dB)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name in ("basic", "basic_plus_s2", "basic_plus_i2"):
        ax.plot(signal, curves[name], label=name)
    ax.set(title="Each second harmonic added to basic p/s/i", ylabel="Gain (dB)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name in ("2s_minus_s2", "2i_minus_i2", "ps_minus_i_s2", "pi_minus_s_i2"):
        ax.plot(signal, mismatch[name], label=name)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.axvline(signal[min_index], color="grey", linestyle="--", linewidth=1)
    ax.set(
        title="Second-harmonic linear phase mismatches",
        xlabel="Signal frequency (GHz)",
        ylabel="Mismatch (rad/mm)",
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(signal, curves["all_8_modes"], color="black", label="all_8_modes gain")
    ax2 = ax.twinx()
    ax2.plot(signal, np.abs(mismatch["2s_minus_s2"]), label="|2s-s2|", alpha=0.8)
    ax2.plot(signal, np.abs(mismatch["2i_minus_i2"]), label="|2i-i2|", alpha=0.8)
    ax2.plot(signal, np.abs(mismatch["ps_minus_i_s2"]), label="|ps-i-s2|", alpha=0.8)
    ax2.plot(signal, np.abs(mismatch["pi_minus_s_i2"]), label="|pi-s-i2|", alpha=0.8)
    ax.set(title="Gain versus harmonic mismatch", xlabel="Signal frequency (GHz)", ylabel="Gain (dB)")
    ax2.set_ylabel("Absolute mismatch (rad/mm)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle("15-4-15 harmonic ripple diagnosis at 11.2 GHz")
    fig.tight_layout()
    fig.savefig(output / "05_harmonic_diagnosis.png", dpi=180)
    fig.savefig(output / "05_harmonic_diagnosis.pdf")
    plt.close(fig)

    print(json.dumps({
        "full_model": full,
        "family_effects": family_effects,
        "at_main_dip": at_main_dip,
        "resolution_differences": resolution_differences,
    }, indent=2))


if __name__ == "__main__":
    main()
