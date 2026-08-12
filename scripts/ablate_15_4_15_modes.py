#!/usr/bin/env python3
"""Mode ablation at the selected physical 15-4-15 operating point.

The experiment holds geometry, HFSS correction, length, pump, and currents
fixed. It tests each parasitic family alone and removes each family from the
full eight-mode configuration. Both ideal propagation and the complete general
model are reported so nonlinear coupling can be distinguished from the
loss/reflection treatment.
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


OPERATING_POINT = {
    "pattern": "15-4-15",
    "left_unloaded_cells": 15,
    "loaded_cells": 4,
    "right_unloaded_cells": 15,
    "cells_per_supercell": 34,
    "supercell_length_um": 68.0,
    "supercell_repeats": 900,
    "device_length_mm": 61.2,
    "pump_GHz": 11.2,
    "Idc_A": 350e-6,
    "Ip0_A": 290e-6,
    "Is0_A": 1.4e-6,
}


CONFIGURATIONS = {
    # Baseline and individual additions identify what each family can do alone.
    "basic_p_s_i": (0, 0, 0),
    "basic_plus_p2": (1, 0, 0),
    "basic_plus_ps_pi": (0, 1, 0),
    "basic_plus_s2_i2": (0, 0, 1),
    # Leave-one-family-out cases expose interactions that only occur when other
    # parasitic families are simultaneously present.
    "all_except_p2": (0, 1, 1),
    "all_except_ps_pi": (1, 0, 1),
    "all_except_s2_i2": (1, 1, 0),
    "all_8_modes": (1, 1, 1),
}


def summarize(frequency: np.ndarray, gain: np.ndarray) -> dict[str, object]:
    metrics = gain_metrics(frequency, gain, band=(4.0, 8.0))
    metrics["minimum_in_band_gain_db"] = float(np.min(gain))
    metrics["maximum_in_band_gain_db"] = float(np.max(gain))
    metrics["strict_pass"] = bool(
        metrics["minimum_in_band_gain_db"] >= 17.0
        and metrics["ripple_peak_to_peak_db"] <= 3.0
    )
    return metrics


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "step_23_15_4_15_mode_ablation"
    output.mkdir(parents=True, exist_ok=True)

    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    f1, s1 = read_touchstone(inputs / "supercell_15_4_15_fine_3_14GHz.s2p")
    f2, s2 = read_touchstone(inputs / "supercell_15_4_15_aux_14_27GHz.s2p")
    if not np.isclose(f1[-1], f2[0]) or not np.allclose(s1[-1], s2[0], atol=1e-12):
        raise ValueError("The 15-4-15 HFSS sweeps disagree at 14 GHz")
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

    lumped = cascade_cells(
        fitted_cell("unloaded"), fitted_cell("loaded"), 15, 4, 15
    )
    measured_phase, _ = phase_delay(measured_s[:, 1, 0], measured_f)
    lumped_phase, _ = phase_delay(lumped[:, 1, 0], measured_f)
    correction_samples = ((-measured_phase) - (-lumped_phase)) / 34

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 15),
            make_cell(cells["loaded"], 4),
            make_cell(cells["unloaded"], 15),
        ],
        N=900,
        Istar=2e-3,
        Idc=OPERATING_POINT["Idc_A"],
        Ip0=OPERATING_POINT["Ip0_A"],
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
    analysis.update_base_data()
    dense_f = np.asarray(analysis.data["freqs"])
    correction = np.zeros_like(dense_f)
    valid = (dense_f >= measured_f[0]) & (dense_f <= measured_f[-1])
    correction[valid] = np.interp(dense_f[valid], measured_f, correction_samples)
    analysis.data["k"] = np.asarray(analysis.data["k"]) + correction

    mode_details: dict[str, dict[str, object]] = {}
    for label, (pump_harmonics, conversions, signal_harmonics) in CONFIGURATIONS.items():
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
        mode_details[label] = {
            "mode_count": len(modes.modes),
            "included_modes": list(modes.modes),
            "p2_included": bool(pump_harmonics),
            "ps_pi_included": bool(conversions),
            "s2_i2_included": bool(signal_harmonics),
        }

    signals = np.arange(4.0, 8.0001, 0.05)
    models = ("general_ideal", "general")
    rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    for model in models:
        for configuration in CONFIGURATIONS:
            result = analysis.gain(
                signals,
                pump=OPERATING_POINT["pump_GHz"],
                Is0=OPERATING_POINT["Is0_A"],
                Ip0=None,
                model=model,
                mode_array_config=configuration,
                thin=300,
                save=False,
            )
            gain = np.asarray(result["gain_db"], dtype=float)
            if not np.all(np.isfinite(gain)):
                raise RuntimeError(f"Non-finite result for {model}/{configuration}")
            curves[f"{model}__{configuration}"] = gain
            rows.append(
                {
                    "model": model,
                    "configuration": configuration,
                    **mode_details[configuration],
                    **summarize(signals, gain),
                }
            )

    all_rows = {
        model: next(
            row
            for row in rows
            if row["model"] == model and row["configuration"] == "all_8_modes"
        )
        for model in models
    }
    for row in rows:
        full = all_rows[str(row["model"])]
        row["ripple_change_vs_all_8_modes_db"] = float(
            row["ripple_peak_to_peak_db"] - full["ripple_peak_to_peak_db"]
        )
        row["minimum_gain_change_vs_all_8_modes_db"] = float(
            row["minimum_in_band_gain_db"] - full["minimum_in_band_gain_db"]
        )

    with (output / "01_mode_ablation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "02_mode_ablation_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *curves])
        writer.writerows(zip(signals, *(curves[key] for key in curves)))

    def model_rows(model: str) -> dict[str, dict[str, object]]:
        return {
            str(row["configuration"]): row for row in rows if row["model"] == model
        }

    complete = model_rows("general")
    removal_effects = {
        "remove_p2": {
            "ripple_reduction_db": float(
                complete["all_8_modes"]["ripple_peak_to_peak_db"]
                - complete["all_except_p2"]["ripple_peak_to_peak_db"]
            ),
            "minimum_gain_change_db": float(
                complete["all_except_p2"]["minimum_in_band_gain_db"]
                - complete["all_8_modes"]["minimum_in_band_gain_db"]
            ),
        },
        "remove_ps_pi": {
            "ripple_reduction_db": float(
                complete["all_8_modes"]["ripple_peak_to_peak_db"]
                - complete["all_except_ps_pi"]["ripple_peak_to_peak_db"]
            ),
            "minimum_gain_change_db": float(
                complete["all_except_ps_pi"]["minimum_in_band_gain_db"]
                - complete["all_8_modes"]["minimum_in_band_gain_db"]
            ),
        },
        "remove_s2_i2": {
            "ripple_reduction_db": float(
                complete["all_8_modes"]["ripple_peak_to_peak_db"]
                - complete["all_except_s2_i2"]["ripple_peak_to_peak_db"]
            ),
            "minimum_gain_change_db": float(
                complete["all_except_s2_i2"]["minimum_in_band_gain_db"]
                - complete["all_8_modes"]["minimum_in_band_gain_db"]
            ),
        },
    }
    ranking = sorted(
        removal_effects,
        key=lambda key: removal_effects[key]["ripple_reduction_db"],
        reverse=True,
    )
    assessment = {
        "status": "15_4_15_selected_point_mode_ablation",
        "operating_point": OPERATING_POINT,
        "experiment": {
            "controlled_variables": "Same geometry, HFSS correction, length, pump, currents, signal grid, and solver settings for every case.",
            "individual_additions": [
                "basic_plus_p2",
                "basic_plus_ps_pi",
                "basic_plus_s2_i2",
            ],
            "leave_one_family_out": [
                "all_except_p2",
                "all_except_ps_pi",
                "all_except_s2_i2",
            ],
        },
        "complete_model_metrics": complete,
        "complete_model_removal_effects": removal_effects,
        "ranking_by_ripple_reduction_when_removed": ranking,
        "ideal_model_metrics": model_rows("general_ideal"),
        "hfss_cell_fit_provenance": provenance,
        "interpretation_caution": (
            "Mode families interact nonlinearly. A leave-one-out change measures "
            "that family's contribution in the presence of the others; it is not "
            "an additive decomposition of ripple."
        ),
    }
    (output / "03_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    addition_labels = (
        "basic_p_s_i",
        "basic_plus_p2",
        "basic_plus_ps_pi",
        "basic_plus_s2_i2",
        "all_8_modes",
    )
    removal_labels = (
        "all_8_modes",
        "all_except_p2",
        "all_except_ps_pi",
        "all_except_s2_i2",
    )
    for column, model in enumerate(models):
        for label in addition_labels:
            axes[0, column].plot(signals, curves[f"{model}__{label}"], label=label)
        axes[0, column].set_title(f"{model}: add each family")
        for label in removal_labels:
            axes[1, column].plot(signals, curves[f"{model}__{label}"], label=label)
        axes[1, column].set_title(f"{model}: remove one family from all modes")
    for axis in axes.flat:
        axis.axhline(17.0, color="black", linestyle=":", linewidth=1)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    axes[0, 0].set_ylabel("Gain (dB)")
    axes[1, 0].set_ylabel("Gain (dB)")
    axes[1, 0].set_xlabel("Signal frequency (GHz)")
    axes[1, 1].set_xlabel("Signal frequency (GHz)")
    fig.suptitle(
        "15-4-15 mode ablation at 11.2 GHz\n"
        "61.2 mm, Idc 350 µA, pump current 290 µA"
    )
    fig.tight_layout()
    fig.savefig(output / "04_mode_ablation.png", dpi=180)
    fig.savefig(output / "04_mode_ablation.pdf")
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
