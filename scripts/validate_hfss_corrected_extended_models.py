#!/usr/bin/env python3
"""Compare minimal and extended nonlinear models with HFSS Bloch parameters."""

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
    read_touchstone,
)
from hfss_bloch import apply_bloch_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402


SELECTED = {"pump_GHz": 12.94, "Idc_A": 315e-6, "Ip0_A": 310e-6}
OUTPUT_STEP = "step_15_extended_nonlinear_validation"
NEARBY_PUMPS_GHZ = (12.93, 12.94, 12.95)
MODELS = (
    ("minimal_3wm", "basic_3wm"),
    ("general_ideal", "hfss_extended"),
    ("general_loss_only", "hfss_extended"),
    ("general", "hfss_extended"),
)


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "bloch_corrected" / OUTPUT_STEP
    output.mkdir(parents=True, exist_ok=True)

    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    fine_f, fine_s = read_touchstone(
        inputs / "supercell_13_3_13_fine_3_14GHz.s2p"
    )
    auxiliary_f, auxiliary_s = read_touchstone(
        inputs / "supercell_13_3_13_aux_14_27GHz.s2p"
    )
    if not np.isclose(fine_f[-1], auxiliary_f[0]):
        raise ValueError("The fine and auxiliary HFSS sweeps do not overlap.")
    if not np.allclose(fine_s[-1], auxiliary_s[0], rtol=1e-10, atol=1e-12):
        raise ValueError("The fine and auxiliary HFSS sweeps disagree at 14 GHz.")
    # Keep the shared 14 GHz point only once.  The matching complex samples make
    # it safe to unwrap S21 phase continuously across the two sweeps.
    measured_f = np.concatenate((fine_f, auxiliary_f[1:]))
    measured_s = np.concatenate((fine_s, auxiliary_s[1:]), axis=0)

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 13),
            make_cell(cells["loaded"], 3),
            make_cell(cells["unloaded"], 13),
        ],
        N=int(SELECTED.get("supercell_repeats", 1407)),
        Istar=2e-3,
        Idc=SELECTED["Idc_A"],
        Ip0=SELECTED["Ip0_A"],
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz")
    analysis.update_base_data()

    apply_bloch_parameters(
        analysis.data, measured_f, measured_s, SUPERCELL_CELLS
    )
    analysis.add_mode_array("basic_3wm", ModeArrayFactory.create_basic(analysis.data))

    # Eight forward modes: pump, signal, idler, sum-frequency conversions,
    # second pump harmonic, and second signal/idler harmonics.
    extended = ModeArrayFactory.create_extended_3wm(
        analysis.data,
        n_pump_harmonics=1,
        n_frequency_conversion=1,
        n_signal_harmonics=1,
    )
    analysis.add_mode_array("hfss_extended", extended)

    signals = np.arange(4.0, 8.0001, 0.1)
    curves: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    def solve(model: str, mode_array: str, pump: float) -> tuple[np.ndarray, dict[str, object]]:
        result = analysis.gain(
            signals,
            pump=pump,
            Is0=1.4e-6,
            Ip0=None,
            model=model,
            mode_array_config=mode_array,
            thin=300,
            save=False,
        )
        gain = np.asarray(result["gain_db"], dtype=float)
        if model == "minimal_3wm":
            pump_index = 0
        else:
            pump_index = int(result["mode_info"]["mode_map"]["p"])
        currents = np.asarray(result["I_triplets"])
        pump_exit_fraction = np.abs(
            currents[:, pump_index, -1] / SELECTED["Ip0_A"]
        ) ** 2
        metrics: dict[str, object] = {
            **gain_metrics(signals, gain, band=(4.0, 8.0)),
            "minimum_in_band_gain_db": float(np.min(gain)),
            "maximum_in_band_gain_db": float(np.max(gain)),
            "pump_exit_power_fraction_median": float(np.median(pump_exit_fraction)),
            "pump_exit_power_fraction_min": float(np.min(pump_exit_fraction)),
            "pump_exit_power_fraction_max": float(np.max(pump_exit_fraction)),
        }
        return gain, metrics

    for model, mode_array in MODELS:
        gain, metrics = solve(model, mode_array, SELECTED["pump_GHz"])
        curves[f"selected__{model}"] = gain
        rows.append(
            {
                "case": "selected",
                "pump_GHz": SELECTED["pump_GHz"],
                "Idc_A": SELECTED["Idc_A"],
                "Ip0_A": SELECTED["Ip0_A"],
                "model": model,
                **metrics,
            }
        )

    # A small pump-frequency check with the most complete available model.
    for pump in NEARBY_PUMPS_GHZ:
        gain, metrics = solve("general", "hfss_extended", pump)
        curves[f"general__pump_{pump:.2f}"] = gain
        if pump != SELECTED["pump_GHz"]:
            rows.append(
                {
                    "case": "nearby_pump",
                    "pump_GHz": pump,
                    "Idc_A": SELECTED["Idc_A"],
                    "Ip0_A": SELECTED["Ip0_A"],
                    "model": "general",
                    **metrics,
                }
            )

    # Mode ablation: identify which extra frequency family causes the smooth
    # three-frequency result to break down.  All cases are ideal here so the
    # comparison isolates nonlinear mode coupling from loss/reflection effects.
    ablation_specs = {
        "basic_3_modes": ModeArrayFactory.create_basic(analysis.data),
        "add_pump_harmonic": ModeArrayFactory.create_extended_3wm(
            analysis.data, n_pump_harmonics=1, n_frequency_conversion=0,
            n_signal_harmonics=0,
        ),
        "add_sum_conversions": ModeArrayFactory.create_extended_3wm(
            analysis.data, n_pump_harmonics=0, n_frequency_conversion=1,
            n_signal_harmonics=0,
        ),
        "add_signal_idler_harmonics": ModeArrayFactory.create_extended_3wm(
            analysis.data, n_pump_harmonics=0, n_frequency_conversion=0,
            n_signal_harmonics=1,
        ),
        "all_8_modes": extended,
    }
    ablation_rows: list[dict[str, object]] = []
    ablation_curves: dict[str, np.ndarray] = {}
    for label, mode_array_object in ablation_specs.items():
        config_name = f"ablation_{label}"
        analysis.add_mode_array(config_name, mode_array_object)
        gain, metrics = solve("general_ideal", config_name, SELECTED["pump_GHz"])
        ablation_curves[label] = gain
        ablation_rows.append(
            {
                "configuration": label,
                "mode_count": len(mode_array_object.modes),
                "included_modes": "+".join(mode_array_object.modes),
                **metrics,
            }
        )

    with (output / "01_extended_model_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "01_extended_model_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *curves])
        writer.writerows(zip(signals, *(curves[key] for key in curves)))
    with (output / "01_mode_ablation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    with (output / "01_mode_ablation_curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", *ablation_curves])
        writer.writerows(
            zip(signals, *(ablation_curves[key] for key in ablation_curves))
        )

    selected_rows = [row for row in rows if row["case"] == "selected"]
    general_selected = next(
        row for row in selected_rows if row["model"] == "general"
    )
    assessment = {
        "status": "extended_model_with_fine_3_to_27_GHz_HFSS_phase_correction",
        "operating_point": SELECTED,
        "mode_count": len(extended.modes),
        "included_modes": list(extended.modes),
        "models": {str(row["model"]): row for row in selected_rows},
        "nearby_general_model": [
            row for row in rows if row["model"] == "general"
        ],
        "ideal_mode_ablation": {
            str(row["configuration"]): row for row in ablation_rows
        },
        "general_model_nominal_pass": bool(
            float(general_selected["minimum_in_band_gain_db"]) >= 17.0
            and float(general_selected["ripple_peak_to_peak_db"]) <= 3.0
        ),
        "limitations": [
            "HFSS Bloch extraction uses 0.1 GHz samples from 3-14 GHz and 0.2 GHz samples from 14-27 GHz.",
            "Only one order of pump, signal/idler harmonics and conversion modes is included.",
            "The nonlinear model still represents the full line by repeating the simulated 13-3-13 physical supercell.",
            "Each signal frequency is solved as a separate single-tone experiment.",
        ],
        "hfss_cell_provenance": provenance,
    }
    (output / "02_extended_model_assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    ax = axes[0]
    labels = {
        "minimal_3wm": "3 frequencies",
        "general_ideal": "8 modes, ideal",
        "general_loss_only": "8 modes + loss",
        "general": "8 modes + loss/reflection",
    }
    for model, _ in MODELS:
        ax.plot(signals, curves[f"selected__{model}"], label=labels[model])
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Predicted gain (dB)",
        title="What changes as more physics is included",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for pump in NEARBY_PUMPS_GHZ:
        ax.plot(
            signals,
            curves[f"general__pump_{pump:.2f}"],
            label=f"pump {pump:.2f} GHz",
        )
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(
        xlabel="Signal frequency (GHz)",
        title="Complete-model sensitivity to pump frequency",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "HFSS-corrected extended nonlinear validation\n"
        "Idc 315 µA, pump current 310 µA",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / "03_extended_nonlinear_validation.png", dpi=180)
    fig.savefig(output / "03_extended_nonlinear_validation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.3))
    for label, gain in ablation_curves.items():
        ax.plot(signals, gain, label=label.replace("_", " "))
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Predicted gain (dB)",
        title="Ideal-model mode ablation at 12.94 GHz",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "04_mode_ablation.png", dpi=180)
    fig.savefig(output / "04_mode_ablation.pdf")
    plt.close(fig)

    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
