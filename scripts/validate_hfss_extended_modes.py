#!/usr/bin/env python3
"""Validate candidate operating points with extended coupled-mode models."""

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
sys.path.insert(0, str(ROOT / "code"))

from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402
from twpasolver.modes_rwa import ModeArrayFactory  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402


log.setLevel(logging.WARNING)


CANDIDATES = {
    "strict_ripple": {"pump_GHz": 13.13, "Idc_A": 350e-6, "Ip0_A": 340e-6},
    "moderate_gain": {"pump_GHz": 13.11, "Idc_A": 325e-6, "Ip0_A": 340e-6},
}


def main() -> None:
    cells, provenance = load_hfss_cell_parameters(
        ROOT / "hfss_inputs" / "hfss_cell_parameters.csv"
    )
    signals = np.arange(4.0, 8.0001, 0.05)
    curves: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}

    for label, candidate in CANDIDATES.items():
        twpa = TWPA(
            cells=[
                make_cell(cells["unloaded"], 13),
                make_cell(cells["loaded"], 3),
                make_cell(cells["unloaded"], 13),
            ],
            N=1407,
            Istar=2e-3,
            Idc=candidate["Idc_A"],
            Ip0=candidate["Ip0_A"],
        )
        analysis = TWPAnalysis(
            twpa=twpa, f_arange=(0.2, 30.0, 0.002), unit="GHz"
        )
        analysis.update_base_data()
        extended = ModeArrayFactory.create_extended_3wm(
            analysis.data,
            n_pump_harmonics=1,
            n_frequency_conversion=1,
            n_signal_harmonics=1,
        )
        analysis.add_mode_array("hfss_extended", extended)
        metrics[label] = {}
        model_specs = (
            ("minimal_3wm", "basic_3wm"),
            ("general_ideal", "hfss_extended"),
            ("general_loss_only", "hfss_extended"),
            ("general", "hfss_extended"),
        )
        for model, mode_config in model_specs:
            result = analysis.gain(
                signals,
                pump=candidate["pump_GHz"],
                Is0=1.4e-6,
                Ip0=None,
                model=model,
                mode_array_config=mode_config,
                thin=300,
                save=False,
            )
            gain = np.asarray(result["gain_db"], dtype=float)
            key = f"{label}__{model}"
            curves[key] = gain
            metrics[label][model] = gain_metrics(signals, gain, band=(4.0, 8.0))

    output = ROOT / "results" / "step_12_hfss_operating_point_optimization"
    with (output / "extended_mode_validation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_GHz", *curves])
        writer.writerows(
            zip(signals, *(curves[name] for name in curves))
        )
    report = {
        "pattern": {
            "unloaded_cells_each_side": 13,
            "loaded_cells": 3,
            "supercells": 1407,
        },
        "candidates": CANDIDATES,
        "extended_mode_limits": {
            "n_pump_harmonics": 1,
            "n_frequency_conversion": 1,
            "n_signal_harmonics": 1,
            "reason": "Keeps the extended modes within the 0.2-30 GHz HFSS fit range.",
        },
        "metrics": metrics,
        "hfss_provenance": provenance,
    }
    (output / "extended_mode_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, label in zip(axes, CANDIDATES):
        for model in ("minimal_3wm", "general_ideal", "general_loss_only", "general"):
            ax.plot(signals, curves[f"{label}__{model}"], label=model)
        ax.axhline(17, color="k", linestyle="--", alpha=0.6)
        ax.set(title=label.replace("_", " "), xlabel="Signal frequency (GHz)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Gain (dB)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "extended_mode_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
