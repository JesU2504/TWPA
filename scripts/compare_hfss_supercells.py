#!/usr/bin/env python3
"""Compare measured HFSS supercells over 3--27 GHz.

The verified 13-3-13 and 16-2-16 exports are required.  The paper-baseline
15-4-15 result is included automatically when both of its exports are present.
The script performs overlap, reciprocity, passivity, transmission, reflection,
phase, group-delay, and linear three-wave-mixing phase-mismatch checks.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "scripts")]

from analyze_hfss_supercell import read_touchstone  # noqa: E402
from hfss_bloch import extract_bloch_parameters  # noqa: E402


PATTERNS = {
    "13-3-13": {
        "fine": "supercell_13_3_13_fine_3_14GHz.s2p",
        "aux": "supercell_13_3_13_aux_14_27GHz.s2p",
        "cells": 29,
        "length_um": 58.0,
        "required": True,
    },
    "16-2-16": {
        "fine": "supercell_16_2_16_fine_3_14GHz.s2p",
        "aux": "supercell_16_2_16_aux_14_27GHz.s2p",
        "cells": 34,
        "length_um": 68.0,
        "required": True,
    },
    "15-4-15": {
        "fine": "supercell_15_4_15_fine_3_14GHz.s2p",
        "aux": "supercell_15_4_15_aux_14_27GHz.s2p",
        "cells": 34,
        "length_um": 68.0,
        "required": False,
    },
}

BANDS = {
    "signal_4_8_GHz": (4.0, 8.0),
    "pump_12p6_13p0_GHz": (12.6, 13.0),
    "sum_products_16p7_21p4_GHz": (16.7, 21.4),
    "second_pump_25_26_GHz": (25.0, 26.0),
}

PUMP_GHZ = 12.8
SIGNAL_GHZ = np.arange(4.0, 8.0001, 0.05)


def db(values: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def join_sweeps(fine_path: Path, aux_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    f1, s1 = read_touchstone(fine_path)
    f2, s2 = read_touchstone(aux_path)
    if not np.isclose(f1[-1], f2[0]):
        raise ValueError(f"Sweep endpoints do not overlap: {f1[-1]} vs {f2[0]} GHz")
    overlap_error = float(np.max(np.abs(s1[-1] - s2[0])))
    return np.concatenate((f1, f2[1:])), np.concatenate((s1, s2[1:]), axis=0), overlap_error


def metrics(f: np.ndarray, s: np.ndarray, length_um: float, overlap_error: float) -> dict[str, object]:
    s11, s21, s12, s22 = s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1]
    bloch_phase = extract_bloch_parameters(s).phase
    group_delay = np.gradient(bloch_phase, f * 1e9) / (2 * np.pi)
    singular_max = np.array([np.linalg.svd(matrix, compute_uv=False)[0] for matrix in s])
    result: dict[str, object] = {
        "overlap_max_complex_error": overlap_error,
        "reciprocity_max_complex_error": float(np.max(np.abs(s21 - s12))),
        "maximum_singular_value": float(np.max(singular_max)),
        "passivity_excess": float(max(0.0, np.max(singular_max) - 1.0)),
        "median_group_delay_ps": float(np.median(group_delay) * 1e12),
        "median_delay_ps_per_mm": float(np.median(group_delay) * 1e12 / (length_um / 1000.0)),
    }
    for name, (low, high) in BANDS.items():
        mask = (f >= low) & (f <= high)
        result[name] = {
            "S21_min_dB": float(np.min(db(s21[mask]))),
            "S21_max_dB": float(np.max(db(s21[mask]))),
            "S11_max_dB": float(np.max(db(s11[mask]))),
            "S22_max_dB": float(np.max(db(s22[mask]))),
        }
    return result


def main() -> None:
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "bloch_corrected" / "step_21_hfss_pattern_comparison"
    output.mkdir(parents=True, exist_ok=True)

    missing_required = [
        inputs / str(config[key])
        for config in PATTERNS.values()
        if bool(config["required"])
        for key in ("fine", "aux")
        if not (inputs / str(config[key])).exists()
    ]
    if missing_required:
        names = "\n".join(f"  - {path}" for path in missing_required)
        raise SystemExit(f"Missing HFSS export(s):\n{names}\nExport/copy them and rerun this script.")

    available_patterns = {
        label: config
        for label, config in PATTERNS.items()
        if all((inputs / str(config[key])).exists() for key in ("fine", "aux"))
    }
    skipped_patterns = [label for label in PATTERNS if label not in available_patterns]

    data: dict[str, dict[str, object]] = {}
    assessment: dict[str, object] = {
        "status": "physical_HFSS_supercell_comparison",
        "bands": BANDS,
        "available_patterns": list(available_patterns),
        "skipped_patterns_missing_exports": skipped_patterns,
        "patterns": {},
    }
    for label, config in available_patterns.items():
        f, s, overlap = join_sweeps(inputs / str(config["fine"]), inputs / str(config["aux"]))
        phase = extract_bloch_parameters(s).phase
        delay = np.gradient(phase, f * 1e9) / (2 * np.pi)
        data[label] = {"f": f, "s": s, "phase": phase, "delay": delay}
        assessment["patterns"][label] = metrics(f, s, float(config["length_um"]), overlap)

        k_per_mm = phase / (float(config["length_um"]) / 1000.0)
        signal_k = np.interp(SIGNAL_GHZ, f, k_per_mm)
        idler_k = np.interp(PUMP_GHZ - SIGNAL_GHZ, f, k_per_mm)
        pump_k = float(np.interp(PUMP_GHZ, f, k_per_mm))
        mismatch = pump_k - signal_k - idler_k
        data[label]["phase_mismatch"] = mismatch
        assessment["patterns"][label]["linear_3wm_phase_mismatch_at_12p8_GHz"] = {
            "definition": "k_p-k_s-k_i in rad/mm, with f_i=f_p-f_s",
            "minimum_rad_per_mm": float(np.min(mismatch)),
            "maximum_rad_per_mm": float(np.max(mismatch)),
            "mean_absolute_rad_per_mm": float(np.mean(np.abs(mismatch))),
            "peak_to_peak_rad_per_mm": float(np.ptp(mismatch)),
        }

    (output / "01_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")

    common_f = np.arange(3.0, 27.0001, 0.05)
    rows: list[list[float]] = []
    interpolated: dict[str, dict[str, np.ndarray]] = {}
    for label, config in available_patterns.items():
        item = data[label]
        f, s = np.asarray(item["f"]), np.asarray(item["s"])
        phase, delay = np.asarray(item["phase"]), np.asarray(item["delay"])
        interpolated[label] = {
            "s11": np.interp(common_f, f, db(s[:, 0, 0])),
            "s21": np.interp(common_f, f, db(s[:, 1, 0])),
            "phase_per_mm": np.interp(common_f, f, phase) / (float(config["length_um"]) / 1000.0),
            "delay_per_mm": np.interp(common_f, f, delay * 1e12) / (float(config["length_um"]) / 1000.0),
        }
    csv_fields = ["frequency_GHz"]
    for label in available_patterns:
        csv_fields.extend(
            [
                f"S11_{label.replace('-', '_')}_dB",
                f"S21_{label.replace('-', '_')}_dB",
                f"bloch_phase_{label.replace('-', '_')}_rad_per_mm",
                f"group_delay_{label.replace('-', '_')}_ps_per_mm",
            ]
        )
    for index, frequency in enumerate(common_f):
        row = [frequency]
        for label in available_patterns:
            row.extend(
                [
                    interpolated[label]["s11"][index],
                    interpolated[label]["s21"][index],
                    interpolated[label]["phase_per_mm"][index],
                    interpolated[label]["delay_per_mm"][index],
                ]
            )
        rows.append(row)
    with (output / "02_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_fields)
        writer.writerows(rows)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12.5))
    colors = {"13-3-13": "#1f77b4", "16-2-16": "#d62728", "15-4-15": "#2ca02c"}
    for label in available_patterns:
        axes[0].plot(common_f, interpolated[label]["s21"], label=label, color=colors[label])
        axes[1].plot(common_f, interpolated[label]["s11"], label=label, color=colors[label])
        axes[2].plot(common_f, interpolated[label]["delay_per_mm"], label=label, color=colors[label])
        axes[3].plot(
            SIGNAL_GHZ,
            np.asarray(data[label]["phase_mismatch"]),
            label=label,
            color=colors[label],
        )
    shades = [(4, 8, "signal"), (12.6, 13, "pump"), (16.7, 21.4, "ps/pi"), (25, 26, "p2")]
    for axis in axes[:3]:
        for low, high, name in shades:
            axis.axvspan(low, high, alpha=0.08, label=name if axis is axes[0] else None)
        axis.grid(True, alpha=0.3)
    axes[3].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[3].grid(True, alpha=0.3)
    axes[0].set_ylabel("S21 (dB)")
    axes[1].set_ylabel("S11 (dB)")
    axes[2].set_ylabel("Group delay (ps/mm)")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[3].set_ylabel("Linear 3WM mismatch (rad/mm)")
    axes[3].set_xlabel("Signal frequency (GHz), pump = 12.8 GHz")
    axes[3].legend(fontsize=8)
    axes[0].set_title("Physical HFSS supercell comparison")
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "03_hfss_pattern_comparison.png", dpi=180)
    fig.savefig(output / "03_hfss_pattern_comparison.pdf")
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
