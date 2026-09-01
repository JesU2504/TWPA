#!/usr/bin/env python3
"""L/C/Lf extraction from two-port HFSS cell sweeps.

The 12.0 µm unloaded and 3.9 µm loaded fits define the baseline structural model.
Output: hfss_inputs/hfss_cell_parameters.csv and its fit-quality report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
import numpy as np

CELL_DEFINITIONS: dict[str, dict[str, float | str]] = {
    "unloaded": {
        "stub_length_um": 12.0,
        "broadband_file": "kinetic_12um_debug.s2p",
        "lowfreq_file": "kinetic_12um_lowfreq.s2p",
    },
    "loaded": {
        "stub_length_um": 3.9,
        "broadband_file": "kinetic_3p9um_debug.s2p",
        "lowfreq_file": "kinetic_3p9um_lowfreq.s2p",
    },
}
FIT_BAND_GHZ = (4.0, 14.0)
OUTPUT_COLUMNS = (
    "cell_type",
    "stub_length_um",
    "L_pH",
    "C_fF",
    "Lf_pH",
    "loss_tangent",
    "Z0_ohm",
    "fit_s21_rmse_db",
    "fit_phase_rmse_deg",
    "source",
)

# Acceptance gates: Z0 of the previously selected HFSS baseline fit.
_SELECTED_UNLOADED_Z0_OHM = 43.08925347829123
_SELECTED_LOADED_Z0_OHM = 70.20100727735449


def read_touchstone_ma(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """GHz Touchstone data as frequencies in Hz and S matrices of shape (n, 2, 2)."""
    frequencies_hz: list[float] = []
    s_matrices: list[np.ndarray] = []
    option_line: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            option_line = line.upper()
            continue
        values = [float(item) for item in line.split()[:9]]
        if len(values) != 9:
            continue
        pairs = [
            values[index] * np.exp(1j * np.deg2rad(values[index + 1])) for index in (1, 3, 5, 7)
        ]
        # Touchstone two-port ordering is S11, S21, S12, S22.
        s_matrices.append(np.array([[pairs[0], pairs[2]], [pairs[1], pairs[3]]]))
        frequencies_hz.append(values[0] * 1e9)
    if (
        option_line is None
        or "GHZ" not in option_line
        or " S " not in option_line
        or " MA " not in option_line
    ):
        raise ValueError(f"{path} must use '# GHz S MA R 50'")
    if "R 50" not in option_line:
        raise ValueError(f"{path} must be normalized to 50 ohms")
    return np.asarray(frequencies_hz), np.asarray(s_matrices)


def s_to_abcd(s_matrices: np.ndarray, z0_ohm: float = 50.0) -> np.ndarray:
    """ABCD matrices for two-port S parameters at reference impedance z0_ohm."""
    s11, s12 = s_matrices[:, 0, 0], s_matrices[:, 0, 1]
    s21, s22 = s_matrices[:, 1, 0], s_matrices[:, 1, 1]
    abcd = np.empty_like(s_matrices)
    abcd[:, 0, 0] = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    abcd[:, 0, 1] = z0_ohm * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    abcd[:, 1, 0] = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * s21 * z0_ohm)
    abcd[:, 1, 1] = ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return abcd


def lclf_cell_s_parameters(
    frequencies_hz: np.ndarray,
    inductance_h: float,
    capacitance_f: float,
    finger_inductance_h: float,
    z0_ohm: float = 50.0,
) -> np.ndarray:
    """Uncentered LCLfBaseCell S parameters with shape (n_frequencies, 2, 2).

    Frequency, inductance, capacitance, and reference impedance use SI units."""
    angular_freq = 2 * np.pi * frequencies_hz
    denominator = 2 - finger_inductance_h * capacitance_f * angular_freq**2
    a = np.ones_like(angular_freq, dtype=complex)
    b = 1j * inductance_h * angular_freq
    c = 1j * 2 * capacitance_f * angular_freq / denominator
    d = 1 - 2 * inductance_h * capacitance_f * angular_freq**2 / denominator
    normalization = a + b / z0_ohm + c * z0_ohm + d
    s_matrices = np.empty((len(frequencies_hz), 2, 2), dtype=complex)
    s_matrices[:, 0, 0] = (a + b / z0_ohm - c * z0_ohm - d) / normalization
    s_matrices[:, 0, 1] = 2 * (a * d - b * c) / normalization
    s_matrices[:, 1, 0] = 2 / normalization
    s_matrices[:, 1, 1] = (-a + b / z0_ohm - c * z0_ohm + d) / normalization
    return s_matrices


def fit_cell(input_dir: Path, definition: dict[str, float | str]) -> dict[str, float]:
    """L/C/Lf estimates and residuals from low-frequency and broadband HFSS sweeps."""
    lowfreq_hz, lowfreq_s = read_touchstone_ma(input_dir / str(definition["lowfreq_file"]))
    broadband_hz, broadband_s = read_touchstone_ma(input_dir / str(definition["broadband_file"]))

    lowfreq_abcd = s_to_abcd(lowfreq_s)
    lowfreq_angular = 2 * np.pi * lowfreq_hz
    inductance_h = float(np.mean(np.imag(lowfreq_abcd[:, 0, 1]) / lowfreq_angular))

    fit_mask = (broadband_hz >= FIT_BAND_GHZ[0] * 1e9) & (broadband_hz <= FIT_BAND_GHZ[1] * 1e9)
    fit_hz = broadband_hz[fit_mask]
    fit_s = broadband_s[fit_mask]
    fit_abcd = s_to_abcd(fit_s)
    fit_angular = 2 * np.pi * fit_hz
    effective_capacitance = np.imag(fit_abcd[:, 1, 0]) / fit_angular

    # For LCLfBaseCell: 1/C_eff = 1/C - (Lf/2) * omega^2.
    slope, intercept = np.polyfit(fit_angular**2, 1 / effective_capacitance, 1)
    capacitance_f = float(1 / intercept)
    finger_inductance_h = float(-2 * slope)
    if min(inductance_h, capacitance_f, finger_inductance_h) <= 0:
        raise ValueError("Fitted L, C, and Lf must all be positive")

    model_s = lclf_cell_s_parameters(fit_hz, inductance_h, capacitance_f, finger_inductance_h)
    hfss_s21, model_s21 = fit_s[:, 1, 0], model_s[:, 1, 0]
    magnitude_error_db = 20 * np.log10(np.abs(model_s21)) - 20 * np.log10(np.abs(hfss_s21))
    phase_error_rad = np.unwrap(np.angle(model_s21)) - np.unwrap(np.angle(hfss_s21))

    return {
        "stub_length_um": float(definition["stub_length_um"]),
        "L_pH": inductance_h * 1e12,
        "C_fF": capacitance_f * 1e15,
        "Lf_pH": finger_inductance_h * 1e12,
        "loss_tangent": 0.0,
        "Z0_ohm": float(np.sqrt(inductance_h / capacitance_f)),
        "fit_s21_rmse_db": float(np.sqrt(np.mean(magnitude_error_db**2))),
        "fit_phase_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(phase_error_rad**2)))),
    }


def acceptance_gates(fitted_cells: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Impedance and residual acceptance flags for the baseline cell pair."""
    selected_ratio = _SELECTED_LOADED_Z0_OHM / _SELECTED_UNLOADED_Z0_OHM
    unloaded, loaded = fitted_cells["unloaded"], fitted_cells["loaded"]
    return {
        "unloaded_Z0_selected_43p09ohm_pm5pct": (
            abs(unloaded["Z0_ohm"] - _SELECTED_UNLOADED_Z0_OHM) <= 0.05 * _SELECTED_UNLOADED_Z0_OHM
        ),
        "loaded_Z0_selected_70p20ohm_pm5pct": (
            abs(loaded["Z0_ohm"] - _SELECTED_LOADED_Z0_OHM) <= 0.05 * _SELECTED_LOADED_Z0_OHM
        ),
        "impedance_ratio_selected_1p629_pm5pct": (
            abs(loaded["Z0_ohm"] / unloaded["Z0_ohm"] - selected_ratio) <= 0.05 * selected_ratio
        ),
        "S21_fit_rmse_below_0p5db": max(row["fit_s21_rmse_db"] for row in fitted_cells.values())
        < 0.5,
        "phase_fit_rmse_below_5deg": max(row["fit_phase_rmse_deg"] for row in fitted_cells.values())
        < 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "hfss_inputs")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "hfss_inputs/hfss_cell_parameters.csv"
    )
    args = parser.parse_args()

    fitted_cells = {
        name: fit_cell(args.input_dir, definition) for name, definition in CELL_DEFINITIONS.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for name in ("unloaded", "loaded"):
            writer.writerow({"cell_type": name, **fitted_cells[name], "source": "HFSS"})

    report: dict[str, Any] = {
        "fit_band_GHz": list(FIT_BAND_GHZ),
        "cells": fitted_cells,
        "acceptance_gates": acceptance_gates(fitted_cells),
    }
    report_path = args.output.with_suffix(".fit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
