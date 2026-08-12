#!/usr/bin/env python3
"""Fit twpasolver LCLfBaseCell parameters from HFSS Touchstone exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CELLS = {
    "unloaded": {
        "stub_length_um": 12.0,
        "broadband": "kinetic_12um_debug.s2p",
        "lowfreq": "kinetic_12um_lowfreq.s2p",
        "target_Z0_ohm": 48.0,
    },
    "loaded": {
        "stub_length_um": 3.9,
        "broadband": "kinetic_3p9um_debug.s2p",
        "lowfreq": "kinetic_3p9um_lowfreq.s2p",
        "target_Z0_ohm": 78.0,
    },
}


def read_touchstone_ma(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-port Touchstone 1 file exported as GHz/S/MA."""
    frequencies: list[float] = []
    matrices: list[np.ndarray] = []
    option_line = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            option_line = line.upper()
            continue
        values = [float(item) for item in line.split()[:9]]
        if len(values) != 9:
            continue
        frequency = values[0] * 1e9
        pairs = [
            values[index] * np.exp(1j * np.deg2rad(values[index + 1]))
            for index in (1, 3, 5, 7)
        ]
        # Touchstone two-port ordering is S11, S21, S12, S22.
        matrices.append(np.array([[pairs[0], pairs[2]], [pairs[1], pairs[3]]]))
        frequencies.append(frequency)
    if option_line is None or "GHZ" not in option_line or " S " not in option_line or " MA " not in option_line:
        raise ValueError(f"{path} must use '# GHz S MA R 50'")
    if "R 50" not in option_line:
        raise ValueError(f"{path} must be normalized to 50 ohms")
    return np.asarray(frequencies), np.asarray(matrices)


def s_to_abcd(s: np.ndarray, z0: float = 50.0) -> np.ndarray:
    s11 = s[:, 0, 0]
    s12 = s[:, 0, 1]
    s21 = s[:, 1, 0]
    s22 = s[:, 1, 1]
    abcd = np.empty_like(s)
    abcd[:, 0, 0] = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    abcd[:, 0, 1] = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    abcd[:, 1, 0] = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * s21 * z0)
    abcd[:, 1, 1] = ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return abcd


def lclf_s(
    frequencies: np.ndarray,
    inductance: float,
    capacitance: float,
    finger_inductance: float,
    z0: float = 50.0,
) -> np.ndarray:
    """Return the uncentered LCLfBaseCell response used by this project."""
    omega = 2 * np.pi * frequencies
    denominator = 2 - finger_inductance * capacitance * omega**2
    a = np.ones_like(omega, dtype=complex)
    b = 1j * inductance * omega
    c = 1j * 2 * capacitance * omega / denominator
    d = 1 - 2 * inductance * capacitance * omega**2 / denominator
    normalization = a + b / z0 + c * z0 + d
    result = np.empty((len(frequencies), 2, 2), dtype=complex)
    result[:, 0, 0] = (a + b / z0 - c * z0 - d) / normalization
    result[:, 0, 1] = 2 * (a * d - b * c) / normalization
    result[:, 1, 0] = 2 / normalization
    result[:, 1, 1] = (-a + b / z0 - c * z0 + d) / normalization
    return result


def fit_cell(input_dir: Path, definition: dict[str, float | str]) -> dict[str, float]:
    low_f, low_s = read_touchstone_ma(input_dir / str(definition["lowfreq"]))
    broad_f, broad_s = read_touchstone_ma(input_dir / str(definition["broadband"]))

    low_abcd = s_to_abcd(low_s)
    low_omega = 2 * np.pi * low_f
    inductance = float(np.mean(np.imag(low_abcd[:, 0, 1]) / low_omega))

    fit_mask = (broad_f >= 4e9) & (broad_f <= 14e9)
    fit_f = broad_f[fit_mask]
    fit_s = broad_s[fit_mask]
    fit_abcd = s_to_abcd(fit_s)
    omega = 2 * np.pi * fit_f
    effective_c = np.imag(fit_abcd[:, 1, 0]) / omega

    # For LCLfBaseCell: 1/Ceff = 1/C - (Lf/2) * omega^2.
    slope, intercept = np.polyfit(omega**2, 1 / effective_c, 1)
    capacitance = float(1 / intercept)
    finger_inductance = float(-2 * slope)
    if min(inductance, capacitance, finger_inductance) <= 0:
        raise ValueError("Fitted L, C, and Lf must all be positive")

    model_s = lclf_s(fit_f, inductance, capacitance, finger_inductance)
    hfss_s21 = fit_s[:, 1, 0]
    model_s21 = model_s[:, 1, 0]
    magnitude_error = 20 * np.log10(np.abs(model_s21)) - 20 * np.log10(np.abs(hfss_s21))
    phase_error = np.unwrap(np.angle(model_s21)) - np.unwrap(np.angle(hfss_s21))
    magnitude_rmse = float(np.sqrt(np.mean(magnitude_error**2)))
    phase_rmse = float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
    characteristic_impedance = float(np.sqrt(inductance / capacitance))

    return {
        "stub_length_um": float(definition["stub_length_um"]),
        "L_pH": inductance * 1e12,
        "C_fF": capacitance * 1e15,
        "Lf_pH": finger_inductance * 1e12,
        "loss_tangent": 0.0,
        "Z0_ohm": characteristic_impedance,
        "fit_s21_rmse_db": magnitude_rmse,
        "fit_phase_rmse_deg": phase_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("hfss_inputs"))
    parser.add_argument("--output", type=Path, default=Path("hfss_inputs/hfss_cell_parameters.csv"))
    args = parser.parse_args()

    rows = {name: fit_cell(args.input_dir, definition) for name, definition in CELLS.items()}
    fieldnames = [
        "cell_type", "stub_length_um", "L_pH", "C_fF", "Lf_pH",
        "loss_tangent", "Z0_ohm", "fit_s21_rmse_db", "fit_phase_rmse_deg", "source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in ("unloaded", "loaded"):
            writer.writerow({"cell_type": name, **rows[name], "source": "HFSS"})

    selected_unloaded_z0 = 43.08925347829123
    selected_loaded_z0 = 70.20100727735449
    selected_ratio = selected_loaded_z0 / selected_unloaded_z0
    gates = {
        "unloaded_Z0_selected_43p09ohm_pm5pct": (
            abs(rows["unloaded"]["Z0_ohm"] - selected_unloaded_z0)
            <= 0.05 * selected_unloaded_z0
        ),
        "loaded_Z0_selected_70p20ohm_pm5pct": (
            abs(rows["loaded"]["Z0_ohm"] - selected_loaded_z0)
            <= 0.05 * selected_loaded_z0
        ),
        "impedance_ratio_selected_1p629_pm5pct": (
            abs(rows["loaded"]["Z0_ohm"] / rows["unloaded"]["Z0_ohm"] - selected_ratio)
            <= 0.05 * selected_ratio
        ),
        "S21_fit_rmse_below_0p5db": max(row["fit_s21_rmse_db"] for row in rows.values()) < 0.5,
        "phase_fit_rmse_below_5deg": max(row["fit_phase_rmse_deg"] for row in rows.values()) < 5.0,
    }
    report = {"fit_band_GHz": [4.0, 14.0], "cells": rows, "acceptance_gates": gates}
    report_path = args.output.with_suffix(".fit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
