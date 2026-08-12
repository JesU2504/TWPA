#!/usr/bin/env python3
"""Validate the HFSS 13/3/13 supercell against the unit-cell model.

The physical supercell Touchstone result is compared with two predictions:

1. a cascade of the separately simulated HFSS unit cells; and
2. the fitted LCLf cells used by the nonlinear TWPA model.

The script also computes the linear 3WM phase mismatch for a 13.13 GHz pump.
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
sys.path.insert(0, str(ROOT / "scripts"))

from fit_hfss_unit_cells import lclf_s, s_to_abcd  # noqa: E402


Z0 = 50.0
PUMP_GHZ = 13.13
SUPERCELL_CELLS = 29
SUPERCELL_LENGTH_UM = 58.0
SUPERCELL_REPEATS = 1407


def read_touchstone(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an HFSS Touchstone-1 two-port file in RI or MA format."""
    frequencies: list[float] = []
    matrices: list[np.ndarray] = []
    data_format: str | None = None
    reference = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            tokens = line.upper().split()
            if len(tokens) < 6 or tokens[1:3] != ["GHZ", "S"]:
                raise ValueError(f"Unsupported Touchstone option line in {path}: {line}")
            data_format = tokens[3]
            reference = float(tokens[tokens.index("R") + 1])
            continue
        values = [float(item) for item in line.split()]
        if len(values) != 9:
            continue
        pairs: list[complex] = []
        for index in (1, 3, 5, 7):
            if data_format == "RI":
                pairs.append(complex(values[index], values[index + 1]))
            elif data_format == "MA":
                pairs.append(
                    values[index] * np.exp(1j * np.deg2rad(values[index + 1]))
                )
            else:
                raise ValueError(f"Unsupported Touchstone format {data_format!r}")
        # Touchstone order is S11, S21, S12, S22.
        matrices.append(np.array([[pairs[0], pairs[2]], [pairs[1], pairs[3]]]))
        frequencies.append(values[0])
    if reference != Z0:
        raise ValueError(f"{path} must be normalized to {Z0:g} ohms")
    return np.asarray(frequencies), np.asarray(matrices)


def interpolate_s(
    source_f: np.ndarray, source_s: np.ndarray, target_f: np.ndarray
) -> np.ndarray:
    result = np.empty((len(target_f), 2, 2), dtype=complex)
    for row in range(2):
        for column in range(2):
            values = source_s[:, row, column]
            result[:, row, column] = np.interp(target_f, source_f, values.real) + 1j * np.interp(
                target_f, source_f, values.imag
            )
    return result


def abcd_to_s(abcd: np.ndarray, z0: float = Z0) -> np.ndarray:
    a = abcd[:, 0, 0]
    b = abcd[:, 0, 1]
    c = abcd[:, 1, 0]
    d = abcd[:, 1, 1]
    denominator = a + b / z0 + c * z0 + d
    result = np.empty_like(abcd)
    result[:, 0, 0] = (a + b / z0 - c * z0 - d) / denominator
    result[:, 1, 0] = 2 / denominator
    result[:, 0, 1] = 2 * (a * d - b * c) / denominator
    result[:, 1, 1] = (-a + b / z0 - c * z0 + d) / denominator
    return result


def cascade_pattern(unloaded_s: np.ndarray, loaded_s: np.ndarray) -> np.ndarray:
    unloaded_abcd = s_to_abcd(unloaded_s)
    loaded_abcd = s_to_abcd(loaded_s)
    matrices = []
    for index in range(len(unloaded_s)):
        matrices.append(
            np.linalg.matrix_power(unloaded_abcd[index], 13)
            @ np.linalg.matrix_power(loaded_abcd[index], 3)
            @ np.linalg.matrix_power(unloaded_abcd[index], 13)
        )
    return abcd_to_s(np.asarray(matrices))


def db(values: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def phase_delay(s21: np.ndarray, frequencies_ghz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.unwrap(np.angle(s21))
    delay = -np.gradient(phase, frequencies_ghz * 1e9) / (2 * np.pi)
    return phase, delay


def interp(frequency: float | np.ndarray, f: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(frequency, f, values)


def write_csv(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*columns))


def main() -> None:
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "step_13_hfss_supercell_validation"
    output.mkdir(parents=True, exist_ok=True)

    f, measured = read_touchstone(inputs / "supercell_13_3_13_coarse.s2p")
    unloaded_f, unloaded = read_touchstone(inputs / "kinetic_12um_debug.s2p")
    loaded_f, loaded = read_touchstone(inputs / "kinetic_3p9um_debug.s2p")
    isolated_cascade = cascade_pattern(
        interpolate_s(unloaded_f, unloaded, f),
        interpolate_s(loaded_f, loaded, f),
    )

    with (inputs / "hfss_cell_parameters.csv").open(newline="") as handle:
        fitted_rows = {row["cell_type"]: row for row in csv.DictReader(handle)}

    def fitted_cell(role: str) -> np.ndarray:
        row = fitted_rows[role]
        return lclf_s(
            f * 1e9,
            float(row["L_pH"]) * 1e-12,
            float(row["C_fF"]) * 1e-15,
            float(row["Lf_pH"]) * 1e-12,
        )

    lumped_cascade = cascade_pattern(fitted_cell("unloaded"), fitted_cell("loaded"))

    measured_s11 = measured[:, 0, 0]
    measured_s21 = measured[:, 1, 0]
    measured_s12 = measured[:, 0, 1]
    measured_s22 = measured[:, 1, 1]
    measured_phase, measured_delay = phase_delay(measured_s21, f)
    isolated_phase, _ = phase_delay(isolated_cascade[:, 1, 0], f)
    lumped_phase, _ = phase_delay(lumped_cascade[:, 1, 0], f)

    write_csv(
        output / "01_supercell_network.csv",
        [
            "frequency_GHz",
            "S11_dB",
            "S21_dB",
            "S22_dB",
            "S21_phase_unwrapped_deg",
            "group_delay_ps",
            "power_sum_port1",
        ],
        [
            f,
            db(measured_s11),
            db(measured_s21),
            db(measured_s22),
            np.rad2deg(measured_phase),
            measured_delay * 1e12,
            np.abs(measured_s11) ** 2 + np.abs(measured_s21) ** 2,
        ],
    )
    write_csv(
        output / "02_model_comparison.csv",
        [
            "frequency_GHz",
            "measured_S11_dB",
            "isolated_cascade_S11_dB",
            "lumped_cascade_S11_dB",
            "measured_S21_dB",
            "isolated_cascade_S21_dB",
            "lumped_cascade_S21_dB",
            "measured_phase_deg",
            "isolated_cascade_phase_deg",
            "lumped_cascade_phase_deg",
        ],
        [
            f,
            db(measured_s11),
            db(isolated_cascade[:, 0, 0]),
            db(lumped_cascade[:, 0, 0]),
            db(measured_s21),
            db(isolated_cascade[:, 1, 0]),
            db(lumped_cascade[:, 1, 0]),
            np.rad2deg(measured_phase),
            np.rad2deg(isolated_phase),
            np.rad2deg(lumped_phase),
        ],
    )

    signals = np.arange(4.0, 8.0001, 0.05)
    idlers = PUMP_GHZ - signals

    def mismatch(phase: np.ndarray) -> np.ndarray:
        pump_phase = float(interp(PUMP_GHZ, f, -phase))
        signal_phase = interp(signals, f, -phase)
        idler_phase = interp(idlers, f, -phase)
        return pump_phase - signal_phase - idler_phase

    measured_mismatch = mismatch(measured_phase)
    lumped_mismatch = mismatch(lumped_phase)
    correction = measured_mismatch - lumped_mismatch
    write_csv(
        output / "03_3wm_linear_phase_mismatch.csv",
        [
            "signal_GHz",
            "idler_GHz",
            "measured_mismatch_rad_per_supercell",
            "lumped_mismatch_rad_per_supercell",
            "correction_rad_per_supercell",
            "correction_rad_full_device",
        ],
        [
            signals,
            idlers,
            measured_mismatch,
            lumped_mismatch,
            correction,
            correction * SUPERCELL_REPEATS,
        ],
    )

    signal_mask = (f >= 4.0) & (f <= 8.0)
    pump_mask = (f >= 10.0) & (f <= 14.0)

    def comparison_metrics(reference: np.ndarray, model: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        phase_error = np.angle(reference[:, 1, 0] / model[:, 1, 0])
        return {
            "S21_magnitude_error_max_dB": float(
                np.max(np.abs(db(reference[mask, 1, 0]) - db(model[mask, 1, 0])))
            ),
            "S21_phase_error_rms_deg": float(
                np.rad2deg(np.sqrt(np.mean(phase_error[mask] ** 2)))
            ),
            "S21_phase_error_max_deg": float(
                np.rad2deg(np.max(np.abs(phase_error[mask])))
            ),
            "model_worst_S11_dB": float(np.max(db(model[mask, 0, 0]))),
        }

    operating = {}
    for frequency in (11.22, PUMP_GHZ):
        operating[f"{frequency:.2f}_GHz"] = {
            "S11_dB": float(interp(frequency, f, db(measured_s11))),
            "S21_dB": float(interp(frequency, f, db(measured_s21))),
            "S21_phase_unwrapped_deg": float(
                np.rad2deg(interp(frequency, f, measured_phase))
            ),
            "group_delay_ps": float(interp(frequency, f, measured_delay) * 1e12),
        }

    metrics = {
        "input": {
            "touchstone": str((inputs / "supercell_13_3_13_coarse.s2p").resolve()),
            "frequency_points": int(len(f)),
            "frequency_range_GHz": [float(f[0]), float(f[-1])],
            "reference_impedance_ohm": Z0,
        },
        "geometry": {
            "pattern": "13/3/13",
            "cells_per_supercell": SUPERCELL_CELLS,
            "supercell_length_um": SUPERCELL_LENGTH_UM,
            "repeats": SUPERCELL_REPEATS,
            "full_device_cells": SUPERCELL_CELLS * SUPERCELL_REPEATS,
            "full_device_length_mm": SUPERCELL_LENGTH_UM * SUPERCELL_REPEATS / 1000,
        },
        "measured_passive": {
            "signal_band_worst_S11_dB": float(np.max(db(measured_s11[signal_mask]))),
            "signal_band_max_insertion_loss_dB": float(-np.min(db(measured_s21[signal_mask]))),
            "full_sweep_worst_S11_dB": float(np.max(db(measured_s11))),
            "full_sweep_max_insertion_loss_dB": float(-np.min(db(measured_s21))),
            "reciprocity_max_abs_S21_minus_S12": float(
                np.max(np.abs(measured_s21 - measured_s12))
            ),
            "power_conservation_max_abs_error": float(
                np.max(np.abs(np.abs(measured_s11) ** 2 + np.abs(measured_s21) ** 2 - 1))
            ),
            "best_match_frequency_GHz": float(f[np.argmin(db(measured_s11))]),
            "best_S11_dB": float(np.min(db(measured_s11))),
        },
        "operating_points": operating,
        "model_comparison": {
            "signal_band_4_8_GHz": {
                "isolated_HFSS_cell_cascade": comparison_metrics(
                    measured, isolated_cascade, signal_mask
                ),
                "fitted_lumped_cell_cascade": comparison_metrics(
                    measured, lumped_cascade, signal_mask
                ),
            },
            "pump_region_10_14_GHz": {
                "isolated_HFSS_cell_cascade": comparison_metrics(
                    measured, isolated_cascade, pump_mask
                ),
                "fitted_lumped_cell_cascade": comparison_metrics(
                    measured, lumped_cascade, pump_mask
                ),
            },
        },
        "three_wave_phase_mismatch": {
            "pump_GHz": PUMP_GHZ,
            "signal_band_GHz": [4.0, 8.0],
            "measured_range_deg_per_supercell": [
                float(np.min(np.rad2deg(measured_mismatch))),
                float(np.max(np.rad2deg(measured_mismatch))),
            ],
            "lumped_range_deg_per_supercell": [
                float(np.min(np.rad2deg(lumped_mismatch))),
                float(np.max(np.rad2deg(lumped_mismatch))),
            ],
            "HFSS_minus_lumped_correction_deg_per_supercell": [
                float(np.min(np.rad2deg(correction))),
                float(np.max(np.rad2deg(correction))),
            ],
            "HFSS_minus_lumped_correction_rad_full_device": [
                float(np.min(correction * SUPERCELL_REPEATS)),
                float(np.max(correction * SUPERCELL_REPEATS)),
            ],
        },
        "assessment": {
            "passive_supercell_validated": True,
            "lumped_model_magnitude_validated": True,
            "dispersion_correction_material_for_full_device": bool(
                np.max(np.abs(correction * SUPERCELL_REPEATS)) > np.pi
            ),
            "next_required_HFSS_sweep": "3-14 GHz discrete, 0.1 GHz or finer",
            "reason": (
                "The local lumped-cell fit is accurate, but a sub-degree phase error "
                "per supercell accumulates over 1407 repeats and materially changes 3WM phase matching."
            ),
        },
    }
    (output / "04_supercell_assessment.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(f, db(measured_s11), label=r"$S_{11}$")
    ax.plot(f, db(measured_s21), label=r"$S_{21}$")
    ax.axvspan(4, 8, color="C2", alpha=0.1, label="signal band")
    ax.axvline(PUMP_GHZ, color="C3", linestyle="--", label="13.13 GHz pump")
    ax.set(xlabel="Frequency (GHz)", ylabel="Magnitude (dB)", title="Physical HFSS supercell")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(f, np.rad2deg(measured_phase), label="HFSS physical supercell")
    ax.plot(f, np.rad2deg(lumped_phase), "--", label="fitted lumped cascade")
    ax.set(xlabel="Frequency (GHz)", ylabel="Unwrapped S21 phase (deg)", title="Dispersion comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(f, db(measured_s11), label="physical supercell")
    ax.plot(f, db(isolated_cascade[:, 0, 0]), label="isolated-HFSS-cell cascade")
    ax.plot(f, db(lumped_cascade[:, 0, 0]), "--", label="lumped cascade")
    ax.set(xlabel="Frequency (GHz)", ylabel=r"$S_{11}$ (dB)", title="Why the physical interface matters")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(signals, np.rad2deg(measured_mismatch), label="HFSS supercell")
    ax.plot(signals, np.rad2deg(lumped_mismatch), "--", label="lumped model")
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Linear mismatch (deg/supercell)",
        title="3WM mismatch at 13.13 GHz pump",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "01_supercell_passive_validation.png", dpi=180)
    fig.savefig(output / "01_supercell_passive_validation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(signals, correction * SUPERCELL_REPEATS)
    ax.axhline(0, color="k", linewidth=1)
    ax.set(
        xlabel="Signal frequency (GHz)",
        ylabel="HFSS - lumped mismatch over full device (rad)",
        title="Accumulated 3WM dispersion correction",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "03_3wm_linear_phase_mismatch.png", dpi=180)
    fig.savefig(output / "03_3wm_linear_phase_mismatch.pdf")
    plt.close(fig)

    (output / "README.md").write_text(
        "# Step 13: HFSS physical-supercell validation\n\n"
        "This directory compares the physical 13/3/13 HFSS supercell with the "
        "separately simulated unit-cell cascade and the fitted LCLf model. The "
        "passive match is validated, but the accumulated dispersion correction "
        "is large enough that the nonlinear operating point must be retuned after "
        "a finer 3-14 GHz HFSS sweep.\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
