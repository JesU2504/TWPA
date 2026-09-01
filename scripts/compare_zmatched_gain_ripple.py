#!/usr/bin/env python3
"""Reflection-ripple comparison for the baseline and stub-only redesign.

The redesigned device uses interpolated 8.34/2.78 µm cells and a 16 GHz pump.
Eq. (S.1) and an eight-mode numerical solve are evaluated in result step 41.
The comparison uses each device's mid-band window, not identical frequencies."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from twpasolver import TWPAnalysis
from twpasolver.models import TWPA, LCLfBaseCell
from twpasolver.modes_rwa import ModeArrayFactory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code"))

from fit_hfss_unit_cells import fit_cell  # noqa: E402
from gain_ripple import db_to_linear_power, gain_ripple_db  # noqa: E402
from hfss_bloch import apply_bloch_parameters  # noqa: E402
from io_utils import read_touchstone, write_csv_rows  # noqa: E402
from plotting import save_figure  # noqa: E402

INPUTS = ROOT / "hfss_inputs"
OUTPUT = ROOT / "results" / "bloch_corrected" / "step_41_zmatched_gain_ripple"
STEP40 = ROOT / "results" / "bloch_corrected" / "step_40_zmatched_supercell_gain"
STEP37 = ROOT / "results" / "bloch_corrected" / "step_37_gain_ripple_analytic"
CELLS_PER_SUPERCELL = 34

TARGET_UNLOADED_UM = 8.34
TARGET_LOADED_UM = 2.78
OUR_GBP_OPTIMAL_PUMP_GHZ = 16.00  # from step_40's provenance
OUR_LINE_TRANSMITTIVITY = 1.0  # zero-loss HFSS policy, same as step_37/step_40
FINE_RIPPLE_WINDOW_GHZ = np.arange(7.0, 9.001, 0.01)  # mid-plateau window, mirrors step_37's choice


def interpolated_cell(lo: dict, hi: dict, target_um: float) -> dict:
    l0, l1 = lo["stub_length_um"], hi["stub_length_um"]
    frac = (target_um - l0) / (l1 - l0)
    return {key: lo[key] + (hi[key] - lo[key]) * frac for key in ("L_pH", "C_fF", "Lf_pH")}


def load_measured_supercell_s_parameters() -> tuple[np.ndarray, np.ndarray]:
    low_freq, low_s = read_touchstone(INPUTS / "supercell_15_4_15_zmatched_low_1_3GHz.s2p")
    fine_freq, fine_s = read_touchstone(INPUTS / "supercell_15_4_15_zmatched_fine_3_14GHz.s2p")
    aux_freq, aux_s = read_touchstone(INPUTS / "supercell_15_4_15_zmatched_aux_14_27GHz.s2p")
    frequencies_ghz = np.concatenate((low_freq, fine_freq[1:], aux_freq[1:]))
    s_matrices = np.concatenate((low_s, fine_s[1:], aux_s[1:]), axis=0)
    return frequencies_ghz, s_matrices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((INPUTS / "amplifier_config.json").read_text())
    initial_signal_current_a = float(config["Is0_A"])
    # The second pump harmonic reaches 32 GHz, beyond the default 30 GHz grid.
    config["base_frequency_stop_GHz"] = 40.0

    unloaded_8 = fit_cell(
        INPUTS,
        {
            "stub_length_um": 8.0,
            "broadband_file": "kinetic_8um_debug.s2p",
            "lowfreq_file": "kinetic_8um_lowfreq.s2p",
        },
    )
    unloaded_9 = fit_cell(
        INPUTS,
        {
            "stub_length_um": 9.0,
            "broadband_file": "kinetic_9um_debug.s2p",
            "lowfreq_file": "kinetic_9um_lowfreq.s2p",
        },
    )
    loaded_2p5 = fit_cell(
        INPUTS,
        {
            "stub_length_um": 2.5,
            "broadband_file": "kinetic_2p5um_debug.s2p",
            "lowfreq_file": "kinetic_2p5um_lowfreq.s2p",
        },
    )
    loaded_3 = fit_cell(
        INPUTS,
        {
            "stub_length_um": 3.0,
            "broadband_file": "kinetic_3um_debug.s2p",
            "lowfreq_file": "kinetic_3um_lowfreq.s2p",
        },
    )
    unloaded_fit = interpolated_cell(unloaded_8, unloaded_9, TARGET_UNLOADED_UM)
    loaded_fit = interpolated_cell(loaded_2p5, loaded_3, TARGET_LOADED_UM)

    unloaded_cell = LCLfBaseCell(
        L=unloaded_fit["L_pH"] * 1e-12,
        C=unloaded_fit["C_fF"] * 1e-15,
        Lf=unloaded_fit["Lf_pH"] * 1e-12,
        N=15,
    )
    loaded_cell = LCLfBaseCell(
        L=loaded_fit["L_pH"] * 1e-12,
        C=loaded_fit["C_fF"] * 1e-15,
        Lf=loaded_fit["Lf_pH"] * 1e-12,
        N=4,
    )
    twpa = TWPA(
        cells=[unloaded_cell, loaded_cell, unloaded_cell],
        N=int(config["supercells"]),
        Istar=float(config["Istar_A"]),
        Idc=float(config["Idc_A"]),
        Ip0=float(config["Ip0_A"]),
    )

    measured_freq_ghz, measured_s = load_measured_supercell_s_parameters()
    engineered = TWPAnalysis(
        twpa=twpa,
        f_arange=(
            config["base_frequency_start_GHz"],
            config["base_frequency_stop_GHz"],
            config["base_frequency_step_GHz"],
        ),
        unit="GHz",
    )
    engineered.update_base_data()
    apply_bloch_parameters(engineered.data, measured_freq_ghz, measured_s, CELLS_PER_SUPERCELL)

    freqs_ghz = np.asarray(engineered.data["freqs"], dtype=float)
    reflection_magnitude_by_freq = np.abs(np.asarray(engineered.data["gammas"], dtype=complex))

    gain_rows = list(csv.DictReader(open(STEP40 / "04_gain_profiles.csv")))
    gbp_column = next(key for key in gain_rows[0] if "max_gbp" in key)
    signal_ghz = np.array([float(row["signal_GHz"]) for row in gain_rows])
    our_gain_db = np.array([float(row[gbp_column]) for row in gain_rows])
    our_reflection_magnitude = np.interp(signal_ghz, freqs_ghz, reflection_magnitude_by_freq)

    write_csv_rows(
        OUTPUT / "01_our_device_gamma_and_gain.csv",
        ["signal_GHz", "our_gain_dB", "our_gamma_mag"],
        [signal_ghz, our_gain_db, our_reflection_magnitude],
    )

    our_ripple_db = gain_ripple_db(
        db_to_linear_power(our_gain_db), OUR_LINE_TRANSMITTIVITY, our_reflection_magnitude
    )
    reflection_term = (
        db_to_linear_power(our_gain_db) * OUR_LINE_TRANSMITTIVITY * our_reflection_magnitude**2
    )
    write_csv_rows(
        OUTPUT / "02_our_device_analytic_ripple.csv",
        ["signal_GHz", "G_eta_Gamma2", "dG_analytic_dB"],
        [signal_ghz, reflection_term, our_ripple_db],
    )

    inband_mask = (signal_ghz >= 5.0) & (signal_ghz <= 9.0)
    median_gamma_inband = float(np.median(our_reflection_magnitude[inband_mask]))

    fig, ax_gain = plt.subplots(figsize=(8, 5))
    ax_gain.plot(signal_ghz, our_gain_db, color="tab:blue", label="Gain, G(f) [dB]")
    ax_gain.set_xlabel("Signal frequency [GHz]")
    ax_gain.set_ylabel("Gain [dB]", color="tab:blue")
    ax_ripple = ax_gain.twinx()
    ax_ripple.plot(signal_ghz, our_ripple_db, color="tab:red", label="Analytic ripple, dG(f) [dB]")
    ax_ripple.set_ylabel("Analytic ripple dG [dB] (Eq. S.1)", color="tab:red")
    ax_gain.set_title(
        f"Redesigned structure: gain and Eq. (S.1) domain check at pump={OUR_GBP_OPTIMAL_PUMP_GHZ} GHz"
    )
    gh, gl = ax_gain.get_legend_handles_labels()
    rh, rl = ax_ripple.get_legend_handles_labels()
    ax_gain.legend(gh + rh, gl + rl, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, OUTPUT, "02_our_device_analytic_ripple")
    plt.close(fig)

    modes = ModeArrayFactory.create_extended_3wm(
        engineered.data, n_pump_harmonics=1, n_frequency_conversion=1, n_signal_harmonics=1
    )
    engineered.add_mode_array("all_8_modes", modes)

    print(
        f"Fine general-model sweep: {len(FINE_RIPPLE_WINDOW_GHZ)} points at pump={OUR_GBP_OPTIMAL_PUMP_GHZ} GHz ..."
    )
    t0 = time.time()
    result = engineered.gain(
        FINE_RIPPLE_WINDOW_GHZ,
        pump=OUR_GBP_OPTIMAL_PUMP_GHZ,
        Is0=initial_signal_current_a,
        model="general",
        mode_array_config="all_8_modes",
        thin=400,
    )
    fine_gain_db = np.asarray(result["gain_db"], dtype=float)
    elapsed_s = time.time() - t0
    print(f"Fine sweep done in {elapsed_s:.1f}s")

    numeric_ripple_pp_db = float(fine_gain_db.max() - fine_gain_db.min())
    mean_gain_window_db = float(np.mean(fine_gain_db))
    gamma_window = np.interp(
        np.mean(FINE_RIPPLE_WINDOW_GHZ), freqs_ghz, reflection_magnitude_by_freq
    )
    analytic_ripple_window_db = float(
        gain_ripple_db(
            np.array([db_to_linear_power(mean_gain_window_db)]),
            OUR_LINE_TRANSMITTIVITY,
            np.array([gamma_window]),
        )[0]
    )

    write_csv_rows(
        OUTPUT / "03_fine_numeric_ripple_check.csv",
        ["signal_GHz", "gain_dB_general_model"],
        [FINE_RIPPLE_WINDOW_GHZ, fine_gain_db],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(FINE_RIPPLE_WINDOW_GHZ, fine_gain_db, color="tab:green", lw=1.2)
    ax.axhline(
        mean_gain_window_db,
        color="gray",
        ls=":",
        lw=1,
        label=f"mean = {mean_gain_window_db:.2f} dB",
    )
    ax.set_xlabel("Signal frequency [GHz]")
    ax.set_ylabel("Gain [dB], general model (reflections on)")
    ax.set_title(
        f"Reflection-enabled numerical model: gain ripple\nobserved peak-to-peak = {numeric_ripple_pp_db:.3f} dB"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, OUTPUT, "03_fine_numeric_ripple_check")
    plt.close(fig)

    original_provenance = json.loads((STEP37 / "07_provenance.json").read_text())
    original_check = original_provenance["fine_numeric_check"]

    provenance = {
        "purpose": "Compare gain ripple between the original and impedance-matched devices",
        "our_pump_GHz": OUR_GBP_OPTIMAL_PUMP_GHZ,
        "median_gamma_inband_5_9GHz": median_gamma_inband,
        "fine_numeric_check": {
            "signal_window_GHz": [
                float(FINE_RIPPLE_WINDOW_GHZ[0]),
                float(FINE_RIPPLE_WINDOW_GHZ[-1]),
            ],
            "solve_time_s": elapsed_s,
            "observed_peak_to_peak_ripple_dB": numeric_ripple_pp_db,
            "mean_gain_in_window_dB": mean_gain_window_db,
            "gamma_mag_in_window": float(gamma_window),
            "analytic_ripple_prediction_dB": analytic_ripple_window_db,
        },
        "comparison_to_original_device_step37": {
            "original_window_GHz": original_check["signal_window_GHz"],
            "original_observed_peak_to_peak_ripple_dB": original_check[
                "observed_peak_to_peak_ripple_dB"
            ],
            "original_gamma_mag_in_window": original_check["gamma_mag_in_window"],
            "note": "Different signal windows (mid-plateau of each device's own dome), so this is a like-for-like 'ripple in the flat part of the gain profile' comparison, not identical frequencies.",
        },
    }
    (OUTPUT / "04_provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"\nMedian |Gamma| in-band (5-9 GHz): {median_gamma_inband:.4f}")
    print(
        f"Numeric peak-to-peak ripple: {numeric_ripple_pp_db:.3f} dB (vs. original device: {original_check['observed_peak_to_peak_ripple_dB']:.3f} dB)"
    )
    print(
        f"Gamma in window: {gamma_window:.4f} (vs. original device: {original_check['gamma_mag_in_window']:.4f})"
    )
    print(f"Wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
