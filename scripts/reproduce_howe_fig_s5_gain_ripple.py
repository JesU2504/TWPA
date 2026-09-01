#!/usr/bin/env python3
"""Analytic and numerical reflection ripple for the baseline device.

Howe et al., arXiv:2507.07706, Eq. (S.1) supplies the analytic estimate.
The baseline calculation uses the step-34 gain profile at 14.10 GHz,
unit power transmittivity, and Bloch boundary reflection coefficients.
An eight-mode solve resolves numerical ripple over 6–8 GHz.
Outputs are stored in result step 37."""

from __future__ import annotations

import csv
import time

from _bootstrap import ROOT

import matplotlib.pyplot as plt
import numpy as np
from twpasolver import TWPAnalysis
from twpasolver.modes_rwa import ModeArrayFactory

from device_models import build_twpa_from_hfss
from gain_ripple import db_to_linear_power, gain_ripple_db
from hfss_bloch import apply_bloch_parameters
from io_utils import read_touchstone, write_csv_rows, write_json
from plotting import save_figure

INPUTS = ROOT / "hfss_inputs"
OUTPUT = ROOT / "results" / "bloch_corrected" / "step_37_gain_ripple_analytic"
STEP34 = ROOT / "results" / "bloch_corrected" / "step_34_howe_fig3_hfss_bloch_reproduction"
CELLS_PER_SUPERCELL = 34

OUR_GBP_OPTIMAL_PUMP_GHZ = 14.10  # from step_34's 05_provenance.json
OUR_LINE_TRANSMITTIVITY = 1.0  # self-consistent with the project's zero-loss HFSS/Bloch policy
SIGNAL_SWEEP_GHZ = np.arange(2.5, 11.51, 0.05)
FINE_RIPPLE_WINDOW_GHZ = np.arange(
    6.0, 8.001, 0.01
)  # narrow, fine window in the low-ripple part of the dome


def load_measured_supercell_s_parameters() -> tuple[np.ndarray, np.ndarray]:
    """Stitched 1–27 GHz supercell frequencies and S matrices."""
    low_freq, low_s = read_touchstone(INPUTS / "supercell_15_4_15_low_1_3GHz.s2p")
    fine_freq, fine_s = read_touchstone(INPUTS / "supercell_15_4_15_fine_3_14GHz.s2p")
    aux_freq, aux_s = read_touchstone(INPUTS / "supercell_15_4_15_aux_14_27GHz.s2p")
    frequencies_ghz = np.concatenate((low_freq, fine_freq[1:], aux_freq[1:]))
    s_matrices = np.concatenate((low_s, fine_s[1:], aux_s[1:]), axis=0)
    return frequencies_ghz, s_matrices


def reproduce_paper_panels() -> None:
    """Analytic Fig. S.5 panels evaluated on the published parameter grids."""
    reflection_db_grid = np.array([-20, -13, -11, -10])
    reflection_magnitude_grid = 10 ** (reflection_db_grid / 20.0)
    gain_db_axis = np.linspace(0.01, 20, 400)
    gain_linear_axis = db_to_linear_power(gain_db_axis)

    panel_a = {
        f"{g}dB": gain_ripple_db(gain_linear_axis, 1.0, gm)
        for g, gm in zip(reflection_db_grid, reflection_magnitude_grid)
    }
    write_csv_rows(
        OUTPUT / "01_paper_fig_s5_panel_a.csv",
        ["gain_dB", *[f"dG_dB_gamma_{g}dB" for g in reflection_db_grid]],
        [gain_db_axis, *panel_a.values()],
    )

    panel_b = {
        f"{g}dB": gain_ripple_db(gain_linear_axis, 0.5, gm)
        for g, gm in zip(reflection_db_grid, reflection_magnitude_grid)
    }
    write_csv_rows(
        OUTPUT / "02_paper_fig_s5_panel_b.csv",
        ["gain_dB", *[f"dG_dB_gamma_{g}dB" for g in reflection_db_grid]],
        [gain_db_axis, *panel_b.values()],
    )

    reflection_db_axis = np.linspace(-20, -10, 400)
    reflection_magnitude_axis = 10 ** (reflection_db_axis / 20.0)
    gain_20db_linear = db_to_linear_power(np.array(20.0))
    panel_c_lossless = gain_ripple_db(gain_20db_linear, 1.0, reflection_magnitude_axis)
    panel_c_3db_loss = gain_ripple_db(gain_20db_linear, 0.5, reflection_magnitude_axis)
    write_csv_rows(
        OUTPUT / "03_paper_fig_s5_panel_c.csv",
        ["gamma_dB", "dG_dB_eta_1", "dG_dB_eta_0.5"],
        [reflection_db_axis, panel_c_lossless, panel_c_3db_loss],
    )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, curve in panel_a.items():
        ax_a.plot(gain_db_axis, curve, label=f"|Gamma|={label}")
    ax_a.set(xlabel="Gain, G [dB]", ylabel="dG [dB]", title="(a) eta=1 (lossless)", ylim=(0, 15))
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    for label, curve in panel_b.items():
        ax_b.plot(gain_db_axis, curve, label=f"|Gamma|={label}")
    ax_b.set(xlabel="Gain, G [dB]", ylabel="dG [dB]", title="(b) eta=0.5 (3 dB IL)", ylim=(0, 5))
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    ax_c.plot(reflection_db_axis, panel_c_lossless, label="eta=1")
    ax_c.plot(reflection_db_axis, panel_c_3db_loss, label="eta=0.5")
    ax_c.set(xlabel="|Gamma| [dB]", ylabel="dG [dB]", title="(c) G=20 dB", ylim=(0, 20))
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3)

    fig.suptitle("Exact reproduction of Fig. S.5 (Eq. S.1 is a disclosed formula, not data)")
    fig.tight_layout()
    save_figure(fig, OUTPUT, "fig_s5_reproduction")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reproduce_paper_panels()

    twpa, provenance = build_twpa_from_hfss(
        INPUTS / "hfss_cell_parameters.csv", INPUTS / "amplifier_config.json"
    )
    config = provenance["amplifier_config"]
    initial_signal_current_a = float(config["Is0_A"])

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

    gain_rows = list(csv.DictReader(open(STEP34 / "04_gain_profiles.csv")))
    gbp_column = next(key for key in gain_rows[0] if "max_gbp" in key)
    signal_ghz = np.array([float(row["signal_GHz"]) for row in gain_rows])
    our_gain_db = np.array([float(row[gbp_column]) for row in gain_rows])
    our_reflection_magnitude = np.interp(signal_ghz, freqs_ghz, reflection_magnitude_by_freq)

    write_csv_rows(
        OUTPUT / "04_our_device_gamma_and_gain.csv",
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
        OUTPUT / "05_our_device_analytic_ripple.csv",
        ["signal_GHz", "G_eta_Gamma2", "dG_analytic_dB"],
        [signal_ghz, reflection_term, our_ripple_db],
    )

    fig, ax_gain = plt.subplots(figsize=(8, 5))
    ax_gain.plot(signal_ghz, our_gain_db, color="tab:blue", label="Our model's gain, G(f) [dB]")
    ax_gain.set_xlabel("Signal frequency [GHz]")
    ax_gain.set_ylabel("Gain [dB]", color="tab:blue")
    ax_ripple = ax_gain.twinx()
    ax_ripple.plot(signal_ghz, our_ripple_db, color="tab:red", label="Analytic ripple, dG(f) [dB]")
    ax_ripple.set_ylabel("Analytic ripple dG [dB] (Eq. S.1)", color="tab:red")
    diverges = ~np.isfinite(our_ripple_db)
    if diverges.any():
        ax_ripple.axvspan(
            signal_ghz[diverges].min(),
            signal_ghz[diverges].max(),
            color="red",
            alpha=0.15,
            label="Eq. S.1 diverges here (G*eta*|Gamma|^2 >= 1)",
        )
    ax_gain.set_title(
        f"Our device: predicted gain and Eq. (S.1) ripple at pump={OUR_GBP_OPTIMAL_PUMP_GHZ} GHz"
    )
    gain_handles, gain_labels = ax_gain.get_legend_handles_labels()
    ripple_handles, ripple_labels = ax_ripple.get_legend_handles_labels()
    ax_gain.legend(
        gain_handles + ripple_handles, gain_labels + ripple_labels, fontsize=8, loc="upper left"
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT, "06_our_device_analytic_ripple")
    plt.close(fig)

    modes = ModeArrayFactory.create_extended_3wm(
        engineered.data, n_pump_harmonics=1, n_frequency_conversion=1, n_signal_harmonics=1
    )
    engineered.add_mode_array("all_8_modes", modes)

    print(
        f"Fine general-model sweep: {len(FINE_RIPPLE_WINDOW_GHZ)} points at pump={OUR_GBP_OPTIMAL_PUMP_GHZ} GHz ..."
    )
    sweep_start_time = time.time()
    result = engineered.gain(
        FINE_RIPPLE_WINDOW_GHZ,
        pump=OUR_GBP_OPTIMAL_PUMP_GHZ,
        Is0=initial_signal_current_a,
        model="general",
        mode_array_config="all_8_modes",
        thin=400,
    )
    fine_gain_db = np.asarray(result["gain_db"], dtype=float)
    elapsed_s = time.time() - sweep_start_time
    print(f"Fine sweep done in {elapsed_s:.1f}s")

    numeric_ripple_peak_to_peak_db = float(fine_gain_db.max() - fine_gain_db.min())
    mean_gain_in_window_db = float(np.mean(fine_gain_db))
    reflection_magnitude_in_window = np.interp(
        np.mean(FINE_RIPPLE_WINDOW_GHZ), freqs_ghz, reflection_magnitude_by_freq
    )
    analytic_ripple_in_window_db = float(
        gain_ripple_db(
            np.array([db_to_linear_power(mean_gain_in_window_db)]),
            OUR_LINE_TRANSMITTIVITY,
            np.array([reflection_magnitude_in_window]),
        )[0]
    )

    write_csv_rows(
        OUTPUT / "06_fine_numeric_ripple_check.csv",
        ["signal_GHz", "gain_dB_general_model"],
        [FINE_RIPPLE_WINDOW_GHZ, fine_gain_db],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(FINE_RIPPLE_WINDOW_GHZ, fine_gain_db, color="tab:green", lw=1.2)
    ax.axhline(
        mean_gain_in_window_db,
        color="gray",
        ls=":",
        lw=1,
        label=f"mean = {mean_gain_in_window_db:.2f} dB",
    )
    ax.set_xlabel("Signal frequency [GHz]")
    ax.set_ylabel("Gain [dB], general model (reflections on)")
    ax.set_title(
        f"Fine numeric ripple check\nobserved peak-to-peak = {numeric_ripple_peak_to_peak_db:.3f} dB, Eq. (S.1) predicts {analytic_ripple_in_window_db:.3f} dB"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, OUTPUT, "07_fine_numeric_ripple_check")
    plt.close(fig)

    provenance_out = {
        "figure_reproduced": "Fig. S.5 / Eq. (S.1), Howe et al. 2507.07706, Appendix B",
        "method": "Eq. (S.1) is a disclosed closed-form formula, reproduced exactly (not digitized)",
        "our_device_gain_source": str((STEP34 / "04_gain_profiles.csv").relative_to(ROOT)),
        "our_device_gamma_source": "engineered.data['gammas'] (Bloch-extracted S11/S22 of the real 15-4-15 HFSS supercell)",
        "eta_assumption": "eta=1; dissipative loss omitted",
        "our_pump_GHz": OUR_GBP_OPTIMAL_PUMP_GHZ,
        "fine_numeric_check": {
            "signal_window_GHz": [
                float(FINE_RIPPLE_WINDOW_GHZ[0]),
                float(FINE_RIPPLE_WINDOW_GHZ[-1]),
            ],
            "step_GHz": 0.01,
            "n_points": len(FINE_RIPPLE_WINDOW_GHZ),
            "observed_peak_to_peak_ripple_dB": numeric_ripple_peak_to_peak_db,
            "mean_gain_in_window_dB": mean_gain_in_window_db,
            "gamma_mag_in_window": float(reflection_magnitude_in_window),
            "analytic_ripple_prediction_dB": analytic_ripple_in_window_db,
        },
        "our_device_ripple_diverges_eq_s1": bool(np.any(~np.isfinite(our_ripple_db))),
    }
    write_json(OUTPUT / "07_provenance.json", provenance_out)

    print(
        f"Numeric peak-to-peak ripple in {FINE_RIPPLE_WINDOW_GHZ[0]}-{FINE_RIPPLE_WINDOW_GHZ[-1]} GHz window: {numeric_ripple_peak_to_peak_db:.3f} dB"
    )
    print(
        f"Eq. (S.1) analytic prediction at mean gain/Gamma in that window: {analytic_ripple_in_window_db:.3f} dB"
    )
    print(f"Wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
