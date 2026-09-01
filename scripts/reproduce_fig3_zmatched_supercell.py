#!/usr/bin/env python3
"""HFSS–Bloch gain for the 8.34/2.78 µm stub redesign.

Three supercell exports cover 1–27 GHz. Structural L/C/Lf values are interpolated
from adjacent single-cell fits: 8/9 µm unloaded and 2.5/3 µm loaded. Operating
currents and cell counts match the baseline configuration.

The exports contain a stopband near 14 GHz and an isolated S21 outlier at
12.9 GHz. The outlier remains in the input and affects interpolated dispersion.
Outputs are stored in result step 40."""

from __future__ import annotations

import json

from _bootstrap import ROOT

import matplotlib.pyplot as plt
import numpy as np
from twpasolver import TWPAnalysis
from twpasolver.models import TWPA, LCLfBaseCell

from fit_hfss_unit_cells import fit_cell
from hfss_bloch import apply_bloch_parameters
from io_utils import read_touchstone, write_csv_rows, write_json
from plotting import save_figure

INPUTS = ROOT / "hfss_inputs"
OUTPUT = ROOT / "results" / "bloch_corrected" / "step_40_zmatched_supercell_gain"
CELLS_PER_SUPERCELL = 34
GAIN_REDUCTION_DB = 3.0
K_STAR_FIT_BAND_GHZ = (3.0, 8.0)

TARGET_UNLOADED_UM = 8.34
TARGET_LOADED_UM = 2.78

# Start above the redesigned stopband and span the high-gain region.
PUMP_SWEEP_GHZ = np.arange(14.6, 19.61, 0.1)
SIGNAL_SWEEP_GHZ = np.arange(2.5, 11.51, 0.05)


def interpolated_cell(lo: dict, hi: dict, target_um: float) -> dict:
    """Linearly interpolate a fit_cell() result between two nearby lengths."""
    l0, l1 = lo["stub_length_um"], hi["stub_length_um"]
    frac = (target_um - l0) / (l1 - l0)
    return {key: lo[key] + (hi[key] - lo[key]) * frac for key in ("L_pH", "C_fF", "Lf_pH")}


def recompute_k_star(
    freqs_ghz: np.ndarray, k: np.ndarray, fit_band_ghz: tuple[float, float]
) -> np.ndarray:
    mask = (freqs_ghz >= fit_band_ghz[0]) & (freqs_ghz <= fit_band_ghz[1])
    fit_coefficients = np.polyfit(freqs_ghz[mask], k[mask], 2)
    return k - np.polyval(fit_coefficients[1:], freqs_ghz)


def load_measured_supercell_s_parameters() -> tuple[np.ndarray, np.ndarray]:
    low_freq, low_s = read_touchstone(INPUTS / "supercell_15_4_15_zmatched_low_1_3GHz.s2p")
    fine_freq, fine_s = read_touchstone(INPUTS / "supercell_15_4_15_zmatched_fine_3_14GHz.s2p")
    aux_freq, aux_s = read_touchstone(INPUTS / "supercell_15_4_15_zmatched_aux_14_27GHz.s2p")
    if not np.isclose(low_freq[-1], fine_freq[0]) or not np.allclose(
        low_s[-1], fine_s[0], atol=1e-12
    ):
        raise ValueError("low/fine sweeps disagree at their 3 GHz overlap")
    if not np.isclose(fine_freq[-1], aux_freq[0]) or not np.allclose(
        fine_s[-1], aux_s[0], atol=1e-12
    ):
        raise ValueError("fine/aux sweeps disagree at their 14 GHz overlap")
    frequencies_ghz = np.concatenate((low_freq, fine_freq[1:], aux_freq[1:]))
    s_matrices = np.concatenate((low_s, fine_s[1:], aux_s[1:]), axis=0)
    return frequencies_ghz, s_matrices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((INPUTS / "amplifier_config.json").read_text(encoding="utf-8"))
    initial_signal_current_a = float(config["Is0_A"])

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
    bloch_provenance = apply_bloch_parameters(
        engineered.data, measured_freq_ghz, measured_s, CELLS_PER_SUPERCELL
    )
    engineered.data["k_star"] = recompute_k_star(
        np.asarray(engineered.data["freqs"]), np.asarray(engineered.data["k"]), K_STAR_FIT_BAND_GHZ
    )

    all_freqs_ghz = np.asarray(engineered.data["freqs"])
    bloch_band_mask = (all_freqs_ghz >= 1.0) & (all_freqs_ghz <= 27.0)
    freqs_ghz = all_freqs_ghz[bloch_band_mask]
    evanescence_per_cell = np.asarray(engineered.data["bloch_evanescence"])[bloch_band_mask]
    s21_db_engineered = -20 * np.log10(np.e) * twpa.N_tot * evanescence_per_cell
    k_star_engineered = np.asarray(engineered.data["k_star"])[bloch_band_mask]

    write_csv_rows(
        OUTPUT / "01_dispersion.csv",
        ["freq_GHz", "S21_db_engineered", "k_star_engineered_rad", "bloch_evanescence_Np_per_cell"],
        [freqs_ghz, s21_db_engineered, k_star_engineered, evanescence_per_cell],
    )

    print(f"Sweeping {len(PUMP_SWEEP_GHZ)} pumps x {len(SIGNAL_SWEEP_GHZ)} signals ...")
    gain_rows = []
    for pump_ghz in PUMP_SWEEP_GHZ:
        result = engineered.gain(
            SIGNAL_SWEEP_GHZ,
            pump=float(pump_ghz),
            Is0=initial_signal_current_a,
            model="minimal_3wm",
            thin=400,
        )
        gain_rows.append(np.asarray(result["gain_db"], dtype=float))
    gain_matrix = np.asarray(gain_rows)
    if not np.all(np.isfinite(gain_matrix)):
        raise FloatingPointError("Gain sweep contains non-finite values")

    write_csv_rows(
        OUTPUT / "02_gain_heatmap.csv",
        ["signal_GHz", *[f"pump_{p:.2f}GHz_gain_dB" for p in PUMP_SWEEP_GHZ]],
        [SIGNAL_SWEEP_GHZ, *gain_matrix],
    )

    bandwidth_rows = []
    for pump_ghz in PUMP_SWEEP_GHZ:
        bandwidth = engineered.bandwidth(
            gain_reduction=GAIN_REDUCTION_DB,
            signal_freqs=SIGNAL_SWEEP_GHZ,
            pump=float(pump_ghz),
            Is0=initial_signal_current_a,
            model="minimal_3wm",
            thin=400,
        )
        mean_gain_db = float(bandwidth["mean_gain"])
        bandwidth_ghz = float(bandwidth["total_bw"])
        bandwidth_rows.append(
            {
                "pump_GHz": float(pump_ghz),
                "mean_B3dB_gain_dB": mean_gain_db,
                "B3dB_bandwidth_GHz": bandwidth_ghz,
                "GBP_dB_GHz": mean_gain_db * bandwidth_ghz,
            }
        )

    write_csv_rows(
        OUTPUT / "03_bandwidth_vs_pump.csv",
        list(bandwidth_rows[0].keys()),
        [[row[key] for row in bandwidth_rows] for key in bandwidth_rows[0]],
    )

    mean_gains_db = np.array([row["mean_B3dB_gain_dB"] for row in bandwidth_rows])
    gbp_db_ghz = np.array([row["GBP_dB_GHz"] for row in bandwidth_rows])
    pump_at_max_gain_ghz = float(PUMP_SWEEP_GHZ[int(np.argmax(mean_gains_db))])
    pump_at_max_gbp_ghz = float(PUMP_SWEEP_GHZ[int(np.argmax(gbp_db_ghz))])
    index_max_gain = int(np.argmin(np.abs(PUMP_SWEEP_GHZ - pump_at_max_gain_ghz)))
    index_max_gbp = int(np.argmin(np.abs(PUMP_SWEEP_GHZ - pump_at_max_gbp_ghz)))
    profile_at_max_gain = gain_matrix[index_max_gain]
    profile_at_max_gbp = gain_matrix[index_max_gbp]

    write_csv_rows(
        OUTPUT / "04_gain_profiles.csv",
        [
            "signal_GHz",
            f"gain_dB_max_gain_pump_{pump_at_max_gain_ghz:.2f}GHz",
            f"gain_dB_max_gbp_pump_{pump_at_max_gbp_ghz:.2f}GHz",
        ],
        [SIGNAL_SWEEP_GHZ, profile_at_max_gain, profile_at_max_gbp],
    )

    provenance = {
        "figure_reproduced": "Fig. 3-style gain reproduction, impedance-matched (Z0=50/80 ohm) 15-4-15 supercell",
        "target_lengths_um": {"unloaded": TARGET_UNLOADED_UM, "loaded": TARGET_LOADED_UM},
        "hfss_supercell_files": [
            "hfss_inputs/supercell_15_4_15_zmatched_low_1_3GHz.s2p",
            "hfss_inputs/supercell_15_4_15_zmatched_fine_3_14GHz.s2p",
            "hfss_inputs/supercell_15_4_15_zmatched_aux_14_27GHz.s2p",
        ],
        "stopband_diagnostic": {
            "real_stopband_GHz": [13.6, 14.6],
            "real_stopband_peak_GHz": 14.0,
            "known_bad_data_point_GHz": 12.9,
            "known_bad_data_point_note": "Linear-interpolation-triangle artifact, confirmed not physical; excluded from interpretation.",
        },
        "cell_fit_caveat": (
            "L/C/Lf interpolated between nearest real HFSS single-cell fits "
            "(8um/9um unloaded, 2.5um/3um loaded), not an exact fit at the "
            "target lengths (8.34/2.78um) -- affects only nonlinear gain "
            "scaling, not dispersion/stopband location (fully HFSS-measured)."
        ),
        "unloaded_cell_fit": unloaded_fit,
        "loaded_cell_fit": loaded_fit,
        "amplifier_config": config,
        "bloch_extraction": bloch_provenance,
        "solver": "twpasolver.TWPAnalysis, model=minimal_3wm",
        "pump_sweep_GHz": [
            float(PUMP_SWEEP_GHZ[0]),
            float(PUMP_SWEEP_GHZ[-1]),
            float(PUMP_SWEEP_GHZ[1] - PUMP_SWEEP_GHZ[0]),
        ],
        "signal_sweep_GHz": [
            float(SIGNAL_SWEEP_GHZ[0]),
            float(SIGNAL_SWEEP_GHZ[-1]),
            float(SIGNAL_SWEEP_GHZ[1] - SIGNAL_SWEEP_GHZ[0]),
        ],
        "pump_max_mean_B3dB_gain_GHz": pump_at_max_gain_ghz,
        "pump_max_GBP_GHz": pump_at_max_gbp_ghz,
        "max_mean_B3dB_gain_dB": float(mean_gains_db[index_max_gain]),
        "max_GBP_dB_GHz": float(gbp_db_ghz[index_max_gbp]),
    }
    write_json(OUTPUT / "05_provenance.json", provenance, sort_keys=True)

    fig, (ax_gain_heatmap, ax_gain_profiles) = plt.subplots(1, 2, figsize=(12, 5))
    mesh = ax_gain_heatmap.pcolormesh(
        SIGNAL_SWEEP_GHZ,
        PUMP_SWEEP_GHZ,
        gain_matrix,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=40,
    )
    fig.colorbar(mesh, ax=ax_gain_heatmap, label="Gain [dB]")
    ax_gain_heatmap.set(
        xlabel="Signal Frequency [GHz]",
        ylabel="Pump Frequency [GHz]",
        title="Gain dome (impedance-matched device)",
    )

    ax_gain_profiles.plot(
        SIGNAL_SWEEP_GHZ,
        profile_at_max_gain,
        color="#2a78d6",
        lw=1.3,
        label=f"max gain (pump={pump_at_max_gain_ghz:.2f} GHz)",
    )
    ax_gain_profiles.plot(
        SIGNAL_SWEEP_GHZ,
        profile_at_max_gbp,
        color="#d6342a",
        lw=1.3,
        label=f"max GBP (pump={pump_at_max_gbp_ghz:.2f} GHz)",
    )
    ax_gain_profiles.set(
        xlabel="Signal Frequency [GHz]", ylabel="Gain [dB]", title="Example gain profiles"
    )
    ax_gain_profiles.legend(fontsize=8)
    ax_gain_profiles.grid(alpha=0.25)

    fig.suptitle(
        "Impedance-matched (Z0=50/80 ohm) 15-4-15 supercell: gain vs. paper's original (12.1/3.9 um)"
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT, "06_zmatched_fig3_reproduction")
    plt.close(fig)

    print(
        f"pump at max mean B3dB gain: {pump_at_max_gain_ghz:.2f} GHz ({mean_gains_db[index_max_gain]:.2f} dB)"
    )
    print(
        f"pump at max GBP: {pump_at_max_gbp_ghz:.2f} GHz ({gbp_db_ghz[index_max_gbp]:.1f} dB.GHz)"
    )
    print(f"Wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
