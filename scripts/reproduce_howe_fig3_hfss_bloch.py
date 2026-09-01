#!/usr/bin/env python3
"""HFSS–Bloch dispersion and minimal-3WM gain for the 15–4–15 supercell.

The baseline exports cover 1–27 GHz and use 12.1 µm long stubs. The structural
single-cell fit uses 12.0 µm stubs; Bloch data overrides its propagation inside
the simulated band. The operating point comes from amplifier_config.json.
Result step 34 supplies the gain profiles used by the reflection calculation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from twpasolver import TWPAnalysis
from twpasolver.models import TWPA, LCLfBaseCell

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from device_models import build_twpa_from_hfss  # noqa: E402
from hfss_bloch import apply_bloch_parameters  # noqa: E402
from io_utils import read_touchstone, write_csv_rows  # noqa: E402
from plotting import save_figure  # noqa: E402

OUTPUT = ROOT / "results" / "bloch_corrected" / "step_34_howe_fig3_hfss_bloch_reproduction"
INPUTS = ROOT / "hfss_inputs"
CELLS_PER_SUPERCELL = 34
GAIN_REDUCTION_DB = 3.0
# Match the solver's nondispersive reference using the corrected Bloch data.
K_STAR_FIT_BAND_GHZ = (3.0, 8.0)

# The pump range covers the full gain dome. Limiting profile signals to
# 11.5 GHz excludes the isolated loaded-cell resonance near 12 GHz.
PUMP_SWEEP_GHZ = np.arange(12.5, 17.51, 0.1)
SIGNAL_SWEEP_GHZ = np.arange(2.5, 11.51, 0.05)
# The wider heatmap is masked where the idler falls below the 1 GHz HFSS limit.
HEATMAP_SIGNAL_SWEEP_GHZ = np.arange(2.5, 14.01, 0.05)
HFSS_MIN_FREQUENCY_GHZ = 1.0


def recompute_k_star(
    freqs_ghz: np.ndarray, k: np.ndarray, fit_band_ghz: tuple[float, float]
) -> np.ndarray:
    """Dispersion after subtracting the constant and linear terms of a quadratic fit.

    The fit uses the specified GHz interval and the current Bloch-corrected k array."""
    mask = (freqs_ghz >= fit_band_ghz[0]) & (freqs_ghz <= fit_band_ghz[1])
    fit_coefficients = np.polyfit(freqs_ghz[mask], k[mask], 2)
    return k - np.polyval(fit_coefficients[1:], freqs_ghz)


def minimal_3wm_phase_mismatch(
    analysis: TWPAnalysis, pumps_ghz: np.ndarray, signals_ghz: np.ndarray
) -> np.ndarray:
    """Undepleted-pump phase mismatch, including pump-induced phase shifts.

    Returns rad/cell with shape (n_pumps, n_signals). The idler is fp - fs."""
    freqs_ghz = np.asarray(analysis.data["freqs"], dtype=float)
    k = np.asarray(analysis.data["k"], dtype=float)
    signal_grid, pump_grid = np.meshgrid(signals_ghz, pumps_ghz)
    idler_grid = pump_grid - signal_grid
    pump_k = np.interp(pump_grid, freqs_ghz, k)
    signal_k = np.interp(signal_grid, freqs_ghz, k)
    idler_k = np.interp(idler_grid, freqs_ghz, k)
    self_phase_coefficient = analysis.twpa.Ip0**2 * analysis.twpa.xi / 8
    return (
        pump_k - signal_k - idler_k + self_phase_coefficient * (pump_k - 2 * signal_k - 2 * idler_k)
    )


def load_measured_supercell_s_parameters() -> tuple[np.ndarray, np.ndarray]:
    """Stitched 1–27 GHz HFSS data, rejecting inconsistent band overlaps."""
    low_freq, low_s = read_touchstone(INPUTS / "supercell_15_4_15_low_1_3GHz.s2p")
    fine_freq, fine_s = read_touchstone(INPUTS / "supercell_15_4_15_fine_3_14GHz.s2p")
    aux_freq, aux_s = read_touchstone(INPUTS / "supercell_15_4_15_aux_14_27GHz.s2p")
    if not np.isclose(low_freq[-1], fine_freq[0]) or not np.allclose(
        low_s[-1], fine_s[0], atol=1e-12
    ):
        raise ValueError("The 15-4-15 low/fine HFSS sweeps disagree at their 3 GHz overlap")
    if not np.isclose(fine_freq[-1], aux_freq[0]) or not np.allclose(
        fine_s[-1], aux_s[0], atol=1e-12
    ):
        raise ValueError("The 15-4-15 fine/aux HFSS sweeps disagree at their 14 GHz overlap")
    frequencies_ghz = np.concatenate((low_freq, fine_freq[1:], aux_freq[1:]))
    s_matrices = np.concatenate((low_s, fine_s[1:], aux_s[1:]), axis=0)
    return frequencies_ghz, s_matrices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

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
    bloch_provenance = apply_bloch_parameters(
        engineered.data, measured_freq_ghz, measured_s, CELLS_PER_SUPERCELL
    )
    # The Bloch correction updates k but leaves k_star cached from the cell model.
    engineered.data["k_star"] = recompute_k_star(
        np.asarray(engineered.data["freqs"]), np.asarray(engineered.data["k"]), K_STAR_FIT_BAND_GHZ
    )

    # No bare-line HFSS export exists; this reference uses the fitted analytic cell.
    unloaded_cell = twpa.cells[0]
    bare_cell = LCLfBaseCell(
        C=unloaded_cell.C, L=unloaded_cell.L, Lf=unloaded_cell.Lf, N=CELLS_PER_SUPERCELL
    )
    bare_twpa = TWPA(cells=[bare_cell], N=twpa.N, Istar=twpa.Istar, Idc=twpa.Idc, Ip0=twpa.Ip0)
    unmodulated = TWPAnalysis(
        twpa=bare_twpa,
        f_arange=(
            config["base_frequency_start_GHz"],
            config["base_frequency_stop_GHz"],
            config["base_frequency_step_GHz"],
        ),
        unit="GHz",
    )
    unmodulated.update_base_data()

    all_freqs_ghz = np.asarray(engineered.data["freqs"])
    # Restrict reported engineered dispersion to the HFSS band, excluding fallback data.
    bloch_band_mask = (all_freqs_ghz >= 1.0) & (all_freqs_ghz <= 27.0)
    freqs_ghz = all_freqs_ghz[bloch_band_mask]
    # The cached S21 belongs to the cell model. Accumulated Bloch evanescence
    # supplies the stopband transmission proxy, excluding finite-length reflections.
    evanescence_per_cell = np.asarray(engineered.data["bloch_evanescence"])[bloch_band_mask]
    s21_db_engineered = -20 * np.log10(np.e) * twpa.N_tot * evanescence_per_cell
    k_star_engineered = np.asarray(engineered.data["k_star"])[bloch_band_mask]
    k_star_bare = np.asarray(unmodulated.data["k_star"])

    write_csv_rows(
        OUTPUT / "01_dispersion.csv",
        ["freq_GHz", "S21_db_engineered", "k_star_engineered_rad"],
        [freqs_ghz, s21_db_engineered, k_star_engineered],
    )
    write_csv_rows(
        OUTPUT / "01b_dispersion_bare_reference.csv",
        ["freq_GHz", "k_star_bare_rad_theoretical"],
        [all_freqs_ghz, k_star_bare],
    )

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

    phase_mismatch = minimal_3wm_phase_mismatch(engineered, PUMP_SWEEP_GHZ, SIGNAL_SWEEP_GHZ)

    # Valid signal limits vary with pump because fi = fp - fs must remain >= 1 GHz.
    # Keep the profile grid separate for the downstream bandwidth calculation.
    gain_heatmap_extended = np.full((len(PUMP_SWEEP_GHZ), len(HEATMAP_SIGNAL_SWEEP_GHZ)), np.nan)
    for pump_index, pump_ghz in enumerate(PUMP_SWEEP_GHZ):
        idlers_ghz = pump_ghz - HEATMAP_SIGNAL_SWEEP_GHZ
        valid = idlers_ghz >= HFSS_MIN_FREQUENCY_GHZ
        valid_signals = HEATMAP_SIGNAL_SWEEP_GHZ[valid]
        if valid_signals.size:
            result = engineered.gain(
                valid_signals,
                pump=float(pump_ghz),
                Is0=initial_signal_current_a,
                model="minimal_3wm",
                thin=400,
            )
            gain_heatmap_extended[pump_index, valid] = np.asarray(result["gain_db"], dtype=float)
    finite_extended = np.isfinite(gain_heatmap_extended)
    if not np.any(finite_extended):
        raise FloatingPointError("Extended panel-(c) sweep contains no valid points")

    phase_mismatch_extended = minimal_3wm_phase_mismatch(
        engineered, PUMP_SWEEP_GHZ, HEATMAP_SIGNAL_SWEEP_GHZ
    )
    phase_mismatch_extended[~finite_extended] = np.nan

    write_csv_rows(
        OUTPUT / "02_gain_heatmap.csv",
        ["signal_GHz", *[f"pump_{pump:.2f}GHz_gain_dB" for pump in PUMP_SWEEP_GHZ]],
        [SIGNAL_SWEEP_GHZ, *gain_matrix],
    )
    write_csv_rows(
        OUTPUT / "02b_phase_mismatch.csv",
        ["signal_GHz", *[f"pump_{pump:.2f}GHz_delta_k_rad_per_cell" for pump in PUMP_SWEEP_GHZ]],
        [SIGNAL_SWEEP_GHZ, *phase_mismatch],
    )
    write_csv_rows(
        OUTPUT / "02c_gain_heatmap_extended_x14.csv",
        ["signal_GHz", *[f"pump_{pump:.2f}GHz_gain_dB" for pump in PUMP_SWEEP_GHZ]],
        [HEATMAP_SIGNAL_SWEEP_GHZ, *gain_heatmap_extended],
    )
    write_csv_rows(
        OUTPUT / "02d_phase_mismatch_extended_x14.csv",
        ["signal_GHz", *[f"pump_{pump:.2f}GHz_delta_k_rad_per_cell" for pump in PUMP_SWEEP_GHZ]],
        [HEATMAP_SIGNAL_SWEEP_GHZ, *phase_mismatch_extended],
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

    provenance_out = {
        "figure_reproduced": "Fig. 3, Howe et al. 2507.07706 (Simulating a full device)",
        "data_source": "real HFSS 15-4-15 supercell full-wave simulation, Bloch-corrected",
        "hfss_supercell_files": [
            "hfss_inputs/supercell_15_4_15_low_1_3GHz.s2p",
            "hfss_inputs/supercell_15_4_15_fine_3_14GHz.s2p",
            "hfss_inputs/supercell_15_4_15_aux_14_27GHz.s2p",
        ],
        "hfss_stub_length_um": 12.1,
        "reported_engineered_dispersion_band_GHz": [1.0, 27.0],
        "cell_placeholder_role": (
            "hfss_inputs/hfss_cell_parameters.csv (single-cell fit at "
            "stub_length=12.0 um, not the corrected 12.1 um) is used only to "
            "instantiate twpasolver's structural cell count (N_tot) and "
            "current-nonlinearity scaling; it contributes no value shown in "
            "any panel or CSV. All reported engineered-line dispersion, gain, "
            "bandwidth, and GBP values come only from the 15-4-15 HFSS "
            "supercell measurement (apply_bloch_parameters), clipped to its "
            "1-27 GHz measured band."
        ),
        "amplifier_config": config,
        "cell_placeholder_provenance": provenance,
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
        "panel_c_signal_sweep_GHz": [
            float(HEATMAP_SIGNAL_SWEEP_GHZ[0]),
            float(HEATMAP_SIGNAL_SWEEP_GHZ[-1]),
            float(HEATMAP_SIGNAL_SWEEP_GHZ[1] - HEATMAP_SIGNAL_SWEEP_GHZ[0]),
        ],
        "panel_c_validity_mask": "idler_GHz = pump_GHz - signal_GHz >= 1.0 (HFSS lower band edge)",
        "pump_max_mean_B3dB_gain_GHz": pump_at_max_gain_ghz,
        "pump_max_GBP_GHz": pump_at_max_gbp_ghz,
        "max_mean_B3dB_gain_dB": float(mean_gains_db[index_max_gain]),
        "max_GBP_dB_GHz": float(gbp_db_ghz[index_max_gbp]),
    }
    (OUTPUT / "05_provenance.json").write_text(
        json.dumps(provenance_out, indent=2, sort_keys=True, default=str) + "\n"
    )

    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.0])
    ax_s21, ax_k_star = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    ax_gain_heatmap, ax_gain_profiles = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    ax_mean_gain, ax_gbp = fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])

    ax_s21.plot(freqs_ghz, s21_db_engineered, color="black", lw=1.0)
    ax_s21.set(
        ylabel="|S21| [dB]", title="(a) HFSS-simulated frequency range", xlim=(1, 27), ylim=(-80, 5)
    )
    ax_s21.grid(alpha=0.25)

    ax_k_star.plot(
        all_freqs_ghz, k_star_bare * 1e3, "--", color="gray", lw=1.2, label="bare (theoretical)"
    )
    ax_k_star.plot(
        freqs_ghz, k_star_engineered * 1e3, color="black", lw=1.2, label="disp. eng. (HFSS-derived)"
    )
    for band_edge_ghz in (1.0, 27.0):
        ax_k_star.axvline(band_edge_ghz, color="#2a78d6", ls=":", lw=0.8)
    ax_k_star.set(xlabel="Frequency [GHz]", ylabel="k* [mrad]", title="(b)", xlim=(0, 30))
    ax_k_star.legend(fontsize=8, frameon=False)
    ax_k_star.grid(alpha=0.25)

    heatmap_cmap = plt.get_cmap("inferno").copy()
    heatmap_cmap.set_bad("#000000")
    mesh = ax_gain_heatmap.pcolormesh(
        HEATMAP_SIGNAL_SWEEP_GHZ,
        PUMP_SWEEP_GHZ,
        np.ma.masked_invalid(gain_heatmap_extended),
        shading="auto",
        cmap=heatmap_cmap,
        vmin=0,
        vmax=40,
    )
    ax_gain_heatmap.contour(
        HEATMAP_SIGNAL_SWEEP_GHZ,
        PUMP_SWEEP_GHZ,
        np.ma.masked_invalid(phase_mismatch_extended),
        levels=[0.0],
        colors="#55d8ff",
        linewidths=1.25,
    )
    ax_gain_heatmap.plot([], [], color="#55d8ff", lw=1.25, label=r"$\Delta k_{\mathrm{eff}}=0$")
    fig.colorbar(mesh, ax=ax_gain_heatmap, label="Gain [dB]", location="top", shrink=0.9)
    ax_gain_heatmap.set(
        xlabel="Signal Frequency [GHz]",
        ylabel="Pump Frequency [GHz]",
        title="(c)",
        xlim=(2.5, 14.0),
    )
    ax_gain_heatmap.legend(
        loc="upper right",
        fontsize=7.0,
        labelcolor="white",
        facecolor="black",
        framealpha=0.35,
        edgecolor="none",
    )

    ax_gain_profiles.plot(
        SIGNAL_SWEEP_GHZ, profile_at_max_gain, color="#2a78d6", lw=1.3, label="max gain"
    )
    ax_gain_profiles.plot(
        SIGNAL_SWEEP_GHZ, profile_at_max_gbp, color="#d6342a", lw=1.3, label="max GBP"
    )
    ax_gain_profiles.set(xlabel="Signal Frequency [GHz]", ylabel="Gain [dB]", title="(d)")
    ax_gain_profiles.legend(fontsize=8, frameon=False)
    ax_gain_profiles.grid(alpha=0.25)

    ax_mean_gain.plot(PUMP_SWEEP_GHZ, mean_gains_db, color="black", lw=1.3)
    ax_mean_gain.axvline(pump_at_max_gbp_ghz, color="gray", ls=":", lw=1.0)
    ax_mean_gain.set(ylabel="Mean B3dB Gain [dB]", title="(e)")
    ax_mean_gain.grid(alpha=0.25)

    ax_gbp.plot(PUMP_SWEEP_GHZ, gbp_db_ghz, color="black", lw=1.3)
    ax_gbp.axvline(pump_at_max_gbp_ghz, color="gray", ls=":", lw=1.0)
    ax_gbp.set(xlabel="Pump Frequency [GHz]", ylabel="GBP [dB.GHz]", title="(f)")
    ax_gbp.grid(alpha=0.25)

    fig.suptitle(
        "Reproduction of Fig. 3 (Howe et al. 2507.07706) -- real HFSS 15-4-15 supercell, Bloch-corrected (stub=12.1 um)",
        fontsize=10.5,
    )
    save_figure(fig, OUTPUT, "06_howe_fig3_hfss_bloch_reproduction")
    plt.close(fig)

    print(
        f"pump at max mean B3dB gain: {pump_at_max_gain_ghz:.2f} GHz ({mean_gains_db[index_max_gain]:.2f} dB)"
    )
    print(
        f"pump at max GBP: {pump_at_max_gbp_ghz:.2f} GHz ({gbp_db_ghz[index_max_gbp]:.1f} dB.GHz)"
    )
    print(f"Saved figure to {OUTPUT / '06_howe_fig3_hfss_bloch_reproduction.png'}")


if __name__ == "__main__":
    main()
