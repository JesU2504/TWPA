#!/usr/bin/env python3
"""Baseline pump-frequency refinement for a target bandwidth of 3.15 GHz."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from twpasolver import TWPAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from device_models import build_twpa_from_hfss  # noqa: E402
from hfss_bloch import apply_bloch_parameters  # noqa: E402
from io_utils import read_touchstone, write_csv_rows  # noqa: E402
from reproduce_howe_fig3_hfss_bloch import recompute_k_star  # noqa: E402

INPUTS = ROOT / "hfss_inputs"
OUTPUT = ROOT / "results" / "refined_pump"
TARGET_BW_GHZ = 3.15
PUMPS_GHZ = np.arange(13.80, 13.9201, 0.005)
SIGNALS_GHZ = np.arange(2.50, 11.5001, 0.01)


def load_supercell():
    low_f, low_s = read_touchstone(INPUTS / "supercell_15_4_15_low_1_3GHz.s2p")
    fine_f, fine_s = read_touchstone(INPUTS / "supercell_15_4_15_fine_3_14GHz.s2p")
    aux_f, aux_s = read_touchstone(INPUTS / "supercell_15_4_15_aux_14_27GHz.s2p")
    freqs = np.concatenate((low_f, fine_f[1:], aux_f[1:]))
    sparams = np.concatenate((low_s, fine_s[1:], aux_s[1:]), axis=0)
    return freqs, sparams


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    twpa, provenance = build_twpa_from_hfss(
        INPUTS / "hfss_cell_parameters.csv", INPUTS / "amplifier_config.json"
    )
    config = provenance["amplifier_config"]
    analysis = TWPAnalysis(
        twpa=twpa,
        f_arange=(
            config["base_frequency_start_GHz"],
            config["base_frequency_stop_GHz"],
            config["base_frequency_step_GHz"],
        ),
        unit="GHz",
    )
    analysis.update_base_data()
    measured_f, measured_s = load_supercell()
    bloch = apply_bloch_parameters(analysis.data, measured_f, measured_s, 34)
    analysis.data["k_star"] = recompute_k_star(
        np.asarray(analysis.data["freqs"]),
        np.asarray(analysis.data["k"]),
        (3.0, 8.0),
    )

    rows = []
    for pump in PUMPS_GHZ:
        result = analysis.bandwidth(
            gain_reduction=3.0,
            signal_freqs=SIGNALS_GHZ,
            pump=float(pump),
            Is0=float(config["Is0_A"]),
            model="minimal_3wm",
            thin=400,
        )
        rows.append(
            {
                "pump_GHz": float(pump),
                "mean_B3dB_gain_dB": float(result["mean_gain"]),
                "B3dB_bandwidth_GHz": float(result["total_bw"]),
            }
        )

    best = dict(min(rows, key=lambda row: abs(row["B3dB_bandwidth_GHz"] - TARGET_BW_GHZ)))
    gain = analysis.gain(
        SIGNALS_GHZ,
        pump=best["pump_GHz"],
        Is0=float(config["Is0_A"]),
        model="minimal_3wm",
        thin=400,
    )
    gain_db = np.asarray(gain["gain_db"], dtype=float)
    best["peak_gain_dB"] = float(np.max(gain_db))
    best["target_bandwidth_GHz"] = TARGET_BW_GHZ
    best["absolute_bandwidth_error_GHz"] = abs(best["B3dB_bandwidth_GHz"] - TARGET_BW_GHZ)

    with open(OUTPUT / "refined_pump_sweep.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_csv_rows(
        OUTPUT / "refined_model_gain_profile.csv",
        ["signal_GHz", "gain_dB"],
        [SIGNALS_GHZ, gain_db],
    )
    (OUTPUT / "refined_result.json").write_text(
        json.dumps({"best": best, "bloch": bloch}, indent=2, default=str)
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
