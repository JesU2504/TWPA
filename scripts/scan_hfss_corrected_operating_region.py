#!/usr/bin/env python3
"""Offline operating-point scan using the measured HFSS supercell dispersion.

This is deliberately a provisional optimization: the physical supercell phase
is only available every 0.5 GHz.  The script therefore searches for a broad,
low-ripple operating region, rather than claiming a final optimum.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "scripts")]

from analyze_hfss_supercell import (  # noqa: E402
    SUPERCELL_CELLS,
    cascade_pattern,
    phase_delay,
    read_touchstone,
)
from fit_hfss_unit_cells import lclf_s  # noqa: E402
from scan_hfss_loading_pattern import make_cell  # noqa: E402
from twpa_project_utils import gain_metrics, load_hfss_cell_parameters  # noqa: E402
from twpasolver import TWPAnalysis  # noqa: E402
from twpasolver.logger import log  # noqa: E402
from twpasolver.models import TWPA  # noqa: E402


ISTAR_A = 2e-3
ISIGNAL_A = 1.4e-6
SUPERCELLS = 1407


def add_metrics(signals: np.ndarray, gain: np.ndarray) -> dict[str, float | bool]:
    metrics = gain_metrics(signals, gain, band=(4.0, 8.0))
    metrics["minimum_in_band_gain_db"] = float(np.min(gain))
    metrics["maximum_in_band_gain_db"] = float(np.max(gain))
    metrics["strict_qualified"] = bool(
        metrics["minimum_in_band_gain_db"] >= 17.0
        and metrics["ripple_peak_to_peak_db"] <= 3.0
    )
    return metrics


def score(row: dict[str, float | bool]) -> tuple[float, ...]:
    """Prefer specification margin, then moderate gain and lower current."""
    minimum = float(row["minimum_in_band_gain_db"])
    ripple = float(row["ripple_peak_to_peak_db"])
    mean = float(row["mean_in_band_gain_db"])
    return (
        float(bool(row["strict_qualified"])),
        -max(0.0, 17.0 - minimum),
        -max(0.0, ripple - 3.0),
        -abs(mean - 22.0),
        -ripple,
        -(float(row["Idc_A"]) + float(row["Ip0_A"])),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    log.setLevel(logging.WARNING)
    inputs = ROOT / "hfss_inputs"
    output = ROOT / "results" / "step_14_offline_operating_region"
    output.mkdir(parents=True, exist_ok=True)

    cells, provenance = load_hfss_cell_parameters(inputs / "hfss_cell_parameters.csv")
    measured_f, measured_s = read_touchstone(inputs / "supercell_13_3_13_coarse.s2p")

    def fitted_cell(role: str) -> np.ndarray:
        row = cells[role]
        return lclf_s(
            measured_f * 1e9,
            row["L_pH"] * 1e-12,
            row["C_fF"] * 1e-15,
            row["Lf_pH"] * 1e-12,
        )

    lumped_supercell = cascade_pattern(fitted_cell("unloaded"), fitted_cell("loaded"))
    measured_phase, _ = phase_delay(measured_s[:, 1, 0], measured_f)
    lumped_phase, _ = phase_delay(lumped_supercell[:, 1, 0], measured_f)
    k_correction_samples = ((-measured_phase) - (-lumped_phase)) / SUPERCELL_CELLS

    twpa = TWPA(
        cells=[
            make_cell(cells["unloaded"], 13),
            make_cell(cells["loaded"], 3),
            make_cell(cells["unloaded"], 13),
        ],
        N=SUPERCELLS,
        Istar=ISTAR_A,
        Idc=350e-6,
        Ip0=340e-6,
    )
    analysis = TWPAnalysis(twpa=twpa, f_arange=(0.2, 20.0, 0.002), unit="GHz")

    def prepare(idc_a: float, ip0_a: float) -> None:
        analysis.twpa.Idc = float(idc_a)
        analysis.twpa.Ip0 = float(ip0_a)
        analysis.update_base_data()
        dense_f = np.asarray(analysis.data["freqs"])
        correction = np.zeros_like(dense_f)
        valid = (dense_f >= measured_f[0]) & (dense_f <= measured_f[-1])
        correction[valid] = np.interp(
            dense_f[valid], measured_f, k_correction_samples
        )
        analysis.data["k"] = np.asarray(analysis.data["k"]) + correction

    def evaluate(
        idc_a: float, ip0_a: float, pump_ghz: float, signals: np.ndarray
    ) -> tuple[dict[str, object], np.ndarray] | None:
        try:
            gain = np.asarray(
                analysis.gain(
                    signals,
                    pump=float(pump_ghz),
                    Is0=ISIGNAL_A,
                    Ip0=None,
                    model="minimal_3wm",
                    mode_array_config="basic_3wm",
                    thin=300,
                    save=False,
                )["gain_db"],
                dtype=float,
            )
        except Exception:
            return None
        if not np.all(np.isfinite(gain)):
            return None
        row: dict[str, object] = {
            "pump_GHz": float(pump_ghz),
            "Idc_A": float(idc_a),
            "Ip0_A": float(ip0_a),
            **add_metrics(signals, gain),
        }
        return row, gain

    # Stage 1: broad currents and pump frequency, with a coarser signal grid.
    coarse_signals = np.arange(4.0, 8.0001, 0.2)
    coarse_pumps = np.arange(12.96, 13.3001, 0.02)
    coarse_idc = np.arange(275.0, 400.1, 25.0) * 1e-6
    coarse_ip = np.arange(280.0, 380.1, 20.0) * 1e-6
    coarse_rows: list[dict[str, object]] = []
    for idc_a in coarse_idc:
        for ip0_a in coarse_ip:
            prepare(float(idc_a), float(ip0_a))
            for pump in coarse_pumps:
                result = evaluate(float(idc_a), float(ip0_a), float(pump), coarse_signals)
                if result is not None:
                    coarse_rows.append(result[0])
    write_rows(output / "01_coarse_operating_scan.csv", coarse_rows)

    # Retain the best pump for every current pair, then refine around the top six.
    pair_rows: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in coarse_rows:
        pair_rows[(float(row["Idc_A"]), float(row["Ip0_A"]))].append(row)
    pair_best = [max(rows, key=score) for rows in pair_rows.values()]
    seeds = sorted(pair_best, key=score, reverse=True)[:6]

    fine_signals = np.arange(4.0, 8.0001, 0.05)
    fine_pairs: set[tuple[float, float]] = set()
    for seed in seeds[:3]:
        for didc_ua in (-10.0, 0.0, 10.0):
            for dip_ua in (-10.0, 0.0, 10.0):
                fine_pairs.add(
                    (
                        float(seed["Idc_A"]) + didc_ua * 1e-6,
                        float(seed["Ip0_A"]) + dip_ua * 1e-6,
                    )
                )

    fine_rows: list[dict[str, object]] = []
    fine_curves: dict[tuple[float, float, float], np.ndarray] = {}
    for idc_a, ip0_a in sorted(fine_pairs):
        prepare(idc_a, ip0_a)
        nearest_seed = min(
            seeds,
            key=lambda row: abs(float(row["Idc_A"]) - idc_a)
            + abs(float(row["Ip0_A"]) - ip0_a),
        )
        center_pump = float(nearest_seed["pump_GHz"])
        for pump in np.arange(center_pump - 0.04, center_pump + 0.0401, 0.01):
            result = evaluate(idc_a, ip0_a, float(pump), fine_signals)
            if result is None:
                continue
            row, gain = result
            fine_rows.append(row)
            fine_curves[(idc_a, ip0_a, float(pump))] = gain
    write_rows(output / "02_fine_operating_scan.csv", fine_rows)

    # Select a point near the center of a passing region.  A single-point score
    # tends to land directly on the 3 dB contour and is too sensitive to small
    # control changes.  Here we first maximize the observed local pass rate on
    # the fine grid, then use the ordinary performance score as a tie-breaker.
    def local_grid_score(row: dict[str, object]) -> tuple[float, ...]:
        neighbors = [
            candidate
            for candidate in fine_rows
            if abs(float(candidate["Idc_A"]) - float(row["Idc_A"])) < 10.1e-6
            and abs(float(candidate["Ip0_A"]) - float(row["Ip0_A"])) < 10.1e-6
            and abs(float(candidate["pump_GHz"]) - float(row["pump_GHz"])) < 0.0101
        ]
        passing = sum(bool(candidate["strict_qualified"]) for candidate in neighbors)
        fraction = passing / len(neighbors)
        return (
            float(bool(row["strict_qualified"])),
            fraction,
            float(passing),
            *score(row),
        )

    def tolerance_cube(
        center: dict[str, object], *, keep_curves: bool = False
    ) -> tuple[list[dict[str, object]], list[np.ndarray]]:
        rows: list[dict[str, object]] = []
        curves: list[np.ndarray] = []
        for didc_ua in (-10.0, 0.0, 10.0):
            idc_a = float(center["Idc_A"]) + didc_ua * 1e-6
            for dip_ua in (-10.0, 0.0, 10.0):
                ip0_a = float(center["Ip0_A"]) + dip_ua * 1e-6
                prepare(idc_a, ip0_a)
                for dpump in (-0.01, 0.0, 0.01):
                    pump = float(center["pump_GHz"]) + dpump
                    result = evaluate(idc_a, ip0_a, pump, fine_signals)
                    if result is not None:
                        rows.append(result[0])
                        if keep_curves:
                            curves.append(result[1])
        return rows, curves

    # Compare several plausible centers using the complete 27-point cube.
    # This avoids selecting a center just because part of its neighborhood was
    # absent from the staged fine-search grid.
    center_candidates = sorted(fine_rows, key=local_grid_score, reverse=True)[:12]
    center_summaries: list[dict[str, object]] = []
    center_cubes: list[list[dict[str, object]]] = []
    for center in center_candidates:
        cube_rows, _ = tolerance_cube(center)
        passing = [row for row in cube_rows if bool(row["strict_qualified"])]
        summary: dict[str, object] = {
            "pump_GHz": center["pump_GHz"],
            "Idc_A": center["Idc_A"],
            "Ip0_A": center["Ip0_A"],
            "nominal_minimum_gain_db": center["minimum_in_band_gain_db"],
            "nominal_mean_gain_db": center["mean_in_band_gain_db"],
            "nominal_ripple_db": center["ripple_peak_to_peak_db"],
            "qualified_points": len(passing),
            "qualified_fraction": len(passing) / len(cube_rows),
            "worst_minimum_gain_db": min(
                float(row["minimum_in_band_gain_db"]) for row in cube_rows
            ),
            "worst_ripple_db": max(
                float(row["ripple_peak_to_peak_db"]) for row in cube_rows
            ),
        }
        center_summaries.append(summary)
        center_cubes.append(cube_rows)

    def robust_center_score(index: int) -> tuple[float, ...]:
        summary = center_summaries[index]
        return (
            float(summary["qualified_fraction"]),
            float(summary["qualified_points"]),
            -max(0.0, float(summary["worst_ripple_db"]) - 3.0),
            float(summary["worst_minimum_gain_db"]),
            *score(center_candidates[index]),
        )

    selected_index = max(range(len(center_candidates)), key=robust_center_score)
    selected = center_candidates[selected_index]
    robustness_rows = center_cubes[selected_index]
    _, robustness_curves = tolerance_cube(selected, keep_curves=True)
    selected_key = (
        float(selected["Idc_A"]),
        float(selected["Ip0_A"]),
        float(selected["pump_GHz"]),
    )
    selected_gain = fine_curves[selected_key]
    write_rows(output / "03_robust_center_comparison.csv", center_summaries)
    write_rows(output / "03_local_robustness_scan.csv", robustness_rows)

    robust_array = np.asarray(robustness_curves)
    passing_fine = [row for row in fine_rows if bool(row["strict_qualified"])]
    passing_robust = [row for row in robustness_rows if bool(row["strict_qualified"])]
    report = {
        "status": "provisional_from_0p5GHz_spaced_HFSS_phase_data",
        "specification": {
            "signal_band_GHz": [4.0, 8.0],
            "minimum_gain_db": 17.0,
            "maximum_ripple_db": 3.0,
        },
        "search": {
            "coarse_evaluated_points": len(coarse_rows),
            "fine_evaluated_points": len(fine_rows),
            "fine_current_pairs": len(fine_pairs),
            "fine_qualified_points": len(passing_fine),
        },
        "selected": selected,
        "local_robustness": {
            "perturbations": {
                "pump_GHz": [-0.01, 0.0, 0.01],
                "Idc_uA": [-10.0, 0.0, 10.0],
                "Ip0_uA": [-10.0, 0.0, 10.0],
            },
            "evaluated_points": len(robustness_rows),
            "qualified_points": len(passing_robust),
            "qualified_fraction": len(passing_robust) / len(robustness_rows),
            "worst_minimum_gain_db": min(
                float(row["minimum_in_band_gain_db"]) for row in robustness_rows
            ),
            "worst_ripple_db": max(
                float(row["ripple_peak_to_peak_db"]) for row in robustness_rows
            ),
        },
        "limitations": [
            "HFSS phase samples are spaced by 0.5 GHz and linearly interpolated.",
            "Gain uses the minimal undepleted-pump 3WM model.",
            "The robustness cube varies operating controls, not fabrication geometry.",
        ],
        "hfss_cell_provenance": provenance,
    }
    (output / "04_operating_region_assessment.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output / "04_selected_gain.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_GHz", "selected_gain_dB"])
        writer.writerows(zip(fine_signals, selected_gain))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(fine_signals, selected_gain, linewidth=2, label="Selected setting")
    ax.axhline(17, color="k", linestyle=":", label="17 dB minimum")
    ax.fill_between(
        fine_signals,
        np.min(robust_array, axis=0),
        np.max(robust_array, axis=0),
        alpha=0.25,
        label="±10 µA, ±0.01 GHz envelope",
    )
    ax.set(xlabel="Signal frequency (GHz)", ylabel="Gain (dB)", title="Gain and local operating tolerance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sc = ax.scatter(
        [float(row["minimum_in_band_gain_db"]) for row in fine_rows],
        [float(row["ripple_peak_to_peak_db"]) for row in fine_rows],
        c=[float(row["pump_GHz"]) for row in fine_rows],
        s=18,
        cmap="viridis",
    )
    ax.axvline(17, color="k", linestyle=":")
    ax.axhline(3, color="k", linestyle=":")
    ax.set(xlabel="Minimum 4–8 GHz gain (dB)", ylabel="Ripple (dB)", title="Fine-search trade-off")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Pump frequency (GHz)")

    ax = axes[1, 0]
    seed_idc = float(selected["Idc_A"])
    seed_ip = float(selected["Ip0_A"])
    nearby = [
        row for row in fine_rows
        if abs(float(row["Idc_A"]) - seed_idc) < 0.1e-6
        and abs(float(row["Ip0_A"]) - seed_ip) < 0.1e-6
    ]
    nearby.sort(key=lambda row: float(row["pump_GHz"]))
    ax.plot(
        [float(row["pump_GHz"]) for row in nearby],
        [float(row["ripple_peak_to_peak_db"]) for row in nearby],
        marker="o",
        label="Ripple",
    )
    ax.axhline(3, color="k", linestyle=":")
    ax.set(xlabel="Pump frequency (GHz)", ylabel="Ripple (dB)", title="Pump-frequency sensitivity")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    q = np.array([bool(row["strict_qualified"]) for row in robustness_rows])
    ax.hist(
        [float(row["ripple_peak_to_peak_db"]) for row in robustness_rows],
        bins=10,
        color="C0",
        alpha=0.8,
    )
    ax.axvline(3, color="k", linestyle=":", label="3 dB limit")
    ax.set(xlabel="Ripple in local tolerance cube (dB)", ylabel="Count", title=f"Local pass rate: {q.sum()}/{len(q)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "HFSS-dispersion-corrected offline operating-region search\n"
        f"Selected: pump {float(selected['pump_GHz']):.3f} GHz, "
        f"Idc {float(selected['Idc_A'])*1e6:.0f} µA, "
        f"Ip {float(selected['Ip0_A'])*1e6:.0f} µA",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / "05_offline_operating_region.png", dpi=180)
    fig.savefig(output / "05_offline_operating_region.pdf")
    plt.close(fig)

    readme = f"""# Offline HFSS-corrected operating-region scan

This scan uses the measured physical-supercell phase correction and the minimal
3WM gain model.  It is provisional because the HFSS phase data are spaced by
0.5 GHz.

## Selected provisional setting

- Pump: {float(selected['pump_GHz']):.3f} GHz
- DC bias: {float(selected['Idc_A'])*1e6:.0f} uA
- Pump current: {float(selected['Ip0_A'])*1e6:.0f} uA
- Minimum 4-8 GHz gain: {float(selected['minimum_in_band_gain_db']):.2f} dB
- Mean 4-8 GHz gain: {float(selected['mean_in_band_gain_db']):.2f} dB
- Ripple: {float(selected['ripple_peak_to_peak_db']):.2f} dB
- Strict target passed: {bool(selected['strict_qualified'])}

## Local tolerance check

The 27-point cube varies pump frequency by +/-0.01 GHz and both currents by
+/-10 uA.  {len(passing_robust)} of {len(robustness_rows)} points pass both the
17 dB minimum-gain and 3 dB ripple requirements.

This result identifies where to operate and what to verify with a fine HFSS
sweep.  It is not yet a final device guarantee.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
