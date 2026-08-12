"""Shared utilities for the KI-TWPA paper-reproduction notebooks.

All notebook-facing frequencies are in GHz.  Circuit quantities passed to
``twpasolver`` use SI units.  The Howe and Faramarzi builders are deliberately
named ``*_surrogate`` because the papers do not publish every fitted lumped
parameter normally obtained from electromagnetic simulation.
"""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from twpasolver import TWPAnalysis
from twpasolver.mathutils import I_to_dBm, dBm_to_I
from twpasolver.models import LCLfBaseCell, LosslessTL, TWPA
from twpasolver.modes_rwa import ModeArrayFactory


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
MODELS_ROOT = ROOT / "models"


PAPERS = {
    "malnou": {
        "title": "Three-Wave Mixing Kinetic Inductance Traveling-Wave Amplifier with Near-Quantum-Limited Noise Performance",
        "doi": "10.1103/PRXQuantum.2.010302",
        "url": "https://doi.org/10.1103/PRXQuantum.2.010302",
    },
    "howe": {
        "title": "Kinetic Inductance Traveling-Wave Parametric Amplifiers Near the Quantum Limit: Methodology and Characterization",
        "doi": "10.1103/yg71-j2dn",
        "url": "https://doi.org/10.1103/yg71-j2dn",
    },
    "faramarzi": {
        "title": "A 4-8 GHz Kinetic Inductance Traveling-Wave Parametric Amplifier Using Four-Wave Mixing",
        "doi": "10.1063/5.0208110",
        "url": "https://doi.org/10.1063/5.0208110",
    },
    "castellanos": {
        "title": "Measurable Improvement in Multi-Qubit Readout Using a Kinetic Inductance Traveling Wave Parametric Amplifier",
        "doi": "10.1109/TASC.2024.3525451",
        "url": "https://doi.org/10.1109/TASC.2024.3525451",
    },
}


HFSS_CELL_COLUMNS = (
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def result_dir(name: str) -> Path:
    path = RESULTS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for package in (
        "twpasolver",
        "numpy",
        "matplotlib",
        "scikit-rf",
        "numba",
        "pydantic",
        "CyRK",
        "nbformat",
        "nbclient",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "units": {
            "frequency": "GHz at notebook interfaces",
            "current": "A internally",
            "power": "dBm",
            "inductance": "H internally",
            "capacitance": "F internally",
        },
        "macos_cyrk_note": (
            "CyRK extensions require an LC_RPATH entry for "
            "/opt/homebrew/opt/libomp/lib on this Apple-silicon environment."
        ),
    }


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return path


def save_curve(path: Path, **columns: Iterable[Any]) -> Path:
    arrays = {name: np.asarray(values) for name, values in columns.items()}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Curve columns have different lengths: {lengths}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(arrays)
        writer.writerows(zip(*(arrays[name] for name in arrays)))
    return path


def save_matrix(
    path: Path,
    row_values: Iterable[float],
    column_values: Iterable[float],
    matrix: np.ndarray,
    *,
    row_name: str = "pump_GHz",
    column_prefix: str = "signal_GHz_",
) -> Path:
    """Save a 2-D sweep in a human-readable wide CSV representation."""
    rows = np.asarray(row_values, dtype=float)
    columns = np.asarray(column_values, dtype=float)
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(rows), len(columns)):
        raise ValueError(
            f"Matrix shape {values.shape} does not match {(len(rows), len(columns))}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_name, *(f"{column_prefix}{x:.6g}" for x in columns)])
        for row_value, row in zip(rows, values):
            writer.writerow([row_value, *row])
    return path


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def save_model_bundle(
    name: str, twpa: TWPA, provenance: dict[str, Any], figures: list[str]
) -> Path:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return save_json(
        MODELS_ROOT / f"{name}.json",
        {
            "provenance": provenance,
            "paper_figures": figures,
            "model": twpa.model_dump(),
        },
    )


def load_hfss_cell_parameters(
    path: str | Path, *, allow_placeholder: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and validate the two fitted cells exported by the HFSS workflow.

    The CSV must contain exactly one ``unloaded`` and one ``loaded`` row. Values
    are stored in human-friendly units in the file and converted to SI only by
    :func:`build_twpa_from_hfss`.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(HFSS_CELL_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"HFSS cell CSV is missing columns: {sorted(missing)}")
        raw_rows = list(reader)
    if len(raw_rows) != 2:
        raise ValueError("HFSS cell CSV must contain exactly two rows")

    numeric = HFSS_CELL_COLUMNS[1:-1]
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        role = raw["cell_type"].strip().lower()
        if role not in {"unloaded", "loaded"} or role in rows:
            raise ValueError("cell_type must contain one unloaded and one loaded row")
        row: dict[str, Any] = {"cell_type": role, "source": raw["source"].strip()}
        for name in numeric:
            try:
                row[name] = float(raw[name])
            except ValueError as exc:
                raise ValueError(f"{role}.{name} is not numeric") from exc
            if not np.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f"{role}.{name} must be finite and non-negative")
        for name in ("stub_length_um", "L_pH", "C_fF", "Lf_pH", "Z0_ohm"):
            if row[name] <= 0:
                raise ValueError(f"{role}.{name} must be positive")
        rows[role] = row

    sources = {row["source"].upper() for row in rows.values()}
    hfss_validated = sources == {"HFSS"}
    if not hfss_validated and not allow_placeholder:
        raise ValueError(
            "Input is not marked HFSS. Replace placeholder rows and set source=HFSS, "
            "or explicitly pass allow_placeholder=True for workflow testing."
        )
    checks = hfss_cell_acceptance_checks(rows)
    metadata = {
        "input_file": str(path.resolve()),
        "hfss_validated": hfss_validated,
        "sources": sorted(sources),
        "acceptance_checks": checks,
        "all_hfss_acceptance_checks_pass": bool(
            hfss_validated and all(checks.values())
        ),
    }
    return rows, metadata


def hfss_cell_acceptance_checks(
    rows: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    """Apply the selected-HFSS-baseline gates from ``HFSS_MODEL_SPEC.md``."""
    unloaded, loaded = rows["unloaded"], rows["loaded"]
    selected_unloaded_z0 = 43.08925347829123
    selected_loaded_z0 = 70.20100727735449
    selected_ratio = selected_loaded_z0 / selected_unloaded_z0
    return {
        "unloaded_Z0_selected_43p09ohm_pm5pct": (
            abs(unloaded["Z0_ohm"] - selected_unloaded_z0)
            <= 0.05 * selected_unloaded_z0
        ),
        "loaded_Z0_selected_70p20ohm_pm5pct": (
            abs(loaded["Z0_ohm"] - selected_loaded_z0)
            <= 0.05 * selected_loaded_z0
        ),
        "impedance_ratio_selected_1p629_pm5pct": (
            abs(loaded["Z0_ohm"] / unloaded["Z0_ohm"] - selected_ratio)
            <= 0.05 * selected_ratio
        ),
        "S21_fit_rmse_below_0p5db": max(
            unloaded["fit_s21_rmse_db"], loaded["fit_s21_rmse_db"]
        ) < 0.5,
        "phase_fit_rmse_below_5deg": max(
            unloaded["fit_phase_rmse_deg"], loaded["fit_phase_rmse_deg"]
        ) < 5.0,
    }


def build_twpa_from_hfss(
    cell_csv: str | Path,
    config_json: str | Path,
    *,
    allow_placeholder: bool = False,
) -> tuple[TWPA, dict[str, Any]]:
    """Construct a complete periodic TWPA from fitted HFSS cell parameters."""
    rows, input_metadata = load_hfss_cell_parameters(
        cell_csv, allow_placeholder=allow_placeholder
    )
    config_path = Path(config_json)
    config = json.loads(config_path.read_text())
    required = {
        "unloaded_cells_each_side",
        "loaded_cells",
        "supercells",
        "Istar_A",
        "Idc_A",
        "Ip0_A",
        "Is0_A",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Amplifier config is missing keys: {sorted(missing)}")
    for key in required:
        value = config[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a non-negative number")
    for key in ("unloaded_cells_each_side", "loaded_cells", "supercells"):
        if int(config[key]) != config[key] or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    def make_cell(role: str, count: int) -> LCLfBaseCell:
        row = rows[role]
        return LCLfBaseCell(
            L=row["L_pH"] * 1e-12,
            C=row["C_fF"] * 1e-15,
            Lf=row["Lf_pH"] * 1e-12,
            delta=row["loss_tangent"],
            N=count,
            centered=bool(config.get("centered_cells", False)),
        )

    side = int(config["unloaded_cells_each_side"])
    loaded_count = int(config["loaded_cells"])
    twpa = TWPA(
        cells=[make_cell("unloaded", side), make_cell("loaded", loaded_count), make_cell("unloaded", side)],
        N=int(config["supercells"]),
        Istar=float(config["Istar_A"]),
        Idc=float(config["Idc_A"]),
        Ip0=float(config["Ip0_A"]),
    )
    expected = (2 * side + loaded_count) * int(config["supercells"])
    if twpa.N_tot != expected:
        raise AssertionError(f"Expected {expected} cells, built {twpa.N_tot}")
    provenance = {
        "status": "HFSS extracted" if input_metadata["hfss_validated"] else "PLACEHOLDER workflow test",
        "warning": None if input_metadata["hfss_validated"] else "These parameters are placeholders and are not HFSS results.",
        "cell_input": input_metadata,
        "amplifier_config_file": str(config_path.resolve()),
        "amplifier_config": config,
        "cells": rows,
        "total_cells": twpa.N_tot,
    }
    return twpa, provenance


def build_malnou_3wm() -> tuple[TWPA, dict[str, Any]]:
    unloaded = LCLfBaseCell(C=18.8e-15, L=45.2e-12, Lf=1.02e-9, N=30)
    loaded = LCLfBaseCell(C=7.0e-15, L=45.2e-12, Lf=0.335e-9, N=6)
    twpa = TWPA(
        cells=[unloaded, loaded, unloaded],
        N=1000,
        Istar=7e-3,
        Idc=1.5e-3,
        Ip0=7e-3 / 60,
    )
    provenance = {
        **PAPERS["malnou"],
        "status": "published lumped parameters",
        "figures": ["2(a)", "2(b)", "7"],
        "finger_lengths_um": {"unloaded": 102.0, "loaded": 33.5},
        "signal_current_A": twpa.Ip0 / 100,
        "pump_frequencies_GHz": [8.8550, 8.8812, 8.8992, 8.9256, 8.9736],
    }
    assert twpa.N_tot == 66_000
    return twpa, provenance


def build_malnou_unmodulated() -> TWPA:
    cell = LCLfBaseCell(C=18.8e-15, L=45.2e-12, Lf=1.02e-9, N=66)
    return TWPA(cells=[cell], N=1000, Istar=7e-3, Idc=1.5e-3, Ip0=7e-3 / 60)


def build_howe_surrogate() -> tuple[TWPA, dict[str, Any]]:
    """Pre-HFSS surrogate constrained by published Z0 and geometry targets."""
    series_L = 60e-12
    z_unloaded, z_loaded = 48.0, 78.0
    c_unloaded = series_L / z_unloaded**2
    c_loaded = series_L / z_loaded**2
    lf_unloaded = 0.4e-9
    lf_loaded = lf_unloaded * 3.9 / 12.0
    unloaded = LCLfBaseCell(
        C=c_unloaded, L=series_L, Lf=lf_unloaded, N=15
    )
    loaded = LCLfBaseCell(C=c_loaded, L=series_L, Lf=lf_loaded, N=4)
    twpa = TWPA(
        cells=[unloaded, loaded, unloaded],
        N=1200,
        Istar=2e-3,
        Idc=220e-6,
        Ip0=100e-6,
    )
    provenance = {
        **PAPERS["howe"],
        "status": "pre-HFSS surrogate",
        "warning": (
            "C and Lf are constrained estimates, not HFSS-extracted values. "
            "Do not use for fabrication."
        ),
        "targets": {
            "unloaded_Z0_ohm": z_unloaded,
            "loaded_Z0_ohm": z_loaded,
            "unloaded_stub_um": 12.0,
            "loaded_stub_um": 3.9,
            "sheet_inductance_pH_per_square": 30.0,
            "dielectric_thickness_nm": 100.0,
            "dielectric_er": 9.1,
        },
        "surrogate_parameters": {
            "series_L_H": series_L,
            "unloaded_C_F": c_unloaded,
            "loaded_C_F": c_loaded,
            "unloaded_Lf_H": lf_unloaded,
            "loaded_Lf_H": lf_loaded,
        },
        "signal_current_A": 1.4e-6,
    }
    assert twpa.N_tot == 40_800
    return twpa, provenance


def build_howe_unmodulated() -> TWPA:
    line, _ = build_howe_surrogate()
    unloaded = line.cells[0]
    cell = LCLfBaseCell(
        C=unloaded.C, L=unloaded.L, Lf=unloaded.Lf, N=34
    )
    return TWPA(
        cells=[cell], N=1200, Istar=2e-3, Idc=220e-6, Ip0=100e-6
    )


def build_faramarzi_surrogate() -> tuple[TWPA, dict[str, Any]]:
    """Stepped-line surrogate for the paper's sinusoidally modulated IMS line."""
    length_m = 0.100
    period_m = 122.7e-6
    phase_velocity = 2.99e6
    z0 = 50.8
    modulation = 0.06
    n_periods = round(length_m / period_m)
    hi = LosslessTL.from_z_vp(
        Z0=z0 * (1 + modulation), vp=phase_velocity, l=period_m / 2
    )
    lo = LosslessTL.from_z_vp(
        Z0=z0 * (1 - modulation), vp=phase_velocity, l=period_m / 2
    )
    twpa = TWPA(
        cells=[hi, lo],
        N=n_periods,
        Istar=3.2e-3,
        Idc=0.0,
        Ip0=380e-6,
    )
    provenance = {
        **PAPERS["faramarzi"],
        "status": "pre-HFSS stepped-line surrogate",
        "warning": (
            "The published sinusoidal stub modulation is represented by a "
            "two-level impedance modulation. Do not use for fabrication."
        ),
        "targets": {
            "length_m": length_m,
            "period_um": period_m * 1e6,
            "phase_velocity_m_per_s": phase_velocity,
            "mean_Z0_ohm": z0,
            "bandgap_GHz": 12.5,
            "pump_GHz": 10.6,
        },
        "surrogate_parameters": {"impedance_modulation_fraction": modulation},
    }
    assert abs(twpa.epsilon) < 1e-15
    return twpa, provenance


def add_extended_3wm(analysis: TWPAnalysis, name: str = "extended_3wm") -> str:
    modes = ModeArrayFactory.create_extended_3wm(
        analysis.data,
        n_pump_harmonics=2,
        n_frequency_conversion=2,
        n_signal_harmonics=1,
    )
    analysis.add_mode_array(name, modes)
    return name


def add_faramarzi_4wm(analysis: TWPAnalysis, name: str = "paper_4wm") -> str:
    modes = ModeArrayFactory.create_custom(
        analysis.data,
        mode_labels=["p", "s", "i", "p3", "p2s", "p4ms"],
        mode_directions=[1, 1, 1, 1, 1, 1],
        relations=[
            ["i", "p+p-s"],
            ["p3", "p+p+p"],
            ["p2s", "p+p+s"],
            ["p4ms", "p+p+p+p-s"],
        ],
    )
    analysis.add_mode_array(name, modes)
    return name


def cell_profile(
    levels: list[float], counts: list[int], repeats: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    if len(levels) != len(counts):
        raise ValueError("levels and counts must have the same length")
    one = np.concatenate(
        [np.full(count, level, dtype=float) for level, count in zip(levels, counts)]
    )
    values = np.tile(one, repeats)
    return np.arange(len(values)), values


def bandwidth_regions(
    freqs_GHz: np.ndarray, gain_db: np.ndarray, reduction_db: float = 3.0
) -> tuple[list[tuple[float, float]], float]:
    f = np.asarray(freqs_GHz, dtype=float)
    g = np.asarray(gain_db, dtype=float)
    finite = np.isfinite(g)
    if finite.sum() < 2:
        return [], 0.0
    threshold = np.nanmax(g) - reduction_db
    indices = np.flatnonzero(finite & (g >= threshold))
    if not len(indices):
        return [], 0.0
    splits = np.where(np.diff(indices) > 1)[0]
    groups = np.split(indices, splits + 1)
    regions = [(float(f[group[0]]), float(f[group[-1]])) for group in groups]
    df = float(np.nanmedian(np.diff(f)))
    total = float(sum(len(group) * df for group in groups))
    return regions, total


def gain_metrics(
    freqs_GHz: np.ndarray,
    gain_db: np.ndarray,
    band: tuple[float, float] = (4.0, 8.0),
) -> dict[str, Any]:
    f = np.asarray(freqs_GHz, dtype=float)
    g = np.asarray(gain_db, dtype=float)
    mask = (f >= band[0]) & (f <= band[1]) & np.isfinite(g)
    if not np.any(mask):
        raise ValueError(f"No finite gain samples inside band {band}")
    gb = g[mask]
    regions, total_bw = bandwidth_regions(f, g)
    return {
        "band_GHz": list(band),
        "peak_gain_db": float(np.nanmax(g)),
        "peak_frequency_GHz": float(f[np.nanargmax(g)]),
        "mean_in_band_gain_db": float(np.mean(gb)),
        "median_in_band_gain_db": float(np.median(gb)),
        "ripple_peak_to_peak_db": float(np.ptp(gb)),
        "ripple_std_db": float(np.std(gb)),
        "fraction_above_15db": float(np.mean(gb >= 15)),
        "fraction_above_17db": float(np.mean(gb >= 17)),
        "fraction_above_20db": float(np.mean(gb >= 20)),
        "bandwidth_regions_GHz": regions,
        "total_3db_bandwidth_GHz": total_bw,
        "finite": bool(np.all(np.isfinite(g))),
    }


def pump_sweep(
    analysis: TWPAnalysis,
    pumps_GHz: np.ndarray,
    signals_GHz: np.ndarray,
    *,
    model: str,
    mode_array_config: str = "basic_3wm",
    Is0: float,
    Ip0: float | None = None,
    thin: int = 200,
) -> np.ndarray:
    rows = []
    for pump in np.asarray(pumps_GHz):
        result = analysis.gain(
            signal_freqs=np.asarray(signals_GHz),
            pump=float(pump),
            Is0=Is0,
            Ip0=Ip0,
            model=model,
            mode_array_config=mode_array_config,
            thin=thin,
        )
        rows.append(np.asarray(result["gain_db"], dtype=float))
    matrix = np.asarray(rows)
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("Gain sweep contains non-finite values")
    return matrix


def four_wave_phase_mismatch(
    analysis: TWPAnalysis,
    signal_GHz: np.ndarray,
    pump_GHz: float,
    pump_current_A: float,
) -> dict[str, np.ndarray]:
    f = np.asarray(analysis.data["freqs"], dtype=float)
    k = np.asarray(analysis.data["k"], dtype=float)
    signal = np.asarray(signal_GHz, dtype=float)
    idler = 2 * pump_GHz - signal
    if signal.min() < f.min() or idler.max() > f.max():
        raise ValueError("Signal/idler frequencies exceed the base-data range")
    ks = np.interp(signal, f, k)
    ki = np.interp(idler, f, k)
    kp = float(np.interp(pump_GHz, f, k))
    linear = ks + ki - 2 * kp
    nonlinear = np.full_like(linear, kp * pump_current_A**2 / (4 * analysis.twpa.Istar**2))
    return {
        "signal_GHz": signal,
        "idler_GHz": idler,
        "linear_delta_beta": linear,
        "nonlinear_correction": nonlinear,
        "total_delta_beta": linear + nonlinear,
    }


def compression_point_dbm(
    input_power_dbm: np.ndarray, mean_gain_db: np.ndarray
) -> float:
    p = np.asarray(input_power_dbm, dtype=float)
    g = np.asarray(mean_gain_db, dtype=float)
    if len(p) != len(g) or len(p) < 2:
        raise ValueError("Compression arrays must have matching length >= 2")
    baseline = float(np.mean(g[: min(3, len(g))]))
    target = baseline - 1.0
    below = np.flatnonzero(g <= target)
    if not len(below):
        return float("nan")
    idx = int(below[0])
    if idx == 0:
        return float(p[0])
    return float(np.interp(target, [g[idx], g[idx - 1]], [p[idx], p[idx - 1]]))


def current_to_dbm(current_A: float | np.ndarray, z0: float = 50.0):
    return I_to_dBm(np.asarray(current_A), Z0=z0)


def dbm_to_current(power_dbm: float | np.ndarray, z0: float = 50.0):
    return dBm_to_I(np.asarray(power_dbm), Z0=z0)


def assert_mode_range(
    base_freqs_GHz: np.ndarray, frequencies: dict[str, np.ndarray]
) -> None:
    lo, hi = float(np.min(base_freqs_GHz)), float(np.max(base_freqs_GHz))
    for label, values in frequencies.items():
        arr = np.asarray(values, dtype=float)
        if np.any(arr < lo) or np.any(arr > hi):
            raise ValueError(f"Mode {label} lies outside [{lo}, {hi}] GHz")
