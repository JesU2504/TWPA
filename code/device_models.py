"""TWPA construction and validation from HFSS cell fits and operating parameters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from twpasolver.models import TWPA, LCLfBaseCell

ROOT = Path(__file__).resolve().parents[1]


def repository_path(path: Path) -> str:
    """POSIX path relative to the repository root."""
    return path.resolve().relative_to(ROOT).as_posix()


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

# Baseline tolerances are defined in docs/HFSS_MODEL_SPEC.md.
_SELECTED_UNLOADED_Z0_OHM = 43.08925347829123
_SELECTED_LOADED_Z0_OHM = 70.20100727735449


def load_hfss_cell_parameters(
    path: str | Path, *, allow_placeholder: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validated unloaded/loaded cell rows and their provenance.

    The CSV must contain exactly one row per role and finite physical parameters.
    Non-HFSS source labels require allow_placeholder=True."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(HFSS_CELL_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"HFSS cell CSV is missing columns: {sorted(missing)}")
        raw_rows = list(reader)
    if len(raw_rows) != 2:
        raise ValueError("HFSS cell CSV must contain exactly two rows")

    numeric_columns = HFSS_CELL_COLUMNS[1:-1]
    rows: dict[str, dict[str, Any]] = {}
    for raw_row in raw_rows:
        role = raw_row["cell_type"].strip().lower()
        if role not in {"unloaded", "loaded"} or role in rows:
            raise ValueError("cell_type must contain one unloaded and one loaded row")
        row: dict[str, Any] = {"cell_type": role, "source": raw_row["source"].strip()}
        for column in numeric_columns:
            try:
                row[column] = float(raw_row[column])
            except ValueError as exc:
                raise ValueError(f"{role}.{column} is not numeric") from exc
            if not np.isfinite(row[column]) or row[column] < 0:
                raise ValueError(f"{role}.{column} must be finite and non-negative")
        for column in ("stub_length_um", "L_pH", "C_fF", "Lf_pH", "Z0_ohm"):
            if row[column] <= 0:
                raise ValueError(f"{role}.{column} must be positive")
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
        "input_file": repository_path(path),
        "hfss_validated": hfss_validated,
        "sources": sorted(sources),
        "fit_within_tolerance": bool(hfss_validated and all(checks.values())),
    }
    return rows, metadata


def hfss_cell_acceptance_checks(rows: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Baseline impedance tolerances and S21/phase residual acceptance flags."""
    unloaded, loaded = rows["unloaded"], rows["loaded"]
    selected_ratio = _SELECTED_LOADED_Z0_OHM / _SELECTED_UNLOADED_Z0_OHM
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
        "S21_fit_rmse_below_0p5db": max(unloaded["fit_s21_rmse_db"], loaded["fit_s21_rmse_db"])
        < 0.5,
        "phase_fit_rmse_below_5deg": max(
            unloaded["fit_phase_rmse_deg"], loaded["fit_phase_rmse_deg"]
        )
        < 5.0,
    }


def build_twpa_from_hfss(
    cell_csv: str | Path,
    config_json: str | Path,
    *,
    allow_placeholder: bool = False,
) -> tuple[TWPA, dict[str, Any]]:
    """Periodic TWPA and provenance from the cell CSV and operating-point JSON.

    Cell counts must be positive integers. Non-HFSS inputs are rejected unless
    allow_placeholder is enabled."""
    rows, input_metadata = load_hfss_cell_parameters(cell_csv, allow_placeholder=allow_placeholder)
    config_path = Path(config_json)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_keys = {
        "unloaded_cells_each_side",
        "loaded_cells",
        "supercells",
        "Istar_A",
        "Idc_A",
        "Ip0_A",
        "Is0_A",
    }
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(f"Amplifier config is missing keys: {sorted(missing_keys)}")
    for key in required_keys:
        value = config[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a non-negative number")
    for key in ("unloaded_cells_each_side", "loaded_cells", "supercells"):
        if int(config[key]) != config[key] or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    def make_cell(role: str, cell_count: int) -> LCLfBaseCell:
        row = rows[role]
        return LCLfBaseCell(
            L=row["L_pH"] * 1e-12,
            C=row["C_fF"] * 1e-15,
            Lf=row["Lf_pH"] * 1e-12,
            delta=row["loss_tangent"],
            N=cell_count,
            centered=bool(config.get("centered_cells", False)),
        )

    unloaded_cells_each_side = int(config["unloaded_cells_each_side"])
    loaded_cell_count = int(config["loaded_cells"])
    twpa = TWPA(
        cells=[
            make_cell("unloaded", unloaded_cells_each_side),
            make_cell("loaded", loaded_cell_count),
            make_cell("unloaded", unloaded_cells_each_side),
        ],
        N=int(config["supercells"]),
        Istar=float(config["Istar_A"]),
        Idc=float(config["Idc_A"]),
        Ip0=float(config["Ip0_A"]),
    )
    expected_total_cells = (2 * unloaded_cells_each_side + loaded_cell_count) * int(
        config["supercells"]
    )
    if twpa.N_tot != expected_total_cells:
        raise AssertionError(f"Expected {expected_total_cells} cells, built {twpa.N_tot}")
    provenance = {
        "status": "HFSS extracted"
        if input_metadata["hfss_validated"]
        else "PLACEHOLDER workflow test",
        "warning": None
        if input_metadata["hfss_validated"]
        else "These parameters are placeholders and are not HFSS results.",
        "cell_input": input_metadata,
        "amplifier_config_file": repository_path(config_path),
        "amplifier_config": config,
        "cells": rows,
        "total_cells": twpa.N_tot,
    }
    return twpa, provenance
