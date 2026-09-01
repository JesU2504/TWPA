"""Touchstone input and CSV output for the HFSS analysis pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _rounded_numbers(value):
    if isinstance(value, (float, np.floating)):
        return float(f"{value:.12g}") if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _rounded_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded_numbers(item) for item in value]
    return value


def write_json(path: Path, value, *, sort_keys: bool = False) -> None:
    """Write calculated metadata with twelve significant digits.

    ``newline="\\n"`` and UTF-8 keep the output byte-identical on Windows, where
    ``Path.write_text`` would otherwise emit CRLF."""
    path.write_text(
        json.dumps(_rounded_numbers(value), indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _csv_value(value):
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return f"{value:.12g}"
    return value


Z0_DEFAULT_OHM = 50.0


def read_touchstone(path: Path, *, z0_ohm: float = Z0_DEFAULT_OHM) -> tuple[np.ndarray, np.ndarray]:
    """Two-port RI/MA Touchstone data with GHz frequencies and fixed reference impedance.

    Returns frequency and S-matrix arrays with shapes (n,) and (n, 2, 2).
    The matrix convention is [[S11, S12], [S21, S22]]."""
    frequencies: list[float] = []
    matrices: list[np.ndarray] = []
    data_format: str | None = None
    reference: float | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
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
                pairs.append(values[index] * np.exp(1j * np.deg2rad(values[index + 1])))
            else:
                raise ValueError(f"Unsupported Touchstone format {data_format!r} in {path}")
        # Touchstone two-port ordering is S11, S21, S12, S22.
        matrices.append(np.array([[pairs[0], pairs[2]], [pairs[1], pairs[3]]]))
        frequencies.append(values[0])
    if reference != z0_ohm:
        raise ValueError(f"{path} must be normalized to {z0_ohm:g} ohms")
    return np.asarray(frequencies), np.asarray(matrices)


def write_csv_rows(path: Path, header: Sequence[str], columns: Iterable[Iterable]) -> None:
    """Write stable CSV output from column iterables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(tuple(_csv_value(value) for value in row) for row in zip(*columns))
