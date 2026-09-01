#!/usr/bin/env python3
"""Impedance-curve digitization from Howe et al., arXiv:2507.07706, Fig. 2.

The externally supplied input is a 400 dpi crop of PDF page 4. Tick coordinates calibrate length
and impedance. Grayscale and brightness thresholds reject colored reference
lines and gridlines; a bounding mask excludes the legend. The median of the
largest contiguous pixel cluster gives the curve position in each column.

The extracted values at 3.9 and 12.1 µm are 76.9 and 48.0 ohms, respectively,
within 1.5% of the published 78.0 and 48.0 ohm reference points."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

import numpy as np
from PIL import Image

from io_utils import write_csv_rows

OUTPUT = ROOT / "results" / "bloch_corrected" / "step_32_fig2_z0_comparison"

# Tick coordinates calibrate the fixed 400 dpi source crop.
X0_PX, X1_PX = 246, 1293
LENGTH0_UM, LENGTH1_UM = 0.0, 80.0
Y0_PX, Y0_OHM = 238.5, 120.0
Y1_PX, Y1_OHM = 851.5, 20.0

# Inset the scan bounds to exclude frame antialiasing.
FRAME_MARGIN_PX = 4
FRAME_LEFT_PX, FRAME_RIGHT_PX = 208 + FRAME_MARGIN_PX, 1344 - FRAME_MARGIN_PX
FRAME_TOP_PX, FRAME_BOTTOM_PX = 193 + FRAME_MARGIN_PX, 888 - FRAME_MARGIN_PX

# Mask the legend to prevent its text from entering the extracted curve.
LEGEND_BOX = (slice(195, 425), slice(860, 1345))

CLUSTER_GAP_PX = 5
MAX_COLUMN_SPAN_PX = 15


def px_to_length_um(px: np.ndarray) -> np.ndarray:
    return LENGTH0_UM + (px - X0_PX) / (X1_PX - X0_PX) * (LENGTH1_UM - LENGTH0_UM)


def py_to_z0_ohm(py: np.ndarray) -> np.ndarray:
    return Y0_OHM + (py - Y0_PX) / (Y1_PX - Y0_PX) * (Y1_OHM - Y0_OHM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="calibrated 400 dpi Fig. 2 crop")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.source).convert("RGB")
    pixels = np.array(image).astype(int)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

    curve_mask = (np.abs(red - green) < 6) & (np.abs(green - blue) < 6) & (red < 150)
    trimmed_mask = np.zeros_like(curve_mask)
    trimmed_mask[FRAME_TOP_PX:FRAME_BOTTOM_PX, FRAME_LEFT_PX:FRAME_RIGHT_PX] = curve_mask[
        FRAME_TOP_PX:FRAME_BOTTOM_PX, FRAME_LEFT_PX:FRAME_RIGHT_PX
    ]
    trimmed_mask[LEGEND_BOX] = False

    lengths_um, z0_ohm = [], []
    for x in range(FRAME_LEFT_PX, FRAME_RIGHT_PX):
        column_ys = np.where(trimmed_mask[:, x])[0]
        if len(column_ys) == 0:
            continue
        if column_ys.max() - column_ys.min() > MAX_COLUMN_SPAN_PX:
            sorted_ys = np.sort(column_ys)
            gaps = np.diff(sorted_ys)
            splits = np.where(gaps > CLUSTER_GAP_PX)[0]
            clusters = np.split(sorted_ys, splits + 1)
            clusters.sort(key=len, reverse=True)
            column_ys = clusters[0]
        lengths_um.append(px_to_length_um(x))
        z0_ohm.append(py_to_z0_ohm(np.median(column_ys)))

    lengths_um = np.array(lengths_um)
    z0_ohm = np.array(z0_ohm)
    order = np.argsort(lengths_um)
    lengths_um, z0_ohm = lengths_um[order], z0_ohm[order]

    write_csv_rows(
        OUTPUT / "01b_paper_digitized_Z0_curve.csv",
        ["finger_length_um", "Z0_ohm"],
        [[f"{length:.4f}" for length in lengths_um], [f"{z0:.3f}" for z0 in z0_ohm]],
    )

    z0_at_3p9um = np.interp(3.9, lengths_um, z0_ohm)
    z0_at_12p1um = np.interp(12.1, lengths_um, z0_ohm)
    print(f"Digitized {len(lengths_um)} points, {lengths_um.min():.2f}-{lengths_um.max():.2f} um")
    print(f"Z0(3.9 um) = {z0_at_3p9um:.2f} ohm  (paper's exact value: 78.0 ohm)")
    print(f"Z0(12.1 um) = {z0_at_12p1um:.2f} ohm  (paper's exact value: 48.0 ohm)")
    print(f"Wrote {OUTPUT / '01b_paper_digitized_Z0_curve.csv'}")


if __name__ == "__main__":
    main()
