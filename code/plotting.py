"""Headless PNG and PDF output for numerical figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FIGURE_DPI = 200


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """PNG and PDF paths for a figure saved at the configured resolution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI)
    fig.savefig(pdf_path)
    return png_path, pdf_path
