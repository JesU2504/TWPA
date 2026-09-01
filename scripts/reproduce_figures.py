#!/usr/bin/env python3
"""Figure-reproduction pipeline for saved results or HFSS recalculation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="rerun fitting and nonlinear solves before plotting",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check results against the archived reference chart data",
    )
    args = parser.parse_args()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(cache / "matplotlib"),
        NUMBA_CACHE_DIR=str(cache / "numba"),
        XDG_CACHE_HOME=str(cache),
    )
    tasks = []
    if args.recompute:
        tasks += [
            "fit_hfss_unit_cells.py",
            "prepare_reference_curves.py",
            "fit_stub_length_sweep.py",
            "fit_extended_stubs.py",
            "reproduce_howe_fig3_hfss_bloch.py",
            "refine_pump_bandwidth.py",
            "reproduce_howe_fig_s5_gain_ripple.py",
            "reproduce_fig3_zmatched_supercell.py",
            "compare_zmatched_gain_ripple.py",
        ]
    for task in tasks:
        print(f"Running {task}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / task)], cwd=ROOT, env=env, check=True
        )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/plot_slide_figures.py")]
        + (["--verify"] if args.verify else []),
        cwd=ROOT,
        env=env,
        check=True,
    )
    report = ROOT / "figures/verification.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    data.update(
        mode="recompute" if args.recompute else "plot_cached_results",
        verified=args.verify,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    report.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Figures: {ROOT / 'figures'}")


if __name__ == "__main__":
    main()
