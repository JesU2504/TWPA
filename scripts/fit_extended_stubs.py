"""Long-stub fits with a documented fallback when raw HFSS exports are absent."""

import csv
from pathlib import Path

from fit_hfss_unit_cells import fit_cell

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "data/reference/hfss_40_50_60um_fits.csv"
    with target.open() as handle:
        rows = list(csv.DictReader(handle))
    missing = []
    for row in rows:
        length = int(float(row["stub_length_um"]))
        definition = {
            "stub_length_um": float(length),
            "broadband_file": f"kinetic_{length}um_debug.s2p",
            "lowfreq_file": f"kinetic_{length}um_lowfreq.s2p",
        }
        absent = [
            definition[k]
            for k in ("broadband_file", "lowfreq_file")
            if not (ROOT / "hfss_inputs" / definition[k]).exists()
        ]
        if absent:
            missing.extend(absent)
        else:
            fit = fit_cell(ROOT / "hfss_inputs", definition)
            row.update({key: fit[key] for key in row if key in fit})
    # Put regenerated/fallback points in results; never overwrite the frozen reference.
    output = ROOT / "results/extended_stubs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if missing:
        print(
            "WARNING: using preserved fitted values; raw files unavailable: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
