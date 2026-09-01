"""Reference gain curves from Howe et al., arXiv:2507.07706, Figs. 3(d) and 4(a).

The externally supplied SVG files are exports of PDF pages 5 and 6. Fixed coordinate calibrations
and path selection reproduce the presentation's digitization. Experimental
and simulated curves represent different paper cases."""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from _bootstrap import ROOT


def points(d):
    return [
        (float(m[1]), float(m[2]))
        for m in re.finditer(r"[ML]\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", d)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="directory containing calibrated paper_fig3d.svg and paper_fig4a.svg exports",
    )
    args = parser.parse_args()
    experiment = [
        e
        for e in ET.parse(args.source_dir / "paper_fig4a.svg").getroot().iter()
        if e.tag.endswith("path")
        and e.get("stroke") == "rgb(54.508972%, 0%, 54.508972%)"
        and len(e.get("d", "")) == 27685
    ]
    if len(experiment) != 1:
        raise ValueError("Expected exactly the selected Fig. 4(a) magenta vector path")
    exp = [
        (
            2 + (x - 58.462374) * 8 / (420.926627 - 58.462374),
            -5 + (y + 286.329962) * 50 / (-55.984671 + 286.329962),
        )
        for x, y in points(experiment[0].get("d"))
    ]
    candidates = []
    for e in ET.parse(args.source_dir / "paper_fig3d.svg").getroot().iter():
        if e.tag.endswith("path") and e.get("stroke") == "rgb(100%, 0%, 0%)":
            for part in e.get("d", "").split("M"):
                pts = points("M" + part)
                if len(pts) > 100:
                    candidates.append(pts)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one Fig. 3(d) maximum-GBP red path")
    sim = [
        (
            1 + (x - 396.597883) * 13 / (644.583122 - 396.597883),
            (y - 55.950730) * 50 / (239.874324 - 55.950730),
        )
        for x, y in candidates[0]
    ]
    sim = [(x, y) for x, y in sim if 2 <= x <= 11]
    output = ROOT / "results/paper_curves"
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [("paper_experiment_fig4a.csv", exp), ("paper_simulation_fig3d.csv", sim)]:
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["signal_GHz", "gain_dB"])
            writer.writerows(data)
        print(name, len(data), "points")


if __name__ == "__main__":
    main()
