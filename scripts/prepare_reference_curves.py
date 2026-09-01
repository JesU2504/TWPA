"""Copy archived digitized reference data into the calculation output directories."""

import csv

from _bootstrap import ROOT

REFERENCE = ROOT / "data/reference"


def main():
    destinations = {
        "paper_impedance_fig2.csv": (
            "results/bloch_corrected/step_32_fig2_z0_comparison/01b_paper_digitized_Z0_curve.csv"
        ),
        "paper_experiment_fig4a.csv": "results/paper_curves/paper_experiment_fig4a.csv",
        "paper_simulation_fig3d.csv": "results/paper_curves/paper_simulation_fig3d.csv",
    }
    for name, relative_path in destinations.items():
        destination = ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (REFERENCE / name).open(newline="") as source:
            rows = list(csv.reader(source))
        # Archived gain traces use different labels; outputs share the solver's schema.
        if name != "paper_impedance_fig2.csv":
            rows[0] = ["signal_GHz", "gain_dB"]
        with destination.open("w", newline="") as output:
            csv.writer(output).writerows(rows)
        print(f"Reference data: {relative_path}")


if __name__ == "__main__":
    main()
