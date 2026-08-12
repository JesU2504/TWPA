from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from twpa_project_utils import (  # noqa: E402
    build_twpa_from_hfss,
    cell_profile,
    gain_metrics,
    load_hfss_cell_parameters,
)


class ProjectTests(unittest.TestCase):
    def test_cell_profile_counts(self):
        _, z = cell_profile([48, 78, 48], [15, 4, 15], repeats=2)
        self.assertEqual(len(z), 68)
        self.assertEqual(np.count_nonzero(z == 78), 8)

    def test_gain_metrics(self):
        f = np.arange(3, 9.1, 0.1)
        gain = 20 - 2 * (f - 6) ** 2
        metrics = gain_metrics(f, gain)
        self.assertEqual(metrics["peak_gain_db"], 20)
        self.assertTrue(0 < metrics["fraction_above_17db"] <= 1)

    def test_hfss_placeholder_is_explicit(self):
        csv_path = ROOT / "hfss_inputs" / "hfss_cell_parameters_TEMPLATE.csv"
        config_path = ROOT / "hfss_inputs" / "amplifier_config.json"
        with self.assertRaises(ValueError):
            load_hfss_cell_parameters(csv_path)
        rows, metadata = load_hfss_cell_parameters(
            csv_path, allow_placeholder=True
        )
        self.assertFalse(metadata["hfss_validated"])
        self.assertFalse(metadata["all_hfss_acceptance_checks_pass"])
        self.assertEqual(set(rows), {"unloaded", "loaded"})
        twpa, provenance = build_twpa_from_hfss(
            csv_path, config_path, allow_placeholder=True
        )
        self.assertEqual(twpa.N_tot, 40_800)
        self.assertEqual(provenance["status"], "PLACEHOLDER workflow test")

    def test_selected_hfss_cells_pass_project_gates(self):
        csv_path = ROOT / "hfss_inputs" / "hfss_cell_parameters.csv"
        rows, metadata = load_hfss_cell_parameters(csv_path)
        self.assertTrue(metadata["hfss_validated"])
        self.assertTrue(metadata["all_hfss_acceptance_checks_pass"])
        self.assertAlmostEqual(rows["unloaded"]["Z0_ohm"], 43.0892534783)
        self.assertAlmostEqual(rows["loaded"]["Z0_ohm"], 70.2010072774)


if __name__ == "__main__":
    unittest.main()
