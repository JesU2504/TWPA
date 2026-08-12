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
from hfss_bloch import apply_bloch_parameters, extract_bloch_parameters  # noqa: E402


class ProjectTests(unittest.TestCase):
    def test_bloch_extraction_recovers_matched_transmission_line(self):
        phase = np.linspace(0.2, 7.0, 101)
        impedance = 50.0
        abcd = np.empty((len(phase), 2, 2), dtype=complex)
        abcd[:, 0, 0] = np.cos(phase)
        abcd[:, 0, 1] = 1j * impedance * np.sin(phase)
        abcd[:, 1, 0] = 1j * np.sin(phase) / impedance
        abcd[:, 1, 1] = np.cos(phase)
        denominator = (
            abcd[:, 0, 0]
            + abcd[:, 0, 1] / impedance
            + abcd[:, 1, 0] * impedance
            + abcd[:, 1, 1]
        )
        s = np.empty_like(abcd)
        s[:, 0, 0] = (
            abcd[:, 0, 0]
            + abcd[:, 0, 1] / impedance
            - abcd[:, 1, 0] * impedance
            - abcd[:, 1, 1]
        ) / denominator
        s[:, 1, 0] = 2 / denominator
        s[:, 0, 1] = 2 / denominator
        s[:, 1, 1] = (
            -abcd[:, 0, 0]
            + abcd[:, 0, 1] / impedance
            - abcd[:, 1, 0] * impedance
            + abcd[:, 1, 1]
        ) / denominator

        extracted = extract_bloch_parameters(s)
        np.testing.assert_allclose(extracted.phase, phase, atol=1e-12)
        np.testing.assert_allclose(extracted.impedance, impedance, atol=1e-10)
        np.testing.assert_allclose(extracted.reflection, 0, atol=1e-12)
        np.testing.assert_allclose(extracted.attenuation, 0, atol=1e-12)

    def test_bloch_extraction_removes_port_mismatch_phase(self):
        phase = np.linspace(0.3, 2.8, 51)
        line_impedance = 70.0
        reference_impedance = 50.0
        abcd = np.empty((len(phase), 2, 2), dtype=complex)
        abcd[:, 0, 0] = np.cos(phase)
        abcd[:, 0, 1] = 1j * line_impedance * np.sin(phase)
        abcd[:, 1, 0] = 1j * np.sin(phase) / line_impedance
        abcd[:, 1, 1] = np.cos(phase)
        denominator = (
            abcd[:, 0, 0]
            + abcd[:, 0, 1] / reference_impedance
            + abcd[:, 1, 0] * reference_impedance
            + abcd[:, 1, 1]
        )
        s = np.empty_like(abcd)
        s[:, 0, 0] = (
            abcd[:, 0, 0]
            + abcd[:, 0, 1] / reference_impedance
            - abcd[:, 1, 0] * reference_impedance
            - abcd[:, 1, 1]
        ) / denominator
        s[:, 1, 0] = 2 / denominator
        s[:, 0, 1] = 2 / denominator
        s[:, 1, 1] = (
            -abcd[:, 0, 0]
            + abcd[:, 0, 1] / reference_impedance
            - abcd[:, 1, 0] * reference_impedance
            + abcd[:, 1, 1]
        ) / denominator

        transmission_phase = -np.unwrap(np.angle(s[:, 1, 0]))
        self.assertGreater(np.max(np.abs(transmission_phase - phase)), 1e-3)
        extracted = extract_bloch_parameters(s, z0=reference_impedance)
        np.testing.assert_allclose(extracted.phase, phase, atol=1e-12)
        np.testing.assert_allclose(extracted.impedance, line_impedance, atol=1e-10)
        expected_gamma = (line_impedance - reference_impedance) / (
            line_impedance + reference_impedance
        )
        np.testing.assert_allclose(extracted.reflection, expected_gamma, atol=1e-12)

    def test_apply_bloch_parameters_replaces_only_measured_band(self):
        frequencies = np.arange(1.0, 6.0)
        phase = np.array([0.4, 0.8, 1.2])
        s = np.zeros((3, 2, 2), dtype=complex)
        s[:, 1, 0] = np.exp(-1j * phase)
        s[:, 0, 1] = s[:, 1, 0]
        data = {
            "freqs": frequencies,
            "k": np.full(5, -1.0),
            "alpha": np.full(5, -2.0),
            "gammas": np.full(5, 3.0 + 4.0j),
        }
        apply_bloch_parameters(data, np.array([2.0, 3.0, 4.0]), s, 2)
        np.testing.assert_allclose(data["k"], [-1.0, 0.2, 0.4, 0.6, -1.0])
        np.testing.assert_allclose(data["alpha"][1:4], 0, atol=1e-12)
        np.testing.assert_allclose(data["gammas"][1:4], 0, atol=1e-12)
        self.assertEqual(data["gammas"][0], 3.0 + 4.0j)
        self.assertEqual(data["gammas"][-1], 3.0 + 4.0j)

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
