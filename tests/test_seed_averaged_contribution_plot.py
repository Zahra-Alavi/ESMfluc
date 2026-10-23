"""Checks for the read-only manuscript contribution-figure exporter."""

import unittest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from esmfluc.visualization.plot_seed_averaged_contributions import (
    assert_profile,
    render,
    validate_bands,
)


class TestSavedContributionFigure(unittest.TestCase):
    def setUp(self):
        self.sequence = "ACDE"
        self.influence = np.array([0.1, 0.2, -0.1, -0.3])
        self.band = {
            "protein_length": "4", "start_residue_1based": "1",
            "end_residue_1based_inclusive": "2", "apex_residue_1based": "2",
            "start_index_0based": "0", "end_index_0based_inclusive": "1",
            "apex_index_0based": "1", "sign": "1", "apex_amino_acid": "C",
            "apex_signed_column_influence": "0.2", "primary_stable_band": "True",
            "seed_support_count": "2", "matching_min_interval_iou": "0.5",
            "minimum_seed_interval_iou": "0.6",
        }

    def test_exact_saved_profile(self):
        self.assertEqual(assert_profile(self.influence, self.influence.tolist(), "test"), 0)

    def test_profile_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            assert_profile(self.influence, -self.influence, "test")

    def test_nonfinite_profile_rejected(self):
        with self.assertRaises(ValueError):
            assert_profile(self.influence, [0.1, 0.2, -0.1, np.nan], "test")

    def test_saved_band_accepted(self):
        self.assertEqual(validate_bands([self.band], self.influence, self.sequence), [self.band])

    def test_bad_coordinate_rejected(self):
        band = dict(self.band, apex_index_0based="2")
        with self.assertRaises(ValueError):
            validate_bands([band], self.influence, self.sequence)

    def test_wrong_sequence_rejected(self):
        with self.assertRaises(ValueError):
            validate_bands([self.band], self.influence, "AAAA")

    def test_wrong_peak_value_rejected(self):
        band = dict(self.band, apex_signed_column_influence="0.3")
        with self.assertRaises(ValueError):
            validate_bands([band], self.influence, self.sequence)

    def test_unstable_band_rejected(self):
        band = dict(self.band, seed_support_count="1")
        with self.assertRaises(ValueError):
            validate_bands([band], self.influence, self.sequence)

    def test_bad_iou_rejected(self):
        band = dict(self.band, minimum_seed_interval_iou="0.49")
        with self.assertRaises(ValueError):
            validate_bands([band], self.influence, self.sequence)

    def test_opposite_sign_interval_rejected(self):
        band = dict(self.band, end_residue_1based_inclusive="3", end_index_0based_inclusive="2")
        with self.assertRaises(ValueError):
            validate_bands([band], self.influence, self.sequence)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            validate_bands([self.band, self.band], self.influence, self.sequence)

    def test_seed_averaging_commutes_with_query_mean(self):
        rng = np.random.default_rng(23)
        matrices = rng.normal(size=(3, 4, 4))
        np.testing.assert_allclose(matrices.mean(axis=0).mean(axis=0),
                                   matrices.mean(axis=1).mean(axis=0))

    def test_exports_are_pdf_and_svg_not_renamed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "test_figure"
            render(np.tile(self.influence, (4, 1)), self.influence,
                   [self.band], prefix, "Test", 1.0)
            self.assertTrue(prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"))
            self.assertEqual(ET.parse(prefix.with_suffix(".svg")).getroot().tag,
                             "{http://www.w3.org/2000/svg}svg")


if __name__ == "__main__":
    unittest.main()
