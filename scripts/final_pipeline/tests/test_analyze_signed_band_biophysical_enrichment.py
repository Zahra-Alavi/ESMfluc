import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_biophysical_enrichment import (
    attach_test_strain,
)


class TestStrainAttachment(unittest.TestCase):
    def residue_frame(self):
        return pd.DataFrame({
            "split": ["test"] * 6 + ["train"] * 3,
            "protein": ["protein_A"] * 6 + ["train_A"] * 3,
            "protein_length": [6] * 6 + [3] * 3,
            "residue_index_0based": list(range(6)) + list(range(3)),
        })

    def test_exact_test_mapping_terminal_exclusion_and_high_strain(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "protein_A"
            folder.mkdir()
            pd.DataFrame({
                "residue": np.arange(1, 7),
                "ensemble_mean": np.arange(1.0, 7.0),
                "ensemble_std": np.arange(0.1, 0.7, 0.1),
            }).to_csv(folder / "strain_summary.csv", index=False)
            merged, audit = attach_test_strain(
                self.residue_frame(), root, terminal_exclusion=1, high_quantile=0.75
            )

        test = merged[merged["split"] == "test"].reset_index(drop=True)
        self.assertTrue(np.isnan(test.loc[0, "strain_ensemble_mean"]))
        self.assertTrue(np.isnan(test.loc[5, "strain_ensemble_mean"]))
        np.testing.assert_allclose(
            test.loc[1:4, "strain_ensemble_mean"], [2.0, 3.0, 4.0, 5.0]
        )
        np.testing.assert_allclose(
            test.loc[2:4, "strain_abs_gradient_from_previous"], [1.0, 1.0, 1.0]
        )
        self.assertEqual(test.loc[4, "strain_top_quantile_within_protein"], 1.0)
        self.assertEqual(test.loc[1, "distance_to_strain_top_quantile"], 3.0)
        self.assertTrue(
            merged.loc[merged["split"] == "train", "strain_ensemble_mean"].isna().all()
        )
        self.assertEqual(audit.loc[0, "strain_status"], "ok")
        self.assertEqual(audit.loc[0, "n_finite_after_terminal_exclusion"], 4)

    def test_missing_file_is_audited_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as root:
            merged, audit = attach_test_strain(
                self.residue_frame(), root, terminal_exclusion=1, high_quantile=0.9
            )
        self.assertTrue(merged["strain_ensemble_mean"].isna().all())
        self.assertEqual(audit.loc[0, "strain_status"], "missing")

    def test_index_mismatch_is_rejected_from_analysis(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "protein_A"
            folder.mkdir()
            pd.DataFrame({
                "residue": [1, 2, 3, 4, 5],
                "ensemble_mean": np.arange(5.0),
                "ensemble_std": np.ones(5),
            }).to_csv(folder / "strain_summary.csv", index=False)
            merged, audit = attach_test_strain(
                self.residue_frame(), root, terminal_exclusion=1, high_quantile=0.9
            )
        self.assertTrue(merged["strain_ensemble_mean"].isna().all())
        self.assertEqual(audit.loc[0, "strain_status"], "index_or_length_mismatch")


if __name__ == "__main__":
    unittest.main()
