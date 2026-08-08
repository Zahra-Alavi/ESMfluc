import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_biophysical_enrichment import (
    add_primary_pvalue_family,
    attach_test_strain,
    deterministic_tie_values,
    headline_union_group_inference,
    matched_covariate_outcomes,
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


class TestCorrectedPhase2Inference(unittest.TestCase):
    def test_matched_covariates_are_not_inferential_outcomes(self):
        self.assertEqual(matched_covariate_outcomes("q3_only"), set())
        self.assertEqual(matched_covariate_outcomes("q3_neq"), {"neq"})
        self.assertEqual(
            matched_covariate_outcomes("q3_neq_rsa_position"),
            {"neq", "rsa", "normalized_position"},
        )

    def test_tie_break_is_reproducible_and_not_index_order(self):
        candidates = np.arange(20)
        first = deterministic_tie_values(
            random_seed=123, condition="c", split="test", protein="p",
            band_id="b", match_scheme="q3_neq", candidate_indices=candidates,
        )
        second = deterministic_tie_values(
            random_seed=123, condition="c", split="test", protein="p",
            band_id="b", match_scheme="q3_neq", candidate_indices=candidates,
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(np.argsort(first), candidates))

    def test_primary_family_uses_only_two_sided_values(self):
        frame = pd.DataFrame({
            "p_upper": [0.001, 0.9],
            "p_lower": [0.9, 0.001],
            "p_two_sided": [0.02, 0.04],
        })
        result = add_primary_pvalue_family(frame)
        np.testing.assert_allclose(result.primary_p_two_sided, [0.02, 0.04])
        np.testing.assert_allclose(result.primary_q_bh, [0.04, 0.04])

    def test_headline_inference_resamples_union_group_means(self):
        per_protein = pd.DataFrame({
            "match_scheme": ["q3_neq_rsa"] * 4,
            "condition": ["c"] * 4, "split": ["test"] * 4,
            "protein": ["p1", "p2", "p3", "p4"], "sign": [1] * 4,
            "label": ["flexibility_supporting"] * 4,
            "metric": ["torsion_change_from_previous"] * 4,
            "case_minus_control": [1.0, 3.0, 5.0, 7.0],
        })
        manifest = pd.DataFrame({
            "name": ["p1", "p2", "p3", "p4"], "split": ["test"] * 4,
            "union_group_id": ["g1", "g1", "g2", "g3"],
        })
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "groups.csv"
            manifest.to_csv(path, index=False)
            result = headline_union_group_inference(
                per_protein, path, n_bootstrap=200, n_sign_flips=100,
                minimum_groups=2, random_seed=7,
            )
        row = result.iloc[0]
        self.assertEqual(row.n_union_groups, 3)
        self.assertEqual(row.n_proteins, 4)
        self.assertAlmostEqual(row.union_group_mean_effect, (2 + 5 + 7) / 3)
        self.assertEqual(row.primary_p_two_sided, row.union_group_p_two_sided)

    def test_headline_family_uses_rsa_before_rsa_matching(self):
        per_protein = pd.DataFrame({
            "match_scheme": ["q3_neq", "q3_neq_rsa", "q3_neq_rsa_position"],
            "condition": ["c"] * 3, "split": ["test"] * 3,
            "protein": ["p1"] * 3, "sign": [1] * 3,
            "label": ["flexibility_supporting"] * 3,
            "metric": ["rsa", "torsion_change_from_previous", "rsa"],
            "case_minus_control": [0.2, 3.0, 0.0],
        })
        manifest = pd.DataFrame({
            "name": ["p1"], "split": ["test"], "union_group_id": ["g1"],
        })
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "groups.csv"
            manifest.to_csv(path, index=False)
            result = headline_union_group_inference(
                per_protein, path, n_bootstrap=20, n_sign_flips=10,
                minimum_groups=2, random_seed=7,
            )
        self.assertEqual(set(result.match_scheme), {"q3_neq", "q3_neq_rsa"})
        self.assertEqual(
            set(result.metric), {"rsa", "torsion_change_from_previous"}
        )


if __name__ == "__main__":
    unittest.main()
