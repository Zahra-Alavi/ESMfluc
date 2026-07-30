import unittest

import numpy as np
import pandas as pd

from signed_band_analysis.compare_observed_uniform_bands import (
    annotate_mean_stability,
    mask_metrics,
    nearest_distances,
    pair_one_group,
)


def band(identifier, apex, start, end):
    return {
        "condition": "c",
        "split": "test",
        "protein": "p",
        "protein_length": 12,
        "sign": 1,
        "label": "flexibility_supporting",
        "band_id": identifier,
        "apex_index_0based": apex,
        "support_start_index_0based": start,
        "support_end_index_0based_inclusive": end,
        "support_width": end - start + 1,
        "apex_standardized_magnitude_R_p": 3.0,
        "band_integrated_magnitude": 1.0,
        "integrated_magnitude_rank_within_protein_sign": 1,
    }


class ObservedUniformComparisonTests(unittest.TestCase):
    def context(self):
        return {
            "condition": "c",
            "split": "test",
            "protein": "p",
            "protein_length": 12,
            "sign": 1,
            "label": "flexibility_supporting",
            "eligible_start_index_0based": 1,
            "eligible_end_index_0based_exclusive": 11,
            "eligible_residue_count": 10,
        }

    def test_binary_mask_overlap(self):
        observed = pd.DataFrame([
            band("o1", 3, 2, 4),
            band("o2", 8, 8, 9),
        ])
        uniform = pd.DataFrame([
            band("u1", 4, 3, 5),
            band("u2", 10, 10, 10),
        ])
        metrics = mask_metrics(
            pd.Series(self.context()), observed, uniform
        )
        self.assertEqual(metrics["intersection_band_residues"], 2)
        self.assertEqual(metrics["union_band_residues"], 7)
        self.assertAlmostEqual(metrics["jaccard"], 2 / 7)
        self.assertAlmostEqual(metrics["dice"], 4 / 9)
        self.assertAlmostEqual(
            metrics["fraction_observed_residues_covered_by_uniform"], 2 / 5
        )
        self.assertAlmostEqual(
            metrics["fraction_uniform_residues_covered_by_observed"], 2 / 4
        )

    def test_nearest_apex_distance_and_missing_target(self):
        np.testing.assert_array_equal(
            nearest_distances(np.array([2, 9]), np.array([4, 10])),
            np.array([2.0, 1.0]),
        )
        self.assertTrue(np.isnan(
            nearest_distances(np.array([2]), np.array([]))[0]
        ))

    def test_hungarian_pairing_maximizes_iou_and_leaves_disjoint_unmatched(self):
        observed = pd.DataFrame([
            band("o1", 3, 2, 5),
            band("o2", 9, 8, 10),
        ])
        uniform = pd.DataFrame([
            band("u1", 4, 3, 5),
            band("u2", 7, 6, 6),
        ])
        matched, observed_only, uniform_only, metrics = pair_one_group(
            self.context(), "mean_candidate", observed, uniform
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["observed_band_id"], "o1")
        self.assertEqual(matched[0]["uniform_band_id"], "u1")
        self.assertAlmostEqual(matched[0]["interval_iou"], 3 / 4)
        self.assertEqual(observed_only[0]["band_id"], "o2")
        self.assertEqual(uniform_only[0]["band_id"], "u2")
        self.assertEqual(metrics["matched_pairs_iou_ge_0_75"], 1)

    def test_all_mean_bands_receive_stability_status(self):
        mean = pd.DataFrame([
            band("b1", 3, 2, 4),
            band("b2", 8, 7, 9),
        ])
        stable = mean.iloc[[0]].copy()
        stable["seed_support_count"] = 3
        stable["seed_support_fraction"] = 1.0
        stable["strict_stable_band"] = True
        stable["primary_stable_band"] = True
        stable["stability_class"] = "strict_3_of_3"
        annotated = annotate_mean_stability(mean, stable, "observed")
        self.assertEqual(
            annotated.set_index("band_id").loc["b1", "stability_status"],
            "supported_3of3",
        )
        self.assertEqual(
            annotated.set_index("band_id").loc["b2", "stability_status"],
            "not_retained_lt2_support_count_unknown",
        )


if __name__ == "__main__":
    unittest.main()
