import types
import unittest

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_interval_structure import (
    STRUCTURE_FEATURES,
    band_case_row,
    candidate_intervals,
    make_protein_arrays,
    matched_effects,
)


class BandIntervalStructureTests(unittest.TestCase):
    def setUp(self):
        length = 12
        frame = pd.DataFrame({
            "protein": ["p"] * length,
            "residue_index_0based": np.arange(length),
            "protein_length_structure": [length] * length,
            "structure_resolved": [True] * length,
            "q3": ["C"] * length,
            "neq": [1.2] * length,
            "rsa": [0.3] * length,
            "normalized_position": np.arange(length) / (length - 1),
        })
        for number, feature in enumerate(STRUCTURE_FEATURES, 1):
            frame[feature] = np.arange(length, dtype=float) + number
        self.record = make_protein_arrays(frame)["p"]
        self.band = types.SimpleNamespace(
            band_id="b", condition="c", split="test", protein="p", sign=1,
            start_index_0based=4, end_index_0based_inclusive=6,
            apex_index_0based=5, eligible_start_index_0based=0,
            eligible_end_index_0based_exclusive=12,
        )
        self.args = types.SimpleNamespace(
            position_caliper=0.30, neq_caliper=0.25, rsa_caliper=0.15,
            max_controls_per_apex=5,
        )

    def test_control_intervals_preserve_width_and_avoid_bands(self):
        mask = np.zeros(12, dtype=bool)
        mask[4:7] = True
        controls = candidate_intervals(
            self.band, self.record, mask, "q3_only", 1.0, self.args,
        )
        widths = (
            controls.control_end_index_0based_inclusive
            - controls.control_start_index_0based + 1
        )
        self.assertTrue((widths == 3).all())
        for row in controls.itertuples(index=False):
            self.assertFalse(mask[
                row.control_start_index_0based: row.control_end_index_0based_inclusive + 1
            ].any())

    def test_position_is_used_only_by_final_nested_scheme(self):
        mask = np.zeros(12, dtype=bool)
        mask[4:7] = True
        q3_only = candidate_intervals(
            self.band, self.record, mask, "q3_only", 1.0, self.args,
        )
        final = candidate_intervals(
            self.band, self.record, mask, "q3_neq_rsa_position", 1.0, self.args,
        )
        self.assertGreater(len(q3_only), len(final))
        self.assertTrue(
            (final.delta_position_control_minus_apex.abs() <= self.args.position_caliper).all()
        )

    def test_band_case_features_are_interval_means(self):
        row = band_case_row(self.band, self.record)
        expected = np.mean(np.arange(12, dtype=float)[4:7] + 1)
        self.assertEqual(row["band_width"], 3)
        self.assertAlmostEqual(row["interval_mean_ca_curvature_degrees"], expected)

    def test_interval_inference_is_protein_clustered(self):
        cases = []
        controls = []
        for index in range(4):
            case = {
                "band_id": f"b{index}", "condition": "c", "split": "test",
                "protein": f"p{index}", "sign": 1,
            }
            control = {
                "band_id": f"b{index}", "cohort": "fully_resolved",
                "match_scheme": "q3_only",
            }
            for feature in STRUCTURE_FEATURES:
                case[f"interval_mean_{feature}"] = 2.0
                control[f"control_interval_mean_{feature}"] = 1.0
            cases.append(case)
            controls.append(control)
        args = types.SimpleNamespace(
            random_seed=123, n_bootstrap=200, n_sign_flips=500,
            minimum_inference_proteins=2,
        )
        by_protein, summary = matched_effects(
            pd.DataFrame(cases), pd.DataFrame(controls), args,
        )
        self.assertEqual(by_protein.protein.nunique(), 4)
        self.assertTrue((summary.mean_band_minus_control == 1.0).all())
        self.assertTrue((summary.bootstrap_ci95_low == 1.0).all())
        self.assertTrue((summary.bootstrap_ci95_high == 1.0).all())
        self.assertTrue(summary.sign_flip_p_two_sided.notna().all())


if __name__ == "__main__":
    unittest.main()
