import types
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_apex_structure import (
    band_masks,
    candidate_control_pool,
    clustered_ols,
    design_matrix,
    matched_effects,
    union_group_network_inference,
)


class BandApexStructureTests(unittest.TestCase):
    def test_controls_are_same_protein_outside_all_bands_and_q3_matched(self):
        bands = pd.DataFrame({
            "condition": ["c", "c"], "protein": ["p", "p"],
            "protein_length": [10, 10],
            "start_index_0based": [2, 6],
            "end_index_0based_inclusive": [3, 7],
        })
        masks = band_masks(bands)
        residues = pd.DataFrame({
            "protein": ["p"] * 10,
            "residue_index_0based": np.arange(10),
            "structure_resolved": [True] * 10,
            "normalized_position": np.arange(10) / 9,
            "q3": list("CCHHCCCCCC"),
            "neq": np.linspace(1, 2, 10),
            "rsa": np.linspace(0, 0.9, 10),
        })
        case = types.SimpleNamespace(
            protein="p", condition="c", split="test", band_id="b1",
            q3="C", neq=1.5, rsa=0.5,
            normalized_position=5 / 9, eligible_start_index_0based=0,
            eligible_end_index_0based_exclusive=10,
        )
        args = types.SimpleNamespace(
            position_caliper=0.5, neq_caliper=1.0, rsa_caliper=1.0,
            max_controls_per_apex=5, random_seed=123, terminal_exclusion=0,
        )
        selected = candidate_control_pool(case, {"p": residues}, masks, "q3_only", args)
        self.assertTrue((selected.q3 == "C").all())
        self.assertTrue(set(selected.residue_index_0based).isdisjoint({2, 3, 6, 7}))

    def test_phase2_nested_matching_adds_position_only_in_final_scheme(self):
        bands = pd.DataFrame({
            "condition": ["c"], "protein": ["p"], "protein_length": [10],
            "start_index_0based": [4], "end_index_0based_inclusive": [4],
        })
        masks = band_masks(bands)
        residues = pd.DataFrame({
            "protein": ["p"] * 10,
            "residue_index_0based": np.arange(10),
            "structure_resolved": [True] * 10,
            "normalized_position": np.arange(10) / 9,
            "q3": ["C"] * 10,
            "neq": [1.0] * 10,
            "rsa": [0.2] * 10,
        })
        case = types.SimpleNamespace(
            protein="p", condition="c", split="test", band_id="b1",
            q3="C", neq=1.0, rsa=0.2,
            normalized_position=4 / 9, eligible_start_index_0based=0,
            eligible_end_index_0based_exclusive=10,
        )
        args = types.SimpleNamespace(
            position_caliper=0.12, neq_caliper=0.25, rsa_caliper=0.15,
            max_controls_per_apex=5, random_seed=123, terminal_exclusion=0,
        )
        q3_only = candidate_control_pool(case, {"p": residues}, masks, "q3_only", args)
        final = candidate_control_pool(
            case, {"p": residues}, masks, "q3_neq_rsa_position", args,
        )
        self.assertGreater(len(q3_only), len(final))
        self.assertTrue((final.abs_delta_position <= args.position_caliper).all())

    def test_matched_inference_uses_sign_flips_and_protein_bootstrap(self):
        apex = pd.DataFrame({
            "band_id": [f"b{i}" for i in range(4)],
            "condition": ["c"] * 4, "split": ["test"] * 4,
            "protein": [f"p{i}" for i in range(4)], "sign": [1] * 4,
            "contact_degree": [2.0, 3.0, 4.0, 5.0],
        })
        controls = pd.DataFrame({
            "band_id": [f"b{i}" for i in range(4)],
            "match_model": ["q3_only"] * 4,
            "control_contact_degree": [1.0, 2.0, 3.0, 4.0],
        })
        for feature in (
            "ca_curvature_degrees", "abs_ca_virtual_torsion_degrees",
            "ca_packing_index", "mean_contact_distance_angstrom", "betweenness",
            "closeness", "participation_coefficient", "community_boundary",
        ):
            apex[feature] = np.nan
            controls[f"control_{feature}"] = np.nan
        args = types.SimpleNamespace(
            random_seed=123, n_bootstrap=200, n_sign_flips=500,
            minimum_inference_proteins=2,
        )
        _, summary = matched_effects(apex, controls, args)
        row = summary.loc[summary.feature.eq("contact_degree")].iloc[0]
        self.assertEqual(row.mean_apex_minus_control, 1.0)
        self.assertEqual(row.bootstrap_ci95_low, 1.0)
        self.assertEqual(row.bootstrap_ci95_high, 1.0)
        self.assertTrue(np.isfinite(row.sign_flip_p_two_sided))
        self.assertTrue(np.isfinite(row.sign_flip_q_bh_global))

    def test_R_on_X_fixed_effect_fit_recovers_positive_association(self):
        rows = []
        for protein_number, protein in enumerate(("p1", "p2", "p3", "p4")):
            for index in range(8):
                x = index - 3.5
                rows.append({
                    "protein": protein, "R": protein_number + 0.4 * x,
                    "contact_degree": x, "normalized_position": index / 7,
                    "strict_3_of_3": index % 2, "q3": "C" if index % 2 else "H",
                    "neq": 1 + index / 10, "rsa": index / 10,
                })
        frame = pd.DataFrame(rows)
        data, matrix, response, names = design_matrix(
            frame, "contact_degree", ["strict_3_of_3"],
        )
        fit = clustered_ols(
            data, matrix, response, names, "contact_degree",
        )
        self.assertIsNotNone(fit)
        self.assertGreater(fit["coefficient_R_per_1sd_X"], 0)
        self.assertEqual(fit["n_proteins"], 4)
        protein_dummies = pd.get_dummies(
            data.protein, drop_first=True
        ).to_numpy(float)
        strict = (
            data.strict_3_of_3.to_numpy(float)
            - data.strict_3_of_3.mean()
        ) / data.strict_3_of_3.std(ddof=0)
        x = (
            data.contact_degree.to_numpy(float)
            - data.contact_degree.mean()
        ) / data.contact_degree.std(ddof=0)
        explicit = np.column_stack([
            np.ones(len(data)), protein_dummies, strict, x,
        ])
        explicit_beta = np.linalg.lstsq(
            explicit, data.R.to_numpy(float), rcond=None,
        )[0][-1]
        self.assertAlmostEqual(
            fit["coefficient_R_per_1sd_X"], explicit_beta, places=10,
        )

    def test_union_group_network_inference_uses_group_means(self):
        by_protein = pd.DataFrame({
            "condition": ["c"] * 4, "split": ["test"] * 4,
            "protein": ["p1", "p2", "p3", "p4"], "sign": [1] * 4,
            "match_model": ["q3_neq"] * 4,
            "feature": ["contact_degree"] * 4,
            "apex_minus_control": [1.0, 3.0, 5.0, 7.0],
        })
        manifest = pd.DataFrame({
            "name": ["p1", "p2", "p3", "p4"], "split": ["test"] * 4,
            "union_group_id": ["g1", "g1", "g2", "g3"],
        })
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "groups.csv"
            manifest.to_csv(path, index=False)
            args = types.SimpleNamespace(
                group_manifest_csv=str(path), random_seed=123,
                n_bootstrap=200, n_sign_flips=100,
                minimum_inference_proteins=2,
            )
            result = union_group_network_inference(by_protein, args)
        row = result.iloc[0]
        self.assertEqual(row.n_union_groups, 3)
        self.assertAlmostEqual(row.union_group_mean_effect, (2 + 5 + 7) / 3)
        self.assertEqual(row.primary_p_two_sided, row.union_group_p_two_sided)


if __name__ == "__main__":
    unittest.main()
