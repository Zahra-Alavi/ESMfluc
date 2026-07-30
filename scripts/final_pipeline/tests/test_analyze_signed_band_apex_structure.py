import types
import unittest

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_apex_structure import (
    band_masks,
    candidate_control_pool,
    clustered_ols,
    design_matrix,
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
            protein="p", condition="c", q3="C", neq=1.5, rsa=0.5,
            normalized_position=5 / 9, eligible_start_index_0based=0,
            eligible_end_index_0based_exclusive=10,
        )
        args = types.SimpleNamespace(
            position_caliper=0.5, neq_caliper=1.0, rsa_caliper=1.0,
            max_controls_per_apex=5,
        )
        selected = candidate_control_pool(
            case, {"p": residues}, masks, "M2_add_q3", args,
        )
        self.assertTrue((selected.q3 == "C").all())
        self.assertTrue(set(selected.residue_index_0based).isdisjoint({2, 3, 6, 7}))
        self.assertTrue((selected.abs_delta_position <= args.position_caliper).all())

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


if __name__ == "__main__":
    unittest.main()
