import types
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_query_receivers import (
    PROFILE_METRICS,
    add_mechanism_interactions,
    receiver_rows_for_profile,
)
from signed_band_analysis.build_band_query_structural_water_features import (
    Atom,
    BAND_FEATURES,
    PAIR_FEATURES,
    adjacency_lists,
    feature_store_is_current,
    multi_source_bfs,
    retained_water_network,
    save_feature_store,
    unique_majority,
)


def atom(
    serial,
    name,
    resname,
    chain,
    resid,
    xyz,
    element,
    occupancy=1.0,
    bfactor=10.0,
    record="ATOM",
):
    return Atom(
        record=record,
        serial=serial,
        name=name,
        altloc="",
        resname=resname,
        chain=chain,
        resid=resid,
        icode="",
        coordinate=xyz,
        occupancy=occupancy,
        bfactor=bfactor,
        element=element,
    )


class Phase4StructuralWaterHelperTests(unittest.TestCase):
    def test_feature_store_reuse_requires_signature_and_band_identity(self):
        bands = pd.DataFrame({
            "band_id": ["band_1"],
            "sign": [1],
            "apex_index_0based": [1],
            "start_index_0based": [0],
            "end_index_0based_inclusive": [2],
        })
        features = {
            name: np.zeros((1, 3), dtype=np.float32)
            for name in PAIR_FEATURES
        }
        features.update({
            name: np.zeros(1, dtype=np.float32)
            for name in BAND_FEATURES
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "feature.npz"
            save_feature_store(
                path, "condition", "test", "protein", bands, 3,
                features, "signature-A",
            )
            self.assertTrue(
                feature_store_is_current(path, bands, 3, "signature-A")
            )
            self.assertFalse(
                feature_store_is_current(path, bands, 3, "signature-B")
            )
            changed = bands.copy()
            changed.loc[0, "apex_index_0based"] = 2
            self.assertFalse(
                feature_store_is_current(path, changed, 3, "signature-A")
            )

    def test_mechanism_interactions_use_phase3c_class_names(self):
        frame = pd.DataFrame({
            "mechanism_class": [
                "combined", "evidence_dominated",
                "consultation_dominated", "not_magnitude_enriched",
                "unclassified",
            ],
            "query_q8": ["H"] * 5,
            "query_amino_acid_class": ["aliphatic"] * 5,
            "neq": [1.0, 2.0, 3.0, 4.0, 5.0],
            "rsa": [0.1] * 5,
            "disorder": [0.0] * 5,
            "torsion_change_from_previous": [10.0] * 5,
            "query_in_any_band": [1.0] * 5,
        })

        result, columns = add_mechanism_interactions(frame)

        evidence = "interaction__evidence_dominated__neq"
        consultation = "interaction__consultation_dominated__neq"
        self.assertIn(evidence, columns)
        self.assertIn(consultation, columns)
        np.testing.assert_allclose(result[evidence], [0, 2, 0, 0, 0])
        np.testing.assert_allclose(result[consultation], [0, 0, 3, 0, 0])

    def test_multisource_bfs_distinguishes_disconnection_from_zero_length(self):
        adjacency = adjacency_lists(
            5,
            [(0, 1, 4.0), (1, 2, 4.0), (3, 4, 4.0)],
        )
        distance = multi_source_bfs(adjacency, [0])
        np.testing.assert_allclose(distance[:3], [0.0, 1.0, 2.0])
        self.assertTrue(np.isnan(distance[3:]).all())

    def test_unique_majority_does_not_break_ties_arbitrarily(self):
        self.assertEqual(unique_majority([1, 1, 2]), 1)
        self.assertIsNone(unique_majority([1, 2]))
        self.assertIsNone(unique_majority([-1, -1]))

    def test_primary_chain_policy_excludes_interface_water(self):
        target_polar = [
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[6.0, 0.0, 0.0]]),
        ]
        target_atoms = [
            atom(1, "O", "ALA", "A", 1, (0.0, 0.0, 0.0), "O"),
            atom(2, "O", "ALA", "A", 2, (6.0, 0.0, 0.0), "O"),
        ]
        foreign = [
            atom(3, "O", "ALA", "B", 1, (0.0, 2.0, 0.0), "O"),
        ]
        atoms = target_atoms + foreign + [
            atom(
                4, "O", "HOH", "A", 10, (0.0, 1.0, 0.0), "O",
                record="HETATM",
            ),
            atom(
                5, "O", "HOH", "A", 11, (3.0, 0.0, 0.0), "O",
                record="HETATM",
            ),
        ]
        args = types.SimpleNamespace(
            maximum_water_bfactor_robust_z=3.0,
            minimum_water_occupancy=0.5,
            protein_water_cutoff=3.1,
            water_water_cutoff=3.5,
            water_chain_policy="target_only_unambiguous",
        )
        result = retained_water_network(
            atoms, target_polar, target_atoms, foreign, args
        )
        self.assertEqual(result["excluded_foreign_chain_contact_count"], 1)
        self.assertEqual(result["retained_water_count"], 1)
        self.assertEqual(result["residue_to_waters"], [{0}, {0}])

    def test_water_chain_identifier_is_not_used_for_assignment(self):
        target_polar = [np.asarray([[0.0, 0.0, 0.0]])]
        target_atoms = [
            atom(1, "O", "ALA", "A", 1, (0.0, 0.0, 0.0), "O"),
        ]
        waters = [
            atom(
                2, "O", "HOH", "Z", 1, (0.0, 2.5, 0.0), "O",
                record="HETATM",
            )
        ]
        args = types.SimpleNamespace(
            maximum_water_bfactor_robust_z=3.0,
            minimum_water_occupancy=0.5,
            protein_water_cutoff=3.5,
            water_water_cutoff=3.5,
            water_chain_policy="target_only_unambiguous",
        )
        result = retained_water_network(
            target_atoms + waters, target_polar, target_atoms, [], args
        )
        self.assertEqual(result["retained_water_count"], 1)
        self.assertEqual(result["residue_to_waters"], [{0}])

    def test_pair_features_follow_receiver_eligibility_mask(self):
        length = 5
        profile = {
            "protein_length": np.asarray([length]),
            "band_id": np.asarray(["band_1"]),
            "sign": np.asarray([1]),
            "apex_index_0based": np.asarray([2]),
            "start_index_0based": np.asarray([2]),
            "end_index_0based_inclusive": np.asarray([2]),
        }
        for offset, metric in enumerate(PROFILE_METRICS):
            profile[metric] = np.asarray(
                [[0.1, 0.2, 0.3, 0.4, 0.5 + offset]]
            )
        residue = pd.DataFrame({
            "residue_index_0based": np.arange(length),
            "amino_acid": list("ACDEF"),
            "q8": ["H"] * length,
            "neq": np.arange(length, dtype=float),
            "rsa": np.arange(length, dtype=float),
            "disorder": np.arange(length, dtype=float),
            "torsion_change_from_previous": np.arange(length, dtype=float),
        })
        bands = pd.DataFrame({
            "band_id": ["band_1"],
            "start_index_0based": [2],
            "end_index_0based_inclusive": [2],
        })
        summary = pd.Series({
            "eligible_start_index_0based": 1,
            "eligible_end_index_0based_exclusive": 4,
        })
        structure = {
            "pair_feature_names": np.asarray(
                ["minimum_ca_distance_angstrom"]
            ),
            "band_feature_names": np.asarray([], dtype=str),
            "minimum_ca_distance_angstrom": np.asarray(
                [[100.0, 101.0, 102.0, 103.0, 104.0]]
            ),
        }

        pairs, _, _ = receiver_rows_for_profile(
            profile,
            "condition",
            "test",
            "protein",
            residue,
            bands,
            summary,
            {},
            receiver_quantile=0.90,
            low_quantile=0.50,
            long_range_min=21,
            structure=structure,
        )

        np.testing.assert_array_equal(
            pairs["query_index_0based"].to_numpy(), [1, 2, 3]
        )
        np.testing.assert_allclose(
            pairs["minimum_ca_distance_angstrom"].to_numpy(),
            [101.0, 102.0, 103.0],
        )


if __name__ == "__main__":
    unittest.main()
