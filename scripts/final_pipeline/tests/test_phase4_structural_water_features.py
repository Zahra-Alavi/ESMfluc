import types
import unittest

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_query_receivers import (
    PROFILE_METRICS,
    receiver_rows_for_profile,
)
from signed_band_analysis.build_band_query_structural_water_features import (
    Atom,
    adjacency_lists,
    multi_source_bfs,
    retained_water_network,
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
