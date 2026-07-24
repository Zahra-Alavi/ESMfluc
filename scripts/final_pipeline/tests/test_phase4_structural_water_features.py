import types
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
