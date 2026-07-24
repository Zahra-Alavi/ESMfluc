import types
import unittest

import numpy as np

from signed_band_analysis.analyze_signed_band_external_structure import (
    audit_contacts,
    ca_curvature,
    ca_virtual_torsion,
    graph_features,
)


class ExternalStructureHelperTests(unittest.TestCase):
    def test_straight_ca_trace_has_zero_curvature_and_virtual_torsion(self):
        coordinates = [[float(index), 0.0, 0.0] for index in range(5)]
        curvature = ca_curvature(coordinates)
        torsion = ca_virtual_torsion(coordinates)
        np.testing.assert_allclose(curvature[1:-1], 0.0)
        self.assertTrue(np.isnan(torsion[1]))  # collinear planes have undefined torsion

    def test_contact_graph_excludes_near_sequence_edges_and_keeps_distances(self):
        record = {
            "name": "test_A", "sequence": "AAAAA", "resolved_mask": [True] * 5,
            "contact_edges": [[0, 1, 3.8], [0, 3, 6.0], [1, 4, 8.0]],
        }
        args = types.SimpleNamespace(
            min_contact_sequence_separation=3, betweenness_samples=64, random_seed=123,
        )
        node, edges = graph_features(record, args)
        self.assertEqual(edges, [(0, 3, 6.0), (1, 4, 8.0)])
        self.assertEqual(node["contact_degree"][0], 1.0)
        self.assertAlmostEqual(node["inverse_distance_weighted_degree"][0], 1 / 6)

    def test_mapping_audit_includes_expected_proteins_without_contact_records(self):
        record = {
            "name": "present_A", "status": "ok", "mapping_status": "exact",
            "sequence": "AAAA", "contact_map_coordinate_system": "input_sequence_0based",
            "alignment": {"identity": 1.0, "input_coverage": 1.0},
            "sequence_check": {"len_input": 4, "len_structure": 4},
        }
        args = types.SimpleNamespace(min_mapping_identity=0.9, min_input_coverage=0.8)
        audit = audit_contacts(
            {"present_A": record}, {"present_A": "train", "missing_A": "test"}, args
        ).set_index("protein")
        self.assertEqual(audit.loc["missing_A", "status"], "missing_contact_record")
        self.assertFalse(bool(audit.loc["missing_A", "mapping_accepted"]))
        self.assertTrue(bool(audit.loc["present_A", "mapping_accepted"]))


if __name__ == "__main__":
    unittest.main()
