import unittest

from Attention.build_contact_maps_from_pdb import (
    align_structure_to_input,
    default_audit_path,
    mapping_quality,
    parse_seq_id,
)


class ContactMapBuilderMappingTests(unittest.TestCase):
    def test_internal_unresolved_residue_is_not_shifted_to_terminus(self):
        result = align_structure_to_input("ACDEFG", "ACEFG")
        self.assertEqual(result["input_to_structure_index"], [0, 1, None, 2, 3, 4])
        self.assertEqual(result["identity"], 1.0)
        self.assertAlmostEqual(result["input_coverage"], 5 / 6)

    def test_input_domain_maps_to_longer_structure_chain(self):
        result = align_structure_to_input("ACDE", "XXACDEYY")
        self.assertEqual(result["input_to_structure_index"], [2, 3, 4, 5])
        self.assertEqual(result["structure_to_input_index"], [None, None, 0, 1, 2, 3, None, None])
        self.assertEqual(result["identity"], 1.0)
        self.assertEqual(result["input_coverage"], 1.0)

    def test_structure_subsequence_marks_unresolved_input_flanks(self):
        result = align_structure_to_input("XXACDEYY", "ACDE")
        self.assertEqual(
            result["input_to_structure_index"],
            [None, None, 0, 1, 2, 3, None, None],
        )
        self.assertEqual(result["input_unresolved_count"], 4)

    def test_quality_status_uses_identity_and_input_coverage(self):
        exact = align_structure_to_input("ACDE", "ACDE")
        aligned = align_structure_to_input("ACDEF", "ACDE")
        low_coverage = align_structure_to_input("XXXXACDEXXXX", "ACDE")
        self.assertEqual(mapping_quality(exact, 0.9, 0.8), "exact")
        self.assertEqual(mapping_quality(aligned, 0.9, 0.8), "aligned")
        self.assertEqual(mapping_quality(low_coverage, 0.9, 0.8), "low_quality")

    def test_identifier_and_default_audit_paths(self):
        self.assertEqual(parse_seq_id("2pbk_B"), ("2pbk", "B"))
        self.assertEqual(parse_seq_id("1a39A00"), ("1a39", "A"))
        self.assertEqual(default_audit_path("out/maps.json.gz"), "out/maps.mapping_audit.csv")


if __name__ == "__main__":
    unittest.main()
