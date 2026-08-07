import inspect
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from prediction_endpoint_analysis.analyze_neq_pb_reliability import (
    _pb_metrics_from_array,
    auc_contribution_matrix,
    binary_performance,
    bootstrap_auc_from_matrix,
    cache_provenance_matches,
    consensus_mask_labels,
    endpoint_mask_labels,
    iter_prediction_records,
    main,
    source_file_provenance,
    validate_pb_cache_tables,
    validate_prediction_arrays,
    weighted_spearman_bootstrap,
)


class NeqPbReliabilityTests(unittest.TestCase):
    def test_pb_state_definitions_exclude_z_and_use_valid_frames(self):
        array = np.asarray(
            [[b"a", b"a", b"Z"], [b"a", b"b", b"Z"], [b"a", b"b", b"Z"]],
            dtype="S1",
        )
        result = _pb_metrics_from_array(array)
        self.assertEqual(result["valid_frames"].tolist(), [3, 3, 0])
        self.assertEqual(result["n_states"].tolist(), [1, 2, 0])
        self.assertEqual(result["neq"][0], 1.0)
        self.assertTrue(np.isnan(result["neq"][2]))
        self.assertTrue(np.isclose(result["secondary_occupancy"][1], 1 / 3))

    def test_two_equal_states_have_neq_two(self):
        array = np.asarray([[b"a"], [b"a"], [b"b"], [b"b"]], dtype="S1")
        self.assertTrue(np.isclose(_pb_metrics_from_array(array)["neq"][0], 2.0))

    def test_consensus_masks_cover_all_four_categories(self):
        counts = np.array([0, 1, 2, 3])
        include, labels = consensus_mask_labels(counts, "any")
        self.assertEqual(include.tolist(), [True, True, True, True])
        self.assertEqual(labels.tolist(), [0, 1, 1, 1])
        include, labels = consensus_mask_labels(counts, "majority")
        self.assertEqual(include.tolist(), [True, False, True, True])
        self.assertEqual(labels[include].tolist(), [0, 1, 1])
        include, labels = consensus_mask_labels(counts, "unanimous")
        self.assertEqual(include.tolist(), [True, False, False, True])
        self.assertEqual(labels[include].tolist(), [0, 1])

    def test_streaming_prediction_reader_skips_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            path.write_text(
                "[\n"
                "  {\n"
                '    "name":"p",\n'
                '    "sequence":"AA",\n'
                '    "attention_weights":[\n'
                "      [1, 0],\n"
                "      [0, 1]\n"
                "    ],\n"
                '    "neq_preds":[0,1],\n'
                '    "flexible_scores":[0.1,0.9],\n'
                '    "class_probs":[[0.9,0.1],[0.1,0.9]]\n'
                "  }\n"
                "]\n"
            )
            records = list(iter_prediction_records(path))
        self.assertEqual(
            records,
            [
                {
                    "name": "p",
                    "sequence": "AA",
                    "neq_preds": [0, 1],
                    "flexible_scores": [0.1, 0.9],
                    "class_probs": [[0.9, 0.1], [0.1, 0.9]],
                }
            ],
        )

    def test_streaming_prediction_reader_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            path.write_text('[\n  {\n    "attention_weights":[\n')
            with self.assertRaises(ValueError):
                list(iter_prediction_records(path))

    def test_prediction_probability_and_hard_label_validation(self):
        valid = {
            "neq_preds": [0, 1],
            "flexible_scores": [0.2, 0.8],
            "class_probs": [[0.8, 0.2], [0.2, 0.8]],
        }
        hard, score, probabilities, difference = validate_prediction_arrays(
            valid, "AA"
        )
        self.assertEqual(hard.tolist(), [0, 1])
        self.assertEqual(score.tolist(), [0.2, 0.8])
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertEqual(difference, 0.0)

        for field, value in (
            ("flexible_scores", [0.3, 0.8]),
            ("neq_preds", [1, 1]),
            ("class_probs", [[0.8, 0.2]]),
        ):
            invalid = dict(valid)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_prediction_arrays(invalid, "AA")

    def test_exclusion_and_relabeling_threshold_semantics(self):
        neq = np.array([1.0, 1.005, 1.02, 2.5])
        include, labels = endpoint_mask_labels(neq, 1.01, "exclusion_margin")
        self.assertEqual(include.tolist(), [True, False, True, True])
        self.assertEqual(labels[include].tolist(), [0, 1, 1])
        include_re, labels_re = endpoint_mask_labels(
            neq, 1.01, "relabeled_threshold"
        )
        self.assertTrue(include_re.all())
        self.assertEqual(labels_re.tolist(), [0, 0, 1, 1])

        include_one, labels_one = endpoint_mask_labels(
            neq, 1.0, "exclusion_margin"
        )
        include_re_one, labels_re_one = endpoint_mask_labels(
            neq, 1.0, "relabeled_threshold"
        )
        self.assertTrue(include_one.all())
        self.assertTrue(include_re_one.all())
        self.assertEqual(labels_one.tolist(), labels_re_one.tolist())

    def test_binary_performance_records_prevalence(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.7, 0.9])
        proteins = np.array(["p1", "p1", "p2", "p2"])
        result = binary_performance(labels, scores, proteins)
        self.assertEqual(result["n_positive"], 2)
        self.assertEqual(result["positive_fraction"], 0.5)
        self.assertEqual(result["auroc"], 1.0)

    def test_source_provenance_detects_size_and_mtime_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.dat"
            path.write_text("abc")
            first = source_file_provenance([path], "test-cache-v1")
            self.assertTrue(
                cache_provenance_matches({"source_provenance": first}, first)
            )
            path.write_text("abcd")
            second = source_file_provenance([path], "test-cache-v1")
            self.assertNotEqual(first["digest"], second["digest"])
            self.assertFalse(
                cache_provenance_matches({"source_provenance": first}, second)
            )
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            third = source_file_provenance([path], "test-cache-v1")
            self.assertNotEqual(second["digest"], third["digest"])

    def test_cache_schema_version_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.dat"
            path.write_text("unchanged")
            version_one = source_file_provenance([path], "pb-cache-v1")
            version_two = source_file_provenance([path], "pb-cache-v2")
            self.assertNotEqual(version_one["digest"], version_two["digest"])
            self.assertFalse(
                cache_provenance_matches(
                    {"source_provenance": version_one}, version_two
                )
            )

    def test_pb_cache_tables_are_revalidated_on_reuse(self):
        residues = pd.DataFrame(
            {
                "row_id": [0, 1],
                "protein": ["p", "p"],
                "residue_index_0based": [0, 1],
                "residue_number_1based": [1, 2],
                "amino_acid": ["A", "C"],
                "sequence_length": [2, 2],
                "original_neq": [1.0, 1.5],
                "original_binary_label": [0, 1],
            }
        )
        replicate_rows = []
        for replicate_id in (1, 2, 3):
            for residue_index in (0, 1):
                replicate_rows.append(
                    {
                        "protein": "p",
                        "residue_index_0based": residue_index,
                        "residue_number_1based": residue_index + 1,
                        "replicate": replicate_id,
                        "total_frames": 10,
                        "valid_pb_frame_count": 10,
                        "valid_pb_frame_fraction": 1.0,
                        "pb_valid": True,
                        "n_observed_non_z_states": residue_index + 1,
                        "single_state": residue_index == 0,
                        "multi_state": residue_index == 1,
                        "pb_entropy": 0.0 if residue_index == 0 else 0.4,
                        "replicate_neq": 1.0 if residue_index == 0 else 1.5,
                        "dominant_pb_state": "a",
                        "dominant_state_occupancy": 1.0,
                        "second_pb_state": "" if residue_index == 0 else "b",
                        "secondary_state_occupancy": 0.0 if residue_index == 0 else 0.2,
                    }
                )
        replicate = pd.DataFrame(replicate_rows)
        consensus = residues.copy()
        consensus = consensus.assign(
            n_valid_replicates=3,
            all_three_replicates_pb_valid=True,
            n_single_state_replicates=[3, 0],
            n_multi_state_replicates=[0, 3],
            multi_state_consensus_category=["0/3", "3/3"],
            mean_replicate_neq=[1.0, 1.5],
            sd_replicate_neq=[0.0, 0.0],
            pooled_dominant_pb_state=["a", "a"],
            dominant_pb_state_agrees_all_replicates=True,
            maximum_secondary_state_occupancy=[0.0, 0.2],
            mean_secondary_state_occupancy=[0.0, 0.2],
            minimum_valid_pb_frame_fraction=1.0,
            mean_valid_pb_frame_fraction=1.0,
        )
        for suffix in ("gt0", "0p1pct", "0p5pct", "1pct", "2pct", "5pct"):
            consensus[f"n_multi_replicates_occ_{suffix}"] = [0, 3]
        comparison = consensus[
            [
                "row_id",
                "protein",
                "residue_index_0based",
                "residue_number_1based",
                "sequence_length",
                "original_neq",
                "original_binary_label",
                "n_valid_replicates",
                "all_three_replicates_pb_valid",
                "n_multi_state_replicates",
                "mean_replicate_neq",
                "minimum_valid_pb_frame_fraction",
                "mean_valid_pb_frame_fraction",
            ]
        ].copy()
        comparison = comparison.assign(
            regenerated_binary_label=[0, 1],
            binary_label_agrees=True,
            absolute_neq_difference=0.0,
            terminal_distance_0based=[0, 0],
            within_two_residues_of_terminus=True,
            within_five_residues_of_terminus=True,
            has_any_z_frames=False,
        )

        validate_pb_cache_tables(replicate, consensus, comparison, residues)

        corruptions = [
            (replicate.drop(columns=["pb_entropy"]), consensus, comparison),
            (
                replicate.assign(
                    replicate=[1, 1, 1, 2, 3, 3]
                ),
                consensus,
                comparison,
            ),
            (replicate, consensus.iloc[::-1].reset_index(drop=True), comparison),
            (replicate, consensus, comparison.iloc[:-1]),
        ]
        for index, tables in enumerate(corruptions):
            with self.subTest(corruption=index), self.assertRaises(ValueError):
                validate_pb_cache_tables(*tables, residues)

    def test_main_generates_documentation_before_required_output_check(self):
        source = inspect.getsource(main)
        required_check = source.index("missing = [")
        self.assertLess(source.index("write_readme("), required_check)
        self.assertLess(source.index("write_manuscript_text("), required_check)
        complete_audit_write = source.index(
            '_json_dump(args.output_dir / "audits" / "complete_analysis_audit.json"'
        )
        self.assertGreater(complete_audit_write, required_check)
        self.assertNotIn("_json_dump(", source[complete_audit_write + 1 :])

    def test_protein_bootstrap_multiplicity_matches_explicit_duplication(self):
        labels = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.3, 0.7, 0.9, 0.2, 0.8])
        protein = np.array([0, 0, 1, 1, 1])
        matrix, positive, negative = auc_contribution_matrix(
            labels, scores, protein, 2
        )
        weights = np.array([[1, 1], [2, 1], [1, 2]])
        observed = bootstrap_auc_from_matrix(
            matrix, positive, negative, weights
        )
        expected = []
        for draw in weights:
            indices = np.concatenate(
                [np.tile(np.flatnonzero(protein == p), draw[p]) for p in range(2)]
            )
            expected.append(roc_auc_score(labels[indices], scores[indices]))
        self.assertTrue(np.allclose(observed, expected))

    def test_weighted_spearman_matches_explicit_protein_duplication(self):
        x = np.array([1.1, 1.3, 2.0, 1.2, 1.8])
        y = np.array([0.2, 0.8, 0.7, 0.1, 0.9])
        protein = np.array([0, 0, 0, 1, 1])
        weights = np.array([[1, 1], [2, 1], [1, 2]])
        observed = weighted_spearman_bootstrap(x, y, protein, weights)
        expected = []
        for draw in weights:
            indices = np.concatenate(
                [np.tile(np.flatnonzero(protein == p), draw[p]) for p in range(2)]
            )
            expected.append(spearmanr(x[indices], y[indices]).statistic)
        self.assertTrue(np.allclose(observed, expected))

    def test_fixed_seed_reproduces_paired_protein_draws(self):
        first = np.random.default_rng(20260730).multinomial(
            3, np.full(3, 1 / 3), size=25
        )
        second = np.random.default_rng(20260730).multinomial(
            3, np.full(3, 1 / 3), size=25
        )
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
