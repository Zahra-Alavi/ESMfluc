import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmark_unified_mdcath import (
    DEFAULT_RUNS_DIR,
    LINEAR_CONDITION,
    LINEAR_CACHE_SCHEMA_VERSION,
    LINEAR_EQUIVALENCE_TOLERANCE,
    LINEAR_INFERENCE_IMPLEMENTATION,
    LINEAR_MODEL_NAME,
    MethodScores,
    _sha256_file,
    _read_netsurfp_table,
    _validate_linear_cache,
    load_cohort,
    load_long_predictors,
    load_verified_linear_model,
    main,
    ordinary_linear_checkpoint_probabilities,
    shared_backbone_linear_probabilities,
    tokenize_linear_sequences,
    validate_method_coverage,
)
from extract_mdcath_rmsf import align_to_reference, rmsf_for_replica


class TestUnifiedBenchmarkValidation(unittest.TestCase):
    def test_load_cohort_excludes_length_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.csv"
            pd.DataFrame({
                "domain": ["ok", "bad"],
                "sequence": ["ACD", "ACDE"],
                "neq": ["[1, 2, 3]", "[1, 2, 3]"],
            }).to_csv(path, index=False)

            cohort, neq, invalid = load_cohort(path)

        self.assertEqual(len(cohort), 2)
        self.assertEqual(set(neq), {"ok"})
        self.assertEqual(invalid, [{
            "domain": "bad", "sequence_length": 4, "neq_length": 3,
            "reason": "neq_length_mismatch",
        }])

    def test_load_cohort_rejects_duplicates_after_prefix_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.csv"
            pd.DataFrame({
                "domain": [">same", "same"],
                "sequence": ["ACD", "ACD"],
                "neq": ["[1, 2, 3]", "[1, 2, 3]"],
            }).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "after normalization"):
                load_cohort(path)

    def test_netsurfp_zip_and_exact_coordinate_join(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            table = pd.DataFrame({
                "id": [">d1", ">d1", ">d1"],
                "seq": ["A", "C", "D"],
                "n": [1, 2, 3],
                "rsa": [0.1, 0.2, 0.3],
                "disorder": [0.4, 0.5, 0.6],
            })
            csv_path = directory / "results.csv"
            table.to_csv(csv_path, index=False)
            zip_path = directory / "results.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(csv_path, arcname="results.csv")

            loaded_table = _read_netsurfp_table(zip_path)
            scores, invalid = load_long_predictors(
                loaded_table, {"d1": "ACD"},
                {"rsa": "rsa", "disorder": "disorder"}, "test",
            )

        self.assertEqual(invalid, {})
        np.testing.assert_allclose(scores["rsa"]["d1"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(scores["disorder"]["d1"], [0.4, 0.5, 0.6])

    def test_coverage_rejects_wrong_length(self):
        method = MethodScores(
            "m", "M", "external_sequence", "synthetic",
            {"d1": np.array([0.1, 0.2]), "d2": np.array([0.3])},
        )
        audit = validate_method_coverage(
            [method], {"d1": "AC", "d2": "AC"}, {"d1", "d2"}
        )["m"]

        self.assertEqual(audit["n_valid"], 1)
        self.assertEqual(audit["n_missing"], 0)
        self.assertEqual(audit["wrong_length"], {
            "d2": {"score_length": 1, "sequence_length": 2}
        })

    def test_rmsf_alignment_removes_rigid_body_motion(self):
        reference = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        moved = reference @ rotation + np.array([4.0, -2.0, 3.0])
        trajectory = np.stack([reference, moved])

        aligned = align_to_reference(trajectory, reference)

        np.testing.assert_allclose(aligned[0], reference, atol=1e-12)
        np.testing.assert_allclose(aligned[1], reference, atol=1e-12)
        np.testing.assert_allclose(rmsf_for_replica(trajectory), 0.0, atol=1e-12)

    def test_tokenizer_guard_rejects_more_than_L_tokens(self):
        import torch

        class DuplicatingTokenizer:
            def __call__(self, sequences, **kwargs):
                width = 2 * max(map(len, sequences))
                return {
                    "input_ids": torch.ones((len(sequences), width), dtype=torch.long),
                    "attention_mask": torch.ones((len(sequences), width), dtype=torch.long),
                }

        with self.assertRaisesRegex(ValueError, "exactly L tokens"):
            tokenize_linear_sequences(DuplicatingTokenizer(), ["ACDE"])

    def test_unversioned_linear_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache = directory / "scores.json"
            cache.write_text(json.dumps({"d1": [0.1]}))
            checkpoints = [directory / f"seed_{seed}.pth" for seed in (1, 2, 3)]
            with self.assertRaisesRegex(ValueError, "Refusing unversioned"):
                _validate_linear_cache(cache, directory / "missing.audit.json", checkpoints)

    def test_linear_cache_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache = directory / "scores.json"
            cache.write_text(json.dumps({"d1": [0.1]}))
            checkpoints = [directory / f"seed_{seed}.pth" for seed in (1, 2, 3)]
            for seed, checkpoint in enumerate(checkpoints, 1):
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
            audit = {
                "cache_schema_version": LINEAR_CACHE_SCHEMA_VERSION,
                "tokenizer_model": LINEAR_MODEL_NAME,
                "add_special_tokens": False,
                "sequence_input_mode": "single_sequence_once",
                "inference_implementation": LINEAR_INFERENCE_IMPLEMENTATION,
                "equivalence_test_tolerance": LINEAR_EQUIVALENCE_TOLERANCE,
                "backbone_equality_passed": True,
                "checkpoint_paths": [str(path) for path in checkpoints],
                "checkpoint_sha256": [_sha256_file(path) for path in checkpoints],
                "cache_sha256": _sha256_file(cache),
            }
            audit_path = directory / "scores.audit.json"
            audit_path.write_text(json.dumps(audit))

            checkpoints[1].write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                _validate_linear_cache(cache, audit_path, checkpoints)

    def test_nonpositive_bootstrap_count_is_rejected(self):
        with patch("sys.argv", ["benchmark_unified_mdcath.py", "--n-bootstrap", "0"]):
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                main()

    def test_force_linear_requires_infer_linear(self):
        with patch("sys.argv", ["benchmark_unified_mdcath.py", "--force-linear"]):
            with self.assertRaisesRegex(ValueError, "requires --infer-linear"):
                main()


class TestFrozenLinearEquivalence(unittest.TestCase):
    def test_real_checkpoints_match_shared_backbone_and_single_proteins(self):
        import torch

        checkpoints = [
            DEFAULT_RUNS_DIR / LINEAR_CONDITION / f"seed_{seed}" / "best_model.pth"
            for seed in (1, 2, 3)
        ]
        device = torch.device("cpu")
        model, tokenizer, heads, verification = load_verified_linear_model(
            checkpoints, device, local_files_only=True
        )
        self.assertTrue(verification["backbone_equality_passed"])

        sequences = ["ACDE", "MKTAYIA", "GGHPEPTIDE"]
        encoded = tokenize_linear_sequences(tokenizer, sequences)
        observed_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        self.assertEqual(observed_lengths, [len(sequence) for sequence in sequences])

        ordinary_by_checkpoint = ordinary_linear_checkpoint_probabilities(
            model, checkpoints, encoded, device
        ).cpu()
        self.assertEqual(ordinary_by_checkpoint.shape[0], 3)
        ordinary = ordinary_by_checkpoint.mean(dim=0)
        optimized = shared_backbone_linear_probabilities(
            model, heads, encoded, device
        ).cpu()

        max_difference = 0.0
        for index, sequence in enumerate(sequences):
            ordinary_vector = ordinary[index, :len(sequence)].numpy()
            optimized_vector = optimized[index, :len(sequence)].numpy()
            self.assertEqual(len(ordinary_vector), len(sequence))
            self.assertEqual(len(optimized_vector), len(sequence))
            difference = float(np.max(np.abs(ordinary_vector - optimized_vector)))
            max_difference = max(max_difference, difference)
            np.testing.assert_array_equal(
                ordinary_vector >= 0.5, optimized_vector >= 0.5
            )

            single_encoded = tokenize_linear_sequences(tokenizer, [sequence])
            single = shared_backbone_linear_probabilities(
                model, heads, single_encoded, device
            )[0, :len(sequence)].cpu().numpy()
            self.assertEqual(len(single), len(sequence))
            np.testing.assert_array_equal(single >= 0.5, optimized_vector >= 0.5)
            max_difference = max(
                max_difference, float(np.max(np.abs(single - optimized_vector)))
            )

        self.assertLessEqual(max_difference, LINEAR_EQUIVALENCE_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
