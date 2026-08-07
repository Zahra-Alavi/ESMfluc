import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_vs_baselines import (
    compute_metrics,
    load_neq_labels,
    paired_bootstrap_auroc_difference,
    validate_score_table,
)


class TestBenchmarkValidation(unittest.TestCase):
    def test_neq_length_mismatch_fails_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame({
                "name": ["protein"], "sequence": ["ACD"], "neq": ["[1, 2]"],
            }).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "sequence length"):
                load_neq_labels(path, 1.0)

    def test_score_table_requires_exact_residue_keys(self):
        scores = pd.DataFrame({
            "name": ["a", "a"], "res_idx": [1, 3], "score": [0.2, 0.4],
        })
        with self.assertRaisesRegex(ValueError, "residue keys"):
            validate_score_table(scores, {"a": 3}, "score", "test")

    def test_metrics_are_threshold_independent(self):
        metrics = compute_metrics(
            np.array([0, 0, 0, 1, 1, 1]),
            np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
            np.array([1.0, 1.0, 1.0, 2.0, 2.5, 3.0]),
        )
        self.assertEqual(metrics["AUROC"], 1.0)
        self.assertNotIn("F1", metrics)
        self.assertNotIn("MCC", metrics)

    def test_paired_protein_bootstrap_preserves_pairing(self):
        names = [f"p{i}" for i in range(20)]
        dynamine = pd.Series(np.linspace(0.4, 0.6, 20), index=names)
        esm = dynamine + 0.2
        result = paired_bootstrap_auroc_difference(
            esm, dynamine, n_bootstrap=500, random_seed=7
        )
        self.assertAlmostEqual(result["mean_delta_AUROC"], 0.2)
        self.assertAlmostEqual(result["mean_delta_AUROC_CI95_lo"], 0.2)
        self.assertAlmostEqual(result["mean_delta_AUROC_CI95_hi"], 0.2)
        self.assertEqual(result["proteins_ESMfluc_better"], 20)


if __name__ == "__main__":
    unittest.main()
