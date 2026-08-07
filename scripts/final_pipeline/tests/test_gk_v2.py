import sys
import unittest
from pathlib import Path

import numpy as np


GK_DIR = Path(__file__).resolve().parents[1] / "gk_mutants"
sys.path.insert(0, str(GK_DIR))

from analyze_gk_v2 import interval_iou  # noqa: E402
from gk_v2_common import bh_adjust, reconstruct_evidence  # noqa: E402


class TestGKV2Numerics(unittest.TestCase):
    def test_exact_contribution_reconstruction(self):
        rng = np.random.default_rng(7)
        attention = rng.random((12, 12))
        attention /= attention.sum(axis=1, keepdims=True)
        evidence = rng.normal(size=12)
        contribution = attention * evidence[None, :]
        recovered, error = reconstruct_evidence(contribution, attention)
        np.testing.assert_allclose(recovered, evidence, atol=1e-12)
        self.assertLess(error, 1e-12)

    def test_interval_iou_uses_inclusive_coordinates(self):
        self.assertEqual(interval_iou(3, 5, 3, 5), 1.0)
        self.assertAlmostEqual(interval_iou(3, 5, 5, 7), 1 / 5)
        self.assertEqual(interval_iou(3, 5, 6, 8), 0.0)

    def test_bh_adjustment(self):
        adjusted = bh_adjust([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 0.20])


if __name__ == "__main__":
    unittest.main()
