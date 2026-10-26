"""Offline tests for the mdCATH cluster and temperature splits."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from esmfluc.datasets.split_mdcath_by_temperature import split_by_temperature
from esmfluc.datasets.split_mdcath_clusters import split_by_clusters


def table(domains):
    return pd.DataFrame({
        "domain": domains,
        "sequence": ["ACDE"] * len(domains),
        "neq_320": [str([1.0, 1.5, 2.0, 1.0])] * len(domains),
        "neq_348": [str([1.0, 1.0, 1.2, 1.0])] * len(domains),
    })


class MdcathSplitTests(unittest.TestCase):
    def test_clusters_stay_together_and_split_is_seeded(self):
        domains = [f"d{i}" for i in range(20)]
        clusters = {d: f"c{i // 2}" for i, d in enumerate(domains)}
        train, test = split_by_clusters(table(domains), clusters, test_size=0.2, random_state=42)
        self.assertEqual(len(train) + len(test), 20)
        self.assertFalse({clusters[d] for d in train.domain} & {clusters[d] for d in test.domain})
        again = split_by_clusters(table(domains), clusters, test_size=0.2, random_state=42)[1]
        self.assertEqual(list(test.domain), list(again.domain))

    def test_temperature_files_keep_domain_sequence_and_neq(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table(["a", "b"]).to_csv(root / "train.csv", index=False)
            table(["c"]).to_csv(root / "test.csv", index=False)
            split_by_temperature(root / "train.csv", root / "test.csv", root / "out")
            test_320 = pd.read_csv(root / "out" / "test_320K.csv")
            self.assertEqual(list(test_320.columns), ["domain", "sequence", "neq"])
            self.assertEqual(test_320.domain.tolist(), ["c"])
            self.assertEqual(test_320.neq[0], str([1.0, 1.5, 2.0, 1.0]))
            self.assertTrue((root / "out" / "train_348K.csv").is_file())


if __name__ == "__main__":
    unittest.main()
