import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from esmfluc.visualization.attention_source_paths import resolve_existing_path, resolve_manifest_path
from esmfluc.visualization.plot_attention_source_grid import (
    averaged_matrix,
    common_proteins,
    condition_names,
    load_condition_records,
)


class TestAttentionSourcePaths(unittest.TestCase):
    def test_resolution_order_and_manifest_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            pipeline_dir = root / "pipeline"
            result_root.mkdir()
            pipeline_dir.mkdir()
            expected = result_root / "attention.json"
            expected.write_text("[]")

            self.assertEqual(
                resolve_existing_path("attention.json", result_root, pipeline_dir),
                expected,
            )
            self.assertEqual(
                resolve_manifest_path(result_root),
                (result_root / "manifest.tsv").resolve(),
            )

    def test_retained_plot_chain_loads_without_legacy_package(self):
        with tempfile.TemporaryDirectory() as directory:
            result_root = Path(directory)
            record = [{
                "name": "protein_a",
                "sequence": "AC",
                "attention_weights": [[0.75, 0.25], [0.20, 0.80]],
            }]
            rows = []
            for condition in condition_names():
                path = result_root / f"{condition}.json"
                path.write_text(json.dumps(record))
                rows.append({"condition": condition, "seed": 1, "attention_json": path.name})

            grouped = load_condition_records(
                pd.DataFrame(rows),
                result_root,
                result_root,
                condition_names(),
                "average",
                None,
            )
            self.assertEqual(common_proteins(grouped), ["protein_a"])
            matrix, sequence, seeds = averaged_matrix(
                grouped, condition_names()[0], "protein_a"
            )
            self.assertEqual(matrix.shape, (2, 2))
            self.assertEqual(sequence, "AC")
            self.assertEqual(seeds, [1])


if __name__ == "__main__":
    unittest.main()
