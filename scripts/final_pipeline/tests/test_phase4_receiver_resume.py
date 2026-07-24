import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_query_receivers import (
    AGGREGATE_CHECKPOINT_DIR,
    PROFILE_METRICS,
    acquire_output_lock,
    aggregate,
    completion_is_reusable,
    reusable_context_checkpoint,
)
from signed_band_analysis.audit_phase4_upgraded import (
    Audit,
    audit_receiver_output,
)


class Phase4ReceiverResumeTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        condition = "test_condition"
        protein = "protein_A"
        source = root / "source"
        output = root / "output"
        manifest_rows = []
        band_rows = []
        summary_rows = []
        residue_rows = []
        for split in ("train", "validation", "test"):
            band_id = f"{condition}__{split}__{protein}__band_1"
            profile = {
                "protein_length": np.asarray([5]),
                "band_id": np.asarray([band_id]),
                "sign": np.asarray([1]),
                "apex_index_0based": np.asarray([2]),
                "start_index_0based": np.asarray([2]),
                "end_index_0based_inclusive": np.asarray([2]),
                "source_seeds": np.asarray([1]),
            }
            for offset, metric in enumerate(PROFILE_METRICS):
                profile[metric] = np.asarray(
                    [[0.1, 0.2, 0.3, 0.4, 0.5 + offset]]
                )
            final = (
                source / "band_query_profiles" / condition / split
                / f"{protein}.npz"
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(final, **profile)
            seed_cache = (
                source / "per_seed_query_profile_cache" / condition
                / "seed_1" / split / f"{protein}.npz"
            )
            seed_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(seed_cache, seed=np.asarray([1]))
            manifest_rows.append({
                "condition": condition,
                "seed": 1,
                "split": split,
            })
            band_rows.append({
                "condition": condition,
                "split": split,
                "protein": protein,
                "band_id": band_id,
                "start_index_0based": 2,
                "end_index_0based_inclusive": 2,
            })
            summary_rows.append({
                "condition": condition,
                "split": split,
                "protein": protein,
                "eligible_start_index_0based": 1,
                "eligible_end_index_0based_exclusive": 4,
            })
            for index, amino_acid in enumerate("ACDEF"):
                residue_rows.append({
                    "split": split,
                    "protein": protein,
                    "residue_index_0based": index,
                    "amino_acid": amino_acid,
                    "q8": "H",
                    "neq": float(index),
                    "rsa": float(index),
                    "disorder": float(index),
                    "torsion_change_from_previous": float(index),
                })
        args = types.SimpleNamespace(
            receiver_cache_source_dir=str(source),
            pairwise_structure_csv=None,
            pairwise_structure_dir=None,
            mechanism_csv=None,
            seeds=[1],
            overwrite_cache=False,
            receiver_quantile=0.90,
            low_receiver_quantile=0.50,
            long_range_min_separation=1,
            minimum_inference_proteins=2,
            max_model_rows_per_class_per_protein=10,
            random_seed=123,
            max_iter=50,
            progress_every=1,
        )
        return (
            args,
            pd.DataFrame(manifest_rows),
            pd.DataFrame(band_rows),
            pd.DataFrame(summary_rows),
            pd.DataFrame(residue_rows),
            output,
        )

    def test_completed_protein_checkpoints_are_reused_after_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            args, manifest, bands, summary, residue, output = fixture
            first = aggregate(
                args, manifest, bands, summary, residue, output, "signature"
            )
            self.assertEqual(first["checkpoints_written"], 3)
            self.assertEqual(first["checkpoints_reused"], 0)
            (output / "parameters.json").write_text(json.dumps({
                "pairwise_structure_source": "fixture",
                "analysis_signature": "signature",
            }))
            (output / "extraction_audit.json").write_text("[]\n")
            (output / "receiver_complete.json").write_text(json.dumps({
                "schema": "esmfluc.receiver.analysis_complete.v1",
                "analysis_signature": "signature",
                "aggregate": first,
            }))
            self.assertIsNotNone(
                completion_is_reusable(output, "signature")
            )
            audit = Audit(20)
            audit_receiver_output(output, audit)
            self.assertEqual(audit.failures, [])

            with patch(
                "signed_band_analysis.analyze_signed_band_query_receivers."
                "receiver_rows_for_profile",
                side_effect=AssertionError("completed profile was recomputed"),
            ):
                resumed = aggregate(
                    args, manifest, bands, summary, residue, output, "signature"
                )

            self.assertEqual(resumed["checkpoints_written"], 0)
            self.assertEqual(resumed["checkpoints_reused"], 3)
            self.assertEqual(
                resumed["band_query_pairs"], first["band_query_pairs"]
            )

    def test_partial_checkpoint_without_marker_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = (
                Path(temporary) / AGGREGATE_CHECKPOINT_DIR
                / "condition" / "test" / "protein"
            )
            directory.mkdir(parents=True)
            (directory / "pairs.csv.gz").write_bytes(b"partial")
            marker = reusable_context_checkpoint(
                directory,
                "signature",
                ("condition", "test", "protein"),
            )
            self.assertIsNone(marker)

    def test_duplicate_runs_for_one_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = acquire_output_lock(output)
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "Another receiver run"
                ):
                    acquire_output_lock(output)
            finally:
                first.close()
            replacement = acquire_output_lock(output)
            replacement.close()


if __name__ == "__main__":
    unittest.main()
