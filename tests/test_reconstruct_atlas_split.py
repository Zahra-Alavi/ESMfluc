"""Fixed-membership reconstruction rejects altered sequences and missing targets."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from esmfluc.datasets.reconstruct_atlas_split import reconstruct


class AtlasReconstructionTests(unittest.TestCase):
    def test_frozen_membership_and_mean_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'split.csv'
            pd.DataFrame([dict(name='example_A', split='test', length=3,
                sequence_sha256=hashlib.sha256(b'ACD').hexdigest())]).to_csv(manifest,index=False)
            table = root/'example_A_Neq.tsv'
            table.write_text('seq\tNeq_R1\tNeq_R2\tNeq_R3\nA\t1\t1\t1\nC\t1\t2\t3\nD\t2\t2\t2\n')
            audit = reconstruct(manifest,root,root/'out')
            self.assertEqual(audit['split_counts']['test']['proteins'],1)
            self.assertEqual(pd.read_csv(root/'out/test_grouped_v1.csv').neq.iloc[0], '[1.0, 2.0, 2.0]')
            table.write_text(table.read_text().replace('A\t1','G\t1'))
            with self.assertRaisesRegex(ValueError,'sequence differs'):
                reconstruct(manifest,root,root/'bad')
