"""Offline checks for helpers retained from earlier workflows."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from esmfluc.datasets.dynamine import parse_fasta, _b2b_submit, _b2b_poll
from esmfluc.analysis.bands.contribution_utils import reconstruct_evidence


class RetainedHelperTests(unittest.TestCase):
    def test_evidence_factorization(self):
        attention = np.array([[0.2, 0.8], [0.6, 0.4]])
        evidence = np.array([-2.0, 3.0])
        recovered, error = reconstruct_evidence(attention * evidence, attention)
        np.testing.assert_allclose(recovered, evidence)
        self.assertLess(error, 1e-12)

    def test_duplicate_fasta_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.fasta'
            path.write_text('>one\nAC\n>one\nDE\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                parse_fasta(path)

    def test_submit_and_completed_queue(self):
        session = Mock()
        session.post.return_value.json.return_value = {'hash_id': 'example'}
        self.assertEqual(_b2b_submit(session, {'one': 'AC'}, 'session-id'), 'example')
        queue, result = Mock(), Mock()
        queue.status_code = 303
        result.json.return_value = {'results': [{'proteinID': 'one', 'backbone': [0.1, 0.2]}]}
        session.get.side_effect = [queue, result]
        self.assertEqual(_b2b_poll(session, 'example', 0, 1)[0]['proteinID'], 'one')
