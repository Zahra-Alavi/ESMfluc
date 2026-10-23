"""Published condition settings and padding safeguards."""
import contextlib
import io
import unittest
import torch
from esmfluc.training.arguments import CONDITIONS, parse_arguments
from esmfluc.training.data_utils import create_classification_func, collate_fn_sequence


class TrainingConditionTests(unittest.TestCase):
    def test_seven_conditions(self):
        self.assertEqual(len(CONDITIONS), 7)
        for condition, expected in CONDITIONS.items():
            args = parse_arguments().parse_args(['--condition', condition])
            self.assertEqual((args.esm_model, args.architecture, args.freeze_all_backbone, args.freeze_layers), expected)
            self.assertEqual((args.hidden_size, args.num_layers, args.num_classes), (512, 3, 2))
            self.assertEqual((args.batch_size, args.epochs, args.patience, args.seed), (2, 80, 5, 1))
            self.assertTrue(args.mixed_precision)

    def test_reject_conflicting_condition_and_excess_epochs(self):
        for flags in (['--epochs', '81'], ['--condition', 'esm2_frozen_linear', '--freeze_layers', '0-4']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments().parse_args(flags)

    def test_compatible_workflow_flags(self):
        args = parse_arguments().parse_args(['--esm_model', 'esm3_sm_open_v1', '--architecture',
            'bilstm_attention', '--freeze_layers', '0-19', '--mixed_precision', '--seed', '3'])
        self.assertEqual(args.condition, 'esm3_top28_bilstm_attn')

    def test_pb_boundary(self):
        classify = create_classification_func(2, [1.0])
        self.assertEqual([classify(x) for x in [1.0, 1.001, 2.0]], [0, 1, 1])

    def test_padding_remains_excluded(self):
        from types import SimpleNamespace
        batch = [dict(input_ids=torch.tensor([1, 2]), attention_mask=torch.ones(2, dtype=torch.long), labels=torch.tensor([0, 1])),
                 dict(input_ids=torch.tensor([1]), attention_mask=torch.ones(1, dtype=torch.long), labels=torch.tensor([1]))]
        actual = collate_fn_sequence(batch, SimpleNamespace(pad_token_id=1))
        self.assertEqual(actual['labels'][1, 1].item(), -1)
        self.assertEqual(actual['attention_mask'][1, 1].item(), 0)
