#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:03:11 2025

Description: This script takes a fasta file and the best model check point.
It returns a JSON file with sequences in the fasta, their attention weights and neq preds,
using best model check point. 
If a CSV file for secondary structure prediction is given, the final JSON will also include ss_pred. 
Such CSV file can be obtained from: https://services.healthtech.dtu.dk/services/NetSurfP-3.0/ 

"""

import pandas as pd
import argparse
import sys
import os
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

# Ensure models.py (in the parent directory) is importable regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from esm.pretrained import ESM3_sm_open_v0
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    # ── Compatibility patch ───────────────────────────────────────────────────
    # Two problems:
    # 1. EsmSequenceTokenizer defines special tokens as read-only @property.
    #    transformers.__init__ calls setattr(self, 'mask_token', ...) → AttributeError.
    # 2. ESM3's internal _get_token calls self.__getattr__(token_name) to retrieve
    #    the stored token string.  This needs (a) a setter that actually stores the
    #    value as self._mask_token etc., and (b) __getattr__ to find it.
    #    Older transformers (< 4.40) don't define __getattr__ in the MRO at all.
    #
    # Fix:
    #   a) Replace each read-only property with a writable one whose setter stores
    #      the value via object.__setattr__(self, '_<name>', value).
    #   b) If no class in the MRO defines __getattr__, inject a simple one that
    #      retrieves those private attrs — compatible with both old and new transformers.
    _SPECIAL_TOK_NAMES = (
        'cls_token', 'eos_token', 'mask_token', 'pad_token',
        'unk_token', 'bos_token', 'sep_token',
    )

    def _make_token_setter(private_name):
        def setter(self, value):
            object.__setattr__(self, private_name, value)
        return setter

    for _tok_name in _SPECIAL_TOK_NAMES:
        _getter = None
        for _klass in EsmSequenceTokenizer.__mro__:
            _cls_attr = _klass.__dict__.get(_tok_name)
            if isinstance(_cls_attr, property):
                _getter = _cls_attr.fget
                break
        if _getter is not None:
            # Always patch on EsmSequenceTokenizer itself (pure-Python class);
            # avoids TypeError when a base class in the MRO is a C-extension type.
            setattr(EsmSequenceTokenizer, _tok_name, property(
                _getter,
                _make_token_setter('_' + _tok_name),
            ))

    # Only inject __getattr__ if the MRO doesn't already provide one.
    # (transformers >= 4.40 defines it in PreTrainedTokenizerBase; older versions don't.)
    _mro_has_getattr = any(
        '__getattr__' in klass.__dict__
        for klass in EsmSequenceTokenizer.__mro__
        if klass is not EsmSequenceTokenizer
    )
    if not _mro_has_getattr:
        def _esm3_compat_getattr(self, name):
            try:
                return object.__getattribute__(self, '_' + name)
            except AttributeError:
                pass
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        EsmSequenceTokenizer.__getattr__ = _esm3_compat_getattr
    # ─────────────────────────────────────────────────────────────────────────

    ESM3_AVAILABLE = True
except Exception as _esm3_import_err:
    print(f"[warn] ESM3 unavailable in get_attn.py: "
          f"{type(_esm3_import_err).__name__}: {_esm3_import_err}")
    ESM3_AVAILABLE = False


from models import (
    BiLSTMWithSelfAttentionModel, ESMLinearTokenClassifier,
    BiLSTMWithSelfAttentionRegressionModel, ESMLinearTokenRegressor
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── ESM3 compatibility wrapper ────────────────────────────────────────────────
# ESM3 (from esm.pretrained.ESM3_sm_open_v0) uses a different forward API than
# HuggingFace ESM2.  This wrapper makes it look like an ESM2 HF model so that
# BiLSTMWithSelfAttentionModel.forward() can call it identically.

class _ESM3Output:
    """Minimal stand-in for HuggingFace model output with .last_hidden_state."""
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyConfig:
    """Provides .hidden_size so BiLSTM __init__ can read embedding dim."""
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class ESM3Wrapper(torch.nn.Module):
    """
    Wraps ESM3_sm_open_v0 to expose the same interface as HuggingFace EsmModel:
      - .config.hidden_size
      - forward(input_ids, attention_mask) -> object with .last_hidden_state

    ESM3-sm hidden size is 1536.  The raw ESM3 forward uses 'sequence_tokens'
    and returns an ESMOutput whose per-residue embeddings are in
    .sequence_last_hidden_states rather than .last_hidden_state.
    """
    ESM3_HIDDEN = 1536  # ESM3-sm-open hidden dimension

    def __init__(self, esm3_model):
        super().__init__()
        self.esm3 = esm3_model
        self.config = _DummyConfig(self.ESM3_HIDDEN)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # kwargs absorbs HF-style arguments like output_attentions, output_hidden_states
        # that the BiLSTM model code may pass — ESM3 doesn't use them.
        out = self.esm3(sequence_tokens=input_ids)
        # ESM3 output field name changed across library versions; try both
        if hasattr(out, 'last_hidden_state'):
            h = out.last_hidden_state
        elif hasattr(out, 'sequence_last_hidden_states'):
            h = out.sequence_last_hidden_states
        elif hasattr(out, 'embeddings'):
            h = out.embeddings
        else:
            raise AttributeError(
                f"Cannot find hidden states in ESM3 output. Keys: {list(vars(out).keys())}")
        return _ESM3Output(h)

    def parameters(self, recurse=True):
        return self.esm3.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.esm3.named_parameters(prefix, recurse)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention weights for sequences in a FASTA file."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (e.g., best_model.pth).")
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="Path to the FASTA file containing sequences.")
    parser.add_argument("--architecture", type=str, default="bilstm_attention",
                        choices=["bilstm_attention", "esm_linear"],
                        help="Model architecture used for training.")
    parser.add_argument("--task_type", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Task type: classification or regression.")
    parser.add_argument("--num_outputs", type=int, default=1,
                        help="Number of output values for regression (default: 1).")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes for classification (default: 2).")
    parser.add_argument("--esm_model", type=str, default="esm2_t33_650M_UR50D",
                        help="ESM model name (e.g., esm2_t12_35M_UR50D, esm2_t33_650M_UR50D).")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden size for BiLSTM (default: 512).")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of BiLSTM layers (default: 3).")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3).")
    parser.add_argument("--bidirectional", type=int, default=1,
                        help="Use bidirectional LSTM (1=True, 0=False, default: 1).")
    parser.add_argument("--layer", type=int, default=-1,
                        help="ESM layer to extract attention from (for esm_linear). Use -1 for last layer.")
    parser.add_argument("--ss_csv", type=str, required=False,
                        help="(Optional) Path to the CSV file containing ss predictions (NetSurfP output).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output JSON file.")
    parser.add_argument(
        "--logit_contributions_output",
        type=str,
        default=None,
        help=("Optional compressed NPZ sidecar for exact flexible-minus-rigid "
              "logit-contribution matrices. Supported only for binary "
              "bilstm_attention classification."),
    )
    parser.add_argument("--is_esm3", action="store_true", default=False,
                        help="Use ESM3 backbone (esm3_sm_open_v0) instead of ESM2 (HuggingFace).")
    return parser.parse_args()


def parse_fasta_file(fasta_path):
    """
    Yields tuples of (sequence_id, sequence_string) from a FASTA file.
    """
    with open(fasta_path, 'r') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_lines)
                seq_id = line[1:]  # everything after ">"
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None and seq_lines:
            yield seq_id, "".join(seq_lines)


def exact_binary_margin_contributions(model, attention, value_vectors, logits):
    """Decompose the binary class-1 minus class-0 logit margin exactly.

    For query i and key/value j, the returned matrix is
    A_ij * V_j dot (w_1 - w_0). Its row sum plus the classifier bias
    difference reconstructs logits[..., 1] - logits[..., 0].
    """
    if model.fc.out_features != 2:
        raise ValueError(
            "Exact flexible-minus-rigid contributions require num_classes=2; "
            f"found {model.fc.out_features}."
        )
    weight_delta = model.fc.weight[1] - model.fc.weight[0]
    value_margin = torch.matmul(value_vectors, weight_delta)  # [B, key]
    contributions = attention * value_margin.unsqueeze(1)     # [B, query, key]
    bias_delta = model.fc.bias[1] - model.fc.bias[0]
    reconstructed = contributions.sum(dim=-1) + bias_delta
    expected = logits[..., 1] - logits[..., 0]
    max_abs_error = torch.max(torch.abs(reconstructed - expected)).item()
    torch.testing.assert_close(reconstructed, expected, rtol=1e-5, atol=1e-5)
    return {
        "matrix": contributions[0].detach().float().cpu().numpy(),
        "margin": expected[0].detach().float().cpu().numpy(),
        "bias": float(bias_delta.detach().float().cpu()),
        "max_abs_error": float(max_abs_error),
    }


def run_model_bilstm_attn(
    model,
    tokenizer,
    sequence,
    device,
    task_type="classification",
    return_logit_contributions=False,
):
    """Extract attention from BiLSTMWithSelfAttentionModel (custom attention layer)."""
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    # ESM3's EsmSequenceTokenizer returns 'sequence_tokens'; ESM2 HF returns 'input_ids'
    if "input_ids" in enc:
        input_ids = enc["input_ids"].to(device)
        attn_mask  = enc["attention_mask"].to(device)
    else:
        input_ids = enc["sequence_tokens"].to(device)
        attn_mask  = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    
    captured = {}
    hook = None
    if return_logit_contributions:
        if task_type != "classification":
            raise ValueError("Logit contribution export is supported only for classification.")
        hook = model.attention.value.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("value", output)
        )
    model.eval()
    try:
        with torch.no_grad():
            logits, feats, attn_weights_torch = model(input_ids, attn_mask, return_attention=True)
    finally:
        if hook is not None:
            hook.remove()
      
    attn_weights = attn_weights_torch[0].cpu().numpy()  # shape=(L,L)
    
    token_ids = input_ids[0].tolist()
    # EsmSequenceTokenizer (ESM3) has convert_ids_to_tokens; fall back to list of residues
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        tokens = list(sequence)  # plain AA characters as fallback
    
    if task_type == "classification":
        y_probs = torch.softmax(logits, dim=-1)
        y_preds = torch.argmax(y_probs, dim=-1)
        y_preds = y_preds.view(-1)
        class_probs = y_probs[0].cpu()
        flexible_scores = class_probs[:, 1] if class_probs.shape[-1] > 1 else class_probs[:, 0]
    else:  # regression
        # logits shape: [B, L, num_outputs] -> squeeze to [B, L] then flatten to [L]
        if logits.dim() == 3 and logits.size(-1) == 1:
            y_preds = logits.squeeze(-1).view(-1)  # [B, L, 1] -> [B, L] -> [L]
        else:
            y_preds = logits.view(-1)  # Fallback: flatten everything
        class_probs = None
        flexible_scores = y_preds.detach().cpu()
    
    contribution_payload = None
    if return_logit_contributions:
        if "value" not in captured:
            raise RuntimeError("Failed to capture the self-attention value projection.")
        contribution_payload = exact_binary_margin_contributions(
            model, attn_weights_torch, captured["value"], logits
        )

    return attn_weights, tokens, y_preds, flexible_scores, class_probs, contribution_payload


def run_model_esm_linear(model, tokenizer, sequence, device, layer_idx=-1, task_type="classification"):
    """Extract attention from ESMLinearTokenClassifier (ESM transformer attention)."""
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        logits, feats, all_attentions = model(input_ids, attn_mask, return_attn=True)
    
    # all_attentions is a tuple of (num_layers,) each with shape [B, num_heads, L, L]
    # Extract the specified layer and average over heads
    selected_layer_attn = all_attentions[layer_idx]  # [B, num_heads, L, L]
    attn_weights = selected_layer_attn[0].mean(dim=0).cpu().numpy()  # Average over heads -> [L, L]
    
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    if task_type == "classification":
        y_probs = torch.softmax(logits, dim=-1)
        y_preds = torch.argmax(y_probs, dim=-1)
        y_preds = y_preds.view(-1)
        class_probs = y_probs[0].cpu()
        flexible_scores = class_probs[:, 1] if class_probs.shape[-1] > 1 else class_probs[:, 0]
    else:  # regression
        # logits shape: [B, L, num_outputs] -> squeeze to [B, L] then flatten to [L]
        if logits.dim() == 3 and logits.size(-1) == 1:
            y_preds = logits.squeeze(-1).view(-1)  # [B, L, 1] -> [B, L] -> [L]
        else:
            y_preds = logits.view(-1)  # Fallback: flatten everything
        class_probs = None
        flexible_scores = y_preds.detach().cpu()
    
    return attn_weights, tokens, y_preds, flexible_scores, class_probs, None

def infer_bilstm_params(checkpoint):
    """Infer hidden_size and num_layers for a BiLSTM model from its checkpoint keys."""
    hidden_size = None
    num_layers = 0
    for key, val in checkpoint.items():
        # e.g. 'lstm.weight_ih_l0', 'lstm.weight_ih_l3' (not '_reverse')
        if key.startswith('lstm.weight_ih_l') and '_reverse' not in key:
            layer_idx = int(key.split('lstm.weight_ih_l')[1])
            num_layers = max(num_layers, layer_idx + 1)
            if layer_idx == 0:
                # shape: [4 * hidden_size_per_direction, input_size]
                hidden_size = val.shape[0] // 4
    return hidden_size, num_layers if num_layers > 0 else None


def parse_nsp3_csv(df):
    """
    Returns a dict mapping { seq_id -> list_of_SS }, 
    where seq_id is the FASTA header without '>',
    and list_of_SS is a list of 'C','H','E' for each residue.
    """

    ss_map = {}

    for row in df.itertuples(index=False):
        raw_id = row[0]    # first column is 'id'
        q3_label = row[5]  # fourth column is 'q3'
        seq_id = raw_id.lstrip(">")
        if seq_id not in ss_map:
            ss_map[seq_id] = []
        ss_map[seq_id].append(q3_label)

    return ss_map


def main():
    
    args = parse_args()

    if args.logit_contributions_output:
        if args.architecture != "bilstm_attention":
            raise ValueError(
                "--logit_contributions_output is only valid for bilstm_attention. "
                "A selected ESM backbone attention layer is not an exact additive "
                "decomposition of the final classifier logit."
            )
        if args.task_type != "classification" or args.num_classes != 2:
            raise ValueError(
                "--logit_contributions_output requires binary classification "
                "(--task_type classification --num_classes 2)."
            )

    # Load ESM model and tokenizer — ESM3 uses a different API from ESM2
    if args.is_esm3:
        if not ESM3_AVAILABLE:
            raise ImportError(
                "ESM3 requires the 'esm' library from EvolutionaryScale.\n"
                "Install: pip install esm\n"
                "Then log in: huggingface-cli login  (and accept the model licence at "
                "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)"
            )
        raw_esm3 = ESM3_sm_open_v0(device)
        embedding_model = ESM3Wrapper(raw_esm3)
        tokenizer = EsmSequenceTokenizer()
        print("Loaded ESM3 model: esm3_sm_open_v0 (wrapped for HF-compatible API)")
    else:
        esm_model_name = f"facebook/{args.esm_model}"
        embedding_model = EsmModel.from_pretrained(esm_model_name)
        tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        print(f"Loaded ESM2 model: {esm_model_name}")

    embedding_model.to(device)
    print(f"Task type: {args.task_type}")
    print(f"Architecture: {args.architecture}")

    # Load checkpoint early so we can infer architecture params from it
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Auto-infer hidden_size / num_layers from the checkpoint so the user
    # doesn't have to remember the exact training flags.
    if args.architecture == 'bilstm_attention':
        ckpt_to_inspect = checkpoint if not args.is_esm3 else {
            k: v for k, v in checkpoint.items()
            if not k.startswith('embedding_model.')
        }
        inferred_hidden, inferred_layers = infer_bilstm_params(ckpt_to_inspect)
        if inferred_hidden is not None and inferred_hidden != args.hidden_size:
            print(f"[info] Auto-detected hidden_size={inferred_hidden} from checkpoint "
                  f"(overriding CLI value {args.hidden_size})")
            args.hidden_size = inferred_hidden
        if inferred_layers is not None and inferred_layers != args.num_layers:
            print(f"[info] Auto-detected num_layers={inferred_layers} from checkpoint "
                  f"(overriding CLI value {args.num_layers})")
            args.num_layers = inferred_layers

    # Build model based on architecture and task type
    if args.task_type == "classification":
        if args.architecture == "bilstm_attention":
            model = BiLSTMWithSelfAttentionModel(
                embedding_model=embedding_model,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_classes=args.num_classes,
                dropout=args.dropout,
                bidirectional=args.bidirectional
            )
            run_fn = lambda m, t, s, d: run_model_bilstm_attn(
                m, t, s, d,
                task_type="classification",
                return_logit_contributions=bool(args.logit_contributions_output),
            )
        elif args.architecture == "esm_linear":
            model = ESMLinearTokenClassifier(
                embedding_model=embedding_model,
                num_classes=args.num_classes
            )
            run_fn = lambda m, t, s, d: run_model_esm_linear(m, t, s, d, layer_idx=args.layer, task_type="classification")
        else:
            raise ValueError(f"Unsupported architecture: {args.architecture}")
    
    elif args.task_type == "regression":
        if args.architecture == "bilstm_attention":
            model = BiLSTMWithSelfAttentionRegressionModel(
                embedding_model=embedding_model,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_outputs=args.num_outputs,
                dropout=args.dropout,
                bidirectional=args.bidirectional
            )
            run_fn = lambda m, t, s, d: run_model_bilstm_attn(
                m, t, s, d, task_type="regression"
            )
        elif args.architecture == "esm_linear":
            model = ESMLinearTokenRegressor(
                embedding_model=embedding_model,
                num_outputs=args.num_outputs
            )
            run_fn = lambda m, t, s, d: run_model_esm_linear(m, t, s, d, layer_idx=args.layer, task_type="regression")
        else:
            raise ValueError(f"Unsupported architecture: {args.architecture}")
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
    
    model.to(device)

    # checkpoint was already loaded above for param inference.
    # Load strictly so attention extraction cannot silently mix checkpointed
    # layers with pretrained/randomly initialized weights.
    model.load_state_dict(checkpoint, strict=True)
    print(f"Loaded checkpoint from {args.checkpoint}")

    ss_map = {}
    ss_available = args.ss_csv is not None
    if ss_available:
        nsp3_df = pd.read_csv(args.ss_csv, index_col=False)
        ss_map = parse_nsp3_csv(nsp3_df)
        print(f"Parsed NetSurfP CSV: found SS for {len(ss_map)} sequences")
    
    rows = []
    contribution_arrays = {}
    contribution_index = []
    for sequence_index, (seq_id, seq_str) in enumerate(parse_fasta_file(args.fasta_file)):
        ss_list = ss_map.get(seq_id, None) if ss_available else None
        if ss_available and ss_list is None:
            print(f"Warning: no SS predictions for {seq_id}")
        if ss_available and ss_list is not None and len(ss_list) != len(seq_str):
            print(f"Warning: length mismatch for {seq_id}, skipping.")
            continue

        attention_weights, tokens, neq_preds, flexible_scores, class_probs, contribution_payload = run_fn(
            model, tokenizer, seq_str, device
        )
        if attention_weights.shape != (len(seq_str), len(seq_str)):
            raise ValueError(
                f"{seq_id}: attention shape {attention_weights.shape} does not match "
                f"sequence length {len(seq_str)}."
            )
        if len(neq_preds) != len(seq_str) or len(flexible_scores) != len(seq_str):
            raise ValueError(
                f"{seq_id}: prediction length mismatch: sequence={len(seq_str)}, "
                f"predictions={len(neq_preds)}, scores={len(flexible_scores)}."
            )
        if class_probs is not None and tuple(class_probs.shape) != (
            len(seq_str), args.num_classes
        ):
            raise ValueError(
                f"{seq_id}: class probability shape {tuple(class_probs.shape)} does not "
                f"match ({len(seq_str)}, {args.num_classes})."
            )
        print(f"{seq_id:15s}  "
              f"seq_len = {len(seq_str):3d}  "
              f"tokens = {len(tokens):3d}  "
              f"attn_shape = {attention_weights.shape}")
        attn_list = attention_weights.tolist()
        neq_list = neq_preds.cpu().numpy().tolist()

        row_dict = {
            "name": seq_id,
            "sequence": seq_str,
            "attention_weights": attn_list,
            "neq_preds": neq_list,
            "flexible_scores": flexible_scores.numpy().tolist()
        }
        if class_probs is not None:
            row_dict["class_probs"] = class_probs.numpy().tolist()
            row_dict["class_index_definition"] = {
                "0": "rigid (Neq <= 1.0)",
                "1": "flexible (Neq > 1.0)",
            }
            row_dict["flexible_score_definition"] = "P(class 1: Neq > 1.0)"

        if contribution_payload is not None:
            key = f"protein_{sequence_index:04d}"
            matrix = contribution_payload["matrix"]
            if matrix.shape != (len(seq_str), len(seq_str)):
                raise ValueError(
                    f"{seq_id}: contribution shape {matrix.shape} does not match "
                    f"sequence length {len(seq_str)}."
                )
            contribution_arrays[key] = matrix
            contribution_index.append((key, seq_id))
            row_dict.update({
                "flex_minus_rigid_logit_contribution_file": os.path.basename(
                    args.logit_contributions_output
                ),
                "flex_minus_rigid_logit_contribution_key": key,
                "flex_minus_rigid_logit_margin": contribution_payload["margin"].tolist(),
                "flex_minus_rigid_logit_margin_bias": contribution_payload["bias"],
                "flex_minus_rigid_logit_reconstruction_max_abs_error": (
                    contribution_payload["max_abs_error"]
                ),
                "flex_minus_rigid_logit_contribution_definition": (
                    "A[query,key] * dot(V[key], fc.weight[1] - fc.weight[0]); "
                    "row_sum + (fc.bias[1] - fc.bias[0]) equals "
                    "logit_flexible - logit_rigid"
                ),
            })

        if ss_available and ss_list is not None:
            row_dict["ss_pred"] = ss_list

        rows.append(row_dict)

        
    # Preserve every output placed in row_dict. Restricting the DataFrame to a
    # legacy column list silently discarded flexible_scores and class_probs.
    final_df = pd.DataFrame(rows)
    print(f"Constructed final DF with {len(final_df)} rows")

    final_df.to_json(args.output, orient="records", indent=2)
    print(f"Saved final JSON to {args.output}")

    if args.logit_contributions_output:
        if len(contribution_arrays) != len(rows):
            raise RuntimeError(
                f"Expected one contribution matrix per sequence; got "
                f"{len(contribution_arrays)} for {len(rows)} sequences."
            )
        contribution_path = os.path.abspath(args.logit_contributions_output)
        os.makedirs(os.path.dirname(contribution_path), exist_ok=True)
        names = np.asarray([name for _, name in contribution_index], dtype=str)
        keys = np.asarray([key for key, _ in contribution_index], dtype=str)
        np.savez_compressed(
            contribution_path,
            **contribution_arrays,
            __protein_names__=names,
            __matrix_keys__=keys,
            __definition__=np.asarray(
                "class 1 (flexible) minus class 0 (rigid) exact final-head logit contribution",
                dtype=str,
            ),
        )
        print(
            f"Saved {len(contribution_arrays)} exact flex-minus-rigid contribution "
            f"matrices to {contribution_path}"
        )
        

if __name__ == "__main__":
    main()
