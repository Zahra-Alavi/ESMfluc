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
import torch
from transformers import EsmModel, EsmTokenizer

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
        for _klass in EsmSequenceTokenizer.__mro__:
            _cls_attr = _klass.__dict__.get(_tok_name)
            if isinstance(_cls_attr, property) and _cls_attr.fset is None:
                setattr(_klass, _tok_name, property(
                    _cls_attr.fget,
                    _make_token_setter('_' + _tok_name),
                    _cls_attr.fdel,
                    _cls_attr.__doc__,
                ))
                break  # patched on the defining class; stop walking MRO

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
except ImportError:
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


def run_model_bilstm_attn(model, tokenizer, sequence, device, task_type="classification"):
    """Extract attention from BiLSTMWithSelfAttentionModel (custom attention layer)."""
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    # ESM3's EsmSequenceTokenizer returns 'sequence_tokens'; ESM2 HF returns 'input_ids'
    if "input_ids" in enc:
        input_ids = enc["input_ids"].to(device)
        attn_mask  = enc["attention_mask"].to(device)
    else:
        input_ids = enc["sequence_tokens"].to(device)
        attn_mask  = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        logits, feats, attn_weights_torch = model(input_ids, attn_mask, return_attention=True)
      
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
    else:  # regression
        # logits shape: [B, L, num_outputs] -> squeeze to [B, L] then flatten to [L]
        if logits.dim() == 3 and logits.size(-1) == 1:
            y_preds = logits.squeeze(-1).view(-1)  # [B, L, 1] -> [B, L] -> [L]
        else:
            y_preds = logits.view(-1)  # Fallback: flatten everything
    
    return attn_weights, tokens, y_preds


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
    else:  # regression
        # logits shape: [B, L, num_outputs] -> squeeze to [B, L] then flatten to [L]
        if logits.dim() == 3 and logits.size(-1) == 1:
            y_preds = logits.squeeze(-1).view(-1)  # [B, L, 1] -> [B, L] -> [L]
        else:
            y_preds = logits.view(-1)  # Fallback: flatten everything
    
    return attn_weights, tokens, y_preds

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
            run_fn = lambda m, t, s, d: run_model_bilstm_attn(m, t, s, d, task_type="classification")
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
            run_fn = lambda m, t, s, d: run_model_bilstm_attn(m, t, s, d, task_type="regression")
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

    # checkpoint was already loaded above for param inference
    # When using ESM3, the backbone (embedding_model.*) is already loaded fresh via
    # ESM3_sm_open_v0 + ESM3Wrapper.  The checkpoint may have been saved with the raw
    # ESM3 directly under 'embedding_model.*' (no '.esm3.' intermediary), so those keys
    # won't match the wrapped model.  Since the backbone was frozen during training its
    # weights haven't changed — we only need to restore the BiLSTM + attention head.
    if args.is_esm3:
        head_ckpt = {k: v for k, v in checkpoint.items()
                     if not k.startswith('embedding_model.')}
        missing, unexpected = model.load_state_dict(head_ckpt, strict=False)
        # 'missing' will be the frozen ESM3 backbone keys (already loaded) — expected.
        # 'unexpected' should be empty if filtering worked correctly.
        if unexpected:
            print(f"[warn] unexpected keys in checkpoint after filtering: {unexpected[:5]}")
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    ss_map = {}
    ss_available = args.ss_csv is not None
    if ss_available:
        nsp3_df = pd.read_csv(args.ss_csv, index_col=False)
        ss_map = parse_nsp3_csv(nsp3_df)
        print(f"Parsed NetSurfP CSV: found SS for {len(ss_map)} sequences")
    
    rows = []
    for seq_id, seq_str in parse_fasta_file(args.fasta_file):
        ss_list = ss_map.get(seq_id, None) if ss_available else None
        if ss_available and ss_list is None:
            print(f"Warning: no SS predictions for {seq_id}")
        if ss_available and ss_list is not None and len(ss_list) != len(seq_str):
            print(f"Warning: length mismatch for {seq_id}, skipping.")
            continue

        attention_weights, tokens, neq_preds = run_fn(model, tokenizer, seq_str, device)
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
            "neq_preds": neq_list
        }

        if ss_available and ss_list is not None:
            row_dict["ss_pred"] = ss_list

        rows.append(row_dict)

        
    columns = ["name", "sequence", "attention_weights", "neq_preds"]
    if ss_available:
        columns.append("ss_pred")

    final_df = pd.DataFrame(rows, columns=columns)
    print(f"Constructed final DF with {len(final_df)} rows")

    final_df.to_json(f"{args.output}.json", orient="records", indent=2)
    print(f"Saved final JSON to {args.output}.json")
        

if __name__ == "__main__":
    main()