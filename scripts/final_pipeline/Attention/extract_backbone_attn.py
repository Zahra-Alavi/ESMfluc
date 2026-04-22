#!/usr/bin/env python3
"""
extract_backbone_attn.py

Extract the backbone (ESM2 or ESM3) last-layer attention for each sequence
in a FASTA file, averaged over all attention heads, and save as a JSON that
is format-compatible with the bilstm_attn.json produced by get_attn.py.

For ESM2  : uses HuggingFace output_attentions=True → last layer → avg heads.
For ESM3  : registers a forward hook on the last transformer block's self-
            attention sub-module to capture post-softmax weights.
            Falls back to a cosine-similarity proxy if the hook does not return
            a [H, L, L] tensor (labelled "proxy" in the output).

The checkpoint is always used so that unfrozen (fine-tuned) backbone weights
are correctly reflected.  For frozen experiments the backbone weights in the
checkpoint are identical to the pretrained weights, so there is no extra cost.

Usage
-----
python extract_backbone_attn.py \
    --checkpoint  ../results/esm2_binary_frozen/best_model.pth \
    --fasta_file  ../../../data/test_data_sequences.fasta \
    --esm_model   esm2_t33_650M_UR50D \
    --output      ../results/esm2_binary_frozen/backbone_attn.json

For ESM3 replace --esm_model with --is_esm3.
"""

import sys
import os
import argparse
import json
import math

import numpy as np
import torch

# ── Add parent directory so we can import ESM3 compatibility helpers ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import EsmModel, EsmTokenizer

# ESM3 optional
try:
    from esm.pretrained import ESM3_sm_open_v0
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    # ── Same compatibility patch as in get_attn.py ───────────────────────────
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
            setattr(EsmSequenceTokenizer, _tok_name, property(
                _getter,
                _make_token_setter('_' + _tok_name),
            ))

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
    print(f"[warn] ESM3 unavailable in extract_backbone_attn.py: "
          f"{type(_esm3_import_err).__name__}: {_esm3_import_err}")
    ESM3_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_fasta(path):
    """Yield (seq_id, sequence) pairs from a FASTA file."""
    seq_id, lines = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(lines)
                seq_id = line[1:].split()[0]
                lines = []
            else:
                lines.append(line)
    if seq_id is not None:
        yield seq_id, "".join(lines)


def attn_to_list(attn_2d):
    """Convert a [L, L] numpy/tensor to a JSON-serialisable list of lists."""
    if torch.is_tensor(attn_2d):
        attn_2d = attn_2d.float().cpu().numpy()
    return attn_2d.tolist()


# ── ESM2 backbone attention ────────────────────────────────────────────────────

def build_esm2_from_checkpoint(checkpoint, esm_model_name, device):
    """
    Build an EsmModel and optionally load fine-tuned weights from a checkpoint.
    Falls back to pretrained weights if the checkpoint has no backbone keys.
    """
    raw_ckpt = checkpoint
    # Filter only the embedding_model.* keys and strip the prefix
    bb_state = {
        k[len("embedding_model."):]: v
        for k, v in raw_ckpt.items()
        if k.startswith("embedding_model.")
    }

    model = EsmModel.from_pretrained(f"facebook/{esm_model_name}")
    if bb_state:
        missing, unexpected = model.load_state_dict(bb_state, strict=False)
        if missing:
            print(f"  [backbone] {len(missing)} keys missing from checkpoint "
                  f"(pretrained weights kept for those).")
        print(f"  [backbone] Loaded {len(bb_state)} ESM2 keys from checkpoint.")
    else:
        print("  [backbone] No embedding_model keys in checkpoint; using pretrained ESM2.")
    model.to(device).eval()
    return model


def extract_esm2_attn(esm2_model, tokenizer, seq, device):
    """
    Run one sequence through ESM2 with output_attentions=True.
    Returns a [L, L] numpy array (last layer, averaged over heads).
    L is the sequence length (special tokens removed).
    """
    enc = tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=True)
    input_ids      = enc["input_ids"].to(device)          # [1, L+2]  (CLS + seq + EOS)
    attention_mask = enc["attention_mask"].to(device)     # [1, L+2]

    with torch.no_grad():
        out = esm2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
    # out.attentions: tuple of (num_layers,) × [1, num_heads, L+2, L+2]
    last_layer = out.attentions[-1][0]          # [H, L+2, L+2]
    avg_heads  = last_layer.mean(dim=0)          # [L+2, L+2]

    # Strip CLS and EOS tokens (first and last position)
    L = len(seq)
    core = avg_heads[1:L+1, 1:L+1]              # [L, L]

    # Row-normalise so each row sums to 1 (CLS removal breaks softmax sum)
    row_sums = core.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    core = core / row_sums

    return core.cpu().numpy()


# ── ESM3 backbone attention ────────────────────────────────────────────────────

_ESM3_HOOK_BUFFER = []


def _make_esm3_hook():
    """Return a fresh forward hook that captures attention weights."""
    def hook_fn(module, inputs, output):
        # output may be:
        #   (context_tensor,)          – if the module only returns context
        #   (context_tensor, weights)  – if weights are also returned
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            w = output[1]
            if w is not None and w.dim() == 4:       # [B, H, L, L]
                _ESM3_HOOK_BUFFER.append(w.detach().cpu())
    return hook_fn


def _find_last_attn_module(model):
    """
    Walk all named modules and return the last one whose name contains 'attn'
    and which has learnable parameters (i.e. is a real attention block,
    not just a dropout or activation).
    """
    candidates = [
        (name, mod)
        for name, mod in model.named_modules()
        if 'attn' in name.lower()
        and sum(1 for _ in mod.parameters()) > 0
    ]
    return candidates[-1] if candidates else (None, None)


def build_esm3_from_checkpoint(checkpoint, device):
    """
    Load the raw ESM3 model and inject fine-tuned backbone weights from checkpoint.
    Returns the raw ESM3 model (not wrapped).
    """
    raw_esm3 = ESM3_sm_open_v0(device)

    # Keys for ESM3 backbone are under "embedding_model.esm3.*"
    prefix = "embedding_model.esm3."
    bb_state = {
        k[len(prefix):]: v
        for k, v in checkpoint.items()
        if k.startswith(prefix)
    }
    if bb_state:
        missing, unexpected = raw_esm3.load_state_dict(bb_state, strict=False)
        print(f"  [backbone] Loaded {len(bb_state)} ESM3 keys from checkpoint.")
        if missing:
            print(f"  [backbone] {len(missing)} ESM3 keys not in checkpoint "
                  "(pretrained kept).")
    else:
        # Try alternative prefix (wrapped differently)
        prefix2 = "embedding_model."
        bb_state2 = {
            k[len(prefix2):]: v
            for k, v in checkpoint.items()
            if k.startswith(prefix2)
        }
        if bb_state2:
            raw_esm3.load_state_dict(bb_state2, strict=False)
            print(f"  [backbone] Loaded ESM3 keys via alternative prefix.")
        else:
            print("  [backbone] No ESM3 keys found in checkpoint; using pretrained weights.")

    raw_esm3.eval()
    return raw_esm3


def extract_esm3_attn(raw_esm3, tokenizer, seq, device):
    """
    Run one sequence through the raw ESM3 model and capture backbone attention
    via a forward hook on the last self-attention sub-module.

    Returns (attn_matrix [L,L], method_str) where method_str is 'hook' or 'proxy'.
    """
    global _ESM3_HOOK_BUFFER
    _ESM3_HOOK_BUFFER = []

    # Register hook on the last attention-like module
    mod_name, attn_mod = _find_last_attn_module(raw_esm3)
    handle = None
    if attn_mod is not None:
        handle = attn_mod.register_forward_hook(_make_esm3_hook())

    # Tokenise
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    L = len(seq)

    with torch.no_grad():
        out = raw_esm3(sequence_tokens=input_ids)

    if handle is not None:
        handle.remove()

    # Try to use hook output
    if _ESM3_HOOK_BUFFER:
        w = _ESM3_HOOK_BUFFER[-1]          # [1, H, L', L']  (may include special tokens)
        w = w[0]                            # [H, L', L']
        avg = w.mean(dim=0)                 # [L', L']

        # Strip special tokens at beginning/end to get [L, L]
        Lp = avg.shape[0]
        if Lp >= L + 2:
            core = avg[1:L+1, 1:L+1]
        elif Lp >= L:
            core = avg[:L, :L]
        else:
            # Unexpected shape – fall through to proxy
            core = None

        if core is not None:
            row_sums = core.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            core = (core / row_sums).numpy()
            return core, "hook"

    # ── Proxy: cosine-similarity attention from backbone embeddings ────────────
    print(f"    [backbone/ESM3] Hook did not return attention; using cosine-similarity proxy.")
    if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        h = out.last_hidden_state[0]
    elif hasattr(out, 'sequence_last_hidden_states') and out.sequence_last_hidden_states is not None:
        h = out.sequence_last_hidden_states[0]
    elif hasattr(out, 'embeddings') and out.embeddings is not None:
        h = out.embeddings[0]
    else:
        available = [k for k, v in vars(out).items() if v is not None]
        raise RuntimeError(
            f"Cannot find hidden states in ESM3 output. "
            f"Non-None fields: {available}")

    # Strip special tokens
    if h.shape[0] >= L + 2:
        h = h[1:L+1]           # [L, D]
    elif h.shape[0] > L:
        h = h[:L]

    # Cosine similarity matrix
    h = h.float()
    norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    h_n  = h / norm
    sim  = torch.mm(h_n, h_n.t()).cpu().numpy()      # [L, L], range [-1, 1]

    # Shift to [0, 1] and row-normalise to mimic softmax attention
    sim  = (sim + 1.0) / 2.0
    row_sums = sim.sum(axis=-1, keepdims=True).clip(1e-12)
    sim  = sim / row_sums
    return sim, "proxy"


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract backbone (ESM2/ESM3) last-layer attention from a checkpoint."
    )
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pth")
    p.add_argument("--fasta_file",  required=True, help="Path to FASTA file")
    p.add_argument("--output",      required=True, help="Output JSON path")
    p.add_argument("--esm_model",   default="esm2_t33_650M_UR50D",
                   help="HuggingFace ESM2 model name (ignored when --is_esm3 is set)")
    p.add_argument("--is_esm3",     action="store_true",
                   help="Use ESM3 backbone instead of ESM2")
    p.add_argument("--device",      default=None,
                   help="Torch device (default: cuda if available, else cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    records = []

    if not args.is_esm3:
        # ── ESM2 path ──────────────────────────────────────────────────────
        print(f"Building ESM2 model: {args.esm_model}")
        tokenizer  = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")
        esm2_model = build_esm2_from_checkpoint(checkpoint, args.esm_model, device)

        seqs = list(parse_fasta(args.fasta_file))
        print(f"Sequences to process: {len(seqs)}")

        for idx, (seq_id, seq) in enumerate(seqs):
            print(f"  [{idx+1}/{len(seqs)}] {seq_id}  (len={len(seq)})")
            attn = extract_esm2_attn(esm2_model, tokenizer, seq, device)
            records.append({
                "name":              seq_id,
                "sequence":          seq,
                "attention_weights": attn_to_list(attn),
                "backbone_type":     "esm2",
                "attention_source":  "last_layer_avg_heads",
            })

    else:
        # ── ESM3 path ──────────────────────────────────────────────────────
        if not ESM3_AVAILABLE:
            raise ImportError(
                "ESM3 requires the 'esm' library from EvolutionaryScale.\n"
                "Install: pip install esm  then accept the licence on HuggingFace."
            )
        print("Building ESM3 model …")
        tokenizer = EsmSequenceTokenizer()
        raw_esm3  = build_esm3_from_checkpoint(checkpoint, device)
        raw_esm3  = raw_esm3.to(device)

        seqs = list(parse_fasta(args.fasta_file))
        print(f"Sequences to process: {len(seqs)}")

        for idx, (seq_id, seq) in enumerate(seqs):
            print(f"  [{idx+1}/{len(seqs)}] {seq_id}  (len={len(seq)})")
            attn, method = extract_esm3_attn(raw_esm3, tokenizer, seq, device)
            records.append({
                "name":              seq_id,
                "sequence":          seq,
                "attention_weights": attn_to_list(attn),
                "backbone_type":     "esm3",
                "attention_source":  f"last_layer_avg_heads_{method}",
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(records, fh)
    print(f"\nSaved {len(records)} records → {args.output}")


if __name__ == "__main__":
    main()
