#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:32:35 2025

@author: zalavi

esmfluc_bench.py

Run an ESMfluc binary classifier checkpoint on sequences in a FASTA file,
merge with per-residue Neq (ground truth) and NetSurfP disorder scores,
and compute Spearman & AUROC per sequence and overall.

Usage:
  python esmfluc_bench.py \
      --checkpoint best_model.pth \
      --fasta_file test_data.fasta \
      --neq_csv test_data.csv \
      --netsurf_csv test_data_nsp3.csv \
      --outbase bench_out

Outputs:
  bench_out_model.json          (per-seq: attn, logits, probs)
  bench_out_long.csv            (tidy per-residue merged table)
  bench_out_metrics_perseq.csv  (metrics per sequence)
  bench_out_metrics_summary.csv (aggregate mean/sem)
"""

import argparse, json, ast, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

from models import LSTMWithSelfAttentionModel


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--fasta_file", required=True, type=str)
    ap.add_argument("--neq_csv", required=True, type=str,
                    help="CSV with columns: name,sequence,neq (list string).")
    ap.add_argument("--netsurf_csv", required=True, type=str,
                    help="NetSurfP long CSV: id, seq, n, ..., disorder.")
    ap.add_argument("--netsurf_disorder_col", default="disorder",
                    help="Column name of disorder prob (after stripping spaces).")
    ap.add_argument("--outbase", required=True, type=str,
                    help="Basename for output files (no extension).")
    ap.add_argument("--neq_thresh", default=1.0, type=float,
                    help="Threshold to call residue flexible (Neq > thresh). Default=1.0.")
    ap.add_argument("--skip_attention", action="store_true",
                    help="Don’t store full LxL attention matrix (saves space).")
    return ap.parse_args()


# --------------------------------------------------------------------------------------
# FASTA parser
# --------------------------------------------------------------------------------------
def parse_fasta_file(fasta_path):
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
                seq_id = line[1:].strip()  # drop > and whitespace
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None and seq_lines:
            yield seq_id, "".join(seq_lines)


# --------------------------------------------------------------------------------------
# Load checkpoint robustly
# --------------------------------------------------------------------------------------
def load_state_dict_safely(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model


# --------------------------------------------------------------------------------------
# Model forward
# --------------------------------------------------------------------------------------
def run_model(model, tokenizer, sequence, device, return_attention=True):
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_weights_torch = model(
            input_ids, attn_mask, return_attention=return_attention
        )

    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # shapes: logits = (1,L,2)
    logits_np = logits[0].cpu().numpy()               # (L,2)
    prob_np   = torch.softmax(logits, dim=-1)[0,:,1].cpu().numpy()  # P(class==1; flexible)
    pred_np   = torch.argmax(logits, dim=-1)[0].cpu().numpy()       # 0/1

    attn_np = None
    if return_attention:
        attn_np = attn_weights_torch[0].cpu().numpy() # (L,L)

    return tokens, logits_np, prob_np, pred_np, attn_np


# --------------------------------------------------------------------------------------
# Parse Neq CSV  (name,sequence,neq=list string)
# --------------------------------------------------------------------------------------
def load_neq_csv(path):
    df = pd.read_csv(path)
    # Try to normalize name column variants
    if "name" not in df.columns:
        # try 'id' or first col fallback
        if "id" in df.columns:
            df = df.rename(columns={"id": "name"})
        else:
            df = df.rename(columns={df.columns[0]: "name"})
    if "sequence" not in df.columns:
        raise ValueError("Need a 'sequence' column in Neq CSV.")
    if "neq" not in df.columns:
        raise ValueError("Need a 'neq' column in Neq CSV.")

    out_rows = []
    for _, row in df.iterrows():
        name = str(row["name"]).lstrip(">")
        seq  = str(row["sequence"])
        raw  = row["neq"]

        # parse list string
        if isinstance(raw, str):
            try:
                vals = ast.literal_eval(raw)
            except Exception:
                vals = [float(x) for x in raw.split(",")]
        else:
            # numeric? maybe single value
            vals = [float(raw)]

        vals = [float(v) for v in vals]
        if len(vals) != len(seq):
            print(f"[WARN] Neq length {len(vals)} != seq length {len(seq)} for {name}", file=sys.stderr)

        L = min(len(vals), len(seq))
        for i in range(L):
            out_rows.append({"name": name, "res_idx": i+1, "aa": seq[i], "Neq": vals[i]})
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------------------
# Parse NetSurfP CSV  (long)
# --------------------------------------------------------------------------------------
def load_netsurf_csv(path, disorder_col="disorder"):
    df = pd.read_csv(path, sep=None, engine="python")
    # strip whitespace from column headers
    df.columns = [c.strip() for c in df.columns]
    disorder_col = disorder_col.strip()
    if disorder_col not in df.columns:
        raise ValueError(f"Disorder column '{disorder_col}' not found. Available: {df.columns.tolist()}")

    rows = []
    for _, r in df.iterrows():
        name = str(r["id"]).lstrip(">")
        aa   = str(r["seq"].strip()) if "seq" in df.columns else str(r[" seq"].strip())
        # NetSurf 'n' column is residue index starting at 1
        if "n" in df.columns:
            idx_col = "n"
        elif " n" in df.columns:
            idx_col = " n"
        else:
            raise ValueError("Could not find residue index column (n).")

        res_idx = int(r[idx_col])
        dis_val = float(r[disorder_col])
        rows.append({"name": name, "res_idx": res_idx, "aa": aa, "netsurf_disorder": dis_val})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def per_chain_metrics(df_chain, score_col, label_col, cont_col):
    """
    df_chain: single-sequence slice
    score_col: predictor column (continuous, higher=more flex)
    label_col: binary label column (0/1)
    cont_col : continuous Neq
    Returns dict with AUROC & Spearman (or None if degenerate)
    """
    # AUROC: need at least 1 pos + 1 neg
    if df_chain[label_col].nunique() < 2:
        auroc = np.nan
    else:
        auroc = roc_auc_score(df_chain[label_col], df_chain[score_col])

    # Spearman: require >1 unique value in both vectors
    if df_chain[cont_col].nunique() < 2 or df_chain[score_col].nunique() < 2:
        rho = np.nan
    else:
        rho = spearmanr(df_chain[cont_col], df_chain[score_col], nan_policy="omit").correlation
    return {"AUROC": auroc, "Spearman": rho}


def run_metrics(df_long, neq_thresh=1.0):
    df_long = df_long.copy()
    df_long["flex_label"] = (df_long["Neq"] > neq_thresh).astype(int)

    methods = ["esm_prob_flex", "esm_logit_flex", "netsurf_disorder"]
    recs = []
    for name, g in df_long.groupby("name"):
        for m in methods:
            res = per_chain_metrics(g, score_col=m, label_col="flex_label", cont_col="Neq")
            recs.append({"name": name, "method": m, **res})
    metrics_df = pd.DataFrame(recs)

    # summary
    summary = (
        metrics_df.groupby("method")[["AUROC","Spearman"]]
        .agg(["mean","sem","count"])
        .sort_values(("AUROC","mean"), ascending=False)
    )
    return metrics_df, summary


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embedding backbone
    print("[INFO] Loading ESM2 650M embeddings …")
    embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    # Build downstream head
    print("[INFO] Building LSTMWithSelfAttentionModel …")
    model = LSTMWithSelfAttentionModel(
        embedding_model=embedding_model,
        hidden_size=512,
        num_layers=3,
        num_classes=2,
        dropout=0.3,
    ).to(device)

    print(f"[INFO] Loading checkpoint {args.checkpoint}")
    load_state_dict_safely(model, args.checkpoint, device)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Run model on FASTA
    perseq_records = []
    long_rows = []
    for seq_id, seq_str in parse_fasta_file(args.fasta_file):
        tokens, logits_np, prob_np, pred_np, attn_np = run_model(
            model, tokenizer, seq_str, device, return_attention=not args.skip_attention
        )
        if len(tokens) != len(seq_str):
            print(f"[WARN] token/seq length mismatch in {seq_id} ({len(tokens)} vs {len(seq_str)})", file=sys.stderr)

        # store full-seq record
        rec = {
            "name": seq_id,
            "sequence": seq_str,
            "esm_logits": logits_np.tolist(),     # (L,2)
            "esm_prob_flex": prob_np.tolist(),    # (L,)
            "esm_pred_class": pred_np.tolist(),   # (L,)
        }
        if attn_np is not None:
            rec["attention_weights"] = attn_np.tolist()
        perseq_records.append(rec)

        # expand to long
        for i, aa in enumerate(seq_str, start=1):
            long_rows.append({
                "name": seq_id,
                "res_idx": i,
                "aa": aa,
                "esm_logit_flex": float(logits_np[i-1,1]),
                "esm_prob_flex": float(prob_np[i-1]),
                "esm_pred_class": int(pred_np[i-1]),
            })

    # Save model output JSON
    out_json = Path(f"{args.outbase}_model.json")
    with open(out_json, "w") as fh:
        json.dump(perseq_records, fh, indent=2)
    print(f"[OK] Wrote model outputs → {out_json}")

    df_model_long = pd.DataFrame(long_rows)

    # Load Neq
    print("[INFO] Loading Neq CSV …")
    df_neq = load_neq_csv(args.neq_csv)
    # Load NetSurf
    print("[INFO] Loading NetSurfP CSV …")
    df_ns = load_netsurf_csv(args.netsurf_csv, disorder_col=args.netsurf_disorder_col)

    # Merge
    print("[INFO] Merging tables …")
    df_long = (
        df_model_long.merge(df_neq, on=["name","res_idx","aa"], how="inner")
                     .merge(df_ns,  on=["name","res_idx","aa"], how="inner")
    )
    print(f"[OK] merged rows: {len(df_long)}")

    # Metrics
    print("[INFO] Computing metrics …")
    metrics_df, summary = run_metrics(df_long, neq_thresh=args.neq_thresh)

    out_long = Path(f"{args.outbase}_long.csv")
    out_metrics_perseq = Path(f"{args.outbase}_metrics_perseq.csv")
    out_summary = Path(f"{args.outbase}_metrics_summary.csv")

    df_long.to_csv(out_long, index=False)
    metrics_df.to_csv(out_metrics_perseq, index=False)
    summary.to_csv(out_summary)

    print(f"[OK] per-res data  → {out_long}")
    print(f"[OK] per-seq metrics → {out_metrics_perseq}")
    print(f"[OK] summary metrics → {out_summary}")
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
