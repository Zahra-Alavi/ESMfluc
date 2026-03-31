#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-protein SVD diagnostics from full attention matrices.

Input JSON format: list of records, each with at least
  - name
  - attention_weights (L x L matrix)

Outputs:
  - per-protein CSV with singular-value diagnostics
  - optional scree CSV with top-k singular values per protein
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Per-protein SVD analysis of attention matrices")
    parser.add_argument("--attention_json", type=str, required=True, help="Path to full attention JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--top_k", type=int, default=20, help="Number of singular values to store (default: 20)")
    parser.add_argument(
        "--normalize_rows",
        action="store_true",
        help="Row-normalize attention matrix before SVD (sum rows to 1)",
    )
    return parser.parse_args()


def safe_row_normalize(a: np.ndarray) -> np.ndarray:
    row_sums = a.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return a / row_sums


def effective_rank_from_singular_values(s: np.ndarray) -> float:
    # Entropy-based effective rank: exp(H(p)), p_i = s_i / sum(s)
    s_sum = float(np.sum(s))
    if s_sum <= 0:
        return 0.0
    p = s / s_sum
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def k_for_energy(s: np.ndarray, energy_threshold: float) -> int:
    # Energy over Frobenius norm: cumulative sum(s^2) / sum(s^2)
    s2 = s ** 2
    total = float(np.sum(s2))
    if total <= 0:
        return 0
    cum = np.cumsum(s2) / total
    return int(np.searchsorted(cum, energy_threshold) + 1)


def analyze_record(record: dict, top_k: int, normalize_rows: bool) -> tuple[dict, dict]:
    seq_id = record.get("name", "unknown")
    a = np.asarray(record["attention_weights"], dtype=float)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{seq_id}: attention matrix must be square 2D, got shape={a.shape}")

    if normalize_rows:
        a = safe_row_normalize(a)

    s = np.linalg.svd(a, compute_uv=False)

    fro_sq = float(np.sum(s ** 2))
    sigma1_sq = float(s[0] ** 2) if len(s) > 0 else 0.0
    stable_rank = (fro_sq / sigma1_sq) if sigma1_sq > 0 else 0.0

    n90 = k_for_energy(s, 0.90)
    n95 = k_for_energy(s, 0.95)
    n99 = k_for_energy(s, 0.99)

    metrics = {
        "seq_id": seq_id,
        "length": int(a.shape[0]),
        "normalize_rows": bool(normalize_rows),
        "frobenius_norm": float(np.sqrt(fro_sq)),
        "spectral_norm": float(s[0]) if len(s) > 0 else 0.0,
        "n_singular": int(len(s)),
        "effective_rank": effective_rank_from_singular_values(s),
        "stable_rank": stable_rank,
        "n90_energy": n90,
        "n95_energy": n95,
        "n99_energy": n99,
    }

    scree = {"seq_id": seq_id, "length": int(a.shape[0])}
    for i in range(min(top_k, len(s))):
        scree[f"sv_{i+1}"] = float(s[i])

    # Also store normalized explained energy of top-k
    if fro_sq > 0:
        for i in range(min(top_k, len(s))):
            scree[f"energy_{i+1}"] = float((s[i] ** 2) / fro_sq)

    return metrics, scree


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.attention_json, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("attention_json must contain a list of protein records")

    metrics_rows = []
    scree_rows = []
    failed = []

    for rec in data:
        seq_id = rec.get("name", "unknown")
        if "attention_weights" not in rec:
            failed.append((seq_id, "missing attention_weights"))
            continue

        try:
            metrics, scree = analyze_record(rec, top_k=args.top_k, normalize_rows=args.normalize_rows)
            metrics_rows.append(metrics)
            scree_rows.append(scree)
        except Exception as exc:
            failed.append((seq_id, str(exc)))

    if not metrics_rows:
        raise RuntimeError("No proteins were successfully analyzed.")

    df_metrics = pd.DataFrame(metrics_rows)
    df_scree = pd.DataFrame(scree_rows)

    out_metrics = os.path.join(args.output_dir, "attention_svd_summary.csv")
    out_scree = os.path.join(args.output_dir, f"attention_svd_top{args.top_k}.csv")
    out_failed = os.path.join(args.output_dir, "attention_svd_failed.csv")

    df_metrics.to_csv(out_metrics, index=False)
    df_scree.to_csv(out_scree, index=False)

    if failed:
        pd.DataFrame(failed, columns=["seq_id", "error"]).to_csv(out_failed, index=False)

    print("✓ SVD analysis complete")
    print(f"Processed proteins: {len(df_metrics)}")
    print(f"Failed proteins: {len(failed)}")
    print(f"Median effective rank: {df_metrics['effective_rank'].median():.2f}")
    print(f"Median n95_energy: {df_metrics['n95_energy'].median():.1f}")
    print(f"Summary CSV: {out_metrics}")
    print(f"Top-k singular values CSV: {out_scree}")
    if failed:
        print(f"Failures CSV: {out_failed}")


if __name__ == "__main__":
    main()
