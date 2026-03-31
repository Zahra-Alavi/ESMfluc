#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-protein PCA-style interaction analysis for attention matrices.

Goal:
- Identify which residue-residue attention interactions are most significant
  under a low-dimensional PCA/SVD model.

Method per protein:
1) Treat each row as a sample and each column as a feature
2) Center features (column-wise mean subtraction)
3) Compute PCA from covariance eigendecomposition
4) Keep top-k PCs and reconstruct PCA signal
5) Rank interactions by |signal[i,j]| (or signed signal[i,j])

Outputs:
- pca_summary.csv: explained variance per protein
- pc_vectors.csv: top PC loadings (vectors of size L)
- top_interactions.csv: top ranked interactions per protein
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="PCA interaction significance from attention matrices")
    p.add_argument("--attention_json", type=str, required=True, help="Path to full attention JSON")
    p.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    p.add_argument("--n_components", type=int, default=3, help="Number of PCA components (default: 3)")
    p.add_argument("--top_n", type=int, default=100, help="Top interactions per protein to save (default: 100)")
    p.add_argument(
        "--top_feature_n",
        type=int,
        default=20,
        help="Top absolute-loading features (residue indices) to save per PC (default: 20)",
    )
    p.add_argument(
        "--score_mode",
        type=str,
        default="abs",
        choices=["abs", "positive"],
        help="Rank by absolute reconstructed score or only strongest positive scores",
    )
    p.add_argument(
        "--exclude_diagonal",
        action="store_true",
        help="Exclude self-attention interactions i->i from ranking",
    )
    p.add_argument(
        "--exclude_window",
        type=int,
        default=0,
        help="Exclude near-diagonal pairs with |i-j| <= window (default: 0)."
             " If >0, diagonal is excluded automatically.",
    )
    return p.parse_args()


def pca_rowwise(a: np.ndarray, k: int):
        """
        PCA where rows are samples (L) and columns are features (L).

        Returns:
            signal_k: centered rank-k reconstruction (no mean added)
            components: shape (k, L), each row is a PC loading vector
            eigenvalues_sorted: all eigenvalues desc
            evr: explained variance ratio desc
            k_eff: effective number of components used
            feature_mean: shape (1, L)
        """
        n_samples = a.shape[0]
        feature_mean = a.mean(axis=0, keepdims=True)
        x = a - feature_mean

        denom = max(n_samples - 1, 1)
        cov = (x.T @ x) / denom

        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # numerical safety
        eigvals = np.where(eigvals > 0, eigvals, 0.0)

        k_eff = max(1, min(k, eigvecs.shape[1]))
        components = eigvecs[:, :k_eff].T  # (k, L)
        scores = x @ components.T          # (L, k)
        signal_k = scores @ components     # centered reconstruction

        total = float(eigvals.sum())
        evr = eigvals / total if total > 0 else np.zeros_like(eigvals)

        return signal_k, components, eigvals, evr, k_eff, feature_mean


def rank_interactions(
    a_raw: np.ndarray,
    a_recon: np.ndarray,
    seq_id: str,
    top_n: int,
    mode: str,
    exclude_diag: bool,
    exclude_window: int,
):
    l = a_raw.shape[0]

    ii, jj = np.indices((l, l))
    flat_i = ii.ravel()
    flat_j = jj.ravel()
    flat_recon = a_recon.ravel()
    flat_raw = a_raw.ravel()

    if exclude_window > 0:
        keep = np.abs(flat_i - flat_j) > exclude_window
        flat_i, flat_j = flat_i[keep], flat_j[keep]
        flat_recon, flat_raw = flat_recon[keep], flat_raw[keep]
    elif exclude_diag:
        keep = flat_i != flat_j
        flat_i, flat_j = flat_i[keep], flat_j[keep]
        flat_recon, flat_raw = flat_recon[keep], flat_raw[keep]

    if mode == "abs":
        rank_score = np.abs(flat_recon)
    else:
        rank_score = flat_recon

    order = np.argsort(-rank_score)
    top = order[: min(top_n, len(order))]

    rows = []
    for r, idx in enumerate(top, start=1):
        rows.append(
            {
                "seq_id": seq_id,
                "rank": r,
                "i": int(flat_i[idx]),
                "j": int(flat_j[idx]),
                "recon_score": float(flat_recon[idx]),
                "abs_recon_score": float(abs(flat_recon[idx])),
                "raw_attention": float(flat_raw[idx]),
            }
        )

    return rows


def rank_pc_features(seq_id: str, components: np.ndarray, top_feature_n: int):
    """Return top loading residue indices per PC for interpretability."""
    rows = []
    k_eff, l = components.shape
    for pc_idx in range(k_eff):
        loadings = components[pc_idx]
        order = np.argsort(-np.abs(loadings))[: min(top_feature_n, l)]
        for r, feat_idx in enumerate(order, start=1):
            rows.append(
                {
                    "seq_id": seq_id,
                    "pc_index": pc_idx + 1,
                    "rank": r,
                    "feature_index": int(feat_idx),
                    "loading": float(loadings[feat_idx]),
                    "abs_loading": float(abs(loadings[feat_idx])),
                }
            )
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.attention_json, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("attention_json must be a list of records")

    summary_rows = []
    pc_rows = []
    pc_top_rows = []
    interaction_rows = []
    failed = []

    for rec in data:
        seq_id = rec.get("name", "unknown")
        try:
            a = np.asarray(rec["attention_weights"], dtype=float)
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ValueError(f"matrix must be square 2D, got {a.shape}")

            l = a.shape[0]
            signal_k, components, eigvals, evr, k_eff, feature_mean = pca_rowwise(a, args.n_components)

            # Reconstructed matrix (optional use/reference): mean + PCA signal
            a_recon = signal_k + feature_mean

            cum = np.cumsum(evr)
            summary_rows.append(
                {
                    "seq_id": seq_id,
                    "length": l,
                    "n_components": k_eff,
                    "pc1_explained_var": float(evr[0]) if len(evr) > 0 else 0.0,
                    "pc2_explained_var": float(evr[1]) if len(evr) > 1 else 0.0,
                    "pc3_explained_var": float(evr[2]) if len(evr) > 2 else 0.0,
                    "cum_explained_var_k": float(cum[k_eff - 1]) if len(cum) > 0 else 0.0,
                    "pc1_eigenvalue": float(eigvals[0]) if len(eigvals) > 0 else 0.0,
                    "pc2_eigenvalue": float(eigvals[1]) if len(eigvals) > 1 else 0.0,
                    "pc3_eigenvalue": float(eigvals[2]) if len(eigvals) > 2 else 0.0,
                }
            )

            # Save PC vectors (length-L loadings)
            for pc_idx in range(k_eff):
                loadings = components[pc_idx]
                for feat_idx, val in enumerate(loadings):
                    pc_rows.append(
                        {
                            "seq_id": seq_id,
                            "pc_index": pc_idx + 1,
                            "feature_index": int(feat_idx),
                            "loading": float(val),
                            "abs_loading": float(abs(val)),
                        }
                    )

            pc_top_rows.extend(rank_pc_features(seq_id, components, args.top_feature_n))

            interaction_rows.extend(
                rank_interactions(
                    a_raw=a,
                    # rank on PCA signal (centered reconstruction), not raw matrix
                    a_recon=signal_k,
                    seq_id=seq_id,
                    top_n=args.top_n,
                    mode=args.score_mode,
                    exclude_diag=args.exclude_diagonal,
                    exclude_window=args.exclude_window,
                )
            )

        except Exception as exc:
            failed.append((seq_id, str(exc)))

    if not summary_rows:
        raise RuntimeError("No proteins processed successfully")

    df_sum = pd.DataFrame(summary_rows)
    df_pc = pd.DataFrame(pc_rows)
    df_pc_top = pd.DataFrame(pc_top_rows)
    df_top = pd.DataFrame(interaction_rows)

    out_sum = os.path.join(args.output_dir, "pca_summary.csv")
    out_pc = os.path.join(args.output_dir, "pc_vectors.csv")
    out_pc_top = os.path.join(args.output_dir, "pc_top_features.csv")
    out_top = os.path.join(args.output_dir, "top_interactions.csv")
    out_fail = os.path.join(args.output_dir, "pca_failed.csv")

    df_sum.to_csv(out_sum, index=False)
    df_pc.to_csv(out_pc, index=False)
    df_pc_top.to_csv(out_pc_top, index=False)
    df_top.to_csv(out_top, index=False)
    if failed:
        pd.DataFrame(failed, columns=["seq_id", "error"]).to_csv(out_fail, index=False)

    print("✓ PCA interaction analysis complete")
    print(f"Processed proteins: {len(df_sum)}")
    print(f"Top interactions rows: {len(df_top)}")
    print(f"Median cum_explained_var_k: {df_sum['cum_explained_var_k'].median():.4f}")
    print(f"Saved summary: {out_sum}")
    print(f"Saved PC vectors: {out_pc}")
    print(f"Saved top PC features: {out_pc_top}")
    print(f"Saved interactions: {out_top}")
    if failed:
        print(f"Saved failures: {out_fail}")


if __name__ == "__main__":
    main()
