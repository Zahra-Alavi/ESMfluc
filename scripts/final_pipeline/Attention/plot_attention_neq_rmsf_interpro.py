#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-protein integrated profile plotting and summary generation.

For each protein present across inputs:
1) Compute average received attention profile (scaled to [0,1])
2) Plot scaled attention + actual Neq + actual RMSF
3) Mark InterPro functional residues on the plot
4) Save summary CSV with:
   - sequence name and sequence
   - top 10% attention residues (index + one-letter AA)
   - top PCA interaction pairs (index + one-letter AA)

Supports InterPro input as JSON (InterProScan) or TSV/CSV (range/position format).
"""

import argparse
import ast
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot per-protein attention/Neq/RMSF with InterPro markers and create summary CSV"
    )
    p.add_argument("--attention_json", type=str, required=True, help="Path to attention JSON")
    p.add_argument("--neq_csv", type=str, required=True, help="Path to CSV with columns: name, sequence, neq")
    p.add_argument("--rmsf_tsv", type=str, required=True, help="Path to RMSF TSV")
    p.add_argument(
        "--interpro_path",
        type=str,
        required=True,
        help="Path to InterPro JSON or TSV/CSV",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--top_percent_attention",
        type=float,
        default=10.0,
        help="Top percent for attention residues (default: 10)",
    )
    p.add_argument(
        "--pca_components",
        type=int,
        default=3,
        help="Number of PCA components for interaction ranking (default: 3)",
    )
    p.add_argument(
        "--top_pca_pairs",
        type=int,
        default=10,
        help="Number of top PCA pairs to store in summary (default: 10)",
    )
    p.add_argument(
        "--exclude_window",
        type=int,
        default=0,
        help="Exclude near-diagonal PCA pairs with |i-j| <= window (default: 0)",
    )
    p.add_argument(
        "--plot_scale_mode",
        type=str,
        default="minmax",
        choices=["quantile", "minmax", "zscore"],
        help=(
            "Scaling mode for plotting all three tracks on one axis. "
            "minmax = plain min-max (same logic as attention-vs-RMSF plot; default), "
            "quantile = clip to 5th/95th then min-max, "
            "minmax = plain min-max, zscore = z-score then min-max"
        ),
    )
    return p.parse_args()


def parse_attention_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("attention_json must contain a list of protein records")
    return {rec["name"]: rec for rec in data if "name" in rec and "attention_weights" in rec}


def parse_listlike(value):
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    raise ValueError(f"Unsupported list-like value type: {type(value)}")


def parse_neq_csv(path):
    df = pd.read_csv(path)
    required = {"name", "sequence", "neq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Neq CSV missing columns: {sorted(missing)}")

    out = {}
    for _, row in df.iterrows():
        seq_id = str(row["name"])
        try:
            neq = parse_listlike(row["neq"])
        except Exception:
            continue
        out[seq_id] = {
            "sequence": str(row["sequence"]),
            "neq": neq,
        }
    return out


def parse_rmsf_tsv(path):
    df = pd.read_csv(path, sep="\t")
    required = {"sequence_name", "sequence", "avg_per_residue_rmsf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RMSF TSV missing columns: {sorted(missing)}")

    out = {}
    for _, row in df.iterrows():
        seq_id = str(row["sequence_name"])
        try:
            rmsf = parse_listlike(row["avg_per_residue_rmsf"])
        except Exception:
            continue
        out[seq_id] = {
            "sequence": str(row["sequence"]),
            "rmsf": rmsf,
        }
    return out


def parse_interpro_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    out = {}

    for entry in results:
        xrefs = entry.get("xref", [])
        seq_id = xrefs[0].get("id") if xrefs else entry.get("md5", "unknown")
        sequence = entry.get("sequence", "")

        indices = set()
        for match in entry.get("matches", []):
            for loc in match.get("locations", []):
                for site in (loc.get("sites") or []):
                    for sloc in site.get("siteLocations", []):
                        s = sloc.get("start")
                        e = sloc.get("end")
                        if s is None:
                            continue
                        if e is None:
                            e = s
                        for pos in range(int(s), int(e) + 1):
                            indices.add(pos - 1)  # 1-based -> 0-based

        out[seq_id] = {
            "sequence": sequence,
            "functional_indices": sorted(indices),
        }

    return out


def _pick_first_column(df, options):
    lower_map = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    return None


def parse_interpro_table(path):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)

    id_col = _pick_first_column(df, ["sequence_name", "name", "seq_id", "protein", "id", "accession"])
    seq_col = _pick_first_column(df, ["sequence", "seq"])
    start_col = _pick_first_column(df, ["start", "site_start", "from", "begin"])
    end_col = _pick_first_column(df, ["end", "site_end", "to", "stop"])
    pos_col = _pick_first_column(df, ["position", "residue_index", "residue", "site"])

    if id_col is None:
        raise ValueError("InterPro table needs an ID column (e.g., sequence_name/name/seq_id)")

    grouped = defaultdict(lambda: {"sequence": "", "functional_indices": set()})

    for _, row in df.iterrows():
        seq_id = str(row[id_col])
        if seq_col is not None and pd.notna(row[seq_col]):
            grouped[seq_id]["sequence"] = str(row[seq_col])

        if start_col is not None:
            if pd.isna(row[start_col]):
                continue
            s = int(row[start_col])
            e = int(row[end_col]) if end_col is not None and pd.notna(row[end_col]) else s
            for pos in range(s, e + 1):
                grouped[seq_id]["functional_indices"].add(pos - 1)  # assume 1-based
        elif pos_col is not None:
            if pd.isna(row[pos_col]):
                continue
            p = int(row[pos_col])
            grouped[seq_id]["functional_indices"].add(p - 1)  # assume 1-based

    out = {}
    for k, v in grouped.items():
        out[k] = {
            "sequence": v["sequence"],
            "functional_indices": sorted(v["functional_indices"]),
        }
    return out


def parse_interpro(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return parse_interpro_json(path)
    if ext in {".tsv", ".csv"}:
        return parse_interpro_table(path)
    # fallback: try JSON first, then table
    try:
        return parse_interpro_json(path)
    except Exception:
        return parse_interpro_table(path)


def avg_received_attention(attn):
    l = attn.shape[0]
    if l <= 1:
        return attn.mean(axis=0)
    col_sum = attn.sum(axis=0)
    return (col_sum - np.diag(attn)) / (l - 1)


def minmax_scale(x):
    x = np.asarray(x, dtype=float)
    # Same normalization style as analyze_attention_vs_rmsf.py
    return (x - x.min()) / (np.ptp(x) + 1e-12)


def scale_profile(x, mode="quantile"):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    if mode == "minmax":
        return minmax_scale(x)

    if mode == "zscore":
        std = np.std(x)
        if std <= 1e-12:
            z = np.zeros_like(x)
        else:
            z = (x - np.mean(x)) / std
        return minmax_scale(z)

    # quantile mode (default): robust to outliers
    lo, hi = np.percentile(x, [5, 95])
    if hi - lo <= 1e-12:
        return minmax_scale(x)
    x_clip = np.clip(x, lo, hi)
    return (x_clip - lo) / (hi - lo)


def top_percent_indices(values, top_percent):
    thr = np.percentile(values, 100 - top_percent)
    return np.where(values >= thr)[0], thr


def pca_rowwise(attn, k):
    n = attn.shape[0]
    mean = attn.mean(axis=0, keepdims=True)
    x = attn - mean

    denom = max(n - 1, 1)
    cov = (x.T @ x) / denom
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.where(eigvals > 0, eigvals, 0.0)

    k_eff = max(1, min(k, eigvecs.shape[1]))
    components = eigvecs[:, :k_eff].T
    scores = x @ components.T
    signal_k = scores @ components
    return signal_k, components, eigvals


def top_pca_pairs(attn, k, top_n, exclude_window=0):
    signal_k, _, _ = pca_rowwise(attn, k)
    l = attn.shape[0]

    ii, jj = np.indices((l, l))
    flat_i = ii.ravel()
    flat_j = jj.ravel()
    flat_s = signal_k.ravel()

    if exclude_window > 0:
        keep = np.abs(flat_i - flat_j) > exclude_window
    else:
        keep = flat_i != flat_j

    flat_i = flat_i[keep]
    flat_j = flat_j[keep]
    flat_s = flat_s[keep]

    order = np.argsort(-np.abs(flat_s))
    top = order[: min(top_n, len(order))]

    rows = []
    for idx in top:
        rows.append((int(flat_i[idx]), int(flat_j[idx]), float(flat_s[idx])))
    return rows


def top_raw_attention_pairs(attn, top_n, exclude_window=0):
    l = attn.shape[0]

    ii, jj = np.indices((l, l))
    flat_i = ii.ravel()
    flat_j = jj.ravel()
    flat_a = attn.ravel()

    if exclude_window > 0:
        keep = np.abs(flat_i - flat_j) > exclude_window
    else:
        keep = flat_i != flat_j

    flat_i = flat_i[keep]
    flat_j = flat_j[keep]
    flat_a = flat_a[keep]

    order = np.argsort(-flat_a)
    top = order[: min(top_n, len(order))]

    rows = []
    for idx in top:
        rows.append((int(flat_i[idx]), int(flat_j[idx]), float(flat_a[idx])))
    return rows


def format_ranked_residue_list(indices, sequence, scores):
    ranked = sorted(
        [int(i) for i in indices],
        key=lambda i: (-float(scores[i]), i),
    )

    out = []
    for i in ranked:
        aa = sequence[i] if 0 <= i < len(sequence) else "?"
        out.append(f"{aa}{i}")
    return "[" + ", ".join(out) + "]"


def format_pair_list(pairs, sequence):
    """
    Input pairs are (giver_i, receiver_j, score).
    Output format is (receiver, giver) as requested.
    """
    out = []
    for i, j, _ in pairs:
        giver_aa = sequence[i] if 0 <= i < len(sequence) else "?"
        recv_aa = sequence[j] if 0 <= j < len(sequence) else "?"
        out.append(f"({recv_aa}{j}, {giver_aa}{i})")
    return "[" + ", ".join(out) + "]"


def make_plot(seq_id, x, attn_scaled, neq_scaled, rmsf_scaled, functional_idx, out_path, scale_mode):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Functional markers (same style logic as attention-vs-RMSF: vertical light lines)
    valid_f = sorted({i for i in functional_idx if 0 <= i < len(x)})
    if valid_f:
        for idx in valid_f:
            ax.axvline(x=idx, color="deeppink", alpha=0.18, linewidth=0.5)
        # add one legend handle for all functional lines
        ax.axvline(
            x=valid_f[0],
            color="deeppink",
            alpha=0.5,
            linewidth=1.0,
            linestyle="-",
            label="InterPro functional residues (vertical lines)",
        )

    ax.plot(x, attn_scaled, color="tab:blue", lw=1.3, label="Avg received attention (scaled)")
    ax.plot(x, neq_scaled, color="tab:green", lw=1.2, label="Neq (scaled)")
    ax.plot(x, rmsf_scaled, color="tab:orange", lw=1.2, label="RMSF (scaled)")

    ax.set_xlabel("Residue index (0-based)")
    ax.set_ylabel("Scaled value [0,1]")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.25)

    ax.legend(loc="upper right", frameon=True)
    plt.title(f"{seq_id}: attention / Neq / RMSF (single-axis, {scale_mode} scaling)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "per_protein_plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading attention: {args.attention_json}")
    attn_map = parse_attention_json(args.attention_json)

    print(f"Loading Neq: {args.neq_csv}")
    neq_map = parse_neq_csv(args.neq_csv)

    print(f"Loading RMSF: {args.rmsf_tsv}")
    rmsf_map = parse_rmsf_tsv(args.rmsf_tsv)

    print(f"Loading InterPro: {args.interpro_path}")
    interpro_map = parse_interpro(args.interpro_path)

    core_ids = sorted(set(attn_map) & set(neq_map) & set(rmsf_map))
    with_interpro = set(core_ids) & set(interpro_map)
    print(f"Proteins with attention+Neq+RMSF: {len(core_ids)}")
    print(f"Of those, with InterPro annotations: {len(with_interpro)}")

    summary_rows = []
    failed_rows = []

    for seq_id in core_ids:
        try:
            rec = attn_map[seq_id]
            attn = np.asarray(rec["attention_weights"], dtype=float)
            if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
                raise ValueError(f"attention matrix must be square, got {attn.shape}")

            seq_attn = str(rec.get("sequence", ""))
            seq_neq = str(neq_map[seq_id]["sequence"])
            seq_rmsf = str(rmsf_map[seq_id]["sequence"])

            neq = np.asarray(neq_map[seq_id]["neq"], dtype=float)
            rmsf = np.asarray(rmsf_map[seq_id]["rmsf"], dtype=float)
            f_idx = interpro_map.get(seq_id, {}).get("functional_indices", [])

            l = min(attn.shape[0], len(seq_attn), len(seq_neq), len(seq_rmsf), len(neq), len(rmsf))
            if l < 2:
                raise ValueError("insufficient aligned length")

            attn = attn[:l, :l]
            sequence = seq_attn[:l]
            neq = neq[:l]
            rmsf = rmsf[:l]
            f_idx = [i for i in f_idx if 0 <= i < l]

            attn_received = avg_received_attention(attn)
            attn_scaled = scale_profile(attn_received, mode=args.plot_scale_mode)
            neq_scaled = scale_profile(neq, mode=args.plot_scale_mode)
            rmsf_scaled = scale_profile(rmsf, mode=args.plot_scale_mode)
            x = np.arange(l)

            top_attn_idx, _ = top_percent_indices(attn_received, args.top_percent_attention)
            top_attn_str = format_ranked_residue_list(top_attn_idx.tolist(), sequence, attn_received)

            pca_pairs = top_pca_pairs(
                attn=attn,
                k=args.pca_components,
                top_n=args.top_pca_pairs,
                exclude_window=args.exclude_window,
            )
            pca_pairs_str = format_pair_list(pca_pairs, sequence)

            raw_pairs = top_raw_attention_pairs(
                attn=attn,
                top_n=args.top_pca_pairs,
                exclude_window=args.exclude_window,
            )
            raw_pairs_str = format_pair_list(raw_pairs, sequence)

            out_plot = os.path.join(plot_dir, f"{seq_id}_profiles.png")
            make_plot(
                seq_id=seq_id,
                x=x,
                attn_scaled=attn_scaled,
                neq_scaled=neq_scaled,
                rmsf_scaled=rmsf_scaled,
                functional_idx=f_idx,
                out_path=out_plot,
                scale_mode=args.plot_scale_mode,
            )

            summary_rows.append(
                {
                    "seq_id": seq_id,
                    "length": l,
                    "sequence": sequence,
                    "n_functional_residues": len(f_idx),
                    "functional_residue_indices_0based": ";".join(map(str, sorted(set(f_idx)))),
                    "top_attention_percent": args.top_percent_attention,
                    "top_attention_residues_ranked": top_attn_str,
                    "top_pca_pairs_ranked_receiver_giver": pca_pairs_str,
                    "top_raw_attention_pairs_ranked_receiver_giver": raw_pairs_str,
                }
            )

            # soft sanity checks for user visibility
            if seq_attn[:l] != seq_neq[:l] or seq_attn[:l] != seq_rmsf[:l]:
                failed_rows.append((seq_id, "sequence mismatch across sources (trimmed to min length)"))

        except Exception as exc:
            failed_rows.append((seq_id, str(exc)))

    if not summary_rows:
        raise RuntimeError("No proteins processed successfully")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "final_summary_attention_neq_rmsf_pca.csv")
    summary_df.to_csv(summary_csv, index=False)

    if failed_rows:
        fail_df = pd.DataFrame(failed_rows, columns=["seq_id", "warning_or_error"])
        fail_df.to_csv(os.path.join(args.output_dir, "processing_warnings_errors.csv"), index=False)

    print("✓ Done")
    print(f"Proteins summarized: {len(summary_df)}")
    print(f"Plots saved in: {plot_dir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
