#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze overlap between attention hubs and high-RMSF residues.

This script:
1. Loads per-protein attention matrices from JSON
2. Loads per-protein RMSF vectors from TSV
3. Computes attention hub scores
4. Compares top attention hubs vs top RMSF residues (overlap/enrichment)
5. Computes ROC/PR/correlation metrics and optional plots
"""

import argparse
import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze overlap between attention hubs and high-RMSF residues"
    )
    parser.add_argument("--attention_json", type=str, required=True, help="Path to attention JSON")
    parser.add_argument("--rmsf_tsv", type=str, required=True, help="Path to RMSF summary TSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument(
        "--hub_metric",
        type=str,
        default="max_received",
        choices=["max_received", "avg_received", "sum_received", "max_given", "avg_given"],
        help="Metric used to define attention hubs",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=10.0,
        help="Top percentage used to define high-attention and high-RMSF residues",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable per-protein plot generation",
    )
    return parser.parse_args()


def parse_rmsf_tsv(rmsf_path):
    """Parse RMSF TSV into {seq_id: {'sequence': str, 'rmsf': np.ndarray}}."""
    print(f"Parsing RMSF TSV from {rmsf_path}")
    df = pd.read_csv(rmsf_path, sep="\t")

    required = {"sequence_name", "sequence", "avg_per_residue_rmsf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RMSF TSV missing required columns: {sorted(missing)}")

    rmsf_map = {}
    for _, row in df.iterrows():
        seq_id = str(row["sequence_name"])
        seq = str(row["sequence"])
        rmsf_raw = row["avg_per_residue_rmsf"]

        try:
            rmsf_vals = ast.literal_eval(rmsf_raw) if isinstance(rmsf_raw, str) else rmsf_raw
            rmsf_vals = np.asarray(rmsf_vals, dtype=float)
        except Exception as exc:
            print(f"  Warning: could not parse RMSF for {seq_id}: {exc}")
            continue

        rmsf_map[seq_id] = {"sequence": seq, "rmsf": rmsf_vals}

    print(f"Parsed RMSF for {len(rmsf_map)} sequences\n")
    return rmsf_map


def compute_attention_hubs(attention_matrix, metric="max_received"):
    L = attention_matrix.shape[0]

    if metric == "max_received":
        return attention_matrix.max(axis=0)
    if metric == "avg_received":
        return np.array([(attention_matrix[:, i].sum() - attention_matrix[i, i]) / (L - 1) for i in range(L)])
    if metric == "sum_received":
        return attention_matrix.sum(axis=0)
    if metric == "max_given":
        return attention_matrix.max(axis=1)
    if metric == "avg_given":
        return np.array([(attention_matrix[i, :].sum() - attention_matrix[i, i]) / (L - 1) for i in range(L)])

    raise ValueError(f"Unknown metric: {metric}")


def top_indices(values, top_percent):
    threshold = np.percentile(values, 100 - top_percent)
    return np.where(values >= threshold)[0], threshold


def compute_enrichment(top_a, top_b, L):
    """Fisher enrichment of overlap between two top-index sets."""
    set_a = set(top_a)
    set_b = set(top_b)
    overlap = len(set_a & set_b)

    n_a = len(set_a)
    n_b = len(set_b)

    a = overlap
    b = n_a - overlap
    c = n_b - overlap
    d = L - n_a - c

    contingency = np.array([[a, b], [c, d]])
    odds_ratio, p_value = stats.fisher_exact(contingency, alternative="greater")

    expected = (n_a * n_b) / L
    fold_enrichment = overlap / expected if expected > 0 else np.nan

    return {
        "n_total": L,
        "n_top_attention": n_a,
        "n_top_rmsf": n_b,
        "overlap": overlap,
        "expected": expected,
        "fold_enrichment": fold_enrichment,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def compute_binary_metrics(scores, positive_indices, L):
    y_true = np.zeros(L)
    y_true[positive_indices] = 1

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def compute_correlations(hub_scores, rmsf_values, top_rmsf_indices, L):
    top_rmsf_binary = np.zeros(L)
    top_rmsf_binary[top_rmsf_indices] = 1

    spearman_top_r, spearman_top_p = stats.spearmanr(hub_scores, top_rmsf_binary)
    pointbiserial_r, pointbiserial_p = stats.pointbiserialr(top_rmsf_binary, hub_scores)
    spearman_cont_r, spearman_cont_p = stats.spearmanr(hub_scores, rmsf_values)
    pearson_cont_r, pearson_cont_p = stats.pearsonr(hub_scores, rmsf_values)

    return {
        "spearman_top_r": spearman_top_r,
        "spearman_top_p": spearman_top_p,
        "pointbiserial_r": pointbiserial_r,
        "pointbiserial_p": pointbiserial_p,
        "spearman_cont_r": spearman_cont_r,
        "spearman_cont_p": spearman_cont_p,
        "pearson_cont_r": pearson_cont_r,
        "pearson_cont_p": pearson_cont_p,
    }


def plot_analysis(
    seq_id,
    hub_scores,
    rmsf_values,
    top_hub_indices,
    top_rmsf_indices,
    enrichment,
    metrics,
    corr,
    top_percent,
    output_dir,
):
    L = len(hub_scores)
    x = np.arange(L)
    top_hub_set = set(top_hub_indices)
    top_rmsf_set = set(top_rmsf_indices)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    hub_scaled = (hub_scores - hub_scores.min()) / (hub_scores.ptp() + 1e-12)
    rmsf_scaled = (rmsf_values - rmsf_values.min()) / (rmsf_values.ptp() + 1e-12)
    ax1.plot(x, hub_scaled, label="Attention hub score (scaled)", color="steelblue", linewidth=1.3)
    ax1.plot(x, rmsf_scaled, label="RMSF (scaled)", color="darkorange", linewidth=1.3, alpha=0.9)
    for idx in top_rmsf_indices:
        ax1.axvline(x=idx, color="red", alpha=0.18, linewidth=0.5)
    ax1.set_title(f"{seq_id}: Attention vs RMSF along sequence")
    ax1.set_xlabel("Residue index")
    ax1.set_ylabel("Scaled value")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    top_rmsf_scores = [hub_scores[i] for i in top_rmsf_indices]
    non_top_rmsf_scores = [hub_scores[i] for i in range(L) if i not in top_rmsf_set]
    ax2.hist(non_top_rmsf_scores, bins=30, alpha=0.6, density=True, label="Not top RMSF", color="gray")
    ax2.hist(top_rmsf_scores, bins=30, alpha=0.6, density=True, label="Top RMSF", color="red")
    ax2.set_title("Attention score distribution")
    ax2.set_xlabel("Hub score")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(metrics["fpr"], metrics["tpr"], linewidth=2, label=f'ROC AUC={metrics["roc_auc"]:.3f}')
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax3.set_title("Top-RMSF prediction by hub score")
    ax3.set_xlabel("False positive rate")
    ax3.set_ylabel("True positive rate")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(metrics["recall"], metrics["precision"], linewidth=2, label=f'AP={metrics["avg_precision"]:.3f}')
    baseline = len(top_rmsf_indices) / L
    ax4.axhline(y=baseline, color="k", linestyle="--", label=f"Baseline={baseline:.3f}")
    ax4.set_title("Precision-Recall")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.legend()
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    txt = (
        f"Top {top_percent:.0f}% overlap\n\n"
        f"n residues: {L}\n"
        f"top attention: {enrichment['n_top_attention']}\n"
        f"top RMSF: {enrichment['n_top_rmsf']}\n"
        f"overlap: {enrichment['overlap']}\n"
        f"expected: {enrichment['expected']:.2f}\n"
        f"fold enrichment: {enrichment['fold_enrichment']:.2f}\n"
        f"odds ratio: {enrichment['odds_ratio']:.2f}\n"
        f"Fisher p: {enrichment['p_value']:.2e}"
    )
    ax5.text(0.05, 0.95, txt, transform=ax5.transAxes, va="top", fontsize=10, fontfamily="monospace")

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    txt_corr = (
        "Correlation\n\n"
        f"Spearman(hub, top-RMSF): {corr['spearman_top_r']:.3f} (p={corr['spearman_top_p']:.2e})\n"
        f"Point-biserial: {corr['pointbiserial_r']:.3f} (p={corr['pointbiserial_p']:.2e})\n"
        f"Spearman(hub, RMSF): {corr['spearman_cont_r']:.3f} (p={corr['spearman_cont_p']:.2e})\n"
        f"Pearson(hub, RMSF): {corr['pearson_cont_r']:.3f} (p={corr['pearson_cont_p']:.2e})"
    )
    ax6.text(0.05, 0.95, txt_corr, transform=ax6.transAxes, va="top", fontsize=10, fontfamily="monospace")

    ax7 = fig.add_subplot(gs[2, 2])
    colors = []
    for i in range(L):
        if i in top_hub_set and i in top_rmsf_set:
            colors.append("purple")
        elif i in top_hub_set:
            colors.append("steelblue")
        elif i in top_rmsf_set:
            colors.append("darkorange")
        else:
            colors.append("lightgray")
    ax7.scatter(hub_scores, rmsf_values, c=colors, alpha=0.7, s=15)
    ax7.set_xlabel("Hub score")
    ax7.set_ylabel("RMSF")
    ax7.set_title("Hub score vs RMSF")
    ax7.grid(alpha=0.3)

    plt.suptitle(f"{seq_id}: attention hubs vs RMSF", fontsize=14, y=0.995)
    out_file = os.path.join(output_dir, f"{seq_id}_hub_rmsf_analysis.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def build_correlation_aggregate(df, hub_metric, top_percent):
    """Return one-row aggregate stats focused only on attention-RMSF correlations."""
    n = len(df)

    def _median(col):
        return float(df[col].median())

    def _mean(col):
        return float(df[col].mean())

    def _std(col):
        return float(df[col].std(ddof=1)) if n > 1 else 0.0

    def _q25(col):
        return float(df[col].quantile(0.25))

    def _q75(col):
        return float(df[col].quantile(0.75))

    out = {
        "n_proteins": int(n),
        "hub_metric": hub_metric,
        "top_percent": float(top_percent),
        # Continuous attention-vs-RMSF correlations
        "spearman_cont_r_mean": _mean("spearman_cont_r"),
        "spearman_cont_r_median": _median("spearman_cont_r"),
        "spearman_cont_r_std": _std("spearman_cont_r"),
        "spearman_cont_r_q25": _q25("spearman_cont_r"),
        "spearman_cont_r_q75": _q75("spearman_cont_r"),
        "pearson_cont_r_mean": _mean("pearson_cont_r"),
        "pearson_cont_r_median": _median("pearson_cont_r"),
        "pearson_cont_r_std": _std("pearson_cont_r"),
        "pearson_cont_r_q25": _q25("pearson_cont_r"),
        "pearson_cont_r_q75": _q75("pearson_cont_r"),
        # Significance summaries
        "n_spearman_p_lt_0_05": int((df["spearman_cont_p"] < 0.05).sum()),
        "frac_spearman_p_lt_0_05": float((df["spearman_cont_p"] < 0.05).mean()),
        "n_pearson_p_lt_0_05": int((df["pearson_cont_p"] < 0.05).sum()),
        "frac_pearson_p_lt_0_05": float((df["pearson_cont_p"] < 0.05).mean()),
        # Directionality
        "n_spearman_positive": int((df["spearman_cont_r"] > 0).sum()),
        "frac_spearman_positive": float((df["spearman_cont_r"] > 0).mean()),
        "n_pearson_positive": int((df["pearson_cont_r"] > 0).sum()),
        "frac_pearson_positive": float((df["pearson_cont_r"] > 0).mean()),
    }
    return pd.DataFrame([out])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading attention data from {args.attention_json}")
    with open(args.attention_json, "r") as f:
        attention_data = json.load(f)
    attn_map = {record["name"]: record for record in attention_data}

    rmsf_map = parse_rmsf_tsv(args.rmsf_tsv)

    common_ids = sorted(set(attn_map.keys()) & set(rmsf_map.keys()))
    print(f"Overlapping proteins: {len(common_ids)}")
    if len(common_ids) == 0:
        print("No shared protein IDs found between attention JSON and RMSF TSV.")
        return

    all_results = []

    for seq_id in common_ids:
        record = attn_map[seq_id]
        rmsf_record = rmsf_map[seq_id]

        attention_matrix = np.asarray(record["attention_weights"], dtype=float)
        attention_seq = record.get("sequence", "")
        rmsf_seq = rmsf_record["sequence"]
        rmsf_values = rmsf_record["rmsf"]
        L = attention_matrix.shape[0]

        # Sequence/shape consistency checks
        if attention_matrix.shape[0] != attention_matrix.shape[1]:
            print(f"Skipping {seq_id}: attention matrix is not square")
            continue
        if len(attention_seq) != L:
            print(f"Skipping {seq_id}: attention sequence length ({len(attention_seq)}) != matrix size ({L})")
            continue
        if len(rmsf_seq) != len(rmsf_values):
            print(f"Skipping {seq_id}: RMSF sequence length ({len(rmsf_seq)}) != RMSF vector ({len(rmsf_values)})")
            continue
        if len(rmsf_seq) != L:
            print(f"Skipping {seq_id}: RMSF length ({len(rmsf_seq)}) != attention length ({L})")
            continue

        seq_identity = np.nan
        mismatch_count = np.nan
        if len(attention_seq) == len(rmsf_seq):
            mismatch_count = sum(1 for a, b in zip(attention_seq, rmsf_seq) if a != b)
            seq_identity = 1.0 - (mismatch_count / len(attention_seq))
            if mismatch_count > 0:
                print(f"Warning {seq_id}: sequence mismatch count={mismatch_count}/{L}")

        # Hub and RMSF top residues
        hub_scores = compute_attention_hubs(attention_matrix, metric=args.hub_metric)
        top_hub_indices, hub_threshold = top_indices(hub_scores, args.top_percent)
        top_rmsf_indices, rmsf_threshold = top_indices(rmsf_values, args.top_percent)

        enrichment = compute_enrichment(top_hub_indices, top_rmsf_indices, L)
        metrics = compute_binary_metrics(hub_scores, top_rmsf_indices, L)
        corr = compute_correlations(hub_scores, rmsf_values, top_rmsf_indices, L)

        print(
            f"{seq_id}: overlap={enrichment['overlap']} "
            f"fold={enrichment['fold_enrichment']:.2f} p={enrichment['p_value']:.2e} "
            f"AUC={metrics['roc_auc']:.3f}"
        )

        if not args.no_plots:
            plot_analysis(
                seq_id=seq_id,
                hub_scores=hub_scores,
                rmsf_values=rmsf_values,
                top_hub_indices=top_hub_indices,
                top_rmsf_indices=top_rmsf_indices,
                enrichment=enrichment,
                metrics=metrics,
                corr=corr,
                top_percent=args.top_percent,
                output_dir=args.output_dir,
            )

        all_results.append(
            {
                "seq_id": seq_id,
                "length": L,
                "hub_metric": args.hub_metric,
                "top_percent": args.top_percent,
                "seq_identity": seq_identity,
                "sequence_mismatch_count": mismatch_count,
                "hub_threshold": hub_threshold,
                "rmsf_threshold": rmsf_threshold,
                "n_top_attention": enrichment["n_top_attention"],
                "n_top_rmsf": enrichment["n_top_rmsf"],
                "overlap": enrichment["overlap"],
                "expected": enrichment["expected"],
                "fold_enrichment": enrichment["fold_enrichment"],
                "odds_ratio": enrichment["odds_ratio"],
                "fisher_p": enrichment["p_value"],
                "roc_auc": metrics["roc_auc"],
                "avg_precision": metrics["avg_precision"],
                "spearman_top_r": corr["spearman_top_r"],
                "spearman_top_p": corr["spearman_top_p"],
                "pointbiserial_r": corr["pointbiserial_r"],
                "pointbiserial_p": corr["pointbiserial_p"],
                "spearman_cont_r": corr["spearman_cont_r"],
                "spearman_cont_p": corr["spearman_cont_p"],
                "pearson_cont_r": corr["pearson_cont_r"],
                "pearson_cont_p": corr["pearson_cont_p"],
            }
        )

    if not all_results:
        print("No proteins could be processed after checks.")
        return

    df = pd.DataFrame(all_results)
    out_csv = os.path.join(args.output_dir, "hub_rmsf_analysis_summary.csv")
    df.to_csv(out_csv, index=False)

    corr_out_csv = os.path.join(args.output_dir, "hub_rmsf_correlation_aggregate.csv")
    corr_df = build_correlation_aggregate(df, hub_metric=args.hub_metric, top_percent=args.top_percent)
    corr_df.to_csv(corr_out_csv, index=False)

    print("\n✓ Analysis complete")
    print(f"Processed proteins: {len(df)}")
    print(f"Median fold enrichment: {df['fold_enrichment'].median():.2f}")
    print(f"Median ROC AUC: {df['roc_auc'].median():.3f}")
    print(f"Proteins with p<0.05: {(df['fisher_p'] < 0.05).sum()}/{len(df)}")
    print(f"Median Spearman(attention,RMSF): {df['spearman_cont_r'].median():.3f}")
    print(f"Median Pearson(attention,RMSF):  {df['pearson_cont_r'].median():.3f}")
    print(
        "Spearman p<0.05 proteins: "
        f"{(df['spearman_cont_p'] < 0.05).sum()}/{len(df)}"
    )
    print(
        "Pearson p<0.05 proteins:  "
        f"{(df['pearson_cont_p'] < 0.05).sum()}/{len(df)}"
    )
    print(f"Summary CSV: {out_csv}")
    print(f"Correlation aggregate CSV: {corr_out_csv}")


if __name__ == "__main__":
    main()
