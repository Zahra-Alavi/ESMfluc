#!/usr/bin/env python3
"""
Analyze seed stability and cross-condition attention agreement for the
publication-oriented comparable ESMfluc result set.

Inputs:
  results/<RESULT_SET>/manifest.tsv
  results/<RESULT_SET>/runs/<condition>/seed_<seed>/attention.json
  test CSV with sequence and Neq values

Outputs:
  results/<RESULT_SET>/analysis/
"""

import argparse
import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except Exception:
    accuracy_score = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze attention stability across seeds and model conditions."
    )
    parser.add_argument(
        "--result_root",
        required=True,
        help="Result root, e.g. results/publication_comparable_20260624_120000",
    )
    parser.add_argument(
        "--test_csv",
        default=None,
        help="Optional test CSV. Defaults to <result_root>/test_data.csv if present.",
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.10,
        help="Fraction of residues used for top-k hub overlap. Default: 0.10",
    )
    parser.add_argument(
        "--diag_window",
        type=int,
        default=5,
        help="Exclude |i-j| <= window for long-range attention. Default: 5",
    )
    return parser.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def rankdata_average(x):
    x = np.asarray(x, dtype=float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    sorted_x = x[sorter]
    obs = np.r_[True, sorted_x[1:] != sorted_x[:-1]]
    dense = obs.cumsum() - 1
    counts = np.bincount(dense)
    starts = np.r_[0, counts.cumsum()[:-1]]
    avg_ranks = starts + (counts - 1) / 2.0 + 1.0
    return avg_ranks[dense][inv]


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return np.nan
    return float(np.sum(a * b) / denom)


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return pearson(rankdata_average(a[mask]), rankdata_average(b[mask]))


def top_indices(x, frac):
    x = np.asarray(x, dtype=float)
    k = max(1, int(math.ceil(len(x) * frac)))
    return set(np.argsort(x)[-k:].tolist())


def jaccard(a, b):
    if not a and not b:
        return np.nan
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def row_entropy(attn):
    eps = 1e-12
    n = attn.shape[1]
    if n <= 1:
        return np.nan
    row_sums = attn.sum(axis=1, keepdims=True)
    probs = np.divide(attn, row_sums, out=np.zeros_like(attn), where=row_sums > 0)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(np.nanmean(entropy / np.log(n)))


def attention_masses(attn, diag_window):
    n = attn.shape[0]
    idx = np.arange(n)
    local = np.abs(idx[:, None] - idx[None, :]) <= diag_window
    total = float(np.sum(attn))
    if total <= 0:
        return np.nan, np.nan
    diagonal_mass = float(np.sum(attn[local]) / total)
    long_range_mass = float(np.sum(attn[~local]) / total)
    return diagonal_mass, long_range_mass


def hub_concentration(received, top_frac):
    received = np.asarray(received, dtype=float)
    total = float(np.sum(received))
    if total <= 0:
        return np.nan
    idx = list(top_indices(received, top_frac))
    return float(np.sum(received[idx]) / total)


def load_test_labels(test_csv):
    df = pd.read_csv(test_csv)
    if "neq" not in df.columns or "sequence" not in df.columns:
        raise ValueError(f"{test_csv} must contain sequence and neq columns.")

    labels = {}
    for row in df.itertuples(index=False):
        sequence = getattr(row, "sequence")
        neq_values = ast.literal_eval(getattr(row, "neq"))
        labels[sequence] = {
            "neq": np.asarray(neq_values, dtype=float),
            "class": classify_neq(neq_values),
        }
    return labels


def safe_metric_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def protein_performance(record, labels_by_sequence):
    sequence = record["sequence"]
    if sequence not in labels_by_sequence:
        return None

    y_true = labels_by_sequence[sequence]["class"]
    neq = labels_by_sequence[sequence]["neq"]
    y_pred = np.asarray(record["neq_preds"], dtype=int)
    scores = np.asarray(record.get("flexible_scores", record["neq_preds"]), dtype=float)

    n = min(len(y_true), len(y_pred), len(scores))
    y_true = y_true[:n]
    neq = neq[:n]
    y_pred = y_pred[:n]
    scores = scores[:n]

    row = {
        "protein": record["name"],
        "n_residues": n,
        "neq_score_spearman": spearman(neq, scores),
    }

    if accuracy_score is None:
        row["accuracy"] = float(np.mean(y_true == y_pred)) if n else np.nan
        row["macro_f1"] = np.nan
        row["auroc"] = np.nan
        row["auprc"] = np.nan
        return row

    row["accuracy"] = safe_metric_float(accuracy_score(y_true, y_pred))
    row["macro_f1"] = safe_metric_float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    row["class0_precision"] = safe_metric_float(precision[0])
    row["class0_recall"] = safe_metric_float(recall[0])
    row["class0_f1"] = safe_metric_float(f1[0])
    row["class1_precision"] = safe_metric_float(precision[1])
    row["class1_recall"] = safe_metric_float(recall[1])
    row["class1_f1"] = safe_metric_float(f1[1])

    if len(np.unique(y_true)) == 2:
        row["auroc"] = safe_metric_float(roc_auc_score(y_true, scores))
        row["auprc"] = safe_metric_float(average_precision_score(y_true, scores))
    else:
        row["auroc"] = np.nan
        row["auprc"] = np.nan

    return row


def summarize_attention(record, diag_window, top_frac):
    attn = np.asarray(record["attention_weights"], dtype=float)
    sequence_len = len(record["sequence"])
    attn = attn[:sequence_len, :sequence_len]

    received = attn.sum(axis=0)
    diagonal_mass, long_range_mass = attention_masses(attn, diag_window)

    return {
        "protein": record["name"],
        "sequence": record["sequence"],
        "n_residues": sequence_len,
        "received": received,
        "top_hubs": top_indices(received, top_frac),
        "normalized_row_entropy": row_entropy(attn),
        "diagonal_mass": diagonal_mass,
        "long_range_mass": long_range_mass,
        "column_hub_top10_mass": hub_concentration(received, top_frac),
    }


def aggregate_mean_std(df, group_cols, value_cols):
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols)[value_cols].agg(["mean", "std", "count"])
    out.columns = ["_".join(c).strip("_") for c in out.columns]
    return out.reset_index()


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    manifest_path = result_root / "manifest.tsv"
    analysis_dir = result_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    test_csv = Path(args.test_csv) if args.test_csv else result_root / "test_data.csv"
    labels_by_sequence = load_test_labels(test_csv)

    manifest = pd.read_csv(manifest_path, sep="\t")
    run_features = {}
    per_run_rows = []
    perf_rows = []

    for run in manifest.itertuples(index=False):
        attention_path = Path(run.attention_json)
        if not attention_path.exists():
            print(f"[skip] missing attention JSON: {attention_path}")
            continue

        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())

        for record in records:
            summary = summarize_attention(record, args.diag_window, args.top_frac)
            key = (run.condition, int(run.seed), summary["protein"])
            run_features[key] = summary

            per_run_rows.append({
                "condition": run.condition,
                "seed": int(run.seed),
                "architecture": run.architecture,
                "esm_model": run.esm_model,
                "protein": summary["protein"],
                "n_residues": summary["n_residues"],
                "normalized_row_entropy": summary["normalized_row_entropy"],
                "diagonal_mass": summary["diagonal_mass"],
                "long_range_mass": summary["long_range_mass"],
                "column_hub_top10_mass": summary["column_hub_top10_mass"],
            })

            perf = protein_performance(record, labels_by_sequence)
            if perf is not None:
                perf.update({
                    "condition": run.condition,
                    "seed": int(run.seed),
                    "architecture": run.architecture,
                    "esm_model": run.esm_model,
                    "protein": summary["protein"],
                })
                perf_rows.append(perf)

    per_run_df = pd.DataFrame(per_run_rows)
    perf_df = pd.DataFrame(perf_rows)

    pair_rows = []
    by_protein = {}
    for key, val in run_features.items():
        condition, seed, protein = key
        by_protein.setdefault(protein, []).append((condition, seed, val))

    for protein, entries in by_protein.items():
        for a, b in itertools.combinations(entries, 2):
            cond_a, seed_a, feat_a = a
            cond_b, seed_b, feat_b = b
            if len(feat_a["received"]) != len(feat_b["received"]):
                continue
            relation = "within_condition" if cond_a == cond_b else "between_condition"
            pair_rows.append({
                "protein": protein,
                "condition_a": cond_a,
                "seed_a": seed_a,
                "condition_b": cond_b,
                "seed_b": seed_b,
                "relation": relation,
                "received_spearman": spearman(feat_a["received"], feat_b["received"]),
                "top10_jaccard": jaccard(feat_a["top_hubs"], feat_b["top_hubs"]),
                "normalized_row_entropy_absdiff": abs(
                    feat_a["normalized_row_entropy"] - feat_b["normalized_row_entropy"]
                ),
                "long_range_mass_absdiff": abs(
                    feat_a["long_range_mass"] - feat_b["long_range_mass"]
                ),
                "diagonal_mass_absdiff": abs(
                    feat_a["diagonal_mass"] - feat_b["diagonal_mass"]
                ),
                "column_hub_top10_mass_absdiff": abs(
                    feat_a["column_hub_top10_mass"] - feat_b["column_hub_top10_mass"]
                ),
            })

    pair_df = pd.DataFrame(pair_rows)

    metric_cols = [
        "received_spearman",
        "top10_jaccard",
        "normalized_row_entropy_absdiff",
        "long_range_mass_absdiff",
        "diagonal_mass_absdiff",
        "column_hub_top10_mass_absdiff",
    ]
    within_df = aggregate_mean_std(
        pair_df[pair_df["relation"] == "within_condition"],
        ["condition_a"],
        metric_cols,
    ).rename(columns={"condition_a": "condition"})

    between = pair_df[pair_df["relation"] == "between_condition"].copy()
    if not between.empty:
        between["condition_pair"] = between.apply(
            lambda r: " vs ".join(sorted([r["condition_a"], r["condition_b"]])),
            axis=1,
        )
    between_df = aggregate_mean_std(between, ["condition_pair"], metric_cols)

    relative_rows = []
    if not between_df.empty and not within_df.empty:
        within_by_condition = within_df.set_index("condition").to_dict(orient="index")
        for row in between_df.itertuples(index=False):
            c1, c2 = row.condition_pair.split(" vs ")
            w1 = within_by_condition.get(c1, {})
            w2 = within_by_condition.get(c2, {})
            within_spear = np.nanmean([
                w1.get("received_spearman_mean", np.nan),
                w2.get("received_spearman_mean", np.nan),
            ])
            within_jacc = np.nanmean([
                w1.get("top10_jaccard_mean", np.nan),
                w2.get("top10_jaccard_mean", np.nan),
            ])
            relative_rows.append({
                "condition_pair": row.condition_pair,
                "between_received_spearman_mean": row.received_spearman_mean,
                "mean_within_received_spearman": within_spear,
                "received_spearman_between_minus_within": row.received_spearman_mean - within_spear,
                "between_top10_jaccard_mean": row.top10_jaccard_mean,
                "mean_within_top10_jaccard": within_jacc,
                "top10_jaccard_between_minus_within": row.top10_jaccard_mean - within_jacc,
            })
    relative_df = pd.DataFrame(relative_rows)

    perf_value_cols = [
        c for c in [
            "accuracy",
            "macro_f1",
            "class0_precision",
            "class0_recall",
            "class0_f1",
            "class1_precision",
            "class1_recall",
            "class1_f1",
            "auroc",
            "auprc",
            "neq_score_spearman",
        ]
        if c in perf_df.columns
    ]
    protein_perf_by_seed = aggregate_mean_std(
        perf_df,
        ["condition", "seed"],
        perf_value_cols,
    )
    protein_perf_across_seeds = aggregate_mean_std(
        protein_perf_by_seed,
        ["condition"],
        [f"{c}_mean" for c in perf_value_cols if f"{c}_mean" in protein_perf_by_seed.columns],
    )

    per_run_df.to_csv(analysis_dir / "per_run_attention_metrics.csv", index=False)
    perf_df.to_csv(analysis_dir / "protein_performance_by_run.csv", index=False)
    protein_perf_by_seed.to_csv(analysis_dir / "protein_performance_by_seed.csv", index=False)
    protein_perf_across_seeds.to_csv(analysis_dir / "protein_performance_across_seeds.csv", index=False)
    pair_df.to_csv(analysis_dir / "attention_pairwise_by_protein.csv", index=False)
    within_df.to_csv(analysis_dir / "within_condition_agreement.csv", index=False)
    between_df.to_csv(analysis_dir / "between_condition_agreement.csv", index=False)
    relative_df.to_csv(analysis_dir / "between_vs_within_agreement.csv", index=False)

    summary_lines = [
        f"Result root: {result_root}",
        f"Runs analyzed: {manifest.shape[0]}",
        f"Proteins with attention features: {len(by_protein)}",
        f"Pairwise comparisons: {pair_df.shape[0]}",
        "",
        "Key files:",
        "  protein_performance_across_seeds.csv",
        "  within_condition_agreement.csv",
        "  between_condition_agreement.csv",
        "  between_vs_within_agreement.csv",
        "  attention_pairwise_by_protein.csv",
    ]
    (analysis_dir / "analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
