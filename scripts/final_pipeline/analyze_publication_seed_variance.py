#!/usr/bin/env python3
"""Analyze prediction performance and attention stability across seeded runs.

The analysis keeps three distinct ideas separate:
  * attention-map morphology (row entropy and local/long-range mass),
  * one-dimensional key-residue profiles and their vertical bands, and
  * prediction performance, reported both per-protein and pooled by residue.

For BiLSTM self-attention, profile analysis includes raw received attention and
the positive/negative parts of the exact flexible-minus-rigid logit contribution.
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


PERFORMANCE_METRICS = [
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

PAIR_METRICS = [
    "profile_spearman",
    "top10_residue_jaccard",
    "band_mask_jaccard",
    "band_peak_match_precision",
    "band_peak_match_recall",
    "band_peak_match_f1",
    "band_peak_mean_distance",
    "band_count_absdiff",
    "both_no_bands",
    "either_no_bands",
    "normalized_row_entropy_absdiff",
    "long_range_mass_absdiff",
    "diagonal_mass_absdiff",
    "top10_profile_mass_absdiff",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze prediction and attention stability across seeded conditions."
    )
    parser.add_argument("--result_root", required=True)
    parser.add_argument(
        "--test_csv",
        default=None,
        help="Defaults to <result_root>/test_data.csv.",
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.10,
        help="Fraction used only for top-residue concentration/overlap. Default: 0.10.",
    )
    parser.add_argument("--diag_window", type=int, default=5)
    parser.add_argument(
        "--band_quantile",
        type=float,
        default=0.90,
        help="Within-profile quantile used in the band threshold. Default: 0.90.",
    )
    parser.add_argument(
        "--band_mad_multiplier",
        type=float,
        default=1.5,
        help="Robust threshold is median + multiplier * scaled MAD. Default: 1.5.",
    )
    parser.add_argument("--band_smooth_window", type=int, default=5)
    parser.add_argument("--band_min_width", type=int, default=2)
    parser.add_argument(
        "--band_max_gap",
        type=int,
        default=1,
        help="Bridge at most this many below-threshold residues inside a band.",
    )
    parser.add_argument(
        "--band_min_prominence_mad",
        type=float,
        default=0.0,
        help="Optional peak-above-threshold requirement in scaled-MAD units.",
    )
    parser.add_argument(
        "--peak_tolerance",
        type=int,
        default=2,
        help="Maximum residue distance for matching peaks across runs. Default: 2.",
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=123)
    args = parser.parse_args()

    if not 0 < args.top_frac <= 1:
        parser.error("--top_frac must be in (0, 1].")
    if not 0 < args.band_quantile < 1:
        parser.error("--band_quantile must be in (0, 1).")
    for name in ("band_smooth_window", "band_min_width"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1.")
    for name in ("band_max_gap", "peak_tolerance", "bootstrap_iterations"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative.")
    return args


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
    if len(x) == 0 or not np.isfinite(x).any() or np.nansum(np.clip(x, 0, None)) <= 0:
        return set()
    k = max(1, int(math.ceil(len(x) * frac)))
    finite = np.where(np.isfinite(x), x, -np.inf)
    return set(np.argsort(finite, kind="mergesort")[-k:].tolist())


def set_jaccard(a, b, both_empty=np.nan):
    a, b = set(a), set(b)
    if not a and not b:
        return both_empty
    return len(a & b) / len(a | b)


def smooth_vector(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) <= 1:
        return values.copy()
    window = min(int(window), len(values))
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def bridge_short_gaps(mask, max_gap):
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0:
        return mask
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
            continue
        start = i
        while i < len(mask) and not mask[i]:
            i += 1
        if start > 0 and i < len(mask) and i - start <= max_gap:
            mask[start:i] = True
    return mask


def contiguous_regions(mask):
    regions = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def detect_vertical_bands(
    values,
    quantile,
    mad_multiplier,
    smooth_window,
    min_width,
    max_gap,
    min_prominence_mad,
):
    """Detect broad high-profile intervals and select exactly one peak per interval."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Band profiles must be non-empty and finite.")
    if np.sum(values) <= 0:
        return [], set(), np.nan, smooth_vector(values, smooth_window), 0.0

    smoothed = smooth_vector(values, smooth_window)
    median = float(np.median(smoothed))
    scaled_mad = float(1.4826 * np.median(np.abs(smoothed - median)))
    quantile_threshold = float(np.quantile(smoothed, quantile))
    robust_threshold = median + mad_multiplier * scaled_mad
    threshold = max(quantile_threshold, robust_threshold)
    high = bridge_short_gaps(smoothed >= threshold, max_gap)

    bands = []
    mask_positions = set()
    for start, end in contiguous_regions(high):
        if end - start + 1 < min_width:
            continue
        region = np.arange(start, end + 1)
        peak = int(region[np.argmax(values[region])])
        smoothed_peak = int(region[np.argmax(smoothed[region])])
        prominence = float(smoothed[smoothed_peak] - threshold)
        if prominence < min_prominence_mad * scaled_mad:
            continue
        mask_positions.update(region.tolist())
        bands.append({
            "band_id": len(bands) + 1,
            "start_idx": int(start),
            "end_idx": int(end),
            "width": int(end - start + 1),
            "peak_idx": peak,
            "peak_value": float(values[peak]),
            "smoothed_peak_value": float(smoothed[smoothed_peak]),
            "smoothed_peak_idx": smoothed_peak,
            "prominence_above_threshold": prominence,
            "profile_mass": float(np.sum(values[region])),
        })
    return bands, mask_positions, threshold, smoothed, scaled_mad


def match_peaks(peaks_a, peaks_b, tolerance):
    """Max-cardinality one-to-one matching, then minimize total peak shift."""
    peaks_a = sorted(set(int(x) for x in peaks_a))
    peaks_b = sorted(set(int(x) for x in peaks_b))
    if not peaks_a and not peaks_b:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_distance": np.nan,
            "n_matches": 0,
            "both_empty": True,
        }

    n_a, n_b = len(peaks_a), len(peaks_b)
    counts = np.zeros((n_a + 1, n_b + 1), dtype=int)
    costs = np.zeros((n_a + 1, n_b + 1), dtype=float)
    actions = np.full((n_a, n_b), "a", dtype="U1")
    for i in range(n_a - 1, -1, -1):
        for j in range(n_b - 1, -1, -1):
            candidates = [
                (counts[i + 1, j], costs[i + 1, j], "a"),
                (counts[i, j + 1], costs[i, j + 1], "b"),
            ]
            distance = abs(peaks_a[i] - peaks_b[j])
            if distance <= tolerance:
                candidates.append(
                    (1 + counts[i + 1, j + 1], distance + costs[i + 1, j + 1], "m")
                )
            best_count = max(candidate[0] for candidate in candidates)
            best = min(
                (candidate for candidate in candidates if candidate[0] == best_count),
                key=lambda candidate: (candidate[1], candidate[2] != "m"),
            )
            counts[i, j], costs[i, j], actions[i, j] = best

    i = j = 0
    distances = []
    while i < n_a and j < n_b:
        action = actions[i, j]
        if action == "m":
            distances.append(abs(peaks_a[i] - peaks_b[j]))
            i += 1
            j += 1
        elif action == "a":
            i += 1
        else:
            j += 1

    matched = len(distances)
    precision = matched / len(peaks_a) if peaks_a else 0.0
    recall = matched / len(peaks_b) if peaks_b else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance": float(np.mean(distances)) if distances else np.nan,
        "n_matches": matched,
        "both_empty": False,
    }


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
    return float(np.sum(attn[local]) / total), float(np.sum(attn[~local]) / total)


def top_concentration(values, top_frac):
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    indices = top_indices(values, top_frac)
    if total <= 0 or not indices:
        return np.nan
    return float(np.sum(values[list(indices)]) / total)


def load_test_labels(test_csv):
    df = pd.read_csv(test_csv)
    if not {"neq", "sequence"}.issubset(df.columns):
        raise ValueError(f"{test_csv} must contain sequence and neq columns.")
    by_name, by_sequence_lists = {}, {}
    for row in df.itertuples(index=False):
        sequence = str(row.sequence)
        neq = np.asarray(ast.literal_eval(row.neq), dtype=float)
        if len(sequence) != len(neq):
            raise ValueError(f"{getattr(row, 'name', '<unknown>')}: sequence/Neq mismatch.")
        payload = {"neq": neq, "class": classify_neq(neq), "sequence": sequence}
        name = getattr(row, "name", None)
        if name is not None:
            if name in by_name:
                raise ValueError(f"Duplicate protein name in {test_csv}: {name}")
            by_name[name] = payload
        by_sequence_lists.setdefault(sequence, []).append(payload)
    by_sequence = {s: rows[0] for s, rows in by_sequence_lists.items() if len(rows) == 1}
    return {"by_name": by_name, "by_sequence": by_sequence}


def labels_for_record(record, label_maps):
    labels = label_maps["by_name"].get(record.get("name"))
    if labels is None:
        labels = label_maps["by_sequence"].get(record["sequence"])
    return labels


def safe_metric_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def calculate_performance(y_true, y_pred, scores, neq):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)
    neq = np.asarray(neq, dtype=float)
    row = {
        "n_residues": int(len(y_true)),
        "neq_score_spearman": spearman(neq, scores),
    }
    if accuracy_score is None:
        row.update({
            "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else np.nan,
            "macro_f1": np.nan,
            "auroc": np.nan,
            "auprc": np.nan,
        })
        return row

    row["accuracy"] = safe_metric_float(accuracy_score(y_true, y_pred))
    row["macro_f1"] = safe_metric_float(
        f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    for idx in (0, 1):
        row[f"class{idx}_precision"] = safe_metric_float(precision[idx])
        row[f"class{idx}_recall"] = safe_metric_float(recall[idx])
        row[f"class{idx}_f1"] = safe_metric_float(f1[idx])
    if len(np.unique(y_true)) == 2:
        row["auroc"] = safe_metric_float(roc_auc_score(y_true, scores))
        row["auprc"] = safe_metric_float(average_precision_score(y_true, scores))
    else:
        row["auroc"] = np.nan
        row["auprc"] = np.nan
    return row


def protein_performance(record, label_maps):
    labels = labels_for_record(record, label_maps)
    if labels is None:
        return None, None
    if "flexible_scores" not in record:
        raise ValueError(
            f"{record.get('name', '<unknown>')}: missing flexible_scores; re-extract attention."
        )
    sequence = record["sequence"]
    arrays = {
        "y_true": labels["class"],
        "neq": labels["neq"],
        "y_pred": np.asarray(record["neq_preds"], dtype=int),
        "scores": np.asarray(record["flexible_scores"], dtype=float),
    }
    lengths = {name: len(value) for name, value in arrays.items()}
    if any(length != len(sequence) for length in lengths.values()):
        raise ValueError(
            f"{record.get('name', '<unknown>')}: sequence={len(sequence)}, {lengths}."
        )
    row = calculate_performance(**arrays)
    row["protein"] = record["name"]
    return row, arrays


def resolve_path(raw_path, result_root):
    path = Path(str(raw_path))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path(__file__).resolve().parent / path, result_root / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def present_path(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def attention_sources_for_run(run, result_root):
    primary_path = resolve_path(run.attention_json, result_root)
    primary_source = (
        "bilstm_self_attention"
        if str(run.architecture) == "bilstm_attention"
        else "esm2_backbone_attention"
    )
    sources = [(primary_source, primary_path, True)]
    backbone_value = getattr(run, "backbone_attention_json", None)
    if present_path(backbone_value):
        backbone_path = resolve_path(backbone_value, result_root)
        if backbone_path != primary_path:
            sources.append(("esm2_backbone_attention", backbone_path, False))
    return sources


def load_contribution_archive(run, result_root):
    value = getattr(run, "logit_contributions_npz", None)
    if not present_path(value):
        return None, {}
    path = resolve_path(value, result_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing contribution archive: {path}")
    archive = np.load(path, allow_pickle=False)
    names = archive["__protein_names__"].astype(str).tolist()
    keys = archive["__matrix_keys__"].astype(str).tolist()
    if len(names) != len(keys) or len(names) != len(set(names)):
        archive.close()
        raise ValueError(f"Invalid contribution index in {path}")
    return archive, dict(zip(names, keys))


def summarize_attention_matrix(record, diag_window):
    attn = np.asarray(record["attention_weights"], dtype=float)
    n = len(record["sequence"])
    if attn.shape != (n, n):
        raise ValueError(
            f"{record.get('name', '<unknown>')}: attention {attn.shape} != {(n, n)}."
        )
    diagonal_mass, long_range_mass = attention_masses(attn, diag_window)
    return attn, {
        "normalized_row_entropy": row_entropy(attn),
        "diagonal_mass": diagonal_mass,
        "long_range_mass": long_range_mass,
    }


def summarize_profile(values, args):
    values = np.asarray(values, dtype=float)
    bands, band_mask, threshold, smoothed, scaled_mad = detect_vertical_bands(
        values,
        args.band_quantile,
        args.band_mad_multiplier,
        args.band_smooth_window,
        args.band_min_width,
        args.band_max_gap,
        args.band_min_prominence_mad,
    )
    total = float(np.sum(values))
    return {
        "values": values,
        "top_residues": top_indices(values, args.top_frac),
        "top10_profile_mass": top_concentration(values, args.top_frac),
        "bands": bands,
        "band_mask": band_mask,
        "band_peaks": {band["peak_idx"] for band in bands},
        "band_count": len(bands),
        "band_residue_fraction": len(band_mask) / len(values) if len(values) else np.nan,
        "band_mass_fraction": (
            float(np.sum(values[list(band_mask)]) / total)
            if total > 0 and band_mask else 0.0 if total > 0 else np.nan
        ),
        "band_threshold": threshold,
        "band_scaled_mad": scaled_mad,
        "smoothed": smoothed,
    }


def profile_pair_metrics(a, b, peak_tolerance):
    peak_match = match_peaks(a["band_peaks"], b["band_peaks"], peak_tolerance)
    return {
        "profile_spearman": spearman(a["values"], b["values"]),
        "top10_residue_jaccard": set_jaccard(a["top_residues"], b["top_residues"]),
        "band_mask_jaccard": set_jaccard(a["band_mask"], b["band_mask"], both_empty=1.0),
        "band_peak_match_precision": peak_match["precision"],
        "band_peak_match_recall": peak_match["recall"],
        "band_peak_match_f1": peak_match["f1"],
        "band_peak_mean_distance": peak_match["mean_distance"],
        "band_peak_n_matches": peak_match["n_matches"],
        "both_no_bands": peak_match["both_empty"],
        "either_no_bands": not a["band_peaks"] or not b["band_peaks"],
        "band_count_absdiff": abs(a["band_count"] - b["band_count"]),
        "top10_profile_mass_absdiff": abs(
            a["top10_profile_mass"] - b["top10_profile_mass"]
        ),
    }


def bootstrap_mean_ci(values, iterations, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or iterations == 0:
        value = float(values[0] if len(values) == 1 else np.mean(values))
        return value, value
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_pair_agreement(pair_df, relation, group_cols, metrics, args, rng):
    if pair_df.empty or "relation" not in pair_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    subset = pair_df[pair_df["relation"] == relation].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_protein = subset.groupby(group_cols + ["protein"], dropna=False)[metrics].mean().reset_index()
    pair_counts = (
        subset.groupby(group_cols + ["protein"], dropna=False)
        .size()
        .rename("n_seed_pair_observations")
        .reset_index()
    )
    per_protein = per_protein.merge(pair_counts, on=group_cols + ["protein"], how="left")

    summary_rows = []
    for key, group in per_protein.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_cols, key)}
        row["n_proteins"] = int(group["protein"].nunique())
        row["n_seed_pair_observations"] = int(group["n_seed_pair_observations"].sum())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{metric}_std_between_proteins"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            )
            row[f"{metric}_n_proteins"] = int(len(finite))
            low, high = bootstrap_mean_ci(finite, args.bootstrap_iterations, rng)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), per_protein


def summarize_performance_by_seed(perf_df):
    rows = []
    for (condition, seed), group in perf_df.groupby(["condition", "seed"], sort=False):
        row = {
            "condition": condition,
            "seed": int(seed),
            "n_proteins": int(group["protein"].nunique()),
            "n_residues": int(group["n_residues"].sum()),
        }
        for metric in PERFORMANCE_METRICS:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{metric}_std_between_proteins"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            )
            row[f"{metric}_n_proteins"] = int(len(finite))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_seeds(df, metric_columns):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for condition, group in df.groupby("condition", sort=False):
        row = {"condition": condition, "n_seeds": int(group["seed"].nunique())}
        for metric in metric_columns:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_across_seed_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{metric}_across_seed_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_relative_agreement(between_df, within_df):
    if between_df.empty or within_df.empty:
        return pd.DataFrame()
    within_lookup = {
        (row.attention_source, row.profile, row.condition): row
        for row in within_df.itertuples(index=False)
    }
    rows = []
    metrics = [
        "profile_spearman",
        "top10_residue_jaccard",
        "band_mask_jaccard",
        "band_peak_match_f1",
    ]
    for between in between_df.itertuples(index=False):
        c1, c2 = between.condition_pair.split(" vs ")
        w1 = within_lookup.get((between.attention_source, between.profile, c1))
        w2 = within_lookup.get((between.attention_source, between.profile, c2))
        row = {
            "attention_source": between.attention_source,
            "profile": between.profile,
            "condition_pair": between.condition_pair,
        }
        for metric in metrics:
            between_value = getattr(between, f"{metric}_mean")
            within_values = [
                getattr(item, f"{metric}_mean")
                for item in (w1, w2)
                if item is not None
            ]
            within_mean = float(np.nanmean(within_values)) if within_values else np.nan
            row[f"between_{metric}_mean"] = between_value
            row[f"mean_within_{metric}"] = within_mean
            row[f"{metric}_between_minus_within"] = between_value - within_mean
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    manifest_path = result_root / "manifest.tsv"
    analysis_dir = result_root / "analysis"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    test_csv = Path(args.test_csv) if args.test_csv else result_root / "test_data.csv"
    label_maps = load_test_labels(test_csv)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    required_manifest = {"condition", "seed", "architecture", "esm_model", "attention_json"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"Manifest lacks {sorted(required_manifest - set(manifest.columns))}")

    run_features = {}
    matrix_rows = []
    profile_rows = []
    band_rows = []
    protein_perf_rows = []
    pooled_arrays = {}
    loaded_run_sources = set()

    for run in manifest.itertuples(index=False):
        contribution_archive, contribution_keys = load_contribution_archive(run, result_root)
        try:
            for attention_source, attention_path, is_primary in attention_sources_for_run(run, result_root):
                if not attention_path.exists():
                    print(f"[skip] missing attention JSON: {attention_path}")
                    continue
                source_key = (run.condition, int(run.seed), attention_source)
                if source_key in loaded_run_sources:
                    raise ValueError(f"Duplicate run/source in manifest: {source_key}")
                loaded_run_sources.add(source_key)
                print(
                    f"[load] {run.condition} seed={run.seed} "
                    f"source={attention_source}: {attention_path}"
                )
                records = json.loads(attention_path.read_text())
                names = [record["name"] for record in records]
                if len(names) != len(set(names)):
                    raise ValueError(f"Duplicate protein names in {attention_path}")

                for record in records:
                    protein = record["name"]
                    n = len(record["sequence"])
                    attn, morphology = summarize_attention_matrix(record, args.diag_window)
                    matrix_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "architecture": run.architecture,
                        "esm_model": run.esm_model,
                        "attention_source": attention_source,
                        "protein": protein,
                        "n_residues": n,
                        "top10_received_mass": top_concentration(
                            attn.sum(axis=0), args.top_frac
                        ),
                        **morphology,
                    })

                    profiles = {"raw_received": attn.sum(axis=0)}
                    if attention_source == "bilstm_self_attention" and contribution_archive is not None:
                        if protein not in contribution_keys:
                            raise ValueError(f"{protein} missing from contribution archive.")
                        contribution = np.asarray(
                            contribution_archive[contribution_keys[protein]], dtype=float
                        )
                        if contribution.shape != (n, n):
                            raise ValueError(
                                f"{protein}: contribution {contribution.shape} != {(n, n)}"
                            )
                        profiles["flexible_support"] = np.clip(contribution, 0, None).sum(axis=0)
                        profiles["rigid_support"] = np.clip(-contribution, 0, None).sum(axis=0)

                    for profile_name, values in profiles.items():
                        summary = summarize_profile(values, args)
                        feature_key = (
                            run.condition,
                            int(run.seed),
                            attention_source,
                            profile_name,
                            protein,
                        )
                        if feature_key in run_features:
                            raise ValueError(f"Duplicate profile feature: {feature_key}")
                        run_features[feature_key] = {**summary, **morphology}
                        profile_rows.append({
                            "condition": run.condition,
                            "seed": int(run.seed),
                            "architecture": run.architecture,
                            "esm_model": run.esm_model,
                            "attention_source": attention_source,
                            "profile": profile_name,
                            "protein": protein,
                            "n_residues": n,
                            "profile_total": float(np.sum(values)),
                            "top10_profile_mass": summary["top10_profile_mass"],
                            "band_count": summary["band_count"],
                            "band_residue_fraction": summary["band_residue_fraction"],
                            "band_mass_fraction": summary["band_mass_fraction"],
                            "band_threshold": summary["band_threshold"],
                            "band_scaled_mad": summary["band_scaled_mad"],
                        })
                        for band in summary["bands"]:
                            band_rows.append({
                                "condition": run.condition,
                                "seed": int(run.seed),
                                "attention_source": attention_source,
                                "profile": profile_name,
                                "protein": protein,
                                "start_position_1based": band["start_idx"] + 1,
                                "end_position_1based": band["end_idx"] + 1,
                                "peak_position_1based": band["peak_idx"] + 1,
                                **band,
                            })

                    if is_primary:
                        perf, arrays = protein_performance(record, label_maps)
                        if perf is not None:
                            perf.update({
                                "condition": run.condition,
                                "seed": int(run.seed),
                                "architecture": run.architecture,
                                "esm_model": run.esm_model,
                            })
                            protein_perf_rows.append(perf)
                            cache = pooled_arrays.setdefault(
                                (run.condition, int(run.seed)),
                                {"y_true": [], "y_pred": [], "scores": [], "neq": []},
                            )
                            for name, values in arrays.items():
                                cache[name].append(values)
                del records
        finally:
            if contribution_archive is not None:
                contribution_archive.close()

    matrix_df = pd.DataFrame(matrix_rows)
    profile_df = pd.DataFrame(profile_rows)
    bands_df = pd.DataFrame(band_rows)
    protein_perf_df = pd.DataFrame(protein_perf_rows)

    pair_rows = []
    by_protein_source_profile = {}
    for key, feature in run_features.items():
        condition, seed, source, profile, protein = key
        by_protein_source_profile.setdefault((protein, source, profile), []).append(
            (condition, seed, feature)
        )
    for (protein, source, profile), entries in by_protein_source_profile.items():
        for a, b in itertools.combinations(entries, 2):
            cond_a, seed_a, feat_a = a
            cond_b, seed_b, feat_b = b
            if (cond_b, seed_b) < (cond_a, seed_a):
                cond_a, seed_a, feat_a, cond_b, seed_b, feat_b = (
                    cond_b, seed_b, feat_b, cond_a, seed_a, feat_a
                )
            if len(feat_a["values"]) != len(feat_b["values"]):
                raise ValueError(f"{protein}: profile length mismatch across runs.")
            relation = "within_condition" if cond_a == cond_b else "between_condition"
            row = {
                "protein": protein,
                "attention_source": source,
                "profile": profile,
                "condition_a": cond_a,
                "seed_a": seed_a,
                "condition_b": cond_b,
                "seed_b": seed_b,
                "relation": relation,
                **profile_pair_metrics(feat_a, feat_b, args.peak_tolerance),
            }
            for metric in ("normalized_row_entropy", "long_range_mass", "diagonal_mass"):
                row[f"{metric}_absdiff"] = (
                    abs(feat_a[metric] - feat_b[metric])
                    if profile == "raw_received" else np.nan
                )
            pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["condition_pair"] = pair_df.apply(
            lambda row: " vs ".join(sorted([row.condition_a, row.condition_b])), axis=1
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    within_df, within_protein_df = summarize_pair_agreement(
        pair_df,
        "within_condition",
        ["attention_source", "profile", "condition_a"],
        PAIR_METRICS,
        args,
        rng,
    )
    if not within_df.empty:
        within_df = within_df.rename(columns={"condition_a": "condition"})
        within_protein_df = within_protein_df.rename(columns={"condition_a": "condition"})
    between_df, between_protein_df = summarize_pair_agreement(
        pair_df,
        "between_condition",
        ["attention_source", "profile", "condition_pair"],
        PAIR_METRICS,
        args,
        rng,
    )
    relative_df = build_relative_agreement(between_df, within_df)

    protein_macro_by_seed = summarize_performance_by_seed(protein_perf_df)
    protein_macro_across_seeds = summarize_across_seeds(
        protein_macro_by_seed,
        [f"{metric}_mean" for metric in PERFORMANCE_METRICS],
    )

    pooled_rows = []
    for (condition, seed), arrays in pooled_arrays.items():
        concatenated = {name: np.concatenate(parts) for name, parts in arrays.items()}
        row = calculate_performance(**concatenated)
        row.update({
            "condition": condition,
            "seed": seed,
            "n_proteins": int(
                protein_perf_df[
                    (protein_perf_df["condition"] == condition)
                    & (protein_perf_df["seed"] == seed)
                ]["protein"].nunique()
            ),
        })
        pooled_rows.append(row)
    pooled_by_seed = pd.DataFrame(pooled_rows)
    pooled_across_seeds = summarize_across_seeds(pooled_by_seed, PERFORMANCE_METRICS)

    matrix_df.to_csv(analysis_dir / "per_run_attention_metrics.csv", index=False)
    profile_df.to_csv(analysis_dir / "per_run_attention_profile_metrics.csv", index=False)
    bands_df.to_csv(analysis_dir / "attention_bands_by_run.csv", index=False)
    protein_perf_df.to_csv(analysis_dir / "protein_performance_by_run.csv", index=False)
    protein_macro_by_seed.to_csv(analysis_dir / "protein_macro_performance_by_seed.csv", index=False)
    protein_macro_across_seeds.to_csv(
        analysis_dir / "protein_macro_performance_across_seeds.csv", index=False
    )
    pooled_by_seed.to_csv(analysis_dir / "pooled_residue_performance_by_seed.csv", index=False)
    pooled_across_seeds.to_csv(
        analysis_dir / "pooled_residue_performance_across_seeds.csv", index=False
    )
    pair_df.to_csv(analysis_dir / "attention_pairwise_by_protein.csv", index=False)
    within_protein_df.to_csv(
        analysis_dir / "within_condition_agreement_by_protein.csv", index=False
    )
    between_protein_df.to_csv(
        analysis_dir / "between_condition_agreement_by_protein.csv", index=False
    )
    within_df.to_csv(analysis_dir / "within_condition_agreement.csv", index=False)
    between_df.to_csv(analysis_dir / "between_condition_agreement.csv", index=False)
    relative_df.to_csv(analysis_dir / "between_vs_within_agreement.csv", index=False)

    parameters = {
        "top_frac": args.top_frac,
        "diag_window": args.diag_window,
        "band_quantile": args.band_quantile,
        "band_mad_multiplier": args.band_mad_multiplier,
        "band_smooth_window": args.band_smooth_window,
        "band_min_width": args.band_min_width,
        "band_max_gap": args.band_max_gap,
        "band_min_prominence_mad": args.band_min_prominence_mad,
        "peak_tolerance": args.peak_tolerance,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (analysis_dir / "analysis_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n"
    )
    summary_lines = [
        f"Result root: {result_root}",
        f"Manifest runs: {len(manifest)}",
        f"Loaded run/source combinations: {len(loaded_run_sources)}",
        f"Proteins with profiles: {len({key[-1] for key in run_features})}",
        f"Pairwise profile comparisons: {len(pair_df)}",
        f"Detected bands: {len(bands_df)}",
        "",
        "Agreement summaries use one mean per protein; 95% confidence intervals",
        "bootstrap whole proteins so all seed-pair observations stay together.",
        "",
        "Key files:",
        "  attention_bands_by_run.csv",
        "  per_run_attention_profile_metrics.csv",
        "  within_condition_agreement.csv",
        "  between_condition_agreement.csv",
        "  pooled_residue_performance_across_seeds.csv",
        "  protein_macro_performance_across_seeds.csv",
    ]
    (analysis_dir / "analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
