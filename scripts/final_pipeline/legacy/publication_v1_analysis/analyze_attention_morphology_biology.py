#!/usr/bin/env python3
"""
Quantify attention morphology and simple biological overlaps for comparable
ESMfluc runs.

This script is meant to follow analyze_publication_seed_variance.py. It reads
the same manifest.tsv and attention.json files, then measures why the maps look
"beehive-like": low-rank structure, column hubs, row similarity, long-range
mass, vertical bands, and overlap of those bands with secondary structure and
flexible Neq residues.

Example:
  python analyze_attention_morphology_biology.py \
    --result_root results/publication_comparable_v1 \
    --test_csv ../../data/test_data_with_names.csv \
    --ss_csv ../../data/test_data_nsp3.csv
"""

import argparse
import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_attention_row_modes import default_analysis_dir, resolve_manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SVD, hub/band morphology, and biological overlap in attention maps."
    )
    parser.add_argument(
        "--result_root",
        required=True,
        help="Result root containing manifest.tsv, e.g. results/publication_comparable_v1.",
    )
    parser.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Defaults to result_root/manifest.tsv. "
             "Use manifest_attention_sources.tsv for the 30-attention source view.",
    )
    parser.add_argument(
        "--test_csv",
        default=None,
        help="CSV with sequence, neq, and optionally name. Defaults to result_root/test_data_with_names.csv, then test_data.csv.",
    )
    parser.add_argument(
        "--ss_csv",
        default=None,
        help="Optional NetSurfP CSV with id and q3 columns, e.g. ../../data/test_data_nsp3.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to result_root/analysis_morphology.",
    )
    parser.add_argument(
        "--pipeline_dir",
        default=None,
        help="Directory used when manifest paths are relative. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--diag_windows",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="Diagonal windows for local/long-range mass. Default: 3 5 10.",
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.10,
        help="Top received-attention fraction for hub/band mass. Default: 0.10.",
    )
    parser.add_argument(
        "--band_quantile",
        type=float,
        default=0.90,
        help="Received-attention quantile for vertical band detection. Default: 0.90.",
    )
    parser.add_argument(
        "--band_z",
        type=float,
        default=1.0,
        help="Received-attention z-score threshold for vertical band detection. Default: 1.0.",
    )
    parser.add_argument(
        "--min_band_width",
        type=int,
        default=2,
        help="Minimum contiguous residues for a vertical band. Default: 2.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Moving-average window for received attention before band detection. Default: 5.",
    )
    parser.add_argument(
        "--svd_top_k",
        type=int,
        default=20,
        help="Number of singular value fractions to save per protein/run. Default: 20.",
    )
    parser.add_argument(
        "--stable_top_n",
        type=int,
        default=30,
        help="Number of stable protein examples to report per condition. Default: 30.",
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


def smooth_vector(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def top_indices(x, frac):
    x = np.asarray(x, dtype=float)
    k = max(1, int(math.ceil(len(x) * frac)))
    return set(np.argsort(x)[-k:].tolist())


def jaccard(a, b):
    union = set(a) | set(b)
    if not union:
        return np.nan
    return len(set(a) & set(b)) / len(union)


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    total = np.sum(x)
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x) / (n * total)) - ((n + 1.0) / n))


def entropy_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)))


def row_entropy(attn):
    eps = 1e-12
    n = attn.shape[1]
    if n <= 1:
        return np.nan
    row_sums = attn.sum(axis=1, keepdims=True)
    probs = np.divide(attn, row_sums, out=np.zeros_like(attn), where=row_sums > 0)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(np.nanmean(entropy / np.log(n)))


def attention_masses(attn, window):
    n = attn.shape[0]
    idx = np.arange(n)
    local = np.abs(idx[:, None] - idx[None, :]) <= window
    total = float(np.sum(attn))
    if total <= 0:
        return np.nan, np.nan
    diagonal_mass = float(np.sum(attn[local]) / total)
    long_range_mass = float(np.sum(attn[~local]) / total)
    return diagonal_mass, long_range_mass


def effective_rank_and_svd(attn, top_k):
    # Row-normalized attention can have a large first singular value; we report
    # both singular-value mass and energy concentration.
    singular_values = np.linalg.svd(attn, compute_uv=False)
    total = float(np.sum(singular_values))
    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan, []
    p = singular_values / total
    effective_rank = float(np.exp(entropy_from_probs(p)))
    energy = singular_values ** 2
    energy_total = float(np.sum(energy))
    top1_energy = float(energy[0] / energy_total) if energy_total > 0 and len(energy) else np.nan
    top3_energy = float(np.sum(energy[:3]) / energy_total) if energy_total > 0 else np.nan
    top5_energy = float(np.sum(energy[:5]) / energy_total) if energy_total > 0 else np.nan
    return effective_rank, top1_energy, top3_energy, top5_energy, p[:top_k].tolist()


def mean_offdiag_cosine(rows, center=False):
    x = np.asarray(rows, dtype=float)
    if x.shape[0] < 2:
        return np.nan
    if center:
        x = x - np.nanmean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    ok = norms[:, 0] > 0
    x = x[ok]
    norms = norms[ok]
    if x.shape[0] < 2:
        return np.nan
    x = x / norms
    summed = np.sum(x, axis=0)
    n = x.shape[0]
    offdiag_sum = float(np.dot(summed, summed) - n)
    return offdiag_sum / (n * (n - 1))


def contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    regions = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def detect_bands(received, quantile=0.90, z_thresh=1.0, min_width=2, smooth_window=5):
    received = np.asarray(received, dtype=float)
    smoothed = smooth_vector(received, smooth_window)
    mu = float(np.mean(smoothed))
    sigma = float(np.std(smoothed))
    q_threshold = float(np.quantile(smoothed, quantile))
    z_threshold = mu + z_thresh * sigma
    threshold = max(q_threshold, z_threshold)
    regions = [
        region for region in contiguous_regions(smoothed >= threshold)
        if (region[1] - region[0] + 1) >= min_width
    ]
    band_positions = set()
    for start, end in regions:
        band_positions.update(range(start, end + 1))
    return regions, band_positions, threshold, smoothed


def fraction(values, target):
    values = list(values)
    if not values:
        return np.nan
    return sum(v == target for v in values) / len(values)


def enrichment(selected_mask, annotation_mask):
    selected_mask = np.asarray(selected_mask, dtype=bool)
    annotation_mask = np.asarray(annotation_mask, dtype=bool)
    n = min(len(selected_mask), len(annotation_mask))
    selected_mask = selected_mask[:n]
    annotation_mask = annotation_mask[:n]
    if n == 0 or selected_mask.sum() == 0:
        return np.nan, np.nan, np.nan
    selected_fraction = float(annotation_mask[selected_mask].mean())
    background_fraction = float(annotation_mask.mean())
    if background_fraction == 0:
        ratio = np.nan
    else:
        ratio = selected_fraction / background_fraction
    return selected_fraction, background_fraction, ratio


def load_neq_maps(test_csv):
    if test_csv is None:
        return {}, {}
    path = Path(test_csv)
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    by_name = {}
    by_sequence = {}
    for _, row in df.iterrows():
        seq = row["sequence"]
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        entry = {
            "neq": neq,
            "flexible": classify_neq(neq),
        }
        by_sequence[seq] = entry
        if "name" in df.columns:
            by_name[str(row["name"])] = entry
    return by_name, by_sequence


def load_ss_map(ss_csv):
    if ss_csv is None:
        return {}
    path = Path(ss_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    columns = {c.strip(): c for c in df.columns}
    id_col = columns.get("id")
    q3_col = columns.get("q3")
    if id_col is None or q3_col is None:
        raise ValueError(f"{ss_csv} must contain id and q3 columns.")
    ss_map = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col]).lstrip(">")
        ss_map.setdefault(seq_id, []).append(str(row[q3_col]).strip())
    return ss_map


def resolve_existing_path(path_value, result_root, pipeline_dir):
    path = Path(str(path_value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            result_root / path,
            result_root.parent / path,
            pipeline_dir / path,
            Path.cwd() / path,
            path,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_neq_entry(record, neq_by_name, neq_by_sequence):
    return neq_by_name.get(record["name"]) or neq_by_sequence.get(record["sequence"])


def summarize_record(record, condition, seed, architecture, esm_model, args, neq_by_name, neq_by_sequence, ss_map):
    sequence = record["sequence"]
    protein = record["name"]
    n = len(sequence)
    attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
    received = np.sum(attn, axis=0)
    received_total = float(np.sum(received))
    top_hubs = top_indices(received, args.top_frac)
    top_hub_mass = float(np.sum(received[list(top_hubs)]) / received_total) if received_total > 0 else np.nan

    effective_rank, top1_energy, top3_energy, top5_energy, sv_fraction = effective_rank_and_svd(
        attn, args.svd_top_k
    )
    regions, band_positions, band_threshold, smoothed_received = detect_bands(
        received,
        quantile=args.band_quantile,
        z_thresh=args.band_z,
        min_width=args.min_band_width,
        smooth_window=args.smooth_window,
    )
    band_mask = np.zeros(n, dtype=bool)
    if band_positions:
        band_mask[list(band_positions)] = True
    widths = [end - start + 1 for start, end in regions]
    band_mass = float(np.sum(received[band_mask]) / received_total) if received_total > 0 and band_mask.any() else 0.0

    neq_entry = get_neq_entry(record, neq_by_name, neq_by_sequence)
    flexible = neq_entry["flexible"][:n] if neq_entry is not None else None
    neq = neq_entry["neq"][:n] if neq_entry is not None else None

    ss = ss_map.get(protein)
    if ss is not None:
        ss = ss[:n]
        if len(ss) != n:
            ss = None

    row = {
        "condition": condition,
        "seed": int(seed),
        "architecture": architecture,
        "esm_model": esm_model,
        "protein": protein,
        "n_residues": n,
        "effective_rank": effective_rank,
        "relative_effective_rank": effective_rank / n if n else np.nan,
        "svd_top1_energy_frac": top1_energy,
        "svd_top3_energy_frac": top3_energy,
        "svd_top5_energy_frac": top5_energy,
        "column_hub_top10_mass": top_hub_mass,
        "column_hub_gini": gini(received),
        "received_max_over_mean": float(np.max(received) / np.mean(received)) if np.mean(received) > 0 else np.nan,
        "normalized_row_entropy": row_entropy(attn),
        "row_cosine_similarity_mean": mean_offdiag_cosine(attn, center=False),
        "row_pearson_similarity_mean": mean_offdiag_cosine(attn, center=True),
        "band_count": len(regions),
        "band_width_mean": float(np.mean(widths)) if widths else 0.0,
        "band_width_max": int(max(widths)) if widths else 0,
        "band_residue_fraction": float(np.mean(band_mask)) if n else np.nan,
        "band_attention_mass": band_mass,
        "band_threshold": band_threshold,
        "band_regions_1based": ";".join(f"{start + 1}-{end + 1}" for start, end in regions),
    }

    for window in args.diag_windows:
        diag_mass, long_mass = attention_masses(attn, window)
        row[f"diagonal_mass_w{window}"] = diag_mass
        row[f"long_range_mass_w{window}"] = long_mass

    if flexible is not None:
        selected, background, ratio = enrichment(band_mask, flexible == 1)
        row["band_flexible_fraction"] = selected
        row["protein_flexible_fraction"] = background
        row["band_flexible_enrichment"] = ratio
        top_mask = np.zeros(n, dtype=bool)
        top_mask[list(top_hubs)] = True
        selected, background, ratio = enrichment(top_mask, flexible == 1)
        row["top_hub_flexible_fraction"] = selected
        row["top_hub_flexible_enrichment"] = ratio
        row["received_vs_neq_spearman"] = spearman(received[:len(neq)], neq) if neq is not None else np.nan
    else:
        row["band_flexible_fraction"] = np.nan
        row["protein_flexible_fraction"] = np.nan
        row["band_flexible_enrichment"] = np.nan
        row["top_hub_flexible_fraction"] = np.nan
        row["top_hub_flexible_enrichment"] = np.nan
        row["received_vs_neq_spearman"] = np.nan

    if ss is not None:
        ss_array = np.asarray(ss)
        for label in ["C", "H", "E"]:
            selected, background, ratio = enrichment(band_mask, ss_array == label)
            row[f"band_ss_{label}_fraction"] = selected
            row[f"protein_ss_{label}_fraction"] = background
            row[f"band_ss_{label}_enrichment"] = ratio
            top_mask = np.zeros(n, dtype=bool)
            top_mask[list(top_hubs)] = True
            selected, background, ratio = enrichment(top_mask, ss_array == label)
            row[f"top_hub_ss_{label}_fraction"] = selected
            row[f"top_hub_ss_{label}_enrichment"] = ratio
        row["has_ss"] = True
    else:
        for label in ["C", "H", "E"]:
            row[f"band_ss_{label}_fraction"] = np.nan
            row[f"protein_ss_{label}_fraction"] = np.nan
            row[f"band_ss_{label}_enrichment"] = np.nan
            row[f"top_hub_ss_{label}_fraction"] = np.nan
            row[f"top_hub_ss_{label}_enrichment"] = np.nan
        row["has_ss"] = False

    sv_rows = []
    for idx, frac in enumerate(sv_fraction, start=1):
        sv_rows.append({
            "condition": condition,
            "seed": int(seed),
            "protein": protein,
            "singular_index": idx,
            "singular_value_fraction": frac,
        })

    band_rows = []
    for band_idx, (start, end) in enumerate(regions, start=1):
        mask = np.zeros(n, dtype=bool)
        mask[start:end + 1] = True
        band_row = {
            "condition": condition,
            "seed": int(seed),
            "protein": protein,
            "band_index": band_idx,
            "start_1based": start + 1,
            "end_1based": end + 1,
            "width": end - start + 1,
            "attention_mass": float(np.sum(received[mask]) / received_total) if received_total > 0 else np.nan,
            "peak_position_1based": int(np.argmax(smoothed_received[start:end + 1]) + start + 1),
            "peak_received": float(np.max(received[start:end + 1])),
        }
        if flexible is not None:
            selected, background, ratio = enrichment(mask, flexible == 1)
            band_row["flexible_fraction"] = selected
            band_row["flexible_enrichment"] = ratio
        if ss is not None:
            ss_array = np.asarray(ss)
            for label in ["C", "H", "E"]:
                selected, background, ratio = enrichment(mask, ss_array == label)
                band_row[f"ss_{label}_fraction"] = selected
                band_row[f"ss_{label}_enrichment"] = ratio
        band_rows.append(band_row)

    feature = {
        "condition": condition,
        "seed": int(seed),
        "protein": protein,
        "received": received,
        "top_hubs": top_hubs,
        "band_positions": band_positions,
        "band_regions": regions,
    }
    return row, sv_rows, band_rows, feature


def aggregate(df, group_cols, value_cols):
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols)[value_cols].agg(["mean", "std", "count"])
    out.columns = ["_".join(c).strip("_") for c in out.columns]
    return out.reset_index()


def build_stable_examples(features, top_n):
    rows = []
    by_condition_protein = {}
    for feat in features:
        key = (feat["condition"], feat["protein"])
        by_condition_protein.setdefault(key, []).append(feat)

    for (condition, protein), entries in by_condition_protein.items():
        if len(entries) < 2:
            continue
        spears = []
        jaccs = []
        band_jaccs = []
        for a, b in itertools.combinations(entries, 2):
            spears.append(spearman(a["received"], b["received"]))
            jaccs.append(jaccard(a["top_hubs"], b["top_hubs"]))
            band_jaccs.append(jaccard(a["band_positions"], b["band_positions"]))
        rows.append({
            "condition": condition,
            "protein": protein,
            "n_seed_pairs": len(spears),
            "received_spearman_mean": float(np.nanmean(spears)),
            "top10_jaccard_mean": float(np.nanmean(jaccs)),
            "band_jaccard_mean": float(np.nanmean(band_jaccs)),
            "stability_score": float(np.nanmean(spears) + np.nanmean(jaccs) + np.nanmean(band_jaccs)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.sort_values(["condition", "stability_score"], ascending=[True, False])
        .groupby("condition", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parents[2]
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest TSV: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_morphology", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = args.test_csv
    if test_csv is None:
        candidates = [result_root / "test_data_with_names.csv", result_root / "test_data.csv"]
        for candidate in candidates:
            if candidate.exists():
                test_csv = str(candidate)
                break
    neq_by_name, neq_by_sequence = load_neq_maps(test_csv)
    ss_map = load_ss_map(args.ss_csv)

    manifest = pd.read_csv(manifest_path, sep="\t")
    metric_rows = []
    svd_rows = []
    band_rows = []
    features = []
    missing = []

    for run in manifest.itertuples(index=False):
        attention_path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            missing.append(str(attention_path))
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            row, sv_rows, b_rows, feature = summarize_record(
                record,
                condition=run.condition,
                seed=run.seed,
                architecture=run.architecture,
                esm_model=run.esm_model,
                args=args,
                neq_by_name=neq_by_name,
                neq_by_sequence=neq_by_sequence,
                ss_map=ss_map,
            )
            metric_rows.append(row)
            svd_rows.extend(sv_rows)
            band_rows.extend(b_rows)
            features.append(feature)

    if missing:
        missing_path = output_dir / "missing_attention_files.txt"
        missing_path.write_text("\n".join(missing) + "\n")
        print(f"[warn] Missing {len(missing)} attention files. See {missing_path}")
    if not metric_rows:
        raise RuntimeError("No attention records were analyzed. Check result_root/path resolution.")

    metrics = pd.DataFrame(metric_rows)
    svd = pd.DataFrame(svd_rows)
    bands = pd.DataFrame(band_rows)

    base_cols = [
        "effective_rank",
        "relative_effective_rank",
        "svd_top1_energy_frac",
        "svd_top3_energy_frac",
        "svd_top5_energy_frac",
        "column_hub_top10_mass",
        "column_hub_gini",
        "received_max_over_mean",
        "normalized_row_entropy",
        "row_cosine_similarity_mean",
        "row_pearson_similarity_mean",
        "band_count",
        "band_width_mean",
        "band_width_max",
        "band_residue_fraction",
        "band_attention_mass",
        "band_flexible_fraction",
        "protein_flexible_fraction",
        "band_flexible_enrichment",
        "top_hub_flexible_fraction",
        "top_hub_flexible_enrichment",
        "received_vs_neq_spearman",
        "band_ss_C_fraction",
        "band_ss_C_enrichment",
        "band_ss_H_fraction",
        "band_ss_H_enrichment",
        "band_ss_E_fraction",
        "band_ss_E_enrichment",
        "top_hub_ss_C_fraction",
        "top_hub_ss_C_enrichment",
        "top_hub_ss_H_fraction",
        "top_hub_ss_H_enrichment",
        "top_hub_ss_E_fraction",
        "top_hub_ss_E_enrichment",
    ]
    for window in args.diag_windows:
        base_cols.extend([f"diagonal_mass_w{window}", f"long_range_mass_w{window}"])
    value_cols = [col for col in base_cols if col in metrics.columns]

    by_seed = aggregate(metrics, ["condition", "seed"], value_cols)
    by_condition = aggregate(by_seed, ["condition"], [f"{c}_mean" for c in value_cols if f"{c}_mean" in by_seed.columns])
    stable = build_stable_examples(features, args.stable_top_n)

    metrics.to_csv(output_dir / "attention_morphology_by_run.csv", index=False)
    svd.to_csv(output_dir / "svd_spectrum_long.csv", index=False)
    bands.to_csv(output_dir / "vertical_bands_by_run.csv", index=False)
    by_seed.to_csv(output_dir / "morphology_summary_by_seed.csv", index=False)
    by_condition.to_csv(output_dir / "morphology_summary_across_seeds.csv", index=False)
    stable.to_csv(output_dir / "stable_protein_examples.csv", index=False)

    summary_lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Runs in manifest: {len(manifest)}",
        f"Protein/run attention records analyzed: {len(metrics)}",
        f"Vertical bands detected: {len(bands)}",
        f"Secondary structure available for records: {int(metrics['has_ss'].sum())}/{len(metrics)}",
        f"Neq labels available for records: {int(metrics['protein_flexible_fraction'].notna().sum())}/{len(metrics)}",
        "",
        "Key outputs:",
        "  attention_morphology_by_run.csv",
        "  morphology_summary_across_seeds.csv",
        "  svd_spectrum_long.csv",
        "  vertical_bands_by_run.csv",
        "  stable_protein_examples.csv",
    ]
    (output_dir / "morphology_analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
