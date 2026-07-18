#!/usr/bin/env python3
"""
Analyze row-level attention modes in ESMfluc attention maps.

Hypothesis tested:
  1. High-entropy query rows are diffuse/unstructured.
  2. Low-entropy query rows split into two structured modes.
  3. Each low-entropy mode preferentially attends to residues in the same mode.
  4. The row modes align with secondary structure and/or flexibility.

This complements analyze_attention_morphology_biology.py. That script shows the
matrices are very low-rank and banded; this one asks whether the low-rank
structure corresponds to a row-mode decomposition.

Example:
  python analyze_attention_row_modes.py \
    --result_root results/publication_comparable_v1 \
    --test_csv ../../data/test_data_with_names.csv \
    --ss_csv ../../data/test_data_nsp3.csv
"""

import argparse
import ast
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster low-entropy attention rows into two modes plus a high-entropy diffuse mode."
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
        help="Output directory. Defaults to result_root/analysis_row_modes.",
    )
    parser.add_argument(
        "--pipeline_dir",
        default=None,
        help="Directory used when manifest paths are relative. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--high_entropy_quantile",
        type=float,
        default=0.67,
        help="Rows above this entropy quantile become the diffuse mode. Default: 0.67.",
    )
    parser.add_argument(
        "--min_low_rows",
        type=int,
        default=8,
        help="Minimum low-entropy rows required to fit two clusters. Default: 8.",
    )
    parser.add_argument(
        "--kmeans_seed",
        type=int,
        default=0,
        help="Random seed for KMeans. Default: 0.",
    )
    parser.add_argument(
        "--kmeans_n_init",
        type=int,
        default=None,
        help="KMeans n_init for low-row mode clustering. Default: ROW_MODE_KMEANS_N_INIT or 20.",
    )
    parser.add_argument(
        "--save_residue_assignments",
        action="store_true",
        help="Also save one row per residue/mode assignment. This can be large.",
    )
    parser.add_argument(
        "--stable_top_n",
        type=int,
        default=40,
        help="Number of strongest row-mode example proteins per condition. Default: 40.",
    )
    return parser.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def row_entropy(attn):
    eps = 1e-12
    n = attn.shape[1]
    if n <= 1:
        return np.zeros(attn.shape[0], dtype=float)
    row_sums = attn.sum(axis=1, keepdims=True)
    probs = np.divide(attn, row_sums, out=np.zeros_like(attn), where=row_sums > 0)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return entropy / np.log(n)


def normalize_rows(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


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
    ratio = selected_fraction / background_fraction if background_fraction > 0 else np.nan
    return selected_fraction, background_fraction, ratio


def jaccard(a, b):
    a = set(a)
    b = set(b)
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


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


def resolve_manifest_path(result_root, manifest_tsv=None):
    path = Path(manifest_tsv).expanduser() if manifest_tsv else result_root / "manifest.tsv"
    if not path.is_absolute():
        path = result_root / path
    return path.resolve()


def default_analysis_dir(result_root, base_name, manifest_path):
    if manifest_path.name == "manifest.tsv":
        return result_root / base_name
    suffix = manifest_path.stem
    if suffix.startswith("manifest_"):
        suffix = suffix[len("manifest_"):]
    return result_root / f"{base_name}_{suffix}"


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


def get_neq_entry(record, neq_by_name, neq_by_sequence):
    return neq_by_name.get(record["name"]) or neq_by_sequence.get(record["sequence"])


def label_low_modes_by_size(labels):
    """Make low-mode labels deterministic: mode_1 is the larger low cluster."""
    labels = labels.copy()
    low = labels > 0
    if not low.any():
        return labels
    count1 = int(np.sum(labels == 1))
    count2 = int(np.sum(labels == 2))
    if count2 > count1:
        labels[labels == 1] = 99
        labels[labels == 2] = 1
        labels[labels == 99] = 2
    return labels


def resolve_kmeans_n_init(kmeans_n_init=None):
    if kmeans_n_init is not None:
        return int(kmeans_n_init)
    return int(os.environ.get("ROW_MODE_KMEANS_N_INIT", "20"))


def analyze_attention_modes(attn, high_entropy_quantile, min_low_rows, kmeans_seed, kmeans_n_init=None):
    n = attn.shape[0]
    ent = row_entropy(attn)
    threshold = float(np.quantile(ent, high_entropy_quantile))
    diffuse_mask = ent > threshold
    low_mask = ~diffuse_mask

    # Guard against very small proteins or degenerate entropy distributions.
    if low_mask.sum() < min_low_rows:
        labels = np.zeros(n, dtype=int)
        labels[low_mask] = 1
        return labels, ent, threshold, False, np.nan, np.nan

    low_rows = normalize_rows(attn[low_mask])
    if len(low_rows) < 2 or np.allclose(low_rows, low_rows[0]):
        labels = np.zeros(n, dtype=int)
        labels[low_mask] = 1
        return labels, ent, threshold, False, np.nan, np.nan

    km = KMeans(n_clusters=2, random_state=kmeans_seed, n_init=resolve_kmeans_n_init(kmeans_n_init))
    low_cluster = km.fit_predict(low_rows) + 1

    labels = np.zeros(n, dtype=int)
    labels[low_mask] = low_cluster
    labels = label_low_modes_by_size(labels)

    c1 = np.mean(normalize_rows(attn[labels == 1]), axis=0) if np.any(labels == 1) else None
    c2 = np.mean(normalize_rows(attn[labels == 2]), axis=0) if np.any(labels == 2) else None
    centroid_cosine = cosine(c1, c2) if c1 is not None and c2 is not None else np.nan

    own_minus_other = []
    norm_rows = normalize_rows(attn)
    for i in np.where(labels > 0)[0]:
        own = c1 if labels[i] == 1 else c2
        other = c2 if labels[i] == 1 else c1
        own_minus_other.append(cosine(norm_rows[i], own) - cosine(norm_rows[i], other))
    separation = float(np.nanmean(own_minus_other)) if own_minus_other else np.nan
    return labels, ent, threshold, True, centroid_cosine, separation


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(label, str(label))


def mode_block_masses(attn, labels):
    rows = []
    for q_label in [0, 1, 2]:
        q_idx = np.where(labels == q_label)[0]
        if len(q_idx) == 0:
            continue
        for k_label in [0, 1, 2]:
            k_idx = np.where(labels == k_label)[0]
            if len(k_idx) == 0:
                mass = np.nan
            else:
                mass = float(attn[np.ix_(q_idx, k_idx)].sum(axis=1).mean())
            rows.append({
                "query_mode": mode_name(q_label),
                "key_mode": mode_name(k_label),
                "mean_attention_mass": mass,
                "query_mode_size": len(q_idx),
                "key_mode_size": len(k_idx),
            })
    return rows


def summarize_mode_composition(labels, ent, ss, flexible):
    rows = []
    n = len(labels)
    ss_array = np.asarray(ss) if ss is not None else None
    flex_array = np.asarray(flexible) if flexible is not None else None
    for label in [0, 1, 2]:
        mask = labels == label
        if mask.sum() == 0:
            continue
        row = {
            "mode_label": label,
            "mode": mode_name(label),
            "residue_count": int(mask.sum()),
            "residue_fraction": float(mask.mean()),
            "mean_row_entropy": float(np.mean(ent[mask])),
        }
        if ss_array is not None and len(ss_array) == n:
            ss_fracs = {}
            for ss_label in ["C", "H", "E"]:
                selected, background, ratio = enrichment(mask, ss_array == ss_label)
                row[f"ss_{ss_label}_fraction"] = selected
                row[f"ss_{ss_label}_background"] = background
                row[f"ss_{ss_label}_enrichment"] = ratio
                ss_fracs[ss_label] = selected
            row["dominant_ss"] = max(ss_fracs, key=lambda key: -np.inf if np.isnan(ss_fracs[key]) else ss_fracs[key])
            row["dominant_ss_fraction"] = ss_fracs[row["dominant_ss"]]
        else:
            for ss_label in ["C", "H", "E"]:
                row[f"ss_{ss_label}_fraction"] = np.nan
                row[f"ss_{ss_label}_background"] = np.nan
                row[f"ss_{ss_label}_enrichment"] = np.nan
            row["dominant_ss"] = ""
            row["dominant_ss_fraction"] = np.nan

        if flex_array is not None and len(flex_array) == n:
            selected, background, ratio = enrichment(mask, flex_array == 1)
            row["flexible_fraction"] = selected
            row["flexible_background"] = background
            row["flexible_enrichment"] = ratio
        else:
            row["flexible_fraction"] = np.nan
            row["flexible_background"] = np.nan
            row["flexible_enrichment"] = np.nan
        rows.append(row)
    return rows


def summarize_one_record(record, run, args, neq_by_name, neq_by_sequence, ss_map):
    sequence = record["sequence"]
    protein = record["name"]
    n = len(sequence)
    attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]

    labels, ent, entropy_threshold, clustered, centroid_cosine, cluster_separation = analyze_attention_modes(
        attn,
        args.high_entropy_quantile,
        args.min_low_rows,
        args.kmeans_seed,
        args.kmeans_n_init,
    )

    neq_entry = get_neq_entry(record, neq_by_name, neq_by_sequence)
    flexible = neq_entry["flexible"][:n] if neq_entry is not None else None
    ss = ss_map.get(protein)
    if ss is not None:
        ss = ss[:n]
        if len(ss) != n:
            ss = None

    blocks = mode_block_masses(attn, labels)
    block_lookup = {
        (row["query_mode"], row["key_mode"]): row["mean_attention_mass"]
        for row in blocks
    }
    low_self = np.nanmean([
        block_lookup.get(("low_mode_1", "low_mode_1"), np.nan),
        block_lookup.get(("low_mode_2", "low_mode_2"), np.nan),
    ])
    low_cross = np.nanmean([
        block_lookup.get(("low_mode_1", "low_mode_2"), np.nan),
        block_lookup.get(("low_mode_2", "low_mode_1"), np.nan),
    ])
    low_to_diffuse = np.nanmean([
        block_lookup.get(("low_mode_1", "diffuse_high_entropy"), np.nan),
        block_lookup.get(("low_mode_2", "diffuse_high_entropy"), np.nan),
    ])
    diffuse_to_low = np.nanmean([
        block_lookup.get(("diffuse_high_entropy", "low_mode_1"), np.nan),
        block_lookup.get(("diffuse_high_entropy", "low_mode_2"), np.nan),
    ])
    self_cross_ratio = low_self / low_cross if np.isfinite(low_cross) and low_cross > 0 else np.nan

    mode_comp = summarize_mode_composition(labels, ent, ss, flexible)
    comp_by_mode = {row["mode"]: row for row in mode_comp}

    low_modes = [comp_by_mode.get("low_mode_1", {}), comp_by_mode.get("low_mode_2", {})]
    low_dominant_ss = [m.get("dominant_ss", "") for m in low_modes]
    low_ss_distinct = (
        len(low_dominant_ss) == 2
        and low_dominant_ss[0] != ""
        and low_dominant_ss[1] != ""
        and low_dominant_ss[0] != low_dominant_ss[1]
    )
    low_ss_purity_mean = float(np.nanmean([m.get("dominant_ss_fraction", np.nan) for m in low_modes]))

    if ss is not None:
        ss_encoded = np.asarray([{"C": 0, "H": 1, "E": 2}.get(x, -1) for x in ss])
        ok = ss_encoded >= 0
        ss_ami = adjusted_mutual_info_score(ss_encoded[ok], labels[ok]) if ok.sum() > 2 else np.nan
        low_ok = ok & (labels > 0)
        low_ss_ami = adjusted_mutual_info_score(ss_encoded[low_ok], labels[low_ok]) if low_ok.sum() > 2 else np.nan
    else:
        ss_ami = np.nan
        low_ss_ami = np.nan

    if flexible is not None:
        flex_ami = adjusted_mutual_info_score(flexible, labels) if len(np.unique(flexible)) > 1 else np.nan
    else:
        flex_ami = np.nan

    summary = {
        "condition": run.condition,
        "seed": int(run.seed),
        "architecture": run.architecture,
        "esm_model": run.esm_model,
        "protein": protein,
        "n_residues": n,
        "clustered_low_entropy_rows": clustered,
        "entropy_threshold": entropy_threshold,
        "diffuse_fraction": float(np.mean(labels == 0)),
        "low_mode_1_fraction": float(np.mean(labels == 1)),
        "low_mode_2_fraction": float(np.mean(labels == 2)),
        "diffuse_entropy_mean": float(np.mean(ent[labels == 0])) if np.any(labels == 0) else np.nan,
        "low_entropy_mean": float(np.mean(ent[labels > 0])) if np.any(labels > 0) else np.nan,
        "low_cluster_centroid_cosine": centroid_cosine,
        "low_cluster_separation": cluster_separation,
        "low_self_attention_mass": low_self,
        "low_cross_attention_mass": low_cross,
        "low_to_diffuse_attention_mass": low_to_diffuse,
        "diffuse_to_low_attention_mass": diffuse_to_low,
        "low_self_to_cross_ratio": self_cross_ratio,
        "low_modes_dominant_ss": "|".join(low_dominant_ss),
        "low_modes_have_distinct_dominant_ss": low_ss_distinct,
        "low_modes_ss_purity_mean": low_ss_purity_mean,
        "mode_ss_adjusted_mutual_info": ss_ami,
        "low_modes_ss_adjusted_mutual_info": low_ss_ami,
        "mode_flexible_adjusted_mutual_info": flex_ami,
    }

    block_rows = []
    for row in blocks:
        row.update({
            "condition": run.condition,
            "seed": int(run.seed),
            "protein": protein,
        })
        block_rows.append(row)

    comp_rows = []
    for row in mode_comp:
        row.update({
            "condition": run.condition,
            "seed": int(run.seed),
            "protein": protein,
        })
        comp_rows.append(row)

    assignment_rows = []
    if args.save_residue_assignments:
        for i, aa in enumerate(sequence):
            assignment_rows.append({
                "condition": run.condition,
                "seed": int(run.seed),
                "protein": protein,
                "position_1based": i + 1,
                "aa": aa,
                "mode_label": int(labels[i]),
                "mode": mode_name(int(labels[i])),
                "row_entropy": float(ent[i]),
                "ss": ss[i] if ss is not None else "",
                "neq": float(neq_entry["neq"][i]) if neq_entry is not None and i < len(neq_entry["neq"]) else np.nan,
                "flexible": int(flexible[i]) if flexible is not None else np.nan,
            })

    feature = {
        "condition": run.condition,
        "seed": int(run.seed),
        "protein": protein,
        "labels": labels,
    }
    return summary, block_rows, comp_rows, assignment_rows, feature


def aggregate(df, group_cols, value_cols):
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols)[value_cols].agg(["mean", "std", "count"])
    out.columns = ["_".join(c).strip("_") for c in out.columns]
    return out.reset_index()


def mode_stability(features):
    rows = []
    by_condition_protein = {}
    for feat in features:
        by_condition_protein.setdefault((feat["condition"], feat["protein"]), []).append(feat)

    for (condition, protein), entries in by_condition_protein.items():
        if len(entries) < 2:
            continue
        ari_full = []
        ari_low_overlap = []
        diffuse_jacc = []
        low_jacc_best = []
        for a, b in itertools.combinations(entries, 2):
            labels_a = a["labels"]
            labels_b = b["labels"]
            n = min(len(labels_a), len(labels_b))
            labels_a = labels_a[:n]
            labels_b = labels_b[:n]
            ari_full.append(adjusted_rand_score(labels_a, labels_b))
            low_overlap = (labels_a > 0) & (labels_b > 0)
            if low_overlap.sum() > 2:
                ari_low_overlap.append(adjusted_rand_score(labels_a[low_overlap], labels_b[low_overlap]))
            else:
                ari_low_overlap.append(np.nan)
            diffuse_jacc.append(jaccard(np.where(labels_a == 0)[0], np.where(labels_b == 0)[0]))
            a1, a2 = set(np.where(labels_a == 1)[0]), set(np.where(labels_a == 2)[0])
            b1, b2 = set(np.where(labels_b == 1)[0]), set(np.where(labels_b == 2)[0])
            direct = np.nanmean([jaccard(a1, b1), jaccard(a2, b2)])
            swapped = np.nanmean([jaccard(a1, b2), jaccard(a2, b1)])
            low_jacc_best.append(max(direct, swapped))
        rows.append({
            "condition": condition,
            "protein": protein,
            "n_seed_pairs": len(ari_full),
            "mode_ari_full_mean": float(np.nanmean(ari_full)),
            "mode_ari_low_overlap_mean": float(np.nanmean(ari_low_overlap)),
            "diffuse_jaccard_mean": float(np.nanmean(diffuse_jacc)),
            "low_mode_best_jaccard_mean": float(np.nanmean(low_jacc_best)),
            "mode_stability_score": float(
                np.nanmean(ari_full)
                + np.nanmean(diffuse_jacc)
                + np.nanmean(low_jacc_best)
            ),
        })
    return pd.DataFrame(rows)


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
        else default_analysis_dir(result_root, "analysis_row_modes", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = args.test_csv
    if test_csv is None:
        for candidate in [result_root / "test_data_with_names.csv", result_root / "test_data.csv"]:
            if candidate.exists():
                test_csv = str(candidate)
                break

    neq_by_name, neq_by_sequence = load_neq_maps(test_csv)
    ss_map = load_ss_map(args.ss_csv)
    manifest = pd.read_csv(manifest_path, sep="\t")

    summary_rows = []
    block_rows = []
    composition_rows = []
    assignment_rows = []
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
            summary, blocks, comps, assignments, feature = summarize_one_record(
                record, run, args, neq_by_name, neq_by_sequence, ss_map
            )
            summary_rows.append(summary)
            block_rows.extend(blocks)
            composition_rows.extend(comps)
            assignment_rows.extend(assignments)
            features.append(feature)

    if missing:
        (output_dir / "missing_attention_files.txt").write_text("\n".join(missing) + "\n")
        print(f"[warn] Missing {len(missing)} attention files.")
    if not summary_rows:
        raise RuntimeError("No attention records were analyzed. Check result_root/path resolution.")

    summary = pd.DataFrame(summary_rows)
    blocks = pd.DataFrame(block_rows)
    composition = pd.DataFrame(composition_rows)
    stability = mode_stability(features)

    value_cols = [
        "clustered_low_entropy_rows",
        "diffuse_fraction",
        "low_mode_1_fraction",
        "low_mode_2_fraction",
        "diffuse_entropy_mean",
        "low_entropy_mean",
        "low_cluster_centroid_cosine",
        "low_cluster_separation",
        "low_self_attention_mass",
        "low_cross_attention_mass",
        "low_to_diffuse_attention_mass",
        "diffuse_to_low_attention_mass",
        "low_self_to_cross_ratio",
        "low_modes_have_distinct_dominant_ss",
        "low_modes_ss_purity_mean",
        "mode_ss_adjusted_mutual_info",
        "low_modes_ss_adjusted_mutual_info",
        "mode_flexible_adjusted_mutual_info",
    ]
    summary_for_agg = summary.copy()
    for col in ["clustered_low_entropy_rows", "low_modes_have_distinct_dominant_ss"]:
        summary_for_agg[col] = summary_for_agg[col].astype(float)
    by_seed = aggregate(summary_for_agg, ["condition", "seed"], value_cols)
    by_condition = aggregate(
        by_seed,
        ["condition"],
        [f"{col}_mean" for col in value_cols if f"{col}_mean" in by_seed.columns],
    )

    stability_by_condition = aggregate(
        stability,
        ["condition"],
        [
            "mode_ari_full_mean",
            "mode_ari_low_overlap_mean",
            "diffuse_jaccard_mean",
            "low_mode_best_jaccard_mean",
            "mode_stability_score",
        ],
    )

    examples = summary.merge(
        stability[["condition", "protein", "mode_stability_score", "mode_ari_full_mean", "diffuse_jaccard_mean", "low_mode_best_jaccard_mean"]],
        on=["condition", "protein"],
        how="left",
    )
    examples["row_mode_signal_score"] = (
        examples["low_self_to_cross_ratio"].replace([np.inf, -np.inf], np.nan)
        + examples["low_modes_ss_purity_mean"]
        + examples["mode_stability_score"]
    )
    examples = (
        examples.sort_values(["condition", "row_mode_signal_score"], ascending=[True, False])
        .groupby("condition", as_index=False)
        .head(args.stable_top_n)
        .reset_index(drop=True)
    )

    summary.to_csv(output_dir / "row_mode_summary_by_run.csv", index=False)
    blocks.to_csv(output_dir / "row_mode_attention_blocks_by_run.csv", index=False)
    composition.to_csv(output_dir / "row_mode_composition_by_run.csv", index=False)
    by_seed.to_csv(output_dir / "row_mode_summary_by_seed.csv", index=False)
    by_condition.to_csv(output_dir / "row_mode_summary_across_seeds.csv", index=False)
    stability.to_csv(output_dir / "row_mode_stability_by_protein.csv", index=False)
    stability_by_condition.to_csv(output_dir / "row_mode_stability_by_condition.csv", index=False)
    examples.to_csv(output_dir / "row_mode_candidate_examples.csv", index=False)

    if args.save_residue_assignments:
        pd.DataFrame(assignment_rows).to_csv(output_dir / "row_mode_assignments_by_residue.csv", index=False)

    summary_lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Runs in manifest: {len(manifest)}",
        f"Protein/run attention records analyzed: {len(summary)}",
        f"High-entropy quantile: {args.high_entropy_quantile}",
        f"KMeans n_init: {resolve_kmeans_n_init(args.kmeans_n_init)}",
        f"Secondary structure maps loaded: {len(ss_map)}",
        f"Neq name maps loaded: {len(neq_by_name)}",
        "",
        "Key outputs:",
        "  row_mode_summary_by_run.csv",
        "  row_mode_summary_across_seeds.csv",
        "  row_mode_attention_blocks_by_run.csv",
        "  row_mode_composition_by_run.csv",
        "  row_mode_stability_by_protein.csv",
        "  row_mode_candidate_examples.csv",
    ]
    (output_dir / "row_mode_analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
