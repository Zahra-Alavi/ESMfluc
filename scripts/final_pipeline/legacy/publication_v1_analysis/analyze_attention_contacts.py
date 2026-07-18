#!/usr/bin/env python3
"""
Analyze whether attention scores recover residue-residue structural contacts.

This is a manifest-based follow-up to Attention/build_contact_maps_from_pdb.py.
It compares attention pair scores against binary C-alpha contact maps using:
  - AUROC / AUPRC over residue pairs
  - precision among top-scoring L, L/2, and L/5 pairs
  - fraction of attention mass assigned to true contacts
  - sequence-separation matched nulls for contact attention mass

The default pair set excludes near-diagonal pairs |i-j| < 6, following common
protein-contact evaluation practice.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

from analyze_attention_row_modes import default_analysis_dir, resolve_existing_path, resolve_manifest_path


DEFAULT_SEP_BINS = "6-11,12-23,24-47,48-inf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Contact-map recovery analysis for attention matrices."
    )
    parser.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    parser.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Defaults to result_root/manifest.tsv. "
             "Use manifest_attention_sources.tsv for the 30-attention source view.",
    )
    parser.add_argument(
        "--contact_json",
        default=None,
        help="Contact map JSON from Attention/build_contact_maps_from_pdb.py. Default: result_root/contact_maps_ca8.json.",
    )
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_contacts.")
    parser.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional condition subset.")
    parser.add_argument(
        "--attention_sources",
        nargs="+",
        default=["bilstm"],
        choices=["bilstm", "backbone"],
        help="Attention sources to analyze. bilstm uses manifest attention_json. backbone searches run_dir.",
    )
    parser.add_argument(
        "--backbone_filenames",
        nargs="+",
        default=["backbone_attention.json", "backbone_attn.json"],
        help="Candidate backbone attention filenames inside each run_dir.",
    )
    parser.add_argument(
        "--symmetrize",
        choices=["max", "mean"],
        default="max",
        help="Convert directed attention A_ij/A_ji to undirected pair score. Default: max.",
    )
    parser.add_argument("--min_seq_sep", type=int, default=6, help="Minimum |i-j| pair separation. Default: 6.")
    parser.add_argument("--sep_bins", default=DEFAULT_SEP_BINS, help=f"Matched-null bins. Default: {DEFAULT_SEP_BINS}")
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument(
        "--allow_length_mismatch",
        action="store_true",
        help="Crop attention/contact maps to common length. Default skips mismatched maps.",
    )
    parser.add_argument(
        "--require_exact_sequence_match",
        action="store_true",
        help="Skip contact maps unless build_contact_maps sequence_check is same length with zero mismatches.",
    )
    return parser.parse_args()


def parse_sep_bins(spec):
    bins = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        low_s, high_s = raw.split("-", 1)
        low = int(low_s)
        high = None if high_s.lower() in {"inf", "none", "max"} else int(high_s)
        if high is not None and high < low:
            raise ValueError(f"Invalid separation bin: {raw}")
        bins.append((raw, low, high))
    if not bins:
        raise ValueError("--sep_bins must define at least one bin.")
    return bins


def load_contact_maps(path, require_exact_sequence_match):
    records = json.loads(Path(path).read_text())
    contacts = {}
    status_rows = []
    for record in records:
        name = record.get("name", "")
        status = record.get("status", "")
        row = {
            "protein": name,
            "status": status,
            "error": record.get("error", ""),
            "has_contact_map": "contact_map" in record,
            "n_residues_structure": record.get("n_residues_structure", np.nan),
        }
        seq_check = record.get("sequence_check", {}) or {}
        row.update({
            "same_length": seq_check.get("same_length", np.nan),
            "mismatch_count": seq_check.get("mismatch_count", np.nan),
            "sequence_identity": seq_check.get("identity", np.nan),
        })
        status_rows.append(row)

        if status != "ok" or "contact_map" not in record:
            continue
        if require_exact_sequence_match:
            if not seq_check.get("same_length", False) or seq_check.get("mismatch_count", 1) != 0:
                continue
        contacts[name] = record
    return contacts, pd.DataFrame(status_rows)


def load_attention_records(path):
    records = json.loads(Path(path).read_text())
    return {record["name"]: record for record in records}


def resolve_backbone_path(run, result_root, pipeline_dir, filenames):
    run_dir = resolve_existing_path(run.run_dir, result_root, pipeline_dir)
    for filename in filenames:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return run_dir / filenames[0]


def symmetrize_attention(attn, mode):
    if mode == "max":
        return np.maximum(attn, attn.T)
    if mode == "mean":
        return 0.5 * (attn + attn.T)
    raise ValueError(mode)


def upper_triangle_pairs(n, min_seq_sep):
    ii, jj = np.triu_indices(n, k=min_seq_sep)
    sep = jj - ii
    return ii, jj, sep


def precision_at_k(y_true, scores, k):
    if len(scores) == 0:
        return np.nan
    k = max(1, min(int(k), len(scores)))
    idx = np.argsort(scores)[-k:]
    return float(np.mean(y_true[idx]))


def pearson_binary_score(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    x = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2 or np.std(x) <= 0:
        return np.nan
    return float(np.corrcoef(y, x)[0, 1])


def assign_bins(separations, bins):
    labels = np.full(len(separations), "", dtype=object)
    for label, low, high in bins:
        if high is None:
            mask = separations >= low
        else:
            mask = (separations >= low) & (separations <= high)
        labels[mask] = label
    return labels


def matched_null(scores, contacts, bin_labels, rng, n_permutations):
    total_score = float(np.sum(scores))
    if total_score <= 0:
        total_score = np.nan

    rows = []
    unique_bins = sorted(b for b in set(bin_labels) if b)
    for perm in range(1, n_permutations + 1):
        sampled_idx = []
        for label in unique_bins:
            in_bin = bin_labels == label
            contact_idx = np.where(in_bin & contacts)[0]
            noncontact_idx = np.where(in_bin & ~contacts)[0]
            if len(contact_idx) == 0 or len(noncontact_idx) == 0:
                continue
            replace = len(noncontact_idx) < len(contact_idx)
            sampled_idx.extend(rng.choice(noncontact_idx, size=len(contact_idx), replace=replace).tolist())

        if not sampled_idx:
            rows.append({
                "permutation": perm,
                "null_contact_mass_fraction": np.nan,
                "null_contact_mean_attention": np.nan,
            })
            continue
        sampled_idx = np.asarray(sampled_idx, dtype=int)
        rows.append({
            "permutation": perm,
            "null_contact_mass_fraction": float(np.sum(scores[sampled_idx]) / total_score) if np.isfinite(total_score) else np.nan,
            "null_contact_mean_attention": float(np.mean(scores[sampled_idx])),
        })
    return rows


def summarize_pairs(attn, contact_map, protein, condition, seed, attention_source, args, bins, rng):
    n = min(attn.shape[0], contact_map.shape[0])
    if (attn.shape[0] != contact_map.shape[0]) and not args.allow_length_mismatch:
        return None, []
    attn = attn[:n, :n]
    contact_map = contact_map[:n, :n].astype(bool)

    pair_scores = symmetrize_attention(attn, args.symmetrize)
    ii, jj, sep = upper_triangle_pairs(n, args.min_seq_sep)
    scores = pair_scores[ii, jj].astype(float)
    contacts = contact_map[ii, jj].astype(bool)
    bin_labels = assign_bins(sep, bins)
    keep = bin_labels != ""
    scores = scores[keep]
    contacts = contacts[keep]
    sep = sep[keep]
    bin_labels = bin_labels[keep]

    n_pairs = len(scores)
    n_contacts = int(np.sum(contacts))
    if n_pairs == 0 or n_contacts == 0 or n_contacts == n_pairs:
        return None, []

    total_score = float(np.sum(scores))
    contact_score = float(np.sum(scores[contacts]))
    noncontact_mean = float(np.mean(scores[~contacts])) if np.any(~contacts) else np.nan
    contact_mean = float(np.mean(scores[contacts]))
    length = int(n)

    row = {
        "condition": condition,
        "seed": int(seed),
        "protein": protein,
        "attention_source": attention_source,
        "n_residues": length,
        "n_pairs": int(n_pairs),
        "n_contacts": n_contacts,
        "contact_rate": float(n_contacts / n_pairs),
        "contact_mass_fraction": contact_score / total_score if total_score > 0 else np.nan,
        "contact_mean_attention": contact_mean,
        "noncontact_mean_attention": noncontact_mean,
        "contact_noncontact_mean_ratio": contact_mean / noncontact_mean if noncontact_mean > 0 else np.nan,
        "pointbiserial_contact_attention": pearson_binary_score(contacts, scores),
        "precision_at_L": precision_at_k(contacts, scores, length),
        "precision_at_L_over_2": precision_at_k(contacts, scores, math.ceil(length / 2)),
        "precision_at_L_over_5": precision_at_k(contacts, scores, math.ceil(length / 5)),
    }
    if roc_auc_score is not None and len(np.unique(contacts)) == 2:
        row["contact_auroc"] = float(roc_auc_score(contacts, scores))
        row["contact_auprc"] = float(average_precision_score(contacts, scores))
    else:
        row["contact_auroc"] = np.nan
        row["contact_auprc"] = np.nan

    null_rows = matched_null(scores, contacts, bin_labels, rng, args.n_permutations)
    for null in null_rows:
        null.update({
            "condition": condition,
            "seed": int(seed),
            "protein": protein,
            "attention_source": attention_source,
        })
    return row, null_rows


def aggregate(df, group_cols, value_cols):
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols)[value_cols].agg(["mean", "std", "count"])
    out.columns = ["_".join(col).strip("_") for col in out.columns]
    return out.reset_index()


def empirical_pvalues(observed, null):
    rows = []
    if observed.empty or null.empty:
        return pd.DataFrame()
    metric_pairs = [
        ("contact_mass_fraction", "null_contact_mass_fraction"),
        ("contact_mean_attention", "null_contact_mean_attention"),
    ]
    for (condition, source), obs_group in observed.groupby(["condition", "attention_source"]):
        null_group = null[(null["condition"] == condition) & (null["attention_source"] == source)]
        for obs_metric, null_metric in metric_pairs:
            obs_val = float(obs_group[obs_metric].mean())
            null_vals = null_group.groupby("permutation")[null_metric].mean().dropna().to_numpy()
            if len(null_vals) == 0:
                continue
            rows.append({
                "condition": condition,
                "attention_source": source,
                "metric": obs_metric,
                "observed_mean": obs_val,
                "null_mean": float(np.mean(null_vals)),
                "observed_minus_null": obs_val - float(np.mean(null_vals)),
                "empirical_p_greater": float((1 + np.sum(null_vals >= obs_val)) / (len(null_vals) + 1)),
                "n_null_permutations": int(len(null_vals)),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parents[2]
    contact_json = Path(args.contact_json).expanduser().resolve() if args.contact_json else result_root / "contact_maps_ca8.json"
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest TSV: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_contacts", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not contact_json.exists():
        raise FileNotFoundError(
            f"Missing contact JSON: {contact_json}\n"
            "Build it first, for example:\n"
            f"  python Attention/build_contact_maps_from_pdb.py --input_csv {result_root / 'test_data_with_names.csv'} "
            f"--output_json {contact_json} --pdb_dir {result_root / 'pdb_cache'} --cutoff 8.0"
        )

    contacts, contact_status = load_contact_maps(contact_json, args.require_exact_sequence_match)
    contact_status.to_csv(output_dir / "contact_map_status.csv", index=False)
    bins = parse_sep_bins(args.sep_bins)
    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()

    rng = np.random.default_rng(args.random_seed)
    metric_rows = []
    null_rows = []
    skip_rows = []

    for run in manifest.itertuples(index=False):
        source_paths = {}
        if "bilstm" in args.attention_sources:
            source_paths["bilstm"] = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        if "backbone" in args.attention_sources:
            source_paths["backbone"] = resolve_backbone_path(run, result_root, pipeline_dir, args.backbone_filenames)

        for source, attention_path in source_paths.items():
            source_label = (
                str(run.attention_kind)
                if source == "bilstm" and hasattr(run, "attention_kind")
                else source
            )
            if not attention_path.exists():
                skip_rows.append({
                    "condition": run.condition,
                    "seed": int(run.seed),
                    "attention_source": source_label,
                    "protein": "",
                    "reason": f"missing attention file: {attention_path}",
                })
                continue
            print(f"[load] {run.condition} seed={run.seed} source={source_label}: {attention_path}")
            records = load_attention_records(attention_path)
            for protein, contact_record in contacts.items():
                if protein not in records:
                    skip_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "attention_source": source_label,
                        "protein": protein,
                        "reason": "protein missing from attention JSON",
                    })
                    continue
                attn_record = records[protein]
                attn = np.asarray(attn_record["attention_weights"], dtype=float)
                contact_map = np.asarray(contact_record["contact_map"], dtype=bool)
                row, nrows = summarize_pairs(
                    attn,
                    contact_map,
                    protein,
                    run.condition,
                    run.seed,
                    source_label,
                    args,
                    bins,
                    rng,
                )
                if row is None:
                    skip_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "attention_source": source_label,
                        "protein": protein,
                        "reason": "no valid pair/contact set or length mismatch",
                    })
                    continue
                metric_rows.append(row)
                null_rows.extend(nrows)

    metrics = pd.DataFrame(metric_rows)
    null = pd.DataFrame(null_rows)
    skipped = pd.DataFrame(skip_rows)

    metrics.to_csv(output_dir / "attention_contact_metrics_by_run.csv", index=False)
    null.to_csv(output_dir / "attention_contact_matched_null.csv", index=False)
    skipped.to_csv(output_dir / "attention_contact_skipped.csv", index=False)

    value_cols = [
        "contact_rate",
        "contact_mass_fraction",
        "contact_mean_attention",
        "noncontact_mean_attention",
        "contact_noncontact_mean_ratio",
        "pointbiserial_contact_attention",
        "contact_auroc",
        "contact_auprc",
        "precision_at_L",
        "precision_at_L_over_2",
        "precision_at_L_over_5",
    ]
    aggregate(metrics, ["condition", "attention_source"], value_cols).to_csv(
        output_dir / "attention_contact_summary_by_condition.csv", index=False
    )
    empirical_pvalues(metrics, null).to_csv(
        output_dir / "attention_contact_matched_pvalues.csv", index=False
    )

    lines = [
        f"Result root: {result_root}",
        f"Contact JSON: {contact_json}",
        f"Output dir: {output_dir}",
        f"Contact maps loaded: {len(contacts)}",
        f"Metric rows: {len(metrics)}",
        f"Null rows: {len(null)}",
        f"Skipped rows: {len(skipped)}",
        f"Attention sources: {', '.join(args.attention_sources)}",
        f"Minimum sequence separation: {args.min_seq_sep}",
        f"Matched separation bins: {args.sep_bins}",
        "",
        "Key outputs:",
        "  attention_contact_metrics_by_run.csv",
        "  attention_contact_summary_by_condition.csv",
        "  attention_contact_matched_pvalues.csv",
        "  contact_map_status.csv",
    ]
    (output_dir / "attention_contact_analysis_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
