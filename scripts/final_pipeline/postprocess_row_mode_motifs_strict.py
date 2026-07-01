#!/usr/bin/env python3
"""
Strict post-processing for row-mode attention k-mers.

This script consumes the outputs from build_row_mode_attention_pwms.py and adds
filters that are closer to something publishable:
  - drop terminal residues and tag-like/poly-residue windows
  - require recurrence across multiple proteins and seeds
  - compare each k-mer against the same-SS matched background instances
  - report protein-level recurrence, not only raw instance counts
"""

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Strict recurrence/enrichment filter for attention k-mers.")
    parser.add_argument("--result_root", required=True, help="Result root, e.g. publication_comparable_v1.")
    parser.add_argument("--motif_dir", default=None, help="Default: result_root/analysis_row_mode_pwms.")
    parser.add_argument("--test_csv", default=None, help="Optional CSV with name and sequence, used for true protein lengths.")
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_row_mode_pwms_strict.")
    parser.add_argument("--terminal_exclusion", type=int, default=10,
                        help="Exclude selected/background centers within this many residues of either terminus.")
    parser.add_argument("--tag_regex", default=r"H{4,}|G{5,}|S{5,}|K{5,}|E{5,}",
                        help="Regex for tag-like/poly-residue windows to exclude.")
    parser.add_argument("--min_observed_count", type=int, default=5)
    parser.add_argument("--min_observed_proteins", type=int, default=3)
    parser.add_argument("--min_observed_seeds", type=int, default=2)
    parser.add_argument("--min_log2_enrichment", type=float, default=0.75)
    parser.add_argument("--min_observed_fraction", type=float, default=0.0005)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    return parser.parse_args()


def load_lengths(test_csv):
    if test_csv is None:
        return {}
    path = Path(test_csv).expanduser()
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    lengths = {}
    for _, row in df.iterrows():
        sequence = str(row["sequence"])
        if "name" in df.columns:
            lengths[str(row["name"])] = len(sequence)
        lengths.setdefault(sequence, len(sequence))
    return lengths


def count_unique(series):
    return int(pd.Series(series).dropna().nunique())


def collapsed_values(series, max_items=12):
    values = sorted(str(x) for x in pd.Series(series).dropna().unique())
    if len(values) <= max_items:
        return "|".join(values)
    return "|".join(values[:max_items]) + f"|...(+{len(values) - max_items})"


def add_quality_flags(df, lengths, terminal_exclusion, tag_pattern):
    df = df.copy()
    if df.empty:
        df["protein_length"] = []
        df["near_terminal"] = []
        df["tag_like_window"] = []
        df["valid_strict_window"] = []
        return df

    inferred_lengths = df.groupby("protein")["position_1based"].transform("max")
    df["protein_length"] = [
        int(lengths.get(protein, inferred))
        for protein, inferred in zip(df["protein"], inferred_lengths)
    ]
    pos = df["position_1based"].astype(int)
    length = df["protein_length"].astype(int)
    df["near_terminal"] = (pos <= terminal_exclusion) | (pos > (length - terminal_exclusion))
    window_text = (
        df.get("motif", pd.Series("", index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("kmer", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    df["tag_like_window"] = window_text.str.contains(tag_pattern, regex=True)
    df["valid_strict_window"] = (
        df["kmer"].notna()
        & ~df["kmer"].astype(str).str.contains("-", regex=False)
        & ~df["near_terminal"]
        & ~df["tag_like_window"]
    )
    return df


def aggregate_counts(obs, bg, pseudocount):
    group_cols = ["condition", "selection", "kmer"]
    obs_counts = (
        obs.groupby(group_cols)
        .agg(
            observed_count=("kmer", "size"),
            observed_unique_proteins=("protein", count_unique),
            observed_unique_seeds=("seed", count_unique),
            observed_proteins=("protein", collapsed_values),
            observed_ss_classes=("ss", collapsed_values),
            observed_modes=("mode", collapsed_values),
            observed_mean_neq=("neq", "mean"),
            observed_peak_fraction=("is_neq_peak", "mean"),
        )
        .reset_index()
    )
    bg_counts = (
        bg.groupby(group_cols)
        .agg(
            background_count=("kmer", "size"),
            background_unique_proteins=("protein", count_unique),
            background_unique_seeds=("seed", count_unique),
            background_proteins=("protein", collapsed_values),
            background_ss_classes=("ss", collapsed_values),
            background_mean_neq=("neq", "mean"),
            background_peak_fraction=("is_neq_peak", "mean"),
        )
        .reset_index()
    )

    totals_obs = obs.groupby(["condition", "selection"]).size().rename("observed_total").reset_index()
    totals_bg = bg.groupby(["condition", "selection"]).size().rename("background_total").reset_index()

    out = obs_counts.merge(bg_counts, on=group_cols, how="left")
    out = out.merge(totals_obs, on=["condition", "selection"], how="left")
    out = out.merge(totals_bg, on=["condition", "selection"], how="left")
    out["background_count"] = out["background_count"].fillna(0).astype(int)
    out["background_unique_proteins"] = out["background_unique_proteins"].fillna(0).astype(int)
    out["background_unique_seeds"] = out["background_unique_seeds"].fillna(0).astype(int)
    out["observed_fraction"] = out["observed_count"] / out["observed_total"].clip(lower=1)
    out["background_fraction"] = out["background_count"] / out["background_total"].clip(lower=1)

    obs_rate = (out["observed_count"] + pseudocount) / (out["observed_total"] + pseudocount * 2.0)
    bg_rate = (out["background_count"] + pseudocount) / (out["background_total"] + pseudocount * 2.0)
    out["enrichment"] = obs_rate / bg_rate
    out["log2_enrichment"] = np.log2(out["enrichment"])
    out["protein_recurrence_fraction"] = out["observed_unique_proteins"] / (
        obs.groupby(["condition", "selection"])["protein"].nunique()
        .rename("n_selection_proteins")
        .reset_index()
        .set_index(["condition", "selection"])
        .reindex(pd.MultiIndex.from_frame(out[["condition", "selection"]]))
        ["n_selection_proteins"]
        .to_numpy()
    )
    return out


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    motif_dir = Path(args.motif_dir).expanduser().resolve() if args.motif_dir else result_root / "analysis_row_mode_pwms"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_root / "analysis_row_mode_pwms_strict"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_path = motif_dir / "attention_motif_instances.csv"
    bg_path = motif_dir / "matched_background_motif_instances.csv"
    if not obs_path.exists() or not bg_path.exists():
        raise FileNotFoundError(f"Expected {obs_path} and {bg_path}. Run build_row_mode_attention_pwms.py first.")

    lengths = load_lengths(args.test_csv)
    obs_raw = pd.read_csv(obs_path)
    bg_raw = pd.read_csv(bg_path)
    if "background_for" in bg_raw.columns:
        split = bg_raw["background_for"].astype(str).str.split("__", n=1, expand=True)
        bg_raw["condition"] = split[0]
        bg_raw["selection"] = split[1]

    obs = add_quality_flags(obs_raw, lengths, args.terminal_exclusion, args.tag_regex)
    bg = add_quality_flags(bg_raw, lengths, args.terminal_exclusion, args.tag_regex)
    obs_valid = obs[obs["valid_strict_window"]].copy()
    bg_valid = bg[bg["valid_strict_window"]].copy()

    obs.to_csv(output_dir / "attention_motif_instances_with_strict_flags.csv", index=False)
    bg.to_csv(output_dir / "matched_background_motif_instances_with_strict_flags.csv", index=False)
    obs_valid.to_csv(output_dir / "attention_motif_instances_strict_valid.csv", index=False)
    bg_valid.to_csv(output_dir / "matched_background_motif_instances_strict_valid.csv", index=False)

    strict = aggregate_counts(obs_valid, bg_valid, args.pseudocount)
    strict["passes_strict_filter"] = (
        (strict["observed_count"] >= args.min_observed_count)
        & (strict["observed_unique_proteins"] >= args.min_observed_proteins)
        & (strict["observed_unique_seeds"] >= args.min_observed_seeds)
        & (strict["log2_enrichment"] >= args.min_log2_enrichment)
        & (strict["observed_fraction"] >= args.min_observed_fraction)
    )
    strict = strict.sort_values(
        ["passes_strict_filter", "condition", "selection", "log2_enrichment", "observed_unique_proteins", "observed_count"],
        ascending=[False, True, True, False, False, False],
    )
    strict.to_csv(output_dir / "strict_kmer_recurrence_enrichment.csv", index=False)
    strict[strict["passes_strict_filter"]].to_csv(output_dir / "strict_kmer_hits.csv", index=False)

    selection_summary = (
        obs_valid.groupby(["condition", "selection"])
        .agg(
            valid_observed_instances=("kmer", "size"),
            valid_observed_proteins=("protein", count_unique),
            valid_observed_seeds=("seed", count_unique),
            mean_neq=("neq", "mean"),
            peak_fraction=("is_neq_peak", "mean"),
            coil_fraction=("ss", lambda x: float(np.mean(pd.Series(x) == "C"))),
            helix_fraction=("ss", lambda x: float(np.mean(pd.Series(x) == "H"))),
            strand_fraction=("ss", lambda x: float(np.mean(pd.Series(x) == "E"))),
        )
        .reset_index()
    )
    hit_counts = (
        strict[strict["passes_strict_filter"]]
        .groupby(["condition", "selection"])
        .size()
        .rename("n_strict_kmer_hits")
        .reset_index()
    )
    selection_summary = selection_summary.merge(hit_counts, on=["condition", "selection"], how="left")
    selection_summary["n_strict_kmer_hits"] = selection_summary["n_strict_kmer_hits"].fillna(0).astype(int)
    selection_summary.to_csv(output_dir / "strict_selection_summary.csv", index=False)

    lines = [
        f"Result root: {result_root}",
        f"Input motif dir: {motif_dir}",
        f"Output dir: {output_dir}",
        f"Observed instances: {len(obs_raw)} raw, {len(obs_valid)} strict-valid",
        f"Background instances: {len(bg_raw)} raw, {len(bg_valid)} strict-valid",
        f"Terminal exclusion: {args.terminal_exclusion} residues",
        f"Tag/poly regex: {args.tag_regex}",
        f"Strict k-mer hits: {int(strict['passes_strict_filter'].sum())}",
        "",
        "Key outputs:",
        "  strict_kmer_hits.csv",
        "  strict_kmer_recurrence_enrichment.csv",
        "  strict_selection_summary.csv",
        "  attention_motif_instances_strict_valid.csv",
        "  matched_background_motif_instances_strict_valid.csv",
    ]
    (output_dir / "strict_motif_postprocess_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
