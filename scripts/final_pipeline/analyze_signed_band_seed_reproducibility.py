#!/usr/bin/env python3
"""Build same-sign seed-consensus bands and block-shift reproducibility nulls.

Inputs are ``signed_bands.csv`` and ``signed_band_protein_summary.csv`` from
extract_signed_contribution_bands.py.  Bands are matched only within the same
condition, split, protein, and sign.  Circular block shifts operate within each
protein's eligible residue interval and preserve a seed's band count, relative
spacing, widths, signs, and magnitudes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


KEY_COLUMNS = ["condition", "split", "protein"]
CONSENSUS_OUTPUT_COLUMNS = [
    "consensus_band_id", "condition", "split", "protein", "protein_length",
    "sign", "label", "seed_support", "n_expected_seeds", "seed_support_fraction",
    "supporting_seeds", "consensus_apex_index_0based",
    "consensus_apex_residue_1based", "consensus_start_index_0based",
    "consensus_end_index_0based_inclusive", "consensus_band_width",
    "member_apex_indices_0based", "max_member_apex_distance",
    "median_apex_signed_column_influence", "median_robust_prominence",
    "member_band_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--protein_summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--min_seeds", type=int, default=2)
    parser.add_argument("--position_tolerance", type=int, default=2)
    parser.add_argument(
        "--max_width_tolerance",
        type=int,
        default=10,
        help="Cap for the half-band-width contribution to apex matching tolerance.",
    )
    parser.add_argument("--n_block_shifts", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=123)
    return parser.parse_args()


def validate_and_load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    bands_path = Path(args.bands_csv).expanduser().resolve()
    summary_path = Path(args.protein_summary_csv).expanduser().resolve()
    bands = pd.read_csv(bands_path)
    summary = pd.read_csv(summary_path)
    band_required = {
        "condition", "seed", "split", "protein", "protein_length", "sign", "label",
        "band_id", "apex_index_0based", "start_index_0based",
        "end_index_0based_inclusive", "band_width",
        "apex_signed_column_influence", "representative_robust_prominence",
    }
    summary_required = {
        "condition", "seed", "split", "protein", "protein_length",
        "eligible_start_index_0based", "eligible_end_index_0based_exclusive",
    }
    missing_bands = band_required - set(bands.columns)
    missing_summary = summary_required - set(summary.columns)
    if missing_bands:
        raise ValueError(f"{bands_path} lacks columns: {sorted(missing_bands)}")
    if missing_summary:
        raise ValueError(f"{summary_path} lacks columns: {sorted(missing_summary)}")
    if bands["band_id"].duplicated().any():
        raise ValueError("band_id values must be unique")
    if summary.duplicated(KEY_COLUMNS + ["seed"]).any():
        raise ValueError("Protein summary has duplicate condition/split/protein/seed rows")
    if not set(bands["sign"].astype(int)).issubset({-1, 1}):
        raise ValueError("Band signs must be -1 or +1")
    for frame in (bands, summary):
        if args.conditions:
            frame.drop(frame.index[~frame["condition"].isin(args.conditions)], inplace=True)
        if args.splits:
            frame.drop(frame.index[~frame["split"].isin(args.splits)], inplace=True)
    if summary.empty:
        raise ValueError("No protein summaries selected")
    insufficient = []
    for (condition, split), group in summary.groupby(["condition", "split"]):
        seed_count = group["seed"].nunique()
        if seed_count < args.min_seeds:
            insufficient.append(f"{condition}/{split}: {seed_count} seed(s)")
    if insufficient:
        raise ValueError(
            "Seed-consensus analysis has fewer than --min_seeds for: "
            + ", ".join(insufficient)
        )
    return bands.reset_index(drop=True), summary.reset_index(drop=True)


def as_band_records(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records") if not frame.empty else []


def match_tolerance(a: dict, b: dict, position_tolerance: int, max_width_tolerance: int) -> int:
    half_width = int(math.ceil(max(float(a["band_width"]), float(b["band_width"])) / 2.0))
    return max(position_tolerance, min(max_width_tolerance, half_width))


def match_band_lists(
    bands_a: list[dict],
    bands_b: list[dict],
    position_tolerance: int,
    max_width_tolerance: int,
) -> list[tuple[int, int, int, float]]:
    if not bands_a or not bands_b:
        return []
    cost = np.full((len(bands_a), len(bands_b)), 1e9, dtype=float)
    distances = np.zeros_like(cost)
    tolerances = np.zeros_like(cost)
    for i, band_a in enumerate(bands_a):
        for j, band_b in enumerate(bands_b):
            distance = abs(
                int(band_a["apex_index_0based"]) - int(band_b["apex_index_0based"])
            )
            tolerance = match_tolerance(
                band_a, band_b, position_tolerance, max_width_tolerance
            )
            distances[i, j] = distance
            tolerances[i, j] = tolerance
            if distance <= tolerance:
                # Tiny deterministic terms prefer prominent matches when distances tie.
                prominence = float(band_a.get("representative_robust_prominence", 0.0)) + float(
                    band_b.get("representative_robust_prominence", 0.0)
                )
                cost[i, j] = distance - min(prominence, 1e3) * 1e-9
    row_indices, column_indices = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_indices, column_indices):
        if cost[i, j] >= 1e8:
            continue
        band_a = bands_a[i]
        band_b = bands_b[j]
        left = max(int(band_a["start_index_0based"]), int(band_b["start_index_0based"]))
        right = min(
            int(band_a["end_index_0based_inclusive"]),
            int(band_b["end_index_0based_inclusive"]),
        )
        intersection = max(0, right - left + 1)
        union = int(band_a["band_width"]) + int(band_b["band_width"]) - intersection
        interval_iou = intersection / union if union else np.nan
        matches.append((int(i), int(j), int(distances[i, j]), float(interval_iou)))
    return matches


def pair_metrics(
    bands_a: list[dict],
    bands_b: list[dict],
    position_tolerance: int,
    max_width_tolerance: int,
) -> dict:
    matches = match_band_lists(
        bands_a, bands_b, position_tolerance, max_width_tolerance
    )
    n_a = len(bands_a)
    n_b = len(bands_b)
    n_match = len(matches)
    denominator = n_a + n_b
    f1 = 2 * n_match / denominator if denominator else np.nan
    jaccard_denominator = n_a + n_b - n_match
    jaccard = n_match / jaccard_denominator if jaccard_denominator else np.nan
    return {
        "n_bands_seed_a": n_a,
        "n_bands_seed_b": n_b,
        "n_matched_bands": n_match,
        "precision_vs_seed_a": n_match / n_a if n_a else np.nan,
        "recall_vs_seed_b": n_match / n_b if n_b else np.nan,
        "f1": f1,
        "jaccard": jaccard,
        "mean_matched_apex_distance": (
            float(np.mean([item[2] for item in matches])) if matches else np.nan
        ),
        "mean_matched_interval_iou": (
            float(np.mean([item[3] for item in matches])) if matches else np.nan
        ),
        "both_seeds_empty": bool(n_a == 0 and n_b == 0),
    }


def aggregate_pair_metrics(rows: list[dict]) -> dict:
    informative = [row for row in rows if row["n_bands_seed_a"] + row["n_bands_seed_b"] > 0]
    total_a = sum(row["n_bands_seed_a"] for row in rows)
    total_b = sum(row["n_bands_seed_b"] for row in rows)
    total_matches = sum(row["n_matched_bands"] for row in rows)
    union = total_a + total_b - total_matches
    f1_values = [row["f1"] for row in informative if np.isfinite(row["f1"])]
    distance_values = []
    for row in rows:
        if row["n_matched_bands"] and np.isfinite(row["mean_matched_apex_distance"]):
            distance_values.extend(
                [row["mean_matched_apex_distance"]] * row["n_matched_bands"]
            )
    return {
        "n_proteins": len(rows),
        "n_informative_proteins": len(informative),
        "total_bands_seed_a": total_a,
        "total_bands_seed_b": total_b,
        "total_matched_bands": total_matches,
        "micro_f1": 2 * total_matches / (total_a + total_b) if total_a + total_b else np.nan,
        "micro_jaccard": total_matches / union if union else np.nan,
        "macro_f1_informative_proteins": float(np.mean(f1_values)) if f1_values else np.nan,
        "mean_matched_apex_distance": float(np.mean(distance_values)) if distance_values else np.nan,
    }


def lookup_bands(
    bands: pd.DataFrame, condition: str, split: str, protein: str, seed: int, sign: int
) -> list[dict]:
    selected = bands[
        (bands["condition"] == condition)
        & (bands["split"] == split)
        & (bands["protein"] == protein)
        & (bands["seed"].astype(int) == int(seed))
        & (bands["sign"].astype(int) == int(sign))
    ]
    return as_band_records(selected)


def consensus_clusters(
    bands: list[dict],
    min_seeds: int,
    position_tolerance: int,
    max_width_tolerance: int,
) -> list[list[dict]]:
    remaining = list(bands)
    clusters = []
    while remaining:
        candidate_clusters = []
        seeds = sorted({int(item["seed"]) for item in remaining})
        for anchor in remaining:
            selected = [anchor]
            anchor_seed = int(anchor["seed"])
            for seed in seeds:
                if seed == anchor_seed:
                    continue
                choices = []
                for candidate in remaining:
                    if int(candidate["seed"]) != seed:
                        continue
                    distance = abs(
                        int(anchor["apex_index_0based"])
                        - int(candidate["apex_index_0based"])
                    )
                    tolerance = match_tolerance(
                        anchor, candidate, position_tolerance, max_width_tolerance
                    )
                    if distance <= tolerance:
                        choices.append((
                            distance,
                            -float(candidate.get("representative_robust_prominence", 0.0)),
                            candidate,
                        ))
                if choices:
                    selected.append(min(choices, key=lambda item: (item[0], item[1]))[2])
            positions = [int(item["apex_index_0based"]) for item in selected]
            score = (
                len({int(item["seed"]) for item in selected}),
                -(max(positions) - min(positions)),
                sum(float(item.get("representative_robust_prominence", 0.0)) for item in selected),
            )
            candidate_clusters.append((score, selected))
        _score, best = max(candidate_clusters, key=lambda item: item[0])
        member_ids = {id(item) for item in best}
        remaining = [item for item in remaining if id(item) not in member_ids]
        if len({int(item["seed"]) for item in best}) >= min_seeds:
            clusters.append(best)
    return clusters


def make_consensus_row(
    cluster: list[dict], condition: str, split: str, protein: str, sign: int,
    protein_length: int, expected_seeds: list[int], number: int,
) -> dict:
    positions = np.asarray([int(item["apex_index_0based"]) for item in cluster])
    starts = np.asarray([int(item["start_index_0based"]) for item in cluster])
    ends = np.asarray([int(item["end_index_0based_inclusive"]) for item in cluster])
    seeds = sorted(int(item["seed"]) for item in cluster)
    apex = int(round(float(np.median(positions))))
    start = min(apex, int(round(float(np.median(starts)))))
    end = max(apex, int(round(float(np.median(ends)))))
    label = "flexibility_supporting" if sign > 0 else "rigidity_supporting"
    return {
        "consensus_band_id": (
            f"{condition}__{split}__{protein}__{label}__consensus_{number:03d}"
        ),
        "condition": condition,
        "split": split,
        "protein": protein,
        "protein_length": protein_length,
        "sign": sign,
        "label": label,
        "seed_support": len(seeds),
        "n_expected_seeds": len(expected_seeds),
        "seed_support_fraction": len(seeds) / len(expected_seeds),
        "supporting_seeds": ",".join(map(str, seeds)),
        "consensus_apex_index_0based": apex,
        "consensus_apex_residue_1based": apex + 1,
        "consensus_start_index_0based": start,
        "consensus_end_index_0based_inclusive": end,
        "consensus_band_width": end - start + 1,
        "member_apex_indices_0based": ",".join(map(str, positions.tolist())),
        "max_member_apex_distance": int(max(positions) - min(positions)),
        "median_apex_signed_column_influence": float(np.median([
            float(item["apex_signed_column_influence"]) for item in cluster
        ])),
        "median_robust_prominence": float(np.median([
            float(item["representative_robust_prominence"]) for item in cluster
        ])),
        "member_band_ids": ";".join(str(item["band_id"]) for item in cluster),
    }


def circular_shift_bands(
    bands: list[dict], eligible_start: int, eligible_end: int, offset: int
) -> list[dict]:
    span = eligible_end - eligible_start
    if span <= 0 or not bands:
        return list(bands)
    shifted = []
    for band in bands:
        item = dict(band)
        old_apex = int(item["apex_index_0based"])
        half_left = old_apex - int(item["start_index_0based"])
        half_right = int(item["end_index_0based_inclusive"]) - old_apex
        new_apex = eligible_start + (
            (old_apex - eligible_start + offset) % span
        )
        item["apex_index_0based"] = new_apex
        # Matching uses apex and width. These approximate limits make null records
        # structurally complete without changing width at circular wrap points.
        item["start_index_0based"] = max(
            eligible_start, new_apex - max(0, half_left)
        )
        item["end_index_0based_inclusive"] = min(
            eligible_end - 1, new_apex + max(0, half_right)
        )
        shifted.append(item)
    return shifted


def empirical_upper_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def z_score(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) < 2 or np.std(null, ddof=1) == 0:
        return np.nan
    return float((observed - np.mean(null)) / np.std(null, ddof=1))


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR adjustment while preserving missing values."""
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return output
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    valid_indices = valid.index.to_numpy()[order]
    output.loc[valid_indices] = adjusted
    return output


def main() -> None:
    args = parse_args()
    if args.min_seeds < 2 or args.position_tolerance < 0 or args.max_width_tolerance < 0:
        raise ValueError("Invalid seed or tolerance settings")
    if args.n_block_shifts < 1:
        raise ValueError("--n_block_shifts must be positive")
    bands, summary = validate_and_load(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    summary_lookup = {
        (row.condition, row.split, row.protein, int(row.seed)): row
        for row in summary.itertuples(index=False)
    }
    protein_seed_lookup = {
        (condition, split, protein): sorted(group["seed"].astype(int).unique().tolist())
        for (condition, split, protein), group in summary.groupby(KEY_COLUMNS, sort=False)
    }
    bands_lookup = {
        (condition, split, protein, int(seed), int(sign)): as_band_records(group)
        for (condition, split, protein, seed, sign), group in bands.groupby(
            KEY_COLUMNS + ["seed", "sign"], sort=False
        )
    }
    protein_groups = list(summary.groupby(KEY_COLUMNS, sort=False))

    pair_rows = []
    consensus_rows = []
    for (condition, split, protein), protein_summary in protein_groups:
        seeds = sorted(protein_summary["seed"].astype(int).unique().tolist())
        protein_length = int(protein_summary["protein_length"].iloc[0])
        for sign in (1, -1):
            all_bands = []
            for seed in seeds:
                all_bands.extend(bands_lookup.get(
                    (condition, split, protein, seed, sign), []
                ))
            clusters = consensus_clusters(
                all_bands,
                args.min_seeds,
                args.position_tolerance,
                args.max_width_tolerance,
            )
            clusters.sort(key=lambda cluster: np.median([
                int(item["apex_index_0based"]) for item in cluster
            ]))
            for number, cluster in enumerate(clusters, start=1):
                consensus_rows.append(make_consensus_row(
                    cluster, condition, split, protein, sign,
                    protein_length, seeds, number,
                ))
            for seed_a, seed_b in itertools.combinations(seeds, 2):
                list_a = bands_lookup.get((condition, split, protein, seed_a, sign), [])
                list_b = bands_lookup.get((condition, split, protein, seed_b, sign), [])
                metrics = pair_metrics(
                    list_a, list_b, args.position_tolerance, args.max_width_tolerance
                )
                pair_rows.append({
                    "condition": condition,
                    "split": split,
                    "protein": protein,
                    "protein_length": protein_length,
                    "sign": sign,
                    "label": "flexibility_supporting" if sign > 0 else "rigidity_supporting",
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    **metrics,
                })

    pair_df = pd.DataFrame(pair_rows)
    consensus_df = pd.DataFrame.from_records(
        consensus_rows, columns=CONSENSUS_OUTPUT_COLUMNS
    )
    pair_df.to_csv(output_dir / "seed_pair_reproducibility_by_protein.csv", index=False)
    consensus_df.to_csv(output_dir / "seed_consensus_bands.csv", index=False)

    pair_null_rows = []
    pair_summary_rows = []
    pair_group_columns = ["condition", "split", "sign", "label", "seed_a", "seed_b"]
    for group_key, observed_group in pair_df.groupby(pair_group_columns, sort=False):
        condition, split, sign, label, seed_a, seed_b = group_key
        observed_rows = observed_group.to_dict(orient="records")
        observed = aggregate_pair_metrics(observed_rows)
        proteins = observed_group["protein"].tolist()
        local_null = []
        for permutation in range(1, args.n_block_shifts + 1):
            permuted_rows = []
            for protein in proteins:
                list_a = bands_lookup.get((condition, split, protein, int(seed_a), int(sign)), [])
                list_b = bands_lookup.get((condition, split, protein, int(seed_b), int(sign)), [])
                summary_b = summary_lookup[(condition, split, protein, int(seed_b))]
                eligible_start = int(summary_b.eligible_start_index_0based)
                eligible_end = int(summary_b.eligible_end_index_0based_exclusive)
                span = eligible_end - eligible_start
                offset = int(rng.integers(0, span)) if span > 0 else 0
                shifted_b = circular_shift_bands(
                    list_b, eligible_start, eligible_end, offset
                )
                permuted_rows.append(pair_metrics(
                    list_a, shifted_b,
                    args.position_tolerance, args.max_width_tolerance,
                ))
            aggregate = aggregate_pair_metrics(permuted_rows)
            row = {
                "condition": condition,
                "split": split,
                "sign": sign,
                "label": label,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "permutation": permutation,
                **aggregate,
            }
            pair_null_rows.append(row)
            local_null.append(row)
        null_micro = np.asarray([row["micro_jaccard"] for row in local_null], dtype=float)
        null_macro = np.asarray([
            row["macro_f1_informative_proteins"] for row in local_null
        ], dtype=float)
        pair_summary_rows.append({
            "condition": condition,
            "split": split,
            "sign": sign,
            "label": label,
            "seed_a": seed_a,
            "seed_b": seed_b,
            **{f"observed_{key}": value for key, value in observed.items()},
            "null_mean_micro_jaccard": float(np.nanmean(null_micro)),
            "null_std_micro_jaccard": float(np.nanstd(null_micro, ddof=1)),
            "micro_jaccard_empirical_p_upper": empirical_upper_p(
                observed["micro_jaccard"], null_micro
            ),
            "micro_jaccard_null_z": z_score(observed["micro_jaccard"], null_micro),
            "null_mean_macro_f1": float(np.nanmean(null_macro)),
            "null_std_macro_f1": float(np.nanstd(null_macro, ddof=1)),
            "macro_f1_empirical_p_upper": empirical_upper_p(
                observed["macro_f1_informative_proteins"], null_macro
            ),
            "macro_f1_null_z": z_score(
                observed["macro_f1_informative_proteins"], null_macro
            ),
        })

    pd.DataFrame(pair_null_rows).to_csv(
        output_dir / "seed_pair_block_shift_null.csv", index=False
    )
    pair_summary_df = pd.DataFrame(pair_summary_rows)
    pair_summary_df["micro_jaccard_empirical_q_bh"] = benjamini_hochberg(
        pair_summary_df["micro_jaccard_empirical_p_upper"]
    )
    pair_summary_df["macro_f1_empirical_q_bh"] = benjamini_hochberg(
        pair_summary_df["macro_f1_empirical_p_upper"]
    )
    pair_summary_df.to_csv(
        output_dir / "seed_pair_reproducibility_summary.csv", index=False
    )

    consensus_null_rows = []
    consensus_summary_rows = []
    condition_split_groups = summary.groupby(["condition", "split"], sort=False)
    for (condition, split), condition_summary in condition_split_groups:
        proteins = condition_summary["protein"].drop_duplicates().tolist()
        for sign in (1, -1):
            observed_count = int((
                (consensus_df["condition"] == condition)
                & (consensus_df["split"] == split)
                & (consensus_df["sign"].astype(int) == sign)
            ).sum()) if not consensus_df.empty else 0
            null_counts = []
            for permutation in range(1, args.n_block_shifts + 1):
                count = 0
                for protein in proteins:
                    seeds = protein_seed_lookup[(condition, split, protein)]
                    shifted_all = []
                    for seed_index, seed in enumerate(seeds):
                        values = bands_lookup.get(
                            (condition, split, protein, seed, sign), []
                        )
                        summary_seed = summary_lookup[(condition, split, protein, seed)]
                        eligible_start = int(summary_seed.eligible_start_index_0based)
                        eligible_end = int(summary_seed.eligible_end_index_0based_exclusive)
                        span = eligible_end - eligible_start
                        # Fix the first seed and shift the other full band patterns.
                        offset = 0 if seed_index == 0 or span <= 0 else int(rng.integers(0, span))
                        shifted_all.extend(circular_shift_bands(
                            values, eligible_start, eligible_end, offset
                        ))
                    count += len(consensus_clusters(
                        shifted_all,
                        args.min_seeds,
                        args.position_tolerance,
                        args.max_width_tolerance,
                    ))
                null_counts.append(count)
                consensus_null_rows.append({
                    "condition": condition,
                    "split": split,
                    "sign": sign,
                    "label": "flexibility_supporting" if sign > 0 else "rigidity_supporting",
                    "permutation": permutation,
                    "consensus_band_count": count,
                })
            null_values = np.asarray(null_counts, dtype=float)
            consensus_summary_rows.append({
                "condition": condition,
                "split": split,
                "sign": sign,
                "label": "flexibility_supporting" if sign > 0 else "rigidity_supporting",
                "n_proteins": len(proteins),
                "observed_consensus_band_count": observed_count,
                "null_mean_consensus_band_count": float(np.mean(null_values)),
                "null_std_consensus_band_count": float(np.std(null_values, ddof=1)),
                "consensus_count_empirical_p_upper": empirical_upper_p(
                    observed_count, null_values
                ),
                "consensus_count_null_z": z_score(observed_count, null_values),
            })

    pd.DataFrame(consensus_null_rows).to_csv(
        output_dir / "consensus_block_shift_null.csv", index=False
    )
    consensus_summary_df = pd.DataFrame(consensus_summary_rows)
    consensus_summary_df["consensus_count_empirical_q_bh"] = benjamini_hochberg(
        consensus_summary_df["consensus_count_empirical_p_upper"]
    )
    consensus_summary_df.to_csv(
        output_dir / "consensus_reproducibility_summary.csv", index=False
    )
    parameters = {
        "bands_csv": str(Path(args.bands_csv).expanduser().resolve()),
        "protein_summary_csv": str(Path(args.protein_summary_csv).expanduser().resolve()),
        "matching": "same condition, split, protein, and sign; one-to-one Hungarian matching by apex distance",
        "match_tolerance": (
            "max(position_tolerance, min(max_width_tolerance, "
            "ceil(max(band_width_a, band_width_b)/2)))"
        ),
        "min_seeds": args.min_seeds,
        "position_tolerance": args.position_tolerance,
        "max_width_tolerance": args.max_width_tolerance,
        "n_block_shifts": args.n_block_shifts,
        "random_seed": args.random_seed,
        "block_shift": (
            "circularly shift each seed's complete same-sign band pattern within the "
            "eligible protein interval; preserve count, spacing, widths, sign, and magnitude"
        ),
        "n_consensus_bands": len(consensus_rows),
        "n_seed_pair_protein_rows": len(pair_rows),
    }
    (output_dir / "seed_reproducibility_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n"
    )
    print(json.dumps({
        "consensus_bands": len(consensus_rows),
        "seed_pair_protein_rows": len(pair_rows),
        "pairwise_null_rows": len(pair_null_rows),
        "consensus_null_rows": len(consensus_null_rows),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
