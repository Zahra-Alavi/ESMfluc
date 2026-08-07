#!/usr/bin/env python3
"""Measure seed reproducibility and write the mean-anchored stable catalog.

Inputs are the per-seed bands, per-seed protein summary, and mean-profile bands
written by ``extract_signed_contribution_bands``. Same-sign bands are matched
within condition, split, and protein by substantial interval overlap. The
script writes pairwise reproducibility diagnostics and the primary stable
catalog: mean-profile bands supported by at least two of three seeds. A 3-of-3
flag is retained in that same catalog, and apex displacement remains
descriptive metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


KEY_COLUMNS = ["condition", "split", "protein"]
GROUP_COLUMNS = KEY_COLUMNS + ["sign"]
MEAN_REQUIRED = {
    "condition", "split", "protein", "protein_length", "sign", "label",
    "apex_id", "apex_index_0based", "apex_residue_1based",
    "apex_signed_column_influence", "representative_robust_prominence",
    "support_start_index_0based", "support_end_index_0based_inclusive",
    "support_width",
}
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
    parser.add_argument(
        "--mean_bands_csv",
        required=True,
        help="Mean-profile signed_bands.csv written by the extractor.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--min_seeds", type=int, default=2)
    parser.add_argument("--position_tolerance", type=int, default=2)
    parser.add_argument(
        "--matching_method",
        choices=["interval_iou", "fixed_apex", "legacy_width"],
        default="interval_iou",
        help=(
            "interval_iou is the Phase 1 primary band-stability method. "
            "fixed_apex and legacy_width remain available for sensitivity and "
            "historical reproduction."
        ),
    )
    parser.add_argument(
        "--min_interval_iou",
        type=float,
        default=0.5,
        help=(
            "Minimum intersection-over-union for interval_iou matching. "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--max_width_tolerance",
        type=int,
        default=10,
        help="Used only with --matching_method legacy_width.",
    )
    parser.add_argument("--n_block_shifts", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=123)
    return parser.parse_args()


def validate_and_load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    bands_path = Path(args.bands_csv).expanduser().resolve()
    summary_path = Path(args.protein_summary_csv).expanduser().resolve()
    bands = pd.read_csv(bands_path)
    summary = pd.read_csv(summary_path)
    aliases = {
        "apex_id": "band_id",
        "support_start_index_0based": "start_index_0based",
        "support_end_index_0based_inclusive": "end_index_0based_inclusive",
        "support_width": "band_width",
    }
    for source, destination in aliases.items():
        if destination not in bands and source in bands:
            bands[destination] = bands[source]
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


def load_mean_bands(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.mean_bands_csv).expanduser().resolve()
    mean_bands = pd.read_csv(path)
    missing = MEAN_REQUIRED - set(mean_bands.columns)
    if missing:
        raise ValueError(f"{path} lacks mean-band columns: {sorted(missing)}")
    if mean_bands["apex_id"].duplicated().any():
        raise ValueError("Mean-profile apex_id values must be unique")
    if args.conditions:
        mean_bands = mean_bands[mean_bands["condition"].isin(args.conditions)]
    if args.splits:
        mean_bands = mean_bands[mean_bands["split"].isin(args.splits)]
    if mean_bands.empty:
        raise ValueError("No mean-profile bands selected")
    return mean_bands.reset_index(drop=True)


def _active_extractor_config(parameters: dict) -> dict:
    method = parameters.get("apex_method")
    support_method = parameters.get("support_method")
    common_keys = [
        "apex_method",
        "support_method",
        "terminal_exclusion",
        "terminal_exclusion_fraction",
    ]
    if method == "raw_mad_amplitude":
        active_keys = common_keys + ["amplitude_mad"]
    elif method == "multiscale_prominence":
        active_keys = common_keys + [
            "smooth_windows",
            "min_scales",
            "scale_tolerance",
            "prominence_mad",
            "min_peak_distance",
        ]
    else:
        raise ValueError(f"Unknown extractor apex_method: {method!r}")
    missing = [key for key in active_keys if key not in parameters]
    if missing:
        raise ValueError(
            f"Extractor parameter file lacks active fields: {missing}"
        )
    if support_method is None:
        raise ValueError("Extractor parameter file lacks support_method")
    return {key: parameters[key] for key in active_keys}


def validated_parameter_set(
    args: argparse.Namespace,
) -> tuple[str, dict, dict[str, str]]:
    """Validate extractor metadata and derive a truthful parameter-set ID."""
    paths = {
        "per_seed": (
            Path(args.bands_csv).expanduser().resolve().parent
            / "signed_band_parameters.json"
        ),
        "mean": (
            Path(args.mean_bands_csv).expanduser().resolve().parent
            / "signed_band_parameters.json"
        ),
    }
    loaded = {}
    for label, path in paths.items():
        if not path.exists():
            raise ValueError(
                f"Missing {label} extractor parameter file beside its CSV: {path}"
            )
        loaded[label] = json.loads(path.read_text())
    per_seed_config = _active_extractor_config(loaded["per_seed"])
    mean_config = _active_extractor_config(loaded["mean"])
    if per_seed_config != mean_config:
        raise ValueError(
            "Per-seed and mean-profile extractor settings do not match: "
            f"{per_seed_config} != {mean_config}"
        )
    influence_field_pairs = {
        "signed_column_influence": "seed_averaged_signed_column_influence",
        "observed_signed_influence": (
            "seed_averaged_observed_signed_influence"
        ),
        "uniform_signed_influence": (
            "seed_averaged_uniform_signed_influence"
        ),
        "shifted_attention_signed_influence": (
            "seed_averaged_shifted_attention_signed_influence"
        ),
    }
    per_seed_field = loaded["per_seed"].get("influence_field")
    mean_field = loaded["mean"].get("influence_field")
    if per_seed_field not in influence_field_pairs:
        raise ValueError(
            "Unsupported per-seed influence field for stability analysis: "
            f"{per_seed_field!r}"
        )
    if mean_field != influence_field_pairs[per_seed_field]:
        raise ValueError(
            "Per-seed and mean-profile influence fields are not a matched "
            f"profile pair: {per_seed_field!r}, {mean_field!r}"
        )
    if loaded["per_seed"].get("selected_proteins") != loaded["mean"].get(
        "selected_proteins"
    ):
        raise ValueError(
            "Per-seed and mean-profile selected_proteins metadata differ"
        )
    canonical = json.dumps(
        per_seed_config, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    if per_seed_config["apex_method"] == "raw_mad_amplitude":
        threshold = format(float(per_seed_config["amplitude_mad"]), "g")
        readable = (
            f"raw_amp_R{threshold}_{per_seed_config['support_method']}"
        )
    else:
        readable = (
            f"multiscale_prominence_{per_seed_config['support_method']}"
        )
    parameter_set_id = f"phase1_{readable}_{digest}"
    return (
        parameter_set_id,
        per_seed_config,
        {label: str(path) for label, path in paths.items()},
    )


def fixed_apex_matches(
    mean_group: pd.DataFrame,
    seed_group: pd.DataFrame,
    tolerance: int,
) -> dict[int, int]:
    """Return one-to-one mean-row-index -> seed-row-index matches."""
    if mean_group.empty or seed_group.empty:
        return {}
    mean_positions = mean_group["apex_index_0based"].astype(int).to_numpy()
    seed_positions = seed_group["apex_index_0based"].astype(int).to_numpy()
    distances = np.abs(mean_positions[:, None] - seed_positions[None, :])
    cost = distances.astype(float)
    cost[distances > tolerance] = 1e9
    seed_amplitudes = np.abs(
        seed_group["apex_signed_column_influence"].astype(float).to_numpy()
    )
    seed_priority = sorted(
        range(len(seed_group)),
        key=lambda index: (-seed_amplitudes[index], seed_positions[index]),
    )
    priority_penalty = np.empty(len(seed_group), dtype=float)
    for rank, index in enumerate(seed_priority):
        priority_penalty[index] = rank * 1e-6 / (len(seed_group) + 1)
    cost += priority_penalty[None, :]
    cost[distances > tolerance] = 1e9
    row_indices, column_indices = linear_sum_assignment(cost)
    mean_labels = mean_group.index.to_numpy()
    seed_labels = seed_group.index.to_numpy()
    return {
        int(mean_labels[row]): int(seed_labels[column])
        for row, column in zip(row_indices, column_indices)
        if distances[row, column] <= tolerance
    }


def interval_iou_values(
    starts_a: np.ndarray,
    ends_a: np.ndarray,
    starts_b: np.ndarray,
    ends_b: np.ndarray,
) -> np.ndarray:
    """Return pairwise inclusive-interval intersection-over-union."""
    intersections = np.maximum(
        0,
        np.minimum(ends_a[:, None], ends_b[None, :])
        - np.maximum(starts_a[:, None], starts_b[None, :])
        + 1,
    )
    widths_a = ends_a - starts_a + 1
    widths_b = ends_b - starts_b + 1
    unions = widths_a[:, None] + widths_b[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )


def interval_iou_matches(
    mean_group: pd.DataFrame,
    seed_group: pd.DataFrame,
    min_interval_iou: float,
) -> dict[int, tuple[int, float]]:
    """Return one-to-one mean-row -> (seed-row, IoU) interval matches."""
    if mean_group.empty or seed_group.empty:
        return {}
    mean_starts = mean_group["support_start_index_0based"].astype(int).to_numpy()
    mean_ends = mean_group[
        "support_end_index_0based_inclusive"
    ].astype(int).to_numpy()
    seed_starts = seed_group["start_index_0based"].astype(int).to_numpy()
    seed_ends = seed_group[
        "end_index_0based_inclusive"
    ].astype(int).to_numpy()
    ious = interval_iou_values(
        mean_starts, mean_ends, seed_starts, seed_ends
    )
    mean_positions = mean_group["apex_index_0based"].astype(int).to_numpy()
    seed_positions = seed_group["apex_index_0based"].astype(int).to_numpy()
    distances = np.abs(mean_positions[:, None] - seed_positions[None, :])
    cost = 1.0 - ious
    cost += distances * 1e-6
    cost[ious < min_interval_iou] = 1e9
    row_indices, column_indices = linear_sum_assignment(cost)
    mean_labels = mean_group.index.to_numpy()
    seed_labels = seed_group.index.to_numpy()
    return {
        int(mean_labels[row]): (
            int(seed_labels[column]),
            float(ious[row, column]),
        )
        for row, column in zip(row_indices, column_indices)
        if ious[row, column] >= min_interval_iou
    }


def audit_mean_intervals(mean_bands: pd.DataFrame) -> dict:
    starts = mean_bands["support_start_index_0based"].astype(int)
    ends = mean_bands["support_end_index_0based_inclusive"].astype(int)
    apices = mean_bands["apex_index_0based"].astype(int)
    widths = mean_bands["support_width"].astype(int)
    if not ((starts <= apices) & (apices <= ends)).all():
        raise ValueError("At least one mean-profile interval excludes its apex")
    if not (widths == ends - starts + 1).all():
        raise ValueError("At least one mean-profile interval width is invalid")
    overlap_count = 0
    for _key, group in mean_bands.groupby(KEY_COLUMNS, sort=False):
        ordered = group.sort_values(
            ["support_start_index_0based", "support_end_index_0based_inclusive"]
        )
        records = list(ordered.itertuples(index=False))
        for left_index, left in enumerate(records):
            for right in records[left_index + 1:]:
                if int(right.support_start_index_0based) > int(
                    left.support_end_index_0based_inclusive
                ):
                    break
                overlap_count += 1
    if overlap_count:
        raise ValueError(
            f"Mean-profile interval catalog contains {overlap_count} overlaps"
        )
    return {
        "n_mean_profile_intervals": int(len(mean_bands)),
        "all_intervals_contain_apex": True,
        "all_interval_widths_exact": True,
        "n_overlapping_interval_pairs": 0,
    }


def write_stable_catalog(
    mean_bands: pd.DataFrame,
    seed_bands: pd.DataFrame,
    seed_summary: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    parameter_set_id: str,
) -> tuple[pd.DataFrame, dict, Path]:
    """Match mean bands to seeds and write the primary stable catalog."""
    interval_audit = audit_mean_intervals(mean_bands)
    seed_bands = seed_bands.copy()
    seed_bands["seed"] = pd.to_numeric(
        seed_bands["seed"], errors="raise"
    ).astype(int)
    seed_summary = seed_summary.copy()
    seed_summary["seed"] = pd.to_numeric(
        seed_summary["seed"], errors="raise"
    ).astype(int)
    expected_lookup = {
        key: sorted(group["seed"].unique().tolist())
        for key, group in seed_summary.groupby(KEY_COLUMNS, sort=False)
    }
    unexpected = {
        key: seeds for key, seeds in expected_lookup.items() if len(seeds) != 3
    }
    if unexpected:
        raise ValueError(
            "Stable catalog requires exactly three seeds per protein; "
            f"examples: {list(unexpected.items())[:5]}"
        )

    seed_groups = {
        key: group
        for key, group in seed_bands.groupby(
            GROUP_COLUMNS + ["seed"], sort=False
        )
    }
    matches_by_mean: dict[int, dict[int, tuple[int, float]]] = {
        int(index): {} for index in mean_bands.index
    }
    for group_key, mean_group in mean_bands.groupby(GROUP_COLUMNS, sort=False):
        condition, split, protein, sign = group_key
        expected = expected_lookup.get((condition, split, protein))
        if expected is None:
            raise ValueError(
                f"No per-seed summary for {condition}/{split}/{protein}"
            )
        for seed in expected:
            seed_group = seed_groups.get(
                (condition, split, protein, sign, seed),
                seed_bands.iloc[0:0],
            )
            if args.matching_method == "interval_iou":
                matches = interval_iou_matches(
                    mean_group, seed_group, args.min_interval_iou
                )
            else:
                matches = {
                    mean_index: (seed_index, np.nan)
                    for mean_index, seed_index in fixed_apex_matches(
                        mean_group, seed_group, args.position_tolerance
                    ).items()
                }
            for mean_index, match in matches.items():
                matches_by_mean[mean_index][seed] = match

    records = []
    for mean_index, mean_row in mean_bands.iterrows():
        row = mean_row.to_dict()
        expected = expected_lookup[
            (row["condition"], row["split"], row["protein"])
        ]
        matched = matches_by_mean[int(mean_index)]
        supporting_seeds = sorted(matched)
        displacements = []
        interval_ious = []
        positions = []
        for seed in expected:
            seed_row = (
                seed_bands.loc[matched[seed][0]] if seed in matched else None
            )
            supported = seed_row is not None
            position = (
                int(seed_row["apex_index_0based"]) if supported else np.nan
            )
            displacement = (
                abs(position - int(row["apex_index_0based"]))
                if supported else np.nan
            )
            row[f"seed_{seed}_supported"] = supported
            row[f"seed_{seed}_apex_index_0based"] = position
            row[f"seed_{seed}_displacement"] = displacement
            row[f"seed_{seed}_support_start_index_0based"] = (
                int(seed_row["start_index_0based"]) if supported else np.nan
            )
            row[f"seed_{seed}_support_end_index_0based_inclusive"] = (
                int(seed_row["end_index_0based_inclusive"])
                if supported else np.nan
            )
            interval_iou = (
                float(matched[seed][1]) if supported else np.nan
            )
            row[f"seed_{seed}_interval_iou"] = interval_iou
            if supported:
                positions.append(str(position))
                displacements.append(float(displacement))
                if np.isfinite(interval_iou):
                    interval_ious.append(interval_iou)
        support_count = len(supporting_seeds)
        row.update({
            "seed_support_count": support_count,
            "seed_support_fraction": support_count / len(expected),
            "supporting_seeds": ",".join(map(str, supporting_seeds)),
            "seed_apex_positions_0based": ",".join(positions),
            "maximum_seed_displacement": (
                max(displacements) if displacements else np.nan
            ),
            "mean_seed_displacement": (
                float(np.mean(displacements)) if displacements else np.nan
            ),
            "minimum_seed_interval_iou": (
                min(interval_ious) if interval_ious else np.nan
            ),
            "mean_seed_interval_iou": (
                float(np.mean(interval_ious)) if interval_ious else np.nan
            ),
            "primary_stable_band": support_count >= args.min_seeds,
            "strict_stable_band": support_count == len(expected),
            "stability_class": (
                "strict_3_of_3"
                if support_count == len(expected)
                else "primary_2_of_3"
                if support_count >= args.min_seeds
                else "unstable_0_or_1"
            ),
            "locked_parameter_set_id": parameter_set_id,
            "matching_method": (
                "one-to-one same-sign interval-IoU Hungarian"
                if args.matching_method == "interval_iou"
                else "one-to-one fixed-position Hungarian; ties use absolute "
                "apex amplitude then residue index"
            ),
            "matching_tolerance_residues": (
                args.position_tolerance
                if args.matching_method != "interval_iou"
                else np.nan
            ),
            "matching_min_interval_iou": (
                args.min_interval_iou
                if args.matching_method == "interval_iou"
                else np.nan
            ),
        })
        records.append(row)

    all_candidates = pd.DataFrame(records)
    stable = all_candidates[all_candidates["primary_stable_band"]].copy()
    all_candidates_path = output_dir / "all_mean_bands_with_stability.csv.gz"
    all_candidates.to_csv(
        all_candidates_path, index=False, compression="gzip"
    )
    stable_path = output_dir / "stable_signed_bands.csv"
    stable.to_csv(stable_path, index=False)
    audit = {
        "n_mean_profile_bands": int(len(all_candidates)),
        "n_primary_stable_bands": int(len(stable)),
        "n_strict_3of3_bands": int(stable["strict_stable_band"].sum()),
        "n_unstable_mean_bands": int(
            (~all_candidates["primary_stable_band"]).sum()
        ),
        "all_mean_bands_with_stability_csv_gz": str(all_candidates_path),
        "primary_definition": f">={args.min_seeds} of 3 seeds",
        "strict_definition": "3 of 3 seeds",
        "matching": (
            "same condition/split/protein/sign; one-to-one interval-IoU "
            f"Hungarian matching with IoU >= {args.min_interval_iou}"
            if args.matching_method == "interval_iou"
            else "same condition/split/protein/sign; one-to-one fixed "
            f"+/-{args.position_tolerance}-residue Hungarian matching; ties "
            "use absolute apex amplitude then residue index"
        ),
        "apex_displacement_role": "descriptive only",
        "mean_profile_intervals_recomputed_after_matching": False,
        "locked_parameter_set_id": parameter_set_id,
        "interval_audit": interval_audit,
    }
    return stable, audit, stable_path


def as_band_records(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records") if not frame.empty else []


CIRCULAR_SEGMENTS_FIELD = "_circular_support_segments_0based_inclusive"
CIRCULAR_ELIGIBLE_START_FIELD = "_circular_eligible_start_index_0based"
CIRCULAR_ELIGIBLE_END_FIELD = "_circular_eligible_end_index_0based_exclusive"


def band_support_segments(band: dict) -> tuple[tuple[int, int], ...]:
    """Return one or two nonoverlapping linear support segments."""
    encoded = band.get(CIRCULAR_SEGMENTS_FIELD)
    if encoded is not None:
        segments = tuple((int(start), int(end)) for start, end in encoded)
    else:
        start = int(band["start_index_0based"])
        end = int(band["end_index_0based_inclusive"])
        if start > end:
            raise ValueError(
                "A wrapped band requires explicit circular support segments"
            )
        segments = ((start, end),)
    if not segments:
        raise ValueError("A band must contain at least one support segment")
    previous_end = None
    for start, end in segments:
        if start > end:
            raise ValueError("Support segment start exceeds its end")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Support segments must be sorted and nonoverlapping")
        previous_end = end
    return segments


def band_support_width(band: dict) -> int:
    """Reconstruct support width from segments, never from cached width."""
    return sum(end - start + 1 for start, end in band_support_segments(band))


def band_interval_iou(a: dict, b: dict) -> float:
    """Return residue-set IoU for ordinary or wrapped band supports."""
    segments_a = band_support_segments(a)
    segments_b = band_support_segments(b)
    intersection = 0
    for start_a, end_a in segments_a:
        for start_b, end_b in segments_b:
            intersection += max(
                0, min(end_a, end_b) - max(start_a, start_b) + 1
            )
    width_a = band_support_width(a)
    width_b = band_support_width(b)
    union = width_a + width_b - intersection
    return intersection / union if union else np.nan


def band_apex_distance(a: dict, b: dict) -> int:
    """Return linear observed distance or circular null-shift distance."""
    apex_a = int(a["apex_index_0based"])
    apex_b = int(b["apex_index_0based"])
    starts = {
        item.get(CIRCULAR_ELIGIBLE_START_FIELD)
        for item in (a, b)
        if item.get(CIRCULAR_ELIGIBLE_START_FIELD) is not None
    }
    ends = {
        item.get(CIRCULAR_ELIGIBLE_END_FIELD)
        for item in (a, b)
        if item.get(CIRCULAR_ELIGIBLE_END_FIELD) is not None
    }
    if not starts and not ends:
        return abs(apex_a - apex_b)
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Circular bands use inconsistent eligible intervals")
    span = int(next(iter(ends))) - int(next(iter(starts)))
    direct = abs(apex_a - apex_b)
    return min(direct, span - direct)


def match_tolerance(
    a: dict,
    b: dict,
    position_tolerance: int,
    max_width_tolerance: int,
    matching_method: str = "fixed_apex",
) -> int:
    if matching_method == "fixed_apex":
        return int(position_tolerance)
    if matching_method != "legacy_width":
        raise ValueError(f"Unknown matching method: {matching_method}")
    half_width = int(math.ceil(max(float(a["band_width"]), float(b["band_width"])) / 2.0))
    return max(position_tolerance, min(max_width_tolerance, half_width))


def match_band_lists(
    bands_a: list[dict],
    bands_b: list[dict],
    position_tolerance: int,
    max_width_tolerance: int,
    matching_method: str = "fixed_apex",
    min_interval_iou: float = 0.5,
) -> list[tuple[int, int, int, float]]:
    if not bands_a or not bands_b:
        return []
    cost = np.full((len(bands_a), len(bands_b)), 1e9, dtype=float)
    distances = np.zeros_like(cost)
    valid_pairs = []
    for i, band_a in enumerate(bands_a):
        for j, band_b in enumerate(bands_b):
            distance = band_apex_distance(band_a, band_b)
            distances[i, j] = distance
            interval_iou = band_interval_iou(band_a, band_b)
            if matching_method == "interval_iou":
                valid = interval_iou >= min_interval_iou
                primary_cost = 1.0 - interval_iou
            else:
                tolerance = match_tolerance(
                    band_a,
                    band_b,
                    position_tolerance,
                    max_width_tolerance,
                    matching_method,
                )
                valid = distance <= tolerance
                primary_cost = float(distance)
            if valid:
                combined_amplitude = (
                    abs(float(band_a["apex_signed_column_influence"]))
                    + abs(float(band_b["apex_signed_column_influence"]))
                )
                valid_pairs.append((
                    i,
                    j,
                    primary_cost,
                    combined_amplitude,
                    int(band_a["apex_index_0based"]),
                    int(band_b["apex_index_0based"]),
                ))
    ordered_pairs = sorted(
        valid_pairs,
        key=lambda item: (item[2], -item[3], item[4], item[5]),
    )
    for rank, (
        i, j, primary_cost, _amplitude, _position_a, _position_b
    ) in enumerate(
        ordered_pairs
    ):
        tie_penalty = rank * 1e-6 / (len(ordered_pairs) + 1)
        cost[i, j] = primary_cost + tie_penalty
    row_indices, column_indices = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_indices, column_indices):
        if cost[i, j] >= 1e8:
            continue
        band_a = bands_a[i]
        band_b = bands_b[j]
        interval_iou = band_interval_iou(band_a, band_b)
        matches.append((int(i), int(j), int(distances[i, j]), float(interval_iou)))
    return matches


def pair_metrics(
    bands_a: list[dict],
    bands_b: list[dict],
    position_tolerance: int,
    max_width_tolerance: int,
    matching_method: str = "fixed_apex",
    min_interval_iou: float = 0.5,
) -> dict:
    matches = match_band_lists(
        bands_a,
        bands_b,
        position_tolerance,
        max_width_tolerance,
        matching_method,
        min_interval_iou,
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
    interval_iou_values_matched = []
    for row in rows:
        if row["n_matched_bands"] and np.isfinite(row["mean_matched_apex_distance"]):
            distance_values.extend(
                [row["mean_matched_apex_distance"]] * row["n_matched_bands"]
            )
        if row["n_matched_bands"] and np.isfinite(
            row["mean_matched_interval_iou"]
        ):
            interval_iou_values_matched.extend(
                [row["mean_matched_interval_iou"]] * row["n_matched_bands"]
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
        "mean_matched_interval_iou": (
            float(np.mean(interval_iou_values_matched))
            if interval_iou_values_matched else np.nan
        ),
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
    matching_method: str = "fixed_apex",
    min_interval_iou: float = 0.5,
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
                    distance = band_apex_distance(anchor, candidate)
                    interval_iou = band_interval_iou(anchor, candidate)
                    if matching_method == "interval_iou":
                        valid = interval_iou >= min_interval_iou
                        match_priority = -interval_iou
                    else:
                        tolerance = match_tolerance(
                            anchor,
                            candidate,
                            position_tolerance,
                            max_width_tolerance,
                            matching_method,
                        )
                        valid = distance <= tolerance
                        match_priority = float(distance)
                    if valid:
                        choices.append((
                            match_priority,
                            distance,
                            -abs(float(candidate["apex_signed_column_influence"])),
                            int(candidate["apex_index_0based"]),
                            candidate,
                        ))
                if choices:
                    selected.append(
                        min(choices, key=lambda item: item[:4])[4]
                    )
            positions = [int(item["apex_index_0based"]) for item in selected]
            score = (
                len({int(item["seed"]) for item in selected}),
                -(max(positions) - min(positions)),
                sum(abs(float(item["apex_signed_column_influence"]))
                    for item in selected),
                -sum(positions),
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
    bands: list[dict],
    eligible_start: int,
    eligible_end: int,
    offset: int,
    shift_intervals: bool = False,
) -> list[dict]:
    """Shift a complete band pattern, wrapping support without clipping."""
    span = eligible_end - eligible_start
    if span <= 0 or not bands:
        return list(bands)
    offset = int(offset) % span
    shifted = []
    for band in bands:
        item = dict(band)
        old_apex = int(item["apex_index_0based"])
        new_apex = eligible_start + (
            (old_apex - eligible_start + offset) % span
        )
        item["apex_index_0based"] = new_apex
        if "apex_residue_1based" in item:
            item["apex_residue_1based"] = new_apex + 1
        if shift_intervals:
            original_segments = band_support_segments(band)
            original_positions = [
                position
                for start, end in original_segments
                for position in range(start, end + 1)
            ]
            if any(
                position < eligible_start or position >= eligible_end
                for position in original_positions
            ):
                raise ValueError("Band support falls outside its eligible interval")
            original_width = len(original_positions)
            if original_width != int(band["band_width"]):
                raise ValueError(
                    "Band width disagrees with its reconstructed support"
                )
            shifted_positions = sorted({
                eligible_start
                + ((position - eligible_start + offset) % span)
                for position in original_positions
            })
            if len(shifted_positions) != original_width:
                raise ValueError("Circular shifting changed band support width")
            segments = []
            segment_start = segment_end = shifted_positions[0]
            for position in shifted_positions[1:]:
                if position == segment_end + 1:
                    segment_end = position
                else:
                    segments.append((segment_start, segment_end))
                    segment_start = segment_end = position
            segments.append((segment_start, segment_end))
            if len(segments) > 2:
                raise ValueError(
                    "A shifted contiguous circular band produced >2 segments"
                )
            old_start = int(item["start_index_0based"])
            old_end = int(item["end_index_0based_inclusive"])
            new_start = eligible_start + (
                (old_start - eligible_start + offset) % span
            )
            new_end = eligible_start + (
                (old_end - eligible_start + offset) % span
            )
            item["start_index_0based"] = new_start
            item["end_index_0based_inclusive"] = new_end
            if "start_residue_1based" in item:
                item["start_residue_1based"] = new_start + 1
            if "end_residue_1based_inclusive" in item:
                item["end_residue_1based_inclusive"] = new_end + 1
            if "support_start_index_0based" in item:
                item["support_start_index_0based"] = new_start
            if "support_end_index_0based_inclusive" in item:
                item["support_end_index_0based_inclusive"] = new_end
            if "support_start_residue_1based" in item:
                item["support_start_residue_1based"] = new_start + 1
            if "support_end_residue_1based_inclusive" in item:
                item["support_end_residue_1based_inclusive"] = new_end + 1
            item[CIRCULAR_SEGMENTS_FIELD] = tuple(segments)
            item[CIRCULAR_ELIGIBLE_START_FIELD] = eligible_start
            item[CIRCULAR_ELIGIBLE_END_FIELD] = eligible_end
            if band_support_width(item) != original_width:
                raise ValueError("Shifted segment reconstruction changed width")
            if not any(
                start <= new_apex <= end
                for start, end in band_support_segments(item)
            ):
                raise ValueError("Shifted apex falls outside shifted support")
        shifted.append(item)
    return shifted


def audit_circular_shift_invariants(
    bands: pd.DataFrame,
    summary: pd.DataFrame,
    shift_intervals: bool,
) -> dict:
    """Smoke-audit representative offsets for every protein/seed/sign pattern."""
    if not shift_intervals:
        return {
            "performed": False,
            "reason": "selected matching method does not use interval support",
        }
    summary_lookup = {
        (row.condition, row.split, row.protein, int(row.seed)): row
        for row in summary.itertuples(index=False)
    }
    magnitude_fields = [
        field for field in (
            "apex_signed_column_influence",
            "apex_absolute_influence",
            "band_integrated_magnitude",
            "band_absolute_influence_sum",
            "support_absolute_influence_sum",
        )
        if field in bands.columns
    ]
    patterns_checked = 0
    band_shifts_checked = 0
    wrapped_band_shifts = 0
    offsets_checked = 0
    for key, group in bands.groupby(
        KEY_COLUMNS + ["seed", "sign"], sort=False
    ):
        condition, split, protein, seed, _sign = key
        summary_row = summary_lookup[
            (condition, split, protein, int(seed))
        ]
        eligible_start = int(summary_row.eligible_start_index_0based)
        eligible_end = int(summary_row.eligible_end_index_0based_exclusive)
        span = eligible_end - eligible_start
        if span <= 0:
            continue
        records = as_band_records(group)
        offsets = sorted({0, 1 % span, (span // 2) % span, (span - 1) % span})
        original_apices = [
            int(item["apex_index_0based"]) for item in records
        ]
        for offset in offsets:
            shifted = circular_shift_bands(
                records, eligible_start, eligible_end, offset, True
            )
            if len(shifted) != len(records):
                raise ValueError("Circular shift changed band count")
            if records:
                original_relative = [
                    (apex - original_apices[0]) % span
                    for apex in original_apices
                ]
                shifted_apices = [
                    int(item["apex_index_0based"]) for item in shifted
                ]
                shifted_relative = [
                    (apex - shifted_apices[0]) % span
                    for apex in shifted_apices
                ]
                if shifted_relative != original_relative:
                    raise ValueError(
                        "Circular shift changed relative apex spacing"
                    )
            for original, moved in zip(records, shifted):
                if band_support_width(moved) != band_support_width(original):
                    raise ValueError("Circular shift changed support width")
                if int(moved["band_width"]) != int(original["band_width"]):
                    raise ValueError("Circular shift changed stored band width")
                if int(moved["sign"]) != int(original["sign"]):
                    raise ValueError("Circular shift changed sign")
                if moved.get("band_id") != original.get("band_id"):
                    raise ValueError("Circular shift changed band identity")
                for field in magnitude_fields:
                    left = original[field]
                    right = moved[field]
                    if pd.isna(left) and pd.isna(right):
                        continue
                    if left != right:
                        raise ValueError(
                            f"Circular shift changed magnitude field {field}"
                        )
                for start, end in band_support_segments(moved):
                    if start < eligible_start or end >= eligible_end:
                        raise ValueError(
                            "Shifted support falls outside eligible interval"
                        )
                apex = int(moved["apex_index_0based"])
                if not any(
                    start <= apex <= end
                    for start, end in band_support_segments(moved)
                ):
                    raise ValueError("Shifted apex is outside shifted support")
                if len(band_support_segments(moved)) == 2:
                    wrapped_band_shifts += 1
                band_shifts_checked += 1
            offsets_checked += 1
        patterns_checked += 1
    return {
        "performed": True,
        "shift_representation": (
            "one or two inclusive linear segments inside the half-open "
            "eligible interval"
        ),
        "patterns_checked": patterns_checked,
        "representative_offsets_checked": offsets_checked,
        "band_shifts_checked": band_shifts_checked,
        "wrapped_band_shifts_checked": wrapped_band_shifts,
        "all_shifted_widths_exact": True,
        "all_band_counts_preserved": True,
        "all_signs_preserved": True,
        "all_magnitudes_preserved": True,
        "all_relative_circular_apex_spacing_preserved": True,
        "all_support_positions_inside_eligible_interval": True,
        "all_shifted_apices_inside_shifted_support": True,
    }


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


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def finite_sample_std(values: np.ndarray) -> float:
    """Return sample SD without warnings when fewer than two values exist."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan


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
    if (
        not 2 <= args.min_seeds <= 3
        or args.position_tolerance < 0
        or args.max_width_tolerance < 0
        or not 0 < args.min_interval_iou <= 1
    ):
        raise ValueError("Invalid seed or tolerance settings")
    if args.n_block_shifts < 1:
        raise ValueError("--n_block_shifts must be positive")
    bands, summary = validate_and_load(args)
    mean_bands = load_mean_bands(args)
    parameter_set_id, active_detector_config, extractor_parameter_files = (
        validated_parameter_set(args)
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_bands, stable_audit, stable_path = write_stable_catalog(
        mean_bands,
        bands,
        summary,
        args,
        output_dir,
        parameter_set_id,
    )
    shift_intervals = args.matching_method in {
        "interval_iou", "legacy_width"
    }
    circular_shift_audit = audit_circular_shift_invariants(
        bands, summary, shift_intervals
    )
    null_inference_mode = (
        "smoke_test_only_n_block_shifts_1"
        if args.n_block_shifts == 1
        else "monte_carlo_block_shift"
    )
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
                args.matching_method,
                args.min_interval_iou,
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
                    list_a,
                    list_b,
                    args.position_tolerance,
                    args.max_width_tolerance,
                    args.matching_method,
                    args.min_interval_iou,
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
                    list_b,
                    eligible_start,
                    eligible_end,
                    offset,
                    shift_intervals=shift_intervals,
                )
                permuted_rows.append(pair_metrics(
                    list_a, shifted_b,
                    args.position_tolerance, args.max_width_tolerance,
                    args.matching_method,
                    args.min_interval_iou,
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
            "null_mean_micro_jaccard": finite_mean(null_micro),
            "null_std_micro_jaccard": finite_sample_std(null_micro),
            "micro_jaccard_empirical_p_upper": empirical_upper_p(
                observed["micro_jaccard"], null_micro
            ),
            "micro_jaccard_null_z": z_score(observed["micro_jaccard"], null_micro),
            "null_mean_macro_f1": finite_mean(null_macro),
            "null_std_macro_f1": finite_sample_std(null_macro),
            "macro_f1_empirical_p_upper": empirical_upper_p(
                observed["macro_f1_informative_proteins"], null_macro
            ),
            "macro_f1_null_z": z_score(
                observed["macro_f1_informative_proteins"], null_macro
            ),
            "null_inference_mode": null_inference_mode,
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
                            values,
                            eligible_start,
                            eligible_end,
                            offset,
                            shift_intervals=shift_intervals,
                        ))
                    count += len(consensus_clusters(
                        shifted_all,
                        args.min_seeds,
                        args.position_tolerance,
                        args.max_width_tolerance,
                        args.matching_method,
                        args.min_interval_iou,
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
                "null_std_consensus_band_count": finite_sample_std(null_values),
                "consensus_count_empirical_p_upper": empirical_upper_p(
                    observed_count, null_values
                ),
                "consensus_count_null_z": z_score(observed_count, null_values),
                "null_inference_mode": null_inference_mode,
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
        "mean_bands_csv": str(Path(args.mean_bands_csv).expanduser().resolve()),
        "per_seed_influence_fields": sorted(
            bands["influence_field"].astype(str).unique().tolist()
        ),
        "mean_profile_influence_fields": sorted(
            mean_bands["influence_field"].astype(str).unique().tolist()
        ),
        "matching": (
            "same condition, split, protein, and sign; one-to-one Hungarian "
            "matching; prominence is never used"
        ),
        "matching_method": args.matching_method,
        "match_tolerance": (
            f"interval intersection-over-union >= {args.min_interval_iou}"
            if args.matching_method == "interval_iou"
            else "fixed positional tolerance; support width does not participate"
            if args.matching_method == "fixed_apex"
            else "max(position_tolerance, min(max_width_tolerance, "
            "ceil(max(band_width_a, band_width_b)/2)))"
        ),
        "min_seeds": args.min_seeds,
        "min_interval_iou": args.min_interval_iou,
        "position_tolerance": args.position_tolerance,
        "max_width_tolerance": args.max_width_tolerance,
        "n_block_shifts": args.n_block_shifts,
        "random_seed": args.random_seed,
        "null_inference_mode": null_inference_mode,
        "null_statistics_inferential": args.n_block_shifts > 1,
        "single_shift_note": (
            "n_block_shifts=1 is smoke-test mode; sample standard deviations "
            "and z-scores are NA and the null is not inferential"
            if args.n_block_shifts == 1 else None
        ),
        "block_shift": (
            "circularly shift every apex and every support residue in each "
            "seed's complete same-sign band pattern modulo the half-open "
            "eligible interval; boundary-crossing support uses two linear "
            "segments and preserves exact band count, width, sign, magnitude, "
            "and within-protein circular spacing"
        ),
        "circular_shift_audit": circular_shift_audit,
        "stable_catalog_independent_of_null_shift": True,
        "consensus_output_role": "diagnostic seed-only clustering",
        "primary_catalog_definition": (
            "mean-profile band with a one-to-one same-sign seed-band interval "
            f"match in at least {args.min_seeds} of 3 seeds"
        ),
        "stable_catalog_csv": str(stable_path),
        "stable_catalog_audit": stable_audit,
        "locked_parameter_set_id": parameter_set_id,
        "validated_active_detector_parameters": active_detector_config,
        "extractor_parameter_files": extractor_parameter_files,
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
        "stable_bands": len(stable_bands),
        "strict_3of3_bands": int(stable_bands["strict_stable_band"].sum()),
        "stable_catalog_csv": str(stable_path),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
