#!/usr/bin/env python3
"""Compare observed and uniform-routing mean and seed-stable bands.

The primary estimand is residue-level interval-mask overlap within the same
condition, split, protein, and sign.  Apex proximity and one-to-one
maximum-IoU band pairing are complementary localization summaries.  All
aggregate estimates weight proteins equally; confidence intervals resample
proteins, never individual bands or residues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


KEY = ["condition", "split", "protein", "sign"]
PROTEIN_KEY = ["condition", "split", "protein"]
SCOPES = ("mean_candidate", "seed_stable")
DISTANCE_THRESHOLDS = (0, 2, 5, 10)
IOU_THRESHOLDS = (0.25, 0.50, 0.75)
BAND_COLUMNS = [
    "condition",
    "split",
    "protein",
    "protein_length",
    "sign",
    "label",
    "band_id",
    "apex_index_0based",
    "support_start_index_0based",
    "support_end_index_0based_inclusive",
    "support_width",
    "apex_standardized_magnitude_R_p",
    "band_integrated_magnitude",
    "integrated_magnitude_rank_within_protein_sign",
]
STABILITY_COLUMNS = [
    "seed_support_count",
    "seed_support_fraction",
    "strict_stable_band",
    "primary_stable_band",
    "stability_class",
]
SUMMARY_COLUMNS = [
    "condition",
    "split",
    "protein",
    "protein_length",
    "eligible_start_index_0based",
    "eligible_end_index_0based_exclusive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed_mean_bands", required=True)
    parser.add_argument("--observed_stable_bands", required=True)
    parser.add_argument(
        "--uniform_mean_bands", "--control_mean_bands",
        dest="uniform_mean_bands", required=True,
    )
    parser.add_argument(
        "--uniform_stable_bands", "--control_stable_bands",
        dest="uniform_stable_bands", required=True,
    )
    parser.add_argument("--observed_protein_summary", required=True)
    parser.add_argument(
        "--uniform_protein_summary", "--control_protein_summary",
        dest="uniform_protein_summary", required=True,
    )
    parser.add_argument(
        "--control_label",
        choices=("uniform", "shifted_attention"),
        default="uniform",
        help=(
            "Name used in output columns and files. Internal calculations retain "
            "the legacy uniform labels for backward compatibility."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260730)
    parser.add_argument(
        "--splits", nargs="*", default=None,
        help="Optional split subset applied identically to both catalogs.",
    )
    return parser.parse_args()


def _atomic_frame(path: Path, frame: pd.DataFrame, *, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
        path.chmod(0o664)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def relabel_control_frame(frame: pd.DataFrame, control_label: str) -> pd.DataFrame:
    """Rename legacy 'uniform' fields for a generic control comparison."""
    if control_label == "uniform":
        return frame
    output = frame.copy()
    output = output.rename(
        columns={column: column.replace("uniform", control_label) for column in output}
    )
    for column in output.select_dtypes(include="object"):
        output[column] = output[column].map(
            lambda value: (
                value.replace("uniform", control_label)
                if isinstance(value, str) else value
            )
        )
    return output


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
        path.chmod(0o664)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_bands(path: Path, *, stable: bool) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    required = set(BAND_COLUMNS)
    if stable:
        required.update(STABILITY_COLUMNS)
    missing = required - set(header)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    columns = BAND_COLUMNS + (STABILITY_COLUMNS if stable else [])
    frame = pd.read_csv(path, usecols=columns)
    if frame["band_id"].isna().any() or frame["band_id"].duplicated().any():
        raise ValueError(f"{path}: band_id values must be nonmissing and unique")
    if not set(frame["sign"].astype(int)).issubset({-1, 1}):
        raise ValueError(f"{path}: signs must be -1 or +1")
    integer_columns = [
        "protein_length",
        "sign",
        "apex_index_0based",
        "support_start_index_0based",
        "support_end_index_0based_inclusive",
        "support_width",
        "integrated_magnitude_rank_within_protein_sign",
    ]
    for column in integer_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    numeric_columns = [
        "apex_standardized_magnitude_R_p",
        "band_integrated_magnitude",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.all(np.isfinite(frame[column])):
            raise ValueError(f"{path}: {column} contains nonfinite values")
    invalid = (
        (frame["support_start_index_0based"] < 0)
        | (
            frame["support_start_index_0based"]
            > frame["apex_index_0based"]
        )
        | (
            frame["apex_index_0based"]
            > frame["support_end_index_0based_inclusive"]
        )
        | (
            frame["support_end_index_0based_inclusive"]
            >= frame["protein_length"]
        )
        | (
            frame["support_width"]
            != frame["support_end_index_0based_inclusive"]
            - frame["support_start_index_0based"]
            + 1
        )
    )
    if invalid.any():
        raise ValueError(f"{path}: {int(invalid.sum())} invalid band intervals")
    return frame.sort_values(
        KEY + ["support_start_index_0based", "band_id"]
    ).reset_index(drop=True)


def load_summary(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = set(SUMMARY_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=SUMMARY_COLUMNS)
    if frame.duplicated(PROTEIN_KEY).any():
        raise ValueError(f"{path}: duplicate condition/split/protein rows")
    for column in (
        "protein_length",
        "eligible_start_index_0based",
        "eligible_end_index_0based_exclusive",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    invalid = (
        (frame["eligible_start_index_0based"] < 0)
        | (
            frame["eligible_start_index_0based"]
            >= frame["eligible_end_index_0based_exclusive"]
        )
        | (
            frame["eligible_end_index_0based_exclusive"]
            > frame["protein_length"]
        )
    )
    if invalid.any():
        raise ValueError(f"{path}: invalid eligible intervals")
    return frame.sort_values(PROTEIN_KEY).reset_index(drop=True)


def build_universe(
    observed_summary: pd.DataFrame, uniform_summary: pd.DataFrame
) -> pd.DataFrame:
    merged = observed_summary.merge(
        uniform_summary,
        on=PROTEIN_KEY,
        how="outer",
        suffixes=("_observed", "_uniform"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Observed and uniform protein-summary universes differ")
    for column in (
        "protein_length",
        "eligible_start_index_0based",
        "eligible_end_index_0based_exclusive",
    ):
        left = merged[f"{column}_observed"]
        right = merged[f"{column}_uniform"]
        if not left.equals(right):
            raise ValueError(f"Observed and uniform summaries differ in {column}")
        merged[column] = left.astype(int)
    base = merged[
        PROTEIN_KEY
        + [
            "protein_length",
            "eligible_start_index_0based",
            "eligible_end_index_0based_exclusive",
        ]
    ]
    universe = pd.concat(
        [base.assign(sign=1), base.assign(sign=-1)], ignore_index=True
    )
    universe["label"] = np.where(
        universe["sign"] > 0,
        "flexibility_supporting",
        "rigidity_supporting",
    )
    universe["eligible_residue_count"] = (
        universe["eligible_end_index_0based_exclusive"]
        - universe["eligible_start_index_0based"]
    )
    return universe.sort_values(KEY).reset_index(drop=True)


def band_lookup(frame: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    return {
        key: group.reset_index(drop=True)
        for key, group in frame.groupby(KEY, sort=False)
    }


def mask_metrics(
    context: pd.Series,
    observed: pd.DataFrame,
    uniform: pd.DataFrame,
) -> dict:
    start = int(context["eligible_start_index_0based"])
    end = int(context["eligible_end_index_0based_exclusive"])
    size = end - start
    observed_mask = np.zeros(size, dtype=bool)
    uniform_mask = np.zeros(size, dtype=bool)
    for frame, mask in ((observed, observed_mask), (uniform, uniform_mask)):
        for row in frame.itertuples(index=False):
            left = int(row.support_start_index_0based)
            right = int(row.support_end_index_0based_inclusive)
            if left < start or right >= end:
                raise ValueError(
                    f"{row.band_id}: band interval falls outside eligibility"
                )
            local_left, local_right = left - start, right - start
            if mask[local_left:local_right + 1].any():
                raise ValueError(
                    f"{row.band_id}: same-catalog band intervals overlap"
                )
            mask[local_left:local_right + 1] = True
    observed_count = int(observed_mask.sum())
    uniform_count = int(uniform_mask.sum())
    intersection = int(np.sum(observed_mask & uniform_mask))
    union = int(np.sum(observed_mask | uniform_mask))
    return {
        "observed_n_bands": len(observed),
        "uniform_n_bands": len(uniform),
        "observed_minus_uniform_n_bands": len(observed) - len(uniform),
        "observed_band_residues": observed_count,
        "uniform_band_residues": uniform_count,
        "intersection_band_residues": intersection,
        "union_band_residues": union,
        "both_masks_empty": union == 0,
        "jaccard": intersection / union if union else math.nan,
        "dice": (
            2 * intersection / (observed_count + uniform_count)
            if observed_count + uniform_count
            else math.nan
        ),
        "fraction_observed_residues_covered_by_uniform": (
            intersection / observed_count if observed_count else math.nan
        ),
        "fraction_uniform_residues_covered_by_observed": (
            intersection / uniform_count if uniform_count else math.nan
        ),
        "observed_band_residue_coverage": observed_count / size,
        "uniform_band_residue_coverage": uniform_count / size,
        "observed_minus_uniform_band_residue_coverage": (
            observed_count - uniform_count
        ) / size,
    }


def compare_masks(
    universe: pd.DataFrame,
    catalogs: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    lookups = {key: band_lookup(frame) for key, frame in catalogs.items()}
    rows = []
    empty = pd.DataFrame(columns=BAND_COLUMNS)
    for scope in SCOPES:
        observed_lookup = lookups[("observed", scope)]
        uniform_lookup = lookups[("uniform", scope)]
        for context in universe.to_dict(orient="records"):
            key = tuple(context[column] for column in KEY)
            metrics = mask_metrics(
                pd.Series(context),
                observed_lookup.get(key, empty),
                uniform_lookup.get(key, empty),
            )
            rows.append({
                **context,
                "catalog_scope": scope,
                **metrics,
            })
    return pd.DataFrame(rows)


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=int)
    target = np.asarray(target, dtype=int)
    if len(source) == 0:
        return np.empty(0, dtype=float)
    if len(target) == 0:
        return np.full(len(source), np.nan)
    return np.min(np.abs(source[:, None] - target[None, :]), axis=1).astype(float)


def compare_apices(
    universe: pd.DataFrame,
    catalogs: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookups = {key: band_lookup(frame) for key, frame in catalogs.items()}
    empty = pd.DataFrame(columns=BAND_COLUMNS)
    distance_rows = []
    protein_rows = []
    for scope in SCOPES:
        for context in universe.to_dict(orient="records"):
            key = tuple(context[column] for column in KEY)
            observed = lookups[("observed", scope)].get(key, empty)
            uniform = lookups[("uniform", scope)].get(key, empty)
            for direction, source, target in (
                ("observed_to_uniform", observed, uniform),
                ("uniform_to_observed", uniform, observed),
            ):
                source_apices = source["apex_index_0based"].to_numpy(dtype=int)
                target_apices = target["apex_index_0based"].to_numpy(dtype=int)
                distances = nearest_distances(source_apices, target_apices)
                for source_row, distance in zip(
                    source.to_dict(orient="records"), distances
                ):
                    distance_rows.append({
                        **{column: context[column] for column in KEY},
                        "protein_length": context["protein_length"],
                        "catalog_scope": scope,
                        "direction": direction,
                        "source_band_id": source_row["band_id"],
                        "source_apex_index_0based": source_row[
                            "apex_index_0based"
                        ],
                        "nearest_same_sign_target_apex_distance": distance,
                        "target_apex_missing": len(target_apices) == 0,
                    })
                protein_row = {
                    **context,
                    "catalog_scope": scope,
                    "direction": direction,
                    "source_n_apices": len(source_apices),
                    "target_n_apices": len(target_apices),
                    "source_present_target_missing": (
                        len(source_apices) > 0 and len(target_apices) == 0
                    ),
                    "median_nearest_apex_distance": (
                        float(np.nanmedian(distances))
                        if len(target_apices) and len(source_apices)
                        else math.nan
                    ),
                }
                for threshold in DISTANCE_THRESHOLDS:
                    protein_row[f"fraction_within_{threshold}_residues"] = (
                        float(np.mean(distances <= threshold))
                        if len(source_apices)
                        else math.nan
                    )
                protein_rows.append(protein_row)
    return pd.DataFrame(distance_rows), pd.DataFrame(protein_rows)


def apex_cdf(distance_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["condition", "split", "catalog_scope", "sign", "direction"]
    for key, group in distance_rows.groupby(group_columns, sort=True):
        finite = group["nearest_same_sign_target_apex_distance"].dropna()
        maximum = int(finite.max()) if len(finite) else 0
        curve_sum = np.zeros(maximum + 1, dtype=float)
        protein_count = 0
        for _protein, protein_group in group.groupby("protein", sort=False):
            distances = protein_group[
                "nearest_same_sign_target_apex_distance"
            ].to_numpy(dtype=float)
            finite_distances = distances[np.isfinite(distances)].astype(int)
            histogram = np.bincount(
                finite_distances, minlength=maximum + 1
            )[:maximum + 1]
            curve_sum += np.cumsum(histogram) / len(distances)
            protein_count += 1
        for distance, value in enumerate(curve_sum / protein_count):
            rows.append({
                **dict(zip(group_columns, key)),
                "distance_residues": distance,
                "mean_within_distance_fraction_equal_protein_weight": value,
                "n_proteins_with_source_apices": protein_count,
            })
    return pd.DataFrame(rows)


def interval_iou_matrix(observed: pd.DataFrame, uniform: pd.DataFrame) -> np.ndarray:
    if observed.empty or uniform.empty:
        return np.empty((len(observed), len(uniform)))
    observed_start = observed["support_start_index_0based"].to_numpy()[:, None]
    observed_end = observed[
        "support_end_index_0based_inclusive"
    ].to_numpy()[:, None]
    uniform_start = uniform["support_start_index_0based"].to_numpy()[None, :]
    uniform_end = uniform[
        "support_end_index_0based_inclusive"
    ].to_numpy()[None, :]
    intersection = np.maximum(
        0,
        np.minimum(observed_end, uniform_end)
        - np.maximum(observed_start, uniform_start)
        + 1,
    )
    union = (
        observed_end - observed_start + 1
        + uniform_end - uniform_start + 1
        - intersection
    )
    return intersection / union


def pair_one_group(
    context: dict,
    scope: str,
    observed: pd.DataFrame,
    uniform: pd.DataFrame,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    matrix = interval_iou_matrix(observed, uniform)
    retained: list[tuple[int, int]] = []
    if matrix.size:
        row_indices, column_indices = linear_sum_assignment(-matrix)
        retained = [
            (int(i), int(j))
            for i, j in zip(row_indices, column_indices)
            if matrix[i, j] > 0
        ]
    matched_observed = {i for i, _ in retained}
    matched_uniform = {j for _, j in retained}
    matched_rows = []
    for i, j in retained:
        observed_row = observed.iloc[i]
        uniform_row = uniform.iloc[j]
        matched_rows.append({
            **{column: context[column] for column in KEY},
            "catalog_scope": scope,
            "observed_band_id": observed_row["band_id"],
            "uniform_band_id": uniform_row["band_id"],
            "interval_iou": float(matrix[i, j]),
            "apex_displacement": abs(
                int(observed_row["apex_index_0based"])
                - int(uniform_row["apex_index_0based"])
            ),
            "observed_apex_index_0based": observed_row[
                "apex_index_0based"
            ],
            "uniform_apex_index_0based": uniform_row[
                "apex_index_0based"
            ],
            "observed_width": observed_row["support_width"],
            "uniform_width": uniform_row["support_width"],
            "observed_standardized_apex_strength": observed_row[
                "apex_standardized_magnitude_R_p"
            ],
            "uniform_standardized_apex_strength": uniform_row[
                "apex_standardized_magnitude_R_p"
            ],
            "observed_integrated_influence_rank": observed_row[
                "integrated_magnitude_rank_within_protein_sign"
            ],
            "uniform_integrated_influence_rank": uniform_row[
                "integrated_magnitude_rank_within_protein_sign"
            ],
            "observed_band_integrated_magnitude": observed_row[
                "band_integrated_magnitude"
            ],
            "uniform_band_integrated_magnitude": uniform_row[
                "band_integrated_magnitude"
            ],
        })
    observed_only = [
        {
            **{column: context[column] for column in KEY},
            "catalog_scope": scope,
            "band_id": row["band_id"],
            "apex_index_0based": row["apex_index_0based"],
            "support_start_index_0based": row[
                "support_start_index_0based"
            ],
            "support_end_index_0based_inclusive": row[
                "support_end_index_0based_inclusive"
            ],
            "support_width": row["support_width"],
            "standardized_apex_strength": row[
                "apex_standardized_magnitude_R_p"
            ],
            "integrated_influence_rank": row[
                "integrated_magnitude_rank_within_protein_sign"
            ],
            "classification": "observed_only_candidate_attention_dependent",
            "unmatched_definition": "no positive-IoU Hungarian assignment",
        }
        for index, row in observed.iterrows()
        if index not in matched_observed
    ]
    uniform_only = [
        {
            **{column: context[column] for column in KEY},
            "catalog_scope": scope,
            "band_id": row["band_id"],
            "apex_index_0based": row["apex_index_0based"],
            "support_start_index_0based": row[
                "support_start_index_0based"
            ],
            "support_end_index_0based_inclusive": row[
                "support_end_index_0based_inclusive"
            ],
            "support_width": row["support_width"],
            "standardized_apex_strength": row[
                "apex_standardized_magnitude_R_p"
            ],
            "integrated_influence_rank": row[
                "integrated_magnitude_rank_within_protein_sign"
            ],
            "classification": "uniform_only_evidence_suppressed_or_displaced",
            "unmatched_definition": "no positive-IoU Hungarian assignment",
        }
        for index, row in uniform.iterrows()
        if index not in matched_uniform
    ]
    metric = {
        **context,
        "catalog_scope": scope,
        "observed_n_bands": len(observed),
        "uniform_n_bands": len(uniform),
        "positive_iou_matched_pairs": len(retained),
        "observed_only_bands": len(observed_only),
        "uniform_only_bands": len(uniform_only),
        "mean_matched_interval_iou": (
            float(np.mean([matrix[i, j] for i, j in retained]))
            if retained else math.nan
        ),
        "median_matched_interval_iou": (
            float(np.median([matrix[i, j] for i, j in retained]))
            if retained else math.nan
        ),
    }
    for threshold in IOU_THRESHOLDS:
        label = str(threshold).replace(".", "_")
        count = sum(matrix[i, j] >= threshold for i, j in retained)
        metric[f"matched_pairs_iou_ge_{label}"] = count
        metric[f"fraction_observed_iou_ge_{label}"] = (
            count / len(observed) if len(observed) else math.nan
        )
        metric[f"fraction_uniform_iou_ge_{label}"] = (
            count / len(uniform) if len(uniform) else math.nan
        )
    return matched_rows, observed_only, uniform_only, metric


def compare_pairs(
    universe: pd.DataFrame,
    catalogs: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookups = {key: band_lookup(frame) for key, frame in catalogs.items()}
    empty = pd.DataFrame(columns=BAND_COLUMNS)
    matched, observed_only, uniform_only, metrics = [], [], [], []
    for scope in SCOPES:
        for context in universe.to_dict(orient="records"):
            key = tuple(context[column] for column in KEY)
            result = pair_one_group(
                context,
                scope,
                lookups[("observed", scope)].get(key, empty),
                lookups[("uniform", scope)].get(key, empty),
            )
            matched.extend(result[0])
            observed_only.extend(result[1])
            uniform_only.extend(result[2])
            metrics.append(result[3])
    return (
        pd.DataFrame(matched),
        pd.DataFrame(observed_only),
        pd.DataFrame(uniform_only),
        pd.DataFrame(metrics),
    )


def annotate_mean_stability(
    mean: pd.DataFrame,
    stable: pd.DataFrame,
    profile_type: str,
) -> pd.DataFrame:
    stable_fields = stable[
        [
            "band_id",
            "seed_support_count",
            "seed_support_fraction",
            "strict_stable_band",
            "stability_class",
        ]
    ]
    if not set(stable_fields["band_id"]).issubset(set(mean["band_id"])):
        raise ValueError(f"{profile_type}: stable catalog is not a mean-band subset")
    annotated = mean.merge(
        stable_fields,
        on="band_id",
        how="left",
        validate="one_to_one",
    )
    annotated["profile_type"] = profile_type
    annotated["supported_at_least_2of3"] = annotated[
        "seed_support_count"
    ].ge(2)
    annotated["supported_3of3"] = annotated["seed_support_count"].eq(3)
    annotated["stability_status"] = np.select(
        [
            annotated["supported_3of3"],
            annotated["supported_at_least_2of3"],
        ],
        ["supported_3of3", "supported_2of3"],
        default="not_retained_lt2_support_count_unknown",
    )
    return annotated


def stability_comparison(
    universe: pd.DataFrame,
    observed_annotated: pd.DataFrame,
    uniform_annotated: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    lookups = {
        "observed": band_lookup(observed_annotated),
        "uniform": band_lookup(uniform_annotated),
    }
    empty = pd.DataFrame()
    for context in universe.to_dict(orient="records"):
        key = tuple(context[column] for column in KEY)
        row = dict(context)
        for profile_type in ("observed", "uniform"):
            frame = lookups[profile_type].get(key, empty)
            candidates = len(frame)
            supported_2 = (
                int(frame["supported_at_least_2of3"].sum())
                if candidates else 0
            )
            supported_3 = int(frame["supported_3of3"].sum()) if candidates else 0
            row[f"{profile_type}_mean_candidate_bands"] = candidates
            row[f"{profile_type}_supported_2of3_bands"] = supported_2
            row[f"{profile_type}_supported_3of3_bands"] = supported_3
            row[f"{profile_type}_fraction_supported_2of3"] = (
                supported_2 / candidates if candidates else math.nan
            )
            row[f"{profile_type}_fraction_supported_3of3"] = (
                supported_3 / candidates if candidates else math.nan
            )
        for suffix in (
            "mean_candidate_bands",
            "supported_2of3_bands",
            "supported_3of3_bands",
            "fraction_supported_2of3",
            "fraction_supported_3of3",
        ):
            row[f"observed_minus_uniform_{suffix}"] = (
                row[f"observed_{suffix}"] - row[f"uniform_{suffix}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def equal_protein_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        row["n_protein_rows"] = len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"mean_{metric}"] = finite.mean() if len(finite) else math.nan
            row[f"median_{metric}"] = (
                finite.median() if len(finite) else math.nan
            )
            row[f"n_finite_{metric}"] = len(finite)
        row["test_set_primary"] = row.get("split") == "test"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["test_set_primary"] + group_columns,
        ascending=[False] + [True] * len(group_columns),
    )


def bootstrap_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
    component: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        matrix = group[metrics].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
        n = len(matrix)
        bootstrap = np.full((n_bootstrap, len(metrics)), np.nan)
        for start in range(0, n_bootstrap, 100):
            stop = min(n_bootstrap, start + 100)
            indices = rng.integers(0, n, size=(stop - start, n))
            sampled = matrix[indices]
            with np.errstate(invalid="ignore"):
                finite_count = np.isfinite(sampled).sum(axis=1)
                total = np.nansum(sampled, axis=1)
                bootstrap[start:stop] = np.divide(
                    total,
                    finite_count,
                    out=np.full_like(total, np.nan),
                    where=finite_count > 0,
                )
        for metric_index, metric in enumerate(metrics):
            values = matrix[:, metric_index]
            finite = values[np.isfinite(values)]
            boot = bootstrap[:, metric_index]
            boot = boot[np.isfinite(boot)]
            rows.append({
                **dict(zip(group_columns, key_tuple)),
                "analysis_component": component,
                "metric": metric,
                "estimate_equal_protein_mean": (
                    float(np.mean(finite)) if len(finite) else math.nan
                ),
                "bootstrap_ci_lower_95": (
                    float(np.quantile(boot, 0.025)) if len(boot) else math.nan
                ),
                "bootstrap_ci_upper_95": (
                    float(np.quantile(boot, 0.975)) if len(boot) else math.nan
                ),
                "n_protein_rows": n,
                "n_finite_protein_rows": len(finite),
                "n_bootstrap": n_bootstrap,
                "test_set_primary": dict(
                    zip(group_columns, key_tuple)
                ).get("split") == "test",
            })
    return pd.DataFrame(rows)


def validate_catalog_contexts(
    universe: pd.DataFrame,
    catalogs: dict[tuple[str, str], pd.DataFrame],
) -> None:
    universe_keys = set(map(tuple, universe[KEY].to_numpy()))
    lengths = {
        tuple(row[column] for column in KEY): int(row["protein_length"])
        for row in universe.to_dict(orient="records")
    }
    for label, frame in catalogs.items():
        keys = set(map(tuple, frame[KEY].to_numpy()))
        extra = keys - universe_keys
        if extra:
            raise ValueError(f"{label}: catalog has contexts outside summaries")
        for row in frame.to_dict(orient="records"):
            key = tuple(row[column] for column in KEY)
            if int(row["protein_length"]) != lengths[key]:
                raise ValueError(f"{label}: protein length mismatch for {key}")


def validate_comparison_outputs(
    catalogs: dict[tuple[str, str], pd.DataFrame],
    mask_frame: pd.DataFrame,
    distance_frame: pd.DataFrame,
    matched: pd.DataFrame,
    observed_only: pd.DataFrame,
    uniform_only: pd.DataFrame,
    stability_frame: pd.DataFrame,
    all_mean_stability: pd.DataFrame,
) -> dict:
    bounded_columns = [
        "jaccard",
        "dice",
        "fraction_observed_residues_covered_by_uniform",
        "fraction_uniform_residues_covered_by_observed",
        "observed_band_residue_coverage",
        "uniform_band_residue_coverage",
    ]
    for column in bounded_columns:
        finite = mask_frame[column].dropna()
        if not finite.between(0, 1).all():
            raise ValueError(f"Mask metric {column} falls outside [0, 1]")
    if (
        mask_frame["intersection_band_residues"]
        > mask_frame["union_band_residues"]
    ).any():
        raise ValueError("Mask intersection exceeds union")
    finite_distances = distance_frame[
        "nearest_same_sign_target_apex_distance"
    ].dropna()
    if (finite_distances < 0).any():
        raise ValueError("Nearest-apex distance is negative")
    if not matched.empty and not matched["interval_iou"].between(0, 1, inclusive="both").all():
        raise ValueError("Matched interval IoU falls outside [0, 1]")
    if not matched.empty and (matched["interval_iou"] <= 0).any():
        raise ValueError("Zero-IoU assignments must be classified as unmatched")
    if not matched.empty and matched.duplicated(["catalog_scope", "observed_band_id"]).any():
        raise ValueError("An observed band is matched more than once")
    if not matched.empty and matched.duplicated(["catalog_scope", "uniform_band_id"]).any():
        raise ValueError("A uniform band is matched more than once")
    for profile in ("observed", "uniform"):
        candidates = stability_frame[f"{profile}_mean_candidate_bands"]
        supported_2 = stability_frame[f"{profile}_supported_2of3_bands"]
        supported_3 = stability_frame[f"{profile}_supported_3of3_bands"]
        if ((supported_3 > supported_2) | (supported_2 > candidates)).any():
            raise ValueError(f"{profile}: invalid nested stability counts")
    if len(all_mean_stability) != (
        len(catalogs[("observed", "mean_candidate")])
        + len(catalogs[("uniform", "mean_candidate")])
    ):
        raise ValueError("Not every mean band received a stability annotation")

    classifications = {}
    for profile in ("observed", "uniform"):
        for scope in SCOPES:
            total = len(catalogs[(profile, scope)])
            matched_count = (
                int(matched["catalog_scope"].eq(scope).sum())
                if "catalog_scope" in matched else 0
            )
            only_frame = observed_only if profile == "observed" else uniform_only
            only_count = (
                int(only_frame["catalog_scope"].eq(scope).sum())
                if "catalog_scope" in only_frame else 0
            )
            if matched_count + only_count != total:
                raise ValueError(
                    f"{profile}/{scope}: matching classification is incomplete"
                )
            classifications[f"{profile}_{scope}"] = {
                "input_bands": total,
                "positive_iou_matched_bands": matched_count,
                "unmatched_bands": only_count,
            }
    return classifications


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n_bootstrap must be positive")
    paths = {
        "observed_mean_bands": Path(args.observed_mean_bands).resolve(),
        "observed_stable_bands": Path(args.observed_stable_bands).resolve(),
        "uniform_mean_bands": Path(args.uniform_mean_bands).resolve(),
        "uniform_stable_bands": Path(args.uniform_stable_bands).resolve(),
        "observed_protein_summary": Path(args.observed_protein_summary).resolve(),
        "uniform_protein_summary": Path(args.uniform_protein_summary).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_mean = load_bands(paths["observed_mean_bands"], stable=False)
    observed_stable = load_bands(paths["observed_stable_bands"], stable=True)
    uniform_mean = load_bands(paths["uniform_mean_bands"], stable=False)
    uniform_stable = load_bands(paths["uniform_stable_bands"], stable=True)
    observed_summary = load_summary(paths["observed_protein_summary"])
    uniform_summary = load_summary(paths["uniform_protein_summary"])
    if args.splits:
        selected_splits = set(map(str, args.splits))
        observed_mean = observed_mean[observed_mean.split.isin(selected_splits)].copy()
        observed_stable = observed_stable[
            observed_stable.split.isin(selected_splits)
        ].copy()
        uniform_mean = uniform_mean[uniform_mean.split.isin(selected_splits)].copy()
        uniform_stable = uniform_stable[
            uniform_stable.split.isin(selected_splits)
        ].copy()
        observed_summary = observed_summary[
            observed_summary.split.isin(selected_splits)
        ].copy()
        uniform_summary = uniform_summary[
            uniform_summary.split.isin(selected_splits)
        ].copy()
    universe = build_universe(observed_summary, uniform_summary)
    catalogs = {
        ("observed", "mean_candidate"): observed_mean,
        ("observed", "seed_stable"): observed_stable,
        ("uniform", "mean_candidate"): uniform_mean,
        ("uniform", "seed_stable"): uniform_stable,
    }
    validate_catalog_contexts(universe, catalogs)

    mask_frame = compare_masks(universe, catalogs)
    distance_frame, apex_protein_frame = compare_apices(universe, catalogs)
    cdf_frame = apex_cdf(distance_frame)
    matched, observed_only, uniform_only, pairing_protein = compare_pairs(
        universe, catalogs
    )
    observed_annotated = annotate_mean_stability(
        observed_mean, observed_stable, "observed"
    )
    uniform_annotated = annotate_mean_stability(
        uniform_mean, uniform_stable, "uniform"
    )
    all_mean_stability = pd.concat(
        [observed_annotated, uniform_annotated], ignore_index=True
    )
    stability_frame = stability_comparison(
        universe, observed_annotated, uniform_annotated
    )
    band_classification_audit = validate_comparison_outputs(
        catalogs,
        mask_frame,
        distance_frame,
        matched,
        observed_only,
        uniform_only,
        stability_frame,
        all_mean_stability,
    )

    mask_metrics_list = [
        "jaccard",
        "dice",
        "fraction_observed_residues_covered_by_uniform",
        "fraction_uniform_residues_covered_by_observed",
        "observed_band_residue_coverage",
        "uniform_band_residue_coverage",
        "observed_minus_uniform_band_residue_coverage",
        "observed_minus_uniform_n_bands",
    ]
    apex_metrics = [
        "median_nearest_apex_distance",
        *[
            f"fraction_within_{threshold}_residues"
            for threshold in DISTANCE_THRESHOLDS
        ],
        "source_present_target_missing",
    ]
    pairing_metrics = [
        "mean_matched_interval_iou",
        "median_matched_interval_iou",
        "observed_only_bands",
        "uniform_only_bands",
        *[
            metric
            for threshold in IOU_THRESHOLDS
            for metric in (
                f"fraction_observed_iou_ge_{str(threshold).replace('.', '_')}",
                f"fraction_uniform_iou_ge_{str(threshold).replace('.', '_')}",
            )
        ],
    ]
    stability_metrics = [
        "observed_mean_candidate_bands",
        "uniform_mean_candidate_bands",
        "observed_fraction_supported_2of3",
        "uniform_fraction_supported_2of3",
        "observed_fraction_supported_3of3",
        "uniform_fraction_supported_3of3",
        "observed_minus_uniform_mean_candidate_bands",
        "observed_minus_uniform_fraction_supported_2of3",
        "observed_minus_uniform_fraction_supported_3of3",
    ]
    group_scope = ["condition", "split", "catalog_scope", "sign"]
    apex_group = group_scope + ["direction"]
    mask_summary = equal_protein_summary(
        mask_frame, group_scope, mask_metrics_list
    )
    apex_summary = equal_protein_summary(
        apex_protein_frame, apex_group, apex_metrics
    )
    pairing_summary = equal_protein_summary(
        pairing_protein, group_scope, pairing_metrics
    )
    stability_summary = equal_protein_summary(
        stability_frame,
        ["condition", "split", "sign"],
        stability_metrics,
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = pd.concat([
        bootstrap_summary(
            mask_frame,
            group_scope,
            mask_metrics_list,
            "residue_mask_overlap",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
        bootstrap_summary(
            apex_protein_frame,
            apex_group,
            apex_metrics,
            "nearest_apex_localization",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
        bootstrap_summary(
            pairing_protein,
            group_scope,
            pairing_metrics,
            "hungarian_interval_iou",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
        bootstrap_summary(
            stability_frame,
            ["condition", "split", "sign"],
            stability_metrics,
            "seed_stability",
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        ),
    ], ignore_index=True).sort_values(
        [
            "test_set_primary",
            "analysis_component",
            "condition",
            "split",
            "sign",
            "metric",
        ],
        ascending=[False, True, True, True, True, True],
    )

    control_label = args.control_label
    if control_label != "uniform":
        mask_frame = relabel_control_frame(mask_frame, control_label)
        mask_summary = relabel_control_frame(mask_summary, control_label)
        distance_frame = relabel_control_frame(distance_frame, control_label)
        apex_protein_frame = relabel_control_frame(apex_protein_frame, control_label)
        apex_summary = relabel_control_frame(apex_summary, control_label)
        cdf_frame = relabel_control_frame(cdf_frame, control_label)
        matched = relabel_control_frame(matched, control_label)
        observed_only = relabel_control_frame(observed_only, control_label)
        uniform_only = relabel_control_frame(uniform_only, control_label)
        pairing_summary = relabel_control_frame(pairing_summary, control_label)
        stability_frame = relabel_control_frame(stability_frame, control_label)
        stability_summary = relabel_control_frame(stability_summary, control_label)
        all_mean_stability = relabel_control_frame(
            all_mean_stability, control_label
        )
        bootstrap = relabel_control_frame(bootstrap, control_label)

    _atomic_frame(
        output_dir / "per_protein_band_mask_overlap.csv", mask_frame
    )
    _atomic_frame(output_dir / "band_mask_overlap_summary.csv", mask_summary)
    _atomic_frame(
        output_dir / "nearest_apex_distances.csv.gz",
        distance_frame,
        compression="gzip",
    )
    _atomic_frame(
        output_dir / "per_protein_apex_localization.csv",
        apex_protein_frame,
    )
    _atomic_frame(output_dir / "apex_distance_summary.csv", apex_summary)
    _atomic_frame(output_dir / "apex_distance_cdf.csv", cdf_frame)
    _atomic_frame(
        output_dir / f"matched_observed_{control_label}_bands.csv", matched
    )
    _atomic_frame(output_dir / "observed_only_bands.csv", observed_only)
    _atomic_frame(output_dir / f"{control_label}_only_bands.csv", uniform_only)
    _atomic_frame(
        output_dir / "band_pairing_threshold_summary.csv", pairing_summary
    )
    _atomic_frame(output_dir / "stability_comparison.csv", stability_frame)
    _atomic_frame(
        output_dir / "stability_comparison_summary.csv", stability_summary
    )
    _atomic_frame(
        output_dir / "all_mean_bands_with_stability.csv.gz",
        all_mean_stability,
        compression="gzip",
    )
    _atomic_frame(
        output_dir / "protein_bootstrap_summary.csv", bootstrap
    )

    parameters = {
        "analysis": f"observed versus {control_label} band localization comparison",
        "control_label": control_label,
        "selected_splits": args.splits,
        "primary_analysis": (
            "test-set residue-level binary band-mask overlap, aggregated with "
            "equal protein weighting"
        ),
        "context_key": KEY,
        "catalog_scopes": list(SCOPES),
        "coverage_denominator": (
            "eligible_end_index_0based_exclusive - "
            "eligible_start_index_0based"
        ),
        "empty_mask_policy": (
            "Jaccard, Dice, and directional overlap are undefined when their "
            "denominator is zero; coverage remains zero"
        ),
        "apex_distance_thresholds": list(DISTANCE_THRESHOLDS),
        "band_matching": (
            "one-to-one Hungarian assignment maximizing same-sign interval "
            "IoU within context; assignments with IoU=0 are treated as unmatched"
        ),
        "band_iou_summary_thresholds": list(IOU_THRESHOLDS),
        "unmatched_catalog_definition": (
            "no positive-IoU assignment in the global maximum-IoU matching; "
            "threshold-specific failures remain in the matched table"
        ),
        "stability_annotation": (
            "mean bands present in stable_signed_bands.csv are supported by at "
            "least 2/3 seeds; absent bands are labelled <2 with exact 0-versus-1 "
            "support unavailable from the supplied stable-only catalog"
        ),
        "aggregation": "equal protein weighting",
        "bootstrap": {
            "unit": "protein row within condition/split/scope/sign/direction",
            "n_bootstrap": args.n_bootstrap,
            "random_seed": args.bootstrap_seed,
            "confidence_interval": "percentile 2.5% to 97.5%",
        },
        "test_set_primary": True,
        "inputs": {
            label.replace("uniform", control_label): {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in paths.items()
        },
    }
    _atomic_json(output_dir / "comparison_parameters.json", parameters)

    audit = {
        "all_invariants_pass": True,
        "input_band_counts": {
            f"{profile.replace('uniform', control_label)}_{scope}": len(frame)
            for (profile, scope), frame in catalogs.items()
        },
        "protein_contexts": len(universe) // 2,
        "protein_sign_contexts": len(universe),
        "mask_rows": len(mask_frame),
        "nearest_apex_rows": len(distance_frame),
        "matched_band_pairs": len(matched),
        "observed_only_bands": len(observed_only),
        f"{control_label}_only_bands": len(uniform_only),
        "band_classification_audit": band_classification_audit,
        "stable_subset_of_mean": {
            "observed": True,
            control_label: True,
        },
        "matching_context_and_sign_invariants": True,
        "input_intervals_nonoverlapping_within_catalog_context": True,
        "test_results_not_used_for_parameter_selection": True,
        "test_set_marked_primary": True,
    }
    _atomic_json(output_dir / "comparison_audit.json", audit)
    print(json.dumps({
        "output_dir": str(output_dir),
        "input_band_counts": audit["input_band_counts"],
        "protein_sign_contexts": len(universe),
        "matched_band_pairs": len(matched),
        "all_invariants_pass": True,
    }, indent=2))


if __name__ == "__main__":
    main()
