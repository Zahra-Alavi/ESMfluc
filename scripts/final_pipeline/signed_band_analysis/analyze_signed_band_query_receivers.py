#!/usr/bin/env python3
"""Analyze which query residues receive signed contribution from each band.

The contribution matrices are read once per condition, seed, and split.  Each
source file is reduced to per-protein band-by-query arrays, then the arrays are
averaged across seeds before receiver labels, matched effects, and predictive
models are calculated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .extract_signed_contribution_bands import MarkerScanner, header_scalar


PROFILE_METRICS = (
    "contribution_sum",
    "directional_contribution_sum",
    "directional_contribution_mean",
    "absolute_contribution_fraction",
    "directional_apex_contribution",
    "band_attention_sum",
    "band_attention_mean",
    "apex_attention",
)

QUERY_FEATURES = (
    "neq", "rsa", "disorder", "torsion_change_from_previous",
    "normalized_query_position", "query_in_any_band",
)

QUERY_MODEL_FEATURE_GROUPS = (
    ("neq", "numeric"),
    ("rsa", "numeric"),
    ("disorder", "numeric"),
    ("torsion_change_from_previous", "numeric"),
    ("query_amino_acid_class", "categorical"),
    ("normalized_query_position", "numeric"),
    ("query_in_any_band", "numeric"),
)

PAIR_GEOMETRY_FEATURES = (
    "minimum_ca_distance_angstrom",
    "minimum_heavy_atom_distance_angstrom",
    "relative_backbone_orientation_cosine",
    "absolute_backbone_orientation_cosine",
)

PAIR_CONTACT_FEATURES = (
    "direct_ca_contact",
    "query_to_band_contact_count",
    "query_to_band_contact_fraction",
    "ordinary_contact_path_exists",
    "shortest_ordinary_contact_path_edges",
    "nonlocal_contact_path_exists",
    "shortest_nonlocal_contact_path_edges",
)

PAIR_ORGANIZATION_FEATURES = (
    "same_community_as_band_apex",
    "same_community_as_band_majority",
    "query_distance_to_community_boundary_edges",
    "same_ecod_domain_as_band_apex",
    "shares_any_ecod_domain_with_band",
    "query_distance_to_ecod_boundary_sequence",
    "query_distance_to_band_domain_boundary_sequence",
)

PAIR_POLAR_FEATURES = (
    "minimum_polar_atom_distance_angstrom",
    "direct_putative_polar_contact",
    "direct_putative_polar_contact_count",
)

PAIR_WATER_FEATURES = (
    "query_water_contact_count",
    "band_water_contact_count",
    "shared_water_count",
    "shared_water_fraction_of_query_contacts",
    "shared_water_fraction_of_band_contacts",
    "one_water_bridge",
    "water_path_exists",
    "shortest_water_path_water_count",
    "water_path_at_most_1",
    "water_path_at_most_2",
    "water_path_at_most_3",
    "water_path_without_direct_ca_contact",
    "water_path_without_direct_polar_contact",
    "reachable_query_water_entry_count",
    "mean_query_contact_water_degree",
)

PAIR_QUALITY_FEATURES = (
    "query_resolved",
    "pair_structure_eligible",
    "water_structure_eligible",
    "band_resolved_fraction",
    "band_resolved_residue_count",
    "band_apex_resolved",
    "band_community_majority_defined",
    "band_ecod_annotation_fraction",
)

PAIR_UPGRADED_FEATURES = (
    *PAIR_GEOMETRY_FEATURES,
    *PAIR_CONTACT_FEATURES,
    *PAIR_ORGANIZATION_FEATURES,
    *PAIR_POLAR_FEATURES,
    *PAIR_WATER_FEATURES,
    *PAIR_QUALITY_FEATURES,
)

PAIR_STRUCTURE_FEATURES = (
    *PAIR_UPGRADED_FEATURES,
    # Legacy names remain accepted for backwards compatibility.
    "ca_distance", "direct_contact", "shortest_contact_network_path",
    "same_structural_community", "same_domain", "relative_orientation",
    "strain_covariance", "mechanical_coupling",
)

PAIR_EFFECT_FEATURES = (
    *PAIR_GEOMETRY_FEATURES,
    *PAIR_CONTACT_FEATURES,
    *PAIR_ORGANIZATION_FEATURES,
    *PAIR_POLAR_FEATURES,
    *PAIR_WATER_FEATURES,
)

MECHANISM_INTERACTION_QUERY_FEATURES = (
    "neq", "rsa", "disorder", "torsion_change_from_previous",
    "query_in_any_band",
)

DISTANCE_BINS = (-1, 0, 5, 20, 50, np.inf)
DISTANCE_LABELS = ("inside", "adjacent_1_5", "local_6_20", "distal_21_50", "distal_gt50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_tsv", required=True)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--protein_summary_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument("--mechanism_csv", default=None)
    parser.add_argument("--pairwise_structure_csv", default=None)
    parser.add_argument(
        "--pairwise_structure_dir",
        default=None,
        help=(
            "Partitioned feature-store root from "
            "signed_band_analysis.build_band_query_structural_water_features"
        ),
    )
    parser.add_argument(
        "--receiver_cache_source_dir",
        default=None,
        help=(
            "Optional existing Phase 4 condition output containing "
            "per_seed_query_profile_cache/ and band_query_profiles/. This "
            "allows an upgraded aggregate-only run without re-reading LxL matrices."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=(1, 2, 3))
    parser.add_argument("--receiver_quantile", type=float, default=0.90)
    parser.add_argument("--low_receiver_quantile", type=float, default=0.50)
    parser.add_argument("--long_range_min_separation", type=int, default=21)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--max_model_rows_per_class_per_protein", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--max_proteins_per_file", type=int, default=0)
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser.parse_args()


def read_matrix(scanner: MarkerScanner, length: int) -> np.ndarray:
    if scanner.take_until(b"[") is None:
        raise ValueError(f"{scanner.path}: missing matrix")
    matrix = np.empty((length, length), dtype=np.float64)
    for row in range(length):
        values = scanner.read_flat_float_array()
        if values.shape != (length,):
            raise ValueError(
                f"{scanner.path}: matrix row {row} has {values.shape}, expected {(length,)}"
            )
        matrix[row] = values
    return matrix


def iter_matrix_records(path: Path) -> tuple[dict, Iterator[dict]]:
    scanner = MarkerScanner(path)
    prefix = scanner.take_until(b'"proteins":[', capture=True)
    if prefix is None:
        scanner.close()
        raise ValueError(f"{path}: missing proteins array")
    metadata = {
        key: header_scalar(prefix, key)
        for key in ("schema_version", "condition", "seed", "split", "protein_count")
    }
    if metadata["schema_version"] != "esmfluc.signed_contributions.v1":
        scanner.close()
        raise ValueError(f"{path}: unsupported schema {metadata['schema_version']!r}")

    def records() -> Iterator[dict]:
        count = 0
        try:
            while scanner.take_until(b'"name":') is not None:
                name = scanner.read_json_string()
                if scanner.take_until(b'"sequence":') is None:
                    raise ValueError(f"{path}: missing sequence for {name}")
                sequence = scanner.read_json_string()
                if scanner.take_until(b'"length":') is None:
                    raise ValueError(f"{path}: missing length for {name}")
                length = scanner.read_integer()
                if length != len(sequence):
                    raise ValueError(f"{path}: length mismatch for {name}")
                if scanner.take_until(b'"contribution_matrix":') is None:
                    raise ValueError(f"{path}: missing contribution matrix for {name}")
                contribution = read_matrix(scanner, length)
                if scanner.take_until(b'"intrinsic_signed_evidence":') is None:
                    raise ValueError(f"{path}: missing evidence for {name}")
                evidence = scanner.read_flat_float_array()
                if scanner.take_until(b'"signed_column_influence":') is None:
                    raise ValueError(f"{path}: missing influence for {name}")
                influence = scanner.read_flat_float_array()
                if scanner.take_until(b'"attention_matrix":') is None:
                    raise ValueError(f"{path}: missing attention matrix for {name}")
                attention = read_matrix(scanner, length)
                if scanner.take_until(b'"attention_column_mean":') is None:
                    raise ValueError(f"{path}: missing attention mean for {name}")
                breadth = scanner.read_flat_float_array()
                for label, values in (
                    ("evidence", evidence), ("influence", influence), ("breadth", breadth)
                ):
                    if values.shape != (length,):
                        raise ValueError(f"{path}: {name} {label} has shape {values.shape}")
                reconstruction_error = float(np.max(np.abs(
                    contribution - attention * evidence[None, :]
                ))) if length else 0.0
                if reconstruction_error > 5e-6:
                    raise ValueError(
                        f"{path}: {name} C=s*A error {reconstruction_error:.3g}"
                    )
                count += 1
                yield {
                    "name": name, "sequence": sequence, "length": length,
                    "contribution": contribution, "attention": attention,
                    "reconstruction_error": reconstruction_error,
                }
            expected = metadata.get("protein_count")
            if expected is not None and count != int(expected):
                raise ValueError(f"{path}: read {count} proteins, expected {expected}")
        finally:
            scanner.close()

    return metadata, records()


def seed_cache_path(root: Path, condition: str, seed: int, split: str, protein: str) -> Path:
    return root / condition / f"seed_{seed}" / split / f"{protein}.npz"


def final_profile_path(root: Path, condition: str, split: str, protein: str) -> Path:
    return root / condition / split / f"{protein}.npz"


def band_arrays(record: dict, bands: pd.DataFrame) -> dict[str, np.ndarray]:
    contribution = record["contribution"]
    attention = record["attention"]
    absolute_total = np.abs(contribution).sum(axis=1)
    metrics = {name: [] for name in PROFILE_METRICS}
    for band in bands.itertuples(index=False):
        start = int(band.start_index_0based)
        end = int(band.end_index_0based_inclusive) + 1
        apex = int(band.apex_index_0based)
        sign = int(band.sign)
        width = end - start
        signed_sum = contribution[:, start:end].sum(axis=1)
        attention_sum = attention[:, start:end].sum(axis=1)
        metrics["contribution_sum"].append(signed_sum)
        metrics["directional_contribution_sum"].append(sign * signed_sum)
        metrics["directional_contribution_mean"].append(sign * signed_sum / width)
        metrics["absolute_contribution_fraction"].append(np.divide(
            np.abs(contribution[:, start:end]).sum(axis=1), absolute_total,
            out=np.zeros_like(absolute_total), where=absolute_total > 0,
        ))
        metrics["directional_apex_contribution"].append(sign * contribution[:, apex])
        metrics["band_attention_sum"].append(attention_sum)
        metrics["band_attention_mean"].append(attention_sum / width)
        metrics["apex_attention"].append(attention[:, apex])
    return {name: np.asarray(values, dtype=np.float32) for name, values in metrics.items()}


def extract_source(
    row, bands_by_context: dict, cache_root: Path, overwrite: bool,
    max_proteins: int,
) -> dict:
    condition, seed, split = str(row.condition), int(row.seed), str(row.split)
    source = Path(row.json_gz).expanduser().resolve()
    metadata, records = iter_matrix_records(source)
    if (str(metadata["condition"]), int(metadata["seed"]), str(metadata["split"])) != (
        condition, seed, split
    ):
        raise ValueError(f"Manifest/header mismatch for {source}")
    written = skipped = proteins_seen = bands_written = 0
    maximum_error = 0.0
    for record in records:
        proteins_seen += 1
        context = (condition, split, str(record["name"]))
        protein_bands = bands_by_context.get(context)
        if protein_bands is not None and not protein_bands.empty:
            destination = seed_cache_path(cache_root, condition, seed, split, record["name"])
            cache_is_current = False
            if destination.exists() and not overwrite:
                cached = load_seed_profile(destination)
                cache_is_current = (
                    "seed" in cached and int(cached["seed"][0]) == seed
                    and all(metric in cached for metric in PROFILE_METRICS)
                )
            if cache_is_current:
                skipped += 1
            else:
                arrays = band_arrays(record, protein_bands)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    band_id=np.asarray(protein_bands.band_id.astype(str).tolist(), dtype=str),
                    seed=np.asarray([seed], dtype=np.int16),
                    sign=protein_bands.sign.to_numpy(np.int8),
                    apex_index_0based=protein_bands.apex_index_0based.to_numpy(np.int32),
                    start_index_0based=protein_bands.start_index_0based.to_numpy(np.int32),
                    end_index_0based_inclusive=protein_bands.end_index_0based_inclusive.to_numpy(np.int32),
                    protein_length=np.asarray([record["length"]], dtype=np.int32),
                    **arrays,
                )
                written += 1
            bands_written += len(protein_bands)
        maximum_error = max(maximum_error, float(record["reconstruction_error"]))
        if max_proteins and proteins_seen >= max_proteins:
            records.close()
            break
    return {
        "condition": condition, "seed": seed, "split": split,
        "source": str(source), "proteins_seen": proteins_seen,
        "protein_caches_written": written, "protein_caches_reused": skipped,
        "bands_encountered": bands_written,
        "maximum_contribution_reconstruction_error": maximum_error,
    }


def load_seed_profile(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def load_partitioned_pair_features(
    root: Path, condition: str, split: str, protein: str,
    expected_band_ids: np.ndarray, expected_length: int,
) -> dict[str, np.ndarray] | None:
    path = root / "pair_features" / condition / split / f"{protein}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as loaded:
        values = {name: loaded[name] for name in loaded.files}
    schema = (
        str(values["schema_version"][0])
        if "schema_version" in values else ""
    )
    if schema != "esmfluc.phase4_structural_water.v1":
        raise ValueError(f"{path}: unsupported feature schema {schema!r}")
    if int(values["protein_length"][0]) != expected_length:
        raise ValueError(f"{path}: protein length differs from receiver profile")
    if not np.array_equal(values["band_id"].astype(str), expected_band_ids.astype(str)):
        raise ValueError(f"{path}: band identities/order differ from receiver profile")
    n_bands = len(expected_band_ids)
    pair_names = values.get("pair_feature_names", np.asarray([], dtype=str)).astype(str)
    band_names = values.get("band_feature_names", np.asarray([], dtype=str)).astype(str)
    for name in pair_names:
        if values[name].shape != (n_bands, expected_length):
            raise ValueError(
                f"{path}: {name} shape {values[name].shape}, "
                f"expected {(n_bands, expected_length)}"
            )
    for name in band_names:
        if values[name].shape != (n_bands,):
            raise ValueError(
                f"{path}: {name} shape {values[name].shape}, expected {(n_bands,)}"
            )
    values["_source_path"] = np.asarray([str(path)])
    return values


def average_seed_profiles(
    paths: list[Path], output_path: Path, expected_seeds: set[int]
) -> dict:
    profiles = [load_seed_profile(path) for path in sorted(paths)]
    found_seeds = {int(profile["seed"][0]) for profile in profiles}
    if found_seeds != expected_seeds:
        raise ValueError(
            f"{output_path}: found seeds {sorted(found_seeds)}, "
            f"expected {sorted(expected_seeds)}"
        )
    identity_fields = (
        "band_id", "sign", "apex_index_0based", "start_index_0based",
        "end_index_0based_inclusive", "protein_length",
    )
    for field in identity_fields:
        if any(not np.array_equal(profiles[0][field], profile[field]) for profile in profiles[1:]):
            raise ValueError(f"{output_path}: seed caches disagree on {field}")
    averaged = {
        metric: np.mean([profile[metric].astype(np.float64) for profile in profiles], axis=0).astype(np.float32)
        for metric in PROFILE_METRICS
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{field: profiles[0][field] for field in identity_fields},
        source_seeds=np.asarray(sorted(found_seeds), dtype=np.int16),
        **averaged,
    )
    return {"bands": len(profiles[0]["band_id"]), "queries": int(profiles[0]["protein_length"][0])}


def q8_segment_numbers(labels: np.ndarray) -> np.ndarray:
    if not len(labels):
        return np.empty(0, dtype=np.int32)
    return np.cumsum(np.r_[True, labels[1:] != labels[:-1]]).astype(np.int32)


def amino_acid_class(amino_acid: str) -> str:
    if amino_acid in "DE":
        return "acidic"
    if amino_acid in "KR":
        return "basic"
    if amino_acid in "AVILM":
        return "aliphatic"
    if amino_acid in "FWY":
        return "aromatic"
    if amino_acid in "STNQ":
        return "polar"
    if amino_acid in "GP":
        return "special"
    return "other"


def bh(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return output
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index.to_numpy()[order]] = np.minimum(adjusted, 1.0)
    return output


def receiver_rows_for_profile(
    profile: dict, condition: str, split: str, protein: str,
    residue: pd.DataFrame, protein_bands: pd.DataFrame,
    summary_row: pd.Series, mechanism: dict[str, str],
    receiver_quantile: float, low_quantile: float, long_range_min: int,
    structure: pd.DataFrame | dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    residue = residue.sort_values("residue_index_0based").reset_index(drop=True)
    length = int(profile["protein_length"][0])
    if len(residue) != length or not np.array_equal(
        residue.residue_index_0based.to_numpy(int), np.arange(length)
    ):
        raise ValueError(f"{condition}/{split}/{protein}: annotation coordinates do not match")
    eligible = np.zeros(length, dtype=bool)
    eligible[int(summary_row.eligible_start_index_0based):int(summary_row.eligible_end_index_0based_exclusive)] = True
    any_band = np.zeros(length, dtype=bool)
    for band in protein_bands.itertuples(index=False):
        any_band[int(band.start_index_0based):int(band.end_index_0based_inclusive) + 1] = True
    q8 = residue.q8.fillna("unknown").astype(str).to_numpy()
    q8_segment = q8_segment_numbers(q8)
    indices = np.arange(length)
    annotations = {
        name: pd.to_numeric(residue[name], errors="coerce").to_numpy(float)
        for name in ("neq", "rsa", "disorder", "torsion_change_from_previous")
    }
    amino_acids = residue.amino_acid.fillna("X").astype(str).to_numpy()
    pair_frames = []
    effect_rows: list[dict] = []
    long_rows: list[dict] = []
    for band_number, band_id in enumerate(profile["band_id"].astype(str)):
        sign = int(profile["sign"][band_number])
        apex = int(profile["apex_index_0based"][band_number])
        start = int(profile["start_index_0based"][band_number])
        end = int(profile["end_index_0based_inclusive"][band_number])
        separation = np.where(indices < start, start - indices, np.where(indices > end, indices - end, 0))
        signed_direction = np.sign(indices - apex).astype(np.int8)
        distance_bin = np.asarray(pd.cut(
            separation, bins=DISTANCE_BINS, labels=DISTANCE_LABELS, include_lowest=True
        ).astype(str))
        directional = profile["directional_contribution_sum"][band_number].astype(float)
        valid_values = directional[eligible & np.isfinite(directional)]
        high_threshold = float(np.quantile(valid_values, receiver_quantile))
        low_threshold = float(np.quantile(valid_values, low_quantile))
        high = eligible & (directional >= high_threshold)
        low = eligible & (directional <= low_threshold) & ~high
        base = pd.DataFrame({
            "condition": condition, "split": split, "protein": protein,
            "band_id": band_id, "sign": sign,
            "mechanism_class": mechanism.get(band_id, "unclassified"),
            "query_index_0based": indices, "query_amino_acid": amino_acids,
            "query_amino_acid_class": [amino_acid_class(value) for value in amino_acids],
            "query_q8": q8, "query_q8_segment_number": q8_segment,
            "source_q8": q8[apex], "source_q8_segment_number": q8_segment[apex],
            "same_q8_segment": q8_segment == q8_segment[apex],
            "absolute_sequence_separation": np.abs(indices - apex),
            "distance_to_band_interval": separation,
            "signed_sequence_direction": signed_direction,
            "distance_bin": distance_bin,
            "query_in_any_band": any_band.astype(np.int8),
            "query_inside_source_band": ((indices >= start) & (indices <= end)).astype(np.int8),
            "normalized_query_position": indices / max(1, length - 1),
            "high_receiver": high.astype(np.int8), "low_receiver": low.astype(np.int8),
            "receiver_threshold": high_threshold, "low_receiver_threshold": low_threshold,
        })
        for name, values in annotations.items():
            base[name] = values
        for metric in PROFILE_METRICS:
            base[metric] = profile[metric][band_number]
        base = base[eligible].copy()
        if isinstance(structure, dict):
            feature_names = [
                *structure.get("pair_feature_names", np.asarray([], dtype=str)).astype(str),
                *structure.get("band_feature_names", np.asarray([], dtype=str)).astype(str),
            ]
            for name in feature_names:
                if name not in PAIR_STRUCTURE_FEATURES:
                    continue
                values = structure[name]
                if values.ndim == 2:
                    if values.shape[1] != length:
                        raise ValueError(
                            f"{condition}/{split}/{protein}: {name} has "
                            f"{values.shape[1]} query positions, expected {length}"
                        )
                    base[name] = values[band_number, eligible]
                elif values.ndim == 1:
                    base[name] = float(values[band_number])
                else:
                    raise ValueError(
                        f"{condition}/{split}/{protein}: invalid shape for {name}"
                    )
        elif structure is not None and not structure.empty:
            source_structure = structure[structure.band_id.astype(str).eq(band_id)].copy()
            keep = ["query_index_0based"] + [
                name for name in PAIR_STRUCTURE_FEATURES if name in source_structure
            ]
            if len(keep) > 1:
                base = base.merge(
                    source_structure[keep], on="query_index_0based", how="left",
                    validate="one_to_one",
                )
        pair_frames.append(base)

        strata = ["query_q8", "distance_bin"]
        matched = base[base.high_receiver.eq(1) | base.low_receiver.eq(1)].copy()
        effect_features = QUERY_FEATURES + ("absolute_sequence_separation",) + tuple(
            name for name in PAIR_EFFECT_FEATURES if name in base
        )
        distal_scope = f"sequence_distal_ge{long_range_min}"
        scopes = {
            "all_eligible_queries": np.ones(len(matched), dtype=bool),
            distal_scope: (
                matched.distance_to_band_interval.to_numpy(float) >= long_range_min
            ),
        }
        if "direct_ca_contact" in matched:
            scopes[f"{distal_scope}_no_direct_ca"] = (
                (matched.distance_to_band_interval.to_numpy(float) >= long_range_min)
                & ~matched.direct_ca_contact.fillna(1).astype(bool).to_numpy()
            )
        for scope, scope_mask in scopes.items():
            scoped = matched.loc[scope_mask]
            if scoped.empty:
                continue
            for feature in effect_features:
                stratum_effects = []
                for _, group in scoped.groupby(strata, sort=False):
                    cases = pd.to_numeric(
                        group.loc[group.high_receiver.eq(1), feature],
                        errors="coerce",
                    )
                    controls = pd.to_numeric(
                        group.loc[group.low_receiver.eq(1), feature],
                        errors="coerce",
                    )
                    if cases.notna().any() and controls.notna().any():
                        stratum_effects.append(float(cases.mean() - controls.mean()))
                if stratum_effects:
                    effect_rows.append({
                        "condition": condition, "split": split, "protein": protein,
                        "band_id": band_id, "sign": sign,
                        "mechanism_class": mechanism.get(band_id, "unclassified"),
                        "receiver_scope": scope,
                        "feature": feature, "n_matched_strata": len(stratum_effects),
                        "high_minus_low": float(np.mean(stratum_effects)),
                    })
        long_mask = base.distance_to_band_interval >= long_range_min
        positive_mass = np.maximum(base.directional_contribution_sum.to_numpy(float), 0)
        attention_mass = base.band_attention_sum.to_numpy(float)
        long_rows.append({
            "condition": condition, "split": split, "protein": protein,
            "band_id": band_id, "sign": sign,
            "mechanism_class": mechanism.get(band_id, "unclassified"),
            "n_queries": len(base), "n_high_receivers": int(base.high_receiver.sum()),
            "fraction_queries_long_range": float(long_mask.mean()),
            "fraction_high_receivers_long_range": float(base.loc[long_mask, "high_receiver"].sum() / max(1, base.high_receiver.sum())),
            "fraction_positive_directional_mass_long_range": float(positive_mass[long_mask].sum() / positive_mass.sum()) if positive_mass.sum() else np.nan,
            "fraction_attention_mass_long_range": float(attention_mass[long_mask].sum() / attention_mass.sum()) if attention_mass.sum() else np.nan,
        })
    return pd.concat(pair_frames, ignore_index=True), effect_rows, long_rows


def summarize_effects(band_effects: pd.DataFrame, minimum_proteins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    protein = band_effects.groupby(
        [
            "condition", "split", "protein", "sign", "mechanism_class",
            "receiver_scope", "feature",
        ],
        as_index=False,
    ).agg(n_bands=("band_id", "nunique"), high_minus_low=("high_minus_low", "mean"))
    rows = []
    keys = [
        "condition", "split", "sign", "mechanism_class",
        "receiver_scope", "feature",
    ]
    for key, group in protein.groupby(keys, sort=False):
        values = group.high_minus_low.dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        eligible = n >= minimum_proteins
        half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if eligible else np.nan
        nonzero = values[values != 0]
        p = (
            float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
            if eligible and len(nonzero) else (1.0 if eligible else np.nan)
        )
        rows.append({
            **dict(zip(keys, key)), "n_proteins": n, "n_bands": int(group.n_bands.sum()),
            "high_minus_low_macro_protein_mean": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "inference_eligible": eligible, "protein_sign_p_two_sided": p,
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["protein_sign_q_bh"] = bh(summary.protein_sign_p_two_sided)
    return protein, summary


def summarize_long_range(band_rows: pd.DataFrame) -> pd.DataFrame:
    protein = band_rows.groupby(
        ["condition", "split", "protein", "sign", "mechanism_class"], as_index=False
    ).agg(
        n_bands=("band_id", "nunique"),
        fraction_high_receivers_long_range=("fraction_high_receivers_long_range", "mean"),
        fraction_positive_directional_mass_long_range=("fraction_positive_directional_mass_long_range", "mean"),
        fraction_attention_mass_long_range=("fraction_attention_mass_long_range", "mean"),
    )
    return protein.groupby(
        ["condition", "split", "sign", "mechanism_class"], as_index=False
    ).agg(
        n_proteins=("protein", "nunique"), n_bands=("n_bands", "sum"),
        fraction_high_receivers_long_range_macro_mean=("fraction_high_receivers_long_range", "mean"),
        fraction_positive_directional_mass_long_range_macro_mean=("fraction_positive_directional_mass_long_range", "mean"),
        fraction_attention_mass_long_range_macro_mean=("fraction_attention_mass_long_range", "mean"),
    )


def sample_model_rows(frame: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby("band_id", sort=False):
        for _, stratum in group.groupby(["query_q8", "distance_bin"], sort=False):
            high = stratum[stratum.high_receiver.eq(1)]
            low = stratum[stratum.low_receiver.eq(1)]
            if high.empty or low.empty:
                continue
            n = min(len(high), len(low))
            selected.append(high.iloc[rng.choice(len(high), n, replace=False)])
            selected.append(low.iloc[rng.choice(len(low), n, replace=False)])
    if not selected:
        return pd.DataFrame()
    data = pd.concat(selected, ignore_index=True)
    parts = []
    for label in (0, 1):
        subset = data[data.high_receiver.eq(label)]
        if len(subset) > cap:
            subset = subset.iloc[rng.choice(len(subset), cap, replace=False)]
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def add_mechanism_interactions(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = data.copy()
    mechanisms = (
        "combined",
        "evidence-dominated",
        "consultation-dominated",
        "not_magnitude_enriched",
        "unclassified",
    )
    interaction_columns = []
    data["mechanism_x_query_q8"] = (
        data.mechanism_class.astype(str) + "__" + data.query_q8.astype(str)
    )
    data["mechanism_x_query_amino_acid_class"] = (
        data.mechanism_class.astype(str)
        + "__"
        + data.query_amino_acid_class.astype(str)
    )
    for mechanism_name in mechanisms:
        indicator = data.mechanism_class.astype(str).eq(mechanism_name).astype(float)
        safe_name = mechanism_name.replace("-", "_")
        for feature in MECHANISM_INTERACTION_QUERY_FEATURES:
            name = f"interaction__{safe_name}__{feature}"
            data[name] = pd.to_numeric(data[feature], errors="coerce") * indicator
            interaction_columns.append(name)
    return data, interaction_columns


def model_scope_mask(
    data: pd.DataFrame, scope: str, long_range_minimum: int
) -> np.ndarray:
    if scope == "all_eligible_queries":
        return np.ones(len(data), dtype=bool)
    distal = (
        pd.to_numeric(data.distance_to_band_interval, errors="coerce").to_numpy(float)
        >= long_range_minimum
    )
    if scope.startswith("sequence_distal_ge") and not scope.endswith("_no_direct_ca"):
        return distal
    if scope.startswith("sequence_distal_ge") and scope.endswith("_no_direct_ca"):
        if "direct_ca_contact" not in data:
            return np.zeros(len(data), dtype=bool)
        no_contact = ~data.direct_ca_contact.fillna(1).astype(bool).to_numpy()
        return distal & no_contact
    raise ValueError(f"Unknown receiver scope {scope}")


def cumulative_stage_specs(
    data: pd.DataFrame, include_structure: bool, include_water: bool,
    interaction_columns: list[str],
) -> list[tuple[str, list[str], list[str]]]:
    baseline_categorical = ["query_q8", "distance_bin"]
    baseline_numeric = [
        "absolute_sequence_separation", "signed_sequence_direction",
        "query_inside_source_band",
    ]
    query_categorical = baseline_categorical + ["query_amino_acid_class"]
    query_numeric = baseline_numeric + list(QUERY_FEATURES)
    stages = [
        ("00_distance_q8", baseline_categorical, baseline_numeric),
        ("01_add_query_biophysics", query_categorical, query_numeric),
    ]
    current_numeric = list(query_numeric)
    if include_structure:
        for stage, features in (
            ("02_add_3d_geometry", PAIR_GEOMETRY_FEATURES),
            ("03_add_contact_network", PAIR_CONTACT_FEATURES),
            ("04_add_community_domain", PAIR_ORGANIZATION_FEATURES),
            ("05_add_putative_polar_contacts", PAIR_POLAR_FEATURES),
        ):
            available = [
                name for name in features
                if name in data
                and pd.to_numeric(data[name], errors="coerce").notna().any()
            ]
            if available:
                current_numeric.extend(available)
                stages.append((stage, query_categorical, list(current_numeric)))
    if include_water:
        available_water = [
            name for name in PAIR_WATER_FEATURES
            if name in data
            and pd.to_numeric(data[name], errors="coerce").notna().any()
        ]
        if available_water:
            current_numeric.extend(available_water)
            stages.append(
                ("06_add_crystallographic_water_network",
                 query_categorical, list(current_numeric))
            )
    interaction_numeric = [
        name for name in interaction_columns
        if name in data
        and pd.to_numeric(data[name], errors="coerce").notna().any()
    ]
    mechanism_categorical = query_categorical + [
        "mechanism_class",
        "mechanism_x_query_q8",
        "mechanism_x_query_amino_acid_class",
    ]
    stages.append((
        "07_add_source_mechanism_interactions",
        mechanism_categorical,
        current_numeric + interaction_numeric,
    ))
    return stages


def receiver_models(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    data, interaction_columns = add_mechanism_interactions(data)
    distal_scope = f"sequence_distal_ge{args.long_range_min_separation}"
    cohort_specs = [
        (
            "all_receivers",
            np.ones(len(data), dtype=bool),
            False,
            False,
            ("all_eligible_queries", distal_scope),
        )
    ]
    if "pair_structure_eligible" in data:
        structure_mask = data.pair_structure_eligible.fillna(0).eq(1).to_numpy()
        cohort_specs.append((
            "structure_eligible_fixed",
            structure_mask,
            True,
            False,
            ("all_eligible_queries", distal_scope),
        ))
    if "water_structure_eligible" in data:
        water_mask = data.water_structure_eligible.fillna(0).eq(1).to_numpy()
        cohort_specs.append((
            "water_eligible_fixed",
            water_mask,
            True,
            True,
            (
                "all_eligible_queries",
                distal_scope,
                f"{distal_scope}_no_direct_ca",
            ),
        ))
    rows = []
    for cohort, cohort_mask, include_structure, include_water, scopes in cohort_specs:
        cohort_data = data.loc[cohort_mask].copy()
        if cohort_data.empty:
            continue
        stages = cumulative_stage_specs(
            cohort_data, include_structure, include_water, interaction_columns
        )
        for scope in scopes:
            scoped = cohort_data.loc[
                model_scope_mask(cohort_data, scope, args.long_range_min_separation)
            ].copy()
            if scoped.empty:
                continue
            for condition in sorted(scoped.condition.unique()):
                for sign in (-1, 1):
                    train = scoped[
                        scoped.condition.eq(condition)
                        & scoped.sign.eq(sign)
                        & scoped.split.eq("train")
                    ]
                    if train.high_receiver.nunique() < 2:
                        continue
                    for stage, categorical, numeric in stages:
                        numeric = [
                            name for name in dict.fromkeys(numeric)
                            if name in train
                            and pd.to_numeric(train[name], errors="coerce").notna().any()
                        ]
                        model = receiver_model_pipeline(categorical, numeric, args)
                        model.fit(train[categorical + numeric], train.high_receiver)
                        for split in ("validation", "test"):
                            evaluate = scoped[
                                scoped.condition.eq(condition)
                                & scoped.sign.eq(sign)
                                & scoped.split.eq(split)
                            ]
                            if evaluate.high_receiver.nunique() < 2:
                                continue
                            metrics = receiver_model_metrics(
                                model, evaluate, categorical, numeric
                            )
                            rows.append({
                                "condition": condition,
                                "sign": sign,
                                "model_cohort": cohort,
                                "receiver_scope": scope,
                                "stage": stage,
                                "evaluation_split": split,
                                "n_pairs": len(evaluate),
                                "n_proteins": int(evaluate.protein.nunique()),
                                **metrics,
                            })
    performance = pd.DataFrame(rows)
    if not performance.empty:
        performance["delta_auroc_from_previous_stage"] = performance.groupby(
            [
                "condition", "sign", "model_cohort",
                "receiver_scope", "evaluation_split",
            ],
            sort=False,
        ).auroc.diff()
    return performance


def receiver_model_pipeline(
    categorical: list[str], numeric: list[str], args: argparse.Namespace
) -> Pipeline:
    categorical = list(dict.fromkeys(categorical))
    numeric = list(dict.fromkeys(numeric))
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    return Pipeline([
        ("transform", transformer),
        ("logistic", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=args.max_iter,
            random_state=args.random_seed,
        )),
    ])


def receiver_model_metrics(
    model: Pipeline, evaluate: pd.DataFrame,
    categorical: list[str], numeric: list[str],
) -> dict:
    score = model.predict_proba(evaluate[categorical + numeric])[:, 1]
    protein_auc = []
    scored = evaluate[["protein", "high_receiver"]].copy()
    scored["score"] = score
    for _, group in scored.groupby("protein", sort=False):
        if group.high_receiver.nunique() == 2:
            protein_auc.append(roc_auc_score(group.high_receiver, group.score))
    return {
        "auroc": float(roc_auc_score(evaluate.high_receiver, score)),
        "average_precision": float(average_precision_score(
            evaluate.high_receiver, score
        )),
        "macro_protein_auroc": (
            float(np.mean(protein_auc)) if protein_auc else np.nan
        ),
    }


def receiver_feature_ablation(
    data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    baseline_categorical = ["query_q8", "distance_bin"]
    baseline_numeric = [
        "absolute_sequence_separation", "signed_sequence_direction",
        "query_inside_source_band",
    ]
    full_categorical = baseline_categorical + ["query_amino_acid_class"]
    full_numeric = baseline_numeric + list(QUERY_FEATURES)
    rows = []
    for condition in sorted(data.condition.unique()):
        for sign in (-1, 1):
            train = data[
                data.condition.eq(condition)
                & data.sign.eq(sign)
                & data.split.eq("train")
            ]
            if train.high_receiver.nunique() < 2:
                continue
            specifications = {
                "baseline": (baseline_categorical, baseline_numeric),
                "full_query": (full_categorical, full_numeric),
            }
            for feature, feature_type in QUERY_MODEL_FEATURE_GROUPS:
                single_categorical = list(baseline_categorical)
                single_numeric = list(baseline_numeric)
                without_categorical = list(full_categorical)
                without_numeric = list(full_numeric)
                if feature_type == "categorical":
                    single_categorical.append(feature)
                    without_categorical.remove(feature)
                else:
                    single_numeric.append(feature)
                    without_numeric.remove(feature)
                specifications[f"single__{feature}"] = (
                    single_categorical, single_numeric
                )
                specifications[f"without__{feature}"] = (
                    without_categorical, without_numeric
                )
            fitted = {}
            for name, (categorical, numeric) in specifications.items():
                model = receiver_model_pipeline(categorical, numeric, args)
                model.fit(train[categorical + numeric], train.high_receiver)
                fitted[name] = (model, categorical, numeric)
            for split in ("validation", "test"):
                evaluate = data[
                    data.condition.eq(condition)
                    & data.sign.eq(sign)
                    & data.split.eq(split)
                ]
                if evaluate.high_receiver.nunique() < 2:
                    continue
                metrics = {
                    name: receiver_model_metrics(
                        model, evaluate, categorical, numeric
                    )
                    for name, (model, categorical, numeric) in fitted.items()
                }
                for feature, feature_type in QUERY_MODEL_FEATURE_GROUPS:
                    baseline = metrics["baseline"]
                    single = metrics[f"single__{feature}"]
                    full = metrics["full_query"]
                    without = metrics[f"without__{feature}"]
                    rows.append({
                        "condition": condition, "sign": sign,
                        "evaluation_split": split,
                        "feature": feature, "feature_type": feature_type,
                        "n_pairs": len(evaluate),
                        "n_proteins": int(evaluate.protein.nunique()),
                        "baseline_auroc": baseline["auroc"],
                        "single_feature_auroc": single["auroc"],
                        "delta_auroc_single_from_baseline": (
                            single["auroc"] - baseline["auroc"]
                        ),
                        "full_query_auroc": full["auroc"],
                        "without_feature_auroc": without["auroc"],
                        "delta_auroc_full_minus_without": (
                            full["auroc"] - without["auroc"]
                        ),
                        "baseline_macro_protein_auroc": (
                            baseline["macro_protein_auroc"]
                        ),
                        "single_feature_macro_protein_auroc": (
                            single["macro_protein_auroc"]
                        ),
                        "delta_macro_auroc_single_from_baseline": (
                            single["macro_protein_auroc"]
                            - baseline["macro_protein_auroc"]
                        ),
                        "full_query_macro_protein_auroc": (
                            full["macro_protein_auroc"]
                        ),
                        "without_feature_macro_protein_auroc": (
                            without["macro_protein_auroc"]
                        ),
                        "delta_macro_auroc_full_minus_without": (
                            full["macro_protein_auroc"]
                            - without["macro_protein_auroc"]
                        ),
                        "baseline_average_precision": (
                            baseline["average_precision"]
                        ),
                        "single_feature_average_precision": (
                            single["average_precision"]
                        ),
                        "full_query_average_precision": (
                            full["average_precision"]
                        ),
                        "without_feature_average_precision": (
                            without["average_precision"]
                        ),
                    })
    return pd.DataFrame(rows)


def receiver_structural_water_ablation(
    data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Evaluate feature groups on one fixed primary-water-eligible cohort."""
    if data.empty or "water_structure_eligible" not in data:
        return pd.DataFrame()
    data = data[data.water_structure_eligible.fillna(0).eq(1)].copy()
    if data.empty:
        return pd.DataFrame()
    categorical = ["query_q8", "distance_bin", "query_amino_acid_class"]
    baseline_numeric = [
        "absolute_sequence_separation", "signed_sequence_direction",
        "query_inside_source_band", *QUERY_FEATURES,
    ]
    feature_groups = {
        "geometry": list(PAIR_GEOMETRY_FEATURES),
        "contact_network": list(PAIR_CONTACT_FEATURES),
        "community_domain": list(PAIR_ORGANIZATION_FEATURES),
        "putative_polar_contacts": list(PAIR_POLAR_FEATURES),
        "water_network": list(PAIR_WATER_FEATURES),
    }
    for group, features in feature_groups.items():
        feature_groups[group] = [
            name for name in features
            if name in data
            and pd.to_numeric(data[name], errors="coerce").notna().any()
        ]
    ordinary = sum(
        [
            feature_groups["geometry"],
            feature_groups["contact_network"],
            feature_groups["community_domain"],
            feature_groups["putative_polar_contacts"],
        ],
        [],
    )
    full = ordinary + feature_groups["water_network"]
    specifications = {
        "query_biophysics": list(baseline_numeric),
        "query_plus_ordinary_structure": baseline_numeric + ordinary,
        "query_plus_structure_and_water": baseline_numeric + full,
        "full_without_rsa": [
            name for name in baseline_numeric + full if name != "rsa"
        ],
    }
    for group, features in feature_groups.items():
        specifications[f"full_without__{group}"] = [
            name for name in baseline_numeric + full if name not in set(features)
        ]

    rows = []
    distal_scope = f"sequence_distal_ge{args.long_range_min_separation}"
    for scope in (
        "all_eligible_queries",
        distal_scope,
        f"{distal_scope}_no_direct_ca",
    ):
        scoped = data.loc[
            model_scope_mask(data, scope, args.long_range_min_separation)
        ].copy()
        if scoped.empty:
            continue
        for condition in sorted(scoped.condition.unique()):
            for sign in (-1, 1):
                train = scoped[
                    scoped.condition.eq(condition)
                    & scoped.sign.eq(sign)
                    & scoped.split.eq("train")
                ]
                if train.high_receiver.nunique() < 2:
                    continue
                fitted = {}
                for specification, numeric in specifications.items():
                    numeric = [
                        name for name in dict.fromkeys(numeric)
                        if name in train
                        and pd.to_numeric(train[name], errors="coerce").notna().any()
                    ]
                    model = receiver_model_pipeline(categorical, numeric, args)
                    model.fit(train[categorical + numeric], train.high_receiver)
                    fitted[specification] = (model, numeric)
                for split in ("validation", "test"):
                    evaluate = scoped[
                        scoped.condition.eq(condition)
                        & scoped.sign.eq(sign)
                        & scoped.split.eq(split)
                    ]
                    if evaluate.high_receiver.nunique() < 2:
                        continue
                    metrics = {
                        name: receiver_model_metrics(
                            model, evaluate, categorical, numeric
                        )
                        for name, (model, numeric) in fitted.items()
                    }
                    query_auc = metrics["query_biophysics"]["auroc"]
                    ordinary_auc = metrics["query_plus_ordinary_structure"]["auroc"]
                    full_auc = metrics["query_plus_structure_and_water"]["auroc"]
                    for specification, result in metrics.items():
                        rows.append({
                            "condition": condition,
                            "sign": sign,
                            "model_cohort": "water_eligible_fixed",
                            "receiver_scope": scope,
                            "evaluation_split": split,
                            "specification": specification,
                            "n_pairs": len(evaluate),
                            "n_proteins": int(evaluate.protein.nunique()),
                            **result,
                            "delta_auroc_from_query_biophysics": (
                                result["auroc"] - query_auc
                            ),
                            "delta_auroc_from_ordinary_structure": (
                                result["auroc"] - ordinary_auc
                            ),
                            "delta_auroc_full_minus_specification": (
                                full_auc - result["auroc"]
                            ),
                        })
    return pd.DataFrame(rows)


def aggregate(args: argparse.Namespace, manifest: pd.DataFrame, bands: pd.DataFrame,
              summary: pd.DataFrame, residue: pd.DataFrame, output: Path) -> dict:
    cache_source = (
        Path(args.receiver_cache_source_dir).expanduser().resolve()
        if args.receiver_cache_source_dir else output
    )
    cache_root = cache_source / "per_seed_query_profile_cache"
    source_profile_root = cache_source / "band_query_profiles"
    generated_profile_root = output / "band_query_profiles"
    expected_seeds = set(map(int, args.seeds))
    cache_groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for row in manifest.itertuples(index=False):
        directory = cache_root / str(row.condition) / f"seed_{int(row.seed)}" / str(row.split)
        if directory.exists():
            for path in directory.glob("*.npz"):
                cache_groups[(str(row.condition), str(row.split), path.stem)].append(path)
    bands_lookup = {
        key: group.sort_values("band_id").reset_index(drop=True)
        for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    residue_lookup = {
        key: group for key, group in residue.groupby(["split", "protein"], sort=False)
    }
    summary_lookup = summary.set_index(["condition", "split", "protein"])
    mechanism = {}
    if args.mechanism_csv:
        mechanism_frame = pd.read_csv(args.mechanism_csv, usecols=["band_id", "mechanism_class"])
        mechanism = dict(zip(mechanism_frame.band_id.astype(str), mechanism_frame.mechanism_class.astype(str)))
    structure_lookup = {}
    if args.pairwise_structure_csv:
        structure_frame = pd.read_csv(args.pairwise_structure_csv)
        required = {"condition", "split", "protein", "band_id", "query_index_0based"}
        missing = required - set(structure_frame)
        if missing:
            raise ValueError(f"Pairwise structure table lacks {sorted(missing)}")
        structure_lookup = {
            key: group for key, group in structure_frame.groupby(
                ["condition", "split", "protein"], sort=False
            )
        }
    structure_root = (
        Path(args.pairwise_structure_dir).expanduser().resolve()
        if args.pairwise_structure_dir else None
    )

    pair_output = output / "band_query_pairs.csv.gz"
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    pair_handle = gzip.open(pair_output, "wt", newline="")
    writer = None
    effect_rows: list[dict] = []
    long_rows: list[dict] = []
    model_rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(args.random_seed)
    profiles_written = pairs_written = 0
    try:
        for context, paths in sorted(cache_groups.items()):
            condition, split, protein = context
            if context not in bands_lookup or (split, protein) not in residue_lookup:
                continue
            final_path = final_profile_path(
                source_profile_root, condition, split, protein
            )
            final_is_current = False
            if final_path.exists() and not args.overwrite_cache:
                existing = load_seed_profile(final_path)
                final_is_current = (
                    "source_seeds" in existing
                    and set(map(int, existing["source_seeds"])) == expected_seeds
                    and all(metric in existing for metric in PROFILE_METRICS)
                )
            if not final_is_current:
                final_path = final_profile_path(
                    generated_profile_root, condition, split, protein
                )
                average_seed_profiles(paths, final_path, expected_seeds)
            profile = load_seed_profile(final_path)
            expected_ids = bands_lookup[context].sort_values("band_id").band_id.astype(str).to_numpy()
            if not np.array_equal(profile["band_id"].astype(str), expected_ids):
                raise ValueError(f"{context}: band identities differ between table and cache")
            structure = structure_lookup.get(context)
            if structure_root is not None:
                structure = load_partitioned_pair_features(
                    structure_root, condition, split, protein,
                    profile["band_id"].astype(str),
                    int(profile["protein_length"][0]),
                )
            pairs, effects, long = receiver_rows_for_profile(
                profile, condition, split, protein, residue_lookup[(split, protein)],
                bands_lookup[context], summary_lookup.loc[context], mechanism,
                args.receiver_quantile, args.low_receiver_quantile,
                args.long_range_min_separation, structure,
            )
            if structure_root is not None:
                for feature in PAIR_UPGRADED_FEATURES:
                    if feature not in pairs:
                        pairs[feature] = np.nan
            if writer is None:
                writer = csv.DictWriter(pair_handle, fieldnames=pairs.columns.tolist())
                writer.writeheader()
            writer.writerows(pairs.to_dict("records"))
            pairs_written += len(pairs)
            effect_rows.extend(effects)
            long_rows.extend(long)
            sampled = sample_model_rows(
                pairs, args.max_model_rows_per_class_per_protein, rng
            )
            if not sampled.empty:
                model_rows.append(sampled)
            profiles_written += 1
    finally:
        pair_handle.close()

    band_effects = pd.DataFrame(effect_rows)
    per_protein, effect_summary = summarize_effects(
        band_effects, args.minimum_inference_proteins
    ) if not band_effects.empty else (pd.DataFrame(), pd.DataFrame())
    long_summary = summarize_long_range(pd.DataFrame(long_rows)) if long_rows else pd.DataFrame()
    model_data = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    performance = receiver_models(model_data, args)
    feature_ablation = receiver_feature_ablation(model_data, args)
    structural_water_ablation = receiver_structural_water_ablation(model_data, args)
    per_protein.to_csv(output / "receiver_feature_effects_by_protein.csv.gz", index=False, compression="gzip")
    effect_summary.to_csv(output / "receiver_feature_summary.csv", index=False)
    performance.to_csv(output / "receiver_model_performance.csv", index=False)
    feature_ablation.to_csv(
        output / "receiver_feature_ablation_performance.csv", index=False
    )
    structural_water_ablation.to_csv(
        output / "receiver_structural_water_ablation_performance.csv", index=False
    )
    long_summary.to_csv(output / "long_range_receiver_summary.csv", index=False)
    return {
        "averaged_protein_profiles": profiles_written,
        "band_query_pairs": pairs_written,
        "protein_feature_effect_rows": len(per_protein),
        "model_performance_rows": len(performance),
        "feature_ablation_rows": len(feature_ablation),
        "structural_water_ablation_rows": len(structural_water_ablation),
    }


def main() -> None:
    args = parse_args()
    if args.extract_only and args.aggregate_only:
        raise ValueError("--extract_only and --aggregate_only are mutually exclusive")
    if args.pairwise_structure_csv and args.pairwise_structure_dir:
        raise ValueError(
            "--pairwise_structure_csv and --pairwise_structure_dir are mutually exclusive"
        )
    if args.receiver_cache_source_dir and not args.aggregate_only:
        raise ValueError(
            "--receiver_cache_source_dir is intended for --aggregate_only runs"
        )
    if not 0.5 < args.receiver_quantile < 1:
        raise ValueError("--receiver_quantile must be between 0.5 and 1")
    if not 0 < args.low_receiver_quantile <= 0.5:
        raise ValueError("--low_receiver_quantile must be in (0, 0.5]")
    if args.minimum_inference_proteins < 2:
        raise ValueError("--minimum_inference_proteins must be at least 2")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_tsv, sep="\t")
    bands = pd.read_csv(args.bands_csv)
    summary = pd.read_csv(args.protein_summary_csv)
    residue = pd.read_csv(args.residue_annotations_csv)
    if args.conditions:
        manifest = manifest[manifest.condition.isin(args.conditions)]
        bands = bands[bands.condition.isin(args.conditions)]
        summary = summary[summary.condition.isin(args.conditions)]
    if args.splits:
        manifest = manifest[manifest.split.isin(args.splits)]
        bands = bands[bands.split.isin(args.splits)]
        summary = summary[summary.split.isin(args.splits)]
        residue = residue[residue.split.isin(args.splits)]
    manifest = manifest[manifest.seed.isin(args.seeds)].copy()
    bands = bands.sort_values(["condition", "split", "protein", "band_id"])
    required_residue = {
        "split", "protein", "residue_index_0based", "amino_acid", "q8",
        "neq", "rsa", "disorder", "torsion_change_from_previous",
    }
    missing = required_residue - set(residue)
    if missing:
        raise ValueError(f"Residue annotations lack {sorted(missing)}")
    bands_by_context = {
        key: group.sort_values("band_id").reset_index(drop=True)
        for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    extraction_reports = []
    if not args.aggregate_only:
        for row in manifest.itertuples(index=False):
            extraction_reports.append(extract_source(
                row, bands_by_context, output / "per_seed_query_profile_cache",
                args.overwrite_cache, args.max_proteins_per_file,
            ))
    aggregate_report = None
    if not args.extract_only:
        aggregate_report = aggregate(args, manifest, bands, summary, residue, output)
    parameters = {
        "receiver_definition": "sign_b * sum_j_in_band C_ij",
        "seed_handling": "compute within seed, then arithmetic mean across seeds",
        "receiver_quantile": args.receiver_quantile,
        "low_receiver_quantile": args.low_receiver_quantile,
        "matching": "same protein, band, exact query Q8, and sequence-distance bin",
        "distance_bins": dict(zip(DISTANCE_LABELS, ["0", "1-5", "6-20", "21-50", ">50"])),
        "long_range_min_separation": args.long_range_min_separation,
        "minimum_inference_proteins": args.minimum_inference_proteins,
        "seeds": sorted(set(args.seeds)),
        "pairwise_structure_source": (
            args.pairwise_structure_dir or args.pairwise_structure_csv
        ),
        "receiver_cache_source_dir": args.receiver_cache_source_dir,
        "model_cohorts": {
            "all_receivers": "original eligible-query cohort",
            "structure_eligible_fixed": (
                "mapping accepted, query resolved, and source band resolved "
                "fraction above threshold; fixed across structural stages"
            ),
            "water_eligible_fixed": (
                "structure-eligible plus primary water method/resolution "
                "criteria; fixed across all water-stage comparisons"
            ),
        },
        "receiver_scopes": [
            "all_eligible_queries",
            f"sequence_distal_ge{args.long_range_min_separation}",
            f"sequence_distal_ge{args.long_range_min_separation}_no_direct_ca",
        ],
        "structure_feature_stages": {
            "geometry": list(PAIR_GEOMETRY_FEATURES),
            "contact_network": list(PAIR_CONTACT_FEATURES),
            "community_domain": list(PAIR_ORGANIZATION_FEATURES),
            "putative_polar_contacts": list(PAIR_POLAR_FEATURES),
            "crystallographic_water_network": list(PAIR_WATER_FEATURES),
            "mechanism_interactions": list(MECHANISM_INTERACTION_QUERY_FEATURES),
        },
        "feature_ablation": {
            "features": [name for name, _ in QUERY_MODEL_FEATURE_GROUPS],
            "single_feature_estimand": (
                "AUROC of baseline plus one feature minus baseline AUROC"
            ),
            "leave_one_out_estimand": (
                "full query-feature AUROC minus AUROC after removing one feature"
            ),
        },
        "random_seed": args.random_seed,
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output / "extraction_audit.json").write_text(json.dumps(extraction_reports, indent=2) + "\n")
    print(json.dumps({
        "extraction": extraction_reports, "aggregate": aggregate_report,
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
