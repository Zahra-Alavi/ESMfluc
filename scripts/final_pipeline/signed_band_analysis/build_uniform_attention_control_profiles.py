#!/usr/bin/env python3
"""Build audited uniform-routing/evidence-only control profiles.

This module implements the detector-independent part of the uniform-attention
control.  It streams the existing signed-contribution JSON files, deliberately
skipping their LxL matrices, and writes only O(L) residue profiles.  It also
constructs aligned three-seed means and profile-level similarity summaries.

It does not call, tune, or lock an apex detector.  Apex comparison must remain a
separate downstream step after the redesigned Phase 1 detector is locked.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .extract_signed_contribution_bands import MarkerScanner, iter_profiles


CONTROL_SCHEMA = "esmfluc.uniform_attention_control.v1"
SEED_AVERAGED_SCHEMA = "esmfluc.uniform_attention_control.seed_average.v1"
EXPECTED_SEEDS = (1, 2, 3)
DEFAULT_G_UNIFORM_ATOL = 1e-6
DEFAULT_G_UNIFORM_RTOL = 1e-6
PROFILE_FIELDS = (
    "intrinsic_signed_evidence",
    "observed_signed_influence",
    "uniform_signed_influence",
    "attention_column_mean",
    "attention_amplification",
    "shifted_attention_column_mean",
    "shifted_attention_signed_influence",
)
SEED_AVERAGED_FIELDS = (
    "seed_averaged_intrinsic_signed_evidence",
    "seed_averaged_observed_signed_influence",
    "seed_averaged_uniform_signed_influence",
    "seed_averaged_attention_column_mean",
    "seed_averaged_attention_amplification",
    "seed_averaged_shifted_attention_column_mean",
    "seed_averaged_shifted_attention_signed_influence",
)
SIMILARITY_METRICS = (
    "pearson_observed_uniform",
    "spearman_observed_uniform",
    "pearson_observed_shifted_attention",
    "spearman_observed_shifted_attention",
    "same_sign_top_decile_overlap_shifted_attention",
    "pearson_complete_absolute_magnitude",
    "same_sign_top_decile_overlap",
    "top_5pct_jaccard",
    "top_10pct_jaccard",
    "mean_high_influence_rank_displacement",
    "median_high_influence_rank_displacement",
    "mean_high_influence_rank_displacement_normalized",
    "g_fraction_gt_1",
    "g_fraction_lt_1",
    "g_fraction_isclose_1",
    "g_minimum",
    "g_median",
    "g_mean",
    "g_maximum",
    "pearson_abs_evidence_g",
    "spearman_abs_evidence_g",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest_tsv",
        required=True,
        help="Contribution manifest with condition, seed, split, and json_gz.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Dedicated analysis_uniform_attention_control output directory.",
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--identity_atol", type=float, default=5e-6)
    parser.add_argument("--identity_rtol", type=float, default=1e-6)
    parser.add_argument("--normalization_atol", type=float, default=5e-6)
    parser.add_argument("--normalization_rtol", type=float, default=1e-6)
    parser.add_argument(
        "--g_uniform_atol", type=float, default=DEFAULT_G_UNIFORM_ATOL
    )
    parser.add_argument(
        "--g_uniform_rtol", type=float, default=DEFAULT_G_UNIFORM_RTOL
    )
    parser.add_argument(
        "--max_proteins_per_file",
        type=int,
        default=0,
        help="Debugging only; zero processes every protein.",
    )
    parser.add_argument(
        "--shift_random_seed",
        type=int,
        default=20260807,
        help=(
            "Seed for a reproducible nonzero circular shift of B_j within each "
            "protein and condition. The same offset is used for all model seeds "
            "so the control does not artificially erase cross-seed stability."
        ),
    )
    return parser.parse_args()


def deterministic_shift_offset(
    *, condition: str, seed: int, split: str, protein: str, length: int,
    random_seed: int,
) -> int:
    """Choose a reproducible nonzero offset shared by all model seeds."""
    if length <= 1:
        return 0
    # Intentionally omit ``seed``. Giving each model seed a different rotation
    # would manufacture instability in the shifted-B control.
    del seed
    token = f"{random_seed}|{condition}|{split}|{protein}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return 1 + value % (length - 1)


def _safe_token(value: object, label: str) -> str:
    token = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", token):
        raise ValueError(f"Unsafe {label} token {token!r}")
    return token


def _atomic_text_path(destination: Path) -> tuple[Path, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    return Path(handle.name), handle


def _write_json_atomic(destination: Path, payload: dict) -> None:
    temporary, handle = _atomic_text_path(destination)
    try:
        with handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_frame_atomic(
    destination: Path, frame: pd.DataFrame, *, sep: str = ",", compression=None
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            sep=sep,
            compression=compression,
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class StreamingControlWriter:
    """Atomically stream one collection of O(L) profiles to JSON.gz."""

    def __init__(
        self,
        destination: Path,
        *,
        schema_version: str,
        metadata: dict,
        fields: tuple[str, ...],
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp.gz",
            dir=destination.parent,
        )
        os.close(descriptor)
        self.destination = destination
        self.temporary = Path(name)
        self.handle = gzip.open(self.temporary, "wt", encoding="utf-8")
        self.fields = fields
        header = {
            "schema_version": schema_version,
            **metadata,
        }
        prefix = json.dumps(header, separators=(",", ":"), allow_nan=False)
        self.handle.write(prefix[:-1])
        self.handle.write(',"proteins":[')
        self.first = True
        self.closed = False

    def write(self, profile: dict) -> None:
        payload = {
            "name": profile["name"],
            "sequence": profile["sequence"],
            "length": int(profile["length"]),
        }
        for field in self.fields:
            values = np.asarray(profile[field], dtype=np.float64)
            payload[field] = values.tolist()
        if not self.first:
            self.handle.write(",")
        json.dump(payload, self.handle, separators=(",", ":"), allow_nan=False)
        self.first = False

    def close(self, *, commit: bool) -> None:
        if self.closed:
            return
        try:
            if commit:
                self.handle.write("]}")
            self.handle.close()
            if commit:
                os.replace(self.temporary, self.destination)
            else:
                self.temporary.unlink(missing_ok=True)
        finally:
            self.closed = True

    def __enter__(self) -> "StreamingControlWriter":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        self.close(commit=exc_type is None)


def iter_control_profiles(
    path: Path,
    fields: tuple[str, ...] = PROFILE_FIELDS,
    expected_schema: str = CONTROL_SCHEMA,
) -> tuple[dict, Iterator[dict]]:
    """Stream a generated compact profile file without materializing it."""
    scanner = MarkerScanner(path)
    prefix = scanner.take_until(b'"proteins":[', capture=True)
    if prefix is None:
        scanner.close()
        raise ValueError(f"{path}: missing proteins array")

    def scalar(key: str):
        marker = json.dumps(key, separators=(",", ":")).encode() + b":"
        index = prefix.find(marker)
        if index < 0:
            return None
        text = prefix[index + len(marker):].decode("utf-8")
        return json.JSONDecoder().raw_decode(text)[0]

    metadata = {
        key: scalar(key)
        for key in (
            "schema_version",
            "condition",
            "seed",
            "split",
            "protein_count",
            "residue_count",
            "source_seeds",
        )
    }
    if metadata["schema_version"] != expected_schema:
        scanner.close()
        raise ValueError(
            f"{path}: expected schema {expected_schema!r}, "
            f"found {metadata['schema_version']!r}"
        )

    def generator() -> Iterator[dict]:
        count = 0
        residues = 0
        try:
            while scanner.take_until(b'"name":') is not None:
                name = scanner.read_json_string()
                if scanner.take_until(b'"sequence":') is None:
                    raise ValueError(f"{path}: missing sequence for {name}")
                sequence = scanner.read_json_string()
                if scanner.take_until(b'"length":') is None:
                    raise ValueError(f"{path}: missing length for {name}")
                length = scanner.read_integer()
                arrays = {}
                for field in fields:
                    marker = json.dumps(field).encode() + b":"
                    if scanner.take_until(marker) is None:
                        raise ValueError(f"{path}: missing {field} for {name}")
                    arrays[field] = scanner.read_flat_float_array()
                if length != len(sequence):
                    raise ValueError(
                        f"{path}: {name} sequence length {len(sequence)} != {length}"
                    )
                for field, values in arrays.items():
                    if values.shape != (length,):
                        raise ValueError(
                            f"{path}: {name} {field} shape {values.shape} != {(length,)}"
                        )
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"{path}: {name} {field} has nonfinite values")
                count += 1
                residues += length
                yield {
                    "name": name,
                    "sequence": sequence,
                    "length": length,
                    **arrays,
                }
            expected_count = metadata["protein_count"]
            expected_residues = metadata["residue_count"]
            if expected_count is not None and count != int(expected_count):
                raise ValueError(
                    f"{path}: streamed {count} proteins, expected {expected_count}"
                )
            if expected_residues is not None and residues != int(expected_residues):
                raise ValueError(
                    f"{path}: streamed {residues} residues, expected {expected_residues}"
                )
        finally:
            scanner.close()

    return metadata, generator()


def load_manifest(
    path: Path,
    conditions: list[str] | None = None,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"condition", "seed", "split", "json_gz", "protein_count", "residue_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    if conditions:
        frame = frame[frame["condition"].astype(str).isin(conditions)]
    if splits:
        frame = frame[frame["split"].astype(str).isin(splits)]
    if frame.empty:
        raise ValueError("No contribution files selected")
    frame = frame.copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    duplicate = frame.duplicated(["condition", "seed", "split"], keep=False)
    if duplicate.any():
        rows = frame.loc[duplicate, ["condition", "seed", "split"]]
        raise ValueError(f"Duplicate manifest keys:\n{rows.to_string(index=False)}")
    for (condition, split), group in frame.groupby(["condition", "split"], sort=False):
        seeds = tuple(sorted(group["seed"].tolist()))
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{condition}/{split}: expected seeds {EXPECTED_SEEDS}, found {seeds}"
            )
    missing_paths = [
        value for value in frame["json_gz"].astype(str) if not Path(value).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing contribution files: " + ", ".join(missing_paths[:10])
        )
    return frame.sort_values(["condition", "seed", "split"]).reset_index(drop=True)


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def build_control_profile(
    profile: dict,
    *,
    shift_offset: int = 0,
    identity_atol: float,
    identity_rtol: float,
    normalization_atol: float,
    normalization_rtol: float,
) -> tuple[dict, dict]:
    length = int(profile["length"])
    evidence = np.asarray(profile["intrinsic_signed_evidence"], dtype=np.float64)
    observed = np.asarray(profile["signed_column_influence"], dtype=np.float64)
    attention = np.asarray(profile["attention_column_mean"], dtype=np.float64)
    arrays = (evidence, observed, attention)
    if any(values.shape != (length,) for values in arrays):
        raise ValueError(f"{profile['name']}: an input profile has the wrong length")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{profile['name']}: nonfinite input profile values")
    if length <= 0:
        raise ValueError(f"{profile['name']}: protein length must be positive")
    if np.any(attention < 0):
        raise ValueError(f"{profile['name']}: attention_column_mean is negative")

    uniform = evidence / length
    amplification = attention * length
    if not 0 <= shift_offset < length:
        raise ValueError(
            f"{profile['name']}: shift offset {shift_offset} outside [0, {length})"
        )
    shifted_attention = np.roll(attention, shift_offset)
    shifted = evidence * shifted_attention
    reconstructed_from_components = evidence * attention
    reconstructed_from_control = uniform * amplification
    component_error = _max_abs(observed - reconstructed_from_components)
    control_error = _max_abs(observed - reconstructed_from_control)
    attention_sum_error = abs(float(np.sum(attention)) - 1.0)
    amplification_mean_error = abs(float(np.mean(amplification)) - 1.0)
    sign_mask = evidence != 0
    sign_disagreements = int(np.sum(
        sign_mask & (
            (np.sign(observed) != np.sign(evidence))
            | (np.sign(uniform) != np.sign(evidence))
        )
    ))

    if not np.allclose(
        observed,
        reconstructed_from_components,
        atol=identity_atol,
        rtol=identity_rtol,
    ):
        raise ValueError(
            f"{profile['name']}: I_observed != s*B; max abs error={component_error}"
        )
    if not np.allclose(
        observed,
        reconstructed_from_control,
        atol=identity_atol,
        rtol=identity_rtol,
    ):
        raise ValueError(
            f"{profile['name']}: I_observed != I_uniform*G; "
            f"max abs error={control_error}"
        )
    if not math.isclose(
        float(np.sum(attention)),
        1.0,
        abs_tol=normalization_atol,
        rel_tol=normalization_rtol,
    ):
        raise ValueError(
            f"{profile['name']}: sum(B) != 1; error={attention_sum_error}"
        )
    if not math.isclose(
        float(np.mean(amplification)),
        1.0,
        abs_tol=normalization_atol,
        rel_tol=normalization_rtol,
    ):
        raise ValueError(
            f"{profile['name']}: mean(G) != 1; "
            f"error={amplification_mean_error}"
        )
    if sign_disagreements:
        raise ValueError(
            f"{profile['name']}: {sign_disagreements} unexpected sign disagreements"
        )

    result = {
        "name": profile["name"],
        "sequence": profile["sequence"],
        "length": length,
        "intrinsic_signed_evidence": evidence,
        "observed_signed_influence": observed,
        "uniform_signed_influence": uniform,
        "attention_column_mean": attention,
        "attention_amplification": amplification,
        "shifted_attention_column_mean": shifted_attention,
        "shifted_attention_signed_influence": shifted,
    }
    audit = {
        "protein": profile["name"],
        "length": length,
        "max_observed_minus_s_b_abs_error": component_error,
        "max_observed_minus_uniform_g_abs_error": control_error,
        "attention_sum_abs_error": attention_sum_error,
        "amplification_mean_abs_error": amplification_mean_error,
        "exact_zero_evidence_residues": int(np.sum(evidence == 0)),
        "unexpected_sign_disagreements": sign_disagreements,
        "positive_evidence_residues": int(np.sum(evidence > 0)),
        "negative_evidence_residues": int(np.sum(evidence < 0)),
        "shift_offset": int(shift_offset),
        "shifted_attention_sum_abs_error": abs(
            float(np.sum(shifted_attention)) - 1.0
        ),
        "max_shifted_minus_s_shifted_b_abs_error": _max_abs(
            shifted - evidence * shifted_attention
        ),
    }
    return result, audit


def _safe_correlation(a: np.ndarray, b: np.ndarray, *, ranked: bool = False) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if len(a) < 2:
        return math.nan
    if ranked:
        a = rankdata(a, method="average")
        b = rankdata(b, method="average")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _top_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=int)
    k = max(1, int(math.ceil(fraction * count)))
    order = np.lexsort((np.arange(count), -np.asarray(values, dtype=np.float64)))
    return np.sort(order[:k])


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else math.nan


def profile_similarity_rows(
    control: dict,
    *,
    condition: str,
    seed: int | str,
    split: str,
    g_uniform_atol: float = DEFAULT_G_UNIFORM_ATOL,
    g_uniform_rtol: float = DEFAULT_G_UNIFORM_RTOL,
) -> list[dict]:
    evidence = np.asarray(control["intrinsic_signed_evidence"])
    observed = np.asarray(control["observed_signed_influence"])
    uniform = np.asarray(control["uniform_signed_influence"])
    shifted = np.asarray(
        control.get("shifted_attention_signed_influence", uniform)
    )
    amplification = np.asarray(control["attention_amplification"])
    complete_absolute_correlation = _safe_correlation(
        np.abs(observed), np.abs(uniform)
    )
    rows = []
    for sign in (1, -1):
        indices = np.flatnonzero(sign * evidence > 0)
        obs = sign * observed[indices]
        uni = sign * uniform[indices]
        shifted_values = sign * shifted[indices]
        g = amplification[indices]
        abs_evidence = np.abs(evidence[indices])
        top_obs_10 = _top_indices(obs, 0.10)
        top_uni_10 = _top_indices(uni, 0.10)
        top_shifted_10 = _top_indices(shifted_values, 0.10)
        top_obs_05 = _top_indices(obs, 0.05)
        top_uni_05 = _top_indices(uni, 0.05)
        high_union = np.union1d(top_obs_10, top_uni_10)
        if len(indices):
            obs_rank = rankdata(-obs, method="average")
            uni_rank = rankdata(-uni, method="average")
            displacements = np.abs(obs_rank[high_union] - uni_rank[high_union])
        else:
            displacements = np.empty(0)
        normalizer = max(1, len(indices) - 1)
        overlap_denominator = min(len(top_obs_10), len(top_uni_10))
        g_is_uniform = np.isclose(
            g,
            1.0,
            atol=g_uniform_atol,
            rtol=g_uniform_rtol,
        )
        rows.append({
            "condition": condition,
            "seed": seed,
            "split": split,
            "protein": control["name"],
            "protein_length": control["length"],
            "sign": sign,
            "label": "flexibility_supporting" if sign > 0 else "rigidity_supporting",
            "n_same_sign_residues": len(indices),
            "pearson_observed_uniform": _safe_correlation(obs, uni),
            "spearman_observed_uniform": _safe_correlation(obs, uni, ranked=True),
            "pearson_observed_shifted_attention": _safe_correlation(
                obs, shifted_values
            ),
            "spearman_observed_shifted_attention": _safe_correlation(
                obs, shifted_values, ranked=True
            ),
            "same_sign_top_decile_overlap_shifted_attention": (
                len(set(top_obs_10) & set(top_shifted_10))
                / min(len(top_obs_10), len(top_shifted_10))
                if min(len(top_obs_10), len(top_shifted_10))
                else math.nan
            ),
            "pearson_complete_absolute_magnitude": (
                complete_absolute_correlation
            ),
            "same_sign_top_decile_overlap": (
                len(set(top_obs_10) & set(top_uni_10)) / overlap_denominator
                if overlap_denominator
                else math.nan
            ),
            "top_5pct_jaccard": _jaccard(top_obs_05, top_uni_05),
            "top_10pct_jaccard": _jaccard(top_obs_10, top_uni_10),
            "mean_high_influence_rank_displacement": (
                float(np.mean(displacements)) if len(displacements) else math.nan
            ),
            "median_high_influence_rank_displacement": (
                float(np.median(displacements)) if len(displacements) else math.nan
            ),
            "mean_high_influence_rank_displacement_normalized": (
                float(np.mean(displacements) / normalizer)
                if len(displacements)
                else math.nan
            ),
            "g_fraction_gt_1": (
                float(np.mean((g > 1) & ~g_is_uniform))
                if len(g) else math.nan
            ),
            "g_fraction_lt_1": (
                float(np.mean((g < 1) & ~g_is_uniform))
                if len(g) else math.nan
            ),
            "g_fraction_isclose_1": (
                float(np.mean(g_is_uniform)) if len(g) else math.nan
            ),
            "g_minimum": float(np.min(g)) if len(g) else math.nan,
            "g_median": float(np.median(g)) if len(g) else math.nan,
            "g_mean": float(np.mean(g)) if len(g) else math.nan,
            "g_maximum": float(np.max(g)) if len(g) else math.nan,
            "pearson_abs_evidence_g": _safe_correlation(abs_evidence, g),
            "spearman_abs_evidence_g": _safe_correlation(
                abs_evidence, g, ranked=True
            ),
        })
    return rows


class SimilarityAccumulator:
    """Metric-wise online means; each protein contributes one row per sign."""

    def __init__(self):
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(int))
        self.protein_rows = defaultdict(int)

    def add(self, row: dict) -> None:
        key = (row["condition"], row["seed"], row["split"], row["sign"])
        self.protein_rows[key] += 1
        for metric in SIMILARITY_METRICS:
            value = float(row[metric])
            if np.isfinite(value):
                self.sums[key][metric] += value
                self.counts[key][metric] += 1

    def rows(self) -> list[dict]:
        output = []
        for key in sorted(self.protein_rows):
            condition, seed, split, sign = key
            row = {
                "condition": condition,
                "seed": seed,
                "split": split,
                "sign": sign,
                "label": (
                    "flexibility_supporting" if sign > 0 else "rigidity_supporting"
                ),
                "n_protein_rows": self.protein_rows[key],
                "weighting": "equal protein weighting",
            }
            for metric in SIMILARITY_METRICS:
                count = self.counts[key][metric]
                row[f"mean_{metric}"] = (
                    self.sums[key][metric] / count if count else math.nan
                )
                row[f"n_finite_{metric}"] = count
            output.append(row)
        return output


class AtomicGzipCsvWriter:
    def __init__(self, destination: Path, fieldnames: list[str]):
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp.gz",
            dir=destination.parent,
        )
        os.close(descriptor)
        self.destination = destination
        self.temporary = Path(name)
        self.handle = gzip.open(self.temporary, "wt", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.closed = False

    def write(self, row: dict) -> None:
        self.writer.writerow(row)

    def close(self, *, commit: bool) -> None:
        if self.closed:
            return
        try:
            self.handle.close()
            if commit:
                os.replace(self.temporary, self.destination)
            else:
                self.temporary.unlink(missing_ok=True)
        finally:
            self.closed = True


def _manifest_count(row, maximum: int) -> int:
    expected = int(row.protein_count)
    return min(expected, maximum) if maximum else expected


def generate_per_seed_profiles(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    identity_atol: float,
    identity_rtol: float,
    normalization_atol: float,
    normalization_rtol: float,
    g_uniform_atol: float = DEFAULT_G_UNIFORM_ATOL,
    g_uniform_rtol: float = DEFAULT_G_UNIFORM_RTOL,
    max_proteins_per_file: int = 0,
    shift_random_seed: int = 20260807,
) -> tuple[pd.DataFrame, dict]:
    profile_dir = output_dir / "profiles"
    comparison_dir = output_dir / "profile_comparison"
    comparison_path = comparison_dir / "per_protein_profile_similarity.csv.gz"
    similarity_fields = [
        "condition", "seed", "split", "protein", "protein_length", "sign",
        "label", "n_same_sign_residues", *SIMILARITY_METRICS,
    ]
    similarity_writer = AtomicGzipCsvWriter(comparison_path, similarity_fields)
    accumulator = SimilarityAccumulator()
    manifest_rows = []
    aggregate_audit = {
        "schema_version": CONTROL_SCHEMA,
        "analysis_type": (
            "evidence-only control / uniform-routing counterfactual / "
            "final-attention-layer decomposition control"
        ),
        "interpretive_limit": (
            "This is not a retrained model or causal attention ablation. "
            "The contextual evidence was learned jointly with attention."
        ),
        "tolerances": {
            "identity_atol": identity_atol,
            "identity_rtol": identity_rtol,
            "normalization_atol": normalization_atol,
            "normalization_rtol": normalization_rtol,
            "g_uniform_isclose_atol": g_uniform_atol,
            "g_uniform_isclose_rtol": g_uniform_rtol,
        },
        "protein_profiles": 0,
        "residues": 0,
        "exact_zero_evidence_residues": 0,
        "unexpected_sign_disagreements": 0,
        "missing_or_invalid_profiles": 0,
        "max_observed_minus_s_b_abs_error": 0.0,
        "max_observed_minus_uniform_g_abs_error": 0.0,
        "max_attention_sum_abs_error": 0.0,
        "max_amplification_mean_abs_error": 0.0,
        "max_shifted_attention_sum_abs_error": 0.0,
        "max_shifted_identity_abs_error": 0.0,
        "nonzero_shift_profiles": 0,
        "counts_by_condition_seed_split": [],
        "selection": {
            "selected_conditions": sorted(
                manifest["condition"].astype(str).unique().tolist()
            ),
            "selected_splits": sorted(
                manifest["split"].astype(str).unique().tolist()
            ),
            "selected_manifest_files": int(len(manifest)),
            "source_expected_protein_profiles": int(
                manifest["protein_count"].sum()
            ),
            "source_expected_residues": int(manifest["residue_count"].sum()),
            "max_proteins_per_file": int(max_proteins_per_file),
            "debug_subset": bool(max_proteins_per_file),
            "run_scope": (
                "debug_subset" if max_proteins_per_file else "full_selected_manifest"
            ),
        },
    }
    try:
        for row in manifest.itertuples(index=False):
            condition = _safe_token(row.condition, "condition")
            split = _safe_token(row.split, "split")
            seed = int(row.seed)
            source = Path(row.json_gz).expanduser().resolve()
            expected_count = _manifest_count(row, max_proteins_per_file)
            destination = (
                profile_dir
                / condition
                / f"seed_{seed}"
                / f"{split}_uniform_attention_profiles.json.gz"
            )
            metadata, profiles = iter_profiles(source)
            if str(metadata["condition"]) != condition:
                raise ValueError(
                    f"{source}: condition {metadata['condition']!r} != {condition!r}"
                )
            if int(metadata["seed"]) != seed or str(metadata["split"]) != split:
                raise ValueError(
                    f"{source}: metadata seed/split do not match the manifest"
                )
            file_count = 0
            file_residues = 0
            file_positive = 0
            file_negative = 0
            with StreamingControlWriter(
                destination,
                schema_version=CONTROL_SCHEMA,
                metadata={
                    "condition": condition,
                    "seed": seed,
                    "split": split,
                    "source_contribution_json": str(source),
                    "protein_count": expected_count,
                    "source_protein_count": int(row.protein_count),
                    "source_residue_count": int(row.residue_count),
                    "max_proteins_per_file": int(max_proteins_per_file),
                    "debug_subset": bool(max_proteins_per_file),
                    "residue_count": (
                        int(row.residue_count)
                        if not max_proteins_per_file
                        else None
                    ),
                },
                fields=PROFILE_FIELDS,
            ) as writer:
                try:
                    for profile in profiles:
                        shift_offset = deterministic_shift_offset(
                            condition=condition,
                            seed=seed,
                            split=split,
                            protein=str(profile["name"]),
                            length=int(profile["length"]),
                            random_seed=shift_random_seed,
                        )
                        control, audit = build_control_profile(
                            profile,
                            shift_offset=shift_offset,
                            identity_atol=identity_atol,
                            identity_rtol=identity_rtol,
                            normalization_atol=normalization_atol,
                            normalization_rtol=normalization_rtol,
                        )
                        writer.write(control)
                        for similarity in profile_similarity_rows(
                            control,
                            condition=condition,
                            seed=seed,
                            split=split,
                            g_uniform_atol=g_uniform_atol,
                            g_uniform_rtol=g_uniform_rtol,
                        ):
                            similarity_writer.write(similarity)
                            accumulator.add(similarity)
                        file_count += 1
                        file_residues += audit["length"]
                        file_positive += audit["positive_evidence_residues"]
                        file_negative += audit["negative_evidence_residues"]
                        aggregate_audit["protein_profiles"] += 1
                        aggregate_audit["residues"] += audit["length"]
                        aggregate_audit["exact_zero_evidence_residues"] += audit[
                            "exact_zero_evidence_residues"
                        ]
                        aggregate_audit["unexpected_sign_disagreements"] += audit[
                            "unexpected_sign_disagreements"
                        ]
                        aggregate_audit["nonzero_shift_profiles"] += int(
                            audit["shift_offset"] != 0
                        )
                        for source_key, destination_key in (
                            (
                                "max_observed_minus_s_b_abs_error",
                                "max_observed_minus_s_b_abs_error",
                            ),
                            (
                                "max_observed_minus_uniform_g_abs_error",
                                "max_observed_minus_uniform_g_abs_error",
                            ),
                            (
                                "attention_sum_abs_error",
                                "max_attention_sum_abs_error",
                            ),
                            (
                                "amplification_mean_abs_error",
                                "max_amplification_mean_abs_error",
                            ),
                            (
                                "shifted_attention_sum_abs_error",
                                "max_shifted_attention_sum_abs_error",
                            ),
                            (
                                "max_shifted_minus_s_shifted_b_abs_error",
                                "max_shifted_identity_abs_error",
                            ),
                        ):
                            aggregate_audit[destination_key] = max(
                                aggregate_audit[destination_key],
                                audit[source_key],
                            )
                        if (
                            max_proteins_per_file
                            and file_count >= max_proteins_per_file
                        ):
                            break
                finally:
                    profiles.close()
                if file_count != expected_count:
                    raise ValueError(
                        f"{source}: wrote {file_count} proteins, "
                        f"expected {expected_count}"
                    )
                if (
                    not max_proteins_per_file
                    and file_residues != int(row.residue_count)
                ):
                    raise ValueError(
                        f"{source}: wrote {file_residues} residues, "
                        f"expected {row.residue_count}"
                    )
            file_record = {
                "condition": condition,
                "seed": seed,
                "split": split,
                "profile_json_gz": str(destination),
                "source_contribution_json_gz": str(source),
                "protein_count": file_count,
                "residue_count": file_residues,
                "source_protein_count": int(row.protein_count),
                "source_residue_count": int(row.residue_count),
                "max_proteins_per_file": int(max_proteins_per_file),
                "debug_subset": bool(max_proteins_per_file),
                "run_scope": (
                    "debug_subset"
                    if max_proteins_per_file
                    else "full_selected_manifest"
                ),
                "positive_evidence_residues": file_positive,
                "negative_evidence_residues": file_negative,
            }
            manifest_rows.append(file_record)
            aggregate_audit["counts_by_condition_seed_split"].append(file_record)
        similarity_writer.close(commit=True)
    except BaseException:
        similarity_writer.close(commit=False)
        raise

    profile_manifest = pd.DataFrame(manifest_rows)
    _write_frame_atomic(
        profile_dir / "profile_manifest.tsv", profile_manifest, sep="\t"
    )
    _write_frame_atomic(
        comparison_dir / "profile_similarity_summary.csv",
        pd.DataFrame(accumulator.rows()),
    )
    aggregate_audit["all_invariants_pass"] = True
    aggregate_audit["selection"]["processed_protein_profiles"] = int(
        aggregate_audit["protein_profiles"]
    )
    aggregate_audit["selection"]["processed_residues"] = int(
        aggregate_audit["residues"]
    )
    aggregate_audit["per_protein_similarity"] = str(comparison_path)
    _write_json_atomic(
        output_dir / "audits" / "profile_generation_audit.json",
        aggregate_audit,
    )
    return profile_manifest, aggregate_audit


def _validate_aligned_profiles(profiles: tuple[dict, ...], context: str) -> None:
    first = profiles[0]
    for other in profiles[1:]:
        for field in ("name", "sequence", "length"):
            if other[field] != first[field]:
                raise ValueError(
                    f"{context}: seed profiles differ in {field}: "
                    f"{first[field]!r} != {other[field]!r}"
                )


def build_seed_average(profiles: tuple[dict, ...]) -> dict:
    _validate_aligned_profiles(profiles, profiles[0]["name"])
    length = profiles[0]["length"]
    result = {
        "name": profiles[0]["name"],
        "sequence": profiles[0]["sequence"],
        "length": length,
    }
    mapping = {
        "seed_averaged_intrinsic_signed_evidence": "intrinsic_signed_evidence",
        "seed_averaged_observed_signed_influence": "observed_signed_influence",
        "seed_averaged_uniform_signed_influence": "uniform_signed_influence",
        "seed_averaged_attention_column_mean": "attention_column_mean",
        "seed_averaged_attention_amplification": "attention_amplification",
        "seed_averaged_shifted_attention_column_mean": (
            "shifted_attention_column_mean"
        ),
        "seed_averaged_shifted_attention_signed_influence": (
            "shifted_attention_signed_influence"
        ),
    }
    for output_field, input_field in mapping.items():
        result[output_field] = np.mean(
            np.stack([profile[input_field] for profile in profiles], axis=0),
            axis=0,
        )
    if not np.allclose(
        result["seed_averaged_uniform_signed_influence"],
        result["seed_averaged_intrinsic_signed_evidence"] / length,
        atol=1e-15,
        rtol=1e-12,
    ):
        raise ValueError(
            f"{profiles[0]['name']}: seed-mean uniform profile identity failed"
        )
    return result


def seed_average_as_control(averaged: dict) -> dict:
    """Expose a seed-mean record through the similarity-profile interface."""
    return {
        "name": averaged["name"],
        "sequence": averaged["sequence"],
        "length": averaged["length"],
        "intrinsic_signed_evidence": averaged[
            "seed_averaged_intrinsic_signed_evidence"
        ],
        "observed_signed_influence": averaged[
            "seed_averaged_observed_signed_influence"
        ],
        "uniform_signed_influence": averaged[
            "seed_averaged_uniform_signed_influence"
        ],
        "attention_column_mean": averaged[
            "seed_averaged_attention_column_mean"
        ],
        "attention_amplification": averaged[
            "seed_averaged_attention_amplification"
        ],
        "shifted_attention_column_mean": averaged[
            "seed_averaged_shifted_attention_column_mean"
        ],
        "shifted_attention_signed_influence": averaged[
            "seed_averaged_shifted_attention_signed_influence"
        ],
    }


def audit_seed_average_profile(
    averaged: dict,
    *,
    normalization_atol: float,
    normalization_rtol: float,
) -> dict:
    """Validate normalization and algebra specific to the seed-mean profiles."""
    length = int(averaged["length"])
    evidence = np.asarray(
        averaged["seed_averaged_intrinsic_signed_evidence"], dtype=np.float64
    )
    uniform = np.asarray(
        averaged["seed_averaged_uniform_signed_influence"], dtype=np.float64
    )
    attention = np.asarray(
        averaged["seed_averaged_attention_column_mean"], dtype=np.float64
    )
    amplification = np.asarray(
        averaged["seed_averaged_attention_amplification"], dtype=np.float64
    )
    shifted_attention = np.asarray(
        averaged["seed_averaged_shifted_attention_column_mean"], dtype=np.float64
    )
    shifted = np.asarray(
        averaged["seed_averaged_shifted_attention_signed_influence"], dtype=np.float64
    )
    arrays = (evidence, uniform, attention, amplification, shifted_attention, shifted)
    if length <= 0 or any(values.shape != (length,) for values in arrays):
        raise ValueError(f"{averaged['name']}: invalid seed-mean profile lengths")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{averaged['name']}: nonfinite seed-mean profile values")
    if np.any(attention < 0):
        raise ValueError(f"{averaged['name']}: negative seed-mean attention")
    uniform_error = _max_abs(uniform - evidence / length)
    attention_sum_error = abs(float(np.sum(attention)) - 1.0)
    amplification_mean_error = abs(float(np.mean(amplification)) - 1.0)
    amplification_identity_error = _max_abs(
        amplification - length * attention
    )
    shifted_attention_sum_error = abs(float(np.sum(shifted_attention)) - 1.0)
    if not math.isclose(
        float(np.sum(attention)),
        1.0,
        abs_tol=normalization_atol,
        rel_tol=normalization_rtol,
    ):
        raise ValueError(
            f"{averaged['name']}: sum(mean_seed(B_j)) != 1; "
            f"error={attention_sum_error}"
        )
    if not math.isclose(
        float(np.mean(amplification)),
        1.0,
        abs_tol=normalization_atol,
        rel_tol=normalization_rtol,
    ):
        raise ValueError(
            f"{averaged['name']}: mean_j(mean_seed(G_j)) != 1; "
            f"error={amplification_mean_error}"
        )
    if not np.allclose(
        amplification,
        length * attention,
        atol=normalization_atol,
        rtol=normalization_rtol,
    ):
        raise ValueError(
            f"{averaged['name']}: mean_seed(G_j) != "
            f"L*mean_seed(B_j); max error={amplification_identity_error}"
        )
    if not math.isclose(
        float(np.sum(shifted_attention)),
        1.0,
        abs_tol=normalization_atol,
        rel_tol=normalization_rtol,
    ):
        raise ValueError(
            f"{averaged['name']}: sum(mean_seed(shift(B_j))) != 1; "
            f"error={shifted_attention_sum_error}"
        )
    return {
        "mean_uniform_identity_abs_error": uniform_error,
        "mean_attention_sum_abs_error": attention_sum_error,
        "mean_amplification_average_abs_error": amplification_mean_error,
        "mean_amplification_identity_abs_error": amplification_identity_error,
        "mean_shifted_attention_sum_abs_error": shifted_attention_sum_error,
    }


def generate_seed_averaged_profiles(
    profile_manifest: pd.DataFrame,
    output_dir: Path,
    *,
    normalization_atol: float = 5e-6,
    normalization_rtol: float = 1e-6,
    g_uniform_atol: float = DEFAULT_G_UNIFORM_ATOL,
    g_uniform_rtol: float = DEFAULT_G_UNIFORM_RTOL,
) -> tuple[pd.DataFrame, dict]:
    required = {
        "condition", "seed", "split", "profile_json_gz",
        "protein_count", "residue_count",
    }
    missing = required - set(profile_manifest.columns)
    if missing:
        raise ValueError(
            "Compact profile manifest lacks required columns: "
            f"{sorted(missing)}"
        )
    if profile_manifest.empty:
        raise ValueError("Compact profile manifest is empty")
    averaged_dir = output_dir / "seed_averaged_profiles"
    comparison_dir = output_dir / "profile_comparison"
    comparison_path = (
        comparison_dir / "per_protein_seed_averaged_profile_similarity.csv.gz"
    )
    similarity_fields = [
        "condition", "seed", "split", "protein", "protein_length", "sign",
        "label", "n_same_sign_residues", *SIMILARITY_METRICS,
    ]
    similarity_writer = AtomicGzipCsvWriter(comparison_path, similarity_fields)
    accumulator = SimilarityAccumulator()
    manifest_rows = []
    audit = {
        "schema_version": SEED_AVERAGED_SCHEMA,
        "source_seeds": list(EXPECTED_SEEDS),
        "formulae": {
            "mean_observed": "mean_seed(s_j * B_j)",
            "mean_uniform": "mean_seed(s_j / L) = mean_seed(s_j) / L",
            "mean_amplification": "mean_seed(L * B_j)",
            "prohibited_factorization": (
                "mean_seed(s_j * B_j) must not be replaced by "
                "mean_seed(s_j) * mean_seed(B_j)"
            ),
        },
        "protein_profiles": 0,
        "residues": 0,
        "alignment_failures": 0,
        "max_mean_uniform_identity_abs_error": 0.0,
        "max_mean_attention_sum_abs_error": 0.0,
        "max_mean_amplification_average_abs_error": 0.0,
        "max_mean_amplification_identity_abs_error": 0.0,
        "max_mean_shifted_attention_sum_abs_error": 0.0,
        "tolerances": {
            "normalization_atol": normalization_atol,
            "normalization_rtol": normalization_rtol,
            "g_uniform_isclose_atol": g_uniform_atol,
            "g_uniform_isclose_rtol": g_uniform_rtol,
        },
        "counts_by_condition_split": [],
    }
    try:
        for (condition, split), group in profile_manifest.groupby(
            ["condition", "split"], sort=True
        ):
            if len(group) != len(EXPECTED_SEEDS) or group["seed"].duplicated().any():
                raise ValueError(
                    f"{condition}/{split}: compact profile manifest must contain "
                    "exactly one row for each seed"
                )
            by_seed = {int(row.seed): row for row in group.itertuples(index=False)}
            if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
                raise ValueError(
                    f"{condition}/{split}: compact profiles do not contain seeds "
                    f"{EXPECTED_SEEDS}"
                )
            iterators = []
            for seed in EXPECTED_SEEDS:
                metadata, iterator = iter_control_profiles(
                    Path(by_seed[seed].profile_json_gz)
                )
                if (
                    str(metadata["condition"]) != str(condition)
                    or int(metadata["seed"]) != seed
                    or str(metadata["split"]) != str(split)
                ):
                    raise ValueError(
                        f"{by_seed[seed].profile_json_gz}: compact metadata mismatch"
                    )
                iterators.append(iterator)
            destination = (
                averaged_dir
                / _safe_token(condition, "condition")
                / f"{_safe_token(split, 'split')}_seed_averaged_profiles.json.gz"
            )
            expected_count = int(group.iloc[0]["protein_count"])
            expected_residues = int(group.iloc[0]["residue_count"])
            if not (group["protein_count"] == expected_count).all():
                raise ValueError(f"{condition}/{split}: seed protein counts differ")
            if not (group["residue_count"] == expected_residues).all():
                raise ValueError(f"{condition}/{split}: seed residue counts differ")
            debug_subset = bool(group.get("debug_subset", pd.Series([False])).any())
            max_proteins = (
                int(group["max_proteins_per_file"].max())
                if "max_proteins_per_file" in group
                else 0
            )
            count = 0
            residues = 0
            with StreamingControlWriter(
                destination,
                schema_version=SEED_AVERAGED_SCHEMA,
                metadata={
                    "condition": condition,
                    "seed": "mean_1_2_3",
                    "split": split,
                    "source_seeds": list(EXPECTED_SEEDS),
                    "protein_count": expected_count,
                    "residue_count": expected_residues,
                    "max_proteins_per_file": max_proteins,
                    "debug_subset": debug_subset,
                },
                fields=SEED_AVERAGED_FIELDS,
            ) as writer:
                try:
                    sentinel = object()
                    for aligned in zip_longest(*iterators, fillvalue=sentinel):
                        if any(item is sentinel for item in aligned):
                            raise ValueError(
                                f"{condition}/{split}: protein order/count differs "
                                "across seeds"
                            )
                        profiles = tuple(aligned)
                        _validate_aligned_profiles(
                            profiles, f"{condition}/{split} protein #{count + 1}"
                        )
                        averaged = build_seed_average(profiles)
                        profile_audit = audit_seed_average_profile(
                            averaged,
                            normalization_atol=normalization_atol,
                            normalization_rtol=normalization_rtol,
                        )
                        writer.write(averaged)
                        for similarity in profile_similarity_rows(
                            seed_average_as_control(averaged),
                            condition=str(condition),
                            seed="mean_1_2_3",
                            split=str(split),
                            g_uniform_atol=g_uniform_atol,
                            g_uniform_rtol=g_uniform_rtol,
                        ):
                            similarity_writer.write(similarity)
                            accumulator.add(similarity)
                        for source_key, destination_key in (
                            (
                                "mean_uniform_identity_abs_error",
                                "max_mean_uniform_identity_abs_error",
                            ),
                            (
                                "mean_attention_sum_abs_error",
                                "max_mean_attention_sum_abs_error",
                            ),
                            (
                                "mean_amplification_average_abs_error",
                                "max_mean_amplification_average_abs_error",
                            ),
                            (
                                "mean_amplification_identity_abs_error",
                                "max_mean_amplification_identity_abs_error",
                            ),
                            (
                                "mean_shifted_attention_sum_abs_error",
                                "max_mean_shifted_attention_sum_abs_error",
                            ),
                        ):
                            audit[destination_key] = max(
                                audit[destination_key], profile_audit[source_key]
                            )
                        count += 1
                        residues += averaged["length"]
                finally:
                    for iterator in iterators:
                        iterator.close()
                if count != expected_count or residues != expected_residues:
                    raise ValueError(
                        f"{condition}/{split}: averaged {count} proteins/{residues} "
                        f"residues, expected {expected_count}/{expected_residues}"
                    )
            record = {
                "condition": condition,
                "seed": "mean_1_2_3",
                "split": split,
                "source_seeds": "1,2,3",
                "profile_json_gz": str(destination),
                "protein_count": count,
                "residue_count": residues,
                "max_proteins_per_file": max_proteins,
                "debug_subset": debug_subset,
                "run_scope": (
                    "debug_subset" if debug_subset else "full_selected_manifest"
                ),
            }
            manifest_rows.append(record)
            audit["counts_by_condition_split"].append(record)
            audit["protein_profiles"] += count
            audit["residues"] += residues
        similarity_writer.close(commit=True)
    except BaseException:
        similarity_writer.close(commit=False)
        raise

    averaged_manifest = pd.DataFrame(manifest_rows)
    _write_frame_atomic(
        averaged_dir / "seed_averaged_profile_manifest.tsv",
        averaged_manifest,
        sep="\t",
    )
    _write_frame_atomic(
        comparison_dir / "seed_averaged_profile_similarity_summary.csv",
        pd.DataFrame(accumulator.rows()),
    )
    audit["all_invariants_pass"] = True
    audit["per_protein_similarity"] = str(comparison_path)
    _write_json_atomic(
        output_dir / "audits" / "seed_averaging_audit.json", audit
    )
    return averaged_manifest, audit


def write_parameters(
    output_dir: Path,
    *,
    manifest_path: Path,
    args: argparse.Namespace,
    profile_manifest: pd.DataFrame,
    averaged_manifest: pd.DataFrame,
    profile_audit: dict,
    averaging_audit: dict,
) -> None:
    debug_subset = bool(args.max_proteins_per_file)
    payload = {
        "analysis_name": "uniform-attention / evidence-only control",
        "status": (
            "detector-independent debug subset complete"
            if debug_subset
            else "detector-independent selected-manifest stages complete"
        ),
        "source_manifest": str(manifest_path),
        "selection": {
            "requested_conditions": args.conditions,
            "requested_splits": args.splits,
            "selected_conditions": sorted(
                profile_manifest["condition"].astype(str).unique().tolist()
            ),
            "selected_splits": sorted(
                profile_manifest["split"].astype(str).unique().tolist()
            ),
            "max_proteins_per_file": int(args.max_proteins_per_file),
            "debug_subset": debug_subset,
            "run_scope": (
                "debug_subset" if debug_subset else "full_selected_manifest"
            ),
            "per_seed_profile_files": int(len(profile_manifest)),
            "seed_averaged_profile_files": int(len(averaged_manifest)),
            "processed_per_seed_protein_profiles": int(
                profile_audit["protein_profiles"]
            ),
            "processed_seed_averaged_protein_profiles": int(
                averaging_audit["protein_profiles"]
            ),
            "processed_per_seed_residues": int(profile_audit["residues"]),
            "processed_seed_averaged_residues": int(
                averaging_audit["residues"]
            ),
        },
        "definitions": {
            "observed_signed_influence": "I_j_observed = s_j * B_j",
            "uniform_signed_influence": "I_j_uniform = s_j / L",
            "attention_amplification": "G_j = L * B_j",
            "decomposition": "I_j_observed = I_j_uniform * G_j",
            "shifted_attention_control": (
                "I_j_shifted = s_j * circular_shift(B_j); each nontrivial "
                "protein/condition receives a reproducible nonzero offset "
                "independent of s_j and B_j values, shared across model seeds "
                "to preserve cross-seed comparability"
            ),
        },
        "shift_random_seed": args.shift_random_seed,
        "tolerances": {
            "identity_atol": args.identity_atol,
            "identity_rtol": args.identity_rtol,
            "normalization_atol": args.normalization_atol,
            "normalization_rtol": args.normalization_rtol,
            "g_uniform_isclose_atol": args.g_uniform_atol,
            "g_uniform_isclose_rtol": args.g_uniform_rtol,
        },
        "profile_similarity": {
            "sign_strata": "residues stratified by the nonzero sign of s_j",
            "pearson_complete_absolute_magnitude": (
                "Pearson correlation between |I_observed| and |I_uniform| over "
                "the complete protein profile; repeated on the two sign-stratum "
                "rows for that protein"
            ),
            "g_uniform_classification": (
                "G_j is classified as uniform with numpy.isclose(G_j, 1, "
                "g_uniform_isclose_atol, g_uniform_isclose_rtol); amplified and "
                "suppressed fractions exclude those isclose residues"
            ),
            "top_sets": (
                "fixed-cardinality top ceil(fraction*n) residues, with residue "
                "index as deterministic tie-breaker"
            ),
            "rank_displacement": (
                "absolute rank difference over the union of observed and "
                "uniform same-sign top-decile residues"
            ),
            "aggregation": "arithmetic mean of per-protein metrics",
            "profiles_reported": [
                "each per-seed observed-versus-uniform profile",
                "each three-seed-mean observed-versus-uniform profile",
            ],
        },
        "outputs": {
            "per_seed_profile_manifest": str(
                output_dir / "profiles" / "profile_manifest.tsv"
            ),
            "seed_averaged_profile_manifest": str(
                output_dir
                / "seed_averaged_profiles"
                / "seed_averaged_profile_manifest.tsv"
            ),
            "per_seed_profile_similarity": str(
                output_dir
                / "profile_comparison"
                / "per_protein_profile_similarity.csv.gz"
            ),
            "seed_averaged_profile_similarity": str(
                output_dir
                / "profile_comparison"
                / "per_protein_seed_averaged_profile_similarity.csv.gz"
            ),
        },
        "apex_analysis": {
            "run": False,
            "reason": (
                "Profile generation does not call the detector. The generated "
                "profile manifests are directly consumable by extract-bands; "
                "invoke it separately with the same locked Phase 1 parameters "
                "for observed and uniform profiles."
            ),
            "test_set_used_for_tuning": False,
        },
        "interpretive_limit": (
            "Final-attention-layer counterfactual only; not a retrained "
            "uniform-attention model, causal ablation, or physical network."
        ),
    }
    _write_json_atomic(output_dir / "uniform_attention_parameters.json", payload)


def validate_cli_args(args: argparse.Namespace) -> None:
    for name in (
        "identity_atol",
        "identity_rtol",
        "normalization_atol",
        "normalization_rtol",
        "g_uniform_atol",
        "g_uniform_rtol",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be nonnegative")
    if args.max_proteins_per_file < 0:
        raise ValueError("--max_proteins_per_file must be nonnegative")


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    manifest_path = Path(args.manifest_tsv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(
        manifest_path,
        conditions=args.conditions,
        splits=args.splits,
    )
    profile_manifest, profile_audit = generate_per_seed_profiles(
        manifest,
        output_dir,
        identity_atol=args.identity_atol,
        identity_rtol=args.identity_rtol,
        normalization_atol=args.normalization_atol,
        normalization_rtol=args.normalization_rtol,
        g_uniform_atol=args.g_uniform_atol,
        g_uniform_rtol=args.g_uniform_rtol,
        max_proteins_per_file=args.max_proteins_per_file,
        shift_random_seed=args.shift_random_seed,
    )
    averaged_manifest, averaging_audit = generate_seed_averaged_profiles(
        profile_manifest,
        output_dir,
        normalization_atol=args.normalization_atol,
        normalization_rtol=args.normalization_rtol,
        g_uniform_atol=args.g_uniform_atol,
        g_uniform_rtol=args.g_uniform_rtol,
    )
    write_parameters(
        output_dir,
        manifest_path=manifest_path,
        args=args,
        profile_manifest=profile_manifest,
        averaged_manifest=averaged_manifest,
        profile_audit=profile_audit,
        averaging_audit=averaging_audit,
    )
    print(json.dumps({
        "per_seed_profile_files": len(profile_manifest),
        "seed_averaged_profile_files": len(averaged_manifest),
        "per_seed_protein_profiles": profile_audit["protein_profiles"],
        "seed_averaged_protein_profiles": averaging_audit["protein_profiles"],
        "output_dir": str(output_dir),
        "apex_analysis_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
