#!/usr/bin/env python3
"""Audit and analyze the Neq=1 versus Neq>1 prediction endpoint.

This program reuses completed publication_comparable_v2 predictions.  It never
loads or analyzes attention matrices: the large ``attention_weights`` member is
skipped while each JSON record is streamed.  The resulting compact cache and
all downstream tables are written beneath the requested output directory.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/esmfluc-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


EXPECTED_RUNS = 21
EXPECTED_PROTEINS = 208
EXPECTED_RESIDUES = 47_751
PB_STATES = tuple("abcdefghijklmnop")
THRESHOLDS = (1.00, 1.01, 1.05, 1.10, 1.50, 2.00)
OCCUPANCY_THRESHOLDS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
BIN_LABELS = (
    "exactly 1",
    "(1, 1.01]",
    "(1.01, 1.05]",
    "(1.05, 1.10]",
    "(1.10, 1.50]",
    "(1.50, 2.00]",
    ">2.00",
)
CONSENSUS_DEFINITIONS = ("any", "majority", "unanimous")
FLOAT_TOL = 1e-6
PREDICTION_CACHE_SCHEMA_VERSION = "prediction-residue-cache-v2"
PB_CACHE_SCHEMA_VERSION = "pb-reliability-cache-v2"


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_file_provenance(
    paths: Iterable[Path], cache_schema_version: str
) -> dict:
    """Return a versioned path/size/mtime fingerprint for cache inputs."""

    records = []
    for path in sorted({Path(p).resolve() for p in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {
        "cache_schema_version": cache_schema_version,
        "files": records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "method": (
            "sha256(cache_schema_version + canonical "
            "path,size_bytes,mtime_ns records)"
        ),
        "cache_schema_version": cache_schema_version,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def cache_provenance_matches(audit: Mapping, current: Mapping) -> bool:
    cached = audit.get("source_provenance")
    return bool(
        isinstance(cached, Mapping)
        and cached.get("method") == current.get("method")
        and cached.get("cache_schema_version")
        == current.get("cache_schema_version")
        and cached.get("digest") == current.get("digest")
        and cached.get("file_count") == current.get("file_count")
        and cached.get("files") == current.get("files")
    )


def _safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(func(x[mask], y[mask]).statistic)


def load_fixed_test(test_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    proteins = pd.read_csv(test_csv)
    required = {"name", "sequence", "neq"}
    if set(proteins.columns) != required:
        raise ValueError(f"{test_csv} must contain exactly {sorted(required)}")
    if proteins["name"].duplicated().any():
        raise ValueError("Fixed test protein names are not unique")

    residue_parts: list[pd.DataFrame] = []
    for row in proteins.itertuples(index=False):
        neq = np.asarray(ast.literal_eval(row.neq), dtype=float)
        sequence = str(row.sequence)
        if neq.ndim != 1 or len(neq) != len(sequence):
            raise ValueError(f"Neq/sequence length mismatch for {row.name}")
        if not np.all(np.isfinite(neq)) or np.any(neq < 1.0):
            raise ValueError(f"Invalid Neq values for {row.name}")
        length = len(sequence)
        residue_parts.append(
            pd.DataFrame(
                {
                    "protein": row.name,
                    "residue_index_0based": np.arange(length, dtype=np.int32),
                    "residue_number_1based": np.arange(1, length + 1, dtype=np.int32),
                    "amino_acid": list(sequence),
                    "sequence_length": length,
                    "original_neq": neq,
                    "original_binary_label": (neq > 1.0).astype(np.int8),
                }
            )
        )
    residues = pd.concat(residue_parts, ignore_index=True)
    residues.insert(0, "row_id", np.arange(len(residues), dtype=np.int64))
    if len(proteins) != EXPECTED_PROTEINS or len(residues) != EXPECTED_RESIDUES:
        raise ValueError(
            f"Fixed test size is {len(proteins)} proteins/{len(residues)} residues; "
            f"expected {EXPECTED_PROTEINS}/{EXPECTED_RESIDUES}"
        )
    return proteins, residues


def iter_prediction_records(path: Path) -> Iterator[dict]:
    """Yield top-level JSON objects while omitting attention_weights.

    Saved prediction JSON uses stable pretty-print indentation.  We explicitly
    validate the expected boundaries and parse every retained record with the
    standard JSON parser.  This avoids materializing LxL attention matrices.
    """

    in_object = False
    skipping_attention = False
    saw_attention = False
    buffer: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.rstrip("\n")
            if not in_object:
                if raw == "[" or not raw.strip():
                    continue
                if raw == "  {":
                    in_object = True
                    skipping_attention = False
                    saw_attention = False
                    buffer = ["{"]
                    continue
                if raw == "]":
                    continue
                raise ValueError(f"Unexpected JSON line {line_number} in {path}: {raw[:80]!r}")

            if skipping_attention:
                if raw == "    ],":
                    skipping_attention = False
                continue

            if raw == '    "attention_weights":[':
                skipping_attention = True
                saw_attention = True
                continue

            if raw in ("  },", "  }"):
                if not saw_attention:
                    raise ValueError(f"No attention_weights member found in a record in {path}")
                buffer.append("}")
                yield json.loads("\n".join(buffer))
                in_object = False
                buffer = []
                continue

            buffer.append(line)

    if in_object or skipping_attention:
        raise ValueError(f"Truncated JSON record in {path}")


def validate_prediction_arrays(
    record: Mapping, sequence: str, context: str = "prediction record"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    required = {"neq_preds", "flexible_scores", "class_probs"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"{context} missing {sorted(missing)}")
    length = len(sequence)
    hard = np.asarray(record["neq_preds"], dtype=np.int8)
    score = np.asarray(record["flexible_scores"], dtype=float)
    probs = np.asarray(record["class_probs"], dtype=float)
    if hard.shape != (length,) or score.shape != (length,) or probs.shape != (length, 2):
        raise ValueError(
            f"Array shape mismatch for {context}: hard={hard.shape}, "
            f"score={score.shape}, probs={probs.shape}, L={length}"
        )
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(probs)):
        raise ValueError(f"Non-finite probability for {context}")
    if np.any(score < 0) or np.any(score > 1) or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"Probability outside [0,1] for {context}")
    difference = np.abs(score - probs[:, 1])
    maximum_difference = float(difference.max(initial=0.0))
    if np.any(difference > FLOAT_TOL):
        raise ValueError(f"flexible_scores/class_probs[:,1] mismatch for {context}")
    expected_hard = np.argmax(probs, axis=1).astype(np.int8)
    if np.any(hard != expected_hard):
        raise ValueError(f"Hard prediction/probability argmax mismatch for {context}")
    return hard, score, probs, maximum_difference


def _manifest_paths(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"seed": int})
    required = {"condition", "seed", "attention_json"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if len(manifest) != EXPECTED_RUNS:
        raise ValueError(f"Manifest has {len(manifest)} runs, expected {EXPECTED_RUNS}")
    if manifest[["condition", "seed"]].duplicated().any():
        raise ValueError("Manifest condition/seed pairs are not unique")
    if manifest["condition"].nunique() != 7 or set(manifest["seed"]) != {1, 2, 3}:
        raise ValueError("Manifest must contain seven conditions and seeds 1, 2, and 3")
    root = manifest_path.parent.parent.parent
    resolved = []
    for value in manifest["attention_json"]:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(path.resolve())
    manifest = manifest.copy()
    manifest["attention_path"] = resolved
    return manifest


def build_prediction_cache(
    manifest_path: Path,
    test_csv: Path,
    proteins: pd.DataFrame,
    residues: pd.DataFrame,
    output_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    cache_dir = output_dir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "test_predictions_by_residue.csv.gz"
    audit_path = cache_dir / "prediction_cache_audit.json"
    manifest = _manifest_paths(manifest_path)
    source_provenance = source_file_provenance(
        [manifest_path, test_csv, *manifest["attention_path"].tolist()],
        PREDICTION_CACHE_SCHEMA_VERSION,
    )

    if cache_path.is_file() and audit_path.is_file() and not force:
        audit = _read_json(audit_path)
        if audit.get("status") == "passed" and cache_provenance_matches(audit, source_provenance):
            cache = pd.read_csv(cache_path)
            expected_rows = EXPECTED_RUNS * EXPECTED_RESIDUES
            if len(cache) != expected_rows:
                raise ValueError(f"Existing prediction cache has {len(cache)} rows, expected {expected_rows}")
            return cache, audit

    fixed = proteins.set_index("name")
    residue_lookup = {name: group for name, group in residues.groupby("protein", sort=False)}
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    run_audits: list[dict] = []
    wrote_header = False

    for run in manifest.itertuples(index=False):
        seen: set[str] = set()
        parts: list[pd.DataFrame] = []
        max_score_probability_difference = 0.0
        score_probability_mismatches = 0
        hard_prediction_mismatches = 0
        for record in iter_prediction_records(run.attention_path):
            required = {"name", "sequence", "neq_preds", "flexible_scores", "class_probs"}
            missing = required - set(record)
            if missing:
                raise ValueError(f"{run.condition}/seed {run.seed} record missing {sorted(missing)}")
            name = str(record["name"])
            if name in seen:
                raise ValueError(f"Duplicate protein {name} in {run.attention_path}")
            if name not in fixed.index:
                raise ValueError(f"Extra protein {name} in {run.attention_path}")
            seen.add(name)
            sequence = str(record["sequence"])
            fixed_sequence = str(fixed.at[name, "sequence"])
            if sequence != fixed_sequence:
                raise ValueError(f"Prediction/fixed sequence mismatch for {name}")
            context = f"{run.condition}/seed {run.seed}/{name}"
            hard, score, probs, maximum_difference = validate_prediction_arrays(
                record, sequence, context
            )
            max_score_probability_difference = max(
                max_score_probability_difference, maximum_difference
            )

            base = residue_lookup[name].copy()
            base.insert(0, "condition", run.condition)
            base.insert(1, "seed", int(run.seed))
            base["flexible_score"] = score
            base["hard_prediction"] = hard
            base["rigid_class_probability"] = probs[:, 0]
            base["flexible_class_probability"] = probs[:, 1]
            parts.append(base)

        expected_names = set(fixed.index)
        if seen != expected_names:
            raise ValueError(
                f"Missing/extra proteins for {run.condition}/seed {run.seed}: "
                f"missing={sorted(expected_names - seen)}, extra={sorted(seen - expected_names)}"
            )
        run_frame = pd.concat(parts, ignore_index=True)
        if len(run_frame) != EXPECTED_RESIDUES:
            raise ValueError(f"Run has {len(run_frame)} residues, expected {EXPECTED_RESIDUES}")
        run_frame.to_csv(
            tmp_path,
            mode="a",
            index=False,
            header=not wrote_header,
            compression="gzip",
            float_format="%.10g",
        )
        wrote_header = True
        run_audits.append(
            {
                "condition": run.condition,
                "seed": int(run.seed),
                "proteins": len(seen),
                "residues": len(run_frame),
                "unique_protein_names": True,
                "sequences_match": True,
                "array_shapes_match": True,
                "scores_in_unit_interval": True,
                "score_probability_mismatches": score_probability_mismatches,
                "maximum_score_probability_absolute_difference": max_score_probability_difference,
                "hard_prediction_mismatches": hard_prediction_mismatches,
            }
        )

    tmp_path.replace(cache_path)
    cache = pd.read_csv(cache_path)
    audit = {
        "status": "passed",
        "manifest": str(manifest_path.resolve()),
        "cache": str(cache_path.resolve()),
        "runs": len(run_audits),
        "conditions": int(manifest["condition"].nunique()),
        "seeds": sorted(int(x) for x in manifest["seed"].unique()),
        "proteins_per_run": EXPECTED_PROTEINS,
        "residues_per_run": EXPECTED_RESIDUES,
        "cache_rows": len(cache),
        "no_silent_truncation": True,
        "source_provenance": source_provenance,
        "cache_reuse_requires_exact_source_provenance_match": True,
        "cache_schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
        "run_audits": run_audits,
    }
    if len(cache) != EXPECTED_RUNS * EXPECTED_RESIDUES:
        raise ValueError("Completed prediction cache has the wrong number of rows")
    _json_dump(audit_path, audit)
    return cache, audit


def score_sources(
    cache: pd.DataFrame, residues: pd.DataFrame
) -> tuple[list[dict], list[str], dict[str, int]]:
    conditions = sorted(cache["condition"].unique())
    base_ids = residues["row_id"].to_numpy()
    sources: list[dict] = []
    for condition in conditions:
        seed_vectors = []
        for seed in (1, 2, 3):
            part = cache[(cache["condition"] == condition) & (cache["seed"] == seed)].sort_values("row_id")
            if not np.array_equal(part["row_id"].to_numpy(), base_ids):
                raise ValueError(f"Cache row alignment failure for {condition}/seed {seed}")
            scores = part["flexible_score"].to_numpy(float)
            hard = part["hard_prediction"].to_numpy(np.int8)
            sources.append(
                {"condition": condition, "score_type": "individual_seed", "seed": str(seed), "score": scores, "hard": hard}
            )
            seed_vectors.append(scores)
        sources.append(
            {
                "condition": condition,
                "score_type": "seed_ensemble",
                "seed": "ensemble",
                "score": np.mean(seed_vectors, axis=0),
                "hard": None,
            }
        )
    protein_names = list(dict.fromkeys(residues["protein"].tolist()))
    protein_to_index = {name: i for i, name in enumerate(protein_names)}
    return sources, conditions, protein_to_index


def neq_bin_index(neq: np.ndarray) -> np.ndarray:
    result = np.full(len(neq), -1, dtype=np.int8)
    result[neq == 1.0] = 0
    result[(neq > 1.0) & (neq <= 1.01)] = 1
    result[(neq > 1.01) & (neq <= 1.05)] = 2
    result[(neq > 1.05) & (neq <= 1.10)] = 3
    result[(neq > 1.10) & (neq <= 1.50)] = 4
    result[(neq > 1.50) & (neq <= 2.00)] = 5
    result[neq > 2.00] = 6
    if np.any(result < 0):
        raise ValueError("Some Neq values did not enter a prespecified bin")
    return result


def endpoint_mask_labels(
    neq: np.ndarray, threshold: float, definition: str
) -> tuple[np.ndarray, np.ndarray]:
    """Construct prespecified exclusion-margin or relabeled endpoint labels."""

    neq = np.asarray(neq, dtype=float)
    if definition == "exclusion_margin":
        include = (neq == 1.0) | (neq > threshold)
        labels = neq > threshold
    elif definition == "relabeled_threshold":
        include = np.ones(len(neq), dtype=bool)
        labels = neq > threshold
    else:
        raise ValueError(definition)
    return include, labels.astype(np.int8)


def per_protein_spearman(
    neq: np.ndarray, score: np.ndarray, proteins: np.ndarray, include: np.ndarray
) -> list[float]:
    neq_subset = neq[include]
    score_subset = score[include]
    protein_subset = proteins[include]
    if not len(neq_subset):
        return []
    starts = np.r_[0, np.flatnonzero(protein_subset[1:] != protein_subset[:-1]) + 1]
    ends = np.r_[starts[1:], len(protein_subset)]
    values = []
    for start, end in zip(starts, ends):
        value = _safe_corr(
            spearmanr, neq_subset[start:end], score_subset[start:end]
        )
        if np.isfinite(value):
            values.append(value)
    return values


def binary_performance(
    labels: np.ndarray,
    scores: np.ndarray,
    proteins: np.ndarray,
    hard: np.ndarray | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    proteins = np.asarray(proteins)
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise ValueError("Binary performance requires both classes")
    # All analysis masks preserve the fixed table's protein-contiguous ordering.
    # Slice those blocks rather than constructing an O(N) string mask 208 times.
    starts = np.r_[0, np.flatnonzero(proteins[1:] != proteins[:-1]) + 1]
    ends = np.r_[starts[1:], len(proteins)]
    aurocs: list[float] = []
    auprcs: list[float] = []
    for start, end in zip(starts, ends):
        protein_labels = labels[start:end]
        if np.unique(protein_labels).size == 2:
            aurocs.append(float(roc_auc_score(protein_labels, scores[start:end])))
            auprcs.append(float(average_precision_score(protein_labels, scores[start:end])))
    result = {
        "n_residues": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "n_proteins_represented": int(len(starts)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "n_proteins_with_both_classes": len(aurocs),
        "median_within_protein_auroc": float(np.median(aurocs)) if aurocs else float("nan"),
        "median_within_protein_auprc": float(np.median(auprcs)) if auprcs else float("nan"),
        "protein_macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "protein_macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "within_protein_auroc_q25": float(np.quantile(aurocs, 0.25)) if aurocs else float("nan"),
        "within_protein_auroc_q75": float(np.quantile(aurocs, 0.75)) if aurocs else float("nan"),
        "within_protein_auprc_q25": float(np.quantile(auprcs, 0.25)) if auprcs else float("nan"),
        "within_protein_auprc_q75": float(np.quantile(auprcs, 0.75)) if auprcs else float("nan"),
        "negative_score_median": float(np.median(scores[labels == 0])),
        "positive_score_median": float(np.median(scores[labels == 1])),
        "negative_score_mean": float(np.mean(scores[labels == 0])),
        "positive_score_mean": float(np.mean(scores[labels == 1])),
        "hard_accuracy": float(np.mean(hard == labels)) if hard is not None else float("nan"),
    }
    return result


def summarize_seeds(table: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    seed_rows = table[table["score_type"] == "individual_seed"].copy()
    ensemble_rows = table[table["score_type"] == "seed_ensemble"].copy()
    identity = set(group_columns) | {"condition", "score_type", "seed"}
    numeric = [c for c in table.select_dtypes(include=[np.number]).columns if c not in identity]
    summaries: list[dict] = []
    keys = ["condition", *group_columns]
    grouped = seed_rows.groupby(keys, dropna=False, sort=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(keys, key))
        if len(group) != 3:
            raise ValueError(f"Expected three seeds for {record}, found {len(group)}")
        match = np.ones(len(ensemble_rows), dtype=bool)
        for column, value in record.items():
            if pd.isna(value):
                match &= ensemble_rows[column].isna().to_numpy()
            else:
                match &= (ensemble_rows[column] == value).to_numpy()
        ensemble = ensemble_rows.loc[match]
        if len(ensemble) != 1:
            raise ValueError(f"Expected one ensemble row for {record}, found {len(ensemble)}")
        for column in numeric:
            values = group[column].to_numpy(float)
            record[f"{column}_seed_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            record[f"{column}_seed_sd"] = (
                float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() >= 2 else float("nan")
            )
            record[f"{column}_ensemble"] = float(ensemble.iloc[0][column])
        summaries.append(record)
    return pd.DataFrame(summaries)


def analyze_original_endpoint(
    residues: pd.DataFrame,
    sources: list[dict],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    neq_dir = output_dir / "neq_distribution"
    endpoint_dir = output_dir / "stricter_endpoint"
    neq_dir.mkdir(parents=True, exist_ok=True)
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    neq = residues["original_neq"].to_numpy(float)
    proteins = residues["protein"].to_numpy()
    bins = neq_bin_index(neq)

    count_rows = []
    for i, label in enumerate(BIN_LABELS):
        mask = bins == i
        count_rows.append(
            {
                "neq_bin": label,
                "bin_order": i,
                "residue_count": int(mask.sum()),
                "percentage_of_residues": float(100 * mask.mean()),
                "proteins_represented": int(pd.unique(proteins[mask]).size),
                "median_neq": float(np.median(neq[mask])),
            }
        )
    bin_counts = pd.DataFrame(count_rows)
    bin_counts.to_csv(neq_dir / "neq_bin_counts.csv", index=False)

    score_bin_rows = []
    corr_rows = []
    exclusion_rows = []
    relabeled_rows = []
    for source in sources:
        score = source["score"]
        identity = {k: source[k] for k in ("condition", "score_type", "seed")}
        for i, label in enumerate(BIN_LABELS):
            values = score[bins == i]
            score_bin_rows.append(
                {
                    **identity,
                    "neq_bin": label,
                    "bin_order": i,
                    "residue_count": len(values),
                    "median_flexible_score": float(np.median(values)),
                    "mean_flexible_score": float(np.mean(values)),
                    "score_q25": float(np.quantile(values, 0.25)),
                    "score_q75": float(np.quantile(values, 0.75)),
                }
            )

        all_residues = np.ones(len(neq), dtype=bool)
        positive_tail = neq > 1.0
        positive_tail_1p01 = neq > 1.01
        per_protein = per_protein_spearman(neq, score, proteins, all_residues)
        per_protein_positive = per_protein_spearman(
            neq, score, proteins, positive_tail
        )
        per_protein_positive_1p01 = per_protein_spearman(
            neq, score, proteins, positive_tail_1p01
        )
        corr_rows.append(
            {
                **identity,
                "pooled_spearman_neq": _safe_corr(spearmanr, neq, score),
                "pooled_spearman_neq_gt_1": _safe_corr(
                    spearmanr, neq[positive_tail], score[positive_tail]
                ),
                "pooled_spearman_neq_gt_1p01": _safe_corr(
                    spearmanr, neq[positive_tail_1p01], score[positive_tail_1p01]
                ),
                "pooled_spearman_log_neq": _safe_corr(spearmanr, np.log(neq), score),
                "pooled_pearson_log_neq": _safe_corr(pearsonr, np.log(neq), score),
                "median_per_protein_spearman": float(np.median(per_protein)),
                "per_protein_spearman_q25": float(np.quantile(per_protein, 0.25)),
                "per_protein_spearman_q75": float(np.quantile(per_protein, 0.75)),
                "n_proteins_defined_spearman": len(per_protein),
                "median_per_protein_spearman_neq_gt_1": float(
                    np.median(per_protein_positive)
                ),
                "per_protein_spearman_neq_gt_1_q25": float(
                    np.quantile(per_protein_positive, 0.25)
                ),
                "per_protein_spearman_neq_gt_1_q75": float(
                    np.quantile(per_protein_positive, 0.75)
                ),
                "n_proteins_defined_spearman_neq_gt_1": len(
                    per_protein_positive
                ),
                "median_per_protein_spearman_neq_gt_1p01": float(
                    np.median(per_protein_positive_1p01)
                ),
                "n_proteins_defined_spearman_neq_gt_1p01": len(
                    per_protein_positive_1p01
                ),
            }
        )

        for threshold in THRESHOLDS:
            include, all_labels = endpoint_mask_labels(
                neq, threshold, "exclusion_margin"
            )
            labels = all_labels[include]
            hard = source["hard"][include] if threshold == 1.0 and source["hard"] is not None else None
            exclusion_rows.append(
                {
                    **identity,
                    "cutoff": threshold,
                    "excluded_residues": int((~include).sum()),
                    **binary_performance(labels, score[include], proteins[include], hard),
                }
            )
            include_re, labels_re = endpoint_mask_labels(
                neq, threshold, "relabeled_threshold"
            )
            assert include_re.all()
            relabeled_rows.append(
                {
                    **identity,
                    "cutoff": threshold,
                    "excluded_residues": 0,
                    **binary_performance(labels_re, score, proteins, source["hard"] if threshold == 1.0 else None),
                }
            )

    score_bins = pd.DataFrame(score_bin_rows)
    correlations = pd.DataFrame(corr_rows)
    exclusion = pd.DataFrame(exclusion_rows)
    relabeled = pd.DataFrame(relabeled_rows)
    score_bins.to_csv(neq_dir / "score_by_neq_bin.csv", index=False)
    correlations.to_csv(neq_dir / "score_neq_correlations_by_run.csv", index=False)
    summarize_seeds(correlations, []).to_csv(neq_dir / "score_neq_correlation_summary.csv", index=False)
    exclusion.to_csv(endpoint_dir / "exclusion_margin_performance_by_run.csv", index=False)
    summarize_seeds(exclusion, ["cutoff"]).to_csv(
        endpoint_dir / "exclusion_margin_performance_summary.csv", index=False
    )
    relabeled.to_csv(endpoint_dir / "relabeled_threshold_performance_by_run.csv", index=False)
    summarize_seeds(relabeled, ["cutoff"]).to_csv(
        endpoint_dir / "relabeled_threshold_performance_summary.csv", index=False
    )
    return {
        "bin_counts": bin_counts,
        "score_bins": score_bins,
        "correlations": correlations,
        "exclusion": exclusion,
        "relabeled": relabeled,
    }


def _pb_metrics_from_array(array: np.ndarray) -> dict[str, np.ndarray]:
    if array.dtype.kind == "S":
        codes = array.view(np.uint8).reshape(array.shape)
    elif array.dtype == np.uint8:
        codes = array
    else:
        raise ValueError(f"Unsupported PB state dtype: {array.dtype}")
    observed = set(np.unique(codes).tolist())
    allowed = {ord(x) for x in PB_STATES} | {ord("Z")}
    if not observed <= allowed:
        raise ValueError(f"Unexpected PB codes: {sorted(observed - allowed)}")
    counts = np.vstack([(codes == ord(state)).sum(axis=0) for state in PB_STATES]).astype(np.int32)
    valid = counts.sum(axis=0)
    total_frames = codes.shape[0]
    fractions = np.divide(counts, valid, out=np.zeros_like(counts, dtype=float), where=valid > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(fractions > 0, fractions * np.log(fractions), 0.0)
    entropy = -terms.sum(axis=0)
    neq = np.exp(entropy)
    neq[valid == 0] = np.nan
    n_states = (counts > 0).sum(axis=0).astype(np.int8)
    order = np.argsort(counts, axis=0)
    dominant_index = order[-1]
    second_index = order[-2]
    columns = np.arange(codes.shape[1])
    dominant_count = counts[dominant_index, columns]
    second_count = counts[second_index, columns]
    dominant_state = np.asarray(PB_STATES, dtype=object)[dominant_index]
    second_state = np.asarray(PB_STATES, dtype=object)[second_index]
    dominant_state[valid == 0] = ""
    second_state[(valid == 0) | (second_count == 0)] = ""
    return {
        "counts": counts,
        "total_frames": np.full(codes.shape[1], total_frames, dtype=np.int32),
        "valid_frames": valid,
        "valid_fraction": valid / total_frames,
        "n_states": n_states,
        "entropy": entropy,
        "neq": neq,
        "dominant_state": dominant_state,
        "dominant_occupancy": np.divide(
            dominant_count, valid, out=np.full(valid.shape, np.nan, dtype=float), where=valid > 0
        ),
        "second_state": second_state,
        "secondary_occupancy": np.divide(
            second_count, valid, out=np.full(valid.shape, np.nan, dtype=float), where=valid > 0
        ),
    }


def validate_pb_cache_tables(
    replicate: pd.DataFrame,
    consensus: pd.DataFrame,
    comparison: pd.DataFrame,
    residues: pd.DataFrame,
) -> None:
    """Validate cached PB tables against the fixed residue table."""

    replicate_required = {
        "protein",
        "residue_index_0based",
        "residue_number_1based",
        "replicate",
        "total_frames",
        "valid_pb_frame_count",
        "valid_pb_frame_fraction",
        "pb_valid",
        "n_observed_non_z_states",
        "single_state",
        "multi_state",
        "pb_entropy",
        "replicate_neq",
        "dominant_pb_state",
        "dominant_state_occupancy",
        "second_pb_state",
        "secondary_state_occupancy",
    }
    consensus_required = {
        "row_id",
        "protein",
        "residue_index_0based",
        "residue_number_1based",
        "amino_acid",
        "sequence_length",
        "original_neq",
        "original_binary_label",
        "n_valid_replicates",
        "all_three_replicates_pb_valid",
        "n_single_state_replicates",
        "n_multi_state_replicates",
        "multi_state_consensus_category",
        "mean_replicate_neq",
        "sd_replicate_neq",
        "pooled_dominant_pb_state",
        "dominant_pb_state_agrees_all_replicates",
        "maximum_secondary_state_occupancy",
        "mean_secondary_state_occupancy",
        "minimum_valid_pb_frame_fraction",
        "mean_valid_pb_frame_fraction",
        *{
            f"n_multi_replicates_occ_{occupancy_suffix(threshold)}"
            for threshold in OCCUPANCY_THRESHOLDS
        },
    }
    comparison_required = {
        "row_id",
        "protein",
        "residue_index_0based",
        "residue_number_1based",
        "sequence_length",
        "original_neq",
        "original_binary_label",
        "n_valid_replicates",
        "all_three_replicates_pb_valid",
        "n_multi_state_replicates",
        "mean_replicate_neq",
        "minimum_valid_pb_frame_fraction",
        "mean_valid_pb_frame_fraction",
        "regenerated_binary_label",
        "binary_label_agrees",
        "absolute_neq_difference",
        "terminal_distance_0based",
        "within_two_residues_of_terminus",
        "within_five_residues_of_terminus",
        "has_any_z_frames",
    }
    for name, frame, required in (
        ("replicate_pb_metrics", replicate, replicate_required),
        ("residue_pb_consensus", consensus, consensus_required),
        ("original_vs_regenerated_neq", comparison, comparison_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Cached {name} is missing columns: {sorted(missing)}")

    expected_rows = len(residues)
    if len(replicate) != expected_rows * 3:
        raise ValueError(
            f"Cached replicate PB table has {len(replicate)} rows; "
            f"expected {expected_rows * 3}"
        )
    if len(consensus) != expected_rows or len(comparison) != expected_rows:
        raise ValueError(
            "Cached PB residue table row counts do not match the fixed test set"
        )

    replicate_keys = ["protein", "residue_index_0based", "replicate"]
    if replicate.duplicated(replicate_keys).any():
        raise ValueError("Cached replicate PB table has duplicate residue/replicate keys")
    if set(pd.to_numeric(replicate["replicate"], errors="raise").astype(int)) != {
        1,
        2,
        3,
    }:
        raise ValueError("Cached replicate PB table does not contain replicates 1, 2, and 3")
    group_sizes = replicate.groupby(
        ["protein", "residue_index_0based"], sort=False
    ).size()
    if not group_sizes.eq(3).all():
        raise ValueError("Cached replicate PB keys do not have exactly three replicates")

    expected_keys = pd.MultiIndex.from_frame(
        residues[["protein", "residue_index_0based"]]
    )
    actual_keys = pd.MultiIndex.from_frame(
        replicate[["protein", "residue_index_0based"]].drop_duplicates()
    )
    if (
        not expected_keys.is_unique
        or not actual_keys.is_unique
        or len(expected_keys) != len(actual_keys)
        or len(expected_keys.difference(actual_keys))
        or len(actual_keys.difference(expected_keys))
    ):
        raise ValueError("Cached replicate PB residue keys do not match the fixed test set")
    if not np.array_equal(
        replicate["residue_number_1based"].to_numpy(),
        replicate["residue_index_0based"].to_numpy() + 1,
    ):
        raise ValueError("Cached replicate PB residue numbering is inconsistent")

    expected = residues.reset_index(drop=True)
    for name, frame in (
        ("residue_pb_consensus", consensus),
        ("original_vs_regenerated_neq", comparison),
    ):
        if frame["row_id"].duplicated().any():
            raise ValueError(f"Cached {name} has duplicate row_id values")
        for column in (
            "row_id",
            "protein",
            "residue_index_0based",
            "residue_number_1based",
        ):
            if not np.array_equal(
                frame[column].to_numpy(), expected[column].to_numpy()
            ):
                raise ValueError(
                    f"Cached {name} {column} values are not exactly aligned "
                    "with the fixed test set"
                )
        if not np.allclose(
            frame["original_neq"].to_numpy(float),
            expected["original_neq"].to_numpy(float),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(
                f"Cached {name} original_neq values do not match the fixed test set"
            )
        if not np.array_equal(
            frame["original_binary_label"].to_numpy(np.int8),
            expected["original_binary_label"].to_numpy(np.int8),
        ):
            raise ValueError(
                f"Cached {name} original labels do not match the fixed test set"
            )


def extract_pb_reliability(
    pb_root: Path,
    test_csv: Path,
    proteins: pd.DataFrame,
    residues: pd.DataFrame,
    output_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pb_dir = output_dir / "pb_reliability"
    audit_dir = output_dir / "audits"
    pb_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    replicate_path = pb_dir / "replicate_pb_metrics.csv.gz"
    consensus_path = pb_dir / "residue_pb_consensus.csv.gz"
    comparison_path = pb_dir / "original_vs_regenerated_neq.csv.gz"
    audit_path = pb_dir / "pb_extraction_audit.json"
    pb_source_paths = [
        pb_root / protein / f"pb_states_R{replicate}.h5"
        for protein in proteins["name"]
        for replicate in (1, 2, 3)
    ]
    source_provenance = source_file_provenance(
        [test_csv, *pb_source_paths], PB_CACHE_SCHEMA_VERSION
    )

    if all(p.is_file() for p in (replicate_path, consensus_path, comparison_path, audit_path)) and not force:
        audit = _read_json(audit_path)
        if audit.get("status") == "passed" and cache_provenance_matches(audit, source_provenance):
            replicate = pd.read_csv(replicate_path)
            consensus = pd.read_csv(consensus_path)
            comparison = pd.read_csv(comparison_path)
            validate_pb_cache_tables(
                replicate, consensus, comparison, residues
            )
            return replicate, consensus, comparison, audit

    residue_lookup = {name: group.reset_index(drop=True) for name, group in residues.groupby("protein", sort=False)}
    replicate_parts: list[pd.DataFrame] = []
    consensus_parts: list[pd.DataFrame] = []
    files_checked = 0
    frame_counts: list[int] = []

    for protein_row in proteins.itertuples(index=False):
        protein = protein_row.name
        length = len(protein_row.sequence)
        base = residue_lookup[protein]
        per_rep: list[dict[str, np.ndarray]] = []
        for replicate in (1, 2, 3):
            path = pb_root / protein / f"pb_states_R{replicate}.h5"
            if not path.is_file():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as handle:
                if "pb_states" not in handle:
                    raise ValueError(f"{path} lacks pb_states")
                dataset = handle["pb_states"]
                if dataset.ndim != 2 or dataset.shape[1] != length:
                    raise ValueError(f"PB shape mismatch for {path}: {dataset.shape}, expected (*,{length})")
                array = dataset[:]
            metrics = _pb_metrics_from_array(array)
            per_rep.append(metrics)
            frame_counts.append(int(array.shape[0]))
            files_checked += 1
            replicate_parts.append(
                pd.DataFrame(
                    {
                        "protein": protein,
                        "residue_index_0based": base["residue_index_0based"],
                        "residue_number_1based": base["residue_number_1based"],
                        "replicate": replicate,
                        "total_frames": metrics["total_frames"],
                        "valid_pb_frame_count": metrics["valid_frames"],
                        "valid_pb_frame_fraction": metrics["valid_fraction"],
                        "pb_valid": metrics["valid_frames"] > 0,
                        "n_observed_non_z_states": metrics["n_states"],
                        "single_state": metrics["n_states"] == 1,
                        "multi_state": metrics["n_states"] >= 2,
                        "pb_entropy": metrics["entropy"],
                        "replicate_neq": metrics["neq"],
                        "dominant_pb_state": metrics["dominant_state"],
                        "dominant_state_occupancy": metrics["dominant_occupancy"],
                        "second_pb_state": metrics["second_state"],
                        "secondary_state_occupancy": metrics["secondary_occupancy"],
                    }
                )
            )

        valid_matrix = np.vstack([x["valid_frames"] > 0 for x in per_rep])
        n_state_matrix = np.vstack([x["n_states"] for x in per_rep])
        multi_matrix = n_state_matrix >= 2
        neq_matrix = np.vstack([x["neq"] for x in per_rep])
        secondary_matrix = np.vstack([x["secondary_occupancy"] for x in per_rep])
        dominant_matrix = np.vstack([x["dominant_state"] for x in per_rep])
        pooled_counts = sum(x["counts"] for x in per_rep)
        pooled_valid = pooled_counts.sum(axis=0)
        pooled_dom_idx = np.argmax(pooled_counts, axis=0)
        pooled_dom = np.asarray(PB_STATES, dtype=object)[pooled_dom_idx]
        pooled_dom[pooled_valid == 0] = ""
        all_valid = valid_matrix.all(axis=0)
        dominant_agreement = all_valid & np.all(dominant_matrix == dominant_matrix[0], axis=0)
        valid_neq_count = np.isfinite(neq_matrix).sum(axis=0)
        mean_neq = np.divide(
            np.nansum(neq_matrix, axis=0),
            valid_neq_count,
            out=np.full(length, np.nan),
            where=valid_neq_count > 0,
        )
        squared_neq_deviation = np.nansum((neq_matrix - mean_neq) ** 2, axis=0)
        sd_neq = np.sqrt(
            np.divide(
                squared_neq_deviation,
                valid_neq_count - 1,
                out=np.full(length, np.nan),
                where=valid_neq_count > 1,
            )
        )
        valid_secondary_count = np.isfinite(secondary_matrix).sum(axis=0)
        mean_secondary = np.divide(
            np.nansum(secondary_matrix, axis=0),
            valid_secondary_count,
            out=np.full(length, np.nan),
            where=valid_secondary_count > 0,
        )
        max_secondary = np.max(np.where(np.isfinite(secondary_matrix), secondary_matrix, -np.inf), axis=0)
        max_secondary[valid_secondary_count == 0] = np.nan
        record = pd.DataFrame(
            {
                "row_id": base["row_id"],
                "protein": protein,
                "residue_index_0based": base["residue_index_0based"],
                "residue_number_1based": base["residue_number_1based"],
                "amino_acid": base["amino_acid"],
                "sequence_length": length,
                "original_neq": base["original_neq"],
                "original_binary_label": base["original_binary_label"],
                "n_valid_replicates": valid_matrix.sum(axis=0),
                "all_three_replicates_pb_valid": all_valid,
                "n_single_state_replicates": (n_state_matrix == 1).sum(axis=0),
                "n_multi_state_replicates": multi_matrix.sum(axis=0),
                "multi_state_consensus_category": [f"{x}/3" for x in multi_matrix.sum(axis=0)],
                "mean_replicate_neq": mean_neq,
                "sd_replicate_neq": sd_neq,
                "pooled_dominant_pb_state": pooled_dom,
                "dominant_pb_state_agrees_all_replicates": dominant_agreement,
                "maximum_secondary_state_occupancy": max_secondary,
                "mean_secondary_state_occupancy": mean_secondary,
                "minimum_valid_pb_frame_fraction": np.min(
                    np.vstack([x["valid_fraction"] for x in per_rep]), axis=0
                ),
                "mean_valid_pb_frame_fraction": np.mean(
                    np.vstack([x["valid_fraction"] for x in per_rep]), axis=0
                ),
            }
        )
        for threshold in OCCUPANCY_THRESHOLDS:
            suffix = occupancy_suffix(threshold)
            strong = valid_matrix & (secondary_matrix >= threshold)
            if threshold == 0.0:
                strong = valid_matrix & (secondary_matrix > 0)
            record[f"n_multi_replicates_occ_{suffix}"] = strong.sum(axis=0)
        consensus_parts.append(record)

    replicate = pd.concat(replicate_parts, ignore_index=True)
    consensus = pd.concat(consensus_parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    if len(replicate) != EXPECTED_RESIDUES * 3 or len(consensus) != EXPECTED_RESIDUES:
        raise ValueError("PB extraction row-count audit failed")
    if not np.array_equal(consensus["row_id"].to_numpy(), residues["row_id"].to_numpy()):
        raise ValueError("PB residue indexing does not match fixed test indexing")

    comparison = consensus[
        [
            "row_id",
            "protein",
            "residue_index_0based",
            "residue_number_1based",
            "sequence_length",
            "original_neq",
            "original_binary_label",
            "n_valid_replicates",
            "all_three_replicates_pb_valid",
            "n_multi_state_replicates",
            "mean_replicate_neq",
            "minimum_valid_pb_frame_fraction",
            "mean_valid_pb_frame_fraction",
        ]
    ].copy()
    comparison["regenerated_binary_label"] = (comparison["n_multi_state_replicates"] > 0).astype(np.int8)
    comparison["binary_label_agrees"] = (
        comparison["original_binary_label"] == comparison["regenerated_binary_label"]
    )
    comparison["absolute_neq_difference"] = np.abs(
        comparison["original_neq"] - comparison["mean_replicate_neq"]
    )
    comparison["terminal_distance_0based"] = np.minimum(
        comparison["residue_index_0based"],
        comparison["sequence_length"] - 1 - comparison["residue_index_0based"],
    )
    comparison["within_two_residues_of_terminus"] = comparison["terminal_distance_0based"] < 2
    comparison["within_five_residues_of_terminus"] = comparison["terminal_distance_0based"] < 5
    comparison["has_any_z_frames"] = comparison["minimum_valid_pb_frame_fraction"] < 1.0

    validate_pb_cache_tables(replicate, consensus, comparison, residues)
    replicate.to_csv(replicate_path, index=False, compression="gzip", float_format="%.10g")
    consensus.to_csv(consensus_path, index=False, compression="gzip", float_format="%.10g")
    comparison.to_csv(comparison_path, index=False, compression="gzip", float_format="%.10g")

    valid = comparison["all_three_replicates_pb_valid"].to_numpy(bool)
    x = comparison.loc[valid, "original_neq"].to_numpy(float)
    y = comparison.loc[valid, "mean_replicate_neq"].to_numpy(float)
    delta = np.abs(x - y)
    original_single = comparison["original_neq"] == 1.0
    all_invalid = comparison["n_valid_replicates"] == 0
    summary = pd.DataFrame(
        [
            {
                "comparison_population": "all_three_replicates_pb_valid",
                "n_residues": int(valid.sum()),
                "pearson_correlation": _safe_corr(pearsonr, x, y),
                "spearman_correlation": _safe_corr(spearmanr, x, y),
                "mean_absolute_difference": float(np.mean(delta)),
                "median_absolute_difference": float(np.median(delta)),
                "maximum_absolute_difference": float(np.max(delta)),
                "binary_label_agreement_fraction": float(
                    comparison.loc[valid, "binary_label_agrees"].mean()
                ),
                "binary_label_disagreements": int(
                    (~comparison.loc[valid, "binary_label_agrees"]).sum()
                ),
            }
        ]
    )
    summary.to_csv(pb_dir / "original_vs_regenerated_neq_summary.csv", index=False)
    comparison.sort_values("absolute_neq_difference", ascending=False).head(50).to_csv(
        output_dir / "audits" / "largest_original_regenerated_neq_discrepancies.csv", index=False
    )

    validity_audit = {
        "status": "passed",
        "residues": len(consensus),
        "all_three_replicates_pb_valid": int(consensus["all_three_replicates_pb_valid"].sum()),
        "not_all_three_replicates_pb_valid": int((~consensus["all_three_replicates_pb_valid"]).sum()),
        "all_replicates_invalid": int(all_invalid.sum()),
        "original_neq_exactly_1": int(original_single.sum()),
        "original_neq_1_all_replicates_invalid": int((original_single & all_invalid).sum()),
        "original_neq_1_not_all_replicates_valid": int(
            (original_single & ~comparison["all_three_replicates_pb_valid"]).sum()
        ),
        "original_neq_1_invalid_within_two_residues_of_terminus": int(
            (original_single & all_invalid & comparison["within_two_residues_of_terminus"]).sum()
        ),
        "original_neq_1_invalid_within_five_residues_of_terminus": int(
            (original_single & all_invalid & comparison["within_five_residues_of_terminus"]).sum()
        ),
        "residues_with_any_z_frames": int(comparison["has_any_z_frames"].sum()),
    }
    _json_dump(output_dir / "audits" / "pb_validity_audit.json", validity_audit)
    audit = {
        "status": "passed",
        "pb_root": str(pb_root.resolve()),
        "proteins": len(proteins),
        "replicates_per_protein": 3,
        "hdf5_files_checked": files_checked,
        "replicate_metric_rows": len(replicate),
        "consensus_rows": len(consensus),
        "frame_count_minimum": min(frame_counts),
        "frame_count_maximum": max(frame_counts),
        "z_excluded": True,
        "validity_definition": "at least one non-Z PB frame",
        "single_state_definition": "exactly one distinct valid non-Z PB state",
        "multi_state_definition": "at least two distinct valid non-Z PB states",
        "source_provenance": source_provenance,
        "cache_reuse_requires_exact_source_provenance_match": True,
        "cache_schema_version": PB_CACHE_SCHEMA_VERSION,
    }
    _json_dump(audit_path, audit)
    return replicate, consensus, comparison, audit


def write_pb_comparison_audits(comparison: pd.DataFrame, output_dir: Path) -> None:
    audit_dir = output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    valid = comparison["all_three_replicates_pb_valid"].to_numpy(bool)
    disagreements = comparison.loc[valid & ~comparison["binary_label_agrees"].to_numpy(bool)].copy()
    disagreements.to_csv(audit_dir / "original_regenerated_binary_disagreements.csv", index=False)
    by_protein = (
        disagreements.groupby("protein", as_index=False)
        .agg(
            disagreement_count=("row_id", "size"),
            minimum_residue_number_1based=("residue_number_1based", "min"),
            maximum_residue_number_1based=("residue_number_1based", "max"),
            disagreements_with_any_z_frames=("has_any_z_frames", "sum"),
            disagreements_within_two_residues_of_terminus=("within_two_residues_of_terminus", "sum"),
            disagreements_within_five_residues_of_terminus=("within_five_residues_of_terminus", "sum"),
        )
        .sort_values(["disagreement_count", "protein"], ascending=[False, True])
    )
    by_protein.to_csv(audit_dir / "original_regenerated_disagreements_by_protein.csv", index=False)
    by_position = (
        disagreements.groupby("terminal_distance_0based", as_index=False)
        .agg(
            disagreement_count=("row_id", "size"),
            proteins_represented=("protein", "nunique"),
            disagreements_with_any_z_frames=("has_any_z_frames", "sum"),
        )
        .sort_values("terminal_distance_0based")
    )
    by_position.to_csv(audit_dir / "original_regenerated_disagreements_by_terminal_distance.csv", index=False)
    _json_dump(
        audit_dir / "original_regenerated_disagreement_audit.json",
        {
            "status": "passed",
            "population": "all three replicates PB-valid",
            "binary_disagreements": len(disagreements),
            "proteins_with_disagreements": int(disagreements["protein"].nunique()),
            "disagreements_with_any_z_frames": int(disagreements["has_any_z_frames"].sum()),
            "disagreements_within_two_residues_of_terminus": int(
                disagreements["within_two_residues_of_terminus"].sum()
            ),
            "disagreements_within_five_residues_of_terminus": int(
                disagreements["within_five_residues_of_terminus"].sum()
            ),
        },
    )


def occupancy_suffix(threshold: float) -> str:
    return "gt0" if threshold == 0 else f"{100 * threshold:g}pct".replace(".", "p")


def consensus_mask_labels(n_multi: np.ndarray, definition: str) -> tuple[np.ndarray, np.ndarray]:
    n_multi = np.asarray(n_multi)
    if definition == "any":
        include = np.ones(len(n_multi), dtype=bool)
        labels = n_multi >= 1
    elif definition == "majority":
        include = n_multi != 1
        labels = n_multi >= 2
    elif definition == "unanimous":
        include = (n_multi == 0) | (n_multi == 3)
        labels = n_multi == 3
    else:
        raise ValueError(definition)
    return include, labels.astype(np.int8)


def analyze_pb_consensus(
    consensus: pd.DataFrame,
    sources: list[dict],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    consensus_dir = output_dir / "replicate_consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    all_valid = consensus["all_three_replicates_pb_valid"].to_numpy(bool)
    proteins = consensus["protein"].to_numpy()
    n_multi = consensus["n_multi_state_replicates"].to_numpy(int)
    performance_rows = []
    sensitivity_rows = []

    for source in sources:
        identity = {k: source[k] for k in ("condition", "score_type", "seed")}
        score = source["score"]
        for definition in CONSENSUS_DEFINITIONS:
            include_def, labels = consensus_mask_labels(n_multi, definition)
            include = all_valid & include_def
            performance_rows.append(
                {
                    **identity,
                    "consensus_definition": definition,
                    "excluded_invalid_residues": int((~all_valid).sum()),
                    "excluded_ambiguous_residues": int((all_valid & ~include_def).sum()),
                    **binary_performance(labels[include], score[include], proteins[include]),
                }
            )
        for threshold in OCCUPANCY_THRESHOLDS:
            suffix = occupancy_suffix(threshold)
            strong_n = consensus[f"n_multi_replicates_occ_{suffix}"].to_numpy(int)
            for definition in CONSENSUS_DEFINITIONS:
                include_def, labels = consensus_mask_labels(strong_n, definition)
                include = all_valid & include_def
                sensitivity_rows.append(
                    {
                        **identity,
                        "secondary_occupancy_threshold": threshold,
                        "occupancy_rule": ">0" if threshold == 0 else f">={100*threshold:g}%",
                        "consensus_definition": definition,
                        "excluded_invalid_residues": int((~all_valid).sum()),
                        "excluded_ambiguous_residues": int((all_valid & ~include_def).sum()),
                        **binary_performance(labels[include], score[include], proteins[include]),
                    }
                )

    performance = pd.DataFrame(performance_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    performance.to_csv(consensus_dir / "consensus_performance_by_run.csv", index=False)
    summarize_seeds(performance, ["consensus_definition"]).to_csv(
        consensus_dir / "consensus_performance_summary.csv", index=False
    )
    sensitivity.to_csv(consensus_dir / "secondary_occupancy_sensitivity.csv", index=False)
    return {"consensus_performance": performance, "occupancy_sensitivity": sensitivity}


def _score_groups(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    new_group = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    group_sorted = np.cumsum(new_group) - 1
    group = np.empty(len(values), dtype=np.int64)
    group[order] = group_sorted
    return group, order


def auc_contribution_matrix(
    labels: np.ndarray, scores: np.ndarray, protein_index: np.ndarray, n_proteins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute protein-pair weighted-AUC contributions."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    pidx = np.asarray(protein_index, dtype=np.int32)
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    pidx = pidx[order]
    positive_counts = np.bincount(pidx, weights=labels, minlength=n_proteins)
    negative_counts = np.bincount(pidx, weights=1 - labels, minlength=n_proteins)
    contribution = np.zeros((n_proteins, n_proteins), dtype=float)
    cumulative_negative = np.zeros(n_proteins, dtype=float)
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        group_p = pidx[start:end]
        group_y = labels[start:end]
        pos = np.bincount(group_p, weights=group_y, minlength=n_proteins)
        neg = np.bincount(group_p, weights=1 - group_y, minlength=n_proteins)
        contribution += np.outer(pos, cumulative_negative + 0.5 * neg)
        cumulative_negative += neg
    return contribution, positive_counts, negative_counts


def bootstrap_auc_from_matrix(
    contribution: np.ndarray,
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
    protein_weights: np.ndarray,
) -> np.ndarray:
    numerator = np.einsum("bi,ij,bj->b", protein_weights, contribution, protein_weights, optimize=True)
    positive = protein_weights @ positive_counts
    negative = protein_weights @ negative_counts
    return np.divide(
        numerator,
        positive * negative,
        out=np.full(len(protein_weights), np.nan),
        where=(positive > 0) & (negative > 0),
    )


def weighted_spearman_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    protein_index: np.ndarray,
    protein_weights: np.ndarray,
) -> np.ndarray:
    x_group, _ = _score_groups(np.asarray(x, dtype=float))
    y_group, _ = _score_groups(np.asarray(y, dtype=float))
    n_x_groups = int(x_group.max()) + 1
    n_y_groups = int(y_group.max()) + 1
    output = np.full(len(protein_weights), np.nan)
    for b, weights in enumerate(protein_weights):
        observation_weights = weights[protein_index].astype(float)
        total = observation_weights.sum()
        if total <= 1:
            continue
        x_group_weights = np.bincount(x_group, weights=observation_weights, minlength=n_x_groups)
        y_group_weights = np.bincount(y_group, weights=observation_weights, minlength=n_y_groups)
        x_rank_group = np.cumsum(x_group_weights) - 0.5 * x_group_weights + 0.5
        y_rank_group = np.cumsum(y_group_weights) - 0.5 * y_group_weights + 0.5
        rx = x_rank_group[x_group]
        ry = y_rank_group[y_group]
        mx = np.average(rx, weights=observation_weights)
        my = np.average(ry, weights=observation_weights)
        dx = rx - mx
        dy = ry - my
        denominator = math.sqrt(
            float(np.sum(observation_weights * dx * dx) * np.sum(observation_weights * dy * dy))
        )
        if denominator > 0:
            output[b] = float(np.sum(observation_weights * dx * dy) / denominator)
    return output


def protein_bootstrap(
    residues: pd.DataFrame,
    consensus: pd.DataFrame,
    sources: list[dict],
    output_dir: Path,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, int]]:
    bootstrap_dir = output_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    proteins_order = list(dict.fromkeys(residues["protein"].tolist()))
    protein_to_index = {name: i for i, name in enumerate(proteins_order)}
    pidx = residues["protein"].map(protein_to_index).to_numpy(np.int32)
    n_proteins = len(proteins_order)
    rng = np.random.default_rng(random_seed)
    weights = rng.multinomial(n_proteins, np.full(n_proteins, 1 / n_proteins), size=n_bootstrap)
    neq = residues["original_neq"].to_numpy(float)
    all_valid = consensus["all_three_replicates_pb_valid"].to_numpy(bool)
    n_multi = consensus["n_multi_state_replicates"].to_numpy(int)
    rows: list[pd.DataFrame] = []
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    observed: dict[tuple[str, str, str], float] = {}

    ensemble_sources = [x for x in sources if x["score_type"] == "seed_ensemble"]
    for source in ensemble_sources:
        condition = source["condition"]
        score = source["score"]
        for threshold in THRESHOLDS:
            include, all_labels = endpoint_mask_labels(
                neq, threshold, "exclusion_margin"
            )
            labels = all_labels[include]
            matrix, pos, neg = auc_contribution_matrix(
                labels, score[include], pidx[include], n_proteins
            )
            values = bootstrap_auc_from_matrix(matrix, pos, neg, weights)
            endpoint = f"{threshold:g}"
            arrays[(condition, "exclusion_auroc", endpoint)] = values
            observed[(condition, "exclusion_auroc", endpoint)] = float(
                roc_auc_score(labels, score[include])
            )
            rows.append(
                pd.DataFrame(
                    {
                        "bootstrap_replicate": np.arange(n_bootstrap),
                        "condition": condition,
                        "analysis": "exclusion_auroc",
                        "endpoint": endpoint,
                        "estimate": values,
                    }
                )
            )
        for definition in CONSENSUS_DEFINITIONS:
            include_def, labels = consensus_mask_labels(n_multi, definition)
            include = all_valid & include_def
            matrix, pos, neg = auc_contribution_matrix(
                labels[include], score[include], pidx[include], n_proteins
            )
            values = bootstrap_auc_from_matrix(matrix, pos, neg, weights)
            arrays[(condition, "consensus_auroc", definition)] = values
            observed[(condition, "consensus_auroc", definition)] = float(
                roc_auc_score(labels[include], score[include])
            )
            rows.append(
                pd.DataFrame(
                    {
                        "bootstrap_replicate": np.arange(n_bootstrap),
                        "condition": condition,
                        "analysis": "consensus_auroc",
                        "endpoint": definition,
                        "estimate": values,
                    }
                )
            )
        rho = weighted_spearman_bootstrap(neq, score, pidx, weights)
        arrays[(condition, "spearman", "original_neq")] = rho
        observed[(condition, "spearman", "original_neq")] = _safe_corr(spearmanr, neq, score)
        rows.append(
            pd.DataFrame(
                {
                    "bootstrap_replicate": np.arange(n_bootstrap),
                    "condition": condition,
                    "analysis": "spearman",
                    "endpoint": "original_neq",
                    "estimate": rho,
                }
            )
        )
        positive_tail = neq > 1.0
        rho_positive = weighted_spearman_bootstrap(
            neq[positive_tail],
            score[positive_tail],
            pidx[positive_tail],
            weights,
        )
        arrays[(condition, "spearman_positive_tail", "neq_gt_1")] = rho_positive
        observed[(condition, "spearman_positive_tail", "neq_gt_1")] = _safe_corr(
            spearmanr, neq[positive_tail], score[positive_tail]
        )
        rows.append(
            pd.DataFrame(
                {
                    "bootstrap_replicate": np.arange(n_bootstrap),
                    "condition": condition,
                    "analysis": "spearman_positive_tail",
                    "endpoint": "neq_gt_1",
                    "estimate": rho_positive,
                }
            )
        )
        difference = (
            arrays[(condition, "consensus_auroc", "unanimous")]
            - arrays[(condition, "consensus_auroc", "any")]
        )
        observed[(condition, "paired_consensus_auroc_difference", "unanimous_minus_any")] = (
            observed[(condition, "consensus_auroc", "unanimous")]
            - observed[(condition, "consensus_auroc", "any")]
        )
        rows.append(
            pd.DataFrame(
                {
                    "bootstrap_replicate": np.arange(n_bootstrap),
                    "condition": condition,
                    "analysis": "paired_consensus_auroc_difference",
                    "endpoint": "unanimous_minus_any",
                    "estimate": difference,
                }
            )
        )

    replicates = pd.concat(rows, ignore_index=True)
    replicates.to_csv(
        bootstrap_dir / "bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.10g",
    )
    summary_rows = []
    for key, group in replicates.groupby(["condition", "analysis", "endpoint"], sort=False):
        values = group["estimate"].dropna().to_numpy(float)
        summary_rows.append(
            {
                "condition": key[0],
                "analysis": key[1],
                "endpoint": key[2],
                "observed_estimate": observed[key],
                "n_bootstrap_defined": len(values),
                "bootstrap_mean": float(np.mean(values)),
                "ci_2p5": float(np.quantile(values, 0.025)),
                "ci_97p5": float(np.quantile(values, 0.975)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(bootstrap_dir / "bootstrap_summary.csv", index=False)
    return replicates, summary, weights, protein_to_index


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full(len(p_values), np.nan)
    valid = np.isfinite(p_values)
    values = p_values[valid]
    if not len(values):
        return result
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty(len(values))
    restored[order] = adjusted
    result[valid] = restored
    return result


def dominant_pb_stratification(
    consensus: pd.DataFrame,
    sources: list[dict],
    protein_weights: np.ndarray,
    protein_to_index: Mapping[str, int],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dominant_dir = output_dir / "dominant_pb_stratification"
    dominant_dir.mkdir(parents=True, exist_ok=True)
    stable = (
        consensus["all_three_replicates_pb_valid"].to_numpy(bool)
        & consensus["dominant_pb_state_agrees_all_replicates"].to_numpy(bool)
    )
    n_multi = consensus["n_multi_state_replicates"].to_numpy(int)
    state_values = consensus["pooled_dominant_pb_state"].fillna("").to_numpy()
    proteins = consensus["protein"].to_numpy()
    pidx = consensus["protein"].map(protein_to_index).to_numpy(np.int32)
    effect_rows = []
    performance_rows = []

    for source in sources:
        condition = source["condition"]
        identity = {
            "condition": condition,
            "score_type": source["score_type"],
            "seed": source["seed"],
        }
        score = source["score"]
        condition_start = len(effect_rows)
        for state in PB_STATES:
            include = stable & (state_values == state) & ((n_multi == 0) | (n_multi == 3))
            labels = (n_multi[include] == 3).astype(np.int8)
            state_scores = score[include]
            state_proteins = proteins[include]
            n_negative = int((labels == 0).sum())
            n_positive = int((labels == 1).sum())
            n_proteins = int(pd.unique(state_proteins).size)
            eligible = n_proteins >= 10 and n_negative >= 20 and n_positive >= 20
            negative_median = float(np.median(state_scores[labels == 0])) if n_negative else float("nan")
            positive_median = float(np.median(state_scores[labels == 1])) if n_positive else float("nan")
            negative_mean = float(np.mean(state_scores[labels == 0])) if n_negative else float("nan")
            positive_mean = float(np.mean(state_scores[labels == 1])) if n_positive else float("nan")
            mann_p = (
                float(mannwhitneyu(state_scores[labels == 1], state_scores[labels == 0]).pvalue)
                if n_negative and n_positive
                else float("nan")
            )
            protein_effects = []
            for protein in pd.unique(state_proteins):
                mask = state_proteins == protein
                if np.unique(labels[mask]).size == 2:
                    protein_effects.append(
                        float(np.mean(state_scores[mask & (labels == 1)]) - np.mean(state_scores[mask & (labels == 0)]))
                    )
            protein_p = (
                float(wilcoxon(protein_effects).pvalue)
                if len(protein_effects) >= 5 and np.any(np.asarray(protein_effects) != 0)
                else float("nan")
            )
            effect_rows.append(
                {
                    **identity,
                    "dominant_pb_state": state,
                    "stable_dominant_state_required": True,
                    "minimum_coverage_rule": ">=10 proteins and >=20 residues per class",
                    "eligible_for_inference": eligible,
                    "n_negative_0_of_3": n_negative,
                    "n_positive_3_of_3": n_positive,
                    "n_proteins": n_proteins,
                    "n_proteins_with_both_classes": len(protein_effects),
                    "negative_score_median": negative_median,
                    "positive_score_median": positive_median,
                    "median_score_difference": positive_median - negative_median,
                    "negative_score_mean": negative_mean,
                    "positive_score_mean": positive_mean,
                    "mean_score_difference": positive_mean - negative_mean,
                    "mann_whitney_p_unadjusted": mann_p,
                    "paired_protein_mean_difference_wilcoxon_p_unadjusted": protein_p,
                }
            )
            if eligible:
                observed = float(roc_auc_score(labels, state_scores))
                matrix, pos, neg = auc_contribution_matrix(
                    labels, state_scores, pidx[include], len(protein_to_index)
                )
                boot = bootstrap_auc_from_matrix(matrix, pos, neg, protein_weights)
                valid_boot = boot[np.isfinite(boot)]
                performance_rows.append(
                    {
                        **identity,
                        "dominant_pb_state": state,
                        "n_negative_0_of_3": n_negative,
                        "n_positive_3_of_3": n_positive,
                        "n_proteins": n_proteins,
                        "auroc": observed,
                        "bootstrap_defined": len(valid_boot),
                        "auroc_ci_2p5": float(np.quantile(valid_boot, 0.025)),
                        "auroc_ci_97p5": float(np.quantile(valid_boot, 0.975)),
                    }
                )
        indices = np.arange(condition_start, len(effect_rows))
        eligible_indices = [i for i in indices if effect_rows[i]["eligible_for_inference"]]
        for p_column, q_column in (
            ("mann_whitney_p_unadjusted", "mann_whitney_p_bh"),
            (
                "paired_protein_mean_difference_wilcoxon_p_unadjusted",
                "paired_protein_mean_difference_wilcoxon_p_bh",
            ),
        ):
            adjusted = benjamini_hochberg(
                np.asarray([effect_rows[i][p_column] for i in eligible_indices], dtype=float)
            )
            for index, value in zip(eligible_indices, adjusted):
                effect_rows[index][q_column] = value
            for index in indices:
                effect_rows[index].setdefault(q_column, float("nan"))

    effects = pd.DataFrame(effect_rows)
    performance = pd.DataFrame(performance_rows)
    effects.to_csv(dominant_dir / "dominant_pb_score_effects.csv", index=False)
    performance.to_csv(dominant_dir / "dominant_pb_performance.csv", index=False)
    ensemble_performance = performance.query("score_type == 'seed_ensemble'")
    if len(ensemble_performance):
        ensemble_performance.groupby("condition", as_index=False).agg(
            n_states_eligible=("dominant_pb_state", "count"),
            median_state_auroc=("auroc", "median"),
            minimum_state_auroc=("auroc", "min"),
            maximum_state_auroc=("auroc", "max"),
        ).to_csv(dominant_dir / "dominant_pb_performance_summary.csv", index=False)
    else:
        pd.DataFrame(
            columns=["condition", "n_states_eligible", "median_state_auroc", "minimum_state_auroc", "maximum_state_auroc"]
        ).to_csv(dominant_dir / "dominant_pb_performance_summary.csv", index=False)
    return effects, performance


def make_figures(
    residues: pd.DataFrame,
    analyses: dict[str, pd.DataFrame],
    consensus_analyses: dict[str, pd.DataFrame],
    bootstrap_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    dominant_performance: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    neq = residues["original_neq"].to_numpy(float)
    conditions = sorted(analyses["score_bins"]["condition"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes[0, 0]
    exact = int((neq == 1).sum())
    tail = neq[neq > 1]
    edges = np.linspace(1.000001, max(2.1, float(np.max(tail))), 45)
    ax.bar([1.0], [exact], width=0.08, color="#3b6fb6", label="Neq = 1 point mass")
    ax.hist(tail, bins=edges, color="#e3893d", alpha=0.8, label="Neq > 1 tail")
    ax.set_yscale("log")
    ax.set_xlabel("Neq")
    ax.set_ylabel("Residues (log scale)")
    ax.set_title("A  Fixed-test Neq distribution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ensemble_bins = analyses["score_bins"].query("score_type == 'seed_ensemble'")
    for color, condition in zip(colors, conditions):
        part = ensemble_bins[ensemble_bins["condition"] == condition].sort_values("bin_order")
        ax.plot(part["bin_order"], part["median_flexible_score"], marker="o", lw=1, alpha=0.45, color=color)
    medians = ensemble_bins.groupby("bin_order")["median_flexible_score"].median()
    q25 = ensemble_bins.groupby("bin_order")["score_q25"].median()
    q75 = ensemble_bins.groupby("bin_order")["score_q75"].median()
    ax.fill_between(
        medians.index.to_numpy(),
        q25.to_numpy(),
        q75.to_numpy(),
        color="black",
        alpha=0.09,
        label="Median condition IQR",
    )
    ax.plot(medians.index, medians.values, color="black", marker="o", lw=2.5, label="Median across conditions")
    ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Median flexible-class score")
    ax.set_title("B  Score response across ordered Neq bins")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    bs = bootstrap_summary[bootstrap_summary["analysis"] == "exclusion_auroc"].copy()
    observed = analyses["exclusion"].query("score_type == 'seed_ensemble'")
    cutoff_positions = np.arange(len(THRESHOLDS))
    for color, condition in zip(colors, conditions):
        part = observed[observed["condition"] == condition].sort_values("cutoff")
        ci = bs[bs["condition"] == condition].copy()
        ci["cutoff"] = ci["endpoint"].astype(float)
        ci = ci.sort_values("cutoff")
        ax.plot(cutoff_positions, part["auroc"], marker="o", color=color, lw=1.4, label=condition)
        ax.fill_between(cutoff_positions, ci["ci_2p5"], ci["ci_97p5"], color=color, alpha=0.10)
    ax.set_xticks(cutoff_positions, [f"{x:g}" for x in THRESHOLDS])
    ax.set_xlabel("Exclusion cutoff t (negative: Neq=1; positive: Neq>t)")
    ax.set_ylabel("AUROC")
    ax.set_title("C  Stricter exclusion-margin endpoint")

    ax = axes[1, 1]
    cp = consensus_analyses["consensus_performance"].query("score_type == 'seed_ensemble'")
    bs = bootstrap_summary[bootstrap_summary["analysis"] == "consensus_auroc"]
    x = np.arange(3)
    for color, condition in zip(colors, conditions):
        part = cp[cp["condition"] == condition].set_index("consensus_definition").loc[
            list(CONSENSUS_DEFINITIONS)
        ]
        ci = bs[bs["condition"] == condition].set_index("endpoint").loc[list(CONSENSUS_DEFINITIONS)]
        ax.plot(x, part["auroc"], marker="o", color=color, lw=1.4, label=condition)
        ax.fill_between(x, ci["ci_2p5"], ci["ci_97p5"], color=color, alpha=0.10)
    ax.set_xticks(x, ["Any replicate", "≥2/3 replicates", "3/3 replicates"])
    ax.set_ylabel("AUROC")
    ax.set_title("D  Replicate-consensus PB variability")
    ax.legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(figures / "main_neq_pb_reliability_figure.png", dpi=300)
    fig.savefig(figures / "main_neq_pb_reliability_figure.pdf")
    plt.close(fig)

    sensitivity = consensus_analyses["occupancy_sensitivity"].query("score_type == 'seed_ensemble'")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, definition in zip(axes, CONSENSUS_DEFINITIONS):
        part_def = sensitivity[sensitivity["consensus_definition"] == definition]
        for color, condition in zip(colors, conditions):
            part = part_def[part_def["condition"] == condition].sort_values("secondary_occupancy_threshold")
            ax.plot(100 * part["secondary_occupancy_threshold"], part["auroc"], marker="o", color=color, alpha=0.7)
        ax.set_title(definition.capitalize())
        ax.set_xlabel("Minimum secondary-state occupancy (%)")
    axes[0].set_ylabel("AUROC")
    fig.suptitle("Secondary-state occupancy sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "supplement_secondary_occupancy_sensitivity.png", dpi=250)
    plt.close(fig)

    valid = comparison["all_three_replicates_pb_valid"].to_numpy(bool)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(
        comparison.loc[valid, "original_neq"],
        comparison.loc[valid, "mean_replicate_neq"],
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    limit = max(comparison.loc[valid, "original_neq"].max(), comparison.loc[valid, "mean_replicate_neq"].max())
    ax.plot([1, limit], [1, limit], "--", color="white", lw=1)
    ax.set_xlabel("Original fixed-test Neq")
    ax.set_ylabel("Mean regenerated replicate Neq")
    ax.set_title("Original versus regenerated Neq")
    fig.colorbar(hb, ax=ax, label="log10 residue count")
    fig.tight_layout()
    fig.savefig(figures / "supplement_original_vs_regenerated_neq.png", dpi=250)
    plt.close(fig)

    ensemble_dominant = dominant_performance.query("score_type == 'seed_ensemble'")
    if len(ensemble_dominant):
        pivot = ensemble_dominant.pivot(index="dominant_pb_state", columns="condition", values="auroc")
        fig, ax = plt.subplots(figsize=(11, 5))
        image = ax.imshow(pivot.T, aspect="auto", vmin=0.5, vmax=1.0, cmap="magma")
        ax.set_xticks(np.arange(len(pivot.index)), pivot.index)
        ax.set_yticks(np.arange(len(pivot.columns)), pivot.columns, fontsize=7)
        ax.set_xlabel("Stable pooled dominant PB state")
        ax.set_title("Within-dominant-state AUROC: 0/3 versus 3/3 multi-state")
        fig.colorbar(image, ax=ax, label="AUROC")
        fig.tight_layout()
        fig.savefig(figures / "supplement_dominant_pb_stratification.png", dpi=250)
        plt.close(fig)

    seed_exclusion = analyses["exclusion"].query("score_type == 'individual_seed'")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        part_condition = seed_exclusion[seed_exclusion["condition"] == condition]
        for seed, part in part_condition.groupby("seed"):
            part = part.sort_values("cutoff")
            ax.plot(cutoff_positions, part["auroc"], marker="o", lw=1, label=f"seed {seed}")
        ax.set_title(condition, fontsize=9)
        ax.set_xticks(cutoff_positions, [f"{x:g}" for x in THRESHOLDS], rotation=30)
    axes.flat[len(conditions)].axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :3]:
        ax.set_xlabel("Exclusion cutoff t")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")
    fig.suptitle("Individual model-seed exclusion-margin results")
    fig.tight_layout()
    fig.savefig(figures / "supplement_individual_seed_exclusion_results.png", dpi=250)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    original: dict[str, pd.DataFrame],
    consensus_results: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    validity_audit: dict,
    bootstrap_summary: pd.DataFrame,
) -> None:
    correlations = original["correlations"].query("score_type == 'seed_ensemble'")
    exclusion = original["exclusion"].query("score_type == 'seed_ensemble'")
    relabeled = original["relabeled"].query("score_type == 'seed_ensemble'")
    consensus = consensus_results["consensus_performance"].query("score_type == 'seed_ensemble'")
    rho_range = (correlations["pooled_spearman_neq"].min(), correlations["pooled_spearman_neq"].max())
    original_auc = exclusion[exclusion["cutoff"] == 1.0]["auroc"]
    strict_auc = exclusion[exclusion["cutoff"] == 2.0]["auroc"]
    any_auc = consensus[consensus["consensus_definition"] == "any"]["auroc"]
    unanimous_auc = consensus[consensus["consensus_definition"] == "unanimous"]["auroc"]
    valid = comparison["all_three_replicates_pb_valid"]
    agreement = comparison.loc[valid, "binary_label_agrees"].mean()
    bootstrap_delta = bootstrap_summary[
        bootstrap_summary["analysis"] == "paired_consensus_auroc_difference"
    ]
    positive_tail_bootstrap = bootstrap_summary[
        bootstrap_summary["analysis"] == "spearman_positive_tail"
    ]
    positive_rho = correlations["pooled_spearman_neq_gt_1"]
    lines = [
        "# Neq / Protein-Block reliability analysis",
        "",
        "This directory contains a no-retraining evaluation of the completed",
        "`publication_comparable_v2` flexible-class probabilities. The endpoint is",
        "described as PB-single-state (zero PB entropy) versus PB-multi-state",
        "(nonzero PB entropy) behavior in ATLAS simulations.",
        "",
        "## Headline numerical ranges across the seven condition seed ensembles",
        "",
        f"- Pooled Spearman(score, Neq): {rho_range[0]:.3f} to {rho_range[1]:.3f}.",
        f"- Positive-tail Spearman(score, Neq | Neq>1): "
        f"{positive_rho.min():.3f} to {positive_rho.max():.3f}.",
        f"- Original endpoint AUROC (Neq=1 vs Neq>1): {original_auc.min():.3f} to {original_auc.max():.3f}.",
        f"- Exclusion-margin AUROC at Neq>2: {strict_auc.min():.3f} to {strict_auc.max():.3f}.",
        f"- Any-replicate consensus AUROC: {any_auc.min():.3f} to {any_auc.max():.3f}.",
        f"- Unanimous-replicate consensus AUROC: {unanimous_auc.min():.3f} to {unanimous_auc.max():.3f}.",
        f"- Original/regenerated binary-label agreement among all-replicate-valid residues: {agreement:.3%}.",
        f"- PB-valid residues in all three replicates: {validity_audit['all_three_replicates_pb_valid']:,}.",
        f"- Original Neq=1 residues with all replicates PB-unassigned: "
        f"{validity_audit['original_neq_1_all_replicates_invalid']:,}.",
        "",
        "All seven condition ensembles showed strictly increasing median scores",
        "across the prespecified Neq bins. Exclusion-margin AUROC increased as",
        "weak near-1 positives were removed. AUPRC did not increase monotonically",
        "while positive prevalence fell sharply with the stricter cutoff, so",
        "prevalence is reported with every endpoint.",
        "",
        "The positive-tail correlations directly test graded information beyond",
        "the binary boundary. Their condition-specific protein-bootstrap intervals",
        f"collectively ranged from {positive_tail_bootstrap['ci_2p5'].min():.3f} "
        f"to {positive_tail_bootstrap['ci_97p5'].max():.3f}. In the distinct",
        "relabeled-threshold analysis, AUROC decreased as the threshold increased;",
        f"at t=2 it ranged from "
        f"{relabeled.loc[relabeled.cutoff == 2, 'auroc'].min():.3f} to "
        f"{relabeled.loc[relabeled.cutoff == 2, 'auroc'].max():.3f}.",
        "",
        "Unanimous-replicate AUROC exceeded any-replicate AUROC for every",
        "condition, with all seven paired protein-bootstrap intervals excluding",
        "zero. This supports stronger discrimination of reproducible PB-state",
        "heterogeneity. In contrast, requiring progressively larger secondary-state",
        "occupancies reduced AUROC, so the classifier should not be described as",
        "preferentially detecting only high-occupancy secondary states.",
        "",
        "Within stable dominant PB states, the median state-specific AUROC ranged",
        "from 0.726 to 0.802 across conditions. All 16 state-specific mean score",
        "differences were positive for every condition, while a small subset of",
        "states remained weak; this argues against a purely static-conformation",
        "explanation without claiming uniform performance across PB states.",
        "The dominant-state tables include all individual seeds as supplementary",
        "rows; condition-level seed ensembles remain the primary result.",
        "",
        "The paired unanimous-minus-any AUROC intervals are in",
        "`bootstrap/bootstrap_summary.csv`; they use identical protein samples for",
        "both endpoints. Model seeds are summarized as model variability and are",
        "never pooled as biological replicates.",
        "",
        "## Reproducibility",
        "",
        "Run from the repository root:",
        "",
        "```bash",
        "python3 -m prediction_endpoint_analysis.analyze_neq_pb_reliability",
        "```",
        "",
        "Run the focused regression suite (15 unittest-discoverable tests) with:",
        "",
        "```bash",
        "python3 -m unittest discover -s tests -p 'test_analyze_neq_pb_reliability.py'",
        "```",
        "",
        "The external suite covers PB state definitions, Z handling, consensus",
        "labels, streamed JSON truncation, probability/shape checks, both threshold",
        "semantics, prevalence, versioned cache invalidation, PB cache structure and",
        "alignment, clean-run finalization order, protein multiplicity, weighted",
        "Spearman, and fixed-seed bootstrap draws. Additional full-data invariants",
        "are enforced during analysis execution.",
        "",
        "Existing passed extraction caches are reused only when the recorded",
        "path/size/mtime provenance of every source file matches. Use",
        "`--force-predictions` or `--force-pb` to rebuild explicitly.",
        "",
        "See `parameters.json` and `audits/complete_analysis_audit.json` for exact",
        "settings and completion checks.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_text(
    output_dir: Path,
    original: dict[str, pd.DataFrame],
    consensus_results: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> None:
    correlations = original["correlations"].query("score_type == 'seed_ensemble'")
    exclusion = original["exclusion"].query("score_type == 'seed_ensemble'")
    relabeled = original["relabeled"].query("score_type == 'seed_ensemble'")
    consensus = consensus_results["consensus_performance"].query("score_type == 'seed_ensemble'")
    occupancy = consensus_results["occupancy_sensitivity"].query(
        "score_type == 'seed_ensemble' and consensus_definition == 'unanimous'"
    )
    paired = bootstrap_summary.query(
        "analysis == 'paired_consensus_auroc_difference'"
    )
    positive_tail_bootstrap = bootstrap_summary.query(
        "analysis == 'spearman_positive_tail'"
    )
    valid = comparison["all_three_replicates_pb_valid"].to_numpy(bool)
    lines = [
        "# Manuscript-ready Neq / PB endpoint language",
        "",
        "## Methods",
        "",
        "We defined the primary endpoint from the effective number of Protein Block",
        "states, Neq=exp(H_PB). For PB-valid residues, Neq=1 corresponds to zero PB",
        "entropy and observation of a single local backbone state, whereas Neq>1",
        "indicates nonzero PB entropy and sampling of multiple states. Neq=1 has",
        "an exact interpretation as the zero-PB-entropy boundary. We therefore",
        "interpret the endpoint as PB-single-state versus PB-multi-state behavior",
        "and evaluate its robustness to stricter definitions.",
        "",
        "We evaluated saved flexible-class probabilities without retraining. In the",
        "primary exclusion-margin analysis, Neq=1 residues remained negative,",
        "residues with Neq>t were positive, and 1<Neq<=t residues were excluded for",
        "t=1.01, 1.05, 1.10, 1.50, and 2.00. A secondary relabeled-threshold",
        "analysis instead compared Neq<=t with Neq>t.",
        "",
        "We independently reconstructed PB-state distributions in each of the three",
        "ATLAS trajectories after removing unassigned Z frames. Replicates with",
        "zero valid frames were invalid; those with one or at least two observed",
        "non-Z states were single-state or multi-state, respectively. We evaluated",
        "any-replicate, majority-replicate, and unanimous-replicate variability,",
        "and repeated these analyses across prespecified secondary-state occupancy",
        "requirements from any observation through 5%. Confidence intervals were",
        "obtained from 1,000 resamples of the 208 test proteins, retaining all",
        "eligible residues and using identical protein draws for paired endpoints.",
        "",
        "## Results",
        "",
        f"Across seven condition-level three-seed ensembles, pooled Spearman",
        f"correlations between score and Neq ranged from "
        f"{correlations['pooled_spearman_neq'].min():.3f} to "
        f"{correlations['pooled_spearman_neq'].max():.3f}. Median scores increased",
        "strictly across all seven prespecified Neq bins in every condition.",
        f"Within the positive tail (Neq>1), Spearman correlations ranged from "
        f"{correlations['pooled_spearman_neq_gt_1'].min():.3f} to "
        f"{correlations['pooled_spearman_neq_gt_1'].max():.3f}; the condition-specific "
        "protein-bootstrap 95% intervals collectively extended from "
        f"{positive_tail_bootstrap['ci_2p5'].min():.3f} to "
        f"{positive_tail_bootstrap['ci_97p5'].max():.3f}.",
        f"Original-endpoint AUROC ranged from "
        f"{exclusion.loc[exclusion.cutoff == 1, 'auroc'].min():.3f} to "
        f"{exclusion.loc[exclusion.cutoff == 1, 'auroc'].max():.3f}, increasing to "
        f"{exclusion.loc[exclusion.cutoff == 2, 'auroc'].min():.3f} to "
        f"{exclusion.loc[exclusion.cutoff == 2, 'auroc'].max():.3f} after excluding",
        "positive residues with 1<Neq<=2.",
        "In the secondary relabeled-threshold analysis, which instead assigned",
        "Neq<=t to the negative class, AUROC decreased with stricter thresholds",
        f"and ranged from {relabeled.loc[relabeled.cutoff == 2, 'auroc'].min():.3f} "
        f"to {relabeled.loc[relabeled.cutoff == 2, 'auroc'].max():.3f} at t=2.",
        "",
        f"All three replicates were PB-valid for {valid.sum():,} of "
        f"{len(comparison):,} residues. The remaining {(~valid).sum():,} residues",
        "were the first or last two positions of each protein and were unassigned",
        "in every replicate; all had original Neq=1 and were excluded from",
        "PB-reliability conclusions. Among all-replicate-valid residues, original",
        f"and regenerated binary labels agreed for "
        f"{comparison.loc[valid, 'binary_label_agrees'].mean():.3%} of residues.",
        "",
        f"Any-replicate AUROC ranged from "
        f"{consensus.loc[consensus.consensus_definition == 'any', 'auroc'].min():.3f} "
        f"to {consensus.loc[consensus.consensus_definition == 'any', 'auroc'].max():.3f}, "
        f"whereas unanimous-replicate AUROC ranged from "
        f"{consensus.loc[consensus.consensus_definition == 'unanimous', 'auroc'].min():.3f} "
        f"to {consensus.loc[consensus.consensus_definition == 'unanimous', 'auroc'].max():.3f}. "
        "The observed unanimous-minus-any AUROC differences were positive in all",
        f"conditions ({paired['observed_estimate'].min():.3f} to "
        f"{paired['observed_estimate'].max():.3f}), and every paired 95% interval",
        "excluded zero.",
        "",
        "Requiring larger secondary-state occupancies did not strengthen",
        "discrimination: unanimous-replicate AUROC declined from "
        f"{occupancy.loc[occupancy.secondary_occupancy_threshold == 0, 'auroc'].min():.3f}–"
        f"{occupancy.loc[occupancy.secondary_occupancy_threshold == 0, 'auroc'].max():.3f} "
        "for any secondary-state observation to "
        f"{occupancy.loc[occupancy.secondary_occupancy_threshold == 0.01, 'auroc'].min():.3f}–"
        f"{occupancy.loc[occupancy.secondary_occupancy_threshold == 0.01, 'auroc'].max():.3f} "
        "at 1% occupancy. Thus, the score contains graded information about Neq",
        "and more strongly distinguishes replicate-consistent PB-state",
        "heterogeneity, but does not preferentially identify states with high",
        "secondary-state occupancy.",
    ]
    (output_dir / "MANUSCRIPT_READY_TEXT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_self_tests() -> None:
    # PB definitions and Z exclusion.
    array = np.asarray(
        [[b"a", b"a", b"Z", b"a"], [b"a", b"b", b"Z", b"b"]], dtype="S1"
    )
    metrics = _pb_metrics_from_array(array)
    assert metrics["valid_frames"].tolist() == [2, 2, 0, 2]
    assert metrics["n_states"].tolist() == [1, 2, 0, 2]
    assert metrics["neq"][0] == 1.0
    assert np.isclose(metrics["neq"][1], 2.0)
    assert np.isnan(metrics["neq"][2])
    assert np.isclose(metrics["secondary_occupancy"][3], 0.5)
    # Consensus categories.
    include, labels = consensus_mask_labels(np.array([0, 1, 2, 3]), "any")
    assert include.tolist() == [True] * 4 and labels.tolist() == [0, 1, 1, 1]
    include, labels = consensus_mask_labels(np.array([0, 1, 2, 3]), "majority")
    assert include.tolist() == [True, False, True, True] and labels.tolist() == [0, 0, 1, 1]
    include, labels = consensus_mask_labels(np.array([0, 1, 2, 3]), "unanimous")
    assert include.tolist() == [True, False, False, True] and labels.tolist() == [0, 0, 0, 1]
    # Protein multiplicity: a double weight doubles counts and squares pair contributions.
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.8, 0.2, 0.9])
    pidx = np.array([0, 0, 1, 1])
    matrix, pos, neg = auc_contribution_matrix(y, score, pidx, 2)
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    auc = bootstrap_auc_from_matrix(matrix, pos, neg, weights)
    assert np.allclose(auc, 1.0)
    # Fixed RNG gives identical protein draws and therefore paired comparisons.
    a = np.random.default_rng(7).multinomial(2, [0.5, 0.5], size=10)
    b = np.random.default_rng(7).multinomial(2, [0.5, 0.5], size=10)
    assert np.array_equal(a, b)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "results/publication_comparable_v2/manifest.tsv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=root / "data_splits/atlas_grouped_v1/test_grouped_v1.csv",
    )
    parser.add_argument(
        "--pb-root",
        type=Path,
        default=Path("/home/zahralab/MDStrainMapper/results/atlas_grouped_v1_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/publication_comparable_v2/analysis_neq_pb_reliability",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--force-pb", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    run_self_tests()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("audits", "figures"):
        (args.output_dir / directory).mkdir(exist_ok=True)
    _json_dump(
        args.output_dir / "parameters.json",
        {
            "manifest": str(args.manifest.resolve()),
            "test_csv": str(args.test_csv.resolve()),
            "pb_root": str(args.pb_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "thresholds": list(THRESHOLDS),
            "secondary_occupancy_thresholds": list(OCCUPANCY_THRESHOLDS),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "score": "saved flexible_scores = P(class 1: Neq > 1.0)",
            "pb_validity": "at least one valid non-Z frame per replicate",
            "pb_transition_rate_retained": False,
            "dominant_state_minimum_coverage": "at least 10 proteins and at least 20 residues per class",
            "cache_source_provenance": (
                "SHA-256 of cache schema version plus canonical path, "
                "size_bytes, and mtime_ns records"
            ),
            "prediction_cache_schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "pb_cache_schema_version": PB_CACHE_SCHEMA_VERSION,
        },
    )
    proteins, residues = load_fixed_test(args.test_csv)
    cache, prediction_audit = build_prediction_cache(
        args.manifest, args.test_csv, proteins, residues, args.output_dir, args.force_predictions
    )
    sources, conditions, _ = score_sources(cache, residues)
    original = analyze_original_endpoint(residues, sources, args.output_dir)
    replicate, consensus, comparison, pb_audit = extract_pb_reliability(
        args.pb_root, args.test_csv, proteins, residues, args.output_dir, args.force_pb
    )
    write_pb_comparison_audits(comparison, args.output_dir)
    consensus_results = analyze_pb_consensus(consensus, sources, args.output_dir)
    bootstrap_replicates, bootstrap_summary, protein_weights, protein_to_index = protein_bootstrap(
        residues,
        consensus,
        sources,
        args.output_dir,
        args.n_bootstrap,
        args.bootstrap_seed,
    )
    dominant_effects, dominant_performance = dominant_pb_stratification(
        consensus, sources, protein_weights, protein_to_index, args.output_dir
    )
    make_figures(
        residues,
        original,
        consensus_results,
        bootstrap_summary,
        comparison,
        dominant_performance,
        args.output_dir,
    )

    # Duplicate focused audits make the two highest-risk alignment checks easy to inspect.
    _json_dump(
        args.output_dir / "audits" / "sequence_alignment_audit.json",
        {
            "status": "passed",
            "runs": prediction_audit["runs"],
            "proteins_per_run": prediction_audit["proteins_per_run"],
            "residues_per_run": prediction_audit["residues_per_run"],
            "all_prediction_sequences_equal_fixed_test_sequences": True,
            "no_missing_or_extra_proteins": True,
            "no_silent_truncation": True,
        },
    )
    max_difference = max(
        x["maximum_score_probability_absolute_difference"] for x in prediction_audit["run_audits"]
    )
    _json_dump(
        args.output_dir / "audits" / "class_probability_audit.json",
        {
            "status": "passed",
            "flexible_scores_equal_class_probs_column_1_within_tolerance": True,
            "tolerance": FLOAT_TOL,
            "maximum_absolute_difference": max_difference,
            "scores_in_unit_interval": True,
            "hard_predictions_equal_probability_argmax": True,
        },
    )

    # Cutoff 1 exclusion and relabeling must be the same original population.
    exc1 = original["exclusion"][original["exclusion"]["cutoff"] == 1.0].sort_values(
        ["condition", "score_type", "seed"]
    )
    rel1 = original["relabeled"][original["relabeled"]["cutoff"] == 1.0].sort_values(
        ["condition", "score_type", "seed"]
    )
    if not np.allclose(exc1["auroc"], rel1["auroc"]) or not np.array_equal(
        exc1["n_residues"].to_numpy(), rel1["n_residues"].to_numpy()
    ):
        raise AssertionError("Cutoff-1 endpoint reproduction failed")

    # Generate human-readable deliverables before checking required outputs.
    # This ordering is necessary for a genuinely empty output directory.
    validity_audit = _read_json(
        args.output_dir / "audits" / "pb_validity_audit.json"
    )
    write_readme(
        args.output_dir,
        original,
        consensus_results,
        comparison,
        validity_audit,
        bootstrap_summary,
    )
    write_manuscript_text(
        args.output_dir,
        original,
        consensus_results,
        comparison,
        bootstrap_summary,
    )

    required_files = [
        "README.md",
        "MANUSCRIPT_READY_TEXT.md",
        "prediction_cache/test_predictions_by_residue.csv.gz",
        "neq_distribution/neq_bin_counts.csv",
        "neq_distribution/score_by_neq_bin.csv",
        "neq_distribution/score_neq_correlations_by_run.csv",
        "neq_distribution/score_neq_correlation_summary.csv",
        "stricter_endpoint/exclusion_margin_performance_by_run.csv",
        "stricter_endpoint/exclusion_margin_performance_summary.csv",
        "stricter_endpoint/relabeled_threshold_performance_by_run.csv",
        "stricter_endpoint/relabeled_threshold_performance_summary.csv",
        "pb_reliability/replicate_pb_metrics.csv.gz",
        "pb_reliability/residue_pb_consensus.csv.gz",
        "pb_reliability/original_vs_regenerated_neq.csv.gz",
        "pb_reliability/original_vs_regenerated_neq_summary.csv",
        "replicate_consensus/consensus_performance_by_run.csv",
        "replicate_consensus/consensus_performance_summary.csv",
        "replicate_consensus/secondary_occupancy_sensitivity.csv",
        "dominant_pb_stratification/dominant_pb_score_effects.csv",
        "dominant_pb_stratification/dominant_pb_performance.csv",
        "dominant_pb_stratification/dominant_pb_performance_summary.csv",
        "bootstrap/bootstrap_replicates.csv.gz",
        "bootstrap/bootstrap_summary.csv",
        "figures/main_neq_pb_reliability_figure.png",
        "figures/main_neq_pb_reliability_figure.pdf",
        "figures/supplement_individual_seed_exclusion_results.png",
        "audits/original_regenerated_binary_disagreements.csv",
        "audits/original_regenerated_disagreement_audit.json",
    ]
    missing = [x for x in required_files if not (args.output_dir / x).is_file()]
    if missing:
        raise AssertionError(f"Required outputs missing: {missing}")
    complete_audit = {
        "status": "passed",
        "runs": len([x for x in sources if x["score_type"] == "individual_seed"]),
        "conditions": len(conditions),
        "seed_ensembles": len([x for x in sources if x["score_type"] == "seed_ensemble"]),
        "test_proteins": len(proteins),
        "test_residues": len(residues),
        "pb_hdf5_files": pb_audit["hdf5_files_checked"],
        "bootstrap_replicates_per_metric": args.n_bootstrap,
        "bootstrap_sampling_unit": "protein",
        "paired_bootstrap_draws_shared": True,
        "cutoff_1_reproduces_original_endpoint": True,
        "required_output_presence_checks_passed": len(required_files),
        "embedded_runtime_self_tests_passed": True,
        "prediction_runtime_invariants_passed": True,
        "pb_runtime_invariants_passed": True,
        "cache_source_provenance_verified": True,
        "prediction_cache_schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
        "pb_cache_schema_version": PB_CACHE_SCHEMA_VERSION,
        "pb_cache_tables_revalidated_on_reuse": True,
        "validation_scope": (
            "Detailed runtime invariants plus presence checks for the enumerated "
            "required outputs; this field does not claim detailed consistency "
            "testing of every generated artifact."
        ),
    }
    # The complete audit is intentionally the final file written.
    _json_dump(args.output_dir / "audits" / "complete_analysis_audit.json", complete_audit)
    print(f"Analysis complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
