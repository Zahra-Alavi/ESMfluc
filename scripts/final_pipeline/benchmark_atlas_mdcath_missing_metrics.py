#!/usr/bin/env python3
"""Complete the missing ATLAS/mdCATH metrics for the seven ESMfluc models.

Outputs seed-specific and probability-averaged ATLAS-test Spearman metrics,
plus seed-specific and probability-averaged strict-mdCATH Neq AUROC metrics.
External mdCATH predictors are unseeded and are evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from benchmark_unified_mdcath import (
    DEFAULT_RUNS_DIR,
    DEFAULT_SOURCE_DIR,
    ESM_BILSTM_CONDITIONS,
    ESM_DISPLAY,
    LINEAR_CONDITION,
    _load_score_json,
    _parse_fasta,
    _portable_path,
    load_clusters,
    load_cohort,
    load_methods,
    load_verified_linear_model,
    shared_backbone_linear_probabilities_by_head,
    tokenize_linear_sequences,
    validate_method_coverage,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ATLAS_CACHE = (
    SCRIPT_DIR / "results" / "publication_comparable_v2"
    / "analysis_neq_pb_reliability" / "prediction_cache"
    / "test_predictions_by_residue.csv.gz"
)
DEFAULT_ATLAS_SPLIT_MANIFEST = (
    SCRIPT_DIR / "data_splits" / "atlas_grouped_v1"
    / "split_manifest_grouped_v1.csv"
)
DEFAULT_NETSURFP = (
    SCRIPT_DIR / "results" / "benchmark_unified_mdcath"
    / "NetSurfP_mdcath_strict.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "benchmark_metric_completion"
ESM_CONDITIONS = (*ESM_BILSTM_CONDITIONS, LINEAR_CONDITION)
CACHE_SCHEMA = "mdcath-seed-score-cache-v1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ATLAS Spearman and mdCATH AUROC, per seed and seed averaged",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--atlas-cache", type=Path, default=DEFAULT_ATLAS_CACHE)
    ap.add_argument(
        "--atlas-split-manifest", type=Path,
        default=DEFAULT_ATLAS_SPLIT_MANIFEST,
    )
    ap.add_argument("--mdcath-source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--esm-runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--netsurfp", type=Path, default=DEFAULT_NETSURFP)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--linear-batch-size", type=int, default=4)
    ap.add_argument("--force-seed-cache", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_stat(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": _portable_path(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_json(path: Path, payload: object, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as fh:
        json.dump(payload, fh, indent=indent)
    temporary.replace(path)


def safe_spearman(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(target) & np.isfinite(score)
    target, score = target[finite], score[finite]
    if len(target) < 3 or np.unique(target).size < 2 or np.unique(score).size < 2:
        return float("nan")
    return float(spearmanr(target, score).statistic)


def safe_auroc(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, score))


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    if len(values) == 0:
        return float("nan"), float("nan")
    unique = np.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.mean(np.concatenate([by_group[group] for group in sampled]))
    low, high = np.percentile(draws, (2.5, 97.5))
    return float(low), float(high)


def summarize_protein_metric(
    per_protein: pd.DataFrame,
    pooled_target: np.ndarray,
    pooled_score: np.ndarray,
    metric: str,
    n_bootstrap: int,
    rng_seed: int,
) -> dict:
    values = per_protein[metric].to_numpy(float)
    finite = np.isfinite(values)
    low, high = cluster_bootstrap_mean(
        values,
        per_protein["cluster"].to_numpy(),
        n_bootstrap,
        np.random.default_rng(rng_seed),
    )
    pooled = (
        safe_spearman(pooled_target, pooled_score)
        if metric == "spearman"
        else safe_auroc(pooled_target, pooled_score)
    )
    return {
        "n_proteins_total": len(per_protein),
        "n_proteins_valid": int(finite.sum()),
        "n_proteins_undefined": int((~finite).sum()),
        "n_clusters_total": int(per_protein["cluster"].nunique()),
        "n_clusters_valid": int(per_protein.loc[finite, "cluster"].nunique()),
        "n_residues": int(per_protein["n_residues"].sum()),
        f"macro_mean_{metric}": float(np.mean(values[finite])),
        f"macro_median_{metric}": float(np.median(values[finite])),
        f"macro_mean_{metric}_ci_low": low,
        f"macro_mean_{metric}_ci_high": high,
        f"pooled_{metric}": pooled,
    }


def validate_atlas_cache(frame: pd.DataFrame) -> dict:
    required = {
        "condition", "seed", "protein", "residue_index_0based",
        "amino_acid", "sequence_length", "original_neq",
        "original_binary_label", "flexible_score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ATLAS prediction cache missing columns: {sorted(missing)}")
    frame_conditions = set(frame["condition"])
    if frame_conditions != set(ESM_CONDITIONS):
        raise ValueError(
            f"ATLAS conditions mismatch: {sorted(frame_conditions)}"
        )
    if set(frame["seed"].unique()) != {1, 2, 3}:
        raise ValueError("ATLAS cache must contain seeds 1, 2, and 3")
    if frame.duplicated(
        ["condition", "seed", "protein", "residue_index_0based"]
    ).any():
        raise ValueError("ATLAS cache contains duplicate residue coordinates")
    if not np.isfinite(frame[["original_neq", "flexible_score"]].to_numpy()).all():
        raise ValueError("ATLAS cache contains nonfinite Neq or scores")
    if not frame["flexible_score"].between(0.0, 1.0).all():
        raise ValueError("ATLAS flexible scores are outside [0, 1]")
    run_counts = frame.groupby(["condition", "seed"]).agg(
        proteins=("protein", "nunique"),
        residues=("protein", "size"),
    )
    if not (run_counts["proteins"] == 208).all() or not (
        run_counts["residues"] == 47751
    ).all():
        raise ValueError(f"ATLAS run coverage mismatch:\n{run_counts}")
    labels = (frame["original_neq"].to_numpy(float) > 1.0).astype(int)
    if not np.array_equal(labels, frame["original_binary_label"].to_numpy(int)):
        raise ValueError("ATLAS binary labels do not equal Neq > 1.0")
    return {
        "conditions": len(frame_conditions),
        "seeds": [1, 2, 3],
        "runs": len(run_counts),
        "proteins_per_run": 208,
        "residues_per_run": 47751,
        "coordinate_duplicates": 0,
        "scores_in_unit_interval": True,
        "binary_definition": "flexible iff Neq > 1.0",
    }


def atlas_seed_average(condition_frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    keys = ["protein", "residue_index_0based"]
    counts = condition_frame.groupby(keys)["seed"].nunique()
    expected_coordinates = len(condition_frame) // 3
    if len(condition_frame) % 3 or len(counts) != expected_coordinates or not (counts == 3).all():
        raise ValueError("ATLAS seed-average coordinates do not have exactly three seeds")
    invariant_columns = [
        "amino_acid", "sequence_length", "original_neq", "original_binary_label"
    ]
    for column in invariant_columns:
        if (condition_frame.groupby(keys)[column].nunique() != 1).any():
            raise ValueError(f"ATLAS seed target mismatch in {column}")
    ordered = condition_frame.sort_values(keys + ["seed"])
    averaged = ordered.groupby(keys, as_index=False).agg(
        amino_acid=("amino_acid", "first"),
        sequence_length=("sequence_length", "first"),
        original_neq=("original_neq", "first"),
        original_binary_label=("original_binary_label", "first"),
        flexible_score=("flexible_score", "mean"),
    )
    direct = ordered.groupby(keys)["flexible_score"].mean().to_numpy()
    maximum_difference = float(np.max(np.abs(direct - averaged["flexible_score"])))
    return averaged, maximum_difference


def run_atlas_spearman(
    cache_path: Path,
    split_manifest_path: Path,
    n_bootstrap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = pd.read_csv(cache_path)
    audit = validate_atlas_cache(frame)
    manifest = pd.read_csv(split_manifest_path)
    manifest = manifest.loc[manifest["split"] == "test", ["name", "union_group_id"]]
    if len(manifest) != 208 or manifest["name"].duplicated().any():
        raise ValueError("ATLAS test split manifest must have 208 unique names")
    cluster_map = dict(zip(manifest["name"], manifest["union_group_id"]))
    if set(frame["protein"]) != set(cluster_map):
        raise ValueError("ATLAS prediction proteins and test split manifest differ")

    protein_rows, summary_rows = [], []
    seed_average_differences = {}
    for condition_index, condition in enumerate(ESM_CONDITIONS):
        condition_frame = frame.loc[frame["condition"] == condition].copy()
        evaluation_frames: list[tuple[str, int | str, pd.DataFrame]] = [
            ("per_seed", seed, condition_frame.loc[condition_frame["seed"] == seed].copy())
            for seed in (1, 2, 3)
        ]
        averaged, difference = atlas_seed_average(condition_frame)
        seed_average_differences[condition] = difference
        evaluation_frames.append(("seed_average", "average", averaged))

        for aggregation_index, (aggregation, seed, evaluation) in enumerate(evaluation_frames):
            per_condition = []
            for protein, part in evaluation.groupby("protein", sort=True):
                part = part.sort_values("residue_index_0based")
                if len(part) != int(part["sequence_length"].iloc[0]):
                    raise ValueError(f"ATLAS length mismatch for {condition}/{seed}/{protein}")
                value = safe_spearman(
                    part["original_neq"].to_numpy(float),
                    part["flexible_score"].to_numpy(float),
                )
                row = {
                    "dataset": "atlas_grouped_v1_test",
                    "target": "neq",
                    "metric": "spearman",
                    "condition": condition,
                    "display_name": ESM_DISPLAY[condition],
                    "aggregation": aggregation,
                    "seed": seed,
                    "protein": protein,
                    "cluster": cluster_map[protein],
                    "n_residues": len(part),
                    "spearman": value,
                }
                per_condition.append(row)
                protein_rows.append(row)
            per_condition_frame = pd.DataFrame(per_condition)
            evaluation = evaluation.sort_values(["protein", "residue_index_0based"])
            summary = summarize_protein_metric(
                per_condition_frame,
                evaluation["original_neq"].to_numpy(float),
                evaluation["flexible_score"].to_numpy(float),
                "spearman",
                n_bootstrap,
                random_seed + condition_index * 101 + aggregation_index,
            )
            summary_rows.append({
                "dataset": "atlas_grouped_v1_test",
                "target": "neq",
                "metric": "spearman",
                "condition": condition,
                "display_name": ESM_DISPLAY[condition],
                "aggregation": aggregation,
                "seed": seed,
                **summary,
            })
    audit.update({
        "prediction_cache": _portable_path(cache_path),
        "split_manifest": _portable_path(split_manifest_path),
        "union_groups": len(set(cluster_map.values())),
        "seed_average_maximum_direct_mean_difference": seed_average_differences,
    })
    return pd.DataFrame(protein_rows), pd.DataFrame(summary_rows), audit


def extract_attention_score_json(
    source: Path,
    expected_sequences: dict[str, str],
) -> dict[str, np.ndarray]:
    """Extract name/sequence/flexible_scores while skipping large attention matrices."""
    output: dict[str, np.ndarray] = {}
    name_key = b'"name"'
    sequence_key = b'"sequence"'
    score_key = b'"flexible_scores"'
    with source.open("rb") as fh, mmap.mmap(
        fh.fileno(), length=0, access=mmap.ACCESS_READ
    ) as mapped:
        position = 0
        while True:
            name_position = mapped.find(name_key, position)
            if name_position < 0:
                break
            name_colon = mapped.find(b":", name_position + len(name_key))
            name_start = mapped.find(b'"', name_colon + 1)
            name_end = mapped.find(b'"', name_start + 1)
            name = mapped[name_start + 1:name_end].decode("utf-8")

            sequence_position = mapped.find(sequence_key, name_end)
            sequence_colon = mapped.find(b":", sequence_position + len(sequence_key))
            sequence_start = mapped.find(b'"', sequence_colon + 1)
            sequence_end = mapped.find(b'"', sequence_start + 1)
            sequence = mapped[sequence_start + 1:sequence_end].decode("ascii")

            score_position = mapped.find(score_key, sequence_end)
            next_name = mapped.find(name_key, sequence_end)
            if score_position < 0 or (next_name >= 0 and next_name < score_position):
                raise ValueError(f"Missing flexible_scores after {name} in {source}")
            array_start = mapped.find(b"[", score_position + len(score_key))
            array_end = mapped.find(b"]", array_start + 1)
            if array_start < 0 or array_end < 0:
                raise ValueError(f"Malformed flexible_scores for {name} in {source}")
            values = np.fromstring(
                mapped[array_start + 1:array_end].decode("ascii"), sep=","
            )
            if name not in expected_sequences:
                position = array_end + 1
                continue
            if name in output:
                raise ValueError(f"Duplicate record {name} in {source}")
            if sequence != expected_sequences[name]:
                raise ValueError(f"Sequence mismatch for {name} in {source}")
            if len(values) != len(sequence) or not np.isfinite(values).all():
                raise ValueError(f"Invalid score vector for {name} in {source}")
            output[name] = values
            position = array_end + 1
    if set(output) != set(expected_sequences):
        missing = sorted(set(expected_sequences) - set(output))[:10]
        raise ValueError(
            f"Per-seed source coverage mismatch for {source}: "
            f"found={len(output)}, missing={missing}"
        )
    return output


def compact_seed_score_cache(
    source: Path,
    output: Path,
    expected_sequences: dict[str, str],
    force: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    audit_path = output.with_suffix(".audit.json")
    expected_source = source_stat(source)
    if output.is_file() and audit_path.is_file() and not force:
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("cache_schema") == CACHE_SCHEMA
            and audit.get("source") == expected_source
            and audit.get("output_sha256") == sha256_file(output)
        ):
            scores = _load_score_json(output)
            if set(scores) == set(expected_sequences) and all(
                len(scores[name]) == len(sequence)
                for name, sequence in expected_sequences.items()
            ):
                return scores, audit

    print(f"[mdCATH] compacting {source.name}")
    scores = extract_attention_score_json(source, expected_sequences)
    atomic_json(output, {name: values.tolist() for name, values in scores.items()})
    audit = {
        "cache_schema": CACHE_SCHEMA,
        "source": expected_source,
        "output": _portable_path(output),
        "output_sha256": sha256_file(output),
        "proteins": len(scores),
        "residues": sum(len(values) for values in scores.values()),
        "field": "flexible_scores = P(class 1: Neq > 1.0)",
    }
    atomic_json(audit_path, audit, indent=2)
    return scores, audit


def linear_seed_score_caches(
    source_dir: Path,
    runs_dir: Path,
    cache_dir: Path,
    expected_sequences: dict[str, str],
    device_name: str,
    batch_size: int,
    force: bool,
) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    import torch

    output_paths = {
        seed: cache_dir / f"{LINEAR_CONDITION}_seed_{seed}_scores.json"
        for seed in (1, 2, 3)
    }
    audit_path = cache_dir / f"{LINEAR_CONDITION}_seed_scores.audit.json"
    mean_audit_path = (
        source_dir / "esmfluc_inference"
        / f"{LINEAR_CONDITION}_mean_scores.audit.json"
    )
    mean_audit = json.loads(mean_audit_path.read_text())
    provenance = {
        "cache_schema": CACHE_SCHEMA,
        "checkpoint_paths": mean_audit["checkpoint_paths"],
        "checkpoint_sha256": mean_audit["checkpoint_sha256"],
        "fasta": source_stat(source_dir / "mdcath_320K_strict.fasta"),
    }
    if audit_path.is_file() and all(path.is_file() for path in output_paths.values()) and not force:
        audit = json.loads(audit_path.read_text())
        output_hashes = {
            str(seed): sha256_file(path) for seed, path in output_paths.items()
        }
        if audit.get("provenance") == provenance and audit.get("output_sha256") == output_hashes:
            scores = {seed: _load_score_json(path) for seed, path in output_paths.items()}
            if all(set(item) == set(expected_sequences) for item in scores.values()):
                return scores, audit

    checkpoints = [
        runs_dir / LINEAR_CONDITION / f"seed_{seed}" / "best_model.pth"
        for seed in (1, 2, 3)
    ]
    requested_cuda = device_name == "cuda"
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")
    if requested_cuda and device.type == "cpu":
        print("[linear] CUDA requested but unavailable; using CPU")
    model, tokenizer, heads, verification = load_verified_linear_model(
        checkpoints, device
    )
    records = sorted(_parse_fasta(source_dir / "mdcath_320K_strict.fasta"), key=lambda x: len(x[1]))
    if dict(records) != expected_sequences:
        raise ValueError("Strict FASTA and mdCATH CSV sequences differ")
    scores: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in (1, 2, 3)}
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        sequences = [sequence for _, sequence in batch]
        encoded = tokenize_linear_sequences(tokenizer, sequences)
        probabilities = shared_backbone_linear_probabilities_by_head(
            model, heads, encoded, device
        ).cpu().numpy()
        for row, (domain, sequence) in enumerate(batch):
            for seed_index, seed in enumerate((1, 2, 3)):
                scores[seed][domain] = probabilities[
                    seed_index, row, :len(sequence)
                ].astype(float)
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            print(f"[linear] per-seed scores {min(start + batch_size, len(records))}/{len(records)}")
    for seed, path in output_paths.items():
        atomic_json(path, {name: value.tolist() for name, value in scores[seed].items()})

    mean_scores = _load_score_json(
        source_dir / "esmfluc_inference" / f"{LINEAR_CONDITION}_mean_scores.json"
    )
    maximum_difference = 0.0
    for domain in expected_sequences:
        averaged = np.mean([scores[seed][domain] for seed in (1, 2, 3)], axis=0)
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(averaged - mean_scores[domain]))),
        )
    if maximum_difference > 1e-6:
        raise ValueError(
            f"Linear seed average differs from canonical mean cache: {maximum_difference}"
        )
    audit = {
        "cache_schema": CACHE_SCHEMA,
        "provenance": provenance,
        "output_sha256": {
            str(seed): sha256_file(path) for seed, path in output_paths.items()
        },
        "proteins_per_seed": len(expected_sequences),
        "residues_per_seed": sum(map(len, expected_sequences.values())),
        "backbone_verification": verification,
        "maximum_seed_average_vs_canonical_mean_difference": maximum_difference,
        "device": str(device),
    }
    atomic_json(audit_path, audit, indent=2)
    return scores, audit


def average_score_maps(seed_maps: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    domains = set(seed_maps[1])
    if any(set(seed_maps[seed]) != domains for seed in (2, 3)):
        raise ValueError("Seed score maps have different domain sets")
    return {
        domain: np.mean([seed_maps[seed][domain] for seed in (1, 2, 3)], axis=0)
        for domain in domains
    }


def run_mdcath_auroc(
    source_dir: Path,
    runs_dir: Path,
    netsurfp: Path,
    output_dir: Path,
    n_bootstrap: int,
    random_seed: int,
    device: str,
    linear_batch_size: int,
    force_seed_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cohort, neq, invalid_neq = load_cohort(source_dir / "mdcath_320K_strict.csv")
    sequences = dict(zip(cohort["domain"], cohort["sequence"]))
    clusters = load_clusters(
        source_dir / "mdcath_cluster_assignments.tsv", set(sequences)
    )
    method_args = SimpleNamespace(
        source_dir=source_dir,
        esm_runs_dir=runs_dir,
        netsurfp=netsurfp,
        infer_linear=False,
        force_linear=False,
        device=device,
        linear_batch_size=linear_batch_size,
    )
    methods, input_audit = load_methods(method_args, cohort)
    coverage = validate_method_coverage(methods, sequences, set(neq))
    incomplete = {
        method: detail for method, detail in coverage.items()
        if detail["n_valid"] != len(neq) or detail["n_missing"]
        or detail["wrong_length"] or detail["nonfinite_domains"]
    }
    if incomplete or len(methods) != 15:
        raise ValueError(
            f"Incomplete canonical mdCATH methods: count={len(methods)}, details={incomplete}"
        )
    canonical_maps = {method.method: method.scores for method in methods}
    display = {method.method: method.display_name for method in methods}

    seed_cache_dir = output_dir / "mdcath_seed_score_cache"
    seed_cache_dir.mkdir(parents=True, exist_ok=True)
    seed_maps: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    seed_cache_audits = {}
    inference_dir = source_dir / "esmfluc_inference"
    for condition in ESM_BILSTM_CONDITIONS:
        condition_maps = {}
        condition_audits = {}
        for seed in (1, 2, 3):
            source = inference_dir / f"{condition}_seed_{seed}_scores.json"
            output = seed_cache_dir / f"{condition}_seed_{seed}_scores.json"
            condition_maps[seed], condition_audits[str(seed)] = compact_seed_score_cache(
                source, output, sequences, force_seed_cache
            )
        seed_maps[condition] = condition_maps
        seed_cache_audits[condition] = condition_audits

    seed_maps[LINEAR_CONDITION], seed_cache_audits[LINEAR_CONDITION] = (
        linear_seed_score_caches(
            source_dir,
            runs_dir,
            seed_cache_dir,
            sequences,
            device,
            linear_batch_size,
            force_seed_cache,
        )
    )

    average_differences = {}
    for condition in ESM_CONDITIONS:
        averaged = average_score_maps(seed_maps[condition])
        maximum = max(
            float(np.max(np.abs(averaged[domain] - canonical_maps[condition][domain])))
            for domain in sequences
        )
        if maximum > 1e-6:
            raise ValueError(
                f"{condition}: seed average differs from canonical cache by {maximum}"
            )
        average_differences[condition] = maximum

    evaluations: list[tuple[str, str, str, int | str, dict[str, np.ndarray]]] = []
    for condition in ESM_CONDITIONS:
        for seed in (1, 2, 3):
            evaluations.append((condition, display[condition], "per_seed", seed, seed_maps[condition][seed]))
        evaluations.append((condition, display[condition], "seed_average", "average", canonical_maps[condition]))
    for method in methods:
        if method.block != "esmfluc":
            evaluations.append((
                method.method, method.display_name, "external_unseeded", "unseeded", method.scores
            ))

    protein_rows, summary_rows = [], []
    valid_domains = sorted(neq)
    for evaluation_index, (condition, display_name, aggregation, seed, score_map) in enumerate(evaluations):
        per_condition = []
        pooled_labels, pooled_scores = [], []
        for domain in valid_domains:
            target = neq[domain]
            score = np.asarray(score_map[domain], dtype=float)
            if len(score) != len(target) or not np.isfinite(score).all():
                raise ValueError(f"Invalid mdCATH score for {condition}/{seed}/{domain}")
            labels = (target > 1.0).astype(int)
            value = safe_auroc(labels, score)
            row = {
                "dataset": "mdcath_320K_strict_atlas_disjoint",
                "target": "neq_binary_gt_1",
                "metric": "auroc",
                "condition": condition,
                "display_name": display_name,
                "aggregation": aggregation,
                "seed": seed,
                "protein": domain,
                "cluster": clusters[domain],
                "n_residues": len(target),
                "n_rigid": int((labels == 0).sum()),
                "n_flexible": int((labels == 1).sum()),
                "auroc": value,
            }
            per_condition.append(row)
            protein_rows.append(row)
            pooled_labels.append(labels)
            pooled_scores.append(score)
        per_condition_frame = pd.DataFrame(per_condition)
        summary = summarize_protein_metric(
            per_condition_frame,
            np.concatenate(pooled_labels),
            np.concatenate(pooled_scores),
            "auroc",
            n_bootstrap,
            random_seed + evaluation_index,
        )
        summary_rows.append({
            "dataset": "mdcath_320K_strict_atlas_disjoint",
            "target": "neq_binary_gt_1",
            "metric": "auroc",
            "condition": condition,
            "display_name": display_name,
            "aggregation": aggregation,
            "seed": seed,
            **summary,
        })

    summary_frame = pd.DataFrame(summary_rows)
    canonical_path = SCRIPT_DIR / "results" / "benchmark_unified_mdcath" / "unified_method_summary.csv"
    canonical_differences = {}
    if canonical_path.is_file():
        canonical = pd.read_csv(canonical_path)
        canonical = canonical.loc[
            (canonical["target"] == "neq")
            & (canonical["scope"] == "shared_loaded_methods")
        ].set_index("method")
        averaged = summary_frame.loc[
            summary_frame["aggregation"].isin(["seed_average", "external_unseeded"])
        ].set_index("condition")
        for condition in averaged.index:
            canonical_differences[condition] = {
                "macro_mean_auroc": float(abs(
                    averaged.loc[condition, "macro_mean_auroc"]
                    - canonical.loc[condition, "macro_mean_auroc"]
                )),
                "pooled_auroc": float(abs(
                    averaged.loc[condition, "pooled_auroc"]
                    - canonical.loc[condition, "pooled_auroc"]
                )),
            }
        if max(
            value for differences in canonical_differences.values()
            for value in differences.values()
        ) > 1e-12:
            raise ValueError("New averaged mdCATH AUROC does not reproduce canonical metrics")

    audit = {
        "strict_csv": _portable_path(source_dir / "mdcath_320K_strict.csv"),
        "cluster_file": _portable_path(source_dir / "mdcath_cluster_assignments.tsv"),
        "n_cohort_domains": len(sequences),
        "n_valid_neq_domains": len(neq),
        "invalid_neq": invalid_neq,
        "n_valid_neq_residues": sum(map(len, neq.values())),
        "n_valid_neq_clusters": len({clusters[domain] for domain in neq}),
        "binary_definition": "flexible iff Neq > 1.0",
        "seed_average_definition": "arithmetic mean of three residue probabilities, then score",
        "score_orientation": "higher means flexible; DynaMine S2 and PEGASUS mean LDDT negated",
        "input_audit": input_audit,
        "canonical_method_coverage": coverage,
        "seed_cache_audits": seed_cache_audits,
        "maximum_seed_average_vs_canonical_cache_difference": average_differences,
        "canonical_averaged_metric_absolute_differences": canonical_differences,
    }
    return pd.DataFrame(protein_rows), summary_frame, audit


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be greater than zero")
    if args.linear_batch_size <= 0:
        raise ValueError("--linear-batch-size must be greater than zero")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atlas_per_protein, atlas_summary, atlas_audit = run_atlas_spearman(
        args.atlas_cache,
        args.atlas_split_manifest,
        args.n_bootstrap,
        args.random_seed,
    )
    atlas_per_protein.to_csv(
        args.output_dir / "atlas_test_spearman_per_protein.csv", index=False
    )
    atlas_summary.to_csv(
        args.output_dir / "atlas_test_spearman_summary.csv", index=False
    )

    mdcath_per_protein, mdcath_summary, mdcath_audit = run_mdcath_auroc(
        args.mdcath_source_dir,
        args.esm_runs_dir,
        args.netsurfp,
        args.output_dir,
        args.n_bootstrap,
        args.random_seed,
        args.device,
        args.linear_batch_size,
        args.force_seed_cache,
    )
    mdcath_per_protein.to_csv(
        args.output_dir / "mdcath_neq_auroc_per_protein.csv", index=False
    )
    mdcath_summary.to_csv(
        args.output_dir / "mdcath_neq_auroc_summary.csv", index=False
    )

    output_files = [
        args.output_dir / "atlas_test_spearman_per_protein.csv",
        args.output_dir / "atlas_test_spearman_summary.csv",
        args.output_dir / "mdcath_neq_auroc_per_protein.csv",
        args.output_dir / "mdcath_neq_auroc_summary.csv",
    ]
    audit = {
        "schema": "atlas-mdcath-missing-metrics-v1",
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed,
        "seed_average_definition": "average residue probabilities across seeds 1,2,3 before metrics",
        "atlas": atlas_audit,
        "mdcath": mdcath_audit,
        "outputs": [
            {"path": _portable_path(path), "sha256": sha256_file(path)}
            for path in output_files
        ],
    }
    atomic_json(args.output_dir / "metric_completion_audit.json", audit, indent=2)

    print("\nATLAS test Spearman (seed-averaged probabilities)")
    print(atlas_summary.loc[
        atlas_summary["aggregation"] == "seed_average",
        ["display_name", "n_proteins_valid", "macro_mean_spearman",
         "macro_mean_spearman_ci_low", "macro_mean_spearman_ci_high",
         "pooled_spearman"],
    ].sort_values("macro_mean_spearman", ascending=False).to_string(index=False))
    print("\nmdCATH Neq AUROC (seed-averaged ESM; unseeded external)")
    print(mdcath_summary.loc[
        mdcath_summary["aggregation"].isin(["seed_average", "external_unseeded"]),
        ["display_name", "aggregation", "n_proteins_valid", "n_proteins_undefined",
         "macro_mean_auroc", "macro_mean_auroc_ci_low",
         "macro_mean_auroc_ci_high", "pooled_auroc"],
    ].sort_values("macro_mean_auroc", ascending=False).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
