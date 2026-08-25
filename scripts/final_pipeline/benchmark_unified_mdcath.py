#!/usr/bin/env python3
"""Unified residue-level flexibility benchmark on strict mdCATH 320 K.

The script joins every predictor to the same ATLAS-disjoint mdCATH cohort by
``(domain, 1-based residue index)`` and refuses length-mismatched records.  It
reports per-protein and pooled Spearman correlations, Neq AUROC, MMseqs2-cluster
bootstrap confidence intervals, and paired macro-Spearman deltas.

Existing ESMfluc/PEGASUS caches are reused.  NetSurfP may be supplied as a CSV,
CSV.GZ, or a ZIP containing one CSV.  The optional linear inference stage runs
the three ESM2-frozen-linear checkpoints without serializing attention maps.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "results" / "benchmark_pegasus_mdcath"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "results" / "publication_comparable_v2" / "runs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "benchmark_unified_mdcath"

ESM_BILSTM_CONDITIONS = (
    "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn",
    "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn",
    "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
)
LINEAR_CONDITION = "esm2_frozen_linear"
LINEAR_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
LINEAR_CACHE_SCHEMA_VERSION = 2
LINEAR_INFERENCE_IMPLEMENTATION = "shared_backbone_single_sequence_v2"
LINEAR_EQUIVALENCE_TOLERANCE = 1e-6
ESM_DISPLAY = {
    "esm2_frozen_linear": "ESMfluc ESM2 frozen linear",
    "esm2_frozen_bilstm_attn": "ESMfluc ESM2 frozen BiLSTM-attn",
    "esm2_top4_bilstm_attn": "ESMfluc ESM2 top-4 BiLSTM-attn",
    "esm2_top28_bilstm_attn": "ESMfluc ESM2 top-28 BiLSTM-attn",
    "esm3_frozen_bilstm_attn": "ESMfluc ESM3 frozen BiLSTM-attn",
    "esm3_top4_bilstm_attn": "ESMfluc ESM3 top-4 BiLSTM-attn",
    "esm3_top28_bilstm_attn": "ESMfluc ESM3 top-28 BiLSTM-attn",
}


@dataclass(frozen=True)
class MethodScores:
    method: str
    display_name: str
    block: str
    score_source: str
    scores: dict[str, np.ndarray]
    derived: bool = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unified ESMfluc/external baseline benchmark on strict mdCATH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--strict-csv", type=Path, default=None)
    ap.add_argument("--clusters-tsv", type=Path, default=None)
    ap.add_argument("--esm-runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--netsurfp", type=Path, default=None,
                    help="NetSurfP-3.0 long CSV/CSV.GZ or ZIP containing it")
    ap.add_argument("--rmsf-csv", type=Path, default=None,
                    help="Optional CSV with domain and per-residue rmsf list")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--infer-linear", action="store_true",
                    help="Run and seed-average ESM2 frozen-linear inference if absent")
    ap.add_argument("--force-linear", action="store_true",
                    help="Regenerate only the ESM2 frozen-linear cache and its provenance audit")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--linear-batch-size", type=int, default=2,
                    help="Batch size for shared-backbone frozen-linear inference")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--require-complete", action="store_true",
                    help="Fail unless all 7 ESMfluc, DynaMine, NetSurfP, and PEGASUS inputs are valid")
    ap.add_argument("--write-long", action="store_true",
                    help="Write a potentially large residue-level tidy CSV")
    return ap.parse_args()


def _vector(value: object, source: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = [part for part in value.replace(";", ",").split(",") if part.strip()]
    arr = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{source}: vector contains missing/nonfinite values")
    return arr


def _read_json(path: Path) -> object:
    with path.open() as fh:
        return json.load(fh)


def load_cohort(path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Strict mdCATH CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"domain", "sequence", "neq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strict CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["domain"] = df["domain"].astype(str).str.lstrip(">")
    df["sequence"] = df["sequence"].astype(str).str.upper()
    if df["domain"].duplicated().any():
        duplicates = sorted(df.loc[df["domain"].duplicated(keep=False), "domain"].unique())
        raise ValueError(f"Strict CSV contains duplicate domain IDs after normalization: {duplicates}")

    neq: dict[str, np.ndarray] = {}
    invalid: list[dict] = []
    for row in df.itertuples(index=False):
        arr = _vector(row.neq, f"Neq {row.domain}")
        if len(arr) != len(row.sequence):
            invalid.append({
                "domain": row.domain,
                "sequence_length": len(row.sequence),
                "neq_length": len(arr),
                "reason": "neq_length_mismatch",
            })
            continue
        neq[row.domain] = arr
    return df, neq, invalid


def load_clusters(path: Path, domains: set[str]) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"MMseqs2 cluster file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    if not {"rep", "member"}.issubset(df.columns):
        raise ValueError("Cluster TSV must contain rep and member columns")
    df["member"] = df["member"].astype(str).str.lstrip(">")
    df["rep"] = df["rep"].astype(str).str.lstrip(">")
    if df["member"].duplicated().any():
        raise ValueError("Cluster TSV assigns at least one member more than once")
    mapping = dict(zip(df["member"], df["rep"]))
    missing = domains - set(mapping)
    extra = set(mapping) - domains
    if missing or extra:
        raise ValueError(
            f"Cluster/cohort mismatch: {len(missing)} missing and {len(extra)} extra members"
        )
    return mapping


def _load_score_json(path: Path) -> dict[str, np.ndarray]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected domain-to-vector JSON object: {path}")
    return {str(k).lstrip(">"): _vector(v, f"{path.name}:{k}") for k, v in raw.items()}


def _torch_load_weights(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    name, parts = None, []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                name, parts = line[1:].strip(), []
            else:
                parts.append(line)
    if name is not None:
        records.append((name, "".join(parts)))
    return records


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(SCRIPT_DIR.resolve()))
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    return [
        args.esm_runs_dir / LINEAR_CONDITION / f"seed_{seed}" / "best_model.pth"
        for seed in (1, 2, 3)
    ]


def _linear_cache_paths(infer_dir: Path) -> tuple[Path, Path]:
    mean_path = infer_dir / f"{LINEAR_CONDITION}_mean_scores.json"
    audit_path = infer_dir / f"{LINEAR_CONDITION}_mean_scores.audit.json"
    return mean_path, audit_path


def tokenize_linear_sequences(tokenizer, sequences: list[str]):
    """Tokenize one copy of each sequence and enforce exactly L real tokens."""
    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    token_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    expected_lengths = [len(sequence) for sequence in sequences]
    if token_lengths != expected_lengths:
        raise ValueError(
            "Single-sequence ESM2 tokenization did not produce exactly L tokens: "
            f"observed={token_lengths}, expected={expected_lengths}; "
            "add_special_tokens=False"
        )
    return encoded


def shared_backbone_linear_probabilities(model, heads, encoded, device):
    """Run one frozen-backbone forward and apply every seed-specific head."""
    return shared_backbone_linear_probabilities_by_head(
        model, heads, encoded, device
    ).mean(dim=0)


def shared_backbone_linear_probabilities_by_head(model, heads, encoded, device):
    """Return one flexible-probability tensor per seed-specific linear head."""
    import torch

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        hidden = model.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        probabilities = [
            torch.softmax(
                torch.nn.functional.linear(hidden, weight.to(device), bias.to(device)),
                dim=-1,
            )[..., 1]
            for weight, bias in heads
        ]
    return torch.stack(probabilities)


def ordinary_linear_checkpoint_probabilities(model, checkpoints, encoded, device):
    """Reference path: load and run each complete checkpoint normally."""
    import torch

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    probabilities = []
    for checkpoint in checkpoints:
        state = _torch_load_weights(checkpoint)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).eval()
        with torch.inference_mode():
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities.append(torch.softmax(logits, dim=-1)[..., 1])
    return torch.stack(probabilities)


def load_verified_linear_model(
    checkpoints: list[Path], device, *, local_files_only: bool = False
):
    """Load seed 1 once and prove all backbone tensors used in inference match."""
    import torch
    from transformers import EsmModel, EsmTokenizer
    from models import ESMLinearTokenClassifier

    if len(checkpoints) != 3:
        raise ValueError(f"Expected three frozen-linear checkpoints; found {len(checkpoints)}")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Linear checkpoint not found: {checkpoint}")

    embedding = EsmModel.from_pretrained(
        LINEAR_MODEL_NAME, local_files_only=local_files_only
    )
    tokenizer = EsmTokenizer.from_pretrained(
        LINEAR_MODEL_NAME, local_files_only=local_files_only
    )
    model = ESMLinearTokenClassifier(embedding, num_classes=2)

    seed1 = _torch_load_weights(checkpoints[0])
    model.load_state_dict(seed1, strict=True)
    heads = [(seed1["fc.weight"].clone(), seed1["fc.bias"].clone())]
    reference = model.state_dict()
    used_backbone_keys = sorted(
        key for key in reference
        if key.startswith("embedding_model.") and ".pooler." not in key
    )
    del seed1

    for checkpoint in checkpoints[1:]:
        state = _torch_load_weights(checkpoint)
        state_used_keys = {
            key for key in state
            if key.startswith("embedding_model.") and ".pooler." not in key
        }
        if state_used_keys != set(used_backbone_keys):
            missing = sorted(set(used_backbone_keys) - state_used_keys)[:5]
            extra = sorted(state_used_keys - set(used_backbone_keys))[:5]
            raise ValueError(
                f"Frozen-linear backbone key mismatch in {checkpoint}: "
                f"missing={missing}, extra={extra}"
            )
        differing = [
            key for key in used_backbone_keys
            if not torch.equal(state[key], reference[key])
        ]
        if differing:
            raise ValueError(
                f"Frozen-linear backbone differs in {checkpoint}: {differing[:5]}"
            )
        heads.append((state["fc.weight"].clone(), state["fc.bias"].clone()))
        del state

    model.to(device).eval()
    heads = [(weight.to(device), bias.to(device)) for weight, bias in heads]
    verification = {
        "backbone_equality_passed": True,
        "n_used_backbone_tensors_compared": len(used_backbone_keys),
        "excluded_unused_backbone_tensors": ["embedding_model.pooler.*"],
    }
    return model, tokenizer, heads, verification


def _validate_linear_cache(
    mean_path: Path, audit_path: Path, checkpoints: list[Path]
) -> dict:
    if not audit_path.is_file():
        raise ValueError(
            f"Refusing unversioned frozen-linear cache {mean_path}. "
            "Regenerate it with --infer-linear --force-linear."
        )
    audit = _read_json(audit_path)
    expected = {
        "cache_schema_version": LINEAR_CACHE_SCHEMA_VERSION,
        "tokenizer_model": LINEAR_MODEL_NAME,
        "add_special_tokens": False,
        "sequence_input_mode": "single_sequence_once",
        "inference_implementation": LINEAR_INFERENCE_IMPLEMENTATION,
        "equivalence_test_tolerance": LINEAR_EQUIVALENCE_TOLERANCE,
        "backbone_equality_passed": True,
    }
    mismatches = {
        key: {"observed": audit.get(key), "expected": value}
        for key, value in expected.items() if audit.get(key) != value
    }
    expected_paths = [_portable_path(path) for path in checkpoints]
    if audit.get("checkpoint_paths") != expected_paths:
        mismatches["checkpoint_paths"] = {
            "observed": audit.get("checkpoint_paths"), "expected": expected_paths,
        }
    observed_cache_hash = _sha256_file(mean_path)
    if audit.get("cache_sha256") != observed_cache_hash:
        mismatches["cache_sha256"] = {
            "observed": audit.get("cache_sha256"), "expected": observed_cache_hash,
        }
    hashes = audit.get("checkpoint_sha256")
    current_hashes = [_sha256_file(path) for path in checkpoints]
    if hashes != current_hashes:
        mismatches["checkpoint_sha256"] = {
            "observed": hashes, "expected": current_hashes,
        }
    if mismatches:
        raise ValueError(
            f"Refusing incompatible frozen-linear cache {mean_path}: {mismatches}. "
            "Regenerate it with --infer-linear --force-linear."
        )
    return audit


def infer_linear_scores(
    args: argparse.Namespace, fasta: Path, infer_dir: Path
) -> tuple[Path, dict | None]:
    mean_path = infer_dir / f"{LINEAR_CONDITION}_mean_scores.json"
    mean_path, audit_path = _linear_cache_paths(infer_dir)
    checkpoints = _linear_checkpoint_paths(args)
    if mean_path.is_file() and not args.force_linear:
        return mean_path, _validate_linear_cache(mean_path, audit_path, checkpoints)
    if not args.infer_linear:
        return mean_path, None

    # The backbone was frozen in all three runs. Load it once, verify every
    # used backbone tensor is identical across seeds, and apply all three heads
    # to the same hidden states. This is mathematically identical to three full
    # forwards while cutting backbone inference by a factor of three.
    import torch

    infer_dir.mkdir(parents=True, exist_ok=True)
    requested_cuda = args.device == "cuda"
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")
    if requested_cuda and device.type == "cpu":
        print("[linear] CUDA requested but unavailable; using CPU")
    model, tokenizer, heads, verification = load_verified_linear_model(
        checkpoints, device
    )

    records = sorted(_parse_fasta(fasta), key=lambda item: len(item[1]))
    record_ids = [domain for domain, _ in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Strict FASTA contains duplicate domain IDs")
    mean: dict[str, list[float]] = {}
    batch_size = args.linear_batch_size
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        sequences = [sequence for _, sequence in batch]
        encoded = tokenize_linear_sequences(tokenizer, sequences)
        averaged = shared_backbone_linear_probabilities(
            model, heads, encoded, device
        )
        averaged = averaged.cpu()
        for row, (domain, sequence) in enumerate(batch):
            mean[domain] = averaged[row, :len(sequence)].tolist()
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(f"[linear] scored {min(start + batch_size, len(records))}/{len(records)} proteins")

    temporary = mean_path.with_suffix(mean_path.suffix + ".tmp")
    with temporary.open("w") as fh:
        json.dump(mean, fh)
    temporary.replace(mean_path)

    checkpoint_hashes = [_sha256_file(path) for path in checkpoints]
    audit = {
        "cache_schema_version": LINEAR_CACHE_SCHEMA_VERSION,
        "cache_file": _portable_path(mean_path),
        "cache_sha256": _sha256_file(mean_path),
        "n_cached_proteins": len(mean),
        "n_cached_residues": sum(len(values) for values in mean.values()),
        "checkpoint_paths": [_portable_path(path) for path in checkpoints],
        "checkpoint_sha256": checkpoint_hashes,
        **verification,
        "tokenizer_model": LINEAR_MODEL_NAME,
        "add_special_tokens": False,
        "sequence_input_mode": "single_sequence_once",
        "inference_implementation": LINEAR_INFERENCE_IMPLEMENTATION,
        "equivalence_test_tolerance": LINEAR_EQUIVALENCE_TOLERANCE,
        "equivalence_test_command": "python3 -m unittest tests.test_benchmark_unified_mdcath",
        "vector_length_policy": "exactly one output per input residue; no truncation",
    }
    audit_temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with audit_temporary.open("w") as fh:
        json.dump(audit, fh, indent=2)
    audit_temporary.replace(audit_path)
    return mean_path, audit


def _read_netsurfp_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith((".csv", ".tsv", ".txt"))]
            if len(members) != 1:
                csv_members = [n for n in members if n.lower().endswith(".csv")]
                if len(csv_members) == 1:
                    members = csv_members
                else:
                    raise ValueError(
                        f"NetSurfP ZIP must contain one unambiguous table; found {members}"
                    )
            with archive.open(members[0]) as fh:
                return pd.read_csv(fh, sep=None, engine="python")
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def load_netsurfp(
    path: Path, sequences: dict[str, str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    if path.suffix.lower() == ".json":
        records = _read_json(path)
        if not isinstance(records, list):
            raise ValueError("NetSurfP JSON must be a list of per-protein records")
        outputs = {"netsurfp_disorder": {}, "netsurfp_rsa": {}}
        invalid: dict[str, str] = {}
        seen = set()
        for record in records:
            raw_id = str(record.get("desc") or record.get("id", "")).lstrip(">")
            domain = raw_id if raw_id in sequences else raw_id.split("_", 1)[-1]
            if domain not in sequences:
                invalid[raw_id] = "domain_not_in_strict_cohort"
                continue
            if domain in seen:
                raise ValueError(f"Duplicate NetSurfP JSON record for {domain}")
            seen.add(domain)
            if str(record.get("seq", "")).upper() != sequences[domain]:
                invalid[domain] = "amino_acid_sequence_mismatch"
                continue
            disorder = _vector(record.get("disorder", []), f"NetSurfP disorder:{domain}")
            rsa = _vector(record.get("rsa", []), f"NetSurfP RSA:{domain}")
            if len(disorder) != len(sequences[domain]) or len(rsa) != len(sequences[domain]):
                invalid[domain] = (
                    f"vector_lengths disorder={len(disorder)} rsa={len(rsa)} "
                    f"expected={len(sequences[domain])}"
                )
                continue
            outputs["netsurfp_disorder"][domain] = disorder
            outputs["netsurfp_rsa"][domain] = rsa
        return outputs, invalid

    table = _read_netsurfp_table(path)
    disorder_col = _find_column(table.columns, ("disorder", "p_disorder", "disorder_probability"))
    rsa_col = _find_column(table.columns, ("rsa", "relative_surface_accessibility"))
    return load_long_predictors(
        table, sequences,
        {"netsurfp_disorder": disorder_col, "netsurfp_rsa": rsa_col},
        str(path),
    )


def _find_column(columns: Iterable[str], aliases: Iterable[str], required: bool = True) -> str | None:
    normalized = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    if required:
        raise ValueError(f"None of columns {list(aliases)} found; available={list(columns)}")
    return None


def load_long_predictors(
    table: pd.DataFrame,
    sequences: dict[str, str],
    value_columns: dict[str, str],
    source: str,
    id_aliases: tuple[str, ...] = ("id", "name", "domain"),
    index_aliases: tuple[str, ...] = ("n", "res_idx", "residue_index"),
    aa_aliases: tuple[str, ...] = ("seq", "aa", "residue"),
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    table = table.copy()
    table.columns = [str(c).strip() for c in table.columns]
    id_col = _find_column(table.columns, id_aliases)
    idx_col = _find_column(table.columns, index_aliases)
    aa_col = _find_column(table.columns, aa_aliases, required=False)
    actual_values = {
        output: _find_column(table.columns, (requested,))
        for output, requested in value_columns.items()
    }
    table[id_col] = table[id_col].astype(str).str.lstrip(">").str.strip()
    table[idx_col] = pd.to_numeric(table[idx_col], errors="raise").astype(int)

    outputs = {name: {} for name in value_columns}
    invalid: dict[str, str] = {}
    for domain, sequence in sequences.items():
        part = table.loc[table[id_col] == domain].sort_values(idx_col)
        if len(part) != len(sequence):
            invalid[domain] = f"row_count={len(part)} expected={len(sequence)}"
            continue
        expected_idx = np.arange(1, len(sequence) + 1)
        if not np.array_equal(part[idx_col].to_numpy(), expected_idx):
            invalid[domain] = "residue_indices_are_not_exactly_1_to_L"
            continue
        if aa_col is not None:
            observed = "".join(part[aa_col].astype(str).str.strip().str.upper())
            if observed != sequence:
                invalid[domain] = "amino_acid_sequence_mismatch"
                continue
        for name, actual in actual_values.items():
            arr = pd.to_numeric(part[actual], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(arr)):
                invalid[domain] = f"{name}_contains_nonfinite_values"
                break
            outputs[name][domain] = arr
        if domain in invalid:
            for score_map in outputs.values():
                score_map.pop(domain, None)
    return outputs, invalid


def load_methods(args: argparse.Namespace, cohort: pd.DataFrame) -> tuple[list[MethodScores], dict]:
    source = args.source_dir
    infer_dir = source / "esmfluc_inference"
    sequences = dict(zip(cohort["domain"], cohort["sequence"]))
    methods: list[MethodScores] = []
    audit: dict[str, object] = {"missing_inputs": [], "long_table_invalid": {}}

    for condition in ESM_BILSTM_CONDITIONS:
        path = infer_dir / f"{condition}_mean_scores.json"
        if path.is_file():
            methods.append(MethodScores(
                condition, ESM_DISPLAY[condition], "esmfluc", _portable_path(path),
                _load_score_json(path)
            ))
        else:
            audit["missing_inputs"].append(str(path))

    fasta = source / "mdcath_320K_strict.fasta"
    linear_path, linear_audit = infer_linear_scores(args, fasta, infer_dir)
    audit["linear_inference"] = linear_audit
    if linear_path.is_file():
        methods.append(MethodScores(
            LINEAR_CONDITION, ESM_DISPLAY[LINEAR_CONDITION], "esmfluc",
            _portable_path(linear_path), _load_score_json(linear_path),
        ))
    else:
        audit["missing_inputs"].append(str(linear_path))

    pegasus_path = source / "pegasus_all_heads.json"
    if pegasus_path.is_file():
        raw = _read_json(pegasus_path)
        peg_specs = (
            ("mean_STD_PHI", "pegasus_phi_std", "PEGASUS phi-std", 1.0, False),
            ("mean_STD_PSI", "pegasus_psi_std", "PEGASUS psi-std", 1.0, False),
            ("phi_psi_combined", "pegasus_phi_psi_mean", "PEGASUS phi/psi mean", 1.0, True),
            ("mean_RMSF", "pegasus_rmsf", "PEGASUS RMSF", 1.0, False),
            ("mean_MEAN_LDDT", "pegasus_neg_lddt", "PEGASUS -mean LDDT", -1.0, False),
        )
        for head, method, display, sign, derived in peg_specs:
            if head not in raw:
                audit["missing_inputs"].append(f"{pegasus_path}:{head}")
                continue
            scores = {str(k).lstrip(">"): sign * _vector(v, f"PEGASUS {head}:{k}")
                      for k, v in raw[head].items()}
            methods.append(MethodScores(
                method, display, "external_structure",
                f"{_portable_path(pegasus_path)}:{head}", scores, derived,
            ))
    else:
        audit["missing_inputs"].append(str(pegasus_path))

    dynamine_path = source / "cache_dynamine_mdcath.csv"
    if dynamine_path.is_file():
        dm_table = pd.read_csv(dynamine_path)
        loaded, invalid = load_long_predictors(
            dm_table, sequences, {"dynamine_s2": "dynamine_bb"}, str(dynamine_path),
            aa_aliases=(),
        )
        audit["long_table_invalid"]["dynamine"] = invalid
        methods.append(MethodScores(
            "dynamine_neg_s2", "DynaMine -S2", "external_sequence",
            f"negative of {_portable_path(dynamine_path)}:dynamine_bb",
            {k: -v for k, v in loaded["dynamine_s2"].items()},
        ))
    else:
        audit["missing_inputs"].append(str(dynamine_path))

    if args.netsurfp is not None:
        if not args.netsurfp.is_file():
            raise FileNotFoundError(f"NetSurfP result not found: {args.netsurfp}")
        loaded, invalid = load_netsurfp(args.netsurfp, sequences)
        audit["long_table_invalid"]["netsurfp"] = invalid
        methods.extend((
            MethodScores("netsurfp_disorder", "NetSurfP-3.0 disorder", "external_sequence", _portable_path(args.netsurfp), loaded["netsurfp_disorder"]),
            MethodScores("netsurfp_rsa", "NetSurfP-3.0 RSA", "external_sequence", _portable_path(args.netsurfp), loaded["netsurfp_rsa"]),
        ))
    else:
        audit["missing_inputs"].append("--netsurfp")
    return methods, audit


def validate_method_coverage(
    methods: list[MethodScores], sequences: dict[str, str], target_domains: set[str]
) -> dict[str, dict]:
    audits = {}
    for method in methods:
        valid = set()
        wrong_length = {}
        nonfinite = []
        for domain in target_domains & set(method.scores):
            arr = method.scores[domain]
            if len(arr) != len(sequences[domain]):
                wrong_length[domain] = {"score_length": len(arr), "sequence_length": len(sequences[domain])}
            elif not np.all(np.isfinite(arr)):
                nonfinite.append(domain)
            else:
                valid.add(domain)
        audits[method.method] = {
            "n_valid": len(valid),
            "n_missing": len(target_domains - set(method.scores)),
            "missing_domains": sorted(target_domains - set(method.scores)),
            "wrong_length": wrong_length,
            "nonfinite_domains": sorted(nonfinite),
            "valid_domains": sorted(valid),
        }
    return audits


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def per_protein_metrics(
    methods: list[MethodScores],
    target_name: str,
    targets: dict[str, np.ndarray],
    sequences: dict[str, str],
    clusters: dict[str, str],
    coverage: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for method in methods:
        valid = set(coverage[method.method]["valid_domains"]) & set(targets)
        for domain in sorted(valid):
            target = targets[domain]
            score = method.scores[domain]
            row = {
                "target": target_name,
                "method": method.method,
                "display_name": method.display_name,
                "block": method.block,
                "derived": method.derived,
                "domain": domain,
                "cluster": clusters[domain],
                "n_residues": len(sequence := sequences[domain]),
                "spearman": _safe_spearman(target, score),
                "auroc": float("nan"),
            }
            if target_name == "neq":
                row["auroc"] = _safe_auroc((target > 1.0).astype(int), score)
            rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    unique = np.unique(groups)
    group_values = {g: values[groups == g] for g in unique}
    draws = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[i] = np.mean(np.concatenate([group_values[g] for g in sampled]))
    return tuple(float(v) for v in np.percentile(draws, (2.5, 97.5)))


def summarize_methods(
    per_protein: pd.DataFrame,
    methods: list[MethodScores],
    targets: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for target_name, target_map in targets.items():
        target_pp = per_protein.loc[per_protein["target"] == target_name]
        shared = set(target_map)
        for method in methods:
            shared &= set(target_pp.loc[target_pp["method"] == method.method, "domain"])
        for method_idx, method in enumerate(methods):
            method_pp = target_pp.loc[target_pp["method"] == method.method]
            for scope, domains in (("method_available", set(method_pp["domain"])),
                                   ("shared_loaded_methods", shared)):
                part = method_pp.loc[method_pp["domain"].isin(domains)].copy()
                if part.empty:
                    continue
                score_parts = [method.scores[d] for d in part["domain"]]
                target_parts = [target_map[d] for d in part["domain"]]
                pooled_score = np.concatenate(score_parts)
                pooled_target = np.concatenate(target_parts)
                rho = part["spearman"].to_numpy(float)
                lo, hi = _cluster_bootstrap_mean(
                    rho, part["cluster"].to_numpy(), n_bootstrap,
                    np.random.default_rng(seed + method_idx * 101 + (0 if scope == "method_available" else 1)),
                )
                row = {
                    "target": target_name,
                    "scope": scope,
                    "method": method.method,
                    "display_name": method.display_name,
                    "block": method.block,
                    "derived": method.derived,
                    "n_proteins": len(part),
                    "n_clusters": part["cluster"].nunique(),
                    "n_residues": int(part["n_residues"].sum()),
                    "macro_mean_spearman": float(np.nanmean(rho)),
                    "macro_median_spearman": float(np.nanmedian(rho)),
                    "macro_mean_spearman_ci_low": lo,
                    "macro_mean_spearman_ci_high": hi,
                    "pooled_spearman": _safe_spearman(pooled_target, pooled_score),
                    "macro_mean_auroc": float("nan"),
                    "pooled_auroc": float("nan"),
                    "score_source": method.score_source,
                }
                if target_name == "neq":
                    row["macro_mean_auroc"] = float(np.nanmean(part["auroc"]))
                    row["pooled_auroc"] = _safe_auroc((pooled_target > 1.0).astype(int), pooled_score)
                rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(
    per_protein: pd.DataFrame,
    methods: list[MethodScores],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    esm = [m for m in methods if m.block == "esmfluc"]
    external = [m for m in methods if m.block != "esmfluc"]
    rows = []
    for target_idx, target in enumerate(sorted(per_protein["target"].unique())):
        pp = per_protein.loc[per_protein["target"] == target]
        for i, left in enumerate(esm):
            ldf = pp.loc[pp["method"] == left.method, ["domain", "cluster", "spearman"]]
            for j, right in enumerate(external):
                rdf = pp.loc[pp["method"] == right.method, ["domain", "spearman"]]
                pair = ldf.merge(rdf, on="domain", suffixes=("_esm", "_external"))
                pair = pair.loc[np.isfinite(pair["spearman_esm"]) & np.isfinite(pair["spearman_external"])]
                if pair.empty:
                    continue
                delta = pair["spearman_esm"].to_numpy() - pair["spearman_external"].to_numpy()
                lo, hi = _cluster_bootstrap_mean(
                    delta, pair["cluster"].to_numpy(), n_bootstrap,
                    np.random.default_rng(seed + target_idx * 10007 + i * 101 + j),
                )
                try:
                    pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
                except ValueError:
                    pvalue = float("nan")
                rows.append({
                    "target": target,
                    "esmfluc_method": left.method,
                    "external_method": right.method,
                    "n_paired_proteins": len(pair),
                    "n_clusters": pair["cluster"].nunique(),
                    "mean_delta_spearman": float(np.mean(delta)),
                    "median_delta_spearman": float(np.median(delta)),
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                    "wilcoxon_pvalue_unadjusted": pvalue,
                })
    return pd.DataFrame(rows)


def load_rmsf(path: Path, sequences: dict[str, str]) -> tuple[dict[str, np.ndarray], dict]:
    df = pd.read_csv(path)
    domain_col = _find_column(df.columns, ("domain", "name", "id"))
    rmsf_col = _find_column(df.columns, ("rmsf", "avg_per_residue_rmsf"))
    seq_col = _find_column(df.columns, ("sequence", "seq"), required=False)
    out, invalid = {}, {}
    for row in df.itertuples(index=False):
        values = row._asdict()
        domain = str(values[domain_col]).lstrip(">")
        if domain not in sequences:
            continue
        arr = _vector(values[rmsf_col], f"RMSF {domain}")
        if len(arr) != len(sequences[domain]):
            invalid[domain] = f"rmsf_length={len(arr)} expected={len(sequences[domain])}"
            continue
        if seq_col is not None and str(values[seq_col]).upper() != sequences[domain]:
            invalid[domain] = "rmsf_sequence_mismatch"
            continue
        out[domain] = arr
    return out, invalid


def write_long_table(
    path: Path,
    methods: list[MethodScores],
    targets: dict[str, dict[str, np.ndarray]],
    sequences: dict[str, str],
) -> None:
    rows = []
    for target_name, target_map in targets.items():
        for method in methods:
            for domain in sorted(set(target_map) & set(method.scores)):
                target, score = target_map[domain], method.scores[domain]
                if len(target) != len(sequence := sequences[domain]) or len(score) != len(sequence):
                    continue
                rows.extend({
                    "target": target_name, "method": method.method, "domain": domain,
                    "res_idx": i + 1, "aa": sequence[i], "target_value": target[i], "score": score[i],
                } for i in range(len(sequence)))
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be greater than zero")
    if args.linear_batch_size <= 0:
        raise ValueError("--linear-batch-size must be greater than zero")
    if args.force_linear and not args.infer_linear:
        raise ValueError("--force-linear requires --infer-linear")
    source = args.source_dir
    strict_csv = args.strict_csv or source / "mdcath_320K_strict.csv"
    clusters_tsv = args.clusters_tsv or source / "mdcath_cluster_assignments.tsv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort, neq, invalid_neq = load_cohort(strict_csv)
    sequences = dict(zip(cohort["domain"], cohort["sequence"]))
    clusters = load_clusters(clusters_tsv, set(sequences))
    methods, input_audit = load_methods(args, cohort)
    coverage = validate_method_coverage(methods, sequences, set(neq))

    expected_methods = set(ESM_BILSTM_CONDITIONS) | {
        LINEAR_CONDITION, "dynamine_neg_s2", "netsurfp_disorder", "netsurfp_rsa",
        "pegasus_phi_std", "pegasus_psi_std", "pegasus_phi_psi_mean",
        "pegasus_rmsf", "pegasus_neg_lddt",
    }
    loaded_methods = {m.method for m in methods}
    incomplete = sorted(expected_methods - loaded_methods)
    coverage_incomplete = sorted(
        method for method, detail in coverage.items() if detail["n_valid"] != len(neq)
    )
    if args.require_complete and (incomplete or coverage_incomplete):
        raise RuntimeError(
            f"Incomplete benchmark: missing methods={incomplete}; incomplete coverage={coverage_incomplete}"
        )

    targets = {"neq": neq}
    target_coverages = {"neq": coverage}
    target_audit = {
        "neq": {
            "source": _portable_path(strict_csv),
            "n_proteins": len(neq),
            "n_residues": sum(len(values) for values in neq.values()),
        }
    }
    rmsf_audit = {}
    if args.rmsf_csv is not None:
        rmsf, rmsf_audit = load_rmsf(args.rmsf_csv, sequences)
        targets["rmsf"] = rmsf
        target_coverages["rmsf"] = validate_method_coverage(
            methods, sequences, set(rmsf)
        )
        target_audit["rmsf"] = {
            "source": _portable_path(args.rmsf_csv),
            "sha256": _sha256_file(args.rmsf_csv),
            "n_proteins": len(rmsf),
            "n_residues": sum(len(values) for values in rmsf.values()),
        }

    if args.require_complete:
        incomplete_targets = {
            target_name: sorted(
                method for method, detail in target_coverages[target_name].items()
                if detail["n_valid"] != len(target_map)
            )
            for target_name, target_map in targets.items()
        }
        incomplete_targets = {
            target: methods for target, methods in incomplete_targets.items() if methods
        }
        if incomplete_targets:
            raise RuntimeError(f"Incomplete target-specific coverage: {incomplete_targets}")

    per_protein_parts = [
        per_protein_metrics(
            methods, target_name, target_map, sequences, clusters,
            target_coverages[target_name],
        )
        for target_name, target_map in targets.items()
    ]
    per_protein = pd.concat(per_protein_parts, ignore_index=True)
    summary = summarize_methods(
        per_protein, methods, targets, args.n_bootstrap, args.random_seed
    )
    paired = paired_deltas(per_protein, methods, args.n_bootstrap, args.random_seed)

    per_protein.to_csv(args.output_dir / "unified_per_protein_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "unified_method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "unified_paired_deltas.csv", index=False)
    if args.write_long:
        write_long_table(args.output_dir / "unified_per_residue_scores.csv", methods, targets, sequences)

    audit = {
        "cohort": {
            "strict_csv": _portable_path(strict_csv),
            "n_rows": len(cohort),
            "n_valid_neq": len(neq),
            "n_invalid_neq": len(invalid_neq),
            "invalid_neq": invalid_neq,
            "cluster_file": _portable_path(clusters_tsv),
            "n_clusters_all_rows": len(set(clusters.values())),
            "n_clusters_valid_neq": len({clusters[d] for d in neq}),
        },
        "targets": target_audit,
        "method_coverage": coverage,
        "target_method_coverage": target_coverages,
        "input_audit": input_audit,
        "rmsf_invalid": rmsf_audit,
        "expected_methods": sorted(expected_methods),
        "loaded_methods": sorted(loaded_methods),
        "missing_methods": incomplete,
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed,
        "primary_metric": "macro mean of within-protein Spearman correlations",
        "ci": "95% percentile bootstrap resampling MMseqs2 clusters",
        "score_orientation": "higher score means more flexible; DynaMine S2 and PEGASUS mean LDDT are negated",
        "neq_binary_definition": "flexible iff Neq > 1.0",
    }
    with (args.output_dir / "unified_benchmark_audit.json").open("w") as fh:
        json.dump(audit, fh, indent=2)

    print(f"Strict cohort: {len(cohort)} rows; {len(neq)} exact Neq vectors; {len(invalid_neq)} excluded")
    print(f"Loaded {len(methods)} methods; missing expected methods: {incomplete or 'none'}")
    print(f"Results written to {args.output_dir}")
    view = summary.loc[(summary["target"] == "neq") & (summary["scope"] == "shared_loaded_methods")]
    print(view[["display_name", "n_proteins", "macro_mean_spearman", "macro_mean_spearman_ci_low", "macro_mean_spearman_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
