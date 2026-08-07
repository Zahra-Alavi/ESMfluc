#!/usr/bin/env python3
"""
benchmark_vs_baselines.py  (rewritten August 2026)

Compare the seven ESMfluc model conditions against DynaMine on the held-out
ATLAS grouped-v1 test set (208 proteins, ~47,751 residues).

ESMfluc metrics are read from pre-computed pipeline outputs stored under
results/publication_comparable_v2/.  No model reload is required.

  For bilstm_attn conditions (6 of 7): per-residue logit margins are
  extracted from flex_rigid_logit_contributions.npz.  The row sum of C_ij
  (flexible-minus-rigid exact logit contribution) gives the per-residue logit
  margin up to a run-wide classifier bias. Margins are averaged across seeds
  1, 2 and 3, defining a logit-margin ensemble used for rank-based metrics.

  For the esm2_frozen_linear condition: per-residue scores are not stored in
  an npz; its metrics are read directly from the pre-computed pipeline CSV.

DynaMine predictions are fetched from the Bio2Byte msatools REST API
(https://bio2byte.be/msatools/api/).  The API is asynchronous: one POST
submits the job, subsequent GETs poll the queue, and a final GET retrieves
the JSON result.  Results are cached locally so re-runs skip the API call.

DynaMine output is a per-residue backbone S^2 prediction. Low values correspond
to greater dynamics, so the predictor used for ranking is -dynamine_bb. Raw API
outputs are not clipped; a small number can fall outside the physical 0-to-1
interpretation of S^2.

The benchmark reports threshold-independent AUROC and AUPRC plus Spearman
association with continuous Neq. It deliberately does not optimize a binary
decision threshold on the held-out test set.

Usage
-----
    python benchmark_vs_baselines.py \\
        --results_root results/publication_comparable_v2 \\
        --neq_csv      data_splits/atlas_grouped_v1/test_grouped_v1.csv \\
        --fasta        data_splits/atlas_grouped_v1/test_grouped_v1.fasta \\
        --output_dir   results/benchmark

Optional flags
--------------
    --neq_thresh   float   Neq threshold for the flexible class (default 1.0)
    --dynamine_cache path  Pre-cached DynaMine CSV (name,res_idx,dynamine_bb).
                           If the file exists the API call is skipped.
    --skip_dynamine        Skip DynaMine; output ESMfluc-only table.
    --batch_size   int     Sequences per API request (max 50, default 40).
    --poll_interval int    Seconds between queue-status polls (default 20).
    --max_polls    int     Maximum queue checks before giving up (default 90).
    --n_bootstrap  int     Paired protein bootstrap repetitions (default 2000).
    --random_seed  int     Bootstrap seed (default 42).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# ── Bio2Byte msatools API ─────────────────────────────────────────────────────
_B2B_API_BASE = "https://bio2byte.be/msatools/api/"
_B2B_BATCH_MAX = 50

# ── ESMfluc conditions (publication-comparable-v2 pipeline) ──────────────────
# Six bilstm_attn conditions have flex_rigid_logit_contributions.npz.
# The linear condition uses pre-computed metrics from the analysis CSVs.
_NPZ_CONDITIONS = [
    "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn",
    "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn",
    "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
]
_LINEAR_CONDITION = "esm2_frozen_linear"
SEEDS = [1, 2, 3]

# Display names for the paper table
_DISPLAY = {
    "esm2_frozen_linear":       "ESMfluc ESM2 frozen linear",
    "esm2_frozen_bilstm_attn":  "ESMfluc ESM2 frozen BiLSTM-Attn",
    "esm2_top4_bilstm_attn":    "ESMfluc ESM2 top-4 BiLSTM-Attn",
    "esm2_top28_bilstm_attn":   "ESMfluc ESM2 top-28 BiLSTM-Attn",
    "esm3_frozen_bilstm_attn":  "ESMfluc ESM3 frozen BiLSTM-Attn",
    "esm3_top4_bilstm_attn":    "ESMfluc ESM3 top-4 BiLSTM-Attn",
    "esm3_top28_bilstm_attn":   "ESMfluc ESM3 top-28 BiLSTM-Attn",
}


# =============================================================================
# Utilities
# =============================================================================
def _random_token(length: int = 10) -> str:
    """Generate a random alphanumeric session token for the Bio2Byte API."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with open(path) as fh:
        name, buf = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    if name in records:
                        raise ValueError(f"Duplicate FASTA name: {name}")
                    records[name] = "".join(buf)
                name = line[1:].strip()
                if not name:
                    raise ValueError("Empty FASTA header")
                buf = []
            else:
                buf.append(line)
        if name:
            if name in records:
                raise ValueError(f"Duplicate FASTA name: {name}")
            records[name] = "".join(buf)
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records


def load_neq_labels(neq_csv: str | Path, thresh: float) -> pd.DataFrame:
    """Return long DataFrame: name, res_idx (1-based), Neq, label."""
    df = pd.read_csv(neq_csv)
    required = {"name", "sequence", "neq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Neq CSV is missing columns: {sorted(missing)}")
    if df["name"].astype(str).duplicated().any():
        duplicates = df.loc[df["name"].astype(str).duplicated(), "name"].tolist()
        raise ValueError(f"Duplicate proteins in Neq CSV: {duplicates[:5]}")
    rows = []
    for _, row in df.iterrows():
        name = str(row["name"])
        seq = str(row["sequence"])
        raw = row["neq"]
        neq_vals = ast.literal_eval(raw) if isinstance(raw, str) else [float(raw)]
        neq_vals = [float(v) for v in neq_vals]
        if len(neq_vals) != len(seq):
            raise ValueError(
                f"{name}: sequence length {len(seq)} != Neq length {len(neq_vals)}"
            )
        if not np.isfinite(neq_vals).all():
            raise ValueError(f"{name}: Neq contains nonfinite values")
        for i in range(len(seq)):
            rows.append({
                "name": name,
                "res_idx": i + 1,
                "Neq": neq_vals[i],
                "label": int(neq_vals[i] > thresh),
            })
    return pd.DataFrame(rows)


def expected_lengths(df_labels: pd.DataFrame) -> dict[str, int]:
    grouped = df_labels.groupby("name")["res_idx"].agg(["count", "min", "max"])
    bad = grouped[(grouped["min"] != 1) | (grouped["count"] != grouped["max"])]
    if not bad.empty:
        raise ValueError(f"Non-contiguous label positions: {bad.index.tolist()[:5]}")
    return grouped["count"].astype(int).to_dict()


def validate_fasta_against_labels(
    fasta_records: dict[str, str], lengths: dict[str, int]
) -> None:
    if set(fasta_records) != set(lengths):
        raise ValueError(
            "FASTA/label protein mismatch: "
            f"missing={sorted(set(lengths)-set(fasta_records))[:5]}, "
            f"extra={sorted(set(fasta_records)-set(lengths))[:5]}"
        )
    bad = [name for name, length in lengths.items()
           if len(fasta_records[name]) != length]
    if bad:
        raise ValueError(f"FASTA/label length mismatch: {bad[:5]}")


# =============================================================================
# ESMfluc score extraction (no model reload)
# =============================================================================
def _load_npz_margins(npz_path: Path) -> dict[str, np.ndarray]:
    """
    Extract per-residue logit margins from a flex_rigid_logit_contributions.npz.

    Each protein entry is the LxL contribution matrix C_ij
    (flexible-minus-rigid exact logit contribution, query × key).
    Row sum over keys j → per-query logit margin (without bias).
    Sigmoid is monotone so AUROC from row sums equals AUROC from P(flexible).

    Returns {protein_name: float32 array of length L}.
    """
    margins: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as npz:
        protein_names = [str(value) for value in npz["__protein_names__"]]
        matrix_keys = [str(value) for value in npz["__matrix_keys__"]]
        if len(protein_names) != len(matrix_keys):
            raise ValueError(f"{npz_path}: protein/key metadata lengths differ")
        if len(set(protein_names)) != len(protein_names):
            raise ValueError(f"{npz_path}: duplicate protein names")
        for name, key in zip(protein_names, matrix_keys):
            mat = np.asarray(npz[key])
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(f"{npz_path}: {name} matrix is not square: {mat.shape}")
            if not np.isfinite(mat).all():
                raise ValueError(f"{npz_path}: {name} matrix contains nonfinite values")
            margins[name] = mat.sum(axis=1).astype(np.float64)
    return margins


def extract_esmfluc_scores(
    results_root: Path,
    conditions: list[str],
    seeds: list[int],
    lengths: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """
    Average per-residue logit margins across seeds for each condition.
    Returns {condition: DataFrame(name, res_idx, esmfluc_score)}.
    """
    out: dict[str, pd.DataFrame] = {}
    for condition in conditions:
        seed_margins: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            npz_path = (
                results_root / "runs" / condition / f"seed_{seed}"
                / "flex_rigid_logit_contributions.npz"
            )
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing required NPZ: {npz_path}")
            margins = _load_npz_margins(npz_path)
            if set(margins) != set(lengths):
                raise ValueError(
                    f"{condition} seed {seed}: protein set does not match test labels"
                )
            bad = [name for name, length in lengths.items()
                   if len(margins[name]) != length]
            if bad:
                raise ValueError(
                    f"{condition} seed {seed}: score/label length mismatch: {bad[:5]}"
                )
            seed_margins.append(margins)

        rows = []
        for name in sorted(lengths):
            arrays = [sm[name] for sm in seed_margins]
            mean_margin = np.mean(np.stack(arrays, axis=0), axis=0)
            for i, score in enumerate(mean_margin):
                rows.append({"name": name, "res_idx": i + 1, "esmfluc_score": float(score)})

        out[condition] = pd.DataFrame(rows)
        print(f"    {condition}: {len(lengths)} proteins, "
              f"{len(seed_margins)}/{len(seeds)} seeds loaded")
    return out


def load_linear_precomputed(results_root: Path, condition: str) -> dict | None:
    """
    Read pre-computed pooled and macro metrics for the linear condition from
    the analysis CSVs.  Returns a dict suitable for the comparison table, or
    None if the files are not found.
    """
    pooled_csv = results_root / "analysis" / "pooled_residue_performance_across_seeds.csv"
    macro_csv  = results_root / "analysis" / "protein_macro_performance_across_seeds.csv"
    if not pooled_csv.exists():
        return None
    try:
        pooled = pd.read_csv(pooled_csv)
        macro  = pd.read_csv(macro_csv) if macro_csv.exists() else None
        row = pooled[pooled["condition"] == condition]
        if row.empty:
            return None
        r = row.iloc[0]
        m: dict = {
            "AUROC":          float(r["auroc_across_seed_mean"]),
            "AUROC_std":      float(r["auroc_across_seed_std"]),
            "AUPRC":          float(r["auprc_across_seed_mean"]),
            "Spearman":       float(r["neq_score_spearman_across_seed_mean"]),
            "score_source":   "pre-computed (no per-residue scores stored)",
        }
        if macro is not None:
            mrow = macro[macro["condition"] == condition]
            if not mrow.empty:
                mr = mrow.iloc[0]
                m["AUROC_macro"] = float(mr["auroc_mean_across_seed_mean"])
                m["AUROC_macro_std"] = float(mr["auroc_mean_across_seed_std"])
        return m
    except Exception as exc:
        print(f"    [WARN] Could not read pre-computed metrics for {condition}: {exc}")
        return None


# =============================================================================
# DynaMine via Bio2Byte msatools API (async submit → poll → retrieve)
# =============================================================================
def _b2b_submit(session, sequences: dict[str, str], token: str) -> str:
    """POST one batch. Returns hash_id."""
    payload: dict = {"tool_list": ["dynamine"], "token": token}
    payload.update(sequences)
    resp = session.post(_B2B_API_BASE, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    hash_id = data.get("hash_id")
    if not hash_id:
        raise RuntimeError(f"API did not return hash_id: {data}")
    return str(hash_id)


def _b2b_poll(session, hash_id: str, poll_interval: int, max_polls: int) -> list[dict]:
    """Poll queue until complete, then retrieve and return results list."""
    queue_url  = f"{_B2B_API_BASE}queue/{hash_id}/"
    result_url = f"{_B2B_API_BASE}{hash_id}/"

    for attempt in range(max_polls):
        resp = session.get(queue_url, timeout=30, allow_redirects=False)
        # 303 = explicit redirect to results.
        # 200 = results ready (observed in practice; status field may be 200).
        # 202 = still queued/processing.
        if resp.status_code == 303:
            break
        try:
            data = resp.json()
        except Exception:
            time.sleep(poll_interval)
            continue
        if data.get("status") == 500 or data.get("Failure"):
            raise RuntimeError(f"Server-side prediction failure: {data.get('Failure')}")
        http_status = data.get("status", resp.status_code)
        # 202 means still processing; anything else (200, 303) means done.
        if http_status != 202:
            break
        if attempt % 3 == 0:
            remaining = data.get("request_text", "still processing")
            print(f"      poll {attempt + 1}/{max_polls}: {remaining}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"DynaMine API timed out after {max_polls} polls "
            f"({max_polls * poll_interval}s)"
        )

    resp = session.get(result_url, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


def run_dynamine_api(
    fasta_records: dict[str, str],
    token: str,
    batch_size: int = 40,
    poll_interval: int = 20,
    max_polls: int = 90,
) -> pd.DataFrame:
    """
    Submit test sequences to the Bio2Byte DynaMine API in batches.
    Returns long DataFrame: name, res_idx (1-based), dynamine_bb.
    Lower DynaMine backbone S² predictions indicate greater dynamics. Raw
    regression output is retained even when it falls slightly outside [0, 1].
    """
    try:
        import requests
    except ImportError:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "requests", "-q"]
        )
        import requests

    batch_size = min(batch_size, _B2B_BATCH_MAX)
    names = list(fasta_records.keys())
    batches = [names[i: i + batch_size] for i in range(0, len(names), batch_size)]

    session = requests.Session()
    all_rows: list[dict] = []

    for b_idx, batch_names in enumerate(batches, 1):
        batch_seqs = {n: fasta_records[n] for n in batch_names}
        print(f"    Batch {b_idx}/{len(batches)}: {len(batch_seqs)} sequences …")
        try:
            hash_id = _b2b_submit(session, batch_seqs, token)
            results = _b2b_poll(session, hash_id, poll_interval, max_polls)
        except Exception as exc:
            print(f"      [WARN] Batch {b_idx} failed: {exc}")
            for n in batch_names:
                for j in range(len(fasta_records[n])):
                    all_rows.append({"name": n, "res_idx": j + 1, "dynamine_bb": np.nan})
            continue

        # Index returned results by proteinID.
        # The API returns per-predictor keys; DynaMine backbone S² is under
        # "backbone", not "dynamine" ("dynamine" key does not exist).
        result_index: dict[str, list] = {
            r["proteinID"]: r.get("backbone", []) for r in results
        }

        for name in batch_names:
            preds = result_index.get(name, [])
            L_expected = len(fasta_records[name])
            if not preds:
                print(f"      [WARN] No predictions for {name}")
                for j in range(L_expected):
                    all_rows.append({"name": name, "res_idx": j + 1, "dynamine_bb": np.nan})
                continue
            for j, val in enumerate(preds[:L_expected]):
                all_rows.append({
                    "name": name,
                    "res_idx": j + 1,
                    "dynamine_bb": float(val) if val is not None else np.nan,
                })

    return pd.DataFrame(all_rows)


def validate_score_table(
    scores: pd.DataFrame,
    lengths: dict[str, int],
    score_col: str,
    source: str,
) -> dict:
    required = {"name", "res_idx", score_col}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    table = scores[["name", "res_idx", score_col]].copy()
    table["name"] = table["name"].astype(str)
    if table.duplicated(["name", "res_idx"]).any():
        raise ValueError(f"{source} contains duplicate residue keys")
    if not np.isfinite(table[score_col].astype(float)).all():
        raise ValueError(f"{source} contains missing or nonfinite scores")
    expected_keys = {
        (name, position)
        for name, length in lengths.items()
        for position in range(1, length + 1)
    }
    actual_keys = set(zip(table["name"], table["res_idx"].astype(int)))
    if actual_keys != expected_keys:
        raise ValueError(
            f"{source} residue keys do not match labels: "
            f"missing={len(expected_keys-actual_keys)}, extra={len(actual_keys-expected_keys)}"
        )
    values = table[score_col].astype(float)
    return {
        "source": source,
        "protein_count": len(lengths),
        "residue_count": len(table),
        "duplicate_residue_keys": 0,
        "nonfinite_scores": 0,
        "score_min": float(values.min()),
        "score_max": float(values.max()),
        "scores_below_zero": int((values < 0).sum()),
        "scores_above_one": int((values > 1).sum()),
        "exact_key_alignment": True,
    }


def merge_scores_exact(
    labels: pd.DataFrame, scores: pd.DataFrame, score_col: str, source: str
) -> pd.DataFrame:
    merged = labels.merge(
        scores[["name", "res_idx", score_col]],
        on=["name", "res_idx"], how="left", validate="one_to_one",
    )
    if len(merged) != len(labels) or merged[score_col].isna().any():
        raise ValueError(f"{source} failed exact merge with test labels")
    return merged


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    neq_continuous: np.ndarray | None = None,
) -> dict:
    nan_result = {k: np.nan for k in
                  ["AUROC", "AUPRC", "Spearman", "n", "n_pos"]}
    if len(np.unique(y_true)) < 2 or len(scores) < 5:
        return nan_result
    m: dict = {"n": int(len(y_true)), "n_pos": int(y_true.sum())}
    try:
        m["AUROC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        m["AUROC"] = np.nan
    try:
        m["AUPRC"] = float(average_precision_score(y_true, scores))
    except Exception:
        m["AUPRC"] = np.nan
    if neq_continuous is not None:
        try:
            rho, _ = spearmanr(neq_continuous, scores, nan_policy="omit")
            m["Spearman"] = float(rho)
        except Exception:
            m["Spearman"] = np.nan
    else:
        m["Spearman"] = np.nan
    return m


def evaluate_method(
    df_long: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    neq_col: str = "Neq",
) -> tuple[dict, pd.DataFrame]:
    """Global and per-protein metrics. Returns (global_dict, per_protein_df)."""
    valid = df_long[[score_col, label_col, neq_col]].dropna()
    global_m = compute_metrics(
        valid[label_col].values, valid[score_col].values, valid[neq_col].values
    )

    recs = []
    for name, g in df_long.groupby("name"):
        g2 = g[[score_col, label_col, neq_col]].dropna()
        m = compute_metrics(g2[label_col].values, g2[score_col].values, g2[neq_col].values)
        recs.append({"name": name, **m})
    per_protein = pd.DataFrame(recs)

    auc_vals = per_protein["AUROC"].dropna().values
    global_m["AUROC_macro"] = float(np.mean(auc_vals)) if len(auc_vals) else np.nan
    if len(auc_vals) >= 10:
        rng = np.random.default_rng(42)
        boots = [
            rng.choice(auc_vals, len(auc_vals), replace=True).mean()
            for _ in range(2000)
        ]
        global_m["AUROC_macro_CI95_lo"] = float(np.percentile(boots, 2.5))
        global_m["AUROC_macro_CI95_hi"] = float(np.percentile(boots, 97.5))

    return global_m, per_protein


def paired_bootstrap_auroc_difference(
    esm: pd.Series,
    dynamine: pd.Series,
    n_bootstrap: int,
    random_seed: int,
) -> dict:
    paired = pd.concat(
        [esm.rename("esm"), dynamine.rename("dynamine")], axis=1, join="inner"
    ).dropna().sort_index()
    if len(paired) < 10:
        raise ValueError("At least ten paired proteins are required")
    differences = (paired["esm"] - paired["dynamine"]).to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    sampled = differences[indices]
    boot_mean = sampled.mean(axis=1)
    boot_median = np.median(sampled, axis=1)
    return {
        "n_proteins": len(differences),
        "mean_delta_AUROC": float(differences.mean()),
        "mean_delta_AUROC_CI95_lo": float(np.quantile(boot_mean, 0.025)),
        "mean_delta_AUROC_CI95_hi": float(np.quantile(boot_mean, 0.975)),
        "median_delta_AUROC": float(np.median(differences)),
        "median_delta_AUROC_CI95_lo": float(np.quantile(boot_median, 0.025)),
        "median_delta_AUROC_CI95_hi": float(np.quantile(boot_median, 0.975)),
        "proteins_ESMfluc_better": int((differences > 0).sum()),
        "proteins_DynaMine_better": int((differences < 0).sum()),
        "proteins_tied": int((differences == 0).sum()),
        "bootstrap_repetitions": n_bootstrap,
        "bootstrap_unit": "test_protein",
    }


def benjamini_hochberg(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        index = order[rank]
        running = min(running, p[index] * len(p) / (rank + 1))
        adjusted[index] = running
    return np.minimum(adjusted, 1.0)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", required=True,
                    help="Path to results/publication_comparable_v2")
    ap.add_argument("--neq_csv", required=True,
                    help="CSV with columns: name, sequence, neq (list string)")
    ap.add_argument("--fasta", required=True,
                    help="FASTA of test sequences for DynaMine submission")
    ap.add_argument("--output_dir", default="results/benchmark")
    ap.add_argument("--neq_thresh", type=float, default=1.0,
                    help="Neq > thresh → flexible (label 1). Default 1.0")
    ap.add_argument("--dynamine_cache", default=None,
                    help="Pre-cached DynaMine CSV (name, res_idx, dynamine_bb). "
                         "Skips API call if the file exists.")
    ap.add_argument("--skip_dynamine", action="store_true",
                    help="Skip DynaMine entirely; output ESMfluc-only table.")
    ap.add_argument("--batch_size", type=int, default=40,
                    help=f"Sequences per API request (max {_B2B_BATCH_MAX}, default 40).")
    ap.add_argument("--poll_interval", type=int, default=20,
                    help="Seconds between queue polls (default 20).")
    ap.add_argument("--max_polls", type=int, default=90,
                    help="Maximum queue polls before giving up (default 90).")
    ap.add_argument("--n_bootstrap", type=int, default=2000,
                    help="Paired protein bootstrap repetitions (default 2000).")
    ap.add_argument("--random_seed", type=int, default=42,
                    help="Random seed for protein bootstrap (default 42).")
    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n_bootstrap must be at least 100")
    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_token = _random_token(10)

    print("=" * 72)
    print("ESMfluc vs DynaMine — publication benchmark")
    print("=" * 72)

    # ── 1. Ground truth ──────────────────────────────────────────────────────
    print("\n[1/4] Loading test-set labels …")
    df_labels = load_neq_labels(args.neq_csv, args.neq_thresh)
    lengths = expected_lengths(df_labels)
    fasta_records = parse_fasta(args.fasta)
    validate_fasta_against_labels(fasta_records, lengths)
    n_prot = df_labels["name"].nunique()
    n_res  = len(df_labels)
    n_pos  = int(df_labels["label"].sum())
    print(f"  {n_prot} proteins, {n_res} residues")
    print(f"  Flexible (Neq > {args.neq_thresh}): {n_pos / n_res * 100:.1f}%")

    # ── 2. ESMfluc scores ────────────────────────────────────────────────────
    print("\n[2/4] Extracting ESMfluc per-residue scores …")
    esmfluc_dfs = extract_esmfluc_scores(
        results_root, _NPZ_CONDITIONS, SEEDS, lengths
    )
    score_audits = {
        condition: validate_score_table(
            scores, lengths, "esmfluc_score", f"ESMfluc {condition}"
        )
        for condition, scores in esmfluc_dfs.items()
    }

    # Pre-computed metrics for the linear condition (no npz)
    linear_precomputed = load_linear_precomputed(results_root, _LINEAR_CONDITION)
    if linear_precomputed:
        print(f"    {_LINEAR_CONDITION}: pre-computed metrics loaded "
              f"(AUROC={linear_precomputed['AUROC']:.4f})")
    else:
        print(f"    [WARN] Could not load pre-computed metrics for {_LINEAR_CONDITION}")

    # ── 3. DynaMine ──────────────────────────────────────────────────────────
    df_dynamine: pd.DataFrame | None = None
    cache_path: Path | None = None
    if not args.skip_dynamine:
        cache_path = (
            Path(args.dynamine_cache)
            if args.dynamine_cache
            else out_dir / "cache_dynamine.csv"
        )
        if cache_path.exists():
            print(f"\n[3/4] Loading cached DynaMine predictions from {cache_path} …")
            df_dynamine = pd.read_csv(cache_path)
            print(f"  {df_dynamine['name'].nunique()} proteins, "
                  f"{df_dynamine['dynamine_bb'].notna().sum()} valid scores")
        else:
            print(f"\n[3/4] Fetching DynaMine predictions (token: {session_token}) …")
            test_names = set(df_labels["name"].unique())
            fasta_filtered = {n: s for n, s in fasta_records.items() if n in test_names}
            print(f"  Submitting {len(fasta_filtered)} sequences in batches of "
                  f"{min(args.batch_size, _B2B_BATCH_MAX)} …")
            df_dynamine = run_dynamine_api(
                fasta_filtered,
                token=session_token,
                batch_size=args.batch_size,
                poll_interval=args.poll_interval,
                max_polls=args.max_polls,
            )
            df_dynamine.to_csv(cache_path, index=False)
            n_valid = df_dynamine["dynamine_bb"].notna().sum()
            print(f"  Done — {n_valid} scores; cached to {cache_path}")
        score_audits["DynaMine"] = validate_score_table(
            df_dynamine, lengths, "dynamine_bb", "DynaMine cache"
        )
        above_one = score_audits["DynaMine"]["scores_above_one"]
        if above_one:
            print(f"  Note: {above_one} raw DynaMine predictions exceed 1; "
                  "values are retained because rank metrics do not require clipping")
    else:
        print("\n[3/4] DynaMine skipped (--skip_dynamine).")

    # ── 4. Compute and save metrics ──────────────────────────────────────────
    print("\n[4/4] Computing metrics …")

    all_global: list[dict] = []
    all_per_protein: dict[str, pd.DataFrame] = {}

    # DynaMine
    if df_dynamine is not None:
        df_dm = merge_scores_exact(
            df_labels, df_dynamine, "dynamine_bb", "DynaMine"
        )
        # Lower S² means greater dynamics. Negation preserves the full raw
        # ordering without implying that every API prediction lies in [0, 1].
        df_dm["dynamine_flex"] = -df_dm["dynamine_bb"]
        g_m, pp_df = evaluate_method(df_dm, score_col="dynamine_flex")
        g_m["method"] = "DynaMine"
        g_m["score_source"] = "Bio2Byte msatools API (negative raw backbone S² prediction)"
        g_m["target_relation"] = "external sequence-only NMR-order-parameter predictor; not trained on ATLAS Neq"
        all_global.append(g_m)
        all_per_protein["DynaMine"] = pp_df

    # ESMfluc bilstm_attn conditions (from npz)
    for condition in _NPZ_CONDITIONS:
        if condition not in esmfluc_dfs:
            continue
        df_esm = merge_scores_exact(
            df_labels, esmfluc_dfs[condition], "esmfluc_score", condition
        )
        g_m, pp_df = evaluate_method(df_esm, score_col="esmfluc_score")
        g_m["method"] = _DISPLAY.get(condition, condition)
        g_m["score_source"] = "mean across three seed-specific row-sum logit margins from exact C_ij"
        g_m["target_relation"] = "supervised ATLAS Neq binary target"
        all_global.append(g_m)
        all_per_protein[g_m["method"]] = pp_df

    # ESMfluc linear (pre-computed)
    if linear_precomputed:
        g_m = dict(linear_precomputed)
        g_m["method"] = _DISPLAY.get(_LINEAR_CONDITION, _LINEAR_CONDITION)
        g_m["target_relation"] = "supervised ATLAS Neq binary target"
        g_m["comparison_note"] = (
            "seed-summary metrics only; no residue-level scores available for paired comparison"
        )
        all_global.append(g_m)
        # No per-protein df; Wilcoxon test will skip it

    # Save
    df_global = pd.DataFrame(all_global).set_index("method")
    df_global.to_csv(out_dir / "benchmark_global_metrics.csv")

    if all_per_protein:
        per_prot_long = pd.concat(
            [df.assign(method=m) for m, df in all_per_protein.items()],
            ignore_index=True,
        )
        per_prot_long.to_csv(out_dir / "benchmark_per_protein_metrics.csv", index=False)

    # Paired protein comparisons: synchronized protein bootstrap plus Wilcoxon.
    comparison_rows: list[dict] = []
    if "DynaMine" in all_per_protein:
        dm_pp = all_per_protein["DynaMine"].set_index("name")["AUROC"].dropna()
        for method, pp_df in all_per_protein.items():
            if method == "DynaMine":
                continue
            esm_pp = pp_df.set_index("name")["AUROC"].dropna()
            summary = paired_bootstrap_auroc_difference(
                esm_pp, dm_pp, args.n_bootstrap, args.random_seed
            )
            common = dm_pp.index.intersection(esm_pp.index).sort_values()
            diff = esm_pp.loc[common].to_numpy() - dm_pp.loc[common].to_numpy()
            stat, pval = wilcoxon(diff, alternative="greater")
            comparison_rows.append({
                "ESMfluc_condition": method,
                **summary,
                "wilcoxon_stat": float(stat),
                "wilcoxon_p_one_sided": float(pval),
            })

    if comparison_rows:
        adjusted = benjamini_hochberg(
            [row["wilcoxon_p_one_sided"] for row in comparison_rows]
        )
        for row, q_value in zip(comparison_rows, adjusted):
            row["wilcoxon_BH_FDR_q"] = float(q_value)
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(
            out_dir / "benchmark_paired_vs_dynamine.csv", index=False
        )
        # Retain the historical filename, now with the corrected, richer schema.
        comparison.to_csv(
            out_dir / "benchmark_wilcoxon_vs_dynamine.csv", index=False
        )

    npz_provenance = []
    for condition in _NPZ_CONDITIONS:
        for seed in SEEDS:
            path = (
                results_root / "runs" / condition / f"seed_{seed}"
                / "flex_rigid_logit_contributions.npz"
            ).resolve()
            stat = path.stat()
            npz_provenance.append({
                "condition": condition, "seed": seed, "path": str(path),
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            })
    methodology = {
        "schema": "esmfluc.dynamine_benchmark.methodology.v2",
        "test_target": f"ATLAS Neq > {args.neq_thresh}",
        "test_proteins": n_prot,
        "test_residues": n_res,
        "reported_metrics": ["AUROC", "AUPRC", "Spearman", "protein_macro_AUROC"],
        "excluded_metrics": {
            "F1": "omitted because no decision threshold was selected independently of test data",
            "MCC": "omitted because no decision threshold was selected independently of test data",
        },
        "ESMfluc_ensemble": (
            "arithmetic mean of the three seed-specific flexible-minus-rigid "
            "logit margins reconstructed as row sums of exact contribution matrices; "
            "run-wide classifier biases do not affect residue ranking"
        ),
        "DynaMine_score": (
            "negative raw Bio2Byte backbone S2 prediction; no clipping or calibration"
        ),
        "DynaMine_target_caveat": (
            "DynaMine is trained on NMR-derived backbone order parameters, not ATLAS Neq"
        ),
        "linear_baseline_caveat": (
            "only precomputed seed-summary metrics are available; it is excluded "
            "from residue-level paired comparisons"
        ),
        "paired_inference": {
            "unit": "test protein",
            "bootstrap_repetitions": args.n_bootstrap,
            "random_seed": args.random_seed,
            "interval": "paired percentile 95% interval for mean and median AUROC difference",
            "test": "one-sided paired Wilcoxon signed-rank, BH corrected across six conditions",
        },
        "sources": {
            "neq_csv": {"path": str(Path(args.neq_csv).resolve()), "sha256": sha256(args.neq_csv)},
            "fasta": {"path": str(Path(args.fasta).resolve()), "sha256": sha256(args.fasta)},
            "dynamine_cache": (
                {"path": str(cache_path.resolve()), "sha256": sha256(cache_path)}
                if cache_path is not None and cache_path.is_file() else None
            ),
            "ESMfluc_npz": npz_provenance,
        },
        "score_table_audits": score_audits,
    }
    methodology_path = out_dir / "benchmark_methodology.json"
    methodology_path.write_text(
        json.dumps(methodology, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    required_outputs = [
        "benchmark_global_metrics.csv", "benchmark_per_protein_metrics.csv",
        "benchmark_paired_vs_dynamine.csv", "benchmark_wilcoxon_vs_dynamine.csv",
        "benchmark_methodology.json",
    ] if df_dynamine is not None else [
        "benchmark_global_metrics.csv", "benchmark_per_protein_metrics.csv",
        "benchmark_methodology.json",
    ]
    checks = {
        "fixed_test_counts": n_prot == 208 and n_res == 47751,
        "six_complete_ESMfluc_score_tables": len(esmfluc_dfs) == 6,
        "all_score_tables_exactly_aligned": all(
            item["exact_key_alignment"] for item in score_audits.values()
        ),
        "test_optimized_threshold_metrics_absent": not {"F1", "MCC"} & set(df_global.columns),
        "expected_global_rows": len(df_global) == (
            6 + int(linear_precomputed is not None) + int(df_dynamine is not None)
        ),
        "expected_paired_comparisons": len(comparison_rows) == (
            6 if df_dynamine is not None else 0
        ),
        "paired_intervals_above_zero": all(
            row["mean_delta_AUROC_CI95_lo"] > 0 for row in comparison_rows
        ),
        "required_outputs_present": all((out_dir / name).is_file() for name in required_outputs),
    }
    completion = {
        "schema": "esmfluc.dynamine_benchmark.audit.v2",
        "passed": all(checks.values()),
        "checks": checks,
        "required_outputs": required_outputs,
        "output_sha256": {
            name: sha256(out_dir / name) for name in required_outputs
        },
    }
    (out_dir / "benchmark_complete_audit.json").write_text(
        json.dumps(completion, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not completion["passed"]:
        raise ValueError("Benchmark completion audit failed")

    # ── Print summary ─────────────────────────────────────────────────────────
    show_cols = [c for c in ["AUROC", "AUROC_macro", "AUPRC", "Spearman"]
                 if c in df_global.columns]
    df_show = df_global[show_cols].sort_values("AUROC", ascending=False, na_position="last")

    print()
    print("=" * 80)
    print(f"BENCHMARK  (atlas_grouped_v1 test set, 208 proteins, Neq > {args.neq_thresh} = flexible)")
    print("=" * 80)
    hdr = f"{'Method':<36}  {'AUROC':>7}  {'macro':>7}  {'AUPRC':>7}  {'Spearman':>9}"
    print(hdr)
    print("-" * len(hdr))
    for method, row in df_show.iterrows():
        auroc_str = f"{row['AUROC']:.4f}" if not np.isnan(row.get("AUROC", np.nan)) else "  —   "
        macro_str = f"{row['AUROC_macro']:.4f}" if "AUROC_macro" in row and not np.isnan(row["AUROC_macro"]) else "  —   "
        auprc_str = f"{row.get('AUPRC', np.nan):.4f}" if not np.isnan(row.get("AUPRC", np.nan)) else "  —   "
        spear_str = f"{row.get('Spearman', np.nan):.4f}" if not np.isnan(row.get("Spearman", np.nan)) else "   —    "
        print(f"  {method:<34}  {auroc_str:>7}  {macro_str:>7}  {auprc_str:>7}  {spear_str:>9}")

    if comparison_rows:
        print()
        print("Paired protein AUROC comparison against DynaMine:")
        for r in comparison_rows:
            sig = ("***" if r["wilcoxon_BH_FDR_q"] < 0.001
                   else "**" if r["wilcoxon_BH_FDR_q"] < 0.01
                   else "*" if r["wilcoxon_BH_FDR_q"] < 0.05
                   else "ns")
            print(f"  {r['ESMfluc_condition']:<36}  "
                  f"mean Δ={r['mean_delta_AUROC']:+.4f} "
                  f"[{r['mean_delta_AUROC_CI95_lo']:+.4f}, "
                  f"{r['mean_delta_AUROC_CI95_hi']:+.4f}]  "
                  f"q={r['wilcoxon_BH_FDR_q']:.3g} {sig}  "
                  f"(n={r['n_proteins']})")

    print(f"\nOutputs saved to: {out_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ── legacy stub to prevent import errors from old code references ─────────────
