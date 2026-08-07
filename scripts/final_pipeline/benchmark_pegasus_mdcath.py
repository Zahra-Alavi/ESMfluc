#!/usr/bin/env python3
"""
benchmark_pegasus_mdcath.py

Cross-dataset ATLAS → mdCATH 320K generalization benchmark.

CONTEXT
-------
PEGASUS (Vander Meersche et al., Protein Science 2025, doi:10.1002/pro.70221)
trains via 10-fold cross-validation over all 1,369 ATLAS proteins, so its
released model has effectively seen every ATLAS protein. A direct ATLAS-test
comparison would be biased in PEGASUS's favour by an unknown and unbounded
amount. This script avoids that entirely.

Both ESMfluc and PEGASUS are evaluated on mdCATH 320K CATH domains that share
no PDB entry with any ATLAS protein. Neither model has trained on these
proteins — the comparison is symmetric and uncontaminated.

PIPELINE STAGES (each stage is cached; re-runs skip completed stages)
----------------------------------------------------------------------
  Stage 1 – filter
      Remove mdCATH 320K test proteins whose 4-char PDB code appears in any
      ATLAS protein (train + validation + test splits combined). Writes a
      filtered CSV and FASTA.

  Stage 2 – infer
      Run the ATLAS-trained ESMfluc model (esm2_frozen_bilstm_attn, seeds 1-3)
      on the filtered FASTA via Attention/get_attn.py. Per-residue P(flexible)
      is averaged across the 3 seeds.

  Stage 3 – pegasus
      Pull the PEGASUS Docker image (dsimb/pegasus) if not present, download
      the model weights if not present, then run PEGASUS on the filtered FASTA.
      PEGASUS predicts per-residue RMSF; higher RMSF = more flexible.

  Stage 4 – compare
      For each protein compute Spearman ρ between per-residue predictor scores
      and the continuous mdCATH 320K Neq. Report macro-average ρ, bootstrapped
      95 % CI (2 000 resamples over proteins), and a paired Wilcoxon signed-rank
      test comparing ESMfluc to PEGASUS.

USAGE (from scripts/final_pipeline/)
-------------------------------------
    python benchmark_pegasus_mdcath.py

All paths auto-resolve relative to this script. Override with flags:

    --mdcath_test        CSV with columns [domain, sequence, neq]  (320K)
    --atlas_splits_dir   dir containing {train,validation,test}_grouped_v1.csv
    --esmfluc_runs_dir   dir containing esm2_frozen_bilstm_attn/seed_{1,2,3}
    --pegasus_models_dir dir for PEGASUS weights  (default: results/pegasus/models)
    --output_dir         (default: results/benchmark_pegasus_mdcath)
    --skip_pegasus       Skip PEGASUS; produce ESMfluc-only table
    --stages             Comma-separated subset of stages to run, e.g. filter,infer
    --n_bootstrap        Paired bootstrap resamples  (default: 2000)
    --random_seed        (default: 42)
    --device             cuda or cpu for ESMfluc inference  (default: cuda)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# ---------------------------------------------------------------------------
# Paths (auto-resolved from this file's location)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE  = _SCRIPT_DIR.parent.parent          # ESMfluc/
_DATA_DIR   = _WORKSPACE / "data"

_DEFAULT_MDCATH_TEST   = _DATA_DIR / "mdcath" / "per_temperature" / "test_320K.csv"
_DEFAULT_SPLITS_DIR    = _SCRIPT_DIR / "data_splits" / "atlas_grouped_v1"
_DEFAULT_RUNS_DIR      = _SCRIPT_DIR / "results" / "publication_comparable_v2" / "runs"
_DEFAULT_OUTPUT_DIR    = _SCRIPT_DIR / "results" / "benchmark_pegasus_mdcath"

# All bilstm_attn conditions — run all, so the table shows ESM2 frozen through
# ESM3 top-28 and the reviewer can see the full model family vs PEGASUS.
_ALL_CONDITIONS = [
    "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn",
    "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn",
    "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
]
_SEEDS                 = [1, 2, 3]
_PEGASUS_IMAGE         = "dsimb/pegasus"
_PEGASUS_WEIGHTS_URL   = "https://dsimb.inserm.fr/PEGASUS/models/pegasus_weights.tar.gz"

_CONDITION_DISPLAY = {
    "esm2_frozen_bilstm_attn":  "ESMfluc ESM2 frozen",
    "esm2_top4_bilstm_attn":    "ESMfluc ESM2 top-4",
    "esm2_top28_bilstm_attn":   "ESMfluc ESM2 top-28",
    "esm3_frozen_bilstm_attn":  "ESMfluc ESM3 frozen",
    "esm3_top4_bilstm_attn":    "ESMfluc ESM3 top-4",
    "esm3_top28_bilstm_attn":   "ESMfluc ESM3 top-28",
}


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ESMfluc vs PEGASUS on contamination-free mdCATH 320K benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mdcath_test", type=Path, default=_DEFAULT_MDCATH_TEST)
    p.add_argument("--atlas_splits_dir", type=Path, default=_DEFAULT_SPLITS_DIR)
    p.add_argument("--esmfluc_runs_dir", type=Path, default=_DEFAULT_RUNS_DIR)
    p.add_argument("--pegasus_models_dir", type=Path,
                   default=_DEFAULT_OUTPUT_DIR / "pegasus_models")
    p.add_argument("--output_dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--skip_pegasus", action="store_true",
                   help="Skip PEGASUS; produce ESMfluc-only table.")
    p.add_argument("--stages", type=str, default="filter,infer,pegasus,compare",
                   help="Comma-separated subset of stages to run.")
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="Compute device for ESMfluc inference.")
    return p


# ===========================================================================
# Stage 1 – filter
# ===========================================================================

def _load_atlas_pdb_codes(splits_dir: Path) -> set[str]:
    """Return the set of lower-case 4-char PDB codes from all three ATLAS splits."""
    pdb_codes: set[str] = set()
    for split in ("train", "validation", "test"):
        csv = splits_dir / f"{split}_grouped_v1.csv"
        if not csv.is_file():
            raise FileNotFoundError(f"Missing ATLAS split: {csv}")
        df = pd.read_csv(csv, usecols=["name"])
        # ATLAS names are like "1btk_A"; first 4 chars = PDB code
        pdb_codes.update(df["name"].str[:4].str.lower())
    return pdb_codes


def run_filter(args: argparse.Namespace) -> Path:
    """
    Stage 1: filter mdCATH 320K test proteins by ATLAS PDB-code overlap.

    Returns path to the filtered FASTA (also writes the filtered CSV).
    """
    out_csv   = args.output_dir / "mdcath_320K_filtered.csv"
    out_fasta = args.output_dir / "mdcath_320K_filtered.fasta"

    if out_csv.is_file() and out_fasta.is_file():
        print(f"[filter] Cached — {out_csv.name} already exists. Skipping.")
        return out_fasta

    print("[filter] Loading mdCATH 320K test set …")
    if not args.mdcath_test.is_file():
        raise FileNotFoundError(f"mdCATH test CSV not found: {args.mdcath_test}")
    mdcath = pd.read_csv(args.mdcath_test)
    required = {"domain", "sequence", "neq"}
    missing = required - set(mdcath.columns)
    if missing:
        raise ValueError(f"mdCATH CSV missing columns: {sorted(missing)}")

    print("[filter] Loading ATLAS PDB codes from all splits …")
    atlas_pdb = _load_atlas_pdb_codes(args.atlas_splits_dir)

    # Filter: remove any domain whose first-4-char PDB code matches ATLAS
    domain_pdb = mdcath["domain"].str[:4].str.lower()
    mask_clean = ~domain_pdb.isin(atlas_pdb)
    n_removed  = (~mask_clean).sum()
    filtered   = mdcath[mask_clean].copy().reset_index(drop=True)

    print(
        f"[filter] mdCATH 320K: {len(mdcath)} proteins  "
        f"→ removed {n_removed} PDB-code overlaps with ATLAS  "
        f"→ {len(filtered)} clean proteins"
    )
    if len(filtered) == 0:
        raise RuntimeError("No proteins remain after filtering.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_csv, index=False)
    print(f"[filter] Wrote filtered CSV → {out_csv}")

    with open(out_fasta, "w") as fh:
        for _, row in filtered.iterrows():
            fh.write(f">{row['domain']}\n{row['sequence']}\n")
    print(f"[filter] Wrote filtered FASTA → {out_fasta}  ({len(filtered)} entries)")

    # Provenance
    prov = {
        "n_total_mdcath":   int(len(mdcath)),
        "n_atlas_pdb_codes": int(len(atlas_pdb)),
        "n_removed_pdb_overlap": int(n_removed),
        "n_clean": int(len(filtered)),
        "filter_criterion": "first-4-char PDB code of mdCATH domain matches any ATLAS protein name",
        "atlas_splits_dir": str(args.atlas_splits_dir),
        "mdcath_test": str(args.mdcath_test),
    }
    with open(args.output_dir / "filter_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)

    return out_fasta


# ===========================================================================
# Stage 2 – ESMfluc inference
# ===========================================================================

def _run_get_attn(
    checkpoint: Path,
    fasta: Path,
    output_json: Path,
    device: str,
    is_esm3: bool = False,
) -> None:
    """Invoke Attention/get_attn.py as a subprocess for one seed."""
    get_attn = _SCRIPT_DIR / "Attention" / "get_attn.py"
    if not get_attn.is_file():
        raise FileNotFoundError(f"get_attn.py not found at {get_attn}")

    cmd = [
        sys.executable, str(get_attn),
        "--checkpoint",    str(checkpoint),
        "--fasta_file",    str(fasta),
        "--output",        str(output_json),
        "--architecture",  "bilstm_attention",
        "--task_type",     "classification",
        "--num_classes",   "2",
    ]
    if is_esm3:
        cmd.append("--is_esm3")
    else:
        cmd += ["--esm_model", "esm2_t33_650M_UR50D"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SCRIPT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    print(f"  [infer] Running: checkpoint={checkpoint.parent.name} (esm3={is_esm3})")
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"get_attn.py failed (exit {result.returncode}) for {checkpoint}"
        )


def _parse_attn_json(json_path: Path) -> dict[str, list[float]]:
    """
    Parse get_attn.py JSON output.
    Returns {domain_name: [P(flexible) per residue]}.
    """
    with open(json_path) as fh:
        records = json.load(fh)
    out: dict[str, list[float]] = {}
    for rec in records:
        name = rec["name"]
        scores = rec.get("flexible_scores")
        if scores is None:
            raise KeyError(f"No 'flexible_scores' in get_attn.py output for {name}")
        out[name] = [float(s) for s in scores]
    return out


def _infer_one_condition(
    condition: str,
    filtered_fasta: Path,
    infer_dir: Path,
    runs_dir: Path,
    device: str,
) -> dict[str, list[float]]:
    """Run get_attn.py for all seeds of one condition and return seed-averaged scores."""
    merged_path = infer_dir / f"{condition}_mean_scores.json"
    if merged_path.is_file():
        print(f"  [infer] {condition}: cached — loading {merged_path.name}.")
        with open(merged_path) as fh:
            return json.load(fh)

    # Detect backbone from condition name
    is_esm3 = condition.startswith("esm3")
    per_seed: dict[int, dict[str, list[float]]] = {}
    for seed in _SEEDS:
        ckpt = runs_dir / condition / f"seed_{seed}" / "best_model.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"ESMfluc checkpoint not found: {ckpt}")
        out_json = infer_dir / f"{condition}_seed_{seed}_scores.json"
        if not out_json.is_file():
            _run_get_attn(ckpt, filtered_fasta, out_json, device, is_esm3=is_esm3)
        else:
            print(f"  [infer] {condition} seed {seed}: cached.")
        per_seed[seed] = _parse_attn_json(out_json)

    domains = list(per_seed[_SEEDS[0]].keys())
    for s in _SEEDS[1:]:
        if set(per_seed[s].keys()) != set(domains):
            raise ValueError(
                f"{condition} seed {s} has different domain names than seed {_SEEDS[0]}"
            )
    mean_scores: dict[str, list[float]] = {}
    for dom in domains:
        arrs = [np.array(per_seed[s][dom]) for s in _SEEDS]
        lengths = {len(a) for a in arrs}
        if len(lengths) != 1:
            raise ValueError(f"{dom}: mismatched lengths across seeds in {condition}: {lengths}")
        mean_scores[dom] = np.mean(arrs, axis=0).tolist()

    with open(merged_path, "w") as fh:
        json.dump(mean_scores, fh)
    print(f"  [infer] {condition}: averaged over {len(_SEEDS)} seeds → {merged_path.name}")
    return mean_scores


def run_infer(
    args: argparse.Namespace,
    filtered_fasta: Path,
) -> dict[str, dict[str, list[float]]]:
    """
    Stage 2: run ESMfluc inference for every condition in _ALL_CONDITIONS.
    Returns {condition: {domain: [mean P(flexible) per residue]}}.
    """
    infer_dir = args.output_dir / "esmfluc_inference"
    infer_dir.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, dict[str, list[float]]] = {}
    for condition in _ALL_CONDITIONS:
        print(f"[infer] Condition: {condition}")
        try:
            all_scores[condition] = _infer_one_condition(
                condition, filtered_fasta, infer_dir, args.esmfluc_runs_dir, args.device
            )
        except (RuntimeError, ImportError) as exc:
            print(f"  [infer] WARNING: skipping {condition} — {exc}")
    if not all_scores:
        raise RuntimeError("No ESMfluc conditions produced scores. Check backbone availability.")
    return all_scores


# ===========================================================================
# Stage 3 – PEGASUS
# ===========================================================================

def _ensure_pegasus_image() -> None:
    """Pull dsimb/pegasus from Docker Hub if not already present."""
    result = subprocess.run(
        ["docker", "images", "-q", _PEGASUS_IMAGE],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"[pegasus] Docker image '{_PEGASUS_IMAGE}' already present.")
        return
    print(f"[pegasus] Pulling Docker image '{_PEGASUS_IMAGE}' …")
    subprocess.run(["docker", "pull", _PEGASUS_IMAGE], check=True)


def _ensure_pegasus_weights(models_dir: Path) -> None:
    """Download and extract PEGASUS weights if not already present."""
    models_dir.mkdir(parents=True, exist_ok=True)
    # PEGASUS expects its weight files directly inside models_dir.
    # We detect presence by checking for any .pt or .pkl file.
    weight_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pkl"))
    if weight_files:
        print(f"[pegasus] Weights already present in {models_dir} ({len(weight_files)} files).")
        return

    tarball = models_dir / "pegasus_weights.tar.gz"
    if not tarball.is_file():
        print(f"[pegasus] Downloading PEGASUS weights to {tarball} …")
        print(f"          URL: {_PEGASUS_WEIGHTS_URL}")
        # Try aria2c first (faster multi-connection), fall back to wget/curl
        if subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0:
            subprocess.run(
                ["aria2c", "-d", str(models_dir), _PEGASUS_WEIGHTS_URL], check=True
            )
        elif subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
            subprocess.run(
                ["wget", "-O", str(tarball), _PEGASUS_WEIGHTS_URL], check=True
            )
        else:
            subprocess.run(
                ["curl", "-L", "-o", str(tarball), _PEGASUS_WEIGHTS_URL], check=True
            )

    print(f"[pegasus] Extracting weights …")
    subprocess.run(
        ["tar", "-xzvf", str(tarball), "-C", str(models_dir)], check=True
    )
    print("[pegasus] Weights extracted.")


def _run_pegasus_docker(
    fasta: Path,
    models_dir: Path,
    output_dir: Path,
) -> Path:
    """Run PEGASUS via Docker on the filtered FASTA. Returns the output directory."""
    peg_out = output_dir / "pegasus_raw_output"
    # PEGASUS creates a unique subdirectory inside output; detect it after the run.
    done_flag = peg_out / ".pegasus_done"
    if done_flag.is_file():
        print(f"[pegasus] Raw output cached — skipping Docker run.")
        return peg_out

    peg_out.mkdir(parents=True, exist_ok=True)

    # Docker volumes: input dir, output dir, models dir
    fasta_abs    = fasta.resolve()
    models_abs   = models_dir.resolve()
    peg_out_abs  = peg_out.resolve()
    input_dir    = fasta_abs.parent

    cmd = [
        "docker", "run", "--rm",
        "-e", f"USER_ID={os.getuid()}",
        "-e", f"GROUP_ID={os.getgid()}",
        "--gpus", "all",
        "-v", f"{input_dir}:/input",
        "-v", f"{peg_out_abs}:/output",
        "-v", f"{models_abs}:/models",
        _PEGASUS_IMAGE,
        "-i", f"/input/{fasta_abs.name}",
        "--output_dir", "/output",
        "--models_dir", "/models",
        "-d", "gpu",
    ]
    print(f"[pegasus] Running Docker: {' '.join(cmd[:8])} …")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"PEGASUS Docker run failed (exit {result.returncode})."
        )
    print(f"[pegasus] PEGASUS finished in {elapsed:.0f} s.")

    done_flag.touch()
    return peg_out


def _parse_pegasus_output(peg_raw_dir: Path) -> dict[str, list[float]]:
    """
    Parse PEGASUS predictions from the raw output directory.

    PEGASUS creates:
      {peg_raw_dir}/{job_id}/id_mapping.tsv          Generated_ID <tab> Original_ID
      {peg_raw_dir}/{job_id}/predictions/{GID}_predictions.tsv
        columns: position  RMSF_mean  RMSF_std  Std_Phi_mean ...

    Returns {original_domain: [RMSF per residue]}.
    """
    # Locate job directory (PEGASUS creates one unique subdirectory)
    subdirs = [d for d in peg_raw_dir.iterdir() if d.is_dir() and (d / "id_mapping.tsv").is_file()]
    if not subdirs:
        raise FileNotFoundError(
            f"No PEGASUS job directory with id_mapping.tsv found in {peg_raw_dir}. "
            "Check that the Docker run completed successfully."
        )
    job_dir = subdirs[0]  # take the first (should be only one)
    print(f"[pegasus] Parsing output from {job_dir.name} …")

    id_map = pd.read_csv(job_dir / "id_mapping.tsv", sep="\t",
                         names=["Generated_ID", "Original_ID"], header=0)
    id_map = id_map.dropna()

    out: dict[str, list[float]] = {}
    missing: list[str] = []
    pred_dir = job_dir / "predictions"

    for _, row in id_map.iterrows():
        gid     = str(row["Generated_ID"]).strip()
        orig_id = str(row["Original_ID"]).strip()
        tsv     = pred_dir / f"{gid}_predictions.tsv"
        if not tsv.is_file():
            missing.append(orig_id)
            continue
        df_pred = pd.read_csv(tsv, sep="\t")
        # Normalize column names (handle potential casing differences)
        df_pred.columns = [c.strip() for c in df_pred.columns]
        rmsf_col = next(
            (c for c in df_pred.columns if c.lower().startswith("rmsf") and "mean" in c.lower()),
            None
        )
        if rmsf_col is None:
            # fallback: first column with "rmsf" (case-insensitive)
            rmsf_col = next(
                (c for c in df_pred.columns if "rmsf" in c.lower()), None
            )
        if rmsf_col is None:
            raise KeyError(
                f"Cannot find RMSF column in {tsv}. Columns: {df_pred.columns.tolist()}"
            )
        out[orig_id] = df_pred[rmsf_col].tolist()

    if missing:
        print(f"[pegasus] WARNING: missing predictions for {len(missing)} proteins: {missing[:5]}")
    print(f"[pegasus] Parsed RMSF for {len(out)} proteins.")
    return out


def run_pegasus(args: argparse.Namespace, filtered_fasta: Path) -> dict[str, list[float]] | None:
    """
    Stage 3: run PEGASUS and return {domain: [RMSF per residue]}.
    Returns None if --skip_pegasus is set.
    """
    if args.skip_pegasus:
        print("[pegasus] Skipped (--skip_pegasus).")
        return None

    merged_path = args.output_dir / "pegasus_rmsf_scores.json"
    if merged_path.is_file():
        print(f"[pegasus] Cached — {merged_path.name} already exists. Skipping.")
        with open(merged_path) as fh:
            return json.load(fh)

    _ensure_pegasus_image()
    _ensure_pegasus_weights(args.pegasus_models_dir)

    peg_raw = _run_pegasus_docker(
        fasta=filtered_fasta,
        models_dir=args.pegasus_models_dir,
        output_dir=args.output_dir,
    )
    rmsf_scores = _parse_pegasus_output(peg_raw)

    with open(merged_path, "w") as fh:
        json.dump(rmsf_scores, fh)
    print(f"[pegasus] Saved RMSF scores → {merged_path}")
    return rmsf_scores


# ===========================================================================
# Stage 4 – compare
# ===========================================================================

def _parse_neq(value) -> list[float]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [float(v) for v in parsed]


def _bootstrap_mean_rho(rho_array: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    """Return (lower_95ci, upper_95ci) of macro-mean Spearman ρ via bootstrap."""
    boot = [
        float(np.mean(rng.choice(rho_array, size=len(rho_array), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_compare(
    args: argparse.Namespace,
    filtered_csv: Path,
    all_esmfluc_scores: dict[str, dict[str, list[float]]],
    pegasus_scores: dict[str, list[float]] | None,
) -> None:
    """Stage 4: compute per-protein Spearman ρ for every ESMfluc condition vs PEGASUS."""

    rng = np.random.default_rng(args.random_seed)

    print("[compare] Loading filtered mdCATH 320K Neq values …")
    df_filt = pd.read_csv(filtered_csv)
    neq_map: dict[str, list[float]] = {}
    for _, row in df_filt.iterrows():
        neq_map[row["domain"]] = _parse_neq(row["neq"])

    # Determine evaluation domain set: intersection of all conditions + PEGASUS + Neq
    # Start with domains valid for all ESMfluc conditions
    valid_per_condition: dict[str, set[str]] = {}
    for cond, scores in all_esmfluc_scores.items():
        valid_per_condition[cond] = {
            d for d in scores
            if d in neq_map and len(scores[d]) == len(neq_map[d])
        }

    # Common across all ESMfluc conditions
    eval_domains_set = set.intersection(*valid_per_condition.values()) if valid_per_condition else set()

    if pegasus_scores is not None:
        peg_valid = {
            d for d in pegasus_scores
            if d in neq_map and len(pegasus_scores[d]) == len(neq_map[d])
        }
        n_before = len(eval_domains_set)
        eval_domains_set &= peg_valid
        n_skip_peg = n_before - len(eval_domains_set)
        if n_skip_peg:
            print(f"[compare] WARNING: {n_skip_peg} proteins dropped (PEGASUS length mismatch/missing).")

    eval_domains = sorted(eval_domains_set)
    print(f"[compare] Evaluating on {len(eval_domains)} proteins, "
          f"{len(all_esmfluc_scores)} ESMfluc conditions "
          f"+ {'PEGASUS' if pegasus_scores else 'no PEGASUS'} …")

    # Per-protein rows
    rho_per_condition: dict[str, list[float]] = {c: [] for c in all_esmfluc_scores}
    rho_peg_list: list[float] = []
    rows = []

    for dom in eval_domains:
        neq = np.array(neq_map[dom])
        row: dict = {"domain": dom, "n_residues": int(len(neq))}

        for cond, scores in all_esmfluc_scores.items():
            sc = np.array(scores[dom])
            rho, _ = spearmanr(sc, neq)
            rho = float(rho) if np.isfinite(rho) else 0.0
            rho_per_condition[cond].append(rho)
            row[f"spearman_rho_{cond}"] = round(rho, 4)

        if pegasus_scores is not None:
            peg_sc = np.array(pegasus_scores[dom])
            rho_peg, _ = spearmanr(peg_sc, neq)
            rho_peg = float(rho_peg) if np.isfinite(rho_peg) else 0.0
            rho_peg_list.append(rho_peg)
            row["spearman_rho_pegasus"] = round(rho_peg, 4)
            for cond in all_esmfluc_scores:
                row[f"delta_{cond}_minus_pegasus"] = round(
                    row[f"spearman_rho_{cond}"] - rho_peg, 4
                )
        rows.append(row)

    per_protein_df = pd.DataFrame(rows)
    per_protein_csv = args.output_dir / "pegasus_mdcath_per_protein.csv"
    per_protein_df.to_csv(per_protein_csv, index=False)
    print(f"[compare] Per-protein results → {per_protein_csv}")

    # Aggregate
    summary: dict = {"n_proteins": len(eval_domains), "conditions": {}}

    for cond, rho_list in rho_per_condition.items():
        arr  = np.array(rho_list)
        mean = float(np.mean(arr))
        ci   = _bootstrap_mean_rho(arr, args.n_bootstrap, rng)
        label = _CONDITION_DISPLAY.get(cond, cond)
        summary["conditions"][cond] = {
            "display": label,
            "mean_spearman_rho": round(mean, 4),
            "ci_95_lower": round(ci[0], 4),
            "ci_95_upper": round(ci[1], 4),
        }
        print(f"[compare] {label:<40s}  ρ={mean:.4f}  CI [{ci[0]:.4f},{ci[1]:.4f}]")

    if pegasus_scores is not None and rho_peg_list:
        peg_arr  = np.array(rho_peg_list)
        mean_peg = float(np.mean(peg_arr))
        ci_peg   = _bootstrap_mean_rho(peg_arr, args.n_bootstrap, rng)
        summary["pegasus"] = {
            "display": "PEGASUS (ATLAS cross-val)",
            "model": "dsimb/pegasus Docker, RMSF output",
            "mean_spearman_rho": round(mean_peg, 4),
            "ci_95_lower": round(ci_peg[0], 4),
            "ci_95_upper": round(ci_peg[1], 4),
        }
        print(f"[compare] {'PEGASUS (ATLAS cross-val)':<40s}  ρ={mean_peg:.4f}  "
              f"CI [{ci_peg[0]:.4f},{ci_peg[1]:.4f}]")

        # Wilcoxon for each condition vs PEGASUS
        comparisons: dict = {}
        for cond, rho_list in rho_per_condition.items():
            delta = np.array(rho_list) - peg_arr
            stat, pval = wilcoxon(delta, alternative="two-sided")
            comparisons[cond] = {
                "mean_delta": round(float(np.mean(delta)), 4),
                "wilcoxon_W": float(stat),
                "wilcoxon_p": float(pval),
            }
            label = _CONDITION_DISPLAY.get(cond, cond)
            print(f"  Δρ {label} − PEGASUS = {np.mean(delta):+.4f}  "
                  f"Wilcoxon p={pval:.4g}")
        summary["wilcoxon_vs_pegasus"] = comparisons

    summary_path = args.output_dir / "pegasus_mdcath_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[compare] Summary → {summary_path}")

    _print_table(summary)


def _print_table(summary: dict) -> None:
    """Print a compact results table to stdout."""
    n = summary["n_proteins"]
    print("\n" + "=" * 68)
    print(f"  mdCATH 320K cross-dataset benchmark  (n = {n} proteins)")
    print("=" * 68)
    fmt = "  {:<42s}  {:>7s}  {:>14s}"
    print(fmt.format("Method", "Spearman ρ", "95 % CI"))
    print("  " + "-" * 64)
    for cond_key in _ALL_CONDITIONS:
        if cond_key not in summary.get("conditions", {}):
            continue
        d = summary["conditions"][cond_key]
        print(fmt.format(
            d["display"],
            f"{d['mean_spearman_rho']:.4f}",
            f"[{d['ci_95_lower']:.4f}, {d['ci_95_upper']:.4f}]",
        ))
    if "pegasus" in summary:
        d = summary["pegasus"]
        print("  " + "-" * 64)
        print(fmt.format(
            d["display"],
            f"{d['mean_spearman_rho']:.4f}",
            f"[{d['ci_95_lower']:.4f}, {d['ci_95_upper']:.4f}]",
        ))
    if "wilcoxon_vs_pegasus" in summary:
        print()
        for cond, c in summary["wilcoxon_vs_pegasus"].items():
            label = _CONDITION_DISPLAY.get(cond, cond)
            print(f"  Δρ {label} − PEGASUS = {c['mean_delta']:+.4f}  p={c['wilcoxon_p']:.4g}")
    print("=" * 68 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = _build_parser().parse_args()
    stages = {s.strip() for s in args.stages.split(",")}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_fasta = args.output_dir / "mdcath_320K_filtered.fasta"
    filtered_csv   = args.output_dir / "mdcath_320K_filtered.csv"

    # Stage 1 – filter
    if "filter" in stages:
        filtered_fasta = run_filter(args)
    else:
        if not filtered_fasta.is_file():
            raise FileNotFoundError(
                f"Filtered FASTA not found at {filtered_fasta}. Run with --stages filter first."
            )
        print(f"[filter] Skipped (not in --stages). Using {filtered_fasta}.")

    # Stage 2 – ESMfluc inference
    all_esmfluc_scores: dict[str, dict[str, list[float]]] | None = None
    if "infer" in stages:
        all_esmfluc_scores = run_infer(args, filtered_fasta)
    else:
        # Try to load each condition from cache
        infer_dir = args.output_dir / "esmfluc_inference"
        cached: dict[str, dict[str, list[float]]] = {}
        for cond in _ALL_CONDITIONS:
            p = infer_dir / f"{cond}_mean_scores.json"
            if p.is_file():
                with open(p) as fh:
                    cached[cond] = json.load(fh)
            else:
                # Legacy: single-condition cache from earlier run
                legacy = infer_dir / "esmfluc_mean_scores.json"
                if legacy.is_file() and not cached:
                    print(f"[infer] Found legacy single-condition cache; "
                          f"mapping to {cond}.")
                    with open(legacy) as fh:
                        cached[cond] = json.load(fh)
                    break
        if cached:
            print(f"[infer] Skipped (not in --stages). Loaded {len(cached)} cached condition(s).")
            all_esmfluc_scores = cached
        else:
            print("[infer] Skipped and no cached scores found; PEGASUS-only mode.")

    # Stage 3 – PEGASUS
    pegasus_scores: dict[str, list[float]] | None = None
    if "pegasus" in stages:
        pegasus_scores = run_pegasus(args, filtered_fasta)
    else:
        cached_peg = args.output_dir / "pegasus_rmsf_scores.json"
        if cached_peg.is_file():
            print(f"[pegasus] Skipped (not in --stages). Loading {cached_peg.name}.")
            with open(cached_peg) as fh:
                pegasus_scores = json.load(fh)
        else:
            print("[pegasus] Skipped and no cached scores found.")

    # Stage 4 – compare
    if "compare" in stages:
        if all_esmfluc_scores is None:
            raise RuntimeError(
                "Cannot run 'compare' without ESMfluc scores. "
                "Include 'infer' in --stages or ensure cached scores exist."
            )
        run_compare(args, filtered_csv, all_esmfluc_scores, pegasus_scores)
    else:
        print("[compare] Skipped (not in --stages).")


if __name__ == "__main__":
    main()
