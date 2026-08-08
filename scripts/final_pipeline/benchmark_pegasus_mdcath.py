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

  Stage 1b – mmseqs (optional, recommended)
      Run MMseqs2 easy-search (ATLAS sequences vs filtered mdCATH) at 30% identity
      / 50% coverage — consistent with the ESMfluc split criterion — and remove
      any mdCATH domain with a hit.  Then cluster the remaining proteins at 30%
      identity to obtain homology groups for the union-group bootstrap.
      Writes mdcath_320K_strict.{csv,fasta} and mdcath_cluster_assignments.tsv.

  Stage 3 – pegasus
      Pull the PEGASUS Docker image (dsimb/pegasus) if not present, download
      the model weights if not present, then run PEGASUS on the filtered FASTA.
      PEGASUS emits four heads per residue: RMSF, STD_PHI, STD_PSI, MEAN_LDDT.
      All four (plus the φ/ψ combined average) are scored against Neq.

  Stage 4 – compare
      For each protein compute Spearman ρ between per-residue predictor scores
      and the continuous mdCATH 320K Neq.  Bootstrap 95 % CI on the macro-mean
      ρ and on Δρ (ESMfluc − best PEGASUS head) are computed by resampling
      homology groups rather than individual proteins (union-group bootstrap).
      If mdcath_cluster_assignments.tsv is present the MMseqs2 clusters are used;
      otherwise the 4-char PDB code is used as a conservative fallback group.

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
from scipy.stats import spearmanr

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

# PEGASUS prediction heads to score against Neq.
# mean_MEAN_LDDT is a confidence score (higher = more rigid), so its Spearman ρ
# vs Neq (flexibility) will be negative — reported as-is.
_PEGASUS_HEADS = ["mean_RMSF", "mean_STD_PHI", "mean_STD_PSI", "mean_MEAN_LDDT"]
_PEGASUS_HEAD_DISPLAY = {
    "mean_RMSF":        "PEGASUS RMSF",
    "mean_STD_PHI":     "PEGASUS φ-std",
    "mean_STD_PSI":     "PEGASUS ψ-std",
    "mean_MEAN_LDDT":   "PEGASUS LDDT (neg. flex.)",
    "phi_psi_combined": "PEGASUS φ/ψ combined",
}

# Path to MMseqs2 binary (ColabFold install takes priority; falls back to PATH).
_MMSEQS_BIN = Path("/home/zahralab/localcolabfold/colabfold-conda/bin/mmseqs")

# Python interpreter that has the EvolutionaryScale ESM3 package installed.
# Used for ESM3 conditions only; ESM2 conditions use the current interpreter.
_ESM3_PYTHON = Path("/home/zahralab/miniconda/envs/esm_env/bin/python")


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
                   help="Comma-separated subset of stages to run. "
                        "Add 'mmseqs' for homology-filtered strict evaluation.")
    p.add_argument("--use_strict", action="store_true",
                   help="Use the MMseqs2-strict filtered set in the compare stage. "
                        "Requires the 'mmseqs' stage to have been run.")
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

    # Use the ESM3-capable interpreter for ESM3 conditions; current interpreter otherwise.
    python = str(_ESM3_PYTHON) if is_esm3 else sys.executable

    cmd = [
        python, str(get_attn),
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

    print(f"  [infer] Running: checkpoint={checkpoint.parent.name} (esm3={is_esm3}, python={Path(python).name})")
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


def _parse_pegasus_all_heads(peg_raw_dir: Path) -> dict[str, dict[str, list[float]]]:
    """
    Parse all PEGASUS prediction heads from the raw output directory.

    PEGASUS TSV columns (predictions/{GID}_predictions.tsv):
      res  mean_RMSF  std_RMSF  mean_STD_PHI  std_STD_PHI
           mean_STD_PSI  std_STD_PSI  mean_MEAN_LDDT  std_MEAN_LDDT

    Returns {head_name: {original_domain: [scores per residue]}}.
    Heads returned: mean_RMSF, mean_STD_PHI, mean_STD_PSI, mean_MEAN_LDDT,
                    phi_psi_combined (average of STD_PHI and STD_PSI means).
    """
    subdirs = [d for d in peg_raw_dir.iterdir() if d.is_dir() and (d / "id_mapping.tsv").is_file()]
    if not subdirs:
        raise FileNotFoundError(
            f"No PEGASUS job directory with id_mapping.tsv found in {peg_raw_dir}."
        )
    job_dir = subdirs[0]
    print(f"[pegasus] Parsing all prediction heads from {job_dir.name} …")

    id_map = pd.read_csv(job_dir / "id_mapping.tsv", sep="\t",
                         names=["Generated_ID", "Original_ID"], header=0).dropna()

    # Initialise output dict: one entry per head + combined
    all_heads: dict[str, dict[str, list[float]]] = {h: {} for h in _PEGASUS_HEADS}
    all_heads["phi_psi_combined"] = {}

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
        df_pred.columns = [c.strip() for c in df_pred.columns]
        # Build lower-case → actual column name map for robust matching
        col_map = {c.lower(): c for c in df_pred.columns}

        for head in _PEGASUS_HEADS:
            actual = col_map.get(head.lower())
            if actual is not None:
                all_heads[head][orig_id] = [float(v) for v in df_pred[actual]]

        # φ/ψ combined: average of the two dihedral-std heads
        phi_col = col_map.get("mean_std_phi")
        psi_col = col_map.get("mean_std_psi")
        if phi_col is not None and psi_col is not None:
            phi = np.array(df_pred[phi_col], dtype=float)
            psi = np.array(df_pred[psi_col], dtype=float)
            all_heads["phi_psi_combined"][orig_id] = ((phi + psi) / 2.0).tolist()

    if missing:
        print(f"[pegasus] WARNING: missing predictions for {len(missing)} proteins.")
    n = len(all_heads["mean_RMSF"])
    print(f"[pegasus] Parsed {n} proteins across "
          f"{len(all_heads)} heads ({', '.join(all_heads)}).")
    return all_heads


def run_pegasus(
    args: argparse.Namespace,
    filtered_fasta: Path,
) -> dict[str, dict[str, list[float]]] | None:
    """
    Stage 3: run PEGASUS and return all prediction heads.

    Returns {head_name: {domain: [scores per residue]}} or None if
    --skip_pegasus is set.
    """
    if args.skip_pegasus:
        print("[pegasus] Skipped (--skip_pegasus).")
        return None

    all_heads_path = args.output_dir / "pegasus_all_heads.json"
    if all_heads_path.is_file():
        print(f"[pegasus] Cached — {all_heads_path.name} already exists. Loading.")
        with open(all_heads_path) as fh:
            return json.load(fh)

    # Legacy cache: only RMSF was saved.  If raw output exists, re-parse all heads.
    peg_raw_dir = args.output_dir / "pegasus_raw_output"
    peg_done    = peg_raw_dir / ".pegasus_done"
    if peg_done.is_file():
        print("[pegasus] Raw output cached; re-parsing all prediction heads …")
        all_heads = _parse_pegasus_all_heads(peg_raw_dir)
        with open(all_heads_path, "w") as fh:
            json.dump(all_heads, fh)
        print(f"[pegasus] Saved all-heads cache → {all_heads_path}")
        return all_heads

    _ensure_pegasus_image()
    _ensure_pegasus_weights(args.pegasus_models_dir)

    peg_raw = _run_pegasus_docker(
        fasta=filtered_fasta,
        models_dir=args.pegasus_models_dir,
        output_dir=args.output_dir,
    )
    all_heads = _parse_pegasus_all_heads(peg_raw)

    with open(all_heads_path, "w") as fh:
        json.dump(all_heads, fh)
    print(f"[pegasus] Saved all-heads cache → {all_heads_path}")
    return all_heads


# ===========================================================================
# Stage 4 – compare
# ===========================================================================

def _parse_neq(value) -> list[float]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [float(v) for v in parsed]


# ---------------------------------------------------------------------------
# Bootstrap helpers (group-level resampling)
# ---------------------------------------------------------------------------

def _build_group_index(
    group_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return (unique_groups, group→[protein_indices]) for bootstrap resampling."""
    groups = np.unique(group_ids)
    g2idx: dict = {g: [] for g in groups}
    for i, g in enumerate(group_ids):
        g2idx[g].append(i)
    return groups, g2idx


def _bootstrap_mean_rho_groups(
    rho_array: np.ndarray,
    group_ids: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI of macro-mean ρ via group-level bootstrap (resample groups, not proteins)."""
    groups, g2idx = _build_group_index(group_ids)
    boot = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idxs = [i for g in sampled for i in g2idx[g]]
        boot.append(float(np.mean(rho_array[idxs])))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bootstrap_ci_delta_groups(
    rho_esm: np.ndarray,
    rho_peg: np.ndarray,
    group_ids: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI on mean(Δρ = ρ_esm − ρ_peg) via group-level bootstrap."""
    groups, g2idx = _build_group_index(group_ids)
    delta = rho_esm - rho_peg
    boot = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idxs = [i for g in sampled for i in g2idx[g]]
        boot.append(float(np.mean(delta[idxs])))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Stage 1b – MMseqs2 homology filter + clustering
# ---------------------------------------------------------------------------

def _find_mmseqs() -> Path:
    """Locate the MMseqs2 binary (ColabFold install or PATH)."""
    if _MMSEQS_BIN.is_file():
        return _MMSEQS_BIN
    result = subprocess.run(["which", "mmseqs"], capture_output=True, text=True)
    if result.returncode == 0:
        return Path(result.stdout.strip())
    raise FileNotFoundError(
        "MMseqs2 not found.  Install with: conda install -c bioconda mmseqs2  "
        f"or set _MMSEQS_BIN in the script (currently {_MMSEQS_BIN})."
    )


def _load_cluster_map(cluster_tsv: Path) -> dict[str, str]:
    df = pd.read_csv(cluster_tsv, sep="\t")
    return dict(zip(df["member"].astype(str), df["rep"].astype(str)))


def run_filter_mmseqs(
    args: argparse.Namespace,
    filtered_csv: Path,
    filtered_fasta: Path,
) -> tuple[Path, Path, dict[str, str]]:
    """
    Stage 1b: MMseqs2-based contamination removal + homology clustering.

    1. easy-search ATLAS (all 1 369 sequences) vs filtered mdCATH FASTA at
       ≥ 30% sequence identity / ≥ 50% coverage (shorter-seq mode) — the same
       threshold used to build the ESMfluc training split.  Any mdCATH domain
       that matches an ATLAS protein is removed.
    2. easy-cluster the remaining proteins at 30% identity to produce homology
       groups for the union-group bootstrap.

    Returns (strict_csv, strict_fasta, cluster_map {domain: cluster_rep}).
    Outputs are cached; re-runs skip completed steps.
    """
    strict_csv   = args.output_dir / "mdcath_320K_strict.csv"
    strict_fasta = args.output_dir / "mdcath_320K_strict.fasta"
    cluster_tsv  = args.output_dir / "mdcath_cluster_assignments.tsv"

    if strict_csv.is_file() and strict_fasta.is_file() and cluster_tsv.is_file():
        print("[mmseqs] Cached — strict files already exist. Loading cluster map.")
        return strict_csv, strict_fasta, _load_cluster_map(cluster_tsv)

    mmseqs  = _find_mmseqs()
    tmp_dir = args.output_dir / "mmseqs_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    atlas_fasta = args.atlas_splits_dir / "mmseqs" / "atlas.fasta"
    if not atlas_fasta.is_file():
        raise FileNotFoundError(
            f"ATLAS FASTA not found at {atlas_fasta}.  "
            "Expected in data_splits/atlas_grouped_v1/mmseqs/atlas.fasta."
        )

    # Step 1 – search
    search_tsv = tmp_dir / "atlas_vs_mdcath.tsv"
    if not search_tsv.is_file():
        print("[mmseqs] Running easy-search: ATLAS vs filtered mdCATH 320K …")
        subprocess.run([
            str(mmseqs), "easy-search",
            str(atlas_fasta), str(filtered_fasta), str(search_tsv),
            str(tmp_dir / "search_tmp"),
            "--min-seq-id", "0.30",
            "-c", "0.50",
            "--cov-mode", "2",   # coverage of shorter sequence
            "--format-output", "query,target,pident,qcov,tcov",
            "-v", "1",
        ], check=True)
    else:
        print("[mmseqs] easy-search cached.")

    if search_tsv.stat().st_size > 0:
        hits = pd.read_csv(search_tsv, sep="\t", header=None,
                           names=["query", "target", "pident", "qcov", "tcov"])
        homologs = set(hits["target"].str.strip())
    else:
        homologs: set[str] = set()

    # Step 2 – filter
    df_filt = pd.read_csv(filtered_csv)
    mask    = ~df_filt["domain"].isin(homologs)
    n_rm    = (~mask).sum()
    df_strict = df_filt[mask].copy().reset_index(drop=True)
    print(
        f"[mmseqs] MMseqs2 removed {n_rm} homologs (≥30% id, ≥50% cov) "
        f"on top of the PDB-code filter; {len(df_strict)} proteins remain."
    )
    df_strict.to_csv(strict_csv, index=False)
    with open(strict_fasta, "w") as fh:
        for _, row in df_strict.iterrows():
            fh.write(f">{row['domain']}\n{row['sequence']}\n")
    print(f"[mmseqs] Wrote strict filtered set → {strict_fasta}")

    # Step 3 – cluster remaining proteins
    print(f"[mmseqs] Clustering {len(df_strict)} proteins at 30% identity …")
    cluster_prefix = str(tmp_dir / "mdcath_clusters")
    subprocess.run([
        str(mmseqs), "easy-cluster",
        str(strict_fasta), cluster_prefix,
        str(tmp_dir / "cluster_tmp"),
        "--min-seq-id", "0.30",
        "-c", "0.50",
        "--cov-mode", "0",
        "-v", "1",
    ], check=True)

    raw_cluster = pd.read_csv(
        cluster_prefix + "_cluster.tsv", sep="\t", header=None,
        names=["rep", "member"]
    )
    cluster_map = dict(zip(raw_cluster["member"].astype(str),
                           raw_cluster["rep"].astype(str)))
    n_clusters = len(set(cluster_map.values()))
    print(f"[mmseqs] {len(cluster_map)} proteins → {n_clusters} clusters.")

    raw_cluster.to_csv(cluster_tsv, sep="\t", index=False)
    print(f"[mmseqs] Wrote cluster assignments → {cluster_tsv}")
    return strict_csv, strict_fasta, cluster_map


# ===========================================================================
# Stage 4 – compare
# ===========================================================================

def run_compare(
    args: argparse.Namespace,
    filtered_csv: Path,
    all_esmfluc_scores: dict[str, dict[str, list[float]]],
    pegasus_all_heads: dict[str, dict[str, list[float]]] | None,
    cluster_map: dict[str, str] | None = None,
) -> None:
    """Stage 4: compute per-protein Spearman ρ for every ESMfluc condition vs all PEGASUS heads."""

    rng = np.random.default_rng(args.random_seed)

    print("[compare] Loading mdCATH Neq values …")
    df_filt = pd.read_csv(filtered_csv)
    neq_map: dict[str, list[float]] = {}
    for _, row in df_filt.iterrows():
        neq_map[row["domain"]] = _parse_neq(row["neq"])

    # Determine group assignment for bootstrap:
    # prefer MMseqs2 clusters; fall back to 4-char PDB code.
    if cluster_map:
        n_clusters = len(set(cluster_map.values()))
        print(f"[compare] Using MMseqs2 cluster groups ({n_clusters} clusters).")
        def get_group(d: str) -> str:
            return cluster_map.get(d, d[:4].lower())  # fallback to PDB if domain not in map
    else:
        print("[compare] No cluster file; using 4-char PDB code as group (conservative fallback).")
        def get_group(d: str) -> str:
            return d[:4].lower()

    # Evaluate on the intersection of all available domains
    valid_esm: dict[str, set[str]] = {}
    for cond, scores in all_esmfluc_scores.items():
        valid_esm[cond] = {
            d for d in scores if d in neq_map and len(scores[d]) == len(neq_map[d])
        }
    eval_domains_set = set.intersection(*valid_esm.values()) if valid_esm else set()

    if pegasus_all_heads is not None:
        # Intersect over all heads so every domain has a complete set of scores
        for head, head_scores in pegasus_all_heads.items():
            peg_valid = {
                d for d in head_scores
                if d in neq_map and len(head_scores[d]) == len(neq_map[d])
            }
            n_before = len(eval_domains_set)
            eval_domains_set &= peg_valid
            n_drop = n_before - len(eval_domains_set)
            if n_drop:
                print(f"[compare] WARNING: {n_drop} domains dropped (PEGASUS head '{head}' "
                      "length mismatch/missing).")

    skipped = len(all_esmfluc_scores[next(iter(all_esmfluc_scores))]) - len(eval_domains_set)
    if skipped:
        print(f"[compare] WARNING: {skipped} ESMfluc proteins skipped (length mismatch or missing Neq).")

    eval_domains = sorted(eval_domains_set)
    n_groups = len({get_group(d) for d in eval_domains})
    print(f"[compare] Evaluating on {len(eval_domains)} proteins "
          f"({n_groups} bootstrap groups) …")

    # Per-protein ρ
    rho_per_esm: dict[str, list[float]]       = {c: [] for c in all_esmfluc_scores}
    rho_per_peg: dict[str, list[float]]       = {h: [] for h in (pegasus_all_heads or {})}
    group_ids:   list[str]                    = []
    rows = []

    for dom in eval_domains:
        neq = np.array(neq_map[dom])
        row: dict = {"domain": dom, "n_residues": int(len(neq)),
                     "group": get_group(dom)}
        group_ids.append(get_group(dom))

        for cond, scores in all_esmfluc_scores.items():
            rho, _ = spearmanr(np.array(scores[dom]), neq)
            rho = float(rho) if np.isfinite(rho) else 0.0
            rho_per_esm[cond].append(rho)
            row[f"rho_{cond}"] = round(rho, 4)

        if pegasus_all_heads is not None:
            for head, head_scores in pegasus_all_heads.items():
                rho_p, _ = spearmanr(np.array(head_scores[dom]), neq)
                rho_p = float(rho_p) if np.isfinite(rho_p) else 0.0
                rho_per_peg[head].append(rho_p)
                row[f"rho_pegasus_{head}"] = round(rho_p, 4)
        rows.append(row)

    per_protein_df = pd.DataFrame(rows)
    per_protein_csv = args.output_dir / "pegasus_mdcath_per_protein.csv"
    per_protein_df.to_csv(per_protein_csv, index=False)
    print(f"[compare] Per-protein results → {per_protein_csv}")

    grp = np.array(group_ids)

    # Identify best PEGASUS head (highest mean ρ vs Neq)
    best_head: str | None = None
    if rho_per_peg:
        best_head = max(rho_per_peg, key=lambda h: float(np.mean(rho_per_peg[h])))

    summary: dict = {
        "n_proteins":   len(eval_domains),
        "n_groups":     n_groups,
        "group_source": "mmseqs2_cluster" if cluster_map else "pdb_code_fallback",
        "conditions":   {},
        "pegasus_heads": {},
    }

    # ESMfluc results
    for cond, rho_list in rho_per_esm.items():
        arr  = np.array(rho_list)
        mean = float(np.mean(arr))
        ci   = _bootstrap_mean_rho_groups(arr, grp, args.n_bootstrap, rng)
        label = _CONDITION_DISPLAY.get(cond, cond)
        summary["conditions"][cond] = {
            "display":  label,
            "mean_rho": round(mean, 4),
            "ci_95":    [round(ci[0], 4), round(ci[1], 4)],
        }
        print(f"[compare] {label:<42s}  ρ={mean:.4f}  CI [{ci[0]:.4f},{ci[1]:.4f}]")

    # PEGASUS head results
    for head, rho_list in rho_per_peg.items():
        arr   = np.array(rho_list)
        mean  = float(np.mean(arr))
        ci    = _bootstrap_mean_rho_groups(arr, grp, args.n_bootstrap, rng)
        label = _PEGASUS_HEAD_DISPLAY.get(head, f"PEGASUS {head}")
        is_best = (head == best_head)
        summary["pegasus_heads"][head] = {
            "display":      label,
            "mean_rho":     round(mean, 4),
            "ci_95":        [round(ci[0], 4), round(ci[1], 4)],
            "is_best_head": is_best,
        }
        marker = "  ← best head" if is_best else ""
        print(f"[compare] {label:<42s}  ρ={mean:.4f}  CI [{ci[0]:.4f},{ci[1]:.4f}]{marker}")

    # Δρ CI: each ESMfluc condition vs best PEGASUS head
    if best_head is not None:
        peg_arr = np.array(rho_per_peg[best_head])
        peg_label = _PEGASUS_HEAD_DISPLAY.get(best_head, best_head)
        delta_summary: dict = {}
        for cond, rho_list in rho_per_esm.items():
            esm_arr = np.array(rho_list)
            delta_mean = float(np.mean(esm_arr - peg_arr))
            ci_d = _bootstrap_ci_delta_groups(esm_arr, peg_arr, grp, args.n_bootstrap, rng)
            delta_summary[cond] = {
                "mean_delta": round(delta_mean, 4),
                "ci_95_delta": [round(ci_d[0], 4), round(ci_d[1], 4)],
            }
            label = _CONDITION_DISPLAY.get(cond, cond)
            print(f"  Δρ {label} − {peg_label} = {delta_mean:+.4f}  "
                  f"95% CI [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]")
        summary["delta_vs_best_pegasus_head"] = {
            "pegasus_head": best_head,
            "conditions":   delta_summary,
        }

    summary_path = args.output_dir / "pegasus_mdcath_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[compare] Summary → {summary_path}")

    _print_table(summary)


def _print_table(summary: dict) -> None:
    """Print a compact results table to stdout."""
    n  = summary["n_proteins"]
    ng = summary.get("n_groups", "?")
    gs = summary.get("group_source", "unknown")
    print("\n" + "=" * 72)
    print(f"  mdCATH 320K cross-dataset benchmark  "
          f"(n={n} proteins, {ng} bootstrap groups [{gs}])")
    print("=" * 72)
    fmt = "  {:<42s}  {:>7s}  {:>16s}"
    print(fmt.format("Method", "Spearman ρ", "95 % CI"))
    print("  " + "-" * 68)

    # ESMfluc conditions
    for cond_key in _ALL_CONDITIONS:
        d = summary.get("conditions", {}).get(cond_key)
        if d is None:
            continue
        ci = d["ci_95"]
        print(fmt.format(d["display"], f"{d['mean_rho']:.4f}",
                         f"[{ci[0]:.4f}, {ci[1]:.4f}]"))

    # PEGASUS heads
    heads = summary.get("pegasus_heads", {})
    if heads:
        print("  " + "-" * 68)
        head_order = ["mean_RMSF", "mean_STD_PHI", "mean_STD_PSI",
                      "mean_MEAN_LDDT", "phi_psi_combined"]
        for hk in head_order:
            d = heads.get(hk)
            if d is None:
                continue
            ci     = d["ci_95"]
            marker = "  *" if d.get("is_best_head") else ""
            print(fmt.format(d["display"] + marker,
                             f"{d['mean_rho']:.4f}",
                             f"[{ci[0]:.4f}, {ci[1]:.4f}]"))
        print("  (* = best PEGASUS head, used for Δρ comparison)")

    # Δρ CI vs best head
    ds = summary.get("delta_vs_best_pegasus_head", {})
    if ds:
        best_hk  = ds["pegasus_head"]
        peg_lbl  = _PEGASUS_HEAD_DISPLAY.get(best_hk, best_hk)
        print()
        for cond, c in ds.get("conditions", {}).items():
            label = _CONDITION_DISPLAY.get(cond, cond)
            ci_d  = c["ci_95_delta"]
            print(f"  Δρ {label} − {peg_lbl} = {c['mean_delta']:+.4f}  "
                  f"95% CI [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]")
    print("=" * 72 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = _build_parser().parse_args()
    stages = {s.strip() for s in args.stages.split(",")}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_fasta = args.output_dir / "mdcath_320K_filtered.fasta"
    filtered_csv   = args.output_dir / "mdcath_320K_filtered.csv"

    # Stage 1 – PDB-code filter
    if "filter" in stages:
        filtered_fasta = run_filter(args)
    else:
        if not filtered_fasta.is_file():
            raise FileNotFoundError(
                f"Filtered FASTA not found at {filtered_fasta}. Run with --stages filter first."
            )
        print(f"[filter] Skipped (not in --stages). Using {filtered_fasta}.")

    # Stage 1b – MMseqs2 homology filter + clustering (optional)
    cluster_map: dict[str, str] | None = None
    compare_csv   = filtered_csv
    if "mmseqs" in stages:
        strict_csv, strict_fasta, cluster_map = run_filter_mmseqs(
            args, filtered_csv, filtered_fasta
        )
        if getattr(args, "use_strict", False):
            compare_csv   = strict_csv
            filtered_fasta = strict_fasta
            print(f"[mmseqs] --use_strict: compare stage will use strict-filtered set.")
    else:
        # Try to load cached cluster assignments from a previous mmseqs run
        cluster_tsv = args.output_dir / "mdcath_cluster_assignments.tsv"
        if cluster_tsv.is_file():
            cluster_map = _load_cluster_map(cluster_tsv)
            print(f"[mmseqs] Loaded cached cluster assignments ({len(set(cluster_map.values()))} clusters).")
        if getattr(args, "use_strict", False):
            strict_csv   = args.output_dir / "mdcath_320K_strict.csv"
            strict_fasta = args.output_dir / "mdcath_320K_strict.fasta"
            if strict_csv.is_file() and strict_fasta.is_file():
                compare_csv    = strict_csv
                filtered_fasta = strict_fasta
                print(f"[mmseqs] --use_strict: using existing strict-filtered set.")
            else:
                print("[mmseqs] WARNING: --use_strict requested but strict files not found. "
                      "Run --stages mmseqs first.")

    # Stage 2 – ESMfluc inference
    all_esmfluc_scores: dict[str, dict[str, list[float]]] | None = None
    if "infer" in stages:
        all_esmfluc_scores = run_infer(args, filtered_fasta)
    else:
        infer_dir = args.output_dir / "esmfluc_inference"
        cached: dict[str, dict[str, list[float]]] = {}
        for cond in _ALL_CONDITIONS:
            p = infer_dir / f"{cond}_mean_scores.json"
            if p.is_file():
                with open(p) as fh:
                    cached[cond] = json.load(fh)
            else:
                legacy = infer_dir / "esmfluc_mean_scores.json"
                if legacy.is_file() and not cached:
                    print(f"[infer] Found legacy single-condition cache; mapping to {cond}.")
                    with open(legacy) as fh:
                        cached[cond] = json.load(fh)
                    break
        if cached:
            print(f"[infer] Skipped (not in --stages). Loaded {len(cached)} cached condition(s).")
            all_esmfluc_scores = cached
        else:
            print("[infer] Skipped and no cached scores found; PEGASUS-only mode.")

    # Stage 3 – PEGASUS
    pegasus_all_heads: dict[str, dict[str, list[float]]] | None = None
    if "pegasus" in stages:
        pegasus_all_heads = run_pegasus(args, filtered_fasta)
    else:
        all_heads_path = args.output_dir / "pegasus_all_heads.json"
        if all_heads_path.is_file():
            print(f"[pegasus] Skipped (not in --stages). Loading {all_heads_path.name}.")
            with open(all_heads_path) as fh:
                pegasus_all_heads = json.load(fh)
        else:
            # Legacy: try to re-parse all heads from cached raw output
            peg_done = args.output_dir / "pegasus_raw_output" / ".pegasus_done"
            if peg_done.is_file():
                print("[pegasus] Skipped; re-parsing all heads from cached raw output …")
                pegasus_all_heads = _parse_pegasus_all_heads(
                    args.output_dir / "pegasus_raw_output"
                )
                with open(all_heads_path, "w") as fh:
                    json.dump(pegasus_all_heads, fh)
            else:
                print("[pegasus] Skipped and no cached output found.")

    # Stage 4 – compare
    if "compare" in stages:
        if all_esmfluc_scores is None:
            raise RuntimeError(
                "Cannot run 'compare' without ESMfluc scores. "
                "Include 'infer' in --stages or ensure cached scores exist."
            )
        run_compare(args, compare_csv, all_esmfluc_scores, pegasus_all_heads, cluster_map)
    else:
        print("[compare] Skipped (not in --stages).")


if __name__ == "__main__":
    main()
