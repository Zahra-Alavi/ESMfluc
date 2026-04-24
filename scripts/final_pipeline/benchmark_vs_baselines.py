#!/usr/bin/env python3
"""
benchmark_vs_baselines.py

Compare ESMfluc models against DynaMine (via b2btools) on the 277-protein
test set.  Produces a publication-ready comparison table.

Metrics
-------
  AUC-ROC   – threshold-independent, primary metric
  AUC-PR    – better for imbalanced labels (Neq>1.0 is minority)
  F1        – at Youden-optimal threshold per method
  MCC       – Matthews Correlation Coefficient at same threshold
  Spearman  – rank correlation with continuous Neq (where applicable)

Usage (remote machine, conda env esm_env)
-----------------------------------------
  python benchmark_vs_baselines.py \\
      --results_root /home/zahralab/Desktop/ESMfluc/scripts/final_pipeline/results \\
      --neq_csv      /home/zahralab/Desktop/ESMfluc/data/test_data_with_names.csv \\
      --nsp3_csv     /home/zahralab/Desktop/ESMfluc/data/test_data_nsp3.csv \\
      --fasta        /home/zahralab/Desktop/ESMfluc/data/test_data_sequences.fasta \\
      --output_dir   /home/zahralab/Desktop/ESMfluc/scripts/final_pipeline/results/benchmark \\
      --neq_thresh   1.0 \\
      --device       cuda

Install missing deps first (once):
  pip install b2btools logomaker
"""

import argparse
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

# ── networkx ≥ 3.3 required: earlier versions generate invalid Python
#    identifiers from config keys like "nx-loopback" (hyphen), breaking
#    Python 3.11 dataclasses at import time.
def _ensure_networkx():
    """
    networkx < 3.3 uses config keys with hyphens (e.g. 'nx-loopback') as
    Python identifiers in generated dataclasses, causing a SyntaxError on
    Python 3.11.  In conda environments pip install is shadowed by the conda
    package, so we must upgrade via conda.  After upgrading we re-exec so the
    new version is loaded from a clean process.
    """
    import os, shutil, importlib.metadata as _im

    needs_upgrade = False
    try:
        ver_str = _im.version("networkx")
        major, minor = [int(x) for x in ver_str.split(".")[:2]]
        if (major, minor) < (3, 3):
            needs_upgrade = True
    except Exception:
        needs_upgrade = True

    if not needs_upgrade or os.environ.get("_NX_UPGRADED") == "1":
        return

    print("[INFO] networkx < 3.3 detected — upgrading and restarting …")
    conda = shutil.which("conda")
    upgraded = False
    if conda:
        try:
            subprocess.check_call(
                [conda, "install", "-c", "conda-forge", "networkx>=3.3", "-y", "-q"],
                timeout=300,
            )
            upgraded = True
        except Exception as e:
            print(f"  [WARN] conda upgrade failed: {e}")
    if not upgraded:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "networkx>=3.3", "-q"]
        )

    env = os.environ.copy()
    env["_NX_UPGRADED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

_ensure_networkx()

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

# ── optional imports resolved at runtime ──────────────────────────────────────
try:
    from b2btools import SingleSeqAnalysis
    B2B_AVAILABLE = True
except ImportError:
    B2B_AVAILABLE = False

try:
    from transformers import EsmModel, EsmTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from esm.pretrained import ESM3_sm_open_v0
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    ESM3_AVAILABLE = True
except Exception:
    ESM3_AVAILABLE = False


# ── local imports ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from models import BiLSTMWithSelfAttentionModel, ESM3Wrapper  # noqa: E402


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark ESMfluc vs DynaMine on the test set."
    )
    ap.add_argument("--results_root", required=True,
                    help="Root dir containing per-experiment sub-dirs.")
    ap.add_argument("--neq_csv", required=True,
                    help="CSV with columns: name,sequence,neq.")
    ap.add_argument("--nsp3_csv", required=True,
                    help="NetSurfP long CSV with disorder column.")
    ap.add_argument("--fasta", required=True,
                    help="FASTA of test sequences (used for b2btools).")
    ap.add_argument("--output_dir", default="./benchmark_results")
    ap.add_argument("--neq_thresh", type=float, default=1.0,
                    help="Neq > thresh → flexible (label=1). Default=1.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_b2b", action="store_true",
                    help="Skip b2btools/DynaMine (use if already cached).")
    ap.add_argument("--b2b_cache", default=None,
                    help="Path to cached b2btools CSV (long format). "
                         "If given, skip running b2btools.")
    ap.add_argument("--force_api", action="store_true",
                    help="Skip local b2btools install and use DynaMine web API directly.")
    return ap.parse_args()


# =============================================================================
# Data helpers
# =============================================================================
def parse_fasta(path):
    records = {}
    with open(path) as fh:
        name, buf = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    records[name] = "".join(buf)
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line)
        if name:
            records[name] = "".join(buf)
    return records


def load_neq_csv(path, thresh):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        name = str(row["name"])
        seq  = str(row["sequence"])
        raw  = row["neq"]
        vals = ast.literal_eval(raw) if isinstance(raw, str) else [float(raw)]
        vals = [float(v) for v in vals]
        L = min(len(vals), len(seq))
        for i in range(L):
            rows.append({
                "name": name,
                "res_idx": i + 1,
                "aa": seq[i],
                "Neq": vals[i],
                "label": int(vals[i] > thresh),
            })
    return pd.DataFrame(rows)


def load_nsp3(path, disorder_col="disorder"):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        name = str(r["id"]).lstrip(">")
        idx_col = "n" if "n" in df.columns else " n"
        rows.append({
            "name": name,
            "res_idx": int(r[idx_col]),
            "nsp3_disorder": float(r[disorder_col.strip()]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# b2btools / DynaMine predictions
# =============================================================================
def _try_install_b2btools():
    """
    b2btools depends on pomegranate which needs Cython.
    Strategy: conda install pomegranate first (has pre-built wheels), then pip.
    Falls back to pip-only if conda is not available.
    """
    import shutil
    conda = shutil.which("conda")
    if conda:
        print("[INFO] Installing pomegranate via conda-forge …")
        try:
            subprocess.check_call(
                [conda, "install", "-c", "conda-forge", "pomegranate", "-y", "-q"],
                timeout=300,
            )
        except Exception as e:
            print(f"  [WARN] conda install pomegranate failed: {e}")
    print("[INFO] Installing b2btools via pip …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "b2btools", "-q"])
    global B2B_AVAILABLE, SingleSeqAnalysis
    from b2btools import SingleSeqAnalysis as _SSA  # noqa: F401
    SingleSeqAnalysis = _SSA
    B2B_AVAILABLE = True


def _run_b2btools_local(fasta_records):
    """Run DynaMine + DisoMine via the installed b2btools package."""
    rows = []
    total = len(fasta_records)
    for i, (name, seq) in enumerate(fasta_records.items(), 1):
        if i % 50 == 0 or i == 1:
            print(f"  [b2btools] {i}/{total}")
        try:
            analyzer = SingleSeqAnalysis()
            analyzer.load_predictors(["dynamine", "disomine"])
            preds = analyzer.query(seq)
            bb   = preds.get("backbone", [None] * len(seq))
            diso = preds.get("disoMine", [None] * len(seq))
            for j in range(len(seq)):
                rows.append({
                    "name": name,
                    "res_idx": j + 1,
                    "dynamine_bb": float(bb[j]) if bb[j] is not None else np.nan,
                    "disomine":    float(diso[j]) if diso[j] is not None else np.nan,
                })
        except Exception as e:
            print(f"  [WARN] b2btools failed for {name}: {e}")
            for j in range(len(seq)):
                rows.append({"name": name, "res_idx": j + 1,
                             "dynamine_bb": np.nan, "disomine": np.nan})
    return rows


def _run_dynamine_api(fasta_records):
    """
    Fall back: call the DynaMine REST API at https://dynamine.ibsquare.be
    POST a FASTA payload, parse the returned CSV.
    Rate limit: submit one sequence at a time with a short sleep.
    """
    import time
    import io
    try:
        import requests
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests

    API_URL = "https://dynamine.ibsquare.be/api/predictions"
    # Suppress SSL warnings — the DynaMine server has a hostname-mismatch cert
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    rows = []
    total = len(fasta_records)
    for i, (name, seq) in enumerate(fasta_records.items(), 1):
        if i % 25 == 0 or i == 1:
            print(f"  [DynaMine API] {i}/{total}")
        fasta_str = f">{name}\n{seq}\n"
        try:
            resp = requests.post(
                API_URL,
                data={"fasta": fasta_str},
                timeout=60,
                verify=False,  # server cert has hostname mismatch (server-side issue)
            )
            resp.raise_for_status()
            # DynaMine returns whitespace/tab-delimited data, possibly with
            # comment lines starting with '#' at the top.
            text = resp.text
            lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
            clean_text = "\n".join(lines)
            df_api = pd.read_csv(
                io.StringIO(clean_text),
                sep=None,
                engine="python",
                header=0,
            )
            df_api.columns = [c.strip().lower() for c in df_api.columns]
            for j, row in enumerate(df_api.itertuples(), 1):
                bb_val = getattr(row, "backbone", np.nan)
                rows.append({
                    "name": name,
                    "res_idx": j,
                    "dynamine_bb": float(bb_val) if bb_val is not None else np.nan,
                    "disomine": np.nan,  # not available from web API alone
                })
        except Exception as e:
            print(f"  [WARN] DynaMine API failed for {name}: {e}")
            for j in range(len(seq)):
                rows.append({"name": name, "res_idx": j + 1,
                             "dynamine_bb": np.nan, "disomine": np.nan})
        time.sleep(0.3)  # be polite to the server
    return rows


def run_b2btools(fasta_records, cache_path=None, force_api=False):
    """
    Return long DataFrame with columns: name, res_idx, dynamine_bb, disomine.
    Priority:
      1. Load from cache if it exists.
      2. Use installed b2btools package (try to install if missing).
      3. Fall back to DynaMine web API if b2btools installation fails.
    Set force_api=True to skip local install and go straight to the web API.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[INFO] Loading b2btools cache from {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    used_api = False

    if not force_api:
        if not B2B_AVAILABLE:
            try:
                _try_install_b2btools()
            except Exception as e:
                print(f"  [WARN] b2btools install failed ({e}), falling back to DynaMine web API …")
                force_api = True

    if force_api or not B2B_AVAILABLE:
        print("[INFO] Using DynaMine web API …")
        rows = _run_dynamine_api(fasta_records)
        used_api = True
    else:
        rows = _run_b2btools_local(fasta_records)

    df = pd.DataFrame(rows)
    if used_api:
        print("  [NOTE] DisoMine scores unavailable via web API (b2btools only).")
    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"[INFO] Cached predictions → {cache_path}")
    return df


# =============================================================================
# ESMfluc model inference
# =============================================================================
EXPERIMENTS = [
    # (exp_name,       is_esm3, frozen)
    ("esm2_binary_frozen",    False, True),
    ("esm2_binary_unfrozen",  False, False),
    ("esm3_binary_frozen",    True,  True),
    ("esm3_binary_unfrozen",  True,  False),
]


def build_esm2_model(device):
    assert HF_AVAILABLE, "transformers not installed"
    backbone = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = BiLSTMWithSelfAttentionModel(
        embedding_model=backbone,
        hidden_size=512,
        num_layers=3,
        num_classes=2,
        dropout=0.3,
        bidirectional=1,
    ).to(device)
    return model, tokenizer


def build_esm3_model(device):
    assert ESM3_AVAILABLE, "esm package not installed"
    raw = ESM3_sm_open_v0("cpu")
    wrapper = ESM3Wrapper(raw).to(device)
    tokenizer = EsmSequenceTokenizer()
    model = BiLSTMWithSelfAttentionModel(
        embedding_model=wrapper,
        hidden_size=512,
        num_layers=3,
        num_classes=2,
        dropout=0.3,
        bidirectional=1,
    ).to(device)
    return model, tokenizer


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = (ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_sequence_esm2(model, tokenizer, seq, device):
    enc = tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits, _ = model(input_ids, attn_mask, return_attention=False)
    probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
    return probs  # P(flexible) per residue


def predict_sequence_esm3(model, tokenizer, seq, device):
    tokens = tokenizer.encode(seq)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits, _ = model(input_ids, attn_mask, return_attention=False)
    L_model = logits.shape[1]
    L_seq   = len(seq)
    # ESM3 tokenizer may add BOS/EOS tokens – trim to match sequence length
    if L_model > L_seq:
        offset = L_model - L_seq
        logits = logits[:, offset // 2: offset // 2 + L_seq, :]
    probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
    return probs


def run_esmfluc_inference(results_root, fasta_records, device, cache_dir):
    """Return dict: exp_name → long DataFrame (name, res_idx, prob_flex)."""
    results_root = Path(results_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_preds = {}
    for exp_name, is_esm3, frozen in EXPERIMENTS:
        cache_csv = cache_dir / f"{exp_name}_probs.csv"
        if cache_csv.exists():
            print(f"[INFO] Loading cached inference: {exp_name}")
            all_preds[exp_name] = pd.read_csv(cache_csv)
            continue

        ckpt_path = results_root / exp_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"[INFO] Running inference: {exp_name}")
        if is_esm3:
            if not ESM3_AVAILABLE:
                print(f"  [SKIP] ESM3 not available")
                continue
            model, tokenizer = build_esm3_model(device)
            predict_fn = predict_sequence_esm3
        else:
            if not HF_AVAILABLE:
                print(f"  [SKIP] transformers not available")
                continue
            model, tokenizer = build_esm2_model(device)
            predict_fn = predict_sequence_esm2

        load_checkpoint(model, ckpt_path, device)

        rows = []
        for i, (name, seq) in enumerate(fasta_records.items(), 1):
            if i % 50 == 0:
                print(f"    [{i}/{len(fasta_records)}]")
            try:
                probs = predict_fn(model, tokenizer, seq, device)
                L = min(len(probs), len(seq))
                for j in range(L):
                    rows.append({"name": name, "res_idx": j + 1, "prob_flex": float(probs[j])})
            except Exception as e:
                print(f"    [WARN] Failed {name}: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(cache_csv, index=False)
        all_preds[exp_name] = df

        # free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_preds


# =============================================================================
# Metrics
# =============================================================================
def youden_threshold(y_true, scores):
    """Find threshold maximising Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thresholds[best_idx])


def compute_metrics(y_true, scores, neq_continuous=None):
    """
    y_true   : binary array  (1 = flexible)
    scores   : continuous predictor  (higher = more flexible)
    neq_continuous : continuous Neq values (for Spearman, optional)
    """
    m = {}
    if len(np.unique(y_true)) < 2 or len(scores) < 5:
        return {k: np.nan for k in ["AUC_ROC", "AUC_PR", "F1", "MCC", "Spearman", "n", "n_pos"]}

    m["n"]     = int(len(y_true))
    m["n_pos"] = int(y_true.sum())

    try:
        m["AUC_ROC"] = float(roc_auc_score(y_true, scores))
    except Exception:
        m["AUC_ROC"] = np.nan

    try:
        m["AUC_PR"] = float(average_precision_score(y_true, scores))
    except Exception:
        m["AUC_PR"] = np.nan

    try:
        thresh = youden_threshold(y_true, scores)
        y_pred = (scores >= thresh).astype(int)
        m["F1"]  = float(f1_score(y_true, y_pred, zero_division=0))
        m["MCC"] = float(matthews_corrcoef(y_true, y_pred))
        m["threshold"] = thresh
    except Exception:
        m["F1"] = m["MCC"] = m["threshold"] = np.nan

    if neq_continuous is not None:
        try:
            rho, _ = spearmanr(neq_continuous, scores, nan_policy="omit")
            m["Spearman"] = float(rho)
        except Exception:
            m["Spearman"] = np.nan
    else:
        m["Spearman"] = np.nan

    return m


def evaluate_method(df_long, score_col, label_col="label", neq_col="Neq"):
    """
    Compute global + per-protein metrics.
    Returns (global_dict, per_protein_df).
    """
    # global
    valid = df_long[[score_col, label_col, neq_col]].dropna()
    global_m = compute_metrics(
        valid[label_col].values,
        valid[score_col].values,
        valid[neq_col].values,
    )

    # per-protein
    recs = []
    for name, g in df_long.groupby("name"):
        g2 = g[[score_col, label_col, neq_col]].dropna()
        m = compute_metrics(g2[label_col].values, g2[score_col].values, g2[neq_col].values)
        recs.append({"name": name, **m})
    per_protein = pd.DataFrame(recs)

    # bootstrap 95% CI on AUC-ROC (1000 resamples of per-protein means)
    auc_vals = per_protein["AUC_ROC"].dropna().values
    if len(auc_vals) >= 10:
        rng = np.random.default_rng(42)
        boots = [rng.choice(auc_vals, len(auc_vals), replace=True).mean() for _ in range(1000)]
        global_m["AUC_ROC_CI95_lo"] = float(np.percentile(boots, 2.5))
        global_m["AUC_ROC_CI95_hi"] = float(np.percentile(boots, 97.5))

    return global_m, per_protein


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ESMfluc Benchmark vs DynaMine")
    print("=" * 70)

    # ── Load ground truth ────────────────────────────────────────────────────
    print("\n[1/5] Loading ground truth …")
    df_neq = load_neq_csv(args.neq_csv, args.neq_thresh)
    print(f"  {df_neq['name'].nunique()} proteins, {len(df_neq)} residues")
    print(f"  Flexible (Neq>{args.neq_thresh}): {df_neq['label'].mean()*100:.1f}% of residues")

    df_nsp3 = load_nsp3(args.nsp3_csv)

    fasta_records = parse_fasta(args.fasta)
    print(f"  FASTA: {len(fasta_records)} sequences")

    # Merge Neq + NSP3 disorder
    df_base = df_neq.merge(df_nsp3, on=["name", "res_idx"], how="inner")
    print(f"  After merging with NSP3: {len(df_base)} rows")

    # ── b2btools / DynaMine ──────────────────────────────────────────────────
    print("\n[2/5] Running b2btools (DynaMine + DisoMine) …")
    if args.skip_b2b:
        print("  Skipped (--skip_b2b)")
        df_b2b = None
    else:
        b2b_cache = args.b2b_cache or str(cache_dir / "b2btools_preds.csv")
        df_b2b = run_b2btools(fasta_records, cache_path=b2b_cache,
                              force_api=getattr(args, "force_api", False))

    # ── ESMfluc inference ────────────────────────────────────────────────────
    print("\n[3/5] Running ESMfluc inference …")
    esmfluc_preds = run_esmfluc_inference(
        args.results_root, fasta_records, args.device, cache_dir
    )

    # ── Build merged long table ──────────────────────────────────────────────
    print("\n[4/5] Merging all predictions …")
    df_long = df_base.copy()

    # Add NSP3 disorder as a predictor (higher = more disordered = more flexible)
    # Already in df_base as nsp3_disorder

    # Add b2btools
    if df_b2b is not None:
        df_long = df_long.merge(df_b2b, on=["name", "res_idx"], how="left")
        # DynaMine backbone S2: low = flexible → invert for scoring
        if "dynamine_bb" in df_long.columns:
            df_long["dynamine_flex"] = 1.0 - df_long["dynamine_bb"]

    # Add ESMfluc probabilities
    for exp_name, df_pred in esmfluc_preds.items():
        df_pred = df_pred.rename(columns={"prob_flex": exp_name})
        df_long = df_long.merge(df_pred, on=["name", "res_idx"], how="left")

    df_long.to_csv(out_dir / "benchmark_long.csv", index=False)
    print(f"  Merged table saved → {out_dir / 'benchmark_long.csv'}")

    # ── Evaluate all methods ─────────────────────────────────────────────────
    print("\n[5/5] Computing metrics …")

    # Define method columns and display names
    method_cols = {}
    if "nsp3_disorder" in df_long.columns:
        method_cols["NSP3-Disorder"] = "nsp3_disorder"
    if "dynamine_flex" in df_long.columns:
        method_cols["DynaMine"] = "dynamine_flex"
    if "disomine" in df_long.columns:
        method_cols["DisoMine"] = "disomine"
    for exp_name in [e[0] for e in EXPERIMENTS]:
        if exp_name in df_long.columns:
            display = {
                "esm2_binary_frozen":    "ESMfluc-ESM2-frozen",
                "esm2_binary_unfrozen":  "ESMfluc-ESM2-unfrozen",
                "esm3_binary_frozen":    "ESMfluc-ESM3-frozen",
                "esm3_binary_unfrozen":  "ESMfluc-ESM3-unfrozen",
            }[exp_name]
            method_cols[display] = exp_name

    all_global = []
    all_perseq_dfs = {}
    for display_name, col in method_cols.items():
        print(f"  Evaluating: {display_name}")
        g_m, pp_df = evaluate_method(df_long, score_col=col)
        g_m["method"] = display_name
        all_global.append(g_m)
        all_perseq_dfs[display_name] = pp_df

    df_global = pd.DataFrame(all_global).set_index("method")

    # ── Save and print ───────────────────────────────────────────────────────
    df_global.to_csv(out_dir / "benchmark_global_metrics.csv")

    per_seq_long = []
    for method, df_pp in all_perseq_dfs.items():
        df_pp["method"] = method
        per_seq_long.append(df_pp)
    pd.concat(per_seq_long, ignore_index=True).to_csv(
        out_dir / "benchmark_per_protein_metrics.csv", index=False
    )

    # ── Wilcoxon test: best ESMfluc vs best baseline ─────────────────────────
    from scipy.stats import wilcoxon

    baseline_methods = [m for m in all_perseq_dfs if "ESMfluc" not in m]
    esmfluc_methods  = [m for m in all_perseq_dfs if "ESMfluc" in m]

    wilcoxon_rows = []
    if baseline_methods and esmfluc_methods:
        # pick best ESMfluc by global AUC-ROC
        best_esm = max(esmfluc_methods,
                       key=lambda m: df_global.loc[m, "AUC_ROC"] if m in df_global.index else 0)
        esm_aucs = all_perseq_dfs[best_esm]["AUC_ROC"].dropna().values

        for bl in baseline_methods:
            bl_aucs = all_perseq_dfs[bl]["AUC_ROC"].dropna().values
            # align on common proteins
            common_names = set(all_perseq_dfs[best_esm]["name"]) & set(all_perseq_dfs[bl]["name"])
            a = all_perseq_dfs[best_esm].set_index("name").loc[list(common_names), "AUC_ROC"].dropna()
            b = all_perseq_dfs[bl].set_index("name").loc[list(common_names), "AUC_ROC"].dropna()
            idx = a.index.intersection(b.index)
            a, b = a.loc[idx].values, b.loc[idx].values
            diff = a - b
            if len(diff) > 10 and not np.all(diff == 0):
                try:
                    stat, pval = wilcoxon(diff, alternative="greater")
                    wilcoxon_rows.append({
                        "comparison": f"{best_esm} vs {bl}",
                        "n_proteins": len(diff),
                        "median_delta_AUC": float(np.median(diff)),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(pval),
                    })
                except Exception as e:
                    print(f"  [WARN] Wilcoxon failed for {bl}: {e}")

    if wilcoxon_rows:
        df_wil = pd.DataFrame(wilcoxon_rows)
        df_wil.to_csv(out_dir / "benchmark_wilcoxon.csv", index=False)

    # ── Print summary table ──────────────────────────────────────────────────
    cols_to_show = ["AUC_ROC", "AUC_PR", "F1", "MCC", "Spearman"]
    df_show = df_global[[c for c in cols_to_show if c in df_global.columns]].copy()
    df_show = df_show.sort_values("AUC_ROC", ascending=False)

    print("\n")
    print("=" * 80)
    print("BENCHMARK RESULTS  (Neq > {:.1f} = flexible, 277 test proteins)".format(args.neq_thresh))
    print("=" * 80)

    header = f"{'Method':<30}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'F1':>8}  {'MCC':>8}  {'Spearman':>9}"
    print(header)
    print("-" * len(header))
    for method, row in df_show.iterrows():
        ci_str = ""
        if "AUC_ROC_CI95_lo" in df_global.columns:
            lo = df_global.loc[method, "AUC_ROC_CI95_lo"]
            hi = df_global.loc[method, "AUC_ROC_CI95_hi"]
            if not np.isnan(lo):
                ci_str = f" [{lo:.3f}–{hi:.3f}]"
        auc_str = f"{row['AUC_ROC']:.4f}{ci_str}" if not np.isnan(row["AUC_ROC"]) else "   N/A"
        print(
            f"  {method:<28}  {auc_str:>18}"
            f"  {row.get('AUC_PR', np.nan):>8.4f}"
            f"  {row.get('F1', np.nan):>8.4f}"
            f"  {row.get('MCC', np.nan):>8.4f}"
            f"  {row.get('Spearman', np.nan):>9.4f}"
        )

    if wilcoxon_rows:
        print("\nPairwise Wilcoxon (per-protein AUC-ROC, one-sided: ESMfluc > baseline):")
        for r in wilcoxon_rows:
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns"))
            print(f"  {r['comparison']}: median ΔAUC={r['median_delta_AUC']:+.4f}, "
                  f"p={r['p_value']:.4f} {sig}  (n={r['n_proteins']})")

    print(f"\nAll outputs saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
