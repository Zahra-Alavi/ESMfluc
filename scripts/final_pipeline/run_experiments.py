#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:54:55 2025

"""

import subprocess, sys, time, json, ast, re, os
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG (shared across all runs) ----------
BASE_ARGS = {
    "--esm_model": "esm2_t33_650M_UR50D",
    "--hidden_size": "512",
    "--num_layers": "3",
    "--dropout": "0.3",
    "--lr_scheduler": "reduce_on_plateau",
    "--epochs": "80",
    "--patience": "3",
    "--batch_size": "2",
    "--freeze_layers": "0-4",
    "--loss_function": "focal",
    "--num_classes": "2",
    "--neq_thresholds": "1.0",
    "--train_data_file": "./train_data.csv",
    "--test_data_file": "./test_data.csv",
    "--device": "cuda",
    #"--mixed_precision": "",    # flags (store_true) are included by key only
    "--amp_dtype": "bf16",
    #"--data_parallel": "",  # flags (store_true) are included by key only
}


HAS_BIDIRECTIONAL_FLAG = True

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_PY = SCRIPT_DIR / "main.py"
# train.py writes to ./results/<timestamp>
RESULTS_ROOTS = [
    (SCRIPT_DIR / "." / "results").resolve(),
    (SCRIPT_DIR / "results").resolve(),  # fallback in case path changed
]
LOG_DIR = (SCRIPT_DIR / "orchestrator_logs")
LOG_DIR.mkdir(exist_ok=True)

# ---------- utilities ----------
def dict_to_argv(d):
    """Turn {'--a':'1','--flag':''} into ['--a','1','--flag'] (store_true flags have empty value)."""
    argv = []
    for k, v in d.items():
        argv.append(k)
        if v != "":
            argv.append(str(v))
    return argv

def results_dirs_after(ts: float):
    """Find result directories created/modified after timestamp across possible roots."""
    found = []
    for root in RESULTS_ROOTS:
        if root.exists():
            for p in sorted(root.glob("*")):
                if p.is_dir() and p.stat().st_mtime >= ts - 1:
                    found.append(p)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

def parse_metrics_from_run_dir(run_dir: Path):
    """Load metrics.json and classification_report.txt if present."""
    metrics_path = run_dir / "metrics.json"
    report_path  = run_dir / "classification_report.txt"
    out = {
        "run_dir": str(run_dir),
        "total_seconds": None,
        "epochs_ran": None,
        "gpu_overall_peak_bytes": None,
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    }
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
            out["total_seconds"] = m.get("total_seconds")
            out["epochs_ran"] = m.get("epochs_ran")
            out["gpu_overall_peak_bytes"] = m.get("gpu_overall_peak_bytes")
        except Exception:
            pass
    if report_path.exists():
        try:
            # classification_report was saved as str(dict)
            rep = ast.literal_eval(report_path.read_text())
            out["accuracy"] = rep.get("accuracy")
            macro = rep.get("macro avg", {})
            wavg  = rep.get("weighted avg", {})
            out["macro_f1"] = macro.get("f1-score")
            out["weighted_f1"] = wavg.get("f1-score")
            
            agg_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}
            for cls_key, cls_stats in rep.items():
                if cls_key in agg_keys:
                    continue
                if not isinstance(cls_stats, dict):
                    continue
                safe = str(cls_key).strip().replace(" ", "_")
                out[f"cls_{safe}_precision"] = cls_stats.get("precision")
                out[f"cls_{safe}_recall"]    = cls_stats.get("recall")
                out[f"cls_{safe}_f1"]        = cls_stats.get("f1-score")
                out[f"cls_{safe}_support"]   = cls_stats.get("support")
                
        except Exception:
            pass
    return out

def guess_run_dir_from_stdout(stdout_text: str):
    # train.py prints: "Saved metrics to <path>/metrics.json"
    m = re.search(r"Saved metrics to (.+?/metrics\.json)", stdout_text)
    if m:
        return Path(m.group(1)).parent
    return None

def run_once(label: str, args_dict: dict, timeout_sec: int = 24*3600):
    """Run one experiment; return result dict with status and metrics."""
    cmd = ["python", "-u", str(MAIN_PY)] + dict_to_argv(args_dict)
    log_file = LOG_DIR / f"{label.replace(' ', '_')}.log"
    start_ts = time.time()
    print(f"\n=== RUN {label} ===")
    print(" ".join(cmd))
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"# {datetime.now().isoformat()} :: {label}\n")
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout_sec, check=False
            )
            lf.write(proc.stdout)
            lf.flush()
            status = "ok" if proc.returncode == 0 else f"fail (rc={proc.returncode})"
            # prefer stdout hint; fallback to newest results dir
            run_dir = guess_run_dir_from_stdout(proc.stdout)
            if run_dir is None:
                cands = results_dirs_after(start_ts)
                run_dir = cands[0] if cands else None
            metrics = parse_metrics_from_run_dir(run_dir) if run_dir else {}
            return {
                "label": label,
                "status": status,
                "run_dir": str(run_dir) if run_dir else None,
                **metrics,
            }
        except subprocess.TimeoutExpired:
            lf.write("\n# TIMEOUT\n")
            return {"label": label, "status": "timeout", "run_dir": None}
        except Exception as e:
            lf.write(f"\n# ERROR: {e}\n")
            return {"label": label, "status": f"error: {e}", "run_dir": None}

# ---------- experiment grids ----------
def base_copy(**overrides):
    d = dict(BASE_ARGS)
    d.update(overrides)
    return d

def grid_models():
    """Model variants under the shared base args."""
    items = []
    # 1) Transformer
    items.append(("Transformer", base_copy(**{"--architecture": "transformer"})))

    # 2) LSTM (uni)  -> bilstm arch with --bidirectional 0
    d = base_copy(**{"--architecture": "bilstm"})
    if HAS_BIDIRECTIONAL_FLAG:
        d["--bidirectional"] = "0"
    items.append(("LSTM", d))

    # 3) BiLSTM (bi)
    d = base_copy(**{"--architecture": "bilstm"})
    if HAS_BIDIRECTIONAL_FLAG:
        d["--bidirectional"] = "1"
    items.append(("BiLSTM", d))

    # 4) LSTM + attention (uni)
    d = base_copy(**{"--architecture": "bilstm_attention"})
    if HAS_BIDIRECTIONAL_FLAG:
        d["--bidirectional"] = "0"
    items.append(("LSTM+Attn", d))

    # 5) BiLSTM + attention (bi)
    d = base_copy(**{"--architecture": "bilstm_attention"})
    if HAS_BIDIRECTIONAL_FLAG:
        d["--bidirectional"] = "1"
    items.append(("BiLSTM+Attn", d))

    # 6) ESM-only linear token classifier
    items.append(("ESM_linear", base_copy(**{"--architecture": "esm_linear"})))

    return items


def grid_losses_on_bilstm_attn():
    """
    Loss/head sweep on bilstm_attention (same other params).
    Only supervised runs (no NC loss).
    """
    items = []
    arch = {"--architecture": "bilstm_attention"}
    if HAS_BIDIRECTIONAL_FLAG:
        arch["--bidirectional"] = "1"  # keep bi-attention by default

    # supervised only
    items.append((
        "Attn_supervised_CE",
        base_copy(**arch, **{"--loss_function": "crossentropy"})
    ))
    items.append((
        "Attn_supervised_Focal",
        base_copy(**arch, **{"--loss_function": "focal"})
    ))

    return items


def load_run_summary_row(label: str, status: str, run_dir: str) -> dict:
    """
    Load one-row summary for a run.
    - Prefer <run_dir>/run_summary.csv (written by train.py at the end).
    - Fallback to metrics/classification_report parsing if needed.
    """
    row = {"label": label, "status": status, "run_dir": run_dir}
    if not run_dir:
        return row

    run_dir = Path(run_dir)
    rs_path = run_dir / "run_summary.csv"
    if rs_path.exists():
        try:
            df = pd.read_csv(rs_path)
            if len(df) > 0:
                row.update(df.iloc[0].to_dict())
                return row
        except Exception:
            pass  # fall through to fallback


    parsed = parse_metrics_from_run_dir(run_dir)
    row.update(parsed)
    return row

# ---------- main ----------
def main():
    results = []

    # phase 1: model variants
    for label, args in grid_models():
        results.append(run_once(label, args))

    # phase 2: loss/head sweeps on BiLSTM+Attn
    for label, args in grid_losses_on_bilstm_attn():
        results.append(run_once(label, args))

    # Aggregate one row per run (prefer per-run run_summary.csv)
    rows = []
    for res in results:
        rows.append(
            load_run_summary_row(
                label=res.get("label", ""),
                status=res.get("status", ""),
                run_dir=res.get("run_dir")
            )
        )

    master_df = pd.DataFrame(rows)

   
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_csv = SCRIPT_DIR / f"all_runs_summary_{ts}.csv"
    master_df.to_csv(master_csv, index=False)

    # Console preview
    cols_pretty = [c for c in [
        "label", "status", "model_architecture", "loss_desc",
        "accuracy", "macro_avg_f1_score", "weighted_avg_f1_score",
        "total_time_seconds", "gpu_overall_peak_bytes", "epochs_ran", "run_dir"
    ] if c in master_df.columns]
    print("\n=== SUMMARY (first few cols) ===")
    if cols_pretty:
        print(master_df[cols_pretty].to_string(index=False))
    else:
        print(master_df.head().to_string(index=False))

    print(f"\nSaved master CSV: {master_csv}")
    print(f"Logs in: {LOG_DIR}")
    

if __name__ == "__main__":
    main()
