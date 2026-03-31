#!/usr/bin/env python3
"""
Pipeline: train one BiLSTM-attention model per temperature, then extract
attention weights for the test set with each best checkpoint.

Temperatures: 320K, 348K, 379K, 413K, 450K

Outputs (inside scripts/final_pipeline/results/):
    results/mdcath_320K/best_model.pth
    results/mdcath_320K/test_attention.json
    ... (same for each temperature)

FASTA (created once):
    data/mdcath/test_split_mmseqs2.fasta

Usage:
    cd scripts/final_pipeline
    python run_mdcath_temp_pipeline.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent          # scripts/final_pipeline/
WORKSPACE    = PIPELINE_DIR.parent.parent               # ESMfluc-1/
DATA_DIR     = WORKSPACE / "data" / "mdcath"
TEMP_DATA_DIR = DATA_DIR / "per_temperature"
RESULTS_DIR  = PIPELINE_DIR / "results"
FASTA_PATH   = DATA_DIR / "test_split_mmseqs2.fasta"

TRAIN_CSV = DATA_DIR / "train_split_mmseqs2.csv"
TEST_CSV  = DATA_DIR / "test_split_mmseqs2.csv"

TEMPERATURES = [320, 348, 379, 413, 450]

# ── Shared training flags ─────────────────────────────────────────────────────
TRAIN_FLAGS = [
    "--epochs",       "80",
    "--lr",           "0.00001",
    "--num_classes",  "2",
    "--neq_thresholds", "1.0",
    "--architecture", "bilstm_attention",
    "--batch_size",   "4",
    "--esm_model",    "esm2_t33_650M_UR50D",
    "--num_layers",   "3",
    "--hidden_size",  "512",
    "--device",       "cuda",
    "--patience",     "3",
    "--freeze_layers","0-4",
    "--lr_scheduler", "reduce_on_plateau",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, cwd: Path):
    """Run a command in the given directory; raise on non-zero exit."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def prepare_temp_csvs():
    """
    For each temperature, create train_TK.csv and test_TK.csv with the
    temperature-specific neq column renamed to 'neq', which is what
    data_utils.load_and_preprocess_data() expects.
    """
    print("Preparing per-temperature CSV files...")
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    for temp in TEMPERATURES:
        col = f"neq_{temp}"
        for df, split in [(train_df, "train"), (test_df, "test")]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {split} CSV. "
                    f"Available columns: {list(df.columns)}"
                )
            out = df[["domain", "sequence", col]].rename(columns={col: "neq"})
            path = TEMP_DATA_DIR / f"{split}_{temp}K.csv"
            out.to_csv(path, index=False)
            print(f"  Saved {path.name} ({len(out)} sequences)")


def make_fasta():
    """Create a FASTA file for the test set (one entry per domain)."""
    print(f"\nCreating FASTA: {FASTA_PATH}")
    df = pd.read_csv(TEST_CSV)
    with open(FASTA_PATH, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['domain']}\n{row['sequence']}\n")
    print(f"  {len(df)} sequences written.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    prepare_temp_csvs()
    make_fasta()

    for temp in TEMPERATURES:
        bar = "━" * 58
        print(f"\n{bar}")
        print(f"  Temperature: {temp}K")
        print(f"{bar}")

        run_name   = f"mdcath_{temp}K"
        train_file = TEMP_DATA_DIR / f"train_{temp}K.csv"
        test_file  = TEMP_DATA_DIR / f"test_{temp}K.csv"
        checkpoint = RESULTS_DIR / run_name / "best_model.pth"
        attn_json  = RESULTS_DIR / run_name / "test_attention.json"

        # ── 1. Train ──────────────────────────────────────────────────────────
        print(f"\n[1/2] Training at {temp}K ...")
        run(
            [
                sys.executable, "train.py",
                "--train_data_file", train_file,
                "--test_data_file",  test_file,
                "--result_foldername", run_name,
                *TRAIN_FLAGS,
            ],
            cwd=PIPELINE_DIR,
        )

        # ── 2. Extract attention ──────────────────────────────────────────────
        if not checkpoint.exists():
            print(f"  WARNING: checkpoint not found at {checkpoint}, skipping attention step.")
            continue

        print(f"\n[2/2] Extracting attention at {temp}K ...")
        run(
            [
                sys.executable, "Attention/get_attn.py",
                "--checkpoint",   checkpoint,
                "--fasta_file",   FASTA_PATH,
                "--architecture", "bilstm_attention",
                "--num_classes",  "2",
                "--esm_model",    "esm2_t33_650M_UR50D",
                "--hidden_size",  "512",
                "--num_layers",   "3",
                "--output",       attn_json,
            ],
            cwd=PIPELINE_DIR,
        )

        print(f"\n  ✓ {temp}K done  →  {attn_json}")

    print(f"\n{'━' * 58}")
    print("All temperatures complete.")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"FASTA       : {FASTA_PATH}")
    for temp in TEMPERATURES:
        run_name  = f"mdcath_{temp}K"
        attn_json = RESULTS_DIR / run_name / "test_attention.json"
        ckpt_path = RESULTS_DIR / run_name / "best_model.pth"
        print(f"  {temp}K  checkpoint={ckpt_path}  attention={attn_json}")


if __name__ == "__main__":
    main()
