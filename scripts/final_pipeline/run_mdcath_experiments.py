#!/usr/bin/env python3
"""
Orchestrate MDCATH temperature experiments.
For each temperature (320K, 348K, 379K, 413K, 450K):
  1. Create FASTA from test CSV
  2. Train classification model (NEQ threshold 1.0)
  3. Extract attention for classification
  4. Train regression model (raw NEQ)
  5. Extract attention for regression
  6. Collect results into summary table
"""

import os
import sys
import subprocess
import pandas as pd
import argparse
from pathlib import Path
import json
import time

# Temperature splits
TEMPERATURES = ["320K", "348K", "379K", "413K", "450K"]

# Model hyperparameters
BATCH_SIZE = 4
ESM_MODEL = "esm2_t33_650M_UR50D"
NUM_LAYERS = 3
HIDDEN_SIZE = 512
PATIENCE = 5
DEVICE = "cuda"

# Paths
DATA_DIR = Path("../../data/mdcath")  # Relative to scripts directory
SCRIPTS_DIR = Path(__file__).parent  # Current scripts directory
TEST_FASTA = DATA_DIR / "test_split.fasta"  # Single FASTA for all temperature experiments


def create_test_fasta(test_csv, output_fasta):
    """
    Convert CSV with sequence and protein_id to FASTA format.
    (Note: This function is not used - FASTA is created via create_mdcath_fasta.py)
    """
    print(f"Creating FASTA from {test_csv}...")
    df = pd.read_csv(test_csv)
    
    # Check required columns
    if 'sequence' not in df.columns or 'protein_id' not in df.columns:
        raise ValueError(f"CSV must have 'sequence' and 'protein_id' columns. Found: {df.columns.tolist()}")
    
    with open(output_fasta, 'w') as f:
        for _, row in df.iterrows():
            protein_id = row['protein_id']
            sequence = row['sequence']
            f.write(f">{protein_id}\n{sequence}\n")
    
    print(f"  Created {output_fasta} with {len(df)} sequences")
    return output_fasta


def run_classification_training(temp, train_csv, test_csv, output_dir):
    """
    Train classification model with NEQ threshold 1.0.
    """
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION TRAINING: {temp}")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # train_unified.py creates ./results/{result_foldername}
    # We need to run from a directory where ./results/ points to our RESULTS_DIR
    # So we'll use absolute paths for data and run from scripts dir
    folder_name = f"{temp}/classification"
    
    cmd = [
        "python", str(SCRIPTS_DIR / "train_unified.py"),
        "--train_data_file", str(train_csv.resolve()),
        "--test_data_file", str(test_csv.resolve()),
        "--task_type", "classification",
        "--neq_thresholds", "1.0",  # Binary classification: NEQ > 1.0
        "--num_classes", "2",
        "--architecture", "bilstm_attention",
        "--batch_size", str(BATCH_SIZE),
        "--esm_model", ESM_MODEL,
        "--num_layers", str(NUM_LAYERS),
        "--hidden_size", str(HIDDEN_SIZE),
        "--device", DEVICE,
        "--patience", str(PATIENCE),
        "--result_foldername", folder_name,
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    start_time = time.time()
    # Change to scripts directory before running so ./results/ works correctly
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=SCRIPTS_DIR)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Classification training failed for {temp}")
        return None
    
    print(f"\nClassification training completed in {elapsed/60:.2f} minutes")
    
    # The model will be in ./results/{folder_name}/best_model.pth relative to SCRIPTS_DIR
    model_path = SCRIPTS_DIR / "results" / folder_name / "best_model.pth"
    return model_path if model_path.exists() else None


def run_regression_training(temp, train_csv, test_csv, output_dir):
    """
    Train regression model on raw NEQ values.
    """
    print(f"\n{'='*80}")
    print(f"REGRESSION TRAINING: {temp}")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    folder_name = f"{temp}/regression"
    
    cmd = [
        "python", str(SCRIPTS_DIR / "train_unified.py"),
        "--train_data_file", str(train_csv.resolve()),
        "--test_data_file", str(test_csv.resolve()),
        "--task_type", "regression",
        "--architecture", "bilstm_attention",
        "--batch_size", str(BATCH_SIZE),
        "--esm_model", ESM_MODEL,
        "--num_layers", str(NUM_LAYERS),
        "--hidden_size", str(HIDDEN_SIZE),
        "--device", DEVICE,
        "--patience", str(PATIENCE),
        "--result_foldername", folder_name,
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=SCRIPTS_DIR)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Regression training failed for {temp}")
        return None
    
    print(f"\nRegression training completed in {elapsed/60:.2f} minutes")
    
    model_path = SCRIPTS_DIR / "results" / folder_name / "best_model.pth"
    return model_path if model_path.exists() else None


def extract_attention(checkpoint, fasta_file, output_dir, task_type):
    """
    Extract attention weights using get_attn.py.
    """
    print(f"\nExtracting attention for {task_type}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file (get_attn.py will append .json)
    output_base = output_dir / "attention_weights"
    
    cmd = [
        "python", str(SCRIPTS_DIR / "get_attn.py"),
        "--checkpoint", str(checkpoint),
        "--fasta_file", str(fasta_file),
        "--output", str(output_base),
        "--esm_model", ESM_MODEL,
        "--architecture", "bilstm_attention",
        "--task_type", task_type,
        "--hidden_size", str(HIDDEN_SIZE),
        "--num_layers", str(NUM_LAYERS),
    ]
    
    # Add task-specific parameters
    if task_type == "classification":
        cmd.extend(["--num_classes", "2"])
    else:  # regression
        cmd.extend(["--num_outputs", "1"])
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"WARNING: Attention extraction failed for {task_type}")
        return None
    
    print(f"Attention extraction completed")
    return output_dir


def collect_results(scripts_dir):
    """
    Collect all run_summary.csv files into a comprehensive table.
    """
    print(f"\n{'='*80}")
    print("COLLECTING RESULTS")
    print(f"{'='*80}\n")
    
    # Results are in scripts/results/{temp}/{task_type}/
    results_root = scripts_dir / "results"
    
    all_results = []
    
    for temp in TEMPERATURES:
        for task_type in ["classification", "regression"]:
            summary_file = results_root / temp / task_type / "run_summary.csv"
            
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                # Add metadata
                df['temperature'] = temp
                df['task_type'] = task_type
                all_results.append(df)
                print(f"✓ Loaded {summary_file}")
            else:
                print(f"✗ Missing {summary_file}")
    
    if not all_results:
        print("\nERROR: No results found!")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns to put metadata first
    metadata_cols = ['temperature', 'task_type']
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]
    
    # Save comprehensive summary
    summary_output = results_root / "mdcath_full_summary.csv"
    combined_df.to_csv(summary_output, index=False)
    print(f"\nSaved full summary: {summary_output}")
    
    # Create condensed summary table
    if 'task_type' in combined_df.columns:
        # For classification: show accuracy, F1, etc.
        # For regression: show MSE, MAE, R², etc.
        
        summary_cols = ['temperature', 'task_type']
        
        # Identify metric columns
        metric_cols = []
        for col in combined_df.columns:
            if any(metric in col.lower() for metric in ['accuracy', 'f1', 'mse', 'mae', 'r2', 'rmse', 'pearson', 'spearman']):
                metric_cols.append(col)
        
        if metric_cols:
            summary_cols.extend(metric_cols)
            condensed_df = combined_df[summary_cols]
        else:
            condensed_df = combined_df
        
        condensed_output = results_root / "mdcath_summary_condensed.csv"
        condensed_df.to_csv(condensed_output, index=False)
        print(f"Saved condensed summary: {condensed_output}")
        
        # Print to console
        print("\n" + "="*80)
        print("MDCATH EXPERIMENT SUMMARY")
        print("="*80)
        print(condensed_df.to_string(index=False))
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Run MDCATH temperature experiments")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only collect results"
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        default=TEMPERATURES,
        choices=TEMPERATURES,
        help="Which temperatures to process (default: all)"
    )
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("MDCATH TEMPERATURE EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Temperatures: {', '.join(args.temperatures)}")
    print(f"Tasks: Classification (threshold=1.0), Regression (raw NEQ)")
    print(f"Model: BiLSTM + {ESM_MODEL}")
    print(f"Params: hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, batch={BATCH_SIZE}")
    print(f"Backbone: Fully unfrozen")
    print(f"Output: {SCRIPTS_DIR}/results/")
    print(f"{'='*80}\n")
    
    # Resolve paths to absolute
    global DATA_DIR, TEST_FASTA
    DATA_DIR = DATA_DIR.resolve()
    TEST_FASTA = TEST_FASTA.resolve()
    
    # Check data directory exists
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Please ensure you're running from the scripts directory.")
        sys.exit(1)
    
    # Create test FASTA file if it doesn't exist
    if not TEST_FASTA.exists():
        print(f"\nCreating test FASTA file...")
        test_csv = DATA_DIR / "test_split_mmseq2.csv"
        subprocess.run([
            "python", str(SCRIPTS_DIR / "create_mdcath_fasta.py"),
            "--test_csv", str(test_csv),
            "--output", str(TEST_FASTA)
        ], check=True)
    else:
        print(f"\n✓ Using existing test FASTA: {TEST_FASTA}")
    
    if not args.skip_training:
        # Process each temperature
        for temp in args.temperatures:
            print(f"\n{'#'*80}")
            print(f"# PROCESSING TEMPERATURE: {temp}")
            print(f"{'#'*80}\n")
            
            # File paths
            train_csv = DATA_DIR / f"train_{temp}.csv"
            test_csv = DATA_DIR / f"test_{temp}.csv"
            
            # Check files exist
            if not train_csv.exists() or not test_csv.exists():
                print(f"ERROR: Missing CSV files for {temp}")
                continue
            
            # 1. CLASSIFICATION
            cls_output_dir = SCRIPTS_DIR / "results" / temp / "classification"
            cls_checkpoint = run_classification_training(temp, train_csv, test_csv, cls_output_dir)
            
            if cls_checkpoint and cls_checkpoint.exists():
                extract_attention(cls_checkpoint, TEST_FASTA, cls_output_dir, "classification")
            
            # 2. REGRESSION
            reg_output_dir = SCRIPTS_DIR / "results" / temp / "regression"
            reg_checkpoint = run_regression_training(temp, train_csv, test_csv, reg_output_dir)
            
            if reg_checkpoint and reg_checkpoint.exists():
                extract_attention(reg_checkpoint, TEST_FASTA, reg_output_dir, "regression")
    
    # Collect all results
    collect_results(SCRIPTS_DIR)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {SCRIPTS_DIR}/results/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
