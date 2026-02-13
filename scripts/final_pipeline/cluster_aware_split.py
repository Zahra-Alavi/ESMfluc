#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 13 2026

Split clustered data into train/test sets while keeping entire clusters together.
This prevents data leakage by ensuring similar proteins are in the same split.

Usage:
    python cluster_aware_split.py --input clustered_data.csv --train_output train.csv --test_output test.csv --test_size 0.2 --seed 42
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split clustered data by cluster to prevent data leakage."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with cluster assignments (must have 'cluster_number' or 'cluster_id' column).",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        required=True,
        help="Output CSV file for training data.",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        required=True,
        help="Output CSV file for test data.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.0,
        help="Fraction of data to use for validation (default: 0.0, no validation split).",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default=None,
        help="Output CSV file for validation data (required if val_size > 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Try to stratify by cluster sizes (experimental).",
    )
    return parser.parse_args()


def split_by_clusters(df, test_size, val_size=0.0, seed=42, stratify=False):
    """
    Split DataFrame by clusters to prevent data leakage.
    
    Args:
        df: DataFrame with cluster assignments
        test_size: Fraction of data for test set
        val_size: Fraction of data for validation set
        seed: Random seed
        stratify: Whether to attempt stratification
    
    Returns:
        Tuple of (train_df, test_df) or (train_df, val_df, test_df) if val_size > 0
    """
    # Get cluster column
    cluster_col = None
    if "cluster_number" in df.columns:
        cluster_col = "cluster_number"
    elif "cluster_id" in df.columns:
        cluster_col = "cluster_id"
    else:
        raise ValueError("DataFrame must have 'cluster_number' or 'cluster_id' column")
    
    # Get unique clusters
    unique_clusters = df[cluster_col].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_clusters)
    
    print(f"[INFO] Total clusters: {len(unique_clusters)}")
    print(f"[INFO] Total sequences: {len(df)}")
    
    # Calculate split sizes
    if val_size > 0:
        # Three-way split
        if test_size + val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
        
        # Split clusters into train, val, test
        clusters_train, clusters_temp = train_test_split(
            unique_clusters,
            test_size=(test_size + val_size),
            random_state=seed,
        )
        
        # Further split temp into val and test
        relative_val_size = val_size / (test_size + val_size)
        clusters_val, clusters_test = train_test_split(
            clusters_temp,
            test_size=(1 - relative_val_size),
            random_state=seed,
        )
        
        # Create splits
        train_df = df[df[cluster_col].isin(clusters_train)].copy()
        val_df = df[df[cluster_col].isin(clusters_val)].copy()
        test_df = df[df[cluster_col].isin(clusters_test)].copy()
        
        print(f"[OK] Train: {len(clusters_train)} clusters, {len(train_df)} sequences ({len(train_df)/len(df)*100:.1f}%)")
        print(f"[OK] Val:   {len(clusters_val)} clusters, {len(val_df)} sequences ({len(val_df)/len(df)*100:.1f}%)")
        print(f"[OK] Test:  {len(clusters_test)} clusters, {len(test_df)} sequences ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    else:
        # Two-way split
        clusters_train, clusters_test = train_test_split(
            unique_clusters,
            test_size=test_size,
            random_state=seed,
        )
        
        # Create splits
        train_df = df[df[cluster_col].isin(clusters_train)].copy()
        test_df = df[df[cluster_col].isin(clusters_test)].copy()
        
        print(f"[OK] Train: {len(clusters_train)} clusters, {len(train_df)} sequences ({len(train_df)/len(df)*100:.1f}%)")
        print(f"[OK] Test:  {len(clusters_test)} clusters, {len(test_df)} sequences ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, test_df


def verify_no_leakage(train_df, test_df, val_df=None):
    """
    Verify that no clusters appear in multiple splits.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        val_df: Optional validation DataFrame
    
    Returns:
        True if no leakage detected, False otherwise
    """
    cluster_col = "cluster_number" if "cluster_number" in train_df.columns else "cluster_id"
    
    train_clusters = set(train_df[cluster_col].unique())
    test_clusters = set(test_df[cluster_col].unique())
    
    train_test_overlap = train_clusters & test_clusters
    
    if val_df is not None:
        val_clusters = set(val_df[cluster_col].unique())
        train_val_overlap = train_clusters & val_clusters
        test_val_overlap = test_clusters & val_clusters
        
        if train_test_overlap or train_val_overlap or test_val_overlap:
            print("[ERROR] Data leakage detected!")
            if train_test_overlap:
                print(f"  - Train/Test overlap: {len(train_test_overlap)} clusters")
            if train_val_overlap:
                print(f"  - Train/Val overlap: {len(train_val_overlap)} clusters")
            if test_val_overlap:
                print(f"  - Test/Val overlap: {len(test_val_overlap)} clusters")
            return False
    else:
        if train_test_overlap:
            print(f"[ERROR] Data leakage detected! Train/Test overlap: {len(train_test_overlap)} clusters")
            return False
    
    print("[OK] No data leakage detected - all clusters are in separate splits")
    return True


def main():
    args = parse_args()
    
    # Validate arguments
    if args.val_size > 0 and args.val_output is None:
        print("[ERROR] --val_output is required when --val_size > 0")
        return 1
    
    # Load clustered data
    print(f"[INFO] Loading clustered data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Check for required columns
    if "cluster_number" not in df.columns and "cluster_id" not in df.columns:
        print("[ERROR] Input CSV must have 'cluster_number' or 'cluster_id' column")
        print("[INFO] Run cluster_sequences.py first to assign clusters")
        return 1
    
    if "sequence" not in df.columns:
        print("[ERROR] Input CSV must have 'sequence' column")
        return 1
    
    print(f"[OK] Loaded {len(df)} sequences")
    
    # Perform cluster-aware split
    print(f"\n[INFO] Splitting data (test_size={args.test_size}, val_size={args.val_size}, seed={args.seed})")
    
    if args.val_size > 0:
        train_df, val_df, test_df = split_by_clusters(
            df, args.test_size, args.val_size, args.seed, args.stratify
        )
    else:
        train_df, test_df = split_by_clusters(
            df, args.test_size, 0.0, args.seed, args.stratify
        )
        val_df = None
    
    # Verify no leakage
    print("\n[INFO] Verifying splits...")
    if not verify_no_leakage(train_df, test_df, val_df):
        print("[ERROR] Split verification failed!")
        return 1
    
    # Save splits
    print(f"\n[INFO] Saving splits...")
    train_df.to_csv(args.train_output, index=False)
    print(f"[OK] Saved training data to {args.train_output}")
    
    test_df.to_csv(args.test_output, index=False)
    print(f"[OK] Saved test data to {args.test_output}")
    
    if val_df is not None:
        val_df.to_csv(args.val_output, index=False)
        print(f"[OK] Saved validation data to {args.val_output}")
    
    # Print summary
    print("\n=== Split Summary ===")
    print(f"Training:   {len(train_df)} sequences")
    if val_df is not None:
        print(f"Validation: {len(val_df)} sequences")
    print(f"Test:       {len(test_df)} sequences")
    
    return 0


if __name__ == "__main__":
    exit(main())
