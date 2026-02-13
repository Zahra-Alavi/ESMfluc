#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for cluster-aware data splitting.
This test creates synthetic clustered data and verifies the split works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_cluster_aware_split():
    """Test the cluster-aware splitting functionality."""
    
    print("="*80)
    print("Testing Cluster-Aware Data Splitting")
    print("="*80)
    print()
    
    # Create synthetic data with clusters
    np.random.seed(42)
    
    # Create 100 sequences in 10 clusters
    sequences = []
    neq_values = []
    cluster_numbers = []
    
    for cluster_id in range(10):
        # Each cluster has 10 sequences
        for seq_in_cluster in range(10):
            # Generate a random sequence
            seq_length = np.random.randint(50, 200)
            seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), seq_length))
            sequences.append(seq)
            
            # Generate random neq values
            neq = [float(np.random.uniform(1.0, 5.0)) for _ in range(seq_length)]
            neq_values.append(str(neq))
            
            # Assign cluster
            cluster_numbers.append(cluster_id)
    
    # Create DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'neq': neq_values,
        'cluster_number': cluster_numbers,
    })
    
    print(f"Created synthetic dataset:")
    print(f"  - Total sequences: {len(df)}")
    print(f"  - Number of clusters: {df['cluster_number'].nunique()}")
    print(f"  - Sequences per cluster: {len(df) / df['cluster_number'].nunique():.1f}")
    print()
    
    # Save to CSV
    input_file = "/tmp/test_clustered_data.csv"
    df.to_csv(input_file, index=False)
    print(f"Saved test data to: {input_file}")
    print()
    
    # Test cluster_aware_split
    print("Running cluster_aware_split.py...")
    print("-"*80)
    
    train_file = "/tmp/test_train.csv"
    test_file = "/tmp/test_test.csv"
    
    import subprocess
    result = subprocess.run([
        "python", "cluster_aware_split.py",
        "--input", input_file,
        "--train_output", train_file,
        "--test_output", test_file,
        "--test_size", "0.2",
        "--seed", "42",
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: cluster_aware_split.py failed with return code {result.returncode}")
        return False
    
    # Verify the split
    print()
    print("Verifying the split...")
    print("-"*80)
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Train set: {len(train_df)} sequences, {train_df['cluster_number'].nunique()} clusters")
    print(f"Test set: {len(test_df)} sequences, {test_df['cluster_number'].nunique()} clusters")
    print()
    
    # Check for data leakage
    train_clusters = set(train_df['cluster_number'].unique())
    test_clusters = set(test_df['cluster_number'].unique())
    overlap = train_clusters & test_clusters
    
    if overlap:
        print(f"ERROR: Data leakage detected! {len(overlap)} clusters in both train and test")
        print(f"Overlapping clusters: {sorted(overlap)}")
        return False
    
    print("✓ No data leakage detected - all clusters are in separate splits")
    
    # Check proportions
    total = len(train_df) + len(test_df)
    test_proportion = len(test_df) / total
    print(f"✓ Test proportion: {test_proportion:.2%} (target: 20%)")
    
    if abs(test_proportion - 0.2) > 0.1:  # Allow 10% deviation
        print(f"WARNING: Test proportion differs from target by more than 10%")
    
    # Clean up
    os.remove(input_file)
    os.remove(train_file)
    os.remove(test_file)
    print()
    print("✓ Test files cleaned up")
    
    print()
    print("="*80)
    print("All tests passed!")
    print("="*80)
    
    return True


def test_validation_split():
    """Test the three-way split (train/val/test)."""
    
    print()
    print("="*80)
    print("Testing Three-Way Split (Train/Val/Test)")
    print("="*80)
    print()
    
    # Create synthetic data
    np.random.seed(42)
    
    sequences = []
    neq_values = []
    cluster_numbers = []
    
    for cluster_id in range(20):  # More clusters for 3-way split
        for seq_in_cluster in range(5):
            seq_length = 100
            seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), seq_length))
            sequences.append(seq)
            neq = [float(np.random.uniform(1.0, 5.0)) for _ in range(seq_length)]
            neq_values.append(str(neq))
            cluster_numbers.append(cluster_id)
    
    df = pd.DataFrame({
        'sequence': sequences,
        'neq': neq_values,
        'cluster_number': cluster_numbers,
    })
    
    print(f"Created synthetic dataset: {len(df)} sequences, {df['cluster_number'].nunique()} clusters")
    print()
    
    # Save to CSV
    input_file = "/tmp/test_clustered_data_3way.csv"
    df.to_csv(input_file, index=False)
    
    # Test three-way split
    print("Running cluster_aware_split.py with validation split...")
    print("-"*80)
    
    train_file = "/tmp/test_train_3way.csv"
    val_file = "/tmp/test_val_3way.csv"
    test_file = "/tmp/test_test_3way.csv"
    
    import subprocess
    result = subprocess.run([
        "python", "cluster_aware_split.py",
        "--input", input_file,
        "--train_output", train_file,
        "--val_output", val_file,
        "--test_output", test_file,
        "--test_size", "0.2",
        "--val_size", "0.1",
        "--seed", "42",
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"ERROR: cluster_aware_split.py failed with return code {result.returncode}")
        return False
    
    # Verify the split
    print()
    print("Verifying the three-way split...")
    print("-"*80)
    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Train set: {len(train_df)} sequences, {train_df['cluster_number'].nunique()} clusters")
    print(f"Val set:   {len(val_df)} sequences, {val_df['cluster_number'].nunique()} clusters")
    print(f"Test set:  {len(test_df)} sequences, {test_df['cluster_number'].nunique()} clusters")
    print()
    
    # Check for data leakage
    train_clusters = set(train_df['cluster_number'].unique())
    val_clusters = set(val_df['cluster_number'].unique())
    test_clusters = set(test_df['cluster_number'].unique())
    
    train_val_overlap = train_clusters & val_clusters
    train_test_overlap = train_clusters & test_clusters
    val_test_overlap = val_clusters & test_clusters
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("ERROR: Data leakage detected!")
        if train_val_overlap:
            print(f"  - Train/Val overlap: {len(train_val_overlap)} clusters")
        if train_test_overlap:
            print(f"  - Train/Test overlap: {len(train_test_overlap)} clusters")
        if val_test_overlap:
            print(f"  - Val/Test overlap: {len(val_test_overlap)} clusters")
        return False
    
    print("✓ No data leakage detected - all clusters are in separate splits")
    
    # Clean up
    os.remove(input_file)
    os.remove(train_file)
    os.remove(val_file)
    os.remove(test_file)
    print("✓ Test files cleaned up")
    
    print()
    print("="*80)
    print("Three-way split test passed!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = True
    
    # Run tests
    success = test_cluster_aware_split() and success
    success = test_validation_split() and success
    
    if success:
        print()
        print("✓ All tests passed successfully!")
        sys.exit(0)
    else:
        print()
        print("✗ Some tests failed")
        sys.exit(1)
