#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 13 2026

Cluster protein sequences using MMseqs2 to avoid data leakage.
This script clusters proteins that are similar (e.g., 30% sequence identity)
so that proteins in the same cluster can be assigned to the same data split.

Usage:
    python cluster_sequences.py --input data.csv --output clusters.csv --min_seq_id 0.3 --coverage 0.8
"""

import argparse
import pandas as pd
import subprocess
import tempfile
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster protein sequences using MMseqs2 to prevent data leakage."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with 'sequence' column (and optional 'name' or ID column).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file with cluster assignments.",
    )
    parser.add_argument(
        "--min_seq_id",
        type=float,
        default=0.3,
        help="Minimum sequence identity threshold for clustering (default: 0.3 = 30%%).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Minimum coverage of alignment (default: 0.8 = 80%%).",
    )
    parser.add_argument(
        "--mmseqs_path",
        type=str,
        default="mmseqs",
        help="Path to mmseqs executable (default: mmseqs in PATH).",
    )
    return parser.parse_args()


def check_mmseqs_installed(mmseqs_path):
    """Check if MMseqs2 is installed and accessible."""
    try:
        result = subprocess.run(
            [mmseqs_path, "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"[INFO] MMseqs2 version: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def create_fasta_from_csv(df, fasta_path, id_column=None):
    """
    Create a FASTA file from DataFrame with sequences.
    
    Args:
        df: DataFrame with sequence data
        fasta_path: Path to write FASTA file
        id_column: Column name to use as sequence ID (default: index)
    """
    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            # Use provided ID column or fall back to index
            if id_column and id_column in df.columns:
                seq_id = str(row[id_column])
            else:
                seq_id = f"seq_{idx}"
            
            sequence = row["sequence"]
            f.write(f">{seq_id}\n{sequence}\n")
    print(f"[OK] Created FASTA file: {fasta_path}")


def run_mmseqs_clustering(fasta_path, output_dir, min_seq_id, coverage, mmseqs_path):
    """
    Run MMseqs2 easy-cluster command.
    
    Args:
        fasta_path: Input FASTA file
        output_dir: Directory for MMseqs2 output
        min_seq_id: Minimum sequence identity (0.0-1.0)
        coverage: Minimum alignment coverage (0.0-1.0)
        mmseqs_path: Path to mmseqs executable
    
    Returns:
        Path to cluster TSV file
    """
    db_prefix = os.path.join(output_dir, "db")
    clust_prefix = os.path.join(output_dir, "clust")
    tmp_dir = os.path.join(output_dir, "tmp")
    
    # Create tmp directory
    os.makedirs(tmp_dir, exist_ok=True)
    
    cmd = [
        mmseqs_path,
        "easy-cluster",
        fasta_path,
        clust_prefix,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
    ]
    
    print(f"[INFO] Running MMseqs2 clustering...")
    print(f"[CMD] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] MMseqs2 failed with return code {result.returncode}")
        print(f"[STDERR] {result.stderr}")
        raise RuntimeError("MMseqs2 clustering failed")
    
    print(f"[OK] MMseqs2 clustering completed")
    
    # The cluster file is typically named clust_cluster.tsv
    cluster_file = f"{clust_prefix}_cluster.tsv"
    return cluster_file


def parse_mmseqs_clusters(cluster_file):
    """
    Parse MMseqs2 cluster output file.
    
    Args:
        cluster_file: Path to MMseqs2 cluster TSV file
    
    Returns:
        Dictionary mapping sequence ID to cluster representative ID
    """
    clusters = {}
    
    with open(cluster_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                clusters[member] = representative
    
    print(f"[OK] Parsed {len(clusters)} sequence-to-cluster mappings")
    print(f"[INFO] Found {len(set(clusters.values()))} unique clusters")
    
    return clusters


def assign_clusters_to_dataframe(df, clusters, id_column=None):
    """
    Assign cluster IDs to DataFrame rows.
    
    Args:
        df: Original DataFrame
        clusters: Dictionary mapping sequence ID to cluster representative
        id_column: Column name used as sequence ID
    
    Returns:
        DataFrame with added 'cluster_id' column
    """
    df = df.copy()
    cluster_ids = []
    
    for idx, row in df.iterrows():
        # Get the sequence ID
        if id_column and id_column in df.columns:
            seq_id = str(row[id_column])
        else:
            seq_id = f"seq_{idx}"
        
        # Get cluster representative (or self if not clustered)
        cluster_rep = clusters.get(seq_id, seq_id)
        cluster_ids.append(cluster_rep)
    
    df["cluster_id"] = cluster_ids
    
    # Convert cluster representatives to integer cluster numbers for easier handling
    unique_clusters = sorted(set(cluster_ids))
    cluster_map = {rep: i for i, rep in enumerate(unique_clusters)}
    df["cluster_number"] = df["cluster_id"].map(cluster_map)
    
    return df


def main():
    args = parse_args()
    
    # Check if MMseqs2 is installed
    if not check_mmseqs_installed(args.mmseqs_path):
        print(f"[ERROR] MMseqs2 not found at '{args.mmseqs_path}'")
        print("[INFO] Please install MMseqs2:")
        print("       - conda install -c bioconda mmseqs2")
        print("       - or visit: https://github.com/soedinglab/mmseqs2")
        return 1
    
    # Load input data
    print(f"[INFO] Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    if "sequence" not in df.columns:
        print("[ERROR] Input CSV must have a 'sequence' column")
        return 1
    
    print(f"[OK] Loaded {len(df)} sequences")
    
    # Determine ID column (prefer 'name', then first column)
    id_column = None
    if "name" in df.columns:
        id_column = "name"
    elif "id" in df.columns:
        id_column = "id"
    
    # Create temporary directory for MMseqs2
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[INFO] Using temporary directory: {tmpdir}")
        
        # Create FASTA file
        fasta_path = os.path.join(tmpdir, "sequences.fasta")
        create_fasta_from_csv(df, fasta_path, id_column)
        
        # Run MMseqs2 clustering
        cluster_file = run_mmseqs_clustering(
            fasta_path,
            tmpdir,
            args.min_seq_id,
            args.coverage,
            args.mmseqs_path,
        )
        
        # Parse cluster assignments
        clusters = parse_mmseqs_clusters(cluster_file)
        
        # Assign clusters to DataFrame
        df_clustered = assign_clusters_to_dataframe(df, clusters, id_column)
    
    # Save output
    df_clustered.to_csv(args.output, index=False)
    print(f"[OK] Saved clustered data to {args.output}")
    
    # Print summary statistics
    print("\n=== Clustering Summary ===")
    print(f"Total sequences: {len(df_clustered)}")
    print(f"Number of clusters: {df_clustered['cluster_number'].nunique()}")
    print(f"Average cluster size: {len(df_clustered) / df_clustered['cluster_number'].nunique():.2f}")
    print(f"Largest cluster size: {df_clustered['cluster_number'].value_counts().max()}")
    print(f"Smallest cluster size: {df_clustered['cluster_number'].value_counts().min()}")
    
    return 0


if __name__ == "__main__":
    exit(main())
