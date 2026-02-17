import pandas as pd
import subprocess
import os
import tempfile
import argparse
from pathlib import Path

def run_mmseqs2_clustering(fasta_file, output_dir, seq_id_threshold=0.3, coverage_threshold=0.8):
    """
    Run MMseqs2 clustering on a FASTA file.
    
    Args:
        fasta_file: Path to input FASTA file
        output_dir: Directory for MMseqs2 outputs
        seq_id_threshold: Sequence identity threshold (default 0.3 = 30%)
        coverage_threshold: Coverage threshold (default 0.8 = 80%)
    
    Returns:
        Dictionary mapping domain IDs to cluster IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    db_file = os.path.join(output_dir, "sequenceDB")
    cluster_db = os.path.join(output_dir, "clusterDB")
    tmp_dir = os.path.join(output_dir, "tmp")
    cluster_tsv = os.path.join(output_dir, "clusters.tsv")
    
    print("Running MMseqs2 clustering...")
    
    # Create sequence database
    print(f"  Creating database...")
    subprocess.run([
        "mmseqs", "createdb", fasta_file, db_file
    ], check=True)
    
    # Cluster sequences
    print(f"  Clustering at {seq_id_threshold*100}% identity, {coverage_threshold*100}% coverage...")
    subprocess.run([
        "mmseqs", "cluster", db_file, cluster_db, tmp_dir,
        "--min-seq-id", str(seq_id_threshold),
        "-c", str(coverage_threshold),
        "--cov-mode", "0"  # coverage of query and target
    ], check=True)
    
    # Create TSV output
    print(f"  Creating TSV output...")
    subprocess.run([
        "mmseqs", "createtsv", db_file, db_file, cluster_db, cluster_tsv
    ], check=True)
    
    # Parse clusters
    print(f"  Parsing clusters...")
    cluster_map = {}
    with open(cluster_tsv, 'r') as f:
        for line in f:
            representative, member = line.strip().split('\t')
            cluster_map[member] = representative
    
    print(f"  Found {len(set(cluster_map.values()))} clusters from {len(cluster_map)} sequences")
    
    return cluster_map


def split_by_clusters(df, cluster_map, test_size=0.20, random_state=42):
    """
    Split dataframe by clusters to prevent data leakage.
    
    Args:
        df: DataFrame with 'domain' column
        cluster_map: Dictionary mapping domain IDs to cluster representatives
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df
    """
    # Add cluster assignment to dataframe
    df['cluster_id'] = df['domain'].map(cluster_map)
    
    # Get unique clusters
    unique_clusters = df['cluster_id'].unique()
    print(f"\nTotal unique clusters: {len(unique_clusters)}")
    
    # Split clusters (not individual sequences)
    import numpy as np
    np.random.seed(random_state)
    
    # Shuffle clusters
    shuffled_clusters = unique_clusters.copy()
    np.random.shuffle(shuffled_clusters)
    
    # Calculate split point
    n_test_clusters = int(len(shuffled_clusters) * test_size)
    test_clusters = set(shuffled_clusters[:n_test_clusters])
    train_clusters = set(shuffled_clusters[n_test_clusters:])
    
    # Split data by cluster assignment
    train_df = df[df['cluster_id'].isin(train_clusters)].copy()
    test_df = df[df['cluster_id'].isin(test_clusters)].copy()
    
    # Drop the cluster_id column from output
    train_df = train_df.drop(columns=['cluster_id'])
    test_df = test_df.drop(columns=['cluster_id'])
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Split mdCATH dataset using MMseqs2 clustering to prevent data leakage"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data/mdcath",
        help="Root directory containing mdcath_neq_dataset.csv"
    )
    parser.add_argument(
        "--seq_id_threshold",
        type=float,
        default=0.3,
        help="Sequence identity threshold for clustering (default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.8,
        help="Coverage threshold for clustering (default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction of clusters for test set (default: 0.20)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Read dataset
    dataset_path = os.path.join(args.data_root, "mdcath_neq_dataset.csv")
    print(f"Reading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Filter out rows with NaN values
    print(f"\nTotal domains: {len(df)}")
    df_clean = df[~df['neq_320'].astype(str).str.contains('nan', case=False)].copy()
    print(f"Valid domains (no NaN): {len(df_clean)}")
    
    # Create temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as fasta_file:
        fasta_path = fasta_file.name
        for _, row in df_clean.iterrows():
            fasta_file.write(f">{row['domain']}\n{row['sequence']}\n")
    
    print(f"\nWrote sequences to {fasta_path}")
    
    # Run MMseqs2 clustering
    mmseqs_output = os.path.join(args.data_root, "mmseqs2_output")
    cluster_map = run_mmseqs2_clustering(
        fasta_path, 
        mmseqs_output,
        seq_id_threshold=args.seq_id_threshold,
        coverage_threshold=args.coverage_threshold
    )
    
    # Clean up FASTA file
    os.unlink(fasta_path)
    
    # Split by clusters
    train_df, test_df = split_by_clusters(
        df_clean, 
        cluster_map, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Save splits
    train_path = os.path.join(args.data_root, "train_split_mmseqs2.csv")
    test_path = os.path.join(args.data_root, "test_split_mmseqs2.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Training set: {len(train_df)} domains")
    print(f"Test set: {len(test_df)} domains")
    print(f"Test fraction: {len(test_df)/(len(train_df)+len(test_df)):.2%}")
    print(f"\nSaved to:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
