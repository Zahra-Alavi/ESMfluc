#!/usr/bin/env python3
"""
Analyze correlation between attention patterns and NEQ fluctuation values.

CORRELATION LOGIC:
- Each sequence has PER-RESIDUE NEQ values (one float per position)
- We compute PER-POSITION metrics (4 metrics per residue in the sequence)
- For each position i in a sequence: correlate attention_metric[i] with neq[i]
- Then correlate: across all positions in all sequences, do positions with high metric values also have high NEQ?

This answers: "Are positions with high NEQ characterized by high incoming attention / high in-degree / high betweenness / high clustering?"

Outputs:
- CSV with all metrics per position per sequence, matched with per-residue NEQ
- Correlation coefficients (Pearson r and p-value)
- Scatter plots showing relationships
"""

import argparse
import json
import csv
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import networkx as nx
except ImportError:
    nx = None


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load attention data from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_csv_neq(path: str) -> Dict[str, np.ndarray]:
    """Load per-residue NEQ values from CSV.
    
    Expects format with columns: name, sequence, neq (tab or comma-delimited)
    Where neq is a JSON array string like [1.0, 2.5, 3.1, ...]
    
    Returns dict mapping sequence_string -> array of per-residue NEQ values
    """
    neq_map = {}
    with open(path, 'r') as f:
        # Try to detect delimiter (tab or comma)
        first_line = f.readline()
        f.seek(0)
        delimiter = '\t' if '\t' in first_line else ','
        
        reader = csv.DictReader(f, delimiter=delimiter)
        row_count = 0
        
        for row in reader:
            if row is None:
                continue
            
            sequence = row.get('sequence', '').strip()
            neq_str = row.get('neq', '').strip()
            
            if not sequence or not neq_str:
                print(f"Warning: Row {row_count} missing 'sequence' or 'neq' column")
                row_count += 1
                continue
            
            # Parse JSON array
            try:
                neq_array = json.loads(neq_str)
                neq_map[sequence] = np.array(neq_array, dtype=np.float32)
                row_count += 1
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Row {row_count} - Could not parse NEQ array for sequence {sequence[:20]}...: {e}")
                row_count += 1
                continue
    
    print(f"  → Successfully loaded {len(neq_map)} sequences with NEQ values")
    return neq_map


def compute_incoming_attention(attn_matrix: np.ndarray) -> np.ndarray:
    """
    Compute average incoming attention per position.
    
    For each position j, sum attention from all positions divided by sequence length.
    """
    # Column j = attention received at position j from all positions
    incoming = attn_matrix.sum(axis=0)  # shape (L,)
    # Normalize by sequence length to get average
    incoming_avg = incoming / attn_matrix.shape[0]
    return incoming_avg


def build_sparse_graph(
    attn_matrix: np.ndarray,
    threshold: float = 0.0,
    topk: int = 0,
    percentile: int = 0,
    local_peaks: bool = False,
    normalize: bool = False,
) -> nx.DiGraph:
    """
    Build sparsified attention graph with multiple filtering strategies.
    
    Args:
        attn_matrix: Attention matrix (L, L)
        threshold: Minimum absolute edge weight to keep
        topk: Keep only top-k strongest targets per query
        percentile: Keep only top-X percentile weights per row (0-100)
        local_peaks: Keep only local maxima (peaks) in each row
        normalize: Row-normalize before filtering
    
    Returns NetworkX directed graph.
    """
    L = attn_matrix.shape[0]
    G = nx.DiGraph()
    
    # Add all nodes
    for i in range(L):
        G.add_node(i)
    
    work = attn_matrix.copy()
    
    if normalize:
        row_sums = work.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            work = np.divide(work, row_sums, where=row_sums != 0)
    
    # Apply sparsification strategy
    for i in range(L):
        row = work[i]
        selected_indices = []
        
        if topk and topk > 0:
            # Keep top-k targets per query
            k = min(topk, L)
            selected_indices = list(np.argpartition(-row, k - 1)[:k])
        elif percentile > 0:
            # Keep only top-X percentile per row
            threshold_val = np.percentile(row, 100 - percentile)
            selected_indices = list(np.where(row >= threshold_val)[0])
        elif local_peaks:
            # Keep only local maxima (peaks) in the row
            for j in range(L):
                is_peak = False
                if j == 0:
                    is_peak = row[j] >= row[j + 1] if L > 1 else True
                elif j == L - 1:
                    is_peak = row[j] >= row[j - 1]
                else:
                    is_peak = row[j] >= row[j - 1] and row[j] >= row[j + 1]
                
                if is_peak and row[j] >= threshold:
                    selected_indices.append(j)
        else:
            # Standard thresholding
            selected_indices = list(np.where(row >= threshold)[0])
        
        # Add edges for selected indices
        for j in selected_indices:
            w = float(row[j])
            if np.isfinite(w) and w >= threshold:
                G.add_edge(i, j, weight=w)
    
    return G


def compute_graph_metrics(G: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute graph metrics for all nodes.
    
    Returns:
    - in_degree: array of in-degree per node (how many attend to it)
    - out_degree: array of out-degree per node (how many it attends to)
    - betweenness: array of betweenness centrality per node
    - clustering: array of clustering coefficient per node
    """
    L = len(G.nodes())
    
    # In-degree (more useful than out-degree when using topk)
    in_degree = np.array([G.in_degree(i) for i in range(L)])
    
    # Out-degree (will be constant if topk is used)
    out_degree = np.array([G.out_degree(i) for i in range(L)])
    
    # Betweenness centrality
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness = np.array([betweenness_dict[i] for i in range(L)])
    
    # Clustering coefficient (for directed graphs, use undirected version)
    G_undirected = G.to_undirected()
    clustering_dict = nx.clustering(G_undirected)
    clustering = np.array([clustering_dict[i] for i in range(L)])
    
    return in_degree, out_degree, betweenness, clustering


def analyze_sequence(
    seq_name: str,
    attn_matrix: np.ndarray,
    sequence: str,
    neq_array: np.ndarray,
    threshold: float = 0.0,
    topk: int = 0,
    percentile: int = 0,
    local_peaks: bool = False,
    normalize: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a single sequence for attention-NEQ correlation.
    
    Args:
        seq_name: Sequence identifier
        attn_matrix: Attention matrix (L, L)
        sequence: Sequence string
        neq_array: Per-residue NEQ values (length L)
        threshold: Minimum absolute edge weight to keep
        topk: Keep only top-k strongest targets per query
        percentile: Keep only top-X percentile weights per row (0-100)
        local_peaks: Keep only local maxima (peaks) in each row
        normalize: Row-normalize before filtering
    
    Returns dict with all metrics per position.
    """
    if nx is None:
        raise RuntimeError("networkx required")
    
    L = attn_matrix.shape[0]
    
    # Validate NEQ array length matches attention matrix
    if len(neq_array) != L:
        raise ValueError(f"NEQ array length {len(neq_array)} != attention matrix length {L}")
    
    # Compute incoming attention
    incoming_attn = compute_incoming_attention(attn_matrix)
    
    # Build sparse graph and compute metrics
    G = build_sparse_graph(
        attn_matrix,
        threshold=threshold,
        topk=topk,
        percentile=percentile,
        local_peaks=local_peaks,
        normalize=normalize
    )
    in_degree, out_degree, betweenness, clustering = compute_graph_metrics(G)
    
    # Return per-position metrics with per-residue NEQ
    result = {
        'seq_name': seq_name,
        'seq_length': L,
        'neq_array': neq_array,  # Per-residue NEQ values
        'incoming_attention': incoming_attn,
        'in_degree': in_degree,
        'betweenness': betweenness,
        'clustering': clustering,
    }
    
    return result


def aggregate_across_sequences(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Aggregate metrics across all sequences, matching per-position metrics with per-residue NEQ.
    
    For each sequence, aligns attention metrics[i] with neq_array[i] position-by-position.
    
    Returns dict with concatenated arrays for correlation analysis.
    """
    all_incoming = []
    all_in_degree = []
    all_betweenness = []
    all_clustering = []
    all_neq = []
    all_seq_info = []
    all_position = []
    
    for res in results:
        L = res['seq_length']
        neq_array = res['neq_array']  # Per-residue NEQ values
        
        # Match position-by-position: metric[i] with neq_array[i]
        all_incoming.extend(res['incoming_attention'].tolist())
        all_in_degree.extend(res['in_degree'].tolist())
        all_betweenness.extend(res['betweenness'].tolist())
        all_clustering.extend(res['clustering'].tolist())
        all_neq.extend(neq_array.tolist())  # Use actual per-residue values
        all_seq_info.extend([res['seq_name']] * L)
        all_position.extend(list(range(L)))
    
    return {
        'incoming_attention': np.array(all_incoming),
        'in_degree': np.array(all_in_degree),
        'betweenness': np.array(all_betweenness),
        'clustering': np.array(all_clustering),
        'neq': np.array(all_neq),
        'seq_name': np.array(all_seq_info),
        'position': np.array(all_position),
    }


def main():
    p = argparse.ArgumentParser(
        description="Analyze correlation between attention patterns and NEQ fluctuation."
    )
    p.add_argument("--attention-json", "-a", required=True, help="Input JSON with attention matrices")
    p.add_argument("--neq-csv", "-n", required=True, help="CSV with per-residue NEQ values")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory for results")
    
    # Sparsification options (use one or combine them)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum absolute edge weight to keep (default: 0.0, keep all)"
    )
    p.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Keep only top-k strongest targets per query (e.g., --topk 5). Overrides threshold."
    )
    p.add_argument(
        "--percentile",
        type=int,
        default=0,
        help="Keep only top-X percentile weights per row (0-100, e.g., --percentile 25 keeps top 25%%). Overrides threshold."
    )
    p.add_argument(
        "--local-peaks",
        action="store_true",
        help="Keep only local maxima (peaks) in each row. Useful for sharp attention patterns."
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Row-normalize attention weights before filtering."
    )
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading attention from {args.attention_json}")
    attn_data = load_json(args.attention_json)
    print(f"  → Found {len(attn_data)} sequences in attention JSON")
    
    print(f"\nLoading NEQ labels from {args.neq_csv}")
    neq_map = load_csv_neq(args.neq_csv)
    
    if not neq_map:
        print("✗ ERROR: No NEQ data loaded! Check CSV format and column names.")
        return
    
    if not attn_data:
        print("✗ ERROR: No attention data loaded! Check JSON format.")
        return
    
    # Analyze each sequence
    print(f"\nMatching {len(attn_data)} attention sequences with {len(neq_map)} NEQ sequences...")
    results = []
    matched_count = 0
    
    for rec in attn_data:
        seq_name = rec.get('name')
        attn_matrix = np.array(rec.get('attention_weights'))
        sequence = rec.get('sequence', '')
        
        # Find matching per-residue NEQ array by sequence string
        neq_array = neq_map.get(sequence)
        if neq_array is None:
            print(f"  ⚠ No NEQ match: {seq_name} (length={attn_matrix.shape[0]})")
            continue
        
        print(f"  ✓ Matched {seq_name} (length={attn_matrix.shape[0]}, NEQ range=[{neq_array.min():.3f}, {neq_array.max():.3f}])")
        matched_count += 1
        
        try:
            res = analyze_sequence(
                seq_name, attn_matrix, sequence, neq_array,
                threshold=args.threshold,
                topk=args.topk,
                percentile=args.percentile,
                local_peaks=args.local_peaks,
                normalize=args.normalize
            )
            results.append(res)
        except Exception as e:
            print(f"  ✗ Error processing {seq_name}: {e}")
            continue
    
    if not results:
        print("\n✗ ERROR: No sequences processed. Check that:")
        print("  1. Sequences in attention JSON match exactly with sequences in NEQ CSV")
        print("  2. NEQ CSV columns are: name, sequence, neq")
        print("  3. NEQ column contains valid JSON arrays")
        return
    
    # Aggregate metrics
    agg = aggregate_across_sequences(results)
    
    # Compute correlations
    metrics = ['incoming_attention', 'in_degree', 'betweenness', 'clustering']
    correlations = {}
    
    print("\n" + "="*60)
    print("CORRELATION RESULTS (Pearson)")
    print("="*60)
    
    for metric_name in metrics:
        metric_vals = agg[metric_name]
        neq_vals = agg['neq']
        
        # Remove NaN/inf values
        mask = np.isfinite(metric_vals) & np.isfinite(neq_vals)
        metric_vals = metric_vals[mask]
        neq_vals = neq_vals[mask]
        
        if len(metric_vals) < 2:
            print(f"{metric_name}: Not enough data")
            continue
        
        r, p = pearsonr(metric_vals, neq_vals)
        correlations[metric_name] = {'r': r, 'p': p}
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{metric_name:25s}: r={r:7.4f}, p={p:.2e} {sig}")
    
    # Save results to CSV
    output_csv = os.path.join(args.output_dir, "position_metrics.csv")
    df = pd.DataFrame({
        'seq_name': agg['seq_name'],
        'position': agg['position'],
        'neq': agg['neq'],
        'incoming_attention': agg['incoming_attention'],
        'in_degree': agg['in_degree'],
        'betweenness': agg['betweenness'],
        'clustering': agg['clustering'],
    })
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved position-level metrics to {output_csv}")
    
    # Save correlation results
    corr_csv = os.path.join(args.output_dir, "correlations.csv")
    with open(corr_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'pearson_r', 'p_value'])
        for metric_name, corr_dict in correlations.items():
            writer.writerow([metric_name, corr_dict['r'], corr_dict['p']])
    print(f"✓ Saved correlations to {corr_csv}")
    
    # Create visualizations
    # Build sparsification strategy description
    strategy_parts = []
    if args.topk > 0:
        strategy_parts.append(f"topk={args.topk}")
    elif args.percentile > 0:
        strategy_parts.append(f"percentile={args.percentile}")
    elif args.local_peaks:
        strategy_parts.append("local-peaks")
    else:
        strategy_parts.append(f"threshold={args.threshold}")
    
    if args.normalize:
        strategy_parts.append("normalized")
    
    strategy_str = ", ".join(strategy_parts)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Attention Metrics vs NEQ ({strategy_str})", fontsize=16)
    
    for ax, metric_name in zip(axes.flat, metrics):
        metric_vals = agg[metric_name]
        neq_vals = agg['neq']
        
        # Remove NaN/inf
        mask = np.isfinite(metric_vals) & np.isfinite(neq_vals)
        metric_vals = metric_vals[mask]
        neq_vals = neq_vals[mask]
        
        if metric_name in correlations:
            r = correlations[metric_name]['r']
            p = correlations[metric_name]['p']
        else:
            r = p = np.nan
        
        ax.scatter(metric_vals, neq_vals, alpha=0.3, s=20)
        ax.set_xlabel(metric_name)
        ax.set_ylabel('NEQ value')
        ax.set_title(f'r={r:.3f}, p={p:.2e}')
        
        # Add trend line
        if len(metric_vals) > 1:
            z = np.polyfit(metric_vals, neq_vals, 1)
            p_trend = np.poly1d(z)
            x_trend = np.linspace(metric_vals.min(), metric_vals.max(), 100)
            ax.plot(x_trend, p_trend(x_trend), 'r--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "correlations.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Saved scatter plots to {plot_path}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
