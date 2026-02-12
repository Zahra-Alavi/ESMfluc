#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Attention Matrix Analysis

Performs three main analyses:
1. 1D Autocorrelation/FFT: Detects periodic patterns in attention (helices, beta-sheets)
2. 2D Patch Mining: Discovers recurring attention motifs using k-means/NMF
3. Periodicity Maps: Visualizes dominant attention periodicities along the sequence

Usage:
    python analyze_attention_patterns.py --attention_json data.json --output_dir ./results
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze attention matrix patterns"
    )
    parser.add_argument(
        "--attention_json",
        type=str,
        required=True,
        help="Path to JSON file with attention data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=11,
        help="Size of k×k patches for motif mining (default: 11)"
    )
    parser.add_argument(
        "--n_motifs",
        type=int,
        default=10,
        help="Number of attention motifs to discover (default: 10)"
    )
    parser.add_argument(
        "--max_lag",
        type=int,
        default=20,
        help="Maximum lag for autocorrelation (default: 20)"
    )
    return parser.parse_args()


def compute_autocorrelation(signal_1d, max_lag):
    """
    Compute autocorrelation of a 1D signal up to max_lag.
    
    Returns:
        lags: array of lag values
        acf: autocorrelation function values
    """
    n = len(signal_1d)
    signal_centered = signal_1d - np.mean(signal_1d)
    
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.corrcoef(signal_centered[:-lag], signal_centered[lag:])[0, 1]
    
    lags = np.arange(max_lag + 1)
    return lags, acf


def find_peaks_in_autocorr(acf, lags, min_height=0.1):
    """
    Find significant peaks in autocorrelation function.
    
    Returns:
        peak_lags: lags where peaks occur
        peak_heights: correlation values at peaks
    """
    # Find peaks (exclude lag 0 which is always 1.0)
    peaks, properties = signal.find_peaks(acf[1:], height=min_height)
    peak_lags = lags[peaks + 1]  # +1 because we excluded lag 0
    peak_heights = properties['peak_heights']
    
    return peak_lags, peak_heights


def analyze_1d_autocorr(attention_matrix, max_lag=20, ss_list=None):
    """
    Analysis 1: 1D Autocorrelation/FFT on columns and rows.
    
    For each residue, compute autocorrelation of its column (received attention)
    and row (given attention). Peaks at lag 3-4 suggest helices, lag 2 suggests sheets.
    
    Returns:
        results: dict with autocorr data per residue
    """
    L = attention_matrix.shape[0]
    
    results = {
        'residue_idx': [],
        'dominant_lag_col': [],
        'dominant_corr_col': [],
        'dominant_lag_row': [],
        'dominant_corr_row': [],
        'ss': []
    }
    
    for i in range(L):
        # Column autocorr (attention received by residue i)
        col_signal = attention_matrix[:, i]
        lags_col, acf_col = compute_autocorrelation(col_signal, max_lag)
        peak_lags_col, peak_heights_col = find_peaks_in_autocorr(acf_col, lags_col)
        
        # Row autocorr (attention given by residue i)
        row_signal = attention_matrix[i, :]
        lags_row, acf_row = compute_autocorrelation(row_signal, max_lag)
        peak_lags_row, peak_heights_row = find_peaks_in_autocorr(acf_row, lags_row)
        
        # Store dominant peak (highest correlation)
        if len(peak_lags_col) > 0:
            max_idx_col = np.argmax(peak_heights_col)
            dom_lag_col = peak_lags_col[max_idx_col]
            dom_corr_col = peak_heights_col[max_idx_col]
        else:
            dom_lag_col = np.nan
            dom_corr_col = np.nan
        
        if len(peak_lags_row) > 0:
            max_idx_row = np.argmax(peak_heights_row)
            dom_lag_row = peak_lags_row[max_idx_row]
            dom_corr_row = peak_heights_row[max_idx_row]
        else:
            dom_lag_row = np.nan
            dom_corr_row = np.nan
        
        results['residue_idx'].append(i)
        results['dominant_lag_col'].append(dom_lag_col)
        results['dominant_corr_col'].append(dom_corr_col)
        results['dominant_lag_row'].append(dom_lag_row)
        results['dominant_corr_row'].append(dom_corr_row)
        results['ss'].append(ss_list[i] if ss_list else None)
    
    return pd.DataFrame(results)


def extract_patches(attention_matrix, patch_size):
    """
    Extract k×k patches from attention matrix.
    
    Returns:
        patches: array of shape (n_patches, patch_size*patch_size)
        positions: array of (i, j) center positions
    """
    L = attention_matrix.shape[0]
    half = patch_size // 2
    
    patches = []
    positions = []
    
    for i in range(half, L - half):
        for j in range(half, L - half):
            patch = attention_matrix[i-half:i+half+1, j-half:j+half+1]
            patches.append(patch.flatten())
            positions.append((i, j))
    
    return np.array(patches), np.array(positions)


def analyze_2d_patches(attention_matrix, patch_size=11, n_motifs=10, ss_list=None, seq_id=""):
    """
    Analysis 2: 2D Patch Mining.
    
    Extract k×k tiles, z-score normalize, and cluster using k-means to find
    recurring attention motifs. Test enrichment for SS classes.
    
    Returns:
        motifs: array of motif prototypes (centroids)
        patch_labels: cluster assignment for each patch
        positions: (i,j) positions of patches
        enrichment: SS enrichment per motif
    """
    # Extract patches
    patches, positions = extract_patches(attention_matrix, patch_size)
    
    # Z-score normalize each patch
    scaler = StandardScaler()
    patches_normalized = scaler.fit_transform(patches)
    
    # K-means clustering to find motifs
    kmeans = KMeans(n_clusters=n_motifs, random_state=42, n_init=10)
    patch_labels = kmeans.fit_predict(patches_normalized)
    motifs = kmeans.cluster_centers_
    
    # Compute SS enrichment for each motif
    enrichment = {}
    if ss_list:
        for motif_idx in range(n_motifs):
            mask = patch_labels == motif_idx
            motif_positions = positions[mask]
            
            # Get SS at center of each patch
            ss_at_patches = [ss_list[pos[0]] for pos in motif_positions]
            ss_counts = pd.Series(ss_at_patches).value_counts(normalize=True).to_dict()
            
            # Compute distance distribution
            distances = [abs(pos[0] - pos[1]) for pos in motif_positions]
            
            enrichment[motif_idx] = {
                'ss_distribution': ss_counts,
                'mean_distance': np.mean(distances),
                'n_patches': mask.sum()
            }
    
    # Reshape motifs back to 2D
    motifs_2d = motifs.reshape(n_motifs, patch_size, patch_size)
    
    return motifs_2d, patch_labels, positions, enrichment


def create_periodicity_map(attention_matrix, max_lag=20):
    """
    Analysis 3: Periodicity Map.
    
    For each residue, find dominant lag in autocorrelation.
    Plot as "periodicity map" showing which residues have helical (3-4) 
    or sheet (2) patterns.
    
    Returns:
        periodicity: array of dominant lags per residue
        strength: array of correlation strength at dominant lag
    """
    L = attention_matrix.shape[0]
    periodicity = np.zeros(L)
    strength = np.zeros(L)
    
    for i in range(L):
        # Use column (received attention)
        col_signal = attention_matrix[:, i]
        lags, acf = compute_autocorrelation(col_signal, max_lag)
        peak_lags, peak_heights = find_peaks_in_autocorr(acf, lags)
        
        if len(peak_lags) > 0:
            max_idx = np.argmax(peak_heights)
            periodicity[i] = peak_lags[max_idx]
            strength[i] = peak_heights[max_idx]
        else:
            periodicity[i] = 0
            strength[i] = 0
    
    return periodicity, strength


def plot_autocorr_analysis(df_autocorr, seq_id, output_dir):
    """
    Visualize autocorrelation analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{seq_id}: Autocorrelation Analysis", fontsize=16)
    
    # 1. Dominant lag distribution (columns)
    ax = axes[0, 0]
    valid_lags = df_autocorr['dominant_lag_col'].dropna()
    ax.hist(valid_lags, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=3.5, color='red', linestyle='--', label='Helix (3-4)', linewidth=2)
    ax.axvline(x=2, color='blue', linestyle='--', label='Sheet (2)', linewidth=2)
    ax.set_xlabel('Dominant Lag')
    ax.set_ylabel('Count')
    ax.set_title('Column Autocorr: Dominant Lag Distribution')
    ax.legend()
    
    # 2. Lag vs position (colored by SS)
    ax = axes[0, 1]
    if 'ss' in df_autocorr.columns and df_autocorr['ss'].notna().any():
        ss_colors = {'H': 'red', 'E': 'blue', 'C': 'gray'}
        for ss_type, color in ss_colors.items():
            mask = df_autocorr['ss'] == ss_type
            ax.scatter(df_autocorr.loc[mask, 'residue_idx'],
                      df_autocorr.loc[mask, 'dominant_lag_col'],
                      c=color, label=ss_type, alpha=0.6, s=30)
        ax.legend(title='SS')
    else:
        ax.scatter(df_autocorr['residue_idx'], df_autocorr['dominant_lag_col'], alpha=0.6, s=30)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Dominant Lag')
    ax.set_title('Column Autocorr: Lag vs Position')
    ax.axhline(y=3.5, color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=2, color='blue', linestyle='--', alpha=0.3)
    
    # 3. Correlation strength
    ax = axes[1, 0]
    ax.plot(df_autocorr['residue_idx'], df_autocorr['dominant_corr_col'], alpha=0.7)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Correlation Strength')
    ax.set_title('Column Autocorr: Strength Along Sequence')
    ax.grid(alpha=0.3)
    
    # 4. SS enrichment by lag range
    ax = axes[1, 1]
    if 'ss' in df_autocorr.columns and df_autocorr['ss'].notna().any():
        # Group by lag ranges
        df_autocorr['lag_group'] = pd.cut(df_autocorr['dominant_lag_col'], 
                                          bins=[0, 2.5, 4.5, 20],
                                          labels=['Sheet-like (≤2)', 'Helix-like (3-4)', 'Other'])
        
        ct = pd.crosstab(df_autocorr['lag_group'], df_autocorr['ss'], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=ax, color=['gray', 'blue', 'red'])
        ax.set_xlabel('Lag Range')
        ax.set_ylabel('Proportion')
        ax.set_title('SS Distribution by Lag Range')
        ax.legend(title='SS')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No SS data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_id}_autocorr_analysis.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved autocorrelation analysis to {seq_id}_autocorr_analysis.pdf")


def plot_motifs(motifs_2d, enrichment, seq_id, output_dir):
    """
    Visualize discovered attention motifs.
    """
    n_motifs = motifs_2d.shape[0]
    n_cols = 5
    n_rows = (n_motifs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    fig.suptitle(f"{seq_id}: Discovered Attention Motifs", fontsize=16)
    
    axes = axes.flatten() if n_motifs > 1 else [axes]
    
    for i in range(n_motifs):
        ax = axes[i]
        im = ax.imshow(motifs_2d[i], cmap='viridis', aspect='auto')
        
        # Add enrichment info
        if enrichment and i in enrichment:
            info = enrichment[i]
            ss_str = ', '.join([f"{k}:{v:.2f}" for k, v in info['ss_distribution'].items()])
            title = f"Motif {i}\nn={info['n_patches']}, dist={info['mean_distance']:.1f}\n{ss_str}"
        else:
            title = f"Motif {i}"
        
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    for i in range(n_motifs, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_id}_motifs.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved motif analysis to {seq_id}_motifs.pdf")


def plot_periodicity_map(periodicity, strength, seq_id, output_dir, ss_list=None):
    """
    Visualize periodicity map.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{seq_id}: Periodicity Map", fontsize=16)
    
    L = len(periodicity)
    x = np.arange(L)
    
    # 1. Periodicity along sequence
    ax = axes[0]
    scatter = ax.scatter(x, periodicity, c=strength, cmap='viridis', s=20, alpha=0.7)
    ax.axhline(y=3.5, color='red', linestyle='--', label='Helix (3-4)', alpha=0.5)
    ax.axhline(y=2, color='blue', linestyle='--', label='Sheet (2)', alpha=0.5)
    ax.set_ylabel('Dominant Lag')
    ax.set_title('Dominant Periodicity Along Sequence')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Correlation Strength')
    
    # 2. Strength along sequence
    ax = axes[1]
    ax.plot(x, strength, alpha=0.7, linewidth=1)
    ax.fill_between(x, 0, strength, alpha=0.3)
    ax.set_ylabel('Correlation Strength')
    ax.set_title('Periodicity Strength Along Sequence')
    ax.grid(alpha=0.3)
    
    # 3. Comparison with SS (if available)
    ax = axes[2]
    if ss_list:
        # Map SS to numeric
        ss_map = {'C': 0, 'H': 1, 'E': 2}
        ss_numeric = [ss_map.get(s, 0) for s in ss_list]
        
        # Create colored background
        for i in range(L):
            color = {'C': 'gray', 'H': 'red', 'E': 'blue'}.get(ss_list[i], 'gray')
            ax.axvspan(i-0.5, i+0.5, alpha=0.3, color=color)
        
        # Overlay periodicity
        ax2 = ax.twinx()
        ax2.plot(x, periodicity, 'k-', alpha=0.7, linewidth=1.5, label='Dominant Lag')
        ax2.set_ylabel('Dominant Lag', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['C', 'H', 'E'])
        ax.set_ylabel('Secondary Structure')
        ax.set_title('Periodicity vs Secondary Structure')
    else:
        ax.text(0.5, 0.5, 'No SS data available', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Residue Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_id}_periodicity_map.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved periodicity map to {seq_id}_periodicity_map.pdf")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load attention data
    print(f"Loading attention data from {args.attention_json}")
    with open(args.attention_json, 'r') as f:
        attention_data = json.load(f)
    
    # Process each sequence
    all_results = []
    
    for record in attention_data:
        seq_id = record['name']
        print(f"\nProcessing {seq_id}...")
        
        attention_matrix = np.array(record['attention_weights'])
        ss_list = record.get('ss_pred', record.get('q3', None))
        
        L = attention_matrix.shape[0]
        print(f"  Sequence length: {L}")
        
        # Analysis 1: Autocorrelation
        print("  Running autocorrelation analysis...")
        df_autocorr = analyze_1d_autocorr(attention_matrix, max_lag=args.max_lag, ss_list=ss_list)
        plot_autocorr_analysis(df_autocorr, seq_id, args.output_dir)
        
        # Save autocorr data
        df_autocorr.to_csv(os.path.join(args.output_dir, f'{seq_id}_autocorr_data.csv'), index=False)
        
        # Analysis 2: Patch mining
        print("  Running patch mining analysis...")
        motifs_2d, patch_labels, positions, enrichment = analyze_2d_patches(
            attention_matrix, 
            patch_size=args.patch_size,
            n_motifs=args.n_motifs,
            ss_list=ss_list,
            seq_id=seq_id
        )
        plot_motifs(motifs_2d, enrichment, seq_id, args.output_dir)
        
        # Analysis 3: Periodicity map
        print("  Creating periodicity map...")
        periodicity, strength = create_periodicity_map(attention_matrix, max_lag=args.max_lag)
        plot_periodicity_map(periodicity, strength, seq_id, args.output_dir, ss_list=ss_list)
        
        # Aggregate results
        result_summary = {
            'seq_id': seq_id,
            'length': L,
            'mean_periodicity': np.nanmean(periodicity[periodicity > 0]),
            'mean_strength': np.nanmean(strength[strength > 0]),
            'helix_like_fraction': np.sum((periodicity >= 3) & (periodicity <= 4)) / L,
            'sheet_like_fraction': np.sum(periodicity == 2) / L
        }
        all_results.append(result_summary)
    
    # Save summary
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(os.path.join(args.output_dir, 'analysis_summary.csv'), index=False)
    print(f"\n✓ Analysis complete! Results saved to {args.output_dir}")
    print(f"\nSummary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
