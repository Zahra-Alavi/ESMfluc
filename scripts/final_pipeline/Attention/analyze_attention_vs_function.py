#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze correlation between attention hubs and functional residues from InterPro.

This script:
1. Parses InterPro JSON to extract functional sites for each sequence
2. Identifies attention hubs using multiple metrics
3. Performs statistical tests (enrichment, ROC-AUC, correlation)
4. Generates visualizations

Usage:
    python analyze_attention_vs_function.py \
        --attention_json attention_data.json \
        --interpro_json interpro_results.json \
        --output_dir ./hub_analysis
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from collections import defaultdict
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze attention hubs vs functional residues"
    )
    parser.add_argument(
        "--attention_json",
        type=str,
        required=True,
        help="Path to JSON file with attention data"
    )
    parser.add_argument(
        "--interpro_json",
        type=str,
        required=True,
        help="Path to InterPro JSON file with functional annotations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--hub_metric",
        type=str,
        default="max_received",
        choices=["max_received", "avg_received", "sum_received", "max_given", "avg_given"],
        help="Metric to define attention hubs (default: max_received)"
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=10.0,
        help="Top percentage of residues to consider as hubs (default: 10)"
    )
    return parser.parse_args()


def parse_interpro_json(interpro_path):
    """
    Parse InterPro JSON and extract functional sites for each sequence.
    
    Returns:
        dict: {seq_id: {'sequence': str, 'sites': {site_type: [residue_indices]}}}
    """
    print(f"Parsing InterPro JSON from {interpro_path}")
    
    with open(interpro_path, 'r') as f:
        interpro_data = json.load(f)
    
    sequences_data = {}
    
    # InterPro JSON has 'results' key at top level
    results = interpro_data.get('results', [])
    
    for entry in results:
        # Extract sequence ID from xref
        xrefs = entry.get('xref', [])
        seq_id = xrefs[0].get('id') if xrefs else entry.get('md5', 'unknown')
        
        # Extract sequence
        sequence = entry.get('sequence', '')
        
        # Extract all sites from matches -> locations -> sites
        all_sites = defaultdict(list)
        
        matches = entry.get('matches', [])
        for match in matches:
            locations = match.get('locations', [])
            for location in locations:
                sites_list = location.get('sites', [])
                if sites_list is None:
                    continue
                for site in sites_list:
                    site_desc = site.get('description', 'unknown_site')
                    site_locations = site.get('siteLocations', [])
                    
                    # Extract residue indices (convert to 0-based)
                    for loc in site_locations:
                        start = loc.get('start')
                        end = loc.get('end')
                        if start is not None:
                            # InterPro uses 1-based indexing, convert to 0-based
                            if start == end:
                                all_sites[site_desc].append(start - 1)
                            else:
                                all_sites[site_desc].extend(range(start - 1, end))
        
        # Flatten all sites into a single set of functional residues
        all_functional_residues = set()
        for site_residues in all_sites.values():
            all_functional_residues.update(site_residues)
        
        if all_sites:  # Only add if we found sites
            sequences_data[seq_id] = {
                'sequence': sequence,
                'sites': dict(all_sites),
                'all_functional': sorted(all_functional_residues),
                'n_sites': len(all_sites),
                'n_functional_residues': len(all_functional_residues)
            }
            print(f"  {seq_id}: {len(all_sites)} site types, {len(all_functional_residues)} functional residues")
    
    print(f"Parsed {len(sequences_data)} sequences with functional annotations\n")
    return sequences_data


def compute_attention_hubs(attention_matrix, metric="max_received"):
    """
    Compute attention hub scores for each residue.
    
    Args:
        attention_matrix: LxL numpy array
        metric: how to define hub score
    
    Returns:
        hub_scores: array of length L with hub scores
    """
    L = attention_matrix.shape[0]
    
    if metric == "max_received":
        # Maximum attention received by each residue
        hub_scores = attention_matrix.max(axis=0)
    elif metric == "avg_received":
        # Average attention received (excluding self)
        hub_scores = np.array([(attention_matrix[:, i].sum() - attention_matrix[i, i]) / (L - 1) 
                               for i in range(L)])
    elif metric == "sum_received":
        # Total attention received
        hub_scores = attention_matrix.sum(axis=0)
    elif metric == "max_given":
        # Maximum attention given by each residue
        hub_scores = attention_matrix.max(axis=1)
    elif metric == "avg_given":
        # Average attention given (excluding self)
        hub_scores = np.array([(attention_matrix[i, :].sum() - attention_matrix[i, i]) / (L - 1) 
                               for i in range(L)])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return hub_scores


def compute_enrichment(hub_scores, functional_residues, top_percent=10):
    """
    Test if top hub residues are enriched for functional residues.
    
    Returns:
        dict with enrichment statistics
    """
    L = len(hub_scores)
    n_functional = len(functional_residues)
    
    # Define top hubs
    threshold = np.percentile(hub_scores, 100 - top_percent)
    top_hubs = np.where(hub_scores >= threshold)[0]
    n_hubs = len(top_hubs)
    
    # Count overlap
    functional_set = set(functional_residues)
    hub_set = set(top_hubs)
    overlap = len(functional_set & hub_set)
    
    # Fisher's exact test
    # Contingency table:
    #               Functional   Not-Functional
    # Top Hub          a              b
    # Not Hub          c              d
    
    a = overlap
    b = n_hubs - overlap
    c = n_functional - overlap
    d = L - n_hubs - c
    
    contingency = np.array([[a, b], [c, d]])
    odds_ratio, p_value = stats.fisher_exact(contingency, alternative='greater')
    
    # Enrichment fold
    expected = (n_hubs * n_functional) / L
    fold_enrichment = overlap / expected if expected > 0 else 0
    
    return {
        'n_total': L,
        'n_hubs': n_hubs,
        'n_functional': n_functional,
        'overlap': overlap,
        'expected': expected,
        'fold_enrichment': fold_enrichment,
        'odds_ratio': odds_ratio,
        'p_value': p_value,
        'top_percent': top_percent
    }


def compute_roc_metrics(hub_scores, functional_residues, L):
    """
    Compute ROC and PR curves for hub scores predicting functional residues.
    
    Returns:
        dict with ROC/PR metrics
    """
    # Create binary labels
    y_true = np.zeros(L)
    y_true[functional_residues] = 1
    
    # Use hub scores as predictions
    y_scores = hub_scores
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }


def compute_correlation(hub_scores, functional_residues, L):
    """
    Compute correlation between hub scores and functional residue locations.
    """
    # Create binary vector for functional residues
    functional_binary = np.zeros(L)
    functional_binary[functional_residues] = 1
    
    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(hub_scores, functional_binary)
    
    # Point-biserial correlation (for binary variable)
    pointbiserial_r, pointbiserial_p = stats.pointbiserialr(functional_binary, hub_scores)
    
    return {
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pointbiserial_r': pointbiserial_r,
        'pointbiserial_p': pointbiserial_p
    }


def plot_analysis(seq_id, hub_scores, functional_residues, enrichment, roc_metrics, 
                 correlation, output_dir, ss_list=None):
    """
    Generate comprehensive visualization of hub vs function analysis.
    """
    L = len(hub_scores)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Hub scores along sequence
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(L)
    ax1.plot(x, hub_scores, alpha=0.7, linewidth=1, color='steelblue')
    
    # Mark functional residues
    for func_idx in functional_residues:
        ax1.axvline(x=func_idx, color='red', alpha=0.3, linewidth=0.5)
    
    # Mark top hubs
    threshold = np.percentile(hub_scores, 100 - enrichment['top_percent'])
    ax1.axhline(y=threshold, color='orange', linestyle='--', label=f'Top {enrichment["top_percent"]:.0f}% threshold')
    
    ax1.set_xlabel('Residue Index')
    ax1.set_ylabel('Hub Score')
    ax1.set_title(f'{seq_id}: Attention Hub Scores vs Functional Residues (red lines)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Distribution comparison
    ax2 = fig.add_subplot(gs[1, 0])
    functional_set = set(functional_residues)
    func_scores = [hub_scores[i] for i in functional_residues]
    non_func_scores = [hub_scores[i] for i in range(L) if i not in functional_set]
    
    ax2.hist(non_func_scores, bins=30, alpha=0.6, label='Non-functional', density=True, color='gray')
    ax2.hist(func_scores, bins=30, alpha=0.6, label='Functional', density=True, color='red')
    ax2.set_xlabel('Hub Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Hub Score Distribution')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(roc_metrics['fpr'], roc_metrics['tpr'], linewidth=2, label=f'ROC (AUC = {roc_metrics["roc_auc"]:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Precision-Recall Curve
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(roc_metrics['recall'], roc_metrics['precision'], linewidth=2, 
             label=f'PR (AP = {roc_metrics["avg_precision"]:.3f})')
    baseline = enrichment['n_functional'] / L
    ax4.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Enrichment stats
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    stats_text = f"""Enrichment Analysis:
    
Total residues: {enrichment['n_total']}
Functional residues: {enrichment['n_functional']} ({100*enrichment['n_functional']/L:.1f}%)
Top {enrichment['top_percent']:.0f}% hubs: {enrichment['n_hubs']}
Overlap: {enrichment['overlap']}
Expected: {enrichment['expected']:.1f}
Fold enrichment: {enrichment['fold_enrichment']:.2f}
Odds ratio: {enrichment['odds_ratio']:.2f}
Fisher's p-value: {enrichment['p_value']:.2e}
    """
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    # 6. Correlation stats
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    corr_text = f"""Correlation Analysis:
    
Spearman r: {correlation['spearman_r']:.3f}
Spearman p: {correlation['spearman_p']:.2e}

Point-biserial r: {correlation['pointbiserial_r']:.3f}
Point-biserial p: {correlation['pointbiserial_p']:.2e}

ROC AUC: {roc_metrics['roc_auc']:.3f}
Avg Precision: {roc_metrics['avg_precision']:.3f}
    """
    ax6.text(0.1, 0.9, corr_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # 7. Scatter: hub score vs position
    ax7 = fig.add_subplot(gs[2, 2])
    functional_binary = np.zeros(L)
    functional_binary[functional_residues] = 1
    
    colors = ['gray' if fb == 0 else 'red' for fb in functional_binary]
    ax7.scatter(x, hub_scores, c=colors, alpha=0.6, s=20)
    ax7.set_xlabel('Residue Index')
    ax7.set_ylabel('Hub Score')
    ax7.set_title('Hub Scores (red = functional)')
    ax7.grid(alpha=0.3)
    
    plt.suptitle(f'{seq_id}: Attention Hubs vs Functional Residues', fontsize=14, y=0.995)
    
    out_file = os.path.join(output_dir, f'{seq_id}_hub_analysis.pdf')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved analysis plot to {seq_id}_hub_analysis.pdf")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load InterPro data
    interpro_data = parse_interpro_json(args.interpro_json)
    
    # Load attention data
    print(f"Loading attention data from {args.attention_json}")
    with open(args.attention_json, 'r') as f:
        attention_data = json.load(f)
    
    # Create mapping
    attn_map = {record['name']: record for record in attention_data}
    
    # Process each sequence
    all_results = []
    
    for seq_id, func_data in interpro_data.items():
        if seq_id not in attn_map:
            print(f"Warning: {seq_id} not found in attention data, skipping...")
            continue
        
        print(f"\nProcessing {seq_id}...")
        
        record = attn_map[seq_id]
        attention_matrix = np.array(record['attention_weights'])
        L = attention_matrix.shape[0]
        functional_residues = func_data['all_functional']
        ss_list = record.get('ss_pred', record.get('q3', None))
        
        # CRITICAL: Verify sequences match between InterPro and attention data
        interpro_seq = func_data['sequence']
        attention_seq = record['sequence']
        
        if interpro_seq != attention_seq:
            print(f"  WARNING: Sequence mismatch!")
            print(f"    InterPro length: {len(interpro_seq)}")
            print(f"    Attention length: {len(attention_seq)}")
            if len(interpro_seq) != len(attention_seq):
                print(f"  ERROR: Length mismatch - skipping {seq_id}")
                continue
            else:
                # Same length but different residues - check differences
                mismatches = sum(1 for a, b in zip(interpro_seq, attention_seq) if a != b)
                print(f"    {mismatches} residue mismatches found")
                if mismatches > 0.05 * len(interpro_seq):  # More than 5% different
                    print(f"  ERROR: Too many mismatches ({mismatches}/{len(interpro_seq)}) - skipping {seq_id}")
                    continue
                else:
                    print(f"  WARNING: Proceeding with minor mismatches")
        
        # Verify attention matrix dimensions match sequence length
        if L != len(attention_seq):
            print(f"  ERROR: Attention matrix size ({L}) != sequence length ({len(attention_seq)}) - skipping {seq_id}")
            continue
        
        # Filter functional residues to valid indices
        invalid_residues = [i for i in functional_residues if i < 0 or i >= L]
        if invalid_residues:
            print(f"  WARNING: {len(invalid_residues)} functional residues out of range: {invalid_residues[:10]}")
        functional_residues = [i for i in functional_residues if 0 <= i < L]
        
        if len(functional_residues) == 0:
            print(f"  No valid functional residues, skipping...")
            continue
        
        print(f"  Sequence length: {L} (InterPro: {len(interpro_seq)}, Attention: {len(attention_seq)})")
        print(f"  Functional residues: {len(functional_residues)} ({100*len(functional_residues)/L:.1f}%)")
        
        # Compute hub scores
        hub_scores = compute_attention_hubs(attention_matrix, metric=args.hub_metric)
        
        # Statistical analyses
        enrichment = compute_enrichment(hub_scores, functional_residues, top_percent=args.top_percent)
        roc_metrics = compute_roc_metrics(hub_scores, functional_residues, L)
        correlation = compute_correlation(hub_scores, functional_residues, L)
        
        # Print results
        print(f"  Enrichment: {enrichment['fold_enrichment']:.2f}x (p={enrichment['p_value']:.2e})")
        print(f"  ROC AUC: {roc_metrics['roc_auc']:.3f}")
        print(f"  Spearman r: {correlation['spearman_r']:.3f} (p={correlation['spearman_p']:.2e})")
        
        # Plot
        plot_analysis(seq_id, hub_scores, functional_residues, enrichment, roc_metrics,
                     correlation, args.output_dir, ss_list=ss_list)
        
        # Store results
        result = {
            'seq_id': seq_id,
            'length': L,
            'n_functional': len(functional_residues),
            'n_sites': func_data['n_sites'],
            'fold_enrichment': enrichment['fold_enrichment'],
            'fisher_p': enrichment['p_value'],
            'roc_auc': roc_metrics['roc_auc'],
            'avg_precision': roc_metrics['avg_precision'],
            'spearman_r': correlation['spearman_r'],
            'spearman_p': correlation['spearman_p'],
            'hub_metric': args.hub_metric,
            'top_percent': args.top_percent
        }
        all_results.append(result)
    
    # Save summary
    if all_results:
        df_summary = pd.DataFrame(all_results)
        out_csv = os.path.join(args.output_dir, 'hub_function_analysis_summary.csv')
        df_summary.to_csv(out_csv, index=False)
        
        print(f"\n✓ Analysis complete! Results saved to {args.output_dir}")
        print(f"\nSummary Statistics:")
        print(f"  Median fold enrichment: {df_summary['fold_enrichment'].median():.2f}x")
        print(f"  Median ROC AUC: {df_summary['roc_auc'].median():.3f}")
        print(f"  Sequences with p<0.05: {(df_summary['fisher_p'] < 0.05).sum()}/{len(df_summary)}")
        print(f"\nFull summary saved to: {out_csv}")
    else:
        print("\nNo sequences could be processed!")


if __name__ == "__main__":
    main()
