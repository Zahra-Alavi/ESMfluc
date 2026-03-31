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
    parser.add_argument(
        "--functional_source",
        type=str,
        default="site_only",
        choices=["site_only", "site_or_location_fallback", "location_only"],
        help=(
            "How to define functional residues from InterPro. "
            "site_only: use siteLocations only (default). "
            "site_or_location_fallback: use siteLocations, but fallback to location ranges if no sites exist. "
            "location_only: use location start/end ranges as functional residues."
        ),
    )
    parser.add_argument(
        "--strict_numbering",
        action="store_true",
        help=(
            "Fail/skip proteins unless numbering checks are fully clean: "
            "sequence mismatch=0, site out-of-range=0, expected-residue mismatch on attention seq=0, "
            "top-hub out-of-range=0"
        ),
    )
    return parser.parse_args()


def parse_interpro_json(interpro_path):
    """
    Parse InterPro JSON and extract functional sites for each sequence.
    
    Returns:
        dict: {seq_id: {'sequence': str, 'sites': {site_type: [residue_indices]},
                        'site_checks': [detailed site records with numbering/residue]}}
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
        all_location_residues = set()
        site_checks = []
        
        matches = entry.get('matches', [])
        for match in matches:
            locations = match.get('locations', [])
            for location in locations:
                loc_start = location.get('start')
                loc_end = location.get('end')
                if loc_start is not None:
                    loc_end = loc_end if loc_end is not None else loc_start
                    all_location_residues.update(range(int(loc_start) - 1, int(loc_end)))

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
                        expected_residue = loc.get('residue', None)
                        if start is not None:
                            # InterPro uses 1-based indexing, convert to 0-based
                            if start == end:
                                all_sites[site_desc].append(start - 1)
                                site_checks.append({
                                    'site_desc': site_desc,
                                    'start_1based': start,
                                    'end_1based': end,
                                    'index_0based': start - 1,
                                    'expected_residue': expected_residue
                                })
                            else:
                                all_sites[site_desc].extend(range(start - 1, end))
                                for idx in range(start - 1, end):
                                    site_checks.append({
                                        'site_desc': site_desc,
                                        'start_1based': start,
                                        'end_1based': end,
                                        'index_0based': idx,
                                        'expected_residue': None
                                    })
        
        # Flatten all sites into a single set of functional residues
        all_functional_residues = set()
        for site_residues in all_sites.values():
            all_functional_residues.update(site_residues)
        
        if all_sites or all_location_residues:  # keep entries even when only broad ranges exist
            sequences_data[seq_id] = {
                'sequence': sequence,
                'sites': dict(all_sites),
                'site_checks': site_checks,
                'all_site_functional': sorted(all_functional_residues),
                'all_location_functional': sorted(all_location_residues),
                'all_functional': sorted(all_functional_residues),
                'n_sites': len(all_sites),
                'n_functional_residues': len(all_functional_residues),
                'n_location_residues': len(all_location_residues),
            }
            print(
                f"  {seq_id}: {len(all_sites)} site types, "
                f"{len(all_functional_residues)} site residues, "
                f"{len(all_location_residues)} location-range residues"
            )
    
    print(f"Parsed {len(sequences_data)} sequences with functional annotations\n")
    return sequences_data


def run_sequence_numbering_checks(seq_id, interpro_seq, attention_seq, attention_matrix,
                                  site_checks, top_hub_indices):
    """
    Validate sequence/index consistency across:
      1) InterPro sequence and attention sequence
      2) InterPro site numbering (1-based) -> sequence residue mapping
      3) High-attention indices -> residue identity consistency
    """
    checks = {
        'seq_len_interpro': len(interpro_seq),
        'seq_len_attention': len(attention_seq),
        'attn_rows': int(attention_matrix.shape[0]),
        'attn_cols': int(attention_matrix.shape[1]),
        'sequence_exact_match': interpro_seq == attention_seq,
        'sequence_mismatch_count': 0,
        'sequence_identity': np.nan,
        'site_total_checks': 0,
        'site_out_of_range_count': 0,
        'site_expected_residue_checks': 0,
        'site_residue_mismatch_interpro_seq': 0,
        'site_residue_mismatch_attention_seq': 0,
        'top_hub_count': int(len(top_hub_indices)),
        'top_hub_out_of_range_count': 0,
        'top_hub_seq_mismatch_count': 0,
    }

    # Global sequence identity
    if len(interpro_seq) == len(attention_seq) and len(interpro_seq) > 0:
        checks['sequence_mismatch_count'] = sum(1 for a, b in zip(interpro_seq, attention_seq) if a != b)
        checks['sequence_identity'] = 1.0 - (checks['sequence_mismatch_count'] / len(interpro_seq))

    # Check InterPro numbering and (when available) residue letter correctness
    for site in site_checks:
        idx = site['index_0based']
        expected = site.get('expected_residue', None)
        checks['site_total_checks'] += 1

        # Out-of-range relative to InterPro sequence
        if idx < 0 or idx >= len(interpro_seq):
            checks['site_out_of_range_count'] += 1
            continue

        # Residue letter checks if expected residue is available
        if expected is not None:
            checks['site_expected_residue_checks'] += 1
            if interpro_seq[idx] != expected:
                checks['site_residue_mismatch_interpro_seq'] += 1

            if 0 <= idx < len(attention_seq) and attention_seq[idx] != expected:
                checks['site_residue_mismatch_attention_seq'] += 1

    # Check top-hub indices against both sequences
    for idx in top_hub_indices:
        if idx < 0 or idx >= len(interpro_seq) or idx >= len(attention_seq):
            checks['top_hub_out_of_range_count'] += 1
            continue
        if interpro_seq[idx] != attention_seq[idx]:
            checks['top_hub_seq_mismatch_count'] += 1

    return checks


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
    
    n_pos = int(y_true.sum())
    n_neg = int(L - n_pos)

    # Degenerate case: ROC is undefined if only one class exists
    if n_pos == 0 or n_neg == 0:
        roc_auc = np.nan
        fpr = np.array([0.0, 1.0])
        tpr = np.array([np.nan, np.nan])
    else:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

    # PR can still be computed, but becomes trivial in degenerate cases
    if n_pos == 0:
        precision = np.array([0.0])
        recall = np.array([0.0])
        avg_precision = np.nan
    elif n_neg == 0:
        precision = np.array([1.0])
        recall = np.array([1.0])
        avg_precision = 1.0
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
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
    
    # Correlations are undefined if binary target has no variance
    if functional_binary.min() == functional_binary.max():
        spearman_r, spearman_p = np.nan, np.nan
        pointbiserial_r, pointbiserial_p = np.nan, np.nan
    else:
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


def build_overall_performance_summary(df_summary):
    """Build one-row aggregate performance summary across all processed proteins."""
    n = len(df_summary)
    if n == 0:
        return pd.DataFrame([])

    out = {
        'n_proteins': int(n),
        'hub_metric': df_summary['hub_metric'].iloc[0] if 'hub_metric' in df_summary.columns else None,
        'top_percent': float(df_summary['top_percent'].iloc[0]) if 'top_percent' in df_summary.columns else np.nan,
        'functional_source_mode': df_summary['functional_source_mode'].iloc[0] if 'functional_source_mode' in df_summary.columns else None,
        'median_fold_enrichment': float(df_summary['fold_enrichment'].median()),
        'mean_fold_enrichment': float(df_summary['fold_enrichment'].mean()),
        'median_roc_auc': float(df_summary['roc_auc'].median()),
        'mean_roc_auc': float(df_summary['roc_auc'].mean()),
        'median_avg_precision': float(df_summary['avg_precision'].median()),
        'mean_avg_precision': float(df_summary['avg_precision'].mean()),
        'median_spearman_r': float(df_summary['spearman_r'].median()),
        'mean_spearman_r': float(df_summary['spearman_r'].mean()),
        'n_fisher_p_lt_0_05': int((df_summary['fisher_p'] < 0.05).sum()),
        'frac_fisher_p_lt_0_05': float((df_summary['fisher_p'] < 0.05).mean()),
        'n_spearman_p_lt_0_05': int((df_summary['spearman_p'] < 0.05).sum()),
        'frac_spearman_p_lt_0_05': float((df_summary['spearman_p'] < 0.05).mean()),
        'n_spearman_positive': int((df_summary['spearman_r'] > 0).sum()),
        'frac_spearman_positive': float((df_summary['spearman_r'] > 0).mean()),
    }

    if 'numbering_ok' in df_summary.columns:
        out['n_numbering_ok'] = int(df_summary['numbering_ok'].sum())
        out['frac_numbering_ok'] = float(df_summary['numbering_ok'].mean())

    if 'functional_source_used' in df_summary.columns:
        counts = df_summary['functional_source_used'].value_counts(dropna=False).to_dict()
        out['n_source_site'] = int(counts.get('site', 0))
        out['n_source_location_fallback'] = int(counts.get('location_fallback', 0))
        out['n_source_location_only'] = int(counts.get('location_only', 0))

    return pd.DataFrame([out])


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
    
    if len(non_func_scores) > 0:
        ax2.hist(non_func_scores, bins=30, alpha=0.6, label='Non-functional', density=True, color='gray')
    if len(func_scores) > 0:
        ax2.hist(func_scores, bins=30, alpha=0.6, label='Functional', density=True, color='red')
    ax2.set_xlabel('Hub Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Hub Score Distribution')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[1, 1])
    roc_auc_val = roc_metrics['roc_auc']
    roc_auc_txt = f"{roc_auc_val:.3f}" if np.isfinite(roc_auc_val) else "NA"
    ax3.plot(roc_metrics['fpr'], roc_metrics['tpr'], linewidth=2, label=f'ROC (AUC = {roc_auc_txt})')
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
    sp_r = correlation['spearman_r']
    sp_p = correlation['spearman_p']
    pb_r = correlation['pointbiserial_r']
    pb_p = correlation['pointbiserial_p']
    roc_auc_disp = roc_metrics['roc_auc']
    ap_disp = roc_metrics['avg_precision']

    corr_text = f"""Correlation Analysis:
    
Spearman r: {sp_r:.3f} {'' if np.isfinite(sp_r) else '(undefined)'}
Spearman p: {sp_p:.2e} {'' if np.isfinite(sp_p) else '(undefined)'}

Point-biserial r: {pb_r:.3f} {'' if np.isfinite(pb_r) else '(undefined)'}
Point-biserial p: {pb_p:.2e} {'' if np.isfinite(pb_p) else '(undefined)'}

ROC AUC: {roc_auc_disp:.3f} {'' if np.isfinite(roc_auc_disp) else '(undefined)'}
Avg Precision: {ap_disp:.3f} {'' if np.isfinite(ap_disp) else '(undefined)'}
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
        site_functional = func_data.get('all_site_functional', func_data.get('all_functional', []))
        location_functional = func_data.get('all_location_functional', [])

        if args.functional_source == 'site_only':
            functional_residues = site_functional
            functional_source_used = 'site_only'
        elif args.functional_source == 'location_only':
            functional_residues = location_functional
            functional_source_used = 'location_only'
        else:
            if len(site_functional) > 0:
                functional_residues = site_functional
                functional_source_used = 'site'
            else:
                functional_residues = location_functional
                functional_source_used = 'location_fallback'

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
        print(
            f"  Functional residues: {len(functional_residues)} "
            f"({100*len(functional_residues)/L:.1f}%) | source={functional_source_used}"
        )
        
        # Compute hub scores
        hub_scores = compute_attention_hubs(attention_matrix, metric=args.hub_metric)

        # High-attention indices (same rule as enrichment)
        hub_threshold = np.percentile(hub_scores, 100 - args.top_percent)
        top_hub_indices = np.where(hub_scores >= hub_threshold)[0]

        # Sequence/index reliability checks
        seq_checks = run_sequence_numbering_checks(
            seq_id=seq_id,
            interpro_seq=interpro_seq,
            attention_seq=attention_seq,
            attention_matrix=attention_matrix,
            site_checks=func_data.get('site_checks', []),
            top_hub_indices=top_hub_indices,
        )

        print("  Sequence/numbering checks:")
        print(f"    seq identity: {seq_checks['sequence_identity']:.4f}" if not np.isnan(seq_checks['sequence_identity'])
              else "    seq identity: N/A (length mismatch)")
        print(f"    site checks: {seq_checks['site_total_checks']} | out-of-range: {seq_checks['site_out_of_range_count']}")
        print(f"    expected-residue checks: {seq_checks['site_expected_residue_checks']} | "
              f"mismatch (InterPro seq): {seq_checks['site_residue_mismatch_interpro_seq']} | "
              f"mismatch (Attention seq): {seq_checks['site_residue_mismatch_attention_seq']}")
        print(f"    top hubs: {seq_checks['top_hub_count']} | out-of-range: {seq_checks['top_hub_out_of_range_count']} | "
              f"InterPro-vs-Attention AA mismatch at top hubs: {seq_checks['top_hub_seq_mismatch_count']}")

        numbering_ok = (
            seq_checks['sequence_mismatch_count'] == 0
            and seq_checks['site_out_of_range_count'] == 0
            and seq_checks['site_residue_mismatch_attention_seq'] == 0
            and seq_checks['top_hub_out_of_range_count'] == 0
        )
        print(f"    numbering_ok: {numbering_ok}")

        if args.strict_numbering and not numbering_ok:
            print("  ERROR: strict numbering check failed - skipping this protein")
            continue
        
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
            'functional_source_mode': args.functional_source,
            'functional_source_used': functional_source_used,
            'fold_enrichment': enrichment['fold_enrichment'],
            'fisher_p': enrichment['p_value'],
            'roc_auc': roc_metrics['roc_auc'],
            'avg_precision': roc_metrics['avg_precision'],
            'spearman_r': correlation['spearman_r'],
            'spearman_p': correlation['spearman_p'],
            'hub_metric': args.hub_metric,
            'top_percent': args.top_percent,
            'seq_identity': seq_checks['sequence_identity'],
            'sequence_mismatch_count': seq_checks['sequence_mismatch_count'],
            'site_total_checks': seq_checks['site_total_checks'],
            'site_out_of_range_count': seq_checks['site_out_of_range_count'],
            'site_expected_residue_checks': seq_checks['site_expected_residue_checks'],
            'site_residue_mismatch_interpro_seq': seq_checks['site_residue_mismatch_interpro_seq'],
            'site_residue_mismatch_attention_seq': seq_checks['site_residue_mismatch_attention_seq'],
            'top_hub_count': seq_checks['top_hub_count'],
            'top_hub_out_of_range_count': seq_checks['top_hub_out_of_range_count'],
            'top_hub_seq_mismatch_count': seq_checks['top_hub_seq_mismatch_count'],
            'numbering_ok': numbering_ok,
        }
        all_results.append(result)
    
    # Save summary
    if all_results:
        df_summary = pd.DataFrame(all_results)
        out_csv = os.path.join(args.output_dir, 'hub_function_analysis_summary.csv')
        df_summary.to_csv(out_csv, index=False)

        overall_csv = os.path.join(args.output_dir, 'hub_function_overall_performance.csv')
        df_overall = build_overall_performance_summary(df_summary)
        df_overall.to_csv(overall_csv, index=False)
        
        print(f"\n✓ Analysis complete! Results saved to {args.output_dir}")
        print(f"\nSummary Statistics:")
        print(f"  Median fold enrichment: {df_summary['fold_enrichment'].median():.2f}x")
        print(f"  Median ROC AUC: {df_summary['roc_auc'].median():.3f}")
        print(f"  Sequences with p<0.05: {(df_summary['fisher_p'] < 0.05).sum()}/{len(df_summary)}")
        if 'numbering_ok' in df_summary.columns:
            print(f"  Sequences with numbering_ok=True: {df_summary['numbering_ok'].sum()}/{len(df_summary)}")
        print(f"\nFull summary saved to: {out_csv}")
        print(f"Overall performance summary saved to: {overall_csv}")
    else:
        print("\nNo sequences could be processed!")


if __name__ == "__main__":
    main()
