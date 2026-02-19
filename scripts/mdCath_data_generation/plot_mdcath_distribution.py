#!/usr/bin/env python3
"""
Plot NEQ distribution for MDCATH dataset across temperatures.
Shows train and test distributions with binned NEQ values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import ast

# Temperatures
TEMPERATURES = ["320K", "348K", "379K", "413K", "450K"]
DATA_DIR = Path("../../data/mdcath/by_temperature")

def parse_neq_lists(neq_col):
    """
    Parse NEQ column from string representation of lists to actual lists.
    Collects ALL per-residue NEQ values across all sequences.
    """
    all_neq_values = []
    for neq_str in neq_col:
        try:
            # Parse string representation of list
            neq_list = ast.literal_eval(neq_str)
            # Collect all residue-level NEQ values
            all_neq_values.extend(neq_list)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse NEQ list: {neq_str[:50]}...")
            continue
    
    return np.array(all_neq_values)

def bin_neq_values(neq_values):
    """
    Bin NEQ values into categories: 1.0, (1,2], (2,3], (3,4], etc.
    neq_values contains all per-residue NEQ values.
    """
    if len(neq_values) == 0:
        raise ValueError("No valid NEQ values found.")
    
    bins = {}
    bins['NEQ = 1.0'] = np.sum(neq_values == 1.0)
    
    max_neq = int(np.ceil(neq_values.max()))
    for i in range(1, max_neq):
        label = f'{i} < NEQ ≤ {i+1}'
        bins[label] = np.sum((neq_values > i) & (neq_values <= i+1))
    
    return bins

def plot_distribution():
    """
    Create bar plots showing NEQ distribution across temperatures.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('MDCATH NEQ Distribution Across Temperatures', fontsize=16, fontweight='bold')
    
    all_stats = []
    
    for idx, temp in enumerate(TEMPERATURES):
        train_file = DATA_DIR / f"train_{temp}.csv"
        test_file = DATA_DIR / f"test_{temp}.csv"
        
        # Read data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Parse NEQ lists and collect all per-residue values
        train_neq = parse_neq_lists(train_df['neq'])
        test_neq = parse_neq_lists(test_df['neq'])
        
        print(f"{temp}: Train={len(train_neq)} residues ({len(train_df)} seqs), Test={len(test_neq)} residues ({len(test_df)} seqs)")
        
        # Bin the values
        train_bins = bin_neq_values(train_neq)
        test_bins = bin_neq_values(test_neq)
        
        # Plot train distribution
        ax_train = axes[0, idx]
        ax_train.bar(range(len(train_bins)), list(train_bins.values()), color='steelblue', alpha=0.7)
        ax_train.set_title(f'{temp} - Train\n({len(train_df)} seqs, {len(train_neq)} residues)', fontsize=10, fontweight='bold')
        ax_train.set_xticks(range(len(train_bins)))
        ax_train.set_xticklabels(train_bins.keys(), rotation=45, ha='right', fontsize=8)
        ax_train.set_ylabel('Count', fontsize=9)
        ax_train.grid(axis='y', alpha=0.3)
        
        # Plot test distribution
        ax_test = axes[1, idx]
        ax_test.bar(range(len(test_bins)), list(test_bins.values()), color='coral', alpha=0.7)
        ax_test.set_title(f'{temp} - Test\n({len(test_df)} seqs, {len(test_neq)} residues)', fontsize=10, fontweight='bold')
        ax_test.set_xticks(range(len(test_bins)))
        ax_test.set_xticklabels(test_bins.keys(), rotation=45, ha='right', fontsize=8)
        ax_test.set_ylabel('Count', fontsize=9)
        ax_test.set_xlabel('NEQ Range', fontsize=9)
        ax_test.grid(axis='y', alpha=0.3)
        
        # Collect statistics
        all_stats.append({
            'temperature': temp,
            'split': 'train',
            'n_sequences': len(train_df),
            'n_residues': len(train_neq),
            'min_neq': train_neq.min(),
            'max_neq': train_neq.max(),
            'mean_neq': train_neq.mean(),
            'median_neq': np.median(train_neq),
            'std_neq': train_neq.std()
        })
        all_stats.append({
            'temperature': temp,
            'split': 'test',
            'n_sequences': len(test_df),
            'n_residues': len(test_neq),
            'min_neq': test_neq.min(),
            'max_neq': test_neq.max(),
            'mean_neq': test_neq.mean(),
            'median_neq': np.median(test_neq),
            'std_neq': test_neq.std()
        })
    
    plt.tight_layout()
    plt.savefig('mdcath_neq_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved mdcath_neq_distribution.png")
    plt.show()
    
    # Print statistics table
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv('mdcath_neq_statistics.csv', index=False)
    print(f"✓ Saved mdcath_neq_statistics.csv\n")
    
    print("="*80)
    print("MDCATH NEQ STATISTICS")
    print("="*80)
    print(stats_df.to_string(index=False))
    print()
    
    # Overall summary
    print("="*80)
    print("OVERALL SUMMARY (Per-Residue NEQ Statistics)")
    print("="*80)
    for temp in TEMPERATURES:
        train_data = stats_df[(stats_df['temperature'] == temp) & (stats_df['split'] == 'train')].iloc[0]
        test_data = stats_df[(stats_df['temperature'] == temp) & (stats_df['split'] == 'test')].iloc[0]
        print(f"{temp}:")
        print(f"  Train: {int(train_data['n_sequences'])} seqs, {int(train_data['n_residues'])} residues | NEQ: {train_data['min_neq']:.2f} - {train_data['max_neq']:.2f} (μ={train_data['mean_neq']:.2f}, σ={train_data['std_neq']:.2f})")
        print(f"  Test:  {int(test_data['n_sequences'])} seqs, {int(test_data['n_residues'])} residues | NEQ: {test_data['min_neq']:.2f} - {test_data['max_neq']:.2f} (μ={test_data['mean_neq']:.2f}, σ={test_data['std_neq']:.2f})")


if __name__ == "__main__":
    plot_distribution()
