#!/usr/bin/env python3
"""
Evaluate and visualize regression model performance.
Creates plots comparing predicted vs actual Neq values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate regression predictions")
    parser.add_argument("--predictions_json", type=str, required=True,
                        help="JSON file with predictions from get_attn.py")
    parser.add_argument("--ground_truth_csv", type=str, required=True,
                        help="CSV file with ground truth Neq values")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory to save plots")
    parser.add_argument("--name_column", type=str, default="name",
                        help="Column name for sequence IDs in CSV")
    parser.add_argument("--neq_column", type=str, default="neq",
                        help="Column name for Neq values in CSV")
    return parser.parse_args()


def load_predictions(json_path):
    """Load predictions from attention JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract per-residue predictions
    seq_predictions = {}
    for record in data:
        seq_id = record.get('name', '')
        sequence = record.get('sequence', '')
        neq_preds = np.array(record['neq_preds'])
        
        # Store per-residue predictions by both ID and sequence (for matching flexibility)
        if seq_id:
            seq_predictions[seq_id] = neq_preds
        if sequence:
            seq_predictions[sequence] = neq_preds
    
    return seq_predictions


def load_ground_truth(csv_path, name_col, neq_col):
    """Load ground truth Neq values from CSV."""
    df = pd.read_csv(csv_path)
    
    # Handle different CSV formats
    if name_col in df.columns:
        # CSV has name/ID column - assume neq_col contains per-residue values
        import ast
        ground_truth = {}
        for idx, row in df.iterrows():
            seq_id = row[name_col]
            neq_values = row[neq_col]
            
            # Parse if it's a string representation of a list
            if isinstance(neq_values, str):
                neq_values = ast.literal_eval(neq_values)
            
            # Store as numpy array
            if isinstance(neq_values, list):
                ground_truth[seq_id] = np.array(neq_values)
            else:
                ground_truth[seq_id] = np.array([float(neq_values)])
                
    elif 'sequence' in df.columns:
        # CSV has sequence column
        import ast
        ground_truth = {}
        for idx, row in df.iterrows():
            seq = row['sequence']
            neq_values = row[neq_col]
            
            # Parse if it's a string representation of a list
            if isinstance(neq_values, str):
                neq_values = ast.literal_eval(neq_values)
            
            # Store as numpy array
            if isinstance(neq_values, list):
                ground_truth[seq] = np.array(neq_values)
            else:
                ground_truth[seq] = np.array([float(neq_values)])
    else:
        raise ValueError(f"CSV must have either '{name_col}' or 'sequence' column")
    
    return ground_truth


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    
    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def plot_predictions_vs_actual(y_true, y_pred, metrics, output_path):
    """Create scatter plot of predictions vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot with regression line
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction', linewidth=2)
    
    # Best fit line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'b-', alpha=0.5, label=f'Best fit (slope={z[0]:.2f})', linewidth=2)
    
    ax.set_xlabel('Actual Neq', fontsize=12)
    ax.set_ylabel('Predicted Neq', fontsize=12)
    ax.set_title(f'Predictions vs Actual\nR²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.3f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual plot
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual Neq', fontsize=12)
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax.set_title(f'Residual Plot\nMAE={metrics["mae"]:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_distributions(y_true, y_pred, output_path):
    """Compare distributions of predicted vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograms
    ax = axes[0]
    ax.hist(y_true, bins=30, alpha=0.5, label='Actual', color='blue', edgecolor='black')
    ax.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='red', edgecolor='black')
    ax.set_xlabel('Neq', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1]
    stats.probplot(y_pred - y_true, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()


def plot_sequence_error_analysis(seq_df, output_path):
    """Analyze errors by sequence."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top errors by sequence
    ax = axes[0]
    top_n = min(20, len(seq_df))
    top_errors = seq_df.head(top_n)
    x = np.arange(top_n)
    width = 0.35
    ax.bar(x - width/2, top_errors['mean_true'], width, label='Actual (mean)', alpha=0.8)
    ax.bar(x + width/2, top_errors['mean_pred'], width, label='Predicted (mean)', alpha=0.8)
    ax.set_xlabel('Sequence', fontsize=12)
    ax.set_ylabel('Mean Neq', fontsize=12)
    ax.set_title(f'Top {top_n} Sequences by Average Error (MAE)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s)[:15] + '...' if len(str(s)) > 15 else str(s) for s in top_errors['seq_id']], 
                       rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error distribution across sequences
    ax = axes[1]
    ax.hist(seq_df['mae'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=seq_df['mae'].mean(), color='r', linestyle='--', linewidth=2, 
               label=f'Mean={seq_df["mae"].mean():.3f}')
    ax.axvline(x=seq_df['mae'].median(), color='g', linestyle='--', linewidth=2, 
               label=f'Median={seq_df["mae"].median():.3f}')
    ax.set_xlabel('Per-Sequence MAE', fontsize=12)
    ax.set_ylabel('Number of Sequences', fontsize=12)
    ax.set_title('Distribution of Per-Sequence MAE', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sequence error analysis to {output_path}")
    plt.close()
    
    # Print worst predictions
    print("\nTop 10 sequences with highest MAE:")
    print(seq_df.head(10)[['seq_id', 'length', 'mean_true', 'mean_pred', 'mae']].to_string(index=False))


def plot_error_analysis(y_true, y_pred, seq_ids, output_path):
    """Analyze errors by sequence."""
    errors = np.abs(y_pred - y_true)
    df = pd.DataFrame({
        'sequence': seq_ids,
        'actual': y_true,
        'predicted': y_pred,
        'error': errors
    })
    df = df.sort_values('error', ascending=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top errors
    ax = axes[0]
    top_n = min(20, len(df))
    top_errors = df.head(top_n)
    x = np.arange(top_n)
    width = 0.35
    ax.bar(x - width/2, top_errors['actual'], width, label='Actual', alpha=0.8)
    ax.bar(x + width/2, top_errors['predicted'], width, label='Predicted', alpha=0.8)
    ax.set_xlabel('Sequence', fontsize=12)
    ax.set_ylabel('Neq', fontsize=12)
    ax.set_title(f'Top {top_n} Sequences by Absolute Error', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:10] for s in top_errors['sequence']], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error histogram
    ax = axes[1]
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={errors.mean():.3f}')
    ax.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median={np.median(errors):.3f}')
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Absolute Errors', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis to {output_path}")
    plt.close()
    
    # Print worst predictions
    print("\nTop 10 sequences with highest errors:")
    print(df.head(10)[['sequence', 'actual', 'predicted', 'error']].to_string(index=False))


def main():
    args = parse_args()
    
    # Load data
    print("Loading predictions...")
    predictions = load_predictions(args.predictions_json)
    
    print("Loading ground truth...")
    ground_truth = load_ground_truth(args.ground_truth_csv, args.name_column, args.neq_column)
    
    # Match sequences
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    print(f"\nFound {len(common_ids)} sequences in both prediction and ground truth")
    
    if len(common_ids) == 0:
        print("ERROR: No common sequences found!")
        print(f"Sample prediction IDs: {list(predictions.keys())[:5]}")
        print(f"Sample ground truth IDs: {list(ground_truth.keys())[:5]}")
        return
    
    # Collect all residues across all sequences
    all_y_true = []
    all_y_pred = []
    seq_ids_list = []
    seq_metrics = []
    
    seq_ids = sorted(common_ids)
    for seq_id in seq_ids:
        y_true_seq = ground_truth[seq_id]
        y_pred_seq = predictions[seq_id]
        
        # Check length match
        if len(y_true_seq) != len(y_pred_seq):
            print(f"Warning: Length mismatch for {seq_id[:50]}... (true={len(y_true_seq)}, pred={len(y_pred_seq)}), skipping")
            continue
        
        # Accumulate all residues
        all_y_true.extend(y_true_seq)
        all_y_pred.extend(y_pred_seq)
        seq_ids_list.extend([seq_id] * len(y_true_seq))
        
        # Calculate per-sequence metrics
        seq_mse = np.mean((y_true_seq - y_pred_seq) ** 2)
        seq_mae = np.mean(np.abs(y_true_seq - y_pred_seq))
        seq_metrics.append({
            'seq_id': seq_id,
            'length': len(y_true_seq),
            'mse': seq_mse,
            'mae': seq_mae,
            'mean_true': y_true_seq.mean(),
            'mean_pred': y_pred_seq.mean()
        })
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    print(f"Total residues evaluated: {len(all_y_true)}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("PER-RESIDUE REGRESSION METRICS")
    print("="*60)
    metrics = calculate_metrics(all_y_true, all_y_pred)
    print(f"MSE:        {metrics['mse']:.4f}")
    print(f"RMSE:       {metrics['rmse']:.4f}")
    print(f"MAE:        {metrics['mae']:.4f}")
    print(f"R²:         {metrics['r2']:.4f}")
    print(f"Pearson r:  {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"Spearman ρ: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    print("="*60)
    
    # Per-sequence summary
    seq_df = pd.DataFrame(seq_metrics)
    print(f"\nPER-SEQUENCE SUMMARY:")
    print(f"Number of sequences: {len(seq_df)}")
    print(f"Average sequence length: {seq_df['length'].mean():.1f}")
    print(f"Average per-sequence MSE: {seq_df['mse'].mean():.4f}")
    print(f"Average per-sequence MAE: {seq_df['mae'].mean():.4f}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_predictions_vs_actual(all_y_true, all_y_pred, metrics, 
                               f"{args.output_dir}/predictions_vs_actual.png")
    plot_distributions(all_y_true, all_y_pred, 
                      f"{args.output_dir}/distributions.png")
    
    # For error analysis, show worst sequences by average error
    seq_df['avg_error'] = seq_df['mae']
    seq_df = seq_df.sort_values('avg_error', ascending=False)
    plot_sequence_error_analysis(seq_df, f"{args.output_dir}/sequence_errors.png")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
