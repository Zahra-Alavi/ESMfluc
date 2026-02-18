#!/usr/bin/env python3
"""
Split MDCATH dataset by temperature for separate model training.
Creates train/test CSV files for each temperature.
"""

import pandas as pd
import ast
from pathlib import Path

def split_by_temperature(input_train, input_test, output_dir):
    """
    Split MDCATH data by temperature.
    
    Args:
        input_train: Path to train_split_mmseqs2.csv
        input_test: Path to test_split_mmseqs2.csv
        output_dir: Directory to save temperature-specific CSVs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading {input_train}...")
    df_train = pd.read_csv(input_train)
    print(f"Loading {input_test}...")
    df_test = pd.read_csv(input_test)
    
    # Ensure neq columns are lists
    for col in df_train.columns:
        if col.startswith('neq_'):
            if df_train[col].dtype == object:
                df_train[col] = df_train[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            if df_test[col].dtype == object:
                df_test[col] = df_test[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
    
    temperatures = ['320', '348', '379', '413', '450']
    
    print(f"\nSplitting data by temperature...")
    for temp in temperatures:
        neq_col = f'neq_{temp}'
        
        if neq_col not in df_train.columns:
            print(f"Warning: {neq_col} not found in data")
            continue
        
        # Create train file
        train_temp = df_train[['domain', 'sequence', neq_col]].copy()
        train_temp.rename(columns={neq_col: 'neq'}, inplace=True)
        train_output = output_dir / f'train_{temp}K.csv'
        train_temp.to_csv(train_output, index=False)
        print(f"  ✓ Saved {len(train_temp)} train sequences to {train_output}")
        
        # Create test file
        test_temp = df_test[['domain', 'sequence', neq_col]].copy()
        test_temp.rename(columns={neq_col: 'neq'}, inplace=True)
        test_output = output_dir / f'test_{temp}K.csv'
        test_temp.to_csv(test_output, index=False)
        print(f"  ✓ Saved {len(test_temp)} test sequences to {test_output}")
    
    print(f"\n✓ Temperature-split datasets saved to {output_dir}/")
    print(f"\nTo train on a specific temperature, use:")
    print(f"  python main.py --task_type regression \\")
    print(f"    --train_data_file {output_dir}/train_348K.csv \\")
    print(f"    --test_data_file {output_dir}/test_348K.csv \\")
    print(f"    --architecture bilstm_attention \\")
    print(f"    --esm_model esm2_t33_650M_UR50D \\")
    print(f"    --epochs 50 --patience 5")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split MDCATH by temperature")
    parser.add_argument(
        "--train_file",
        type=str,
        default="../../data/mdcath/train_split_mmseqs2.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../../data/mdcath/test_split_mmseqs2.csv",
        help="Path to test data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/mdcath/by_temperature",
        help="Output directory for temperature-split files"
    )
    
    args = parser.parse_args()
    
    split_by_temperature(args.train_file, args.test_file, args.output_dir)
