#!/usr/bin/env python3
"""
Create FASTA file from MDCATH test split.
The test split is identical across all temperature experiments.
"""

import pandas as pd
from pathlib import Path
import argparse

def create_test_fasta(test_csv, output_fasta):
    """
    Create FASTA file from test CSV.
    """
    test_csv = Path(test_csv)
    
    if not test_csv.exists():
        print(f"ERROR: {test_csv} not found")
        return None
    
    print(f"Reading test CSV from {test_csv}...")
    df = pd.read_csv(test_csv)
    
    # Check required columns
    if 'sequence' not in df.columns or 'domain' not in df.columns:
        print(f"ERROR: CSV missing required columns")
        print(f"  Found columns: {df.columns.tolist()}")
        return None
    
    print(f"  Found {len(df)} sequences")
    
    # Write FASTA
    print(f"Writing to {output_fasta}...")
    with open(output_fasta, 'w') as f:
        for _, row in df.iterrows():
            protein_id = row['domain']
            sequence = row['sequence']
            f.write(f">{protein_id}\n{sequence}\n")
    
    print(f"✓ Created {output_fasta} with {len(df)} sequences\n")
    return output_fasta


def main():
    parser = argparse.ArgumentParser(description="Create FASTA from MDCATH test split")
    parser.add_argument(
        "--test_csv",
        type=str,
        default="../../data/mdcath/test_split_mmseqs2.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../data/mdcath/test_split.fasta",
        help="Output FASTA file path"
    )
    
    args = parser.parse_args()
    
    test_csv = Path(args.test_csv)
    output_fasta = Path(args.output)
    
    # Create output directory if needed
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    
    result = create_test_fasta(test_csv, output_fasta)
    
    if result is None:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
