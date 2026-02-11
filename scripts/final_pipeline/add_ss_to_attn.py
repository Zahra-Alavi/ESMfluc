#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add secondary structure predictions from NetSurfP CSV to existing attention JSON.

Usage:
    python add_ss_to_attn.py --attention_json input.json --netsurf_csv netsurf_results.csv --output output.json

Adds the following fields from NetSurfP to each sequence:
    - ss_pred (q3): 3-state secondary structure (C/H/E)
    - q8: 8-state secondary structure
    - rsa: Relative solvent accessibility
    - asa: Absolute solvent accessibility
    - disorder: Disorder prediction
"""

import argparse
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add SS predictions from NetSurfP CSV to attention JSON."
    )
    parser.add_argument(
        "--attention_json",
        type=str,
        required=True,
        help="Path to the attention JSON file (without SS predictions)."
    )
    parser.add_argument(
        "--netsurf_csv",
        type=str,
        required=True,
        help="Path to the NetSurfP CSV output file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the updated JSON file (with SS predictions)."
    )
    return parser.parse_args()


def parse_netsurf_csv(csv_path):
    """
    Parse NetSurfP CSV and return a dict mapping seq_id to structural predictions.
    
    NetSurfP format:
        id, seq, n, rsa, asa, q3, p[q3_H], p[q3_E], p[q3_C], q8, p[q8_G], p[q8_H], 
        p[q8_I], p[q8_B], p[q8_E], p[q8_S], p[q8_T], p[q8_C], phi, psi, disorder
    
    Returns:
        dict: {seq_id: {
            'rsa': [val1, val2, ...],
            'asa': [val1, val2, ...],
            'q3': [label1, label2, ...],
            'q8': [label1, label2, ...],
            'disorder': [val1, val2, ...]
        }}
    """
    df = pd.read_csv(csv_path, sep=',', skipinitialspace=True)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    predictions_map = {}
    
    for _, row in df.iterrows():
        raw_id = row['id']
        rsa = row['rsa']
        asa = row['asa']
        q3_label = row['q3']
        q8_label = row['q8']
        disorder = row['disorder']
        
        # Remove leading '>' if present
        seq_id = str(raw_id).lstrip(">")
        
        if seq_id not in predictions_map:
            predictions_map[seq_id] = {
                'rsa': [],
                'asa': [],
                'q3': [],
                'q8': [],
                'disorder': []
            }
        
        predictions_map[seq_id]['rsa'].append(float(rsa))
        predictions_map[seq_id]['asa'].append(float(asa))
        predictions_map[seq_id]['q3'].append(str(q3_label).strip())
        predictions_map[seq_id]['q8'].append(str(q8_label).strip())
        predictions_map[seq_id]['disorder'].append(float(disorder))
    
    return predictions_map


def main():
    args = parse_args()
    
    # Load attention JSON
    print(f"Loading attention JSON from {args.attention_json}")
    with open(args.attention_json, 'r') as f:
        attention_data = json.load(f)
    
    # Parse NetSurfP CSV
    print(f"Parsing NetSurfP CSV from {args.netsurf_csv}")
    predictions_map = parse_netsurf_csv(args.netsurf_csv)
    print(f"Found predictions for {len(predictions_map)} sequences")
    
    # Add predictions to attention data
    added_count = 0
    missing_count = 0
    mismatch_count = 0
    
    for record in attention_data:
        seq_id = record['name']
        seq_len = len(record['sequence'])
        
        if seq_id in predictions_map:
            preds = predictions_map[seq_id]
            
            # Check length match (using q3 as reference)
            if len(preds['q3']) == seq_len:
                record['ss_pred'] = preds['q3']
                record['rsa'] = preds['rsa']
                record['asa'] = preds['asa']
                record['q8'] = preds['q8']
                record['disorder'] = preds['disorder']
                added_count += 1
                print(f"✓ Added predictions for {seq_id} ({len(preds['q3'])} residues)")
            else:
                mismatch_count += 1
                print(f"✗ Length mismatch for {seq_id}: sequence={seq_len}, predictions={len(preds['q3'])}")
        else:
            missing_count += 1
            print(f"⚠ No predictions found for {seq_id}")
    
    # Save updated JSON
    print(f"\nSaving updated JSON to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(attention_data, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Predictions added: {added_count}")
    print(f"  Missing predictions: {missing_count}")
    print(f"  Length mismatches: {mismatch_count}")
    print(f"  Total sequences: {len(attention_data)}")
    print(f"\nAdded fields: ss_pred (q3), q8, rsa, asa, disorder")
    

if __name__ == "__main__":
    main()
