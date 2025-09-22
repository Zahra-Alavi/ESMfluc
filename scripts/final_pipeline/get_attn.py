#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:03:11 2025

Description: This script takes a fasta file and the best model check point.
It returns a JSON file with sequences in the fasta, their attention weights and neq preds,
using best model check point. 
If a CSV file for secondary structure prediction is given, the final JSON will also include ss_pred. 
Such CSV file can be obtained from: https://services.healthtech.dtu.dk/services/NetSurfP-3.0/ 

@author: zalavi
"""

import pandas as pd
import argparse
import torch
from transformers import EsmModel, EsmTokenizer


from models import BiLSTMWithSelfAttentionModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention weights for sequences in a FASTA file."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (e.g., best_model.pth).")
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="Path to the FASTA file containing sequences.")
    parser.add_argument("--ss_csv", type=str, required=False,
                        help="(Optional) Path to the CSV file containing ss predictions (NetSurfP output).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output JSON file.")
    return parser.parse_args()


def parse_fasta_file(fasta_path):
    """
    Yields tuples of (sequence_id, sequence_string) from a FASTA file.
    """
    with open(fasta_path, 'r') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_lines)
                seq_id = line[1:]  # everything after ">"
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None and seq_lines:
            yield seq_id, "".join(seq_lines)


            
def run_model(model, tokenizer, sequence, device):     
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attn_weights_torch = model(input_ids, attn_mask, return_attention=True)
      
    attn_weights = attn_weights_torch[0].cpu().numpy()  # shape=(L,L)
    
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # length=L
    
    y_probs = torch.softmax(logits, dim=-1)
    y_preds = torch.argmax(y_probs, dim=-1)
    y_preds = y_preds.view(-1)
    
    return attn_weights, tokens, y_preds

def parse_nsp3_csv(df):
    """
    Returns a dict mapping { seq_id -> list_of_SS }, 
    where seq_id is the FASTA header without '>',
    and list_of_SS is a list of 'C','H','E' for each residue.
    """

    ss_map = {}

    for row in df.itertuples(index=False):
        raw_id = row[0]    # first column is 'id'
        q3_label = row[5]  # fourth column is 'q3'
        seq_id = raw_id.lstrip(">")
        if seq_id not in ss_map:
            ss_map[seq_id] = []
        ss_map[seq_id].append(q3_label)

    return ss_map


def main():
    
    args = parse_args()

    embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    embedding_model.to(device)
        

    model = BiLSTMWithSelfAttentionModel(
        embedding_model=embedding_model,
        hidden_size=512,
        num_layers=3,
        num_classes=2,
        dropout=0.3
    )
    model.to(device)


    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)


    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")


    ss_map = {}
    ss_available = args.ss_csv is not None
    if ss_available:
        nsp3_df = pd.read_csv(args.ss_csv, index_col=False)
        ss_map = parse_nsp3_csv(nsp3_df)
        print(f"Parsed NetSurfP CSV: found SS for {len(ss_map)} sequences")
    
    rows = []
    for seq_id, seq_str in parse_fasta_file(args.fasta_file):
        ss_list = ss_map.get(seq_id, None) if ss_available else None
        if ss_available and ss_list is None:
            print(f"Warning: no SS predictions for {seq_id}")
        if ss_available and len(ss_list) != len(seq_str):
            print(f"Warning: length mismatch for {seq_id}, skipping.")
            continue

        attention_weights, tokens, neq_preds = run_model(model, tokenizer, seq_str, device)
        print(f"{seq_id:15s}  "
              f"seq_len = {len(seq_str):3d}  "
              f"tokens = {len(tokens):3d}  "
              f"attn_shape = {attention_weights.shape}")
        attn_list = attention_weights.tolist()
        neq_list = neq_preds.cpu().numpy().tolist()

        row_dict = {
            "name": seq_id,
            "sequence": seq_str,
            "attention_weights": attn_list,
            "neq_preds": neq_list
        }

        if ss_available and ss_list is not None:
            row_dict["ss_pred"] = ss_list

        rows.append(row_dict)

        
    columns = ["name", "sequence", "attention_weights", "neq_preds"]
    if ss_available:
        columns.append("ss_pred")

    final_df = pd.DataFrame(rows, columns=columns)
    print(f"Constructed final DF with {len(final_df)} rows")

    final_df.to_json(f"{args.output}.json", orient="records", indent=2)
    print(f"Saved final JSON to {args.output}.json")
        

if __name__ == "__main__":
    main()