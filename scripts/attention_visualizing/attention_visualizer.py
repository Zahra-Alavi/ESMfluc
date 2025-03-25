#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:11:16 2025

@author: zalavi
"""

import argparse
import torch
import plotly.graph_objects as go
from transformers import EsmModel, EsmTokenizer

from models import BiLSTMWithSelfAttentionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention weights for sequences in a FASTA file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (e.g., best_model_shuffled.pth)."
    )
    parser.add_argument(
        "--fasta_file",
        type=str,
        required=True,
        help="Path to the FASTA file containing sequences."
    )
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
                # If we already have a seq_id, yield the previous record
                if seq_id is not None:
                    yield seq_id, "".join(seq_lines)
                seq_id = line[1:]  # everything after ">"
                seq_lines = []
            else:
                seq_lines.append(line)
        # Yield the last record if present
        if seq_id is not None and seq_lines:
            yield seq_id, "".join(seq_lines)




def visualize_attention(model, tokenizer, seq, seq_id):
    """
    Generate and save an attention heatmap for a given sequence.
    seq_id will be used to name the output PNG file.
    """
    
    enc = tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(input_ids, attn_mask, return_attention=True)

    # attn_weights: [batch_size, seq_len, seq_len]
    attn_weights = attn_weights[0].cpu().numpy()    # shape = [seq_len, seq_len]
    
    token_ids = input_ids[0].tolist()  # shape = [seq_len], since batch_size=1
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    seq_len = attn_weights.shape[0]
    
    # Display pixel info 
    hover_text = []
    for r in range(seq_len):
        row_text = []
        for c in range(seq_len):
            row_text.append(
                f"Query idx:{r+1} ({tokens[r]})<br>"
                f"Key idx:{c+1} ({tokens[c]})<br>"
                f"Weight: {attn_weights[r,c]:.4f}")
        hover_text.append(row_text)
        
        
    fig = go.Figure(
        data=go.Heatmap(
            z=attn_weights,
            text=hover_text,     
            hoverinfo='text',      
            colorscale='Viridis'))

   
    fig.update_layout(
        title=f"Attention Heatmap: {seq_id}",
        xaxis=dict(title='Key positions'),
        yaxis=dict(title='Query positions'),  
        width=1200,
        height=1200
        )
    
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1)
        )

    step = 25
    tick_positions = list(range(0, seq_len, step))
    tick_labels = [f"{i+1}-{tokens[i]}" for i in tick_positions]

    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels
    )

    # Save as an interactive HTML file
    out_html = f"{seq_id}_attention.html"
    fig.write_html(out_html)
    print(f"Saved interactive Plotly heatmap to {out_html}")
    
def main():
    
    args = parse_args()

    embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    embedding_model.to(device)


    freeze_list = range(0, 5)
    for name, param in embedding_model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split(".")[2])
            param.requires_grad = not layer_num in freeze_list
        else:   
            param.requires_grad = True
        

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
    
    for seq_id, seq_str in parse_fasta_file(args.fasta_file):
        visualize_attention(model, tokenizer, seq_str, seq_id)



if __name__ == "__main__":
    main()
