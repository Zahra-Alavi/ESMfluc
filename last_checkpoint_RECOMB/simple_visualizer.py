#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:46:27 2025

@author: zalavi
"""

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
                # if we already have a seq_id, yield the previous record
                if seq_id is not None:
                    yield seq_id, "".join(seq_lines)
                seq_id = line[1:]  # everything after ">"
                seq_lines = []
            else:
                seq_lines.append(line)
        # yield the last record if present
        if seq_id is not None and seq_lines:
            yield seq_id, "".join(seq_lines)
            
            
            
def run_model(model, tokenizer, sequence, device):     
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attn_weights_torch = model(input_ids, attn_mask, return_attention=True)
        # attn_weights_torch shape: [batch_size, seq_len, seq_len]
    attn_weights = attn_weights_torch[0].cpu().numpy()  # shape=(L,L)
    
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # length=L
    
    y_probs = torch.softmax(logits, dim=-1)
    y_preds = torch.argmax(y_probs, dim=-1)
    y_preds = y_preds.view(-1)
    
    return attn_weights, tokens, y_preds


def visualize_attention(attention_weights, tokens, seq, seq_id):
    """
    Generate and save an attention heatmap for a given sequence.
    seq_id will be used to name the output PNG file.
    """
    
   
    seq_len = attention_weights.shape[0]
    
    # display pixel info 
    hover_text = []
    for r in range(seq_len):
        row_text = []
        for c in range(seq_len):
            row_text.append(
                f"Query idx:{r+1} ({tokens[r]})<br>"
                f"Key idx:{c+1} ({tokens[c]})<br>"
                f"Weight: {attention_weights[r,c]:.4f}")
        hover_text.append(row_text)
        
    
    
    
    fig = go.Figure(data = go.Heatmap(
            z=attention_weights,
            text=hover_text,
            hoverinfo='text',
            colorscale='Viridis',
            )
        )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    step = 25
    tick_positions = list(range(0, seq_len, step))
    tick_labels = [f"{i+1}-{tokens[i]}" for i in tick_positions]

    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
    )
    
    fig.update_layout(
        title=f"{seq_id}: Attention",
        width=1200, height=1200
    )
    
    fig.update_xaxes(title="Key Residue")
    fig.update_yaxes(title="Query Residue")
    

    # save as an interactive HTML file
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
        
        attention_weights, tokens, neq_preds = run_model(model, tokenizer, seq_str, device)
        visualize_attention(attention_weights, tokens, seq_str, seq_id)
      



if __name__ == "__main__":
    main()
