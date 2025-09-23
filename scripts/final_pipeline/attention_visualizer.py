#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:11:16 2025

@author: zalavi
"""

import argparse
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import EsmModel, EsmTokenizer
import biolib
import pandas as pd
import numpy as np
from pheatmap import pheatmap
import os
from models import BiLSTMWithSelfAttentionModel



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



def ss_preds(fasta_path):

    nsp3 = biolib.load('DTU/NetSurfP-3')
    nsp3_results = nsp3.cli(args=f"-i {fasta_path}")
    print("STDOUT:\n", nsp3_results.get_stdout())
    print("STDERR:\n", nsp3_results.get_stderr())
    nsp3_results.save_files("nsp3_results/")
    
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



def visualize_attention(model, tokenizer, seq, seq_id, ss_list, device):
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
    
    # display pixel info 
    hover_text = []
    for r in range(seq_len):
        row_text = []
        for c in range(seq_len):
            row_text.append(
                f"Query idx:{r+1} ({tokens[r]})<br>"
                f"Key idx:{c+1} ({tokens[c]})<br>"
                f"Weight: {attn_weights[r,c]:.4f}")
        hover_text.append(row_text)
        
    ss_map = {'C': 0, 'H': 1, 'E': 2}
    ss_nums = [ss_map[s] for s in ss_list]
    ss_array = np.array(ss_nums)       # shape (L,)
    ss_array_tiled = np.tile(ss_array, (5, 1))  # shape (5, L), to make the colored row in ss plot thicker
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # so they align horizontally
        vertical_spacing=0.005, # gap size
        row_heights=[0.1, 0.9]  # top row smaller
    )
    
    # top subplot is a single row for ss-preds
    ss_colorscale = [
        [0.00, "orange"],  [0.33, "orange"],  # coil
        [0.33, "blue"],    [0.66, "blue"],    # helix
        [0.66, "red"],     [1.00, "red"]      # sheet
    ]
    
    fig.add_trace(
        go.Heatmap(
            z=ss_array_tiled,
            colorscale=ss_colorscale,
            showscale=True,
            hoverinfo="none",
            colorbar=dict(
                title="Sec. Str.",
                tickvals=[0,1,2],
                ticktext=["Coil", "Helix", "Sheet"],
                x=1.08,  # move colorbar to the right
                y = 0.95, # palce near top
                len=0.1, # shorter volorbar
            ),
        ),
        row=1, col=1
    )
    
    
    # hide top subplot’s axes ticks/grids so it looks like a single color bar
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.5, 4.5], row=1, col=1)

    # bottom subplot: attention heatmap 
    fig.add_trace(
        go.Heatmap(
            z=attn_weights,
            text=hover_text,
            hoverinfo='text',
            colorscale='Viridis',
            # separate colorbar for attention
            colorbar=dict(
                title="Attention",
                x=1.08,   # same x offset so colorbars align
                y = 0.45,
                len=0.9
            ),
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=2, col=1)

    step = 25
    tick_positions = list(range(0, seq_len, step))
    tick_labels = [f"{i+1}-{tokens[i]}" for i in tick_positions]

    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=2, col=1
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{seq_id}: Attention + 3-state SS",
        width=1200, height=1200
    )
    
    fig.update_xaxes(title="Key Residue", row=2, col=1)
    fig.update_yaxes(title="Query Residue", row=2, col=1)
    

    # save as an interactive HTML file
    out_html = f"{seq_id}_attention_ss.html"
    fig.write_html(out_html)
    print(f"Saved interactive Plotly heatmap to {out_html}")
    
def find_local_peaks(array, global_threshold):
    """
    Return indices 'i' where array[i] is:
      - >= array[i-1]
      - >= array[i+1]
      - >= global_threshold
    For i in [1..len-2].
    """
    peaks = []
    L = len(array)
    for i in range(1, L-1):
        if (array[i] >= array[i-1] and
            array[i] >= array[i+1] and
            array[i] >= global_threshold):
            peaks.append(i)
    # Edge cases for i=0 or i=L-1 if you want to consider them
    return peaks
    
    
def pheatmap_visualizer( attention_weights, tokens: list, seq: str, seq_id: str, ss_list: list,neq_preds):
    
    L = attention_weights.shape[0]
    
    threshold = np.quantile(attention_weights, 0.90)

    
    col_maxes = attention_weights.max(axis=0)  
    col_peaks = find_local_peaks(col_maxes, threshold)
    col_names = ["" for _ in range(L)]
    for i in col_peaks:
        col_names[i] = f"{i+1}-{tokens[i]}"

    row_maxes = attention_weights.max(axis=1)
    row_peaks = find_local_peaks(row_maxes, threshold)
    row_names = ["" for _ in range(L)]
    for r in row_peaks:
        row_names[r] = f"{r+1}-{tokens[r]}"
    
    
    df_mat = pd.DataFrame(attention_weights,index=row_names,columns=col_names)
    df_mat = df_mat.round(2)
    
    neq_labels = [f"NEQ_{int(v)}" for v in neq_preds]  # length L

    
    anno_row = pd.DataFrame({"SS": ss_list}, index=row_names)
    
    anno_col = pd.DataFrame({"NEQ": neq_labels, "SS": ss_list}, index=col_names)

    annotation_col_cmaps = {"NEQ": "Accent", "SS": "Set1"}
    annotation_row_cmaps = {"SS": "Set1"}
    
    fig = pheatmap(
        df_mat,cmap = "viridis",
        annotation_row=anno_row,
        annotation_col=anno_col,
        annotation_col_cmaps=annotation_col_cmaps,
        annotation_row_cmaps=annotation_row_cmaps,
        rownames_style = dict(rotation=45, size=4),
        colnames_style = dict(rotation=90, size=6),
        annotation_bar_space=0.3,
        show_rownames=True,
        show_colnames=True   
    )

    out_file = f"{seq_id}_pheatmap.pdf"
    fig.savefig(out_file, dpi=300)
    print(f"[visualize_attention] Saved {out_file}")
    
def main():
    
    
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ss_preds(args.fasta_file)
    df = pd.read_csv("nsp3_results/results.csv", index_col=False)
    ss_map = parse_nsp3_csv(df)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        if seq_id not in ss_map:
            print(f"Warning: no SS predictions for {seq_id}")
            continue
        ss_list = ss_map[seq_id]
        if len(ss_list) != len(seq_str):
            print(f"Warning: length mismatch for {seq_id}, skipping.")
            continue
        attention_weights, tokens, neq_preds = run_model(model, tokenizer, seq_str, device)
        pheatmap_visualizer(attention_weights, tokens, seq_str, seq_id, ss_list, neq_preds)
      



if __name__ == "__main__":
    main()
