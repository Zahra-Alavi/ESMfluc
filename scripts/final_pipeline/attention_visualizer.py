#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize attention weights from get_attn.py JSON output.
Takes pre-computed attention weights from JSON instead of running the model.
"""

import argparse
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pheatmap import pheatmap
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention weights from JSON file."
    )
    parser.add_argument(
        "--attention_json",
        type=str,
        required=True,
        help="Path to JSON file with attention data (from get_attn.py output)."
    )
    parser.add_argument(
        "--fasta_file",
        type=str,
        required=True,
        help="Path to the FASTA file containing sequences (for matching)."
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


def get_tokens_from_sequence(sequence):
    """
    Convert amino acid sequence to token list (one token per amino acid).
    """
    return list(sequence)


def visualize_attention(attention_weights, tokens: list, seq_id: str, ss_list=None):
    """
    Generate and save an attention heatmap for a given sequence.
    seq_id will be used to name the output HTML file.
    ss_list is optional (may be None if not available).
    """
    seq_len = attention_weights.shape[0]
    
    # Build hover text for each cell
    hover_text = []
    for r in range(seq_len):
        row_text = []
        for c in range(seq_len):
            row_text.append(
                f"Query idx:{r+1} ({tokens[r]})<br>"
                f"Key idx:{c+1} ({tokens[c]})<br>"
                f"Weight: {attention_weights[r,c]:.4f}")
        hover_text.append(row_text)
    
    # Only create SS plot if ss_list is provided
    has_ss = ss_list is not None and len(ss_list) > 0
    
    if has_ss:
        ss_map = {'C': 0, 'H': 1, 'E': 2}
        ss_nums = [ss_map.get(s, 0) for s in ss_list]  # default to coil if unknown
        ss_array = np.array(ss_nums)       # shape (L,)
        ss_array_tiled = np.tile(ss_array, (5, 1))  # shape (5, L)
        row_heights = [0.1, 0.9]
        num_rows = 2
    else:
        row_heights = [1.0]
        num_rows = 1
    
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
        row_heights=row_heights
    )
    
    # top subplot is a single row for ss-preds (if available)
    if has_ss:
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
                    x=1.08,
                    y = 0.95,
                    len=0.1,
                ),
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, range=[-0.5, 4.5], row=1, col=1)
        attn_row = 2
    else:
        attn_row = 1

    # attention heatmap 
    fig.add_trace(
        go.Heatmap(
            z=attention_weights,
            text=hover_text,
            hoverinfo='text',
            colorscale='Viridis',
            colorbar=dict(
                title="Attention",
                x=1.08,
                y = 0.45,
                len=0.9
            ),
        ),
        row=attn_row, col=1
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=attn_row, col=1)

    step = 25
    tick_positions = list(range(0, seq_len, step))
    tick_labels = [f"{i+1}-{tokens[i]}" for i in tick_positions]

    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=attn_row, col=1
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=attn_row, col=1
    )
    
    title_str = f"{seq_id}: Attention"
    if has_ss:
        title_str += " + 3-state SS"
    fig.update_layout(
        title=title_str,
        width=1200, height=1200 if has_ss else 800
    )
    
    fig.update_xaxes(title="Key Residue", row=attn_row, col=1)
    fig.update_yaxes(title="Query Residue", row=attn_row, col=1)
    
    # save as an interactive HTML file
    out_html = f"{seq_id}_attention.html"
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
    return peaks
    
    
def pheatmap_visualizer(attention_weights, tokens: list, seq_id: str, ss_list=None, neq_preds=None):
    """
    Generate pheatmap PDF visualization of attention with optional annotations.
    """
    L = attention_weights.shape[0]
    
    threshold = np.quantile(attention_weights, 0.90)

    # Find peaks in column and row maxima
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
    
    # Create DataFrame with peak labels
    df_mat = pd.DataFrame(attention_weights, index=row_names, columns=col_names)
    df_mat = df_mat.round(2)
    
    # Build annotations dict based on available data
    anno_col_dict = {}
    annotation_col_cmaps = {}
    annotation_row_cmaps = {}
    
    if neq_preds is not None:
        neq_labels = [f"NEQ_{int(v)}" for v in neq_preds]
        anno_col_dict["NEQ"] = neq_labels
        annotation_col_cmaps["NEQ"] = "Accent"
    
    if ss_list is not None:
        anno_col_dict["SS"] = ss_list
        annotation_col_cmaps["SS"] = "Set1"
        anno_row_dict = {"SS": ss_list}
        annotation_row_cmaps["SS"] = "Set1"
    else:
        anno_row_dict = {}
    
    anno_row = pd.DataFrame(anno_row_dict, index=row_names)
    anno_col = pd.DataFrame(anno_col_dict, index=col_names)
    
    fig = pheatmap(
        df_mat, cmap="viridis",
        annotation_row=anno_row,
        annotation_col=anno_col,
        annotation_col_cmaps=annotation_col_cmaps,
        annotation_row_cmaps=annotation_row_cmaps,
        rownames_style=dict(rotation=45, size=4),
        colnames_style=dict(rotation=90, size=6),
        annotation_bar_space=0.3,
        show_rownames=True,
        show_colnames=True   
    )

    out_file = f"{seq_id}_pheatmap.pdf"
    fig.savefig(out_file, dpi=300)
    print(f"[pheatmap_visualizer] Saved {out_file}")
    
    
def main():
    """
    Load attention data from JSON and generate visualizations for each sequence.
    """
    args = parse_args()
    
    # Load attention data from JSON
    print(f"Loading attention data from {args.attention_json}")
    with open(args.attention_json, 'r') as f:
        attention_data = json.load(f)
    
    # Create a mapping of seq_id to attention record
    attn_map = {record['name']: record for record in attention_data}
    
    # Parse FASTA file and visualize
    for seq_id, seq_str in parse_fasta_file(args.fasta_file):
        if seq_id not in attn_map:
            print(f"Warning: no attention data for {seq_id}")
            continue
        
        record = attn_map[seq_id]
        attention_weights = np.array(record['attention_weights'])
        neq_preds = np.array(record.get('neq_preds', []))
        ss_list = record.get('ss_pred', None)
        
        # Check sequence length match with attention matrix
        if attention_weights.shape[0] != len(seq_str):
            print(f"Warning: length mismatch for {seq_id} (seq={len(seq_str)}, attn={attention_weights.shape[0]}), skipping.")
            continue
        
        # Get tokens from sequence
        tokens = get_tokens_from_sequence(seq_str)
        
        # Generate visualizations
        visualize_attention(attention_weights, tokens, seq_id, ss_list)
        pheatmap_visualizer(attention_weights, tokens, seq_id, ss_list, neq_preds)
      


if __name__ == "__main__":
    main()
