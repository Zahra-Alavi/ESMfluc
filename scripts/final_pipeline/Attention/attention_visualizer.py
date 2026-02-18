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
    parser.add_argument(
        "--annotations",
        type=str,
        nargs="*",
        default=[],
        choices=["ss_pred", "q3", "q8", "rsa", "asa", "disorder"],
        help="Structural features to plot as annotations (choose from: ss_pred, q3, q8, rsa, asa, disorder). Can specify multiple."
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


def prepare_annotation_array(data, data_type, feat_name, orientation='horizontal'):
    """
    Prepare annotation data for visualization.
    
    Args:
        data: list of values (categorical or continuous)
        data_type: 'categorical' or 'continuous'
        feat_name: name of the feature
        orientation: 'horizontal' or 'vertical'
    
    Returns:
        (z_array, colorscale, colorbar_config)
    """
    L = len(data)
    
    if data_type == 'categorical':
        if feat_name == 'q3':
            # 3-state secondary structure
            label_map = {'C': 0, 'H': 1, 'E': 2}
            nums = [label_map.get(str(s).strip(), 0) for s in data]
            colorscale = [
                [0.00, "orange"], [0.33, "orange"],  # coil
                [0.33, "blue"],   [0.66, "blue"],    # helix
                [0.66, "red"],    [1.00, "red"]      # sheet
            ]
            colorbar_config = dict(
                title="q3",
                tickvals=[0, 1, 2],
                ticktext=["C", "H", "E"]
            )
        elif feat_name == 'q8':
            # 8-state secondary structure
            label_map = {'G': 0, 'H': 1, 'I': 2, 'B': 3, 'E': 4, 'S': 5, 'T': 6, 'C': 7}
            nums = [label_map.get(str(s).strip(), 7) for s in data]
            # Create a discrete colorscale for 8 states
            colorscale = [
                [0.00, "#8dd3c7"], [0.125, "#8dd3c7"],  # G - light blue
                [0.125, "#ffffb3"], [0.25, "#ffffb3"],  # H - light yellow
                [0.25, "#bebada"], [0.375, "#bebada"],  # I - light purple
                [0.375, "#fb8072"], [0.5, "#fb8072"],   # B - light red
                [0.5, "#80b1d3"], [0.625, "#80b1d3"],   # E - blue
                [0.625, "#fdb462"], [0.75, "#fdb462"],  # S - orange
                [0.75, "#b3de69"], [0.875, "#b3de69"],  # T - light green
                [0.875, "#fccde5"], [1.00, "#fccde5"]   # C - pink
            ]
            colorbar_config = dict(
                title="q8",
                tickvals=list(range(8)),
                ticktext=['G', 'H', 'I', 'B', 'E', 'S', 'T', 'C']
            )
        else:
            nums = list(range(L))
            colorscale = 'Viridis'
            colorbar_config = dict(title=feat_name)
    else:
        # Continuous data
        nums = [float(v) for v in data]
        if feat_name == 'disorder':
            colorscale = 'YlOrRd'
        elif feat_name in ['rsa', 'asa']:
            colorscale = 'Blues'
        else:
            colorscale = 'Viridis'
        colorbar_config = dict(title=feat_name)
    
    # Create array with proper shape
    if orientation == 'horizontal':
        z_array = np.tile(np.array(nums), (5, 1))  # shape (5, L)
    else:  # vertical
        z_array = np.tile(np.array(nums).reshape(-1, 1), (1, 5))  # shape (L, 5)
    
    return z_array, colorscale, colorbar_config


def visualize_attention(attention_weights, tokens: list, seq_id: str, record: dict, annotation_features: list):
    """
    Generate and save an attention heatmap for a given sequence.
    seq_id will be used to name the output HTML file.
    record: the full data record containing all available features
    annotation_features: list of feature names to plot (e.g., ['ss_pred', 'rsa', 'disorder'])
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
    
    # Prepare annotations from the record
    annotations = []
    for feat_name in annotation_features:
        # Handle both 'q3' and 'ss_pred' as the same thing
        if feat_name in ['q3', 'ss_pred']:
            data = record.get('ss_pred', record.get('q3', None))
            if data and len(data) == seq_len:
                annotations.append(('q3', data, 'categorical'))
        elif feat_name in record and record[feat_name] and len(record[feat_name]) == seq_len:
            data_type = 'categorical' if feat_name == 'q8' else 'continuous'
            annotations.append((feat_name, record[feat_name], data_type))
    
    has_annotations = len(annotations) > 0
    
    if has_annotations:
        # Each annotation gets 0.05 height, attention gets the rest
        annotation_height = 0.05 * len(annotations)
        row_heights = [annotation_height, 1.0 - annotation_height]
        num_rows = 2
    else:
        row_heights = [1.0]
        num_rows = 1
    
    # Determine column widths (left annotation bars + main plot)
    if has_annotations:
        column_widths = [0.05 * len(annotations), 1.0 - 0.05 * len(annotations)]
        num_cols = 2
    else:
        column_widths = [1.0]
        num_cols = 1
    
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.005,
        horizontal_spacing=0.005,
        row_heights=row_heights,
        column_widths=column_widths
    )
    
    # Add top annotation bars (horizontal)
    if has_annotations:
        for idx, (feat_name, data, data_type) in enumerate(annotations):
            z_array, colorscale, colorbar_config = prepare_annotation_array(
                data, data_type, feat_name, orientation='horizontal'
            )
            # Position colorbar on the right side, stacked vertically
            colorbar_y = 0.95 - (idx * 0.15)
            colorbar_config['x'] = 1.15
            colorbar_config['y'] = colorbar_y
            colorbar_config['len'] = 0.1
            
            fig.add_trace(
                go.Heatmap(
                    z=z_array,
                    colorscale=colorscale,
                    showscale=True,
                    hoverinfo="none",
                    colorbar=colorbar_config,
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        
        # Add left annotation bars (vertical)
        for idx, (feat_name, data, data_type) in enumerate(annotations):
            z_array, colorscale, _ = prepare_annotation_array(
                data, data_type, feat_name, orientation='vertical'
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=z_array,
                    colorscale=colorscale,
                    showscale=False,
                    hoverinfo="none",
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)
        
        attn_row = 2
        attn_col = 2
    else:
        attn_row = 1
        attn_col = 1

    # attention heatmap 
    fig.add_trace(
        go.Heatmap(
            z=attention_weights,
            text=hover_text,
            hoverinfo='text',
            colorscale='Viridis',
            colorbar=dict(
                title="Attention",
                x=1.15,
                y=0.45,
                len=0.9
            ),
        ),
        row=attn_row, col=attn_col
    )
    
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=attn_row, col=attn_col)

    step = 25
    tick_positions = list(range(0, seq_len, step))
    tick_labels = [f"{i+1}-{tokens[i]}" for i in tick_positions]

    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=attn_row, col=attn_col
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=attn_row, col=attn_col
    )
    
    title_str = f"{seq_id}: Attention"
    if has_annotations:
        title_str += f" + {', '.join([a[0] for a in annotations])}"
    
    # Calculate dimensions to keep attention plot square
    if has_annotations:
        # Account for annotation bars and spacing
        plot_size = 1200
        total_width = plot_size + 300  # Extra space for colorbars and annotation bars
        total_height = plot_size + 200  # Extra space for top annotation bars
    else:
        plot_size = 1000
        total_width = plot_size + 200
        total_height = plot_size + 100
    
    fig.update_layout(
        title=title_str,
        width=total_width,
        height=total_height
    )
    
    fig.update_xaxes(title="Key Residue", row=attn_row, col=attn_col)
    fig.update_yaxes(title="Query Residue", row=attn_row, col=attn_col, scaleanchor=f"x{attn_col if num_cols > 1 else ''}")
    
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
    
    
def pheatmap_visualizer(attention_weights, tokens: list, seq_id: str, record: dict, annotation_features: list):
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
    
    # Create DataFrame with peak labels - keep full precision for smooth gradients
    df_mat = pd.DataFrame(attention_weights, index=row_names, columns=col_names)
    
    # Build annotations dict based on selected features
    anno_col_dict = {}
    anno_row_dict = {}
    annotation_col_cmaps = {}
    annotation_row_cmaps = {}
    
    # Always add NEQ if available
    neq_preds = record.get('neq_preds', None)
    if neq_preds is not None and len(neq_preds) == L:
        neq_labels = [f"NEQ_{int(v)}" for v in neq_preds]
        anno_col_dict["NEQ"] = neq_labels
        annotation_col_cmaps["NEQ"] = "Accent"
    
    # Add selected annotations
    for feat_name in annotation_features:
        # Handle both 'q3' and 'ss_pred' as the same thing
        if feat_name in ['q3', 'ss_pred']:
            data = record.get('ss_pred', record.get('q3', None))
            if data and len(data) == L:
                anno_col_dict["SS_q3"] = data
                anno_row_dict["SS_q3"] = data
                annotation_col_cmaps["SS_q3"] = "Set1"
                annotation_row_cmaps["SS_q3"] = "Set1"
        elif feat_name in record and record[feat_name] and len(record[feat_name]) == L:
            if feat_name == 'q8':
                anno_col_dict[feat_name] = record[feat_name]
                anno_row_dict[feat_name] = record[feat_name]
                annotation_col_cmaps[feat_name] = "Set3"
                annotation_row_cmaps[feat_name] = "Set3"
            else:
                # Continuous features (rsa, asa, disorder)
                anno_col_dict[feat_name] = record[feat_name]
                anno_row_dict[feat_name] = record[feat_name]
                if feat_name == 'disorder':
                    annotation_col_cmaps[feat_name] = "YlOrRd"
                    annotation_row_cmaps[feat_name] = "YlOrRd"
                elif feat_name in ['rsa', 'asa']:
                    annotation_col_cmaps[feat_name] = "Blues"
                    annotation_row_cmaps[feat_name] = "Blues"
                else:
                    annotation_col_cmaps[feat_name] = "viridis"
                    annotation_row_cmaps[feat_name] = "viridis"
    
    # Only build annotation DataFrames if we actually have annotations
    anno_row = pd.DataFrame(anno_row_dict, index=row_names) if anno_row_dict else None
    anno_col = pd.DataFrame(anno_col_dict, index=col_names) if anno_col_dict else None

    # Only pass colormap dicts if they are non-empty
    annotation_col_cmaps_arg = annotation_col_cmaps if annotation_col_cmaps else None
    annotation_row_cmaps_arg = annotation_row_cmaps if annotation_row_cmaps else None

    # Calculate square figure size based on sequence length
    base_size = max(10, L * 0.035)  # Smaller: 0.035 inch per residue, min 10 inches
    fig_width = base_size + 3
    fig_height = base_size + 3  # Square: same as width
    
    # Configure matplotlib for high-quality vector output
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editability
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['image.interpolation'] = 'none'  # No interpolation for crisp edges
    
    fig = pheatmap(
        df_mat, cmap="viridis",
        annotation_row=anno_row,
        annotation_col=anno_col,
        annotation_col_cmaps=annotation_col_cmaps_arg,
        annotation_row_cmaps=annotation_row_cmaps_arg,
        rownames_style=dict(rotation=45, size=5),
        colnames_style=dict(rotation=90, size=5),
        annotation_bar_space=0.3,
        show_rownames=True,
        show_colnames=True,
        width=fig_width,
        height=fig_height
    )

    out_file = f"{seq_id}_pheatmap.pdf"
    # Save as high-resolution PDF
    fig.savefig(out_file, dpi=600, bbox_inches='tight', format='pdf')
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
        
        # Check sequence length match with attention matrix
        if attention_weights.shape[0] != len(seq_str):
            print(f"Warning: length mismatch for {seq_id} (seq={len(seq_str)}, attn={attention_weights.shape[0]}), skipping.")
            continue
        
        # Get tokens from sequence
        tokens = get_tokens_from_sequence(seq_str)
        
        # Generate visualizations
        visualize_attention(attention_weights, tokens, seq_id, record, args.annotations)
        pheatmap_visualizer(attention_weights, tokens, seq_id, record, args.annotations)
      


if __name__ == "__main__":
    main()
