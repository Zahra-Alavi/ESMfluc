#!/usr/bin/env python3
"""
compare_attn_temperatures.py

Randomly samples N proteins common to all 5 temperature attention files
and generates one HTML figure per protein showing the attention matrix
at 320K, 348K, 379K, 413K, 450K side-by-side.

Usage
-----
python compare_attn_temperatures.py \\
    --results_dir /home/zahralab/Desktop/ESMfluc/scripts/final_pipeline/results \\
    --output_dir  /home/zahralab/Desktop/ESMfluc/results/temp_attn_comparison \\
    [--n_proteins 10] \\
    [--seed 42] \\
    [--proteins 1abc_A 2xyz_B]   # optional: specific proteins instead of random
"""

import argparse
import json
import os
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPERATURES = [320, 348, 379, 413, 450]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_attention_file(path):
    """Load a test_attention.json and return {name: record} dict."""
    with open(path) as f:
        data = json.load(f)
    # handle both list-of-dicts and dataframe-orient records
    if isinstance(data, list):
        return {r["name"]: r for r in data}
    raise ValueError(f"Unexpected JSON format in {path}")


def find_common_proteins(all_dicts):
    """Return sorted list of protein names present in every temperature dict."""
    common = set(all_dicts[0].keys())
    for d in all_dicts[1:]:
        common &= set(d.keys())
    return sorted(common)


# ── Heatmap builder ───────────────────────────────────────────────────────────

def make_tick_config(seq, step=25):
    """Produce tickvals/ticktext at every `step` residues (1-based labels)."""
    n = len(seq)
    positions = list(range(0, n, step))
    labels = [f"{i+1}{seq[i]}" for i in positions]
    return positions, labels


def build_figure(protein_name, temp_records):
    """
    Build a 1-row × 5-column Plotly figure with attention heatmaps.

    temp_records : list of 5 records in temperature order
    """
    seq = temp_records[0]["sequence"]
    n = len(seq)

    # Shared color range across all temperatures for fair comparison
    all_values = np.concatenate([
        np.array(r["attention_weights"]).ravel() for r in temp_records
    ])
    vmin, vmax = float(all_values.min()), float(all_values.max())

    tick_vals, tick_text = make_tick_config(seq, step=max(1, n // 8))

    fig = make_subplots(
        rows=1,
        cols=5,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=[f"{t}K" for t in TEMPERATURES],
    )

    for col_idx, (record, temp) in enumerate(zip(temp_records, TEMPERATURES), start=1):
        A = np.array(record["attention_weights"])

        # hover text
        hover = [
            [
                f"Q:{ri+1}({seq[ri]})  K:{ci+1}({seq[ci]})<br>weight:{A[ri,ci]:.4f}"
                for ci in range(n)
            ]
            for ri in range(n)
        ]

        show_scale = col_idx == 5  # only rightmost panel shows the colorbar

        fig.add_trace(
            go.Heatmap(
                z=A,
                text=hover,
                hoverinfo="text",
                colorscale="Viridis",
                zmin=vmin,
                zmax=vmax,
                showscale=show_scale,
                colorbar=dict(
                    title="Attention",
                    thickness=15,
                    len=0.8,
                    x=1.01,
                ),
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45,
            title_text="Key",
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            title_text="Query" if col_idx == 1 else "",
            row=1,
            col=col_idx,
        )

    panel_size = max(300, min(600, n * 4))  # scale panel size with protein length

    fig.update_layout(
        title=dict(
            text=f"{protein_name}  (L={n})  — attention at 5 temperatures",
            font=dict(size=16),
        ),
        width=panel_size * 5 + 200,
        height=panel_size + 150,
        plot_bgcolor="white",
    )

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir",
        required=True,
        help=(
            "Directory containing mdcath_320K/, mdcath_348K/, ... subdirs, "
            "each with test_attention.json inside."
        ),
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write one HTML file per protein.",
    )
    p.add_argument(
        "--n_proteins",
        type=int,
        default=10,
        help="Number of proteins to sample randomly (default 10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default 42).",
    )
    p.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of protein names to visualize. "
            "Overrides --n_proteins / --seed."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load all 5 files ──────────────────────────────────────────────────────
    all_dicts = []
    for temp in TEMPERATURES:
        path = os.path.join(args.results_dir, f"mdcath_{temp}K", "test_attention.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Expected attention file not found: {path}\n"
                "Check --results_dir points to the directory containing mdcath_320K/ etc."
            )
        print(f"Loading {temp}K  →  {path}")
        all_dicts.append(load_attention_file(path))

    # ── select proteins ───────────────────────────────────────────────────────
    common = find_common_proteins(all_dicts)
    print(f"\n{len(common)} proteins common to all 5 temperatures.")

    if args.proteins:
        chosen = []
        for name in args.proteins:
            if name in set(common):
                chosen.append(name)
            else:
                print(f"  WARNING: {name} not found in all temperatures — skipping.")
        if not chosen:
            raise ValueError("None of the requested proteins are available in all temperatures.")
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(common, min(args.n_proteins, len(common)))
        chosen.sort()

    print(f"Visualizing {len(chosen)} proteins: {chosen}\n")

    # ── generate figures ──────────────────────────────────────────────────────
    for protein in chosen:
        records = [d[protein] for d in all_dicts]
        print(f"  {protein} (L={len(records[0]['sequence'])}) ... ", end="", flush=True)

        fig = build_figure(protein, records)

        safe_name = protein.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(args.output_dir, f"{safe_name}_temps.html")
        fig.write_html(out_path)
        print(f"saved → {out_path}")

    print(f"\nDone. {len(chosen)} files written to {args.output_dir}")


if __name__ == "__main__":
    main()
