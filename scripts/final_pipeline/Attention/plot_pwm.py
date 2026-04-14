#!/usr/bin/env python3
"""
Visualize PWM enrichment matrices from extract_bright_region_motifs.py output.

Produces per-pattern sequence logos (letter height ∝ enrichment, like DNA logos)
and an enrichment heatmap, saved as PDF.

Letter colors follow physicochemical groups:
  hydrophobic (GAVLIPFWM) → black
  polar/small  (STCNQ)    → green
  acidic       (DE)        → red
  basic        (KRH)       → blue
  aromatic     (FWY)       → purple  (overrides hydrophobic for F/W/Y)
  proline      (P)         → orange
  glycine      (G)         → gray
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker


# Physicochemical color scheme for amino acids
AA_COLORS = {
    'G': '#888888',  # gray   – glycine
    'A': '#222222',  'V': '#222222',  'L': '#222222',  'I': '#222222',
    'M': '#222222',                                                      # black  – aliphatic
    'P': '#e67e00',                                                      # orange – proline
    'F': '#7b2d8b',  'W': '#7b2d8b',  'Y': '#7b2d8b',                  # purple – aromatic
    'S': '#2ca02c',  'T': '#2ca02c',  'C': '#2ca02c',
    'N': '#2ca02c',  'Q': '#2ca02c',                                    # green  – polar
    'D': '#d62728',  'E': '#d62728',                                    # red    – acidic
    'K': '#1f77b4',  'R': '#1f77b4',  'H': '#1f77b4',                  # blue   – basic
}


def load_pwm(csv_path: str):
    """Load a pwm CSV and return a DataFrame[position × AA] of enrichment values."""
    df = pd.read_csv(csv_path)
    positions = sorted(df['position'].unique())
    aas = sorted(df['AA'].unique())
    mat = pd.DataFrame(0.0, index=positions, columns=aas)
    for _, row in df.iterrows():
        mat.loc[row['position'], row['AA']] = row['enrichment']
    return mat  # enrichment: values > 1 = enriched, < 1 = depleted


def plot_logo(mat: pd.DataFrame, title: str, out_path: str):
    """
    Sequence logo where letter height = max(0, enrichment - 1.0).
    Depleted residues are shown below zero as downward letters (enrichment - 1, capped at -1).
    """
    positions = list(mat.index)
    n_pos = len(positions)

    # logomaker wants a DataFrame[position × AA] with numeric heights
    logo_df_pos = mat.copy().clip(lower=1.0) - 1.0   # positive part
    logo_df_neg = mat.copy().clip(upper=1.0) - 1.0   # negative part (≤ 0)

    # Build color scheme dict
    color_scheme = {aa: AA_COLORS.get(aa, '#333333') for aa in mat.columns}

    fig, axes = plt.subplots(2, 1, figsize=(max(6, n_pos * 1.0), 4),
                             gridspec_kw={'height_ratios': [3, 1]})

    # ── top panel: enriched letters (height > 0) ──────────────────────────
    ax_pos = axes[0]
    if logo_df_pos.values.max() > 0:
        logomaker.Logo(logo_df_pos, ax=ax_pos, color_scheme=color_scheme, show_spines=False)
    ax_pos.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax_pos.set_ylabel("Enrichment − 1", fontsize=9)
    ax_pos.set_title(title, fontsize=11, fontweight='bold')
    ax_pos.set_xticks(range(n_pos))
    ax_pos.set_xticklabels([f"{p:+d}" for p in positions])
    ax_pos.tick_params(axis='x', bottom=False)

    # ── bottom panel: depleted letters (height < 0, flipped) ──────────────
    ax_neg = axes[1]
    logo_df_neg_flipped = (-logo_df_neg)   # make positive so logomaker renders them
    if logo_df_neg_flipped.values.max() > 0:
        logomaker.Logo(logo_df_neg_flipped, ax=ax_neg, color_scheme=color_scheme,
                       show_spines=False, flip_below=False)
        ax_neg.invert_yaxis()
    ax_neg.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax_neg.set_ylabel("Depletion", fontsize=9)
    ax_neg.set_xticks(range(n_pos))
    ax_neg.set_xticklabels([f"{p:+d}" for p in positions])
    ax_neg.set_xlabel("Position relative to bright residue", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap(mat: pd.DataFrame, title: str, out_path: str):
    """Enrichment heatmap (AAs × positions), diverging color around 1.0."""
    z = mat.T.values   # shape (AAs, positions)
    aa_labels = list(mat.columns)
    pos_labels = [f"{p:+d}" for p in mat.index]

    vdev = max(abs(np.nanmax(z) - 1.0), abs(np.nanmin(z) - 1.0), 0.3)
    vmin, vmax = 1.0 - vdev, 1.0 + vdev

    fig, ax = plt.subplots(figsize=(max(5, len(pos_labels) * 0.8), max(5, len(aa_labels) * 0.35)))
    im = ax.imshow(z, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pos_labels)))
    ax.set_xticklabels(pos_labels)
    ax.set_yticks(range(len(aa_labels)))
    ax.set_yticklabels(aa_labels, fontsize=8)
    ax.set_xlabel("Position relative to bright residue")
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Enrichment (obs/expected)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PWM sequence logos and heatmaps.")
    parser.add_argument(
        "--motif_dir",
        type=str,
        default="scripts/final_pipeline/Attention/motif_analysis",
        help="Directory containing pwm_*.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save figures (default: same as --motif_dir).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.motif_dir
    os.makedirs(out_dir, exist_ok=True)

    pwm_files = {
        "Coil (PatA)":   "pwm_coil_pata.csv",
        "Helix (PatB)":  "pwm_helix_patb.csv",
        "Strand (Bg)":   "pwm_strand_bg.csv",
        "All":           "pwm_all.csv",
    }
    file_stems = {
        "Coil (PatA)":  "coil_PatA",
        "Helix (PatB)": "helix_PatB",
        "Strand (Bg)":  "strand_Bg",
        "All":          "all",
    }

    for label, fname in pwm_files.items():
        fpath = os.path.join(args.motif_dir, fname)
        if not os.path.exists(fpath):
            print(f"[skip] {fpath} not found")
            continue
        stem = file_stems[label]
        mat = load_pwm(fpath)
        plot_logo(mat, title=f"Sequence logo — {label}", out_path=os.path.join(out_dir, f"logo_{stem}.png"))
        plot_heatmap(mat, title=f"Enrichment — {label}", out_path=os.path.join(out_dir, f"heatmap_{stem}.png"))


if __name__ == "__main__":
    main()
