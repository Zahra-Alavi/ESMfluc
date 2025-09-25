#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:49:53 2025

Description: This script takes a JSON file including seq names, sequences, attention weights, neq_pred and optionally ss_pred. 
(input JSON is the output of get_attn.py)

It can:
    - plot attention heatmaps, optionally with neq_pred/ss_pred as annotation col/row.results will be saved as png files in ../../results.
    - perform PCA on a single attention matrix (one sequence)
    - if the input file contains viriants you could: 
        . plot attention differences between a mutant and wildtype 
        . perform PCA across all mutants
        . perfrom tSNE on the collection

"""

import pandas as pd
import numpy  as np
from pheatmap import pheatmap
import matplotlib.pyplot as plt      
from sklearn.decomposition import PCA
from sklearn.preprocessing  import StandardScaler

# helper function to get aa indices receiving peak attention, to mark in plots
def local_peaks(arr, perc=0.80):
    """
    arr  : 1‑D numpy array
    perc : global percentile threshold (0–1)
    Return indices i that are (strict) local maxima and ≥ `perc`‑quantile.
    """
    thr   = np.quantile(arr, perc)
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] >= thr and arr[i] > arr[i-1] and arr[i] >= arr[i+1]:
            peaks.append(i)
    return peaks


def plot_attention_heatmap(
        seq_id,
        seq_str,
        attention_weights,           # (L,L) list‑of‑lists or ndarray
        vmin = None,
        vmax = None,
        ss_list=None,                # length L
        neq_list=None,               # length L
        out_prefix="attention_",
        peak_quantile=0.80,
        ann_space=0.35,                
        tick_rot=60, tick_size=6):

    A  = np.asarray(attention_weights, dtype=float)
    L  = len(seq_str)
    assert A.shape == (L, L), "attention matrix must be L×L"
    
    if vmin is None:           
        vmin = 0.0
    if vmax is None:
        vmax = round(A.max(), 2)

    # optionally change axis for rows 
    col_peaks = local_peaks(A.max(axis=0), perc=peak_quantile)
    row_peaks = local_peaks(A.max(axis=1), perc=peak_quantile)

    # make label lists, same length as sequence
    col_names = [f"{i+1}-{seq_str[i]}" if i in col_peaks else "" for i in range(L)]
    row_names = [f"{i+1}-{seq_str[i]}" if i in row_peaks else "" for i in range(L)]
    
    # uncomment if you just wanto to mark every 10th index instead of peaks 
    #row_names = [f"{i+1}" if i % 10 == 0 else "" for i in range(L)]
    #col_names = [f"{i+1}" if i % 10 == 0 else "" for i in range(L)]

    df_mat = pd.DataFrame(A, index=row_names, columns=col_names)

    #  annotation bars 
    row_anno, col_anno = {}, {}
    if ss_list is not None:
        row_anno["SS"] = ss_list
        col_anno["SS"] = ss_list
    if neq_list is not None:
        row_anno["NEQ"] = list(map(str, neq_list))
        col_anno["NEQ"] = list(map(str, neq_list))

    row_anno_df = pd.DataFrame(row_anno, index=row_names) if row_anno else None
    col_anno_df = pd.DataFrame(col_anno, index=col_names) if col_anno else None

    cmap_row = cmap_col = {"SS": "Set1", "NEQ": "Set2"}

    # draw heat‑map 
    fig = pheatmap(
        df_mat,
        cmap="viridis",
        vmin=vmin,          # same limits everywhere
        vmax=vmax,
        annotation_row=row_anno_df,
        annotation_col=col_anno_df,
        annotation_row_cmaps={k: cmap_row[k] for k in row_anno} if row_anno else None,
        annotation_col_cmaps={k: cmap_col[k] for k in col_anno} if col_anno else None,
        show_rownames=True,
        show_colnames=True,
        annotation_bar_space=ann_space,
        rownames_style=dict(rotation=tick_rot, size=tick_size, ha="right"),
        colnames_style=dict(rotation=tick_rot, size=tick_size, va="top")
    )

    
    fig.suptitle(seq_id, fontsize=12, y=0.97)

    out_file = f"../../results/{out_prefix}_{seq_id}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
   
               
    print(f"Saved {out_file}")


def plot_all(df, out_pref="attn_"):
    """Plot each attention matrix as‑is (no subtraction)."""
    for _, row in df.iterrows():
        plot_attention_heatmap(
            seq_id   = row["name"],
            seq_str  = row["sequence"],
            attention_weights = row["attention_weights"],
            ss_list  = row.get("ss_pred"),
            neq_list = row.get("neq_preds"),
            out_prefix = out_pref
    )


def pca_single(attn_matrix, n_pc=5, seq_label=""):
    A   = np.asarray(attn_matrix, float)
    A_c = A - A.mean(axis=0, keepdims=True)

    pca = PCA(n_components=n_pc).fit(A_c)
    scores = pca.transform(A_c)         # (L, n_pc)
    loads  = pca.components_            # (n_pc, L)

    print(f"[{seq_label}] explained variance:",
          ", ".join(f"{v:.3f}" for v in pca.explained_variance_ratio_[:3]))

    # scatter of first two PCs (residues as points) 
    plt.figure(figsize=(5, 4))
    plt.scatter(scores[:, 0], scores[:, 1], s=15, c="k")
    for i in range(len(A)):
        plt.text(scores[i, 0], scores[i, 1], str(i+1), fontsize=6)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Residues in PC space – {seq_label}")
    plt.tight_layout(); plt.show()

    # loading profiles of PC1/PC2
    plt.figure(figsize=(8, 3))
    plt.plot(loads[0], label="PC1 loadings")
    plt.plot(loads[1], label="PC2 loadings")
    plt.xlabel("Key position"); plt.legend(); plt.tight_layout(); plt.show()
    
    
    
    
def plot_diff_heatmaps(df, wt_row=0, peak_q=0.80, out_pref="diff_"):
    """
    Draw one heat‑map per sequence showing |A_mut – A_WT|.

    df       : DataFrame with columns
               [name, sequence, attention_weights, ss_pred?, neq_preds?]
    wt_row   : index of the WT inside df   (default = 0)
    peak_q   : percentile for local_peaks  (passed on to plot_attention_heatmap)
    out_pref : prefix for the PNG files    (“diff_<seq>.png”)
    """
    
    wt_A = np.asarray(df.loc[wt_row, "attention_weights"], float)

    # set global limits for a comparable colour scale 
    vmin = min(
        np.abs(np.asarray(m, float) - wt_A).min() for m in df["attention_weights"]
    )
    vmax = max(
        np.abs(np.asarray(m, float) - wt_A).max() for m in df["attention_weights"]
    )
    print(f"[Δ‑attn] common scale  vmin={vmin:.4f}  vmax={vmax:.4f}")

    # draw every mutant incl. WT (WT Δ will be zero everywhere) 
    for _, row in df.iterrows():
        diff_A = np.abs(np.asarray(row["attention_weights"], float) - wt_A)

        plot_attention_heatmap(
            seq_id   = row["name"],
            seq_str  = row["sequence"],
            attention_weights = diff_A,
            vmin=vmin, vmax=vmax,                #force same range
            ss_list  = row.get("ss_pred"),
            neq_list = row.get("neq_preds"),
            out_prefix = out_pref,
            peak_quantile = peak_q
        )
        
        
def pca_on_stack(df, n_pc=2): # PCA across the whole collection (mutant‑space)
    X = np.stack([np.asarray(m, float).ravel()      # L²‑vector
                  for m in df["attention_weights"]])
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_pc).fit(X_std)
    Z   = pca.transform(X_std)
    
    print("explained variance:",
          ", ".join(f"{v:.3f}" for v in pca.explained_variance_ratio_[:3]))

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c="steelblue")
    for i, name in enumerate(df["name"]):
        plt.text(Z[i, 0], Z[i, 1], name, fontsize=7)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    plt.title("PCA of attention matrices"); plt.tight_layout(); plt.show()

    return pca, Z          




def tsne_on_stack(df, perplexity=30, lr=100):
    from sklearn.manifold import TSNE

    X = np.stack([np.asarray(m, float).ravel()
                  for m in df["attention_weights"]])
    X_std = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=perplexity,
                learning_rate=lr, random_state=0, init="pca")
    Z = tsne.fit_transform(X_std)

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c="slateblue")
    for i, name in enumerate(df["name"]):
        plt.text(Z[i, 0], Z[i, 1], name, fontsize=8)
    plt.title("t‑SNE projection of attention matrices")
    plt.tight_layout(); plt.show()

    return Z                   

df = pd.read_json("../../results/output.json", orient="records")


import random

random_numbers = random.sample(range(0, 279), 10)
    
for number in random_numbers:
    plot_attention_heatmap(
            seq_id   = df.loc[number,"name"],
            seq_str  = df.loc[number,"sequence"],
            attention_weights = df.loc[number,"attention_weights"],
            ss_list  = df.loc[number,"ss_pred"],
            neq_list = df.loc[number,"neq_preds"],
    )
    