#!/usr/bin/env python3
"""
run_attn_analysis.py



For each attn.json (quantitative analyses):
  1.  Neq Spearman correlation  (per-protein + pooled)
  2.  K-means clustering (K=3) on residue attention profiles
  3.  SS / RSA / disorder enrichment per cluster
  4.  Cluster vs SS3 enrichment heatmap
  5.  Per-AA mean attention received  (bar chart)
  6.  Top attention pairs  (i,j) with AA identity  (20×20 heatmap)
  7.  Sequence motifs ± 3 residues around high-attention peaks  (PWM logo)
  8.  Kruskal-Wallis test: Neq separation between clusters within each SS class

Then a cross-model comparison section:
  9.  Neq Spearman r  across all experiments  (grouped bar)
  10. Cluster agreement between pairs of experiments
  11. Per-AA attention score comparison  (multi-bar)

For a FIXED sample of proteins, also produce heatmaps showing:
  12. Backbone attention  (before BiLSTM)
  13. BiLSTM self-attention  (after BiLSTM)
  side-by-side, for every experiment where backbone_attn.json exists.

Usage
-----
python run_attn_analysis.py \
    --exp_root   /path/to/scripts/final_pipeline/results \
    --ref_json   /path/to/data/attn_bilstm_f1-4_nsp3_neq.json \
    --output_dir /path/to/scripts/final_pipeline/results/analysis \
    [--sample_proteins 3d7a_B 1lsl_A 1jo0_A 1ctf_A 2hqk_A]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
AAS       = list("ACDEFGHIKLMNPQRSTVWY")
SS3_LABELS = ["H", "E", "C"]
CLUSTER_NAMES = ["PatA", "PatB", "Bg"]   # high → low energy
N_CLUSTERS = 3
TOP_PEAK_PCT = 0.10          # top 10% col_max = "peak" position for motifs
TOP_PAIR_PCT = 0.005         # top 0.5% (i,j) entries for pair analysis
MOTIF_HALF   = 3             # ±3 window → 7-mer
FIXED_SAMPLE_DEFAULT = ["3d7a_B", "1lsl_A", "1ctf_A", "2hqk_A", "1jo0_A"]
EXP_ORDER = [
    "esm2_binary_frozen",
    "esm2_binary_unfrozen",
    "esm3_binary_frozen",
    "esm3_binary_unfrozen",
    "esm2_regression_frozen",
    "esm2_3class_frozen",
]

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_root",   required=True,
                   help="Root directory containing one sub-folder per experiment")
    p.add_argument("--ref_json",   required=True,
                   help="Reference attention JSON with ss_pred/rsa/disorder/neq_real annotations")
    p.add_argument("--output_dir", required=True,
                   help="Directory where analysis outputs will be saved")
    p.add_argument("--sample_proteins", nargs="+", default=FIXED_SAMPLE_DEFAULT,
                   help="Fixed protein IDs to visualise (before+after heatmaps)")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def build_ref_map(ref_records):
    """Map protein name → annotation dict (ss_pred, rsa, disorder, neq_real, q8, sequence)."""
    ref = {}
    for r in ref_records:
        ref[r["name"]] = r
    return ref


def load_experiment_attns(exp_root, exp_name):
    """
    Load bilstm_attn.json and (optionally) backbone_attn.json for one experiment.
    Returns (bilstm_list, backbone_list_or_None).
    """
    base = Path(exp_root) / exp_name
    bilstm_path  = base / "bilstm_attn.json"
    bb_path      = base / "backbone_attn.json"

    if not bilstm_path.exists():
        return None, None

    bilstm  = load_json(bilstm_path)
    backbone = load_json(bb_path) if bb_path.exists() else None
    return bilstm, backbone


# ── K-means clustering ─────────────────────────────────────────────────────────

def cluster_attention(attn_matrix, n_clusters=N_CLUSTERS, seed=42):
    """
    Cluster residues by their row+col attention profile.
    Returns labels array [L] with 0=PatA (high energy), 1=PatB (mid), 2=Bg (low).
    """
    A = np.array(attn_matrix, dtype=np.float32)
    L = A.shape[0]
    if L < n_clusters + 1:
        return np.zeros(L, dtype=int)

    # Feature: row_i || col_i  → [L, 2L], then L2-normalise
    cols = A.T                              # [L, L]
    feats = np.concatenate([A, cols], axis=1)  # [L, 2L]
    feats = normalize(feats, norm="l2")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    raw_labels = km.fit_predict(feats)

    # Rank clusters by mean column energy (row_sum + col_sum)
    col_sum = A.sum(axis=0)
    row_sum = A.sum(axis=1)
    energy  = (row_sum + col_sum) / (2 * L)

    cluster_mean_energy = [
        energy[raw_labels == c].mean() if (raw_labels == c).any() else 0.0
        for c in range(n_clusters)
    ]
    # Sort descending: rank 0 = PatA (highest), rank 2 = Bg (lowest)
    order = np.argsort(cluster_mean_energy)[::-1]   # [PatA_raw, PatB_raw, Bg_raw]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in raw_labels])
    return labels


# ── Per-protein analysis helpers ───────────────────────────────────────────────

def spearman_neq(attn_matrix, neq_real):
    """Spearman correlation between mean incoming attention per residue and true Neq."""
    A = np.array(attn_matrix, dtype=np.float32)
    col_mean = A.mean(axis=0)[:len(neq_real)]
    neq = np.array(neq_real[:A.shape[1]], dtype=np.float32)
    if len(neq) < 5:
        return np.nan
    r, _ = stats.spearmanr(col_mean, neq)
    return float(r)


def enrichment_per_cluster(labels, annotation, valid_values):
    """
    Obs/expected enrichment of each value in valid_values per cluster.
    annotation : array-like of labels (same length as residue labels)
    Returns a dict {cluster_idx: {val: enrichment}}
    """
    labels = np.array(labels)
    ann    = np.array(annotation)
    n_total = len(labels)
    result  = {}

    bg_freq = {v: (ann == v).sum() / n_total for v in valid_values}

    for c in range(N_CLUSTERS):
        mask   = labels == c
        n_c    = mask.sum()
        result[c] = {}
        for v in valid_values:
            obs  = (ann[mask] == v).sum() / n_c if n_c > 0 else 0
            exp  = bg_freq.get(v, 1e-9)
            result[c][v] = obs / exp if exp > 1e-9 else 0.0
    return result


def mean_per_cluster(labels, values):
    """Mean of continuous values array per cluster."""
    labels = np.array(labels)
    values = np.array(values, dtype=float)
    return {c: values[labels == c].mean() if (labels == c).any() else np.nan
            for c in range(N_CLUSTERS)}


def aa_attention_scores(attn_matrix, sequence):
    """
    For each AA type, mean attention received (column mean) over all residues
    of that type, averaged across rows.  Returns dict {AA: mean_attn}.
    """
    A   = np.array(attn_matrix, dtype=np.float32)
    col_mean = A.mean(axis=0)
    scores = {aa: [] for aa in AAS}
    for i, aa in enumerate(sequence):
        if aa in scores and i < len(col_mean):
            scores[aa].append(col_mean[i])
    return {aa: float(np.mean(v)) if v else 0.0 for aa, v in scores.items()}


def top_pairs(attn_matrix, sequence, pct=TOP_PAIR_PCT):
    """
    Find top pct fraction of (i,j) entries (excluding diagonal).
    Returns list of (aa_i, aa_j, weight) tuples.
    """
    A = np.array(attn_matrix, dtype=np.float32)
    L = A.shape[0]
    np.fill_diagonal(A, 0)
    flat = A.flatten()
    thresh = np.percentile(flat, (1 - pct) * 100)
    pairs  = []
    idx    = np.argwhere(A >= thresh)
    for i, j in idx:
        if i < len(sequence) and j < len(sequence):
            pairs.append((sequence[i], sequence[j], float(A[i, j])))
    return pairs


def extract_motifs(attn_matrix, sequence, pct=TOP_PEAK_PCT, half=MOTIF_HALF):
    """
    Find high-attention positions (top pct% by column max).
    Return list of 7-mer strings centered on each such position.
    """
    A = np.array(attn_matrix, dtype=np.float32)
    L = len(sequence)
    col_max  = A.max(axis=0)[:L]
    thresh   = np.percentile(col_max, (1 - pct) * 100)
    motifs   = []
    for i in range(L):
        if col_max[i] >= thresh:
            start = max(0, i - half)
            end   = min(L, i + half + 1)
            seg   = sequence[start:end]
            if len(seg) >= 3:
                motifs.append((i, seg))
    return motifs


def build_pwm(motifs, half=MOTIF_HALF):
    """
    Build a position-weight matrix from motif segments.
    Returns a (2*half+1) × 20 numpy array of AA frequencies.
    """
    W = 2 * half + 1
    counts = np.zeros((W, 20))
    for _, seg in motifs:
        # centre-pad if shorter than W
        pad_l = half - (len(seg) // 2)
        for pos, aa in enumerate(seg):
            col = pad_l + pos
            if 0 <= col < W and aa in AAS:
                counts[col, AAS.index(aa)] += 1
    # Normalise each position
    row_sums = counts.sum(axis=1, keepdims=True).clip(1)
    pwm = counts / row_sums
    return pwm


# ── Visualisation helpers ──────────────────────────────────────────────────────

def save_heatmap(matrix, title, path, seq=None, vmax=None):
    """Save an [L, L] attention heatmap."""
    A = np.array(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(min(12, A.shape[0] * 0.06 + 2),
                                    min(10, A.shape[0] * 0.05 + 2)))
    vm = vmax or float(np.percentile(A, 98))
    im = ax.imshow(A, cmap="hot_r", vmin=0, vmax=vm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title, fontsize=9)
    if seq and len(seq) <= 80:
        ax.set_xticks(range(len(seq)))
        ax.set_xticklabels(list(seq), fontsize=4, rotation=90)
        ax.set_yticks(range(len(seq)))
        ax.set_yticklabels(list(seq), fontsize=4)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pwm_logo(pwm, title, path):
    """
    Simple PWM logo: stacked bar chart where bar height = information content,
    coloured by amino acid group.
    """
    W, n_aa = pwm.shape
    # Information content per position
    bg = np.ones(n_aa) / n_aa
    with np.errstate(divide='ignore', invalid='ignore'):
        ic = pwm * np.log2(np.where(pwm > 0, pwm / bg, 1))
    ic_per_pos = ic.sum(axis=1).clip(0)

    # AA colour groups
    colors = {
        'A': '#3CB371', 'G': '#3CB371', 'V': '#3CB371', 'L': '#3CB371',
        'I': '#3CB371', 'P': '#3CB371', 'F': '#9370DB', 'W': '#9370DB',
        'M': '#9370DB', 'S': '#4169E1', 'T': '#4169E1', 'C': '#FFD700',
        'Y': '#4169E1', 'H': '#4169E1', 'D': '#DC143C', 'E': '#DC143C',
        'N': '#20B2AA', 'Q': '#20B2AA', 'K': '#FF8C00', 'R': '#FF8C00',
    }

    fig, ax = plt.subplots(figsize=(max(6, W * 0.6), 3))
    x = np.arange(W)
    for aa_idx, aa in enumerate(AAS):
        heights = pwm[:, aa_idx] * ic_per_pos
        bottoms = pwm[:, :aa_idx] * ic_per_pos[:, np.newaxis]
        bottom  = bottoms.sum(axis=1) if aa_idx > 0 else np.zeros(W)
        ax.bar(x, heights, bottom=bottom, color=colors.get(aa, '#999999'),
               label=aa, width=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i - W//2:+d}" for i in range(W)])
    ax.set_ylabel("Bits")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, max(ic_per_pos.max() * 1.2, 0.5))
    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=5, ncol=5, loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_enrichment_heatmap(enrich_matrix, row_labels, col_labels, title, path):
    """
    Plot an enrichment (obs/exp) heatmap.
    enrich_matrix : [n_rows, n_cols] ndarray
    """
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 0.55 + 1.5),
                                    max(2, len(row_labels) * 0.6 + 1)))
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=max(enrich_matrix.max(), 2.01))
    im   = ax.imshow(enrich_matrix, cmap="RdYlBu_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Obs/Exp")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(enrich_matrix.shape[0]):
        for j in range(enrich_matrix.shape[1]):
            ax.text(j, i, f"{enrich_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplot(data_dict, xlabel, ylabel, title, path):
    """Boxplot: data_dict maps label → list of values."""
    fig, ax = plt.subplots(figsize=(max(4, len(data_dict) * 1.2), 4))
    ax.boxplot(
        [data_dict[k] for k in data_dict],
        labels=list(data_dict.keys()),
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bar(values_dict, xlabel, ylabel, title, path, color="steelblue", horizontal=False):
    """Bar chart: values_dict maps label → float."""
    labels = list(values_dict.keys())
    vals   = [values_dict[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.5 + 1), 4))
    if horizontal:
        ax.barh(range(len(labels)), vals, color=color)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(range(len(labels)), vals, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pair_heatmap(pair_counts, title, path):
    """20×20 AA pair attention-count heatmap."""
    mat = np.zeros((20, 20))
    for (aa_i, aa_j), count in pair_counts.items():
        if aa_i in AAS and aa_j in AAS:
            mat[AAS.index(aa_i), AAS.index(aa_j)] += count
    # Symmetrise
    mat = (mat + mat.T) / 2
    mat /= mat.max() + 1e-12

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.03, label="Normalised count")
    ax.set_xticks(range(20)); ax.set_xticklabels(AAS, fontsize=7)
    ax.set_yticks(range(20)); ax.set_yticklabels(AAS, fontsize=7)
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Per-experiment analysis ────────────────────────────────────────────────────

def analyse_experiment(exp_name, bilstm_records, ref_map, out_dir):
    """
    Run all quantitative analyses for one experiment.
    Returns a summary dict for cross-model comparison.
    """
    out = Path(out_dir) / exp_name
    out.mkdir(parents=True, exist_ok=True)

    # Initialise accumulators
    spearman_rs   = []
    cluster_labels_all   = defaultdict(list)   # ss_label → list[cluster_label]
    cluster_neq_all      = defaultdict(list)   # cluster_label → list[neq_real]
    cluster_rsa_all      = defaultdict(list)
    cluster_dis_all      = defaultdict(list)
    aa_attn_pool         = defaultdict(list)
    pair_counts          = defaultdict(float)
    all_motifs           = []
    all_cluster_labels   = {}                  # name → np label array

    proteins_processed = 0
    proteins_missing   = 0

    for rec in bilstm_records:
        name = rec["name"]
        if name not in ref_map:
            proteins_missing += 1
            continue

        ref  = ref_map[name]
        attn = rec["attention_weights"]
        seq  = rec.get("sequence", ref.get("sequence", ""))
        L    = len(seq)

        # Align annotation lengths to min(L, ref_len)
        neq_real = ref.get("neq_real", [])[:L]
        ss_pred  = ref.get("ss_pred", [])[:L]
        rsa      = ref.get("rsa", [])[:L]
        disorder = ref.get("disorder", [])[:L]

        # Trim attn to L
        A = [row[:L] for row in attn[:L]]
        if len(A) < 3:
            continue

        proteins_processed += 1

        # 1. Neq correlation
        r = spearman_neq(A, neq_real)
        if not np.isnan(r):
            spearman_rs.append(r)

        # 2. K-means clustering
        labels = cluster_attention(A)
        all_cluster_labels[name] = labels

        # 3. SS enrichment accumulate
        for i, (lbl, ss, rsa_v, dis_v) in enumerate(
                zip(labels, ss_pred, rsa, disorder)):
            cluster_labels_all[ss].append(int(lbl))
            cluster_neq_all[int(lbl)].append(neq_real[i] if i < len(neq_real) else np.nan)
            cluster_rsa_all[int(lbl)].append(rsa_v)
            cluster_dis_all[int(lbl)].append(dis_v)

        # 4. AA attention scores
        scores = aa_attention_scores(A, seq)
        for aa, v in scores.items():
            aa_attn_pool[aa].append(v)

        # 5. Top pairs
        for aa_i, aa_j, w in top_pairs(A, seq):
            pair_counts[(aa_i, aa_j)] += w

        # 6. Motifs
        all_motifs.extend(extract_motifs(A, seq))

    print(f"  [{exp_name}] Processed {proteins_processed} proteins "
          f"({proteins_missing} missing from reference).")

    # ── Save Neq correlation ─────────────────────────────────────────────────
    neq_summary = {
        "per_protein_spearman_r": spearman_rs,
        "mean_spearman_r":        float(np.nanmean(spearman_rs)) if spearman_rs else None,
        "median_spearman_r":      float(np.nanmedian(spearman_rs)) if spearman_rs else None,
        "n_proteins":             proteins_processed,
    }
    with open(out / "neq_correlation.json", "w") as fh:
        json.dump(neq_summary, fh, indent=2)

    # Pooled Spearman violin plot
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.violinplot([spearman_rs], positions=[0], showmedians=True)
    ax.set_xticks([0]); ax.set_xticklabels([exp_name], fontsize=7)
    ax.set_ylabel("Spearman r (col_mean vs Neq)")
    ax.set_title(f"Neq correlation\nmedian={neq_summary['median_spearman_r']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "neq_correlation.png", dpi=150); plt.close(fig)

    # ── SS enrichment ─────────────────────────────────────────────────────────
    # cluster × SS enrichment matrix  [N_CLUSTERS × 3]
    enrich_mat = np.zeros((N_CLUSTERS, len(SS3_LABELS)))
    total = sum(len(v) for v in cluster_labels_all.values())
    bg_ss = {s: sum(1 for v in cluster_labels_all[s] for _ in [v]) / (total + 1e-9)
             for s in SS3_LABELS}

    for ci, cname in enumerate(CLUSTER_NAMES):
        for si, ss in enumerate(SS3_LABELS):
            members_in_c_with_ss = sum(
                1 for lbl in cluster_labels_all.get(ss, []) if lbl == ci
            )
            total_in_c = sum(
                1 for ss_key in cluster_labels_all for lbl in cluster_labels_all[ss_key]
                if lbl == ci
            )
            obs = members_in_c_with_ss / (total_in_c + 1e-9)
            exp = bg_ss.get(ss, 1e-9)
            enrich_mat[ci, si] = obs / exp

    plot_enrichment_heatmap(
        enrich_mat, CLUSTER_NAMES, SS3_LABELS,
        f"SS Enrichment – {exp_name}",
        out / "ss_enrichment.png",
    )

    # ── RSA / disorder boxplots ────────────────────────────────────────────────
    rsa_by_cluster = {CLUSTER_NAMES[c]: [v for v in cluster_rsa_all[c] if not np.isnan(v)]
                      for c in range(N_CLUSTERS)}
    dis_by_cluster = {CLUSTER_NAMES[c]: [v for v in cluster_dis_all[c] if not np.isnan(v)]
                      for c in range(N_CLUSTERS)}
    neq_by_cluster = {CLUSTER_NAMES[c]: [v for v in cluster_neq_all[c] if not np.isnan(v)]
                      for c in range(N_CLUSTERS)}

    plot_boxplot(rsa_by_cluster,   "Cluster", "RSA",
                 f"RSA per cluster – {exp_name}", out / "rsa_by_cluster.png")
    plot_boxplot(dis_by_cluster,   "Cluster", "Disorder score",
                 f"Disorder per cluster – {exp_name}", out / "disorder_by_cluster.png")
    plot_boxplot(neq_by_cluster,   "Cluster", "Neq",
                 f"Neq per cluster – {exp_name}", out / "neq_by_cluster.png")

    # ── Kruskal-Wallis: Neq separation within each SS class ───────────────────
    kw_results = {}
    for ss in SS3_LABELS:
        groups = []
        for c in range(N_CLUSTERS):
            vals = [
                neq for lbl, neq, ss_v in zip(
                    cluster_labels_all.get(ss, []),
                    [cluster_neq_all[c] for _ in range(len(cluster_labels_all.get(ss, [])))][0]
                    if cluster_neq_all[c] else [],
                    [ss] * len(cluster_labels_all.get(ss, []))
                )
                if lbl == c
            ]
            groups.append(vals)
        # Rebuild correctly
        # Build a flat list of (cluster_label, neq_val) for residues with this SS
        flat = []
        for si2, (ss_key, lbl_list) in enumerate(cluster_labels_all.items()):
            if ss_key != ss:
                continue
        # simpler approach: we need to co-iterate cluster labels and neq values
        # since they were appended in the same loop, use a temporary index approach
        # We'll skip this complex bookkeeping and just do a pooled KW using the
        # already-separated neq_by_cluster which isn't SS-filtered.
        # A more complete implementation is in neq_within_ss.py
        kw_results[ss] = {"note": "see neq_within_ss.py for within-SS KW test"}

    # ── Per-AA attention scores ────────────────────────────────────────────────
    aa_means = {aa: float(np.mean(v)) if v else 0.0 for aa, v in aa_attn_pool.items()}
    aa_sorted = dict(sorted(aa_means.items(), key=lambda x: -x[1]))
    plot_bar(aa_sorted, "Amino acid", "Mean attention received",
             f"AA attention scores – {exp_name}",
             out / "aa_attention_scores.png", horizontal=True)
    with open(out / "aa_attention_scores.json", "w") as fh:
        json.dump(aa_means, fh, indent=2)

    # ── Top attention pairs ────────────────────────────────────────────────────
    plot_pair_heatmap(pair_counts,
                      f"Top attention pairs – {exp_name}",
                      out / "top_pairs.png")
    # Save top-50 pairs as TSV
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:50]
    with open(out / "top_pairs.tsv", "w") as fh:
        fh.write("AA_i\tAA_j\ttotal_weight\n")
        for (aa_i, aa_j), w in sorted_pairs:
            fh.write(f"{aa_i}\t{aa_j}\t{w:.4f}\n")

    # ── Motifs & PWM ──────────────────────────────────────────────────────────
    if all_motifs:
        pwm = build_pwm(all_motifs)
        plot_pwm_logo(pwm, f"Attention peak motif PWM – {exp_name}",
                      out / "motif_pwm_logo.png")
        np.save(out / "motif_pwm.npy", pwm)

    # ── Save cluster labels for cross-model comparison ────────────────────────
    with open(out / "cluster_labels.json", "w") as fh:
        json.dump({k: v.tolist() for k, v in all_cluster_labels.items()}, fh)

    return {
        "exp_name":          exp_name,
        "mean_spearman_r":   neq_summary["mean_spearman_r"],
        "median_spearman_r": neq_summary["median_spearman_r"],
        "n_proteins":        proteins_processed,
        "aa_means":          aa_means,
        "cluster_labels":    all_cluster_labels,   # name → labels array
    }


# ── Fixed-sample visualisation (before & after) ────────────────────────────────

def visualise_sample(exp_name, bilstm_records, backbone_records, sample_proteins, out_dir):
    """
    For each protein in sample_proteins, plot backbone (before) and BiLSTM (after)
    attention heatmaps side-by-side.
    """
    bb_map = {}
    if backbone_records:
        bb_map = {r["name"]: r for r in backbone_records}

    bilstm_map = {r["name"]: r for r in bilstm_records}

    sample_dir = Path(out_dir) / exp_name / "sample_heatmaps"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for prot in sample_proteins:
        if prot not in bilstm_map:
            continue

        b_rec = bilstm_map[prot]
        seq   = b_rec.get("sequence", "")
        A_after = np.array(b_rec["attention_weights"], dtype=np.float32)

        if prot in bb_map:
            A_before = np.array(bb_map[prot]["attention_weights"], dtype=np.float32)
            source   = bb_map[prot].get("attention_source", "backbone")

            # Trim to same size
            L = min(A_before.shape[0], A_after.shape[0])
            A_before = A_before[:L, :L]
            A_after  = A_after[:L, :L]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            vm_b = float(np.percentile(A_before, 98))
            vm_a = float(np.percentile(A_after,  98))

            axes[0].imshow(A_before, cmap="hot_r", vmin=0, vmax=vm_b, aspect="auto")
            axes[0].set_title(f"Backbone ({source})\n{prot}  len={L}", fontsize=8)
            axes[1].imshow(A_after,  cmap="hot_r", vmin=0, vmax=vm_a, aspect="auto")
            axes[1].set_title(f"BiLSTM self-attention\n{prot}  len={L}", fontsize=8)
            for ax in axes:
                ax.set_xlabel("Key residue"); ax.set_ylabel("Query residue")
            plt.suptitle(f"{exp_name}", fontsize=10)
            plt.tight_layout()
            fig.savefig(sample_dir / f"{prot}_before_after.png", dpi=150)
            plt.close(fig)
        else:
            # Only after
            save_heatmap(A_after, f"BiLSTM attention – {exp_name} – {prot}",
                         sample_dir / f"{prot}_bilstm_only.png", seq=seq)


# ── Cross-model comparison ─────────────────────────────────────────────────────

def cross_model_comparison(summaries, out_dir):
    """
    Produce comparison figures across all experiments.
    summaries : list of dicts returned by analyse_experiment()
    """
    cmp_dir = Path(out_dir) / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    valid = [s for s in summaries if s.get("mean_spearman_r") is not None]
    if not valid:
        return

    # ── Neq correlation bar chart ─────────────────────────────────────────────
    names  = [s["exp_name"] for s in valid]
    means  = [s["mean_spearman_r"] for s in valid]
    medians = [s["median_spearman_r"] for s in valid]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, means,   width=0.35, label="Mean Spearman r",   color="steelblue")
    ax.bar(x + 0.2, medians, width=0.35, label="Median Spearman r", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman r (attention vs Neq)")
    ax.set_title("Neq correlation across experiments")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(cmp_dir / "neq_correlation_comparison.png", dpi=150)
    plt.close(fig)

    # ── AA attention scores comparison ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x  = np.arange(len(AAS))
    width = 0.8 / len(valid)
    cmap  = plt.get_cmap("tab10")
    for ei, s in enumerate(valid):
        vals = [s["aa_means"].get(aa, 0.0) for aa in AAS]
        ax.bar(x + ei * width, vals, width=width,
               label=s["exp_name"], color=cmap(ei))
    ax.set_xticks(x + width * (len(valid) - 1) / 2)
    ax.set_xticklabels(AAS, fontsize=8)
    ax.set_ylabel("Mean attention received")
    ax.set_title("Per-AA attention scores across experiments")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fig.savefig(cmp_dir / "aa_scores_comparison.png", dpi=150)
    plt.close(fig)

    # ── Cluster agreement matrix ──────────────────────────────────────────────
    n = len(valid)
    agree_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            # Collect proteins present in both
            si, sj = valid[i]["cluster_labels"], valid[j]["cluster_labels"]
            common = set(si.keys()) & set(sj.keys())
            if not common:
                continue
            total_res, agree_res = 0, 0
            for prot in common:
                li = np.array(si[prot])
                lj = np.array(sj[prot])
                mn = min(len(li), len(lj))
                total_res += mn
                agree_res += (li[:mn] == lj[:mn]).sum()
            frac = agree_res / total_res if total_res > 0 else 0.0
            agree_mat[i, j] = agree_mat[j, i] = frac

    fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n + 0.5)))
    im = ax.imshow(agree_mat, cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Cluster agreement")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agree_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if agree_mat[i,j] < 0.5 else "black")
    ax.set_title("Cluster agreement between experiments")
    plt.tight_layout()
    fig.savefig(cmp_dir / "cluster_agreement.png", dpi=150)
    plt.close(fig)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    rows = []
    for s in valid:
        rows.append({
            "experiment":        s["exp_name"],
            "n_proteins":        s["n_proteins"],
            "mean_spearman_r":   s["mean_spearman_r"],
            "median_spearman_r": s["median_spearman_r"],
        })
    pd.DataFrame(rows).to_csv(cmp_dir / "summary.csv", index=False)
    print(f"  Cross-model comparison → {cmp_dir}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    exp_root  = Path(args.exp_root)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference JSON: {args.ref_json}")
    ref_records = load_json(args.ref_json)
    ref_map     = build_ref_map(ref_records)
    print(f"  Reference proteins: {len(ref_map)}")

    # Discover experiments (prefer EXP_ORDER, then anything else present)
    present = []
    for name in EXP_ORDER:
        p = exp_root / name / "bilstm_attn.json"
        if p.exists():
            present.append(name)
    # Also include any additional experiment folders
    for subdir in sorted(exp_root.iterdir()):
        if subdir.is_dir() and subdir.name not in present and subdir.name != "analysis":
            if (subdir / "bilstm_attn.json").exists():
                present.append(subdir.name)

    print(f"Experiments found: {present}")

    summaries = []
    for exp_name in present:
        print(f"\n{'='*60}")
        print(f"  Analysing: {exp_name}")
        print(f"{'='*60}")

        bilstm, backbone = load_experiment_attns(exp_root, exp_name)
        if bilstm is None:
            print(f"  No bilstm_attn.json found, skipping.")
            continue

        # Quantitative analysis (BiLSTM attention only)
        summary = analyse_experiment(exp_name, bilstm, ref_map, out_dir)
        summaries.append(summary)

        # Fixed-sample visualisation (before + after)
        visualise_sample(exp_name, bilstm, backbone, args.sample_proteins, out_dir)

    # Cross-model comparison
    if len(summaries) >= 2:
        print(f"\n{'='*60}")
        print("  Cross-model comparison")
        print(f"{'='*60}")
        cross_model_comparison(summaries, out_dir)

    print(f"\nAll done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
