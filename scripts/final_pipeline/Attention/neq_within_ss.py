#!/usr/bin/env python3
"""
neq_within_ss.py

Tests whether attention clusters separate residues by Neq *within* each
secondary structure class (H / E / C).

Hypothesis A (ESM2-frozen):
  Clusters reflect SS as a mechanism to predict Neq.
  → Within a single SS class, clusters should lose Neq-separation power,
    because SS already explains most of the variance.

Hypothesis B (ESM3-frozen):
  SS is already encoded in the embeddings, so BiLSTM attention captures
  something *beyond* SS — likely dynamics / flexibility.
  → Clusters should retain Neq-separation power even *within* each SS class.

Method
------
For every residue:
  - Assign to a cluster (PatA / PatB / Bg) via K-means on attention matrix
  - Record SS label (H/E/C) and neq_real from the reference JSON

For each model × SS class:
  - Run Kruskal-Wallis test across the 3 cluster groups
  - Compute eta-squared (effect size)
  - Record mean Neq per cluster

Outputs
-------
  neq_within_ss_stats.csv       — KW statistic, p-value, eta2, means per cell
  neq_within_ss_boxplots.png    — Within-SS Neq distributions per cluster (2 × 3 grid)
  neq_within_ss_means.png       — Mean Neq per cluster per SS class, both models

Usage
-----
python neq_within_ss.py \\
    --esm2 esm2_attn_frozen.json \\
    --esm3 esm3_attn.json \\
    --ref  ../../data/attn_bilstm_f1-4_nsp3_neq.json \\
    --output_dir results/neq_within_ss
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        recs = json.load(f)
    return {r['name']: r for r in recs}


def cluster_labels_np(A_np, n_init=10, seed=42):
    N = A_np.shape[0]
    if N < 3:
        return None
    raw  = np.concatenate([A_np, A_np.T], axis=1)
    feat = normalize(raw, norm='l2', axis=1)
    km   = KMeans(n_clusters=3, n_init=n_init, random_state=seed, max_iter=100)
    raw_labels = km.fit_predict(feat)
    energy = (A_np.sum(axis=1) + A_np.sum(axis=0)) / N
    ranked = sorted(range(3), key=lambda c: energy[raw_labels == c].mean(), reverse=True)
    mapping = {ranked[0]: 'PatA', ranked[1]: 'PatB', ranked[2]: 'Bg'}
    return [mapping[int(l)] for l in raw_labels]


def kruskal_wallis(groups):
    """
    groups: list of arrays (one per group), all values pooled.
    Returns (H-statistic, p-value, eta-squared).
    Uses scipy if available, otherwise falls back to manual computation.
    """
    try:
        from scipy.stats import kruskal
        stat, pval = kruskal(*groups)
    except ImportError:
        # Manual Kruskal-Wallis
        all_vals = np.concatenate(groups)
        n_total  = len(all_vals)
        # rank all values
        order  = np.argsort(all_vals)
        ranks  = np.empty(n_total)
        ranks[order] = np.arange(1, n_total + 1, dtype=float)
        # tie correction: adjust ranks for ties
        # (simple version: midrank)
        i = 0
        while i < n_total:
            j = i
            while j < n_total - 1 and all_vals[order[j+1]] == all_vals[order[j]]:
                j += 1
            if j > i:
                ranks[order[i:j+1]] = (i + j) / 2.0 + 1
            i = j + 1
        stat = 0.0
        idx  = 0
        for g in groups:
            ng  = len(g)
            rg  = ranks[idx:idx+ng]
            stat += ng * (rg.mean() - (n_total + 1) / 2) ** 2
            idx += ng
        stat = 12 / (n_total * (n_total + 1)) * stat
        pval = float('nan')   # no easy chi2 without scipy

    # eta-squared (effect size)
    k = len(groups)
    n = sum(len(g) for g in groups)
    eta2 = (stat - k + 1) / (n - k) if n > k else float('nan')
    return float(stat), float(pval), float(eta2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--esm2',   required=True)
    p.add_argument('--esm3',   required=True)
    p.add_argument('--ref',    required=True,
                   help='Reference JSON with ss_pred and neq_real fields')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_init', type=int, default=10)
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    MODELS     = ['ESM2-frozen', 'ESM3-frozen']
    SS_CLASSES = ['H', 'E', 'C']
    CLUSTERS   = ['PatA', 'PatB', 'Bg']
    COLORS     = {'PatA': '#4C72B0', 'PatB': '#55A868', 'Bg': '#C44E52'}

    jsons = {
        'ESM2-frozen': load_json(args.esm2),
        'ESM3-frozen': load_json(args.esm3),
    }
    ref = load_json(args.ref)
    common = sorted(set(jsons['ESM2-frozen']) & set(jsons['ESM3-frozen']) & set(ref))
    print(f"Common proteins: {len(common)}")

    # Accumulate residue-level records: per model, per (ss, cluster) → list of neq_real
    # neq_real is continuous (from the reference JSON)
    neq_by = {
        m: {ss: {cl: [] for cl in CLUSTERS} for ss in SS_CLASSES}
        for m in MODELS
    }

    print("Processing proteins ...")
    for idx, name in enumerate(common):
        print(f"  [{idx+1}/{len(common)}] {name}", end='\r', flush=True)
        ref_rec  = ref[name]
        ss3_list = ref_rec.get('ss_pred', [])
        neq_real = ref_rec.get('neq_real', [])

        if not ss3_list or not neq_real:
            continue

        for m in MODELS:
            A_np   = np.array(jsons[m][name]['attention_weights'], dtype=np.float32)
            labels = cluster_labels_np(A_np, n_init=args.n_init, seed=args.seed)
            if labels is None:
                continue

            N = len(labels)
            for i, (cl, ss, neq) in enumerate(zip(labels, ss3_list, neq_real)):
                if cl and ss in SS_CLASSES and neq is not None:
                    neq_by[m][ss][cl].append(float(neq))

    print(f"\nDone.\n")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("=== Kruskal-Wallis: Clusters → Neq, stratified by SS ===\n")
    print(f"  {'Model':<14} {'SS':>4}  {'H':>8}  {'p':>10}  {'eta2':>7}  "
          f"{'n_PatA':>7} {'n_PatB':>7} {'n_Bg':>6}  "
          f"{'mean_PatA':>10} {'mean_PatB':>10} {'mean_Bg':>9}")
    print("  " + "-" * 105)

    stat_rows = []
    for m in MODELS:
        for ss in SS_CLASSES:
            groups = [np.array(neq_by[m][ss][cl]) for cl in CLUSTERS]
            counts = [len(g) for g in groups]
            means  = [g.mean() if len(g) > 0 else float('nan') for g in groups]
            groups_nonempty = [g for g in groups if len(g) >= 5]
            if len(groups_nonempty) < 2:
                H, pval, eta2 = float('nan'), float('nan'), float('nan')
            else:
                H, pval, eta2 = kruskal_wallis(groups_nonempty)

            print(f"  {m:<14} {ss:>4}  {H:>8.2f}  {pval:>10.2e}  {eta2:>7.4f}  "
                  f"{counts[0]:>7} {counts[1]:>7} {counts[2]:>6}  "
                  f"{means[0]:>10.4f} {means[1]:>10.4f} {means[2]:>9.4f}")

            stat_rows.append({
                'model': m, 'ss': ss,
                'KW_H': H, 'p_value': pval, 'eta_squared': eta2,
                'n_PatA': counts[0], 'n_PatB': counts[1], 'n_Bg': counts[2],
                'mean_neq_PatA': means[0],
                'mean_neq_PatB': means[1],
                'mean_neq_Bg':   means[2],
            })
        print()

    csv_path = os.path.join(args.output_dir, 'neq_within_ss_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)
    print(f"Saved → {csv_path}")

    # ── Boxplot grid: model × SS ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
    fig.suptitle('Neq distribution per attention cluster,\nstratified by secondary structure class',
                 fontsize=12)

    for row_idx, m in enumerate(MODELS):
        for col_idx, ss in enumerate(SS_CLASSES):
            ax = axes[row_idx][col_idx]
            data   = [neq_by[m][ss][cl] for cl in CLUSTERS]
            labels = [f"{cl}\n(n={len(d)})" for cl, d in zip(CLUSTERS, data)]

            # filter out empty
            nonempty = [(lbl, d) for lbl, d in zip(labels, data) if len(d) > 0]
            if nonempty:
                lbls, vals = zip(*nonempty)
                bp = ax.boxplot(vals, labels=lbls, patch_artist=True,
                                medianprops=dict(color='black', lw=2),
                                showfliers=False)
                for patch, cl in zip(bp['boxes'], CLUSTERS):
                    patch.set_facecolor(COLORS[cl])
                    patch.set_alpha(0.75)

            # annotate KW result
            row = next((r for r in stat_rows if r['model']==m and r['ss']==ss), None)
            if row:
                pstr = f"p={row['p_value']:.2e}" if not math.isnan(row['p_value']) else 'p=N/A'
                eta_str = f"η²={row['eta_squared']:.3f}" if not math.isnan(row['eta_squared']) else ''
                ax.set_title(f"{m} | SS={ss}\nKW {pstr}  {eta_str}", fontsize=9)
            ax.set_ylabel('Neq (true)' if col_idx == 0 else '')

    plt.tight_layout()
    path = os.path.join(args.output_dir, 'neq_within_ss_boxplots.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")

    # ── Mean Neq per cluster per SS: side-by-side grouped bars ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.suptitle('Mean Neq per cluster, within each SS class\n'
                 '(if ESM3 bars separate more → attention captures flexibility beyond SS)',
                 fontsize=11)

    x = np.arange(len(CLUSTERS))
    width = 0.35
    model_colors = {'ESM2-frozen': '#4C72B0', 'ESM3-frozen': '#DD8452'}

    for col_idx, ss in enumerate(SS_CLASSES):
        ax = axes[col_idx]
        for offset, m in zip([-width/2, width/2], MODELS):
            means = [
                next(r['mean_neq_' + cl] for r in stat_rows
                     if r['model'] == m and r['ss'] == ss)
                for cl in CLUSTERS
            ]
            counts = [
                next(r['n_' + cl] for r in stat_rows
                     if r['model'] == m and r['ss'] == ss)
                for cl in CLUSTERS
            ]
            bars = ax.bar(x + offset, means, width, label=m,
                          color=model_colors[m], alpha=0.8)
            # annotate n
            for bar, n in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f'n={n}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(CLUSTERS)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Neq')
        ax.set_title(f'SS = {ss}')
        if col_idx == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(args.output_dir, 'neq_within_ss_means.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")

    # ── Summary interpretation ─────────────────────────────────────────────────
    print("\n=== Summary: does cluster→Neq survive SS control? ===")
    for ss in SS_CLASSES:
        rows = {m: next(r for r in stat_rows if r['model']==m and r['ss']==ss)
                for m in MODELS}
        e2  = rows['ESM2-frozen']['eta_squared']
        e3  = rows['ESM3-frozen']['eta_squared']
        p2  = rows['ESM2-frozen']['p_value']
        p3  = rows['ESM3-frozen']['p_value']
        verdict = ''
        if not math.isnan(e2) and not math.isnan(e3):
            if e3 > e2 * 1.2:
                verdict = '→ ESM3 clusters carry MORE Neq signal within this SS class'
            elif e2 > e3 * 1.2:
                verdict = '→ ESM2 clusters carry more Neq signal'
            else:
                verdict = '→ Both similar'
        sig2 = 'sig' if (not math.isnan(p2) and p2 < 0.05) else 'ns'
        sig3 = 'sig' if (not math.isnan(p3) and p3 < 0.05) else 'ns'
        print(f"  SS={ss}: ESM2 η²={e2:.4f}({sig2})  ESM3 η²={e3:.4f}({sig3})  {verdict}")


if __name__ == '__main__':
    main()
