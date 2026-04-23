#!/usr/bin/env python3
"""
compare_three_attentions.py

Compare attention matrices from three models:
  - ESM2-cls  : ESM2 backbone, BiLSTM classifier (original)
  - ESM3-cls  : ESM3 backbone, BiLSTM classifier
  - ESM2-reg  : ESM2 backbone, BiLSTM regression

For each model, and for all 277 test proteins, this script computes:
  1. Raw attention statistics (mean, max, sparsity, diagonal dominance)
  2. K-means cluster assignments (PatA / PatB / Bg) using the same
     cluster_attention_residues.py logic used elsewhere
  3. SS3 enrichment per cluster (using ss_pred from the ESM2-cls JSON,
     which has NetSurfP annotations — the same proteins appear in all three)
  4. Neq-pred vs. neq-real correlation (where available)
  5. Cluster overlap: how often do two models agree on the cluster of a residue

Outputs (all in --output_dir):
  attention_stats.csv         — per-model raw stats
  cluster_fractions.csv       — PatA/PatB/Bg fractions per model
  ss_enrichment.csv           — enrichment of each SS in each cluster × model
  neq_corr.csv                — Spearman(neq_preds, neq_real) per model
  cluster_overlap.csv         — pairwise residue-level cluster agreement (%)
  attention_energy_violin.png — attention energy distributions side-by-side
  cluster_agreement_bar.png   — pairwise cluster overlap bar chart

Usage
-----
python compare_three_attentions.py \\
    --esm2_cls  ../../data/attn_bilstm_f1-4_nsp3_neq.json \\
    --esm3_cls  esm3_attn.json \\
    --esm2_reg  regression_bilstm_attn_unfrozen.json \\
    --output_dir results/three_model_comparison
"""

import argparse
import json
import math
import os
import sys
import random
import csv

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


def spearman_r(x, y):
    """Spearman correlation between two equal-length lists (no scipy needed)."""
    n = len(x)
    if n < 3:
        return float('nan')
    def rank(v):
        sv = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and v[sv[j+1]] == v[sv[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                r[sv[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mean_rx) ** 2 for i in range(n)) *
        sum((ry[i] - mean_ry) ** 2 for i in range(n))
    )
    return num / den if den > 1e-12 else float('nan')


def attn_stats(A):
    """Return dict of scalar statistics for an NxN attention matrix."""
    N = len(A)
    flat = [A[i][j] for i in range(N) for j in range(N)]
    diag = [A[i][i] for i in range(N)]
    off  = [A[i][j] for i in range(N) for j in range(N) if i != j]
    # sparsity: fraction of entries below 0.01
    n_sparse = sum(1 for v in flat if v < 0.01)
    # col_max distribution
    col_maxes = [max(A[j][i] for j in range(N)) for i in range(N)]
    return {
        'mean':            sum(flat) / len(flat),
        'max':             max(flat),
        'diag_mean':       sum(diag) / N,
        'offdiag_mean':    sum(off) / len(off) if off else 0.0,
        'sparsity_frac':   n_sparse / len(flat),
        'col_max_mean':    sum(col_maxes) / N,
        'col_max_std':     math.sqrt(sum((v - sum(col_maxes)/N)**2 for v in col_maxes) / N),
    }


def cluster_labels_for(rec, n_init=10, seed=42):
    """Return per-residue cluster label list ['PatA'|'PatB'|'Bg'] for one record.

    Features: L2-normalised row+col concatenation (same as cluster_attention_residues.py).
    Clusters ranked by mean attention energy (row_sum + col_sum):
      highest → PatA, middle → PatB, lowest → Bg.
    """
    A = np.array(rec['attention_weights'], dtype=np.float32)
    N = A.shape[0]
    if N < 3:
        return None
    # Build 2N-dim features: row_i || col_i
    raw = np.concatenate([A, A.T], axis=1)          # (N, 2N)
    feat = normalize(raw, norm='l2', axis=1)         # (N, 2N) L2-normalised rows

    km = KMeans(n_clusters=3, n_init=n_init, random_state=seed, max_iter=100)
    raw_labels = km.fit_predict(feat)                # int array (N,)

    # Energy per cluster: mean of (row_sum + col_sum) for members
    row_sum = A.sum(axis=1)
    col_sum = A.sum(axis=0)
    energy  = (row_sum + col_sum) / N

    cluster_energy = {c: energy[raw_labels == c].mean() for c in range(3)}
    ranked = sorted(cluster_energy, key=cluster_energy.get, reverse=True)
    # ranked[0]=highest energy → PatA, ranked[1]→PatB, ranked[2]→Bg
    mapping = {ranked[0]: 'PatA', ranked[1]: 'PatB', ranked[2]: 'Bg'}
    return [mapping[int(l)] for l in raw_labels]


def enrichment_table(counts, row_labels, col_labels):
    """obs/expected enrichment."""
    grand = sum(counts.get((r, c), 0) for r in row_labels for c in col_labels)
    if grand == 0:
        return {}
    row_tot = {r: sum(counts.get((r, c), 0) for c in col_labels) for r in row_labels}
    col_tot = {c: sum(counts.get((r, c), 0) for r in row_labels) for c in col_labels}
    enr = {}
    for r in row_labels:
        for c in col_labels:
            obs = counts.get((r, c), 0)
            exp = row_tot[r] * col_tot[c] / grand
            enr[(r, c)] = obs / exp if exp > 0 else float('nan')
    return enr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--esm2_cls', required=True,
                   help='Path to ESM2-classifier attention JSON (has ss_pred, neq_real)')
    p.add_argument('--esm3_cls', required=True,
                   help='Path to ESM3-classifier attention JSON')
    p.add_argument('--esm2_reg', required=True,
                   help='Path to ESM2-regression attention JSON')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_init', type=int, default=10)
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    MODEL_NAMES = ['ESM2-cls', 'ESM3-cls', 'ESM2-reg']
    jsons = {
        'ESM2-cls': load_json(args.esm2_cls),
        'ESM3-cls': load_json(args.esm3_cls),
        'ESM2-reg': load_json(args.esm2_reg),
    }

    # Common protein names
    common_names = sorted(
        set(jsons['ESM2-cls']) & set(jsons['ESM3-cls']) & set(jsons['ESM2-reg'])
    )
    print(f"Common proteins: {len(common_names)}")

    # ── 1. Raw attention statistics ──────────────────────────────────────────
    print("\n=== 1. Raw Attention Statistics ===")
    stat_keys = ['mean', 'max', 'diag_mean', 'offdiag_mean',
                 'sparsity_frac', 'col_max_mean', 'col_max_std']
    stat_accum = {m: {k: [] for k in stat_keys} for m in MODEL_NAMES}

    for name in common_names:
        for m in MODEL_NAMES:
            s = attn_stats(jsons[m][name]['attention_weights'])
            for k in stat_keys:
                stat_accum[m][k].append(s[k])

    stat_rows = []
    for m in MODEL_NAMES:
        row = {'model': m}
        for k in stat_keys:
            vals = stat_accum[m][k]
            row[k + '_mean'] = sum(vals) / len(vals)
            row[k + '_std']  = math.sqrt(sum((v - row[k+'_mean'])**2 for v in vals) / len(vals))
        stat_rows.append(row)

    stat_cols = ['model'] + [k + sfx for k in stat_keys for sfx in ('_mean', '_std')]
    csv_path = os.path.join(args.output_dir, 'attention_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=stat_cols)
        w.writeheader()
        w.writerows(stat_rows)
    print(f"  Saved → {csv_path}")

    # Print summary table
    print(f"\n  {'Model':<12} {'mean':>8} {'max':>8} {'diag':>8} {'sparse%':>9} {'col_max_mean':>13}")
    print("  " + "-" * 60)
    for row in stat_rows:
        print(f"  {row['model']:<12} "
              f"{row['mean_mean']:>8.5f} "
              f"{row['max_mean']:>8.4f} "
              f"{row['diag_mean_mean']:>8.5f} "
              f"{row['sparsity_frac_mean']*100:>8.1f}% "
              f"{row['col_max_mean_mean']:>13.5f}")

    # ── 2. Clustering ─────────────────────────────────────────────────────────
    print("\n=== 2. K-means Clustering (K=3) ===")
    cluster_labels = {m: {} for m in MODEL_NAMES}   # name -> list of labels
    CLUSTER_LBLS = ['PatA', 'PatB', 'Bg']
    SS3_LBLS     = ['H', 'E', 'C']

    frac_accum   = {m: {'PatA': [], 'PatB': [], 'Bg': []} for m in MODEL_NAMES}
    # SS3 contingency: model -> (cluster, ss) -> count
    ct_ss = {m: {(cl, ss): 0 for cl in CLUSTER_LBLS for ss in SS3_LBLS}
             for m in MODEL_NAMES}
    # Neq pred vs real
    neq_corrs = {m: [] for m in MODEL_NAMES}

    for idx, name in enumerate(common_names):
        print(f"  [{idx+1}/{len(common_names)}] {name}", end='', flush=True)

        # ss_pred and neq_real come from the ESM2-cls JSON (only one with annotations)
        ref_rec    = jsons['ESM2-cls'][name]
        ss3_list   = ref_rec.get('ss_pred', [])
        neq_real   = ref_rec.get('neq_real', [])

        for m in MODEL_NAMES:
            rec    = jsons[m][name]
            labels = cluster_labels_for(rec, n_init=args.n_init, seed=args.seed)
            if labels is None:
                print(f" SKIP-{m}", end='')
                continue
            cluster_labels[m][name] = labels
            N = len(labels)

            # Cluster fractions
            n_A  = labels.count('PatA')
            n_B  = labels.count('PatB')
            n_Bg = labels.count('Bg')
            frac_accum[m]['PatA'].append(n_A  / N)
            frac_accum[m]['PatB'].append(n_B  / N)
            frac_accum[m]['Bg'].append(n_Bg / N)

            # SS3 contingency (use ESM2-cls annotations)
            if ss3_list:
                for i, cl in enumerate(labels):
                    if cl and i < len(ss3_list) and ss3_list[i] in SS3_LBLS:
                        ct_ss[m][(cl, ss3_list[i])] += 1

            # Neq: model's predicted neq vs real neq from ESM2-cls
            np_preds = rec.get('neq_preds', [])
            # Regression model outputs floats; classification outputs 0/1
            if neq_real and np_preds and len(np_preds) == len(neq_real):
                valid = [(p, r) for p, r in zip(np_preds, neq_real)
                         if r is not None and p is not None]
                if len(valid) >= 10:
                    preds, reals = zip(*valid)
                    neq_corrs[m].append(spearman_r(list(preds), list(reals)))

        print()

    # ── 3. Cluster fractions ──────────────────────────────────────────────────
    print("\n=== 3. Cluster Fractions ===")
    frac_rows = []
    print(f"\n  {'Model':<12} {'PatA%':>7} {'PatB%':>7} {'Bg%':>7}")
    print("  " + "-" * 38)
    for m in MODEL_NAMES:
        def msd(vals):
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v-mu)**2 for v in vals)/len(vals))
            return mu, sd
        a_mu, a_sd = msd(frac_accum[m]['PatA'])
        b_mu, b_sd = msd(frac_accum[m]['PatB'])
        g_mu, g_sd = msd(frac_accum[m]['Bg'])
        frac_rows.append({'model': m,
                          'PatA_mean': a_mu, 'PatA_std': a_sd,
                          'PatB_mean': b_mu, 'PatB_std': b_sd,
                          'Bg_mean':   g_mu, 'Bg_std':   g_sd})
        print(f"  {m:<12} {a_mu*100:>6.1f}% {b_mu*100:>6.1f}% {g_mu*100:>6.1f}%")

    csv_path = os.path.join(args.output_dir, 'cluster_fractions.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model'] + [
            k + sfx for k in ['PatA', 'PatB', 'Bg'] for sfx in ('_mean', '_std')
        ])
        w.writeheader()
        w.writerows(frac_rows)
    print(f"\n  Saved → {csv_path}")

    # ── 4. SS3 enrichment ─────────────────────────────────────────────────────
    print("\n=== 4. SS3 Enrichment (obs/expected) ===")
    enr_rows = []
    for m in MODEL_NAMES:
        enr = enrichment_table(ct_ss[m], CLUSTER_LBLS, SS3_LBLS)
        for cl in CLUSTER_LBLS:
            for ss in SS3_LBLS:
                enr_rows.append({
                    'model': m, 'cluster': cl, 'ss': ss,
                    'enrichment': enr.get((cl, ss), float('nan')),
                    'count': ct_ss[m].get((cl, ss), 0),
                })

    # Print a readable summary
    print(f"\n  {'Model':<12} {'Clust':<7} {'H':>8} {'E':>8} {'C':>8}")
    print("  " + "-" * 48)
    for m in MODEL_NAMES:
        enr = {row['cluster'] + '|' + row['ss']: row['enrichment']
               for row in enr_rows if row['model'] == m}
        for cl in CLUSTER_LBLS:
            h = enr.get(f'{cl}|H', float('nan'))
            e = enr.get(f'{cl}|E', float('nan'))
            c = enr.get(f'{cl}|C', float('nan'))
            print(f"  {m:<12} {cl:<7} {h:>8.3f} {e:>8.3f} {c:>8.3f}")
        print()

    csv_path = os.path.join(args.output_dir, 'ss_enrichment.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'cluster', 'ss', 'enrichment', 'count'])
        w.writeheader()
        w.writerows(enr_rows)
    print(f"  Saved → {csv_path}")

    # ── 5. Neq correlation ────────────────────────────────────────────────────
    print("\n=== 5. Spearman(neq_pred, neq_real) ===")
    neq_rows = []
    print(f"\n  {'Model':<12} {'mean_r':>8} {'std_r':>8} {'n_proteins':>12}")
    print("  " + "-" * 44)
    for m in MODEL_NAMES:
        vals = [v for v in neq_corrs[m] if not math.isnan(v)]
        if vals:
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v-mu)**2 for v in vals) / len(vals))
        else:
            mu, sd = float('nan'), float('nan')
        neq_rows.append({'model': m, 'mean_r': mu, 'std_r': sd, 'n_proteins': len(vals)})
        print(f"  {m:<12} {mu:>8.4f} {sd:>8.4f} {len(vals):>12}")

    csv_path = os.path.join(args.output_dir, 'neq_corr.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'mean_r', 'std_r', 'n_proteins'])
        w.writeheader()
        w.writerows(neq_rows)
    print(f"\n  Saved → {csv_path}")

    # ── 6. Pairwise residue-level cluster agreement ───────────────────────────
    print("\n=== 6. Pairwise Cluster Agreement ===")
    MODEL_PAIRS = [
        ('ESM2-cls', 'ESM3-cls'),
        ('ESM2-cls', 'ESM2-reg'),
        ('ESM3-cls', 'ESM2-reg'),
    ]
    overlap_rows = []
    print(f"\n  {'Pair':<25} {'agree%':>8} {'PatA-same':>10} {'PatB-same':>10} {'Bg-same':>10}")
    print("  " + "-" * 65)

    for mA, mB in MODEL_PAIRS:
        total = 0; agree = 0
        same_cl = {cl: 0 for cl in CLUSTER_LBLS}
        tot_cl  = {cl: 0 for cl in CLUSTER_LBLS}

        for name in common_names:
            lA = cluster_labels[mA].get(name)
            lB = cluster_labels[mB].get(name)
            if lA is None or lB is None:
                continue
            for a, b in zip(lA, lB):
                if a and b:
                    total += 1
                    if a == b:
                        agree += 1
                        same_cl[a] += 1
                    tot_cl[a] += 1

        pct = agree / total * 100 if total else float('nan')
        pair_label = f"{mA} vs {mB}"
        row = {
            'pair': pair_label,
            'agreement_pct': pct,
            'total_residues': total,
        }
        for cl in CLUSTER_LBLS:
            row[f'{cl}_same_pct'] = same_cl[cl] / tot_cl[cl] * 100 if tot_cl[cl] else float('nan')

        overlap_rows.append(row)
        print(f"  {pair_label:<25} {pct:>8.1f}% "
              f"{row['PatA_same_pct']:>9.1f}% "
              f"{row['PatB_same_pct']:>9.1f}% "
              f"{row['Bg_same_pct']:>9.1f}%")

    csv_path = os.path.join(args.output_dir, 'cluster_overlap.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['pair', 'agreement_pct', 'total_residues'] + \
                     [f'{cl}_same_pct' for cl in CLUSTER_LBLS]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(overlap_rows)
    print(f"\n  Saved → {csv_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n=== 7. Generating Plots ===")

    # 7a. Attention energy violin (col_max distribution)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    fig.suptitle('Column-max Attention Distribution per Model', fontsize=13)
    COLORS = {'ESM2-cls': '#4C72B0', 'ESM3-cls': '#DD8452', 'ESM2-reg': '#55A868'}

    for ax, m in zip(axes, MODEL_NAMES):
        all_col_maxes = []
        for name in common_names:
            A = jsons[m][name]['attention_weights']
            N = len(A)
            col_maxes = [max(A[j][i] for j in range(N)) for i in range(N)]
            all_col_maxes.extend(col_maxes)

        # Subsample for speed (max 50k points)
        if len(all_col_maxes) > 50000:
            rng = random.Random(42)
            all_col_maxes = rng.sample(all_col_maxes, 50000)

        ax.violinplot(all_col_maxes, positions=[0], showmedians=True)
        ax.set_title(m)
        ax.set_xticks([])
        ax.set_ylabel('col-max attention weight')
        ax.set_xlabel(m)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'attention_energy_violin.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved → {plot_path}")

    # 7b. Cluster fraction stacked bars
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MODEL_NAMES))
    width = 0.5
    pA = [r['PatA_mean'] * 100 for r in frac_rows]
    pB = [r['PatB_mean'] * 100 for r in frac_rows]
    pG = [r['Bg_mean']   * 100 for r in frac_rows]
    b1 = ax.bar(x, pA, width, label='PatA', color='#4C72B0')
    b2 = ax.bar(x, pB, width, bottom=pA, label='PatB', color='#55A868')
    b3 = ax.bar(x, pG, width,
                bottom=[a + b for a, b in zip(pA, pB)],
                label='Background', color='#C44E52', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel('Fraction of residues (%)')
    ax.set_title('Cluster composition per model')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'cluster_fractions_bar.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved → {plot_path}")

    # 7c. SS enrichment heatmaps side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axes, MODEL_NAMES):
        enr_mat = np.array([
            [next((r['enrichment'] for r in enr_rows
                   if r['model'] == m and r['cluster'] == cl and r['ss'] == ss), float('nan'))
             for ss in SS3_LBLS]
            for cl in CLUSTER_LBLS
        ])
        im = ax.imshow(enr_mat, vmin=0.3, vmax=2.2, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(SS3_LBLS)))
        ax.set_xticklabels(SS3_LBLS)
        ax.set_yticks(range(len(CLUSTER_LBLS)))
        ax.set_yticklabels(CLUSTER_LBLS)
        ax.set_title(m)
        for i, cl in enumerate(CLUSTER_LBLS):
            for j, ss in enumerate(SS3_LBLS):
                v = enr_mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                        color='black' if 0.7 < v < 1.6 else 'white')
        plt.colorbar(im, ax=ax, label='obs/expected')
    fig.suptitle('SS3 enrichment per cluster × model', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'ss_enrichment_heatmap.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved → {plot_path}")

    # 7d. Pairwise cluster agreement bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = [r['pair'] for r in overlap_rows]
    y = np.arange(len(pairs))
    height = 0.2
    for k, cl in enumerate(CLUSTER_LBLS):
        vals = [r[f'{cl}_same_pct'] for r in overlap_rows]
        ax.barh(y + (k - 1) * height, vals, height, label=cl)
    overall = [r['agreement_pct'] for r in overlap_rows]
    ax.barh(y + height * (len(CLUSTER_LBLS) - 1) + height,
            overall, height, label='Overall', color='grey', alpha=0.5)
    ax.set_yticks(y + height)
    ax.set_yticklabels(pairs)
    ax.set_xlabel('% residues assigned same cluster')
    ax.set_title('Pairwise cluster agreement between models')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'cluster_agreement_bar.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved → {plot_path}")

    print("\nDone. All outputs in:", args.output_dir)


if __name__ == '__main__':
    main()
