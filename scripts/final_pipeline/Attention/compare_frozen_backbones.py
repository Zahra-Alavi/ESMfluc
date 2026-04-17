#!/usr/bin/env python3
"""
compare_frozen_backbones.py

Focused comparison of:
  - ESM2-frozen  : ESM2 backbone (frozen), BiLSTM classifier
  - ESM3-frozen  : ESM3 backbone (frozen), BiLSTM classifier

Both models share the same head architecture and training objective.
Only the backbone differs.  Because ESM3 is structure-aware (trained on
sequence + structure + function), differences in the downstream BiLSTM
attention reveal how much structural knowledge the backbone injects.

Analyses
--------
1. Raw Attention Statistics (mean, entropy, sparsity, diagonal dominance)
2. Sequence-separation profile of high-attention pairs
   For each (i,j) where attention > threshold, record |i-j|.
   Peaks near diagonal → local / contact-like attention.
   Long-range enrichment → co-evolutionary / domain signal.
3. Per-residue attention entropy (H = -sum p log p over row)
   Low entropy = focused; high entropy = diffuse.
4. K-means cluster assignment + SS3 enrichment
5. Residue-level cluster agreement across the two models

Outputs
-------
  stats_comparison.csv
  separation_profile.png   — sequence-separation of top pairs (side-by-side)
  entropy_distribution.png — per-residue entropy violin
  ss_enrichment_compare.png
  cluster_agreement.txt

Usage
-----
python compare_frozen_backbones.py \\
    --esm2  esm2_attn_frozen.json \\
    --esm3  esm3_attn.json \\
    --esm2_ref  ../../data/attn_bilstm_f1-4_nsp3_neq.json \\
    --output_dir results/frozen_backbone_comparison
"""

import argparse
import csv
import json
import math
import os
import sys

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


def row_entropy(row):
    """Shannon entropy of a probability row (H = -sum p log p, base 2)."""
    out = 0.0
    for p in row:
        if p > 1e-12:
            out -= p * math.log2(p)
    return out


def attn_stats(A_np):
    """Return dict of scalar stats for an NxN numpy attention matrix."""
    N = A_np.shape[0]
    mean_val   = float(A_np.mean())
    max_val    = float(A_np.max())
    diag_mean  = float(np.diag(A_np).mean())
    off_mean   = float((A_np.sum() - np.trace(A_np)) / (N * N - N))
    sparsity   = float((A_np < 0.01).mean())
    entropies  = [row_entropy(A_np[i].tolist()) for i in range(N)]
    col_maxes  = A_np.max(axis=0)
    return {
        'mean':           mean_val,
        'max':            max_val,
        'diag_mean':      diag_mean,
        'off_diag_mean':  off_mean,
        'sparsity':       sparsity,
        'entropy_mean':   float(np.mean(entropies)),
        'entropy_std':    float(np.std(entropies)),
        'col_max_mean':   float(col_maxes.mean()),
        'col_max_std':    float(col_maxes.std()),
    }


def cluster_labels_np(A_np, n_init=10, seed=42):
    """K=3 clustering on L2-normalised row+col features. Returns label list."""
    N = A_np.shape[0]
    if N < 3:
        return None
    raw  = np.concatenate([A_np, A_np.T], axis=1)       # (N, 2N)
    feat = normalize(raw, norm='l2', axis=1)
    km   = KMeans(n_clusters=3, n_init=n_init, random_state=seed, max_iter=100)
    raw_labels = km.fit_predict(feat)
    energy = (A_np.sum(axis=1) + A_np.sum(axis=0)) / N
    ranked = sorted(range(3), key=lambda c: energy[raw_labels == c].mean(), reverse=True)
    mapping = {ranked[0]: 'PatA', ranked[1]: 'PatB', ranked[2]: 'Bg'}
    return [mapping[int(l)] for l in raw_labels]


def separation_profile(A_np, top_pct=5.0):
    """
    For the top `top_pct` percent of non-diagonal (i,j) pairs by attention weight,
    return the list of sequence separations |i-j|.
    """
    N = A_np.shape[0]
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    vals = A_np[mask]
    threshold = np.percentile(vals, 100.0 - top_pct)
    seps = []
    rows, cols = np.where((A_np >= threshold) & mask)
    seps = np.abs(rows - cols).tolist()
    return seps


def enrichment_table(counts, row_labels, col_labels):
    grand = sum(counts.get((r, c), 0) for r in row_labels for c in col_labels)
    if grand == 0:
        return {}
    row_tot = {r: sum(counts.get((r, c), 0) for c in col_labels) for r in row_labels}
    col_tot = {c: sum(counts.get((r, c), 0) for r in row_labels) for c in col_labels}
    return {
        (r, c): counts.get((r, c), 0) / (row_tot[r] * col_tot[c] / grand)
        if row_tot[r] * col_tot[c] > 0 else float('nan')
        for r in row_labels for c in col_labels
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--esm2',    required=True, help='ESM2-frozen attention JSON')
    p.add_argument('--esm3',    required=True, help='ESM3-frozen attention JSON')
    p.add_argument('--esm2_ref', required=True,
                   help='Reference ESM2-cls JSON with ss_pred / neq_real annotations')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--n_init', type=int, default=10)
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--top_pct', type=float, default=5.0,
                   help='Top %% of attention pairs to include in separation profile (default 5)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    MODELS = ['ESM2-frozen', 'ESM3-frozen']
    jsons = {
        'ESM2-frozen': load_json(args.esm2),
        'ESM3-frozen': load_json(args.esm3),
    }
    ref = load_json(args.esm2_ref)   # has ss_pred, neq_real

    common = sorted(set(jsons['ESM2-frozen']) & set(jsons['ESM3-frozen']) & set(ref))
    print(f"Common proteins: {len(common)}\n")

    CLUSTER_LBLS = ['PatA', 'PatB', 'Bg']
    SS3_LBLS     = ['H', 'E', 'C']

    # Accumulators
    stat_accum    = {m: [] for m in MODELS}
    entropy_all   = {m: [] for m in MODELS}
    seps_all      = {m: [] for m in MODELS}
    ct_ss         = {m: {(cl, ss): 0 for cl in CLUSTER_LBLS for ss in SS3_LBLS}
                     for m in MODELS}
    cluster_store = {m: {} for m in MODELS}   # name -> label list

    # ── Per-protein loop ──────────────────────────────────────────────────────
    print("Processing proteins ...")
    for idx, name in enumerate(common):
        print(f"  [{idx+1}/{len(common)}] {name}", end='\r', flush=True)
        ref_rec  = ref[name]
        ss3_list = ref_rec.get('ss_pred', [])

        for m in MODELS:
            A_np = np.array(jsons[m][name]['attention_weights'], dtype=np.float32)
            N    = A_np.shape[0]

            # -- stats
            stat_accum[m].append(attn_stats(A_np))

            # -- per-residue entropy
            for i in range(N):
                entropy_all[m].append(row_entropy(A_np[i].tolist()))

            # -- separation profile
            seps_all[m].extend(separation_profile(A_np, top_pct=args.top_pct))

            # -- clustering
            labels = cluster_labels_np(A_np, n_init=args.n_init, seed=args.seed)
            if labels is not None:
                cluster_store[m][name] = labels
                if ss3_list:
                    for i, cl in enumerate(labels):
                        if cl and i < len(ss3_list) and ss3_list[i] in SS3_LBLS:
                            ct_ss[m][(cl, ss3_list[i])] += 1

    print(f"\nDone processing {len(common)} proteins.\n")

    # ── 1. Stats summary ──────────────────────────────────────────────────────
    print("=== 1. Attention Statistics ===")
    STAT_KEYS = ['mean', 'max', 'diag_mean', 'off_diag_mean',
                 'sparsity', 'entropy_mean', 'entropy_std',
                 'col_max_mean', 'col_max_std']
    stat_rows = []
    for m in MODELS:
        row = {'model': m}
        for k in STAT_KEYS:
            vals = [s[k] for s in stat_accum[m]]
            row[k + '_mean'] = float(np.mean(vals))
            row[k + '_std']  = float(np.std(vals))
        stat_rows.append(row)

    print(f"\n  {'Metric':<22} {'ESM2-frozen':>14} {'ESM3-frozen':>14}")
    print("  " + "-" * 52)
    for k in STAT_KEYS:
        v2 = [r[k + '_mean'] for r in stat_rows if r['model'] == 'ESM2-frozen'][0]
        v3 = [r[k + '_mean'] for r in stat_rows if r['model'] == 'ESM3-frozen'][0]
        diff = '↑ESM3' if v3 > v2 else '↓ESM3'
        print(f"  {k:<22} {v2:>14.5f} {v3:>14.5f}  {diff}")

    csv_path = os.path.join(args.output_dir, 'stats_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        cols = ['model'] + [k + sfx for k in STAT_KEYS for sfx in ('_mean', '_std')]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(stat_rows)
    print(f"\n  Saved → {csv_path}")

    # ── 2. Sequence-separation profile ───────────────────────────────────────
    print("\n=== 2. Sequence-Separation Profile (top-{:.0f}% pairs) ===".format(args.top_pct))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    COLORS = {'ESM2-frozen': '#4C72B0', 'ESM3-frozen': '#DD8452'}
    max_sep = max(max(seps_all['ESM2-frozen']), max(seps_all['ESM3-frozen']))
    bins = np.arange(0, min(max_sep + 2, 202), 2)

    for ax, m in zip(axes, MODELS):
        seps = np.array(seps_all[m])
        counts, edges = np.histogram(seps, bins=bins, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(centres, counts, width=2, color=COLORS[m], alpha=0.8, edgecolor='none')
        ax.axvline(4,  color='grey', lw=1, ls='--', label='α-helix turn (~3.6)')
        ax.axvline(2,  color='green', lw=1, ls=':', label='β-strand skip (~2)')
        ax.set_xlabel('Sequence separation |i − j|')
        ax.set_ylabel('Density')
        ax.set_title(m)
        ax.legend(fontsize=8)

    fig.suptitle(f'Sequence separation of top-{args.top_pct:.0f}% attention pairs', fontsize=12)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'separation_profile.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    # Summary stats on separation
    for m in MODELS:
        seps = np.array(seps_all[m])
        local_frac  = (seps <= 5).mean()
        medium_frac = ((seps > 5) & (seps <= 20)).mean()
        long_frac   = (seps > 20).mean()
        print(f"  {m}: local(≤5)={local_frac*100:.1f}%  "
              f"medium(6-20)={medium_frac*100:.1f}%  "
              f"long(>20)={long_frac*100:.1f}%  "
              f"median={np.median(seps):.0f}")

    # ── 3. Per-residue entropy ────────────────────────────────────────────────
    print("\n=== 3. Per-Residue Attention Row Entropy ===")

    # Subsample for violin (max 30k per model to keep plot readable)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7, 5))
    data_to_plot = []
    labels_to_plot = []
    for m in MODELS:
        e = np.array(entropy_all[m])
        if len(e) > 30000:
            e = rng.choice(e, 30000, replace=False)
        data_to_plot.append(e)
        labels_to_plot.append(m)

    parts = ax.violinplot(data_to_plot, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(COLORS.values())[i])
        pc.set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels_to_plot)
    ax.set_ylabel('Row entropy (bits)')
    ax.set_title('Per-residue attention entropy\n(lower = more focused, higher = more diffuse)')

    for i, m in enumerate(MODELS):
        mu  = np.mean(entropy_all[m])
        med = np.median(entropy_all[m])
        ax.text(i + 1, ax.get_ylim()[0] + 0.1, f'μ={mu:.2f}\nmed={med:.2f}',
                ha='center', va='bottom', fontsize=9)
        print(f"  {m}: mean_entropy={mu:.4f}  median={med:.4f}")

    plt.tight_layout()
    path = os.path.join(args.output_dir, 'entropy_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    # ── 4. SS3 enrichment ─────────────────────────────────────────────────────
    print("\n=== 4. SS3 Enrichment per Cluster ===")
    enr_all = {}
    for m in MODELS:
        enr_all[m] = enrichment_table(ct_ss[m], CLUSTER_LBLS, SS3_LBLS)

    print(f"\n  {'Model':<14} {'Clust':<7} {'H':>8} {'E':>8} {'C':>8}")
    print("  " + "-" * 48)
    for m in MODELS:
        for cl in CLUSTER_LBLS:
            h = enr_all[m].get((cl, 'H'), float('nan'))
            e = enr_all[m].get((cl, 'E'), float('nan'))
            c = enr_all[m].get((cl, 'C'), float('nan'))
            print(f"  {m:<14} {cl:<7} {h:>8.3f} {e:>8.3f} {c:>8.3f}")
        print()

    # Side-by-side heatmap
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, MODELS):
        enr_mat = np.array([
            [enr_all[m].get((cl, ss), float('nan')) for ss in SS3_LBLS]
            for cl in CLUSTER_LBLS
        ])
        im = ax.imshow(enr_mat, vmin=0.3, vmax=2.3, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(SS3_LBLS)))
        ax.set_xticklabels(SS3_LBLS, fontsize=11)
        ax.set_yticks(range(len(CLUSTER_LBLS)))
        ax.set_yticklabels(CLUSTER_LBLS)
        ax.set_title(m)
        for i, cl in enumerate(CLUSTER_LBLS):
            for j, ss in enumerate(SS3_LBLS):
                v = enr_mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=10,
                        color='black' if 0.7 < v < 1.7 else 'white')
        plt.colorbar(im, ax=ax, label='obs/expected')
    fig.suptitle('SS3 enrichment by cluster\n(> 1 = over-represented, < 1 = under-represented)',
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'ss_enrichment_compare.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    # ── 4b. Enrichment delta (ESM3 - ESM2): where does ESM3 improve? ─────────
    print("  Enrichment delta (ESM3 − ESM2):")
    print(f"  {'Clust':<7} {'ΔH':>8} {'ΔE':>8} {'ΔC':>8}  interpretation")
    print("  " + "-" * 60)
    for cl in CLUSTER_LBLS:
        dH = enr_all['ESM3-frozen'].get((cl,'H'),0) - enr_all['ESM2-frozen'].get((cl,'H'),0)
        dE = enr_all['ESM3-frozen'].get((cl,'E'),0) - enr_all['ESM2-frozen'].get((cl,'E'),0)
        dC = enr_all['ESM3-frozen'].get((cl,'C'),0) - enr_all['ESM2-frozen'].get((cl,'C'),0)
        strongest = max([('H',dH),('E',dE),('C',dC)], key=lambda x: abs(x[1]))
        interp = f"ESM3 cluster {cl} gains more {strongest[0]} signal" if abs(strongest[1]) > 0.05 else "similar"
        print(f"  {cl:<7} {dH:>+8.3f} {dE:>+8.3f} {dC:>+8.3f}  {interp}")

    # ── 5. Cluster agreement ──────────────────────────────────────────────────
    print("\n=== 5. Residue-level Cluster Agreement ===")
    total = 0; agree = 0
    per_cl_same = {cl: 0 for cl in CLUSTER_LBLS}
    per_cl_tot  = {cl: 0 for cl in CLUSTER_LBLS}

    for name in common:
        lA = cluster_store['ESM2-frozen'].get(name)
        lB = cluster_store['ESM3-frozen'].get(name)
        if lA is None or lB is None:
            continue
        for a, b in zip(lA, lB):
            if a and b:
                total += 1
                if a == b:
                    agree += 1
                    per_cl_same[a] += 1
                per_cl_tot[a] += 1

    overall_pct = agree / total * 100 if total else float('nan')
    print(f"\n  Overall agreement: {overall_pct:.1f}% ({agree}/{total} residues)")
    for cl in CLUSTER_LBLS:
        pct = per_cl_same[cl] / per_cl_tot[cl] * 100 if per_cl_tot[cl] else float('nan')
        print(f"  {cl}: {pct:.1f}%")

    result_path = os.path.join(args.output_dir, 'cluster_agreement.txt')
    with open(result_path, 'w') as f:
        f.write(f"Overall agreement: {overall_pct:.1f}% ({agree}/{total} residues)\n")
        for cl in CLUSTER_LBLS:
            pct = per_cl_same[cl] / per_cl_tot[cl] * 100 if per_cl_tot[cl] else float('nan')
            f.write(f"{cl}: {pct:.1f}%\n")
    print(f"\n  Saved → {result_path}")

    # ── 6. Long-range vs local enrichment side-by-side plot ──────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins_fine = np.arange(0, 101, 1)
    for m in MODELS:
        seps = np.array(seps_all[m])
        seps_clipped = np.clip(seps, 0, 100)
        counts, edges = np.histogram(seps_clipped, bins=bins_fine, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.plot(centres, counts, color=COLORS[m], lw=1.8, label=m)
    ax.set_xlabel('Sequence separation |i − j|')
    ax.set_ylabel('Density')
    ax.set_title(f'Sequence separation of top-{args.top_pct:.0f}% attention pairs\n'
                 'Local peak = contact-like; long-range = co-evolutionary / global signal')
    ax.legend()
    ax.set_xlim(0, 100)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'separation_overlay.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved → {path}")

    print("\nAll done. Outputs in:", args.output_dir)


if __name__ == '__main__':
    main()
