#!/usr/bin/env python3
"""
differential_peaks.py

Characterises residues that are "differentially peaked" between ESM2 and ESM3
attention matrices — i.e., residues that one model specifically brightens
(high col_max percentile rank) while the other largely ignores (low rank).

Peak categories (per-protein):
  ESM3-specific  : col_max rank in ESM3 ≥ top_pct  AND  rank in ESM2 ≤ bot_pct
  ESM2-specific  : col_max rank in ESM2 ≥ top_pct  AND  rank in ESM3 ≤ bot_pct
  Shared         : col_max rank ≥ top_pct in BOTH models
  Background     : everything else

For each category this script reports:
  - Q8 secondary structure enrichment (vs background)
  - Mean Neq, RSA, disorder, ASA
  - Local amino-acid composition (window ±3 around peak residue)
  - Fraction of residues inside an InterPro functional domain

Outputs
-------
  differential_peaks_stats.csv
  q8_enrichment_heatmap.png
  aa_composition_heatmap.png
  neq_rsa_boxplots.png
  interpro_overlap_bar.png

Usage
-----
python differential_peaks.py \\
    --esm2  esm2_attn_frozen.json \\
    --esm3  esm3_attn.json \\
    --ref   ../../data/attn_bilstm_f1-4_nsp3_neq.json \\
    --interpro ../../data/test_data_interpro.json \\
    --output_dir results/differential_peaks
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
import matplotlib.colors as mcolors

AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
Q8_LIST = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']
CATS    = ['ESM3-specific', 'ESM2-specific', 'Shared', 'Background']
CAT_COLORS = {
    'ESM3-specific': '#DD8452',
    'ESM2-specific': '#4C72B0',
    'Shared':        '#55A868',
    'Background':    '#C7C7C7',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        recs = json.load(f)
    if isinstance(recs, list):
        return {r['name']: r for r in recs}
    return recs


def col_max_ranks(A_np):
    """Return per-residue percentile rank of col_max (0=lowest, 100=highest)."""
    N = A_np.shape[0]
    col_maxes = A_np.max(axis=0)  # (N,)
    # percentile rank: (rank / N) * 100
    order = np.argsort(col_maxes)
    ranks = np.empty(N)
    ranks[order] = np.arange(N) / (N - 1) * 100 if N > 1 else np.zeros(N)
    return ranks  # 0–100


def build_interpro_map(interpro_path):
    """
    Returns dict: protein_name -> list of (start, end) 1-based domain intervals.
    """
    with open(interpro_path) as f:
        data = json.load(f)
    domain_map = {}
    for result in data.get('results', []):
        xrefs = result.get('xref', [])
        if not xrefs:
            continue
        name = xrefs[0].get('id', xrefs[0].get('name', ''))
        intervals = []
        for match in result.get('matches', []):
            for loc in match.get('locations', []):
                start = loc.get('start', None)
                end   = loc.get('end',   None)
                if start is not None and end is not None:
                    intervals.append((int(start), int(end)))
        if name not in domain_map:
            domain_map[name] = []
        domain_map[name].extend(intervals)
    return domain_map


def in_domain(pos_0based, intervals):
    """True if 0-based pos falls inside any 1-based (start, end) interval."""
    p = pos_0based + 1  # convert to 1-based
    return any(s <= p <= e for s, e in intervals)


def enrichment(cat_counts, bg_counts, labels):
    """obs/expected for each label."""
    cat_total = sum(cat_counts.get(l, 0) for l in labels)
    bg_total  = sum(bg_counts.get(l, 0) for l in labels)
    result = {}
    for l in labels:
        obs = cat_counts.get(l, 0) / cat_total if cat_total > 0 else 0
        exp = bg_counts.get(l, 0)  / bg_total  if bg_total  > 0 else 0
        result[l] = obs / exp if exp > 1e-9 else float('nan')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--esm2',      required=True)
    p.add_argument('--esm3',      required=True)
    p.add_argument('--ref',       required=True,
                   help='Reference JSON with q8, ss_pred, rsa, asa, disorder, neq_real')
    p.add_argument('--interpro',  required=True,
                   help='InterPro JSON (test_data_interpro.json)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--top_pct',   type=float, default=25.0,
                   help='Percentile threshold for "peak" (default: top 25%%)')
    p.add_argument('--bot_pct',   type=float, default=25.0,
                   help='Percentile threshold for "not peak" (default: bottom 25%%)')
    p.add_argument('--window',    type=int, default=3,
                   help='Sequence window radius for motif analysis (default: ±3)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    TOP = 100.0 - args.top_pct   # rank threshold for "peak"
    BOT = args.bot_pct           # rank threshold for "not peak"

    jsons = {
        'ESM2': load_json(args.esm2),
        'ESM3': load_json(args.esm3),
    }
    ref    = load_json(args.ref)
    ipr    = build_interpro_map(args.interpro)
    common = sorted(set(jsons['ESM2']) & set(jsons['ESM3']) & set(ref))
    print(f"Proteins: {len(common)}  |  InterPro coverage: {len(ipr)}")
    print(f"Peak threshold: top {args.top_pct:.0f}%  |  Not-peak: bottom {args.bot_pct:.0f}%\n")

    # ── Accumulators ─────────────────────────────────────────────────────────
    q8_counts  = {c: {q: 0 for q in Q8_LIST} for c in CATS}
    neq_vals   = {c: [] for c in CATS}
    rsa_vals   = {c: [] for c in CATS}
    dis_vals   = {c: [] for c in CATS}
    asa_vals   = {c: [] for c in CATS}
    # AA composition: for each cat, count AAs in ±window around each peak residue
    aa_counts  = {c: {a: 0 for a in AA_LIST} for c in CATS}
    # InterPro: how many residues fall in a domain
    ipr_in     = {c: 0 for c in CATS}
    ipr_tot    = {c: 0 for c in CATS}

    print("Processing proteins ...")
    for idx, name in enumerate(common):
        print(f"  [{idx+1}/{len(common)}] {name}", end='\r', flush=True)
        ref_rec = ref[name]
        seq     = ref_rec['sequence']
        N       = len(seq)

        q8_list  = ref_rec.get('q8',       [])
        neq_real = ref_rec.get('neq_real', [])
        rsa_list = ref_rec.get('rsa',      [])
        dis_list = ref_rec.get('disorder', [])
        asa_list = ref_rec.get('asa',      [])
        intervals= ipr.get(name, [])

        A2 = np.array(jsons['ESM2'][name]['attention_weights'], dtype=np.float32)
        A3 = np.array(jsons['ESM3'][name]['attention_weights'], dtype=np.float32)
        if A2.shape[0] != N or A3.shape[0] != N:
            continue

        r2 = col_max_ranks(A2)  # ESM2 col_max percentile rank per residue
        r3 = col_max_ranks(A3)  # ESM3 col_max percentile rank per residue

        # Assign each residue to a category
        cat_of = []
        for i in range(N):
            peak2 = r2[i] >= TOP
            peak3 = r3[i] >= TOP
            low2  = r2[i] <= BOT
            low3  = r3[i] <= BOT
            if peak3 and low2:
                cat_of.append('ESM3-specific')
            elif peak2 and low3:
                cat_of.append('ESM2-specific')
            elif peak2 and peak3:
                cat_of.append('Shared')
            else:
                cat_of.append('Background')

        for i, cat in enumerate(cat_of):
            # Q8
            if i < len(q8_list) and q8_list[i] in Q8_LIST:
                q8_counts[cat][q8_list[i]] += 1

            # Continuous quantities
            if i < len(neq_real) and neq_real[i] is not None:
                neq_vals[cat].append(float(neq_real[i]))
            if i < len(rsa_list) and rsa_list[i] is not None:
                rsa_vals[cat].append(float(rsa_list[i]))
            if i < len(dis_list) and dis_list[i] is not None:
                dis_vals[cat].append(float(dis_list[i]))
            if i < len(asa_list) and asa_list[i] is not None:
                asa_vals[cat].append(float(asa_list[i]))

            # Local AA composition (window ±args.window)
            w = args.window
            for j in range(max(0, i - w), min(N, i + w + 1)):
                aa = seq[j]
                if aa in AA_LIST:
                    aa_counts[cat][aa] += 1

            # InterPro domain membership
            ipr_tot[cat] += 1
            if intervals and in_domain(i, intervals):
                ipr_in[cat] += 1

    print(f"\nDone. Residue counts per category:")
    for c in CATS:
        n = sum(q8_counts[c].values())
        print(f"  {c:<18}: {n:>7} residues")

    # ── 1. Q8 enrichment table ────────────────────────────────────────────────
    print("\n=== Q8 Enrichment (obs/expected vs Background) ===")
    bg = q8_counts['Background']
    q8_enr = {}
    for cat in CATS[:-1]:   # skip Background itself
        q8_enr[cat] = enrichment(q8_counts[cat], bg, Q8_LIST)

    print(f"\n  {'Cat':<18} " + "  ".join(f"{q:>6}" for q in Q8_LIST))
    print("  " + "-" * (18 + len(Q8_LIST) * 8))
    for cat in CATS[:-1]:
        vals = "  ".join(f"{q8_enr[cat][q]:>6.3f}" for q in Q8_LIST)
        print(f"  {cat:<18} {vals}")

    # ── 2. Quantitative stats ─────────────────────────────────────────────────
    print("\n=== Mean Neq / RSA / Disorder per category ===")
    print(f"\n  {'Category':<18} {'mean_Neq':>10} {'mean_RSA':>10} {'mean_dis':>10} {'ipr_frac':>10}")
    print("  " + "-" * 55)
    stat_rows = []
    for cat in CATS:
        mn  = float(np.mean(neq_vals[cat])) if neq_vals[cat] else float('nan')
        mr  = float(np.mean(rsa_vals[cat])) if rsa_vals[cat] else float('nan')
        md  = float(np.mean(dis_vals[cat])) if dis_vals[cat] else float('nan')
        ipr = ipr_in[cat] / ipr_tot[cat] if ipr_tot[cat] > 0 else float('nan')
        print(f"  {cat:<18} {mn:>10.4f} {mr:>10.4f} {md:>10.4f} {ipr:>10.4f}")
        q8e = q8_enr.get(cat, {q: float('nan') for q in Q8_LIST})
        row = {'category': cat, 'mean_neq': mn, 'mean_rsa': mr,
               'mean_disorder': md, 'interpro_frac': ipr,
               'n_residues': sum(q8_counts[cat].values())}
        for q in Q8_LIST:
            row[f'q8_enr_{q}'] = q8e.get(q, float('nan'))
        stat_rows.append(row)

    csv_path = os.path.join(args.output_dir, 'differential_peaks_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)
    print(f"\n  Saved → {csv_path}")

    # ── 3. AA enrichment (ESM3-specific vs ESM2-specific) ────────────────────
    print("\n=== AA Enrichment of peak categories vs Background ===")
    aa_enr = {}
    for cat in ['ESM3-specific', 'ESM2-specific', 'Shared']:
        aa_enr[cat] = enrichment(aa_counts[cat], aa_counts['Background'], AA_LIST)

    print(f"\n  {'Cat':<18} " + "  ".join(f"{a:<5}" for a in AA_LIST))
    print("  " + "-" * (18 + len(AA_LIST) * 7))
    for cat in ['ESM3-specific', 'ESM2-specific', 'Shared']:
        vals = "  ".join(f"{aa_enr[cat][a]:>5.3f}" for a in AA_LIST)
        print(f"  {cat:<18} {vals}")

    # ── Plot 1: Q8 enrichment heatmap ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_cats = CATS[:-1]  # ESM3-specific, ESM2-specific, Shared
    mat = np.array([[q8_enr[c].get(q, float('nan')) for q in Q8_LIST]
                    for c in plot_cats])
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.3, vmax=2.2, aspect='auto')
    ax.set_xticks(range(len(Q8_LIST)))
    ax.set_xticklabels(Q8_LIST, fontsize=11)
    ax.set_yticks(range(len(plot_cats)))
    ax.set_yticklabels(plot_cats, fontsize=10)
    for i, cat in enumerate(plot_cats):
        for j, q in enumerate(Q8_LIST):
            v = mat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                    color='black' if 0.7 < v < 1.7 else 'white')
    plt.colorbar(im, ax=ax, label='obs/expected vs Background')
    ax.set_title('Q8 enrichment of differential attention peaks\n'
                 'H=α-helix, B=β-bridge, E=β-strand, G=3₁₀-helix, I=π-helix, T=turn, S=bend, C=coil',
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'q8_enrichment_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved → {path}")

    # ── Plot 2: AA composition heatmap ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    aa_mat = np.array([[aa_enr[c].get(a, float('nan')) for a in AA_LIST]
                       for c in ['ESM3-specific', 'ESM2-specific', 'Shared']])
    im = ax.imshow(aa_mat, cmap='RdYlGn', vmin=0.5, vmax=1.8, aspect='auto')
    ax.set_xticks(range(len(AA_LIST)))
    ax.set_xticklabels(AA_LIST, fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['ESM3-specific', 'ESM2-specific', 'Shared'], fontsize=10)
    for i, cat in enumerate(['ESM3-specific', 'ESM2-specific', 'Shared']):
        for j, a in enumerate(AA_LIST):
            v = aa_mat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='black' if 0.7 < v < 1.5 else 'white')
    plt.colorbar(im, ax=ax, label='obs/expected vs Background')
    ax.set_title(f'Local AA composition (±{args.window} window) enrichment of peak categories',
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'aa_composition_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    # ── Plot 3: Neq / RSA boxplots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (vals_dict, label) in zip(axes, [
        (neq_vals, 'True Neq'),
        (rsa_vals, 'RSA (relative solvent access)'),
        (dis_vals, 'Disorder score'),
    ]):
        data   = [vals_dict[c] for c in CATS]
        colors = [CAT_COLORS[c] for c in CATS]
        bp = ax.boxplot(data, labels=CATS, patch_artist=True,
                        medianprops=dict(color='black', lw=2),
                        showfliers=False)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        ax.set_title(label)
        ax.set_xticklabels(CATS, rotation=20, ha='right', fontsize=8)
    fig.suptitle('Biophysical properties of differential attention peaks', fontsize=11)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'neq_rsa_boxplots.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    # ── Plot 4: InterPro domain overlap bar ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    fracs = [ipr_in[c] / ipr_tot[c] * 100 if ipr_tot[c] > 0 else 0.0 for c in CATS]
    bars  = ax.bar(CATS, fracs, color=[CAT_COLORS[c] for c in CATS], alpha=0.85)
    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{frac:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('% residues inside an InterPro domain')
    ax.set_title('Functional domain coverage of differential attention peaks\n'
                 '(higher = more often inside a known domain)')
    ax.set_xticklabels(CATS, rotation=15, ha='right')
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'interpro_overlap_bar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    print("\nAll done. Outputs in:", args.output_dir)


if __name__ == '__main__':
    main()
