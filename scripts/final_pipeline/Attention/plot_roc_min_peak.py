#!/usr/bin/env python3
"""
plot_roc_min_peak.py

Plots per-class ROC curves for the attention cluster → SS assignment,
sweeping the col_max percentile threshold (= --min_peak_pct) post-hoc
after running K-means **once** per protein (no demotion pass needed).

Assignment rule at threshold t (= min_peak_pct):
  PatA residue with col_max_pct_rank >= (100-t) → predicts C (coil)
  PatB residue with col_max_pct_rank >= (100-t) → predicts H (helix)
  Everything else                                → predicts E (strand)

Score used to build the ROC:
  H : col_max_pct_rank  if raw_km == PatB  else −1
  C : col_max_pct_rank  if raw_km == PatA  else −1
  E : (100-col_max_pct_rank)  if raw_km ∈ {PatA,PatB}  else 101
      (Bg-km residues always predict E → capped at 101 so they rank above
       any demoted PatA/PatB residue at every threshold)

Sweeping from high→low score:
  H/C: high score = stays in PatB/PatA even at strict min_peak_pct (high precision)
       low score  = demoted to Bg at strict thresholds, only captured at loose ones
  E:   high score = always-Bg (strands the model never captured in a pattern)
       lower score = PatA/PatB residues that get demoted only at loose thresholds

Usage
-----
python plot_roc_min_peak.py \\
    --input data/attn_bilstm_f1-4_nsp3_neq.json \\
    --output roc_min_peak.png \\
    [--n_init 10] [--max_iter 100] [--seed 42]
"""

import argparse
import json
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_attention_residues import cluster_residues


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',    required=True, help='Path to attn JSON')
    p.add_argument('--output',   default='roc_min_peak.png', help='Output PNG path')
    p.add_argument('--n_init',   type=int, default=10)
    p.add_argument('--max_iter', type=int, default=100)
    p.add_argument('--seed',     type=int, default=42)
    return p.parse_args()


def roc_from_scores(scores: np.ndarray, y_true: list):
    """
    Compute ROC curve by sweeping all unique score thresholds.

    Returns (fpr, tpr) arrays anchored at (0,0) and approaching (1,1).
    """
    n_pos = int(sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0., 1.]), np.array([0., 1.])

    order = np.argsort(scores)[::-1]          # descending score
    y_sorted = np.array(y_true, dtype=float)[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1.0 - y_sorted)

    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg

    # anchor at origin
    tpr = np.concatenate([[0.], tpr])
    fpr = np.concatenate([[0.], fpr])
    return fpr, tpr


def auc_trapz(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr, fpr))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.input) as fh:
        data = json.load(fh)
    print(f"Loaded {len(data)} proteins.")

    SS3 = ['H', 'E', 'C']
    scores = {ss: [] for ss in SS3}
    labels = {ss: [] for ss in SS3}

    for idx, rec in enumerate(data):
        name = rec['name']
        seq  = rec['sequence']
        N    = len(seq)
        print(f"[{idx+1}/{len(data)}] {name} (N={N})", end='\r', flush=True)

        ss_list = rec.get('ss_pred', [])

        try:
            result = cluster_residues(
                rec['attention_weights'],
                n_init=args.n_init,
                max_iter=args.max_iter,
                seed=args.seed,
                min_peak_pct=None,   # raw K-means, no percentile demotion
            )
        except Exception as e:
            print(f"\nSKIP {name}: {e}")
            continue

        patA_set = set(result['pattern_A'])
        patB_set = set(result['pattern_B'])
        # Bg = everything else (K-means background before any threshold)

        # compute col_max for each residue
        A = rec['attention_weights']
        col_max = [max(A[j][i] for j in range(N)) for i in range(N)]

        # percentile rank of col_max within this protein  (0 = dimmest, 100 = brightest)
        sorted_cm  = sorted(range(N), key=lambda i: col_max[i])
        pct_rank   = [0.0] * N
        for rank, i in enumerate(sorted_cm):
            pct_rank[i] = rank / (N - 1) * 100.0 if N > 1 else 50.0

        # accumulate per-residue scores + ground-truth labels
        for i in range(N):
            true_ss = ss_list[i] if i < len(ss_list) else None
            if true_ss not in SS3:
                continue

            p = pct_rank[i]
            in_patA = i in patA_set
            in_patB = i in patB_set
            in_bg   = not in_patA and not in_patB

            sc_H = p      if in_patB else -1.0
            sc_C = p      if in_patA else -1.0
            sc_E = 101.0  if in_bg   else (100.0 - p)

            for ss, sc in zip(SS3, [sc_H, sc_E, sc_C]):
                # reorder to H/E/C
                pass

            scores['H'].append(sc_H)
            scores['E'].append(sc_E)
            scores['C'].append(sc_C)
            labels['H'].append(1 if true_ss == 'H' else 0)
            labels['E'].append(1 if true_ss == 'E' else 0)
            labels['C'].append(1 if true_ss == 'C' else 0)

    print(f"\nCollected {len(scores['H'])} residues across all proteins.")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    class_style = {
        'H': dict(color='#e74c3c', label='Helix  (PatB → H)'),
        'C': dict(color='#3498db', label='Coil   (PatA → C)'),
        'E': dict(color='#2ecc71', label='Strand (Bg    → E)'),
    }

    for ss in SS3:
        sc  = np.array(scores[ss], dtype=float)
        lb  = labels[ss]
        fpr, tpr = roc_from_scores(sc, lb)
        auc_val  = auc_trapz(fpr, tpr)
        style = class_style[ss]
        ax.plot(fpr, tpr, lw=2,
                color=style['color'],
                label=f"{style['label']}   AUC = {auc_val:.3f}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate',  fontsize=13)
    ax.set_ylabel('True Positive Rate',   fontsize=13)
    ax.set_title(
        'ROC: Attention Cluster as SS Classifier\n'
        '(col_max percentile threshold swept = --min_peak_pct equivalent)',
        fontsize=12,
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved ROC plot: {args.output}")

    # also print AUC summary
    print("\nAUC summary:")
    for ss in SS3:
        sc  = np.array(scores[ss], dtype=float)
        lb  = labels[ss]
        fpr, tpr = roc_from_scores(sc, lb)
        print(f"  {ss}: AUC = {auc_trapz(fpr, tpr):.4f}   "
              f"(n_pos={sum(lb)}, n_neg={len(lb)-sum(lb)})")


if __name__ == '__main__':
    main()
