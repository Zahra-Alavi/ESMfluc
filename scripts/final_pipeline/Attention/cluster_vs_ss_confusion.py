#!/usr/bin/env python3
"""
cluster_vs_ss_confusion.py

Cross-tabulates attention-cluster assignment (PatA / PatB / Background)
against secondary structure label (H / E / C) from NetSurfP, across all
proteins in the test set.

Logic
-----
For each residue in each protein:
  1. K-means on the residue's attention profile shape → cluster label
     (PatA = highest energy cluster, PatB = middle, Bg = lowest)
  2. NetSurfP 3-class SS label from the JSON field 'ss_pred' (H / E / C)

Build a 3×3 contingency table (cluster × SS) summed over all proteins.
Report:
  - Raw counts
  - Row-normalized (given cluster, what fraction is each SS?)
  - Column-normalized (given SS, what fraction is each cluster?)
  - Enrichment: obs / expected_if_independent
    expected[k,s] = row_total[k] * col_total[s] / grand_total
  - Chi-square statistic + p-value (scipy) or manual (fallback)

Also reports the same analysis broken down by:
  - Q8 (8-class SS): H B E G I T S C
  - RSA quartile (low / mid-low / mid-high / high solvent exposure)
  - Disorder quartile

Usage
-----
python cluster_vs_ss_confusion.py \\
    --input ../../../data/attn_bilstm_f1-4_nsp3_neq.json \\
    [--min_peak_pct 25] \\
    [--n_init 10] \\
    [--seed 42] \\
    [--output ../../../data/cluster_vs_ss_results.json]
"""

import argparse
import json
import sys
import os
import math

# ── import cluster_residues from sibling script ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_attention_residues import cluster_residues


# ── helpers ──────────────────────────────────────────────────────────────────

def chi2_pvalue(table, row_labels, col_labels):
    """
    Compute chi-square statistic and p-value for a contingency table.
    Uses scipy if available, otherwise returns stat only.

    table : dict {(row, col): count}
    """
    rows = row_labels
    cols = col_labels
    grand = sum(table.values())
    if grand == 0:
        return float('nan'), float('nan')

    row_totals = {r: sum(table.get((r, c), 0) for c in cols) for r in rows}
    col_totals = {c: sum(table.get((r, c), 0) for r in rows) for c in cols}

    chi2 = 0.0
    for r in rows:
        for c in cols:
            obs = table.get((r, c), 0)
            exp = row_totals[r] * col_totals[c] / grand
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

    try:
        from scipy.stats import chi2 as chi2_dist
        df = (len(rows) - 1) * (len(cols) - 1)
        pval = 1.0 - chi2_dist.cdf(chi2, df)
    except ImportError:
        pval = float('nan')

    return chi2, pval


def enrichment_table(table, row_labels, col_labels):
    """
    Returns enrichment[r][c] = obs / expected_if_independent.
    >1 means over-represented, <1 under-represented.
    """
    grand = sum(table.values())
    if grand == 0:
        return {}
    row_totals = {r: sum(table.get((r, c), 0) for c in col_labels) for r in row_labels}
    col_totals = {c: sum(table.get((r, c), 0) for r in row_labels) for c in col_labels}

    enrich = {}
    for r in row_labels:
        enrich[r] = {}
        for c in col_labels:
            obs = table.get((r, c), 0)
            exp = row_totals[r] * col_totals[c] / grand
            enrich[r][c] = obs / exp if exp > 0 else float('nan')
    return enrich


def print_table(label, table, row_labels, col_labels, fmt=".0f"):
    width = max(len(str(r)) for r in row_labels) + 2
    col_w = max(max(len(str(c)) for c in col_labels), 8)
    header = " " * width + "  ".join(f"{c:>{col_w}}" for c in col_labels)
    print(f"\n{label}")
    print(header)
    for r in row_labels:
        vals = "  ".join(f"{table.get((r,c), 0):>{col_w}{fmt}}" for c in col_labels)
        print(f"{str(r):<{width}}{vals}")


def quartile_label(values):
    """Assign quartile labels (Q1..Q4) to a list of floats."""
    n = len(values)
    if n == 0:
        return []
    sorted_v = sorted(values)
    q1 = sorted_v[n // 4]
    q2 = sorted_v[n // 2]
    q3 = sorted_v[3 * n // 4]
    labels = []
    for v in values:
        if v <= q1:
            labels.append('Q1_low')
        elif v <= q2:
            labels.append('Q2_mid_low')
        elif v <= q3:
            labels.append('Q3_mid_high')
        else:
            labels.append('Q4_high')
    return labels


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to attn JSON (with ss_pred, q8, rsa, disorder)')
    p.add_argument('--min_peak_pct', type=float, default=None,
                   help='Percentile threshold for cluster assignment (e.g. 25)')
    p.add_argument('--n_init',  type=int, default=10)
    p.add_argument('--max_iter', type=int, default=100)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--output',  default=None, help='Save full results to JSON')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"  {len(data)} proteins loaded.\n")

    # ── accumulators ─────────────────────────────────────────────────────────
    CLUSTER_LABELS = ['PatA', 'PatB', 'Bg']
    SS3_LABELS     = ['H', 'E', 'C']
    Q8_LABELS      = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']
    RSA_LABELS     = ['Q1_low', 'Q2_mid_low', 'Q3_mid_high', 'Q4_high']
    DIS_LABELS     = ['Q1_low', 'Q2_mid_low', 'Q3_mid_high', 'Q4_high']
    NEQ_LABELS     = ['neq=0(rigid)', 'neq=1(flex)']
    NEQR_LABELS    = ['Q1_low', 'Q2_mid_low', 'Q3_mid_high', 'Q4_high']

    ct_ss3   = {(c, s): 0 for c in CLUSTER_LABELS for s in SS3_LABELS}
    ct_q8    = {(c, s): 0 for c in CLUSTER_LABELS for s in Q8_LABELS}
    ct_rsa   = {(c, s): 0 for c in CLUSTER_LABELS for s in RSA_LABELS}
    ct_dis   = {(c, s): 0 for c in CLUSTER_LABELS for s in DIS_LABELS}
    ct_neq   = {(c, s): 0 for c in CLUSTER_LABELS for s in NEQ_LABELS}
    ct_neqr  = {(c, s): 0 for c in CLUSTER_LABELS for s in NEQR_LABELS}

    # per-protein records for JSON output
    records = []

    for idx, rec in enumerate(data):
        name = rec['name']
        seq  = rec['sequence']
        N    = len(seq)

        print(f"[{idx+1}/{len(data)}] {name} (N={N}) ...", end=' ', flush=True)

        # run clustering
        try:
            result = cluster_residues(
                rec['attention_weights'],
                n_init=args.n_init,
                max_iter=args.max_iter,
                seed=args.seed,
                min_peak_pct=args.min_peak_pct,
            )
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # build per-residue cluster label list
        cluster_of = [''] * N
        for i in result['pattern_A']:
            cluster_of[i] = 'PatA'
        for i in result['pattern_B']:
            cluster_of[i] = 'PatB'
        for i in result['background']:
            cluster_of[i] = 'Bg'

        ss3_list  = rec.get('ss_pred', [])
        q8_list   = rec.get('q8', [])
        rsa_list  = rec.get('rsa', [None] * N)
        dis_list  = rec.get('disorder', [None] * N)
        neq_list  = rec.get('neq_preds', [])
        neqr_list = rec.get('neq_real', [None] * N)

        rsa_q  = quartile_label([v for v in rsa_list  if v is not None]) if rsa_list  else []
        dis_q  = quartile_label([v for v in dis_list  if v is not None]) if dis_list  else []
        neqr_q = quartile_label([v for v in neqr_list if v is not None]) if neqr_list else []

        # fill contingency tables
        rsa_valid_idx  = 0
        dis_valid_idx  = 0
        neqr_valid_idx = 0
        for i in range(N):
            cl = cluster_of[i]
            if not cl:
                continue

            # SS3
            if i < len(ss3_list) and ss3_list[i] in SS3_LABELS:
                ct_ss3[(cl, ss3_list[i])] += 1

            # Q8
            if i < len(q8_list) and q8_list[i] in Q8_LABELS:
                ct_q8[(cl, q8_list[i])] += 1

            # RSA quartile
            if rsa_list[i] is not None and rsa_valid_idx < len(rsa_q):
                ct_rsa[(cl, rsa_q[rsa_valid_idx])] += 1
                rsa_valid_idx += 1

            # Disorder quartile
            if dis_list[i] is not None and dis_valid_idx < len(dis_q):
                ct_dis[(cl, dis_q[dis_valid_idx])] += 1
                dis_valid_idx += 1

            # Predicted Neq class
            if i < len(neq_list):
                neq_lbl = 'neq=1(flex)' if neq_list[i] == 1 else 'neq=0(rigid)'
                ct_neq[(cl, neq_lbl)] += 1

            # True Neq quartile
            if neqr_list[i] is not None and neqr_valid_idx < len(neqr_q):
                ct_neqr[(cl, neqr_q[neqr_valid_idx])] += 1
                neqr_valid_idx += 1

        patA_count = len(result['pattern_A'])
        patB_count = len(result['pattern_B'])
        bg_count   = len(result['background'])
        print(f"PatA={patA_count}  PatB={patB_count}  Bg={bg_count}")

        records.append({
            'name': name,
            'length': N,
            'pattern_A': result['pattern_A'],
            'pattern_B': result['pattern_B'],
            'background': result['background'],
            'ss3': ss3_list,
            'q8': q8_list,
            'rsa': rsa_list,
            'disorder': dis_list,
            'neq_preds': neq_list,
            'neq_real': neqr_list,
        })

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("CLUSTER vs SECONDARY STRUCTURE (3-class: H/E/C)")
    print("="*70)

    # Raw counts
    print_table("Raw counts:", ct_ss3, CLUSTER_LABELS, SS3_LABELS, fmt=".0f")

    # Row-normalized (cluster → SS composition)
    row_totals_ss3 = {r: sum(ct_ss3.get((r, c), 0) for c in SS3_LABELS) for r in CLUSTER_LABELS}
    ct_ss3_rownorm = {(r, c): ct_ss3.get((r,c), 0) / row_totals_ss3[r] * 100
                      if row_totals_ss3[r] > 0 else 0.0
                      for r in CLUSTER_LABELS for c in SS3_LABELS}
    print_table("Row-normalized (% SS within each cluster):", ct_ss3_rownorm, CLUSTER_LABELS, SS3_LABELS, fmt=".1f")

    # Enrichment
    enrich_ss3 = enrichment_table(ct_ss3, CLUSTER_LABELS, SS3_LABELS)
    ct_enrich_ss3 = {(r, c): enrich_ss3[r][c] for r in CLUSTER_LABELS for c in SS3_LABELS}
    print_table("Enrichment (obs/expected):", ct_enrich_ss3, CLUSTER_LABELS, SS3_LABELS, fmt=".2f")

    chi2, pval = chi2_pvalue(ct_ss3, CLUSTER_LABELS, SS3_LABELS)
    print(f"\nChi-square = {chi2:.2f},  p-value = {pval:.2e}  (df={(len(CLUSTER_LABELS)-1)*(len(SS3_LABELS)-1)})")

    print("\n" + "="*70)
    print("CLUSTER vs SECONDARY STRUCTURE (8-class Q8)")
    print("="*70)
    print_table("Row-normalized (% Q8 within each cluster):",
                {(r,c): ct_q8.get((r,c),0) / max(sum(ct_q8.get((r,s),0) for s in Q8_LABELS),1) * 100
                 for r in CLUSTER_LABELS for c in Q8_LABELS},
                CLUSTER_LABELS, Q8_LABELS, fmt=".1f")
    enrich_q8 = enrichment_table(ct_q8, CLUSTER_LABELS, Q8_LABELS)
    ct_enrich_q8 = {(r, c): enrich_q8[r][c] for r in CLUSTER_LABELS for c in Q8_LABELS}
    print_table("Enrichment (obs/expected):", ct_enrich_q8, CLUSTER_LABELS, Q8_LABELS, fmt=".2f")

    print("\n" + "="*70)
    print("CLUSTER vs RSA QUARTILE (solvent accessibility)")
    print("="*70)
    print_table("Row-normalized (% RSA quartile within each cluster):",
                {(r,c): ct_rsa.get((r,c),0) / max(sum(ct_rsa.get((r,s),0) for s in RSA_LABELS),1) * 100
                 for r in CLUSTER_LABELS for c in RSA_LABELS},
                CLUSTER_LABELS, RSA_LABELS, fmt=".1f")
    enrich_rsa = enrichment_table(ct_rsa, CLUSTER_LABELS, RSA_LABELS)
    ct_enrich_rsa = {(r, c): enrich_rsa[r][c] for r in CLUSTER_LABELS for c in RSA_LABELS}
    print_table("Enrichment (obs/expected):", ct_enrich_rsa, CLUSTER_LABELS, RSA_LABELS, fmt=".2f")

    print("\n" + "="*70)
    print("CLUSTER vs DISORDER QUARTILE")
    print("="*70)
    print_table("Row-normalized (% disorder quartile within each cluster):",
                {(r,c): ct_dis.get((r,c),0) / max(sum(ct_dis.get((r,s),0) for s in DIS_LABELS),1) * 100
                 for r in CLUSTER_LABELS for c in DIS_LABELS},
                CLUSTER_LABELS, DIS_LABELS, fmt=".1f")
    enrich_dis = enrichment_table(ct_dis, CLUSTER_LABELS, DIS_LABELS)
    ct_enrich_dis = {(r, c): enrich_dis[r][c] for r in CLUSTER_LABELS for c in DIS_LABELS}
    print_table("Enrichment (obs/expected):", ct_enrich_dis, CLUSTER_LABELS, DIS_LABELS, fmt=".2f")

    print("\n" + "="*70)
    print("CLUSTER vs PREDICTED NEQ CLASS (0=rigid / 1=flexible)")
    print("="*70)
    print_table("Row-normalized (% Neq class within each cluster):",
                {(r,c): ct_neq.get((r,c),0) / max(sum(ct_neq.get((r,s),0) for s in NEQ_LABELS),1) * 100
                 for r in CLUSTER_LABELS for c in NEQ_LABELS},
                CLUSTER_LABELS, NEQ_LABELS, fmt=".1f")
    enrich_neq = enrichment_table(ct_neq, CLUSTER_LABELS, NEQ_LABELS)
    ct_enrich_neq = {(r, c): enrich_neq[r][c] for r in CLUSTER_LABELS for c in NEQ_LABELS}
    print_table("Enrichment (obs/expected):", ct_enrich_neq, CLUSTER_LABELS, NEQ_LABELS, fmt=".2f")
    chi2_neq, pval_neq = chi2_pvalue(ct_neq, CLUSTER_LABELS, NEQ_LABELS)
    print(f"\nChi-square = {chi2_neq:.2f},  p-value = {pval_neq:.2e}  (df={(len(CLUSTER_LABELS)-1)*(len(NEQ_LABELS)-1)})")

    print("\n" + "="*70)
    print("CLUSTER vs TRUE NEQ QUARTILE (continuous neq_real)")
    print("="*70)
    print_table("Row-normalized (% true-Neq quartile within each cluster):",
                {(r,c): ct_neqr.get((r,c),0) / max(sum(ct_neqr.get((r,s),0) for s in NEQR_LABELS),1) * 100
                 for r in CLUSTER_LABELS for c in NEQR_LABELS},
                CLUSTER_LABELS, NEQR_LABELS, fmt=".1f")
    enrich_neqr = enrichment_table(ct_neqr, CLUSTER_LABELS, NEQR_LABELS)
    ct_enrich_neqr = {(r, c): enrich_neqr[r][c] for r in CLUSTER_LABELS for c in NEQR_LABELS}
    print_table("Enrichment (obs/expected):", ct_enrich_neqr, CLUSTER_LABELS, NEQR_LABELS, fmt=".2f")

    # ── SS classification performance (treat clusters as SS predictions) ──────
    # Fixed assignment: PatA → C (coil), PatB → H (helix), Bg → E (strand)
    # This asks: if I use the attention cluster as a 3-class SS classifier,
    # how good is it?  Uses the ct_ss3 contingency table already built.
    print("\n" + "="*70)
    print("CLUSTER AS SS CLASSIFIER (PatA→C, PatB→H, Bg→E)")
    print("Fixed assignment: attention clustering used as SS prediction")
    print("="*70)

    # mapping: cluster → predicted SS class
    CLUSTER_TO_SS = {'PatA': 'C', 'PatB': 'H', 'Bg': 'E'}

    # Build classic confusion matrix: rows=true SS, cols=predicted SS
    # TP[ss] = ct_ss3[(cluster_that_maps_to_ss, ss)]
    ss_to_cluster = {v: k for k, v in CLUSTER_TO_SS.items()}  # C→PatA, H→PatB, E→Bg

    grand_total = sum(ct_ss3.values())
    correct = sum(ct_ss3.get((ss_to_cluster[ss], ss), 0) for ss in SS3_LABELS)
    accuracy = correct / grand_total if grand_total > 0 else float('nan')

    print(f"\nOverall accuracy: {correct}/{grand_total} = {accuracy*100:.1f}%\n")
    print(f"  {'Class':<6}  {'TP':>7}  {'FP':>7}  {'FN':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*8}")

    f1_scores = []
    for ss in SS3_LABELS:
        cluster = ss_to_cluster[ss]
        tp = ct_ss3.get((cluster, ss), 0)
        # FP: other true-SS residues predicted as this class (other SS in same cluster row)
        fp = sum(ct_ss3.get((cluster, other), 0) for other in SS3_LABELS if other != ss)
        # FN: true-ss residues predicted as a different class (this SS column, other cluster rows)
        fn = sum(ct_ss3.get((other_cl, ss), 0) for other_cl in CLUSTER_LABELS if other_cl != cluster)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else float('nan')
        f1_scores.append(f1)
        print(f"  {ss:<6}  {tp:>7}  {fp:>7}  {fn:>7}  {prec*100:>9.1f}%  {recall*100:>7.1f}%  {f1*100:>7.1f}%")

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n  Macro-F1: {macro_f1*100:.1f}%")

    # weighted F1
    class_totals = {ss: sum(ct_ss3.get((cl, ss), 0) for cl in CLUSTER_LABELS) for ss in SS3_LABELS}
    weighted_f1 = sum(f1_scores[i] * class_totals[ss] for i, ss in enumerate(SS3_LABELS)) / grand_total
    print(f"  Weighted-F1: {weighted_f1*100:.1f}%")

    # ── save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        output = {
            'settings': {
                'input': args.input,
                'min_peak_pct': args.min_peak_pct,
                'n_init': args.n_init,
                'seed': args.seed,
            },
            'contingency_ss3':  {f"{r}_{c}": ct_ss3.get((r,c),0)   for r in CLUSTER_LABELS for c in SS3_LABELS},
            'enrichment_ss3':   {f"{r}_{c}": enrich_ss3[r][c]       for r in CLUSTER_LABELS for c in SS3_LABELS},
            'contingency_q8':   {f"{r}_{c}": ct_q8.get((r,c),0)    for r in CLUSTER_LABELS for c in Q8_LABELS},
            'enrichment_q8':    {f"{r}_{c}": enrich_q8[r][c]        for r in CLUSTER_LABELS for c in Q8_LABELS},
            'enrichment_rsa':   {f"{r}_{c}": enrich_rsa[r][c]       for r in CLUSTER_LABELS for c in RSA_LABELS},
            'enrichment_dis':   {f"{r}_{c}": enrich_dis[r][c]       for r in CLUSTER_LABELS for c in DIS_LABELS},
            'contingency_neq':  {f"{r}_{c}": ct_neq.get((r,c),0)    for r in CLUSTER_LABELS for c in NEQ_LABELS},
            'enrichment_neq':   {f"{r}_{c}": enrich_neq[r][c]        for r in CLUSTER_LABELS for c in NEQ_LABELS},
            'enrichment_neqr':  {f"{r}_{c}": enrich_neqr[r][c]       for r in CLUSTER_LABELS for c in NEQR_LABELS},
            'proteins': records,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
