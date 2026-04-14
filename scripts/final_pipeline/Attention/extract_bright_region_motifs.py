#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract sequence motifs from bright regions in NxN attention matrices,
split by secondary structure type of the blob's centre residue.

Key design decisions (updated after cluster analysis):
  - Blob detection uses col_max (max attention received by each column from
    any query) instead of raw cell values.  This isolates residues that are
    genuine key targets rather than just being in a dense row.
  - Threshold is percentile-based on col_max within each protein
    (--min_peak_pct, default 25 = top 25% brightest key columns).
    Protein-agnostic; adapts to per-protein attention contrast.
  - Every detected blob centre is tagged with its SS type (H/E/C) from the
    'ss_pred' field, and all counters/PWMs are maintained separately for
    Coil (PatA), Helix (PatB), and Strand (Background).

Outputs (per SS type H/E/C, plus "ALL"):
  - bright_region_motifs.csv            all blobs with SS label
  - motif_frequency_col_{SS}.csv        key-column motif frequencies
  - col_kmer_brightness_{SS}.csv        k-mer enrichment table
  - pwm_coil.csv / pwm_helix.csv        positional weight matrices
  - motif_summary.json

Usage
-----
python extract_bright_region_motifs.py \\
    --attention_json  ../../../data/attn_bilstm_f1-4_nsp3_neq.json \\
    --output_dir      ../../../data/motif_analysis \\
    [--min_peak_pct   25] \\
    [--min_component_size 4] \\
    [--motif_len 7] \\
    [--kmer_len 4]
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract motifs from bright attention regions, split by SS type")
    p.add_argument("--attention_json", required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--matrix_key",     default="attention_weights")
    p.add_argument("--min_peak_pct",   type=float, default=25.0,
                   help="Only treat a column as a bright key if its col_max is in the "
                        "top N%% within this protein (default 25 = top quarter).")
    p.add_argument("--min_component_size", type=int, default=4)
    p.add_argument("--motif_len",      type=int, default=7,
                   help="Window length for motif/PWM extraction (default 7, use odd numbers).")
    p.add_argument("--kmer_len",       type=int, default=4)
    p.add_argument("--exclude_diagonal_window", type=int, default=0)
    return p.parse_args()


def extract_motif(sequence: str, center_idx: int, motif_len: int) -> str:
    if motif_len <= 0:
        return ""
    L = len(sequence)
    start = center_idx - (motif_len // 2)
    end   = start + motif_len
    if start < 0:
        end  += -start
        start = 0
    if end > L:
        start -= (end - L)
        end = L
    start = max(0, start)
    return sequence[start:end]


def iter_kmers(sequence: str, k: int):
    if k <= 0 or len(sequence) < k:
        return
    for i in range(len(sequence) - k + 1):
        yield sequence[i:i + k]


def neighbors8(r: int, c: int, n: int) -> List[Tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                out.append((rr, cc))
    return out


def connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    n = mask.shape[0]
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors8(rr, cc, n):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


# ── PWM helpers ───────────────────────────────────────────────────────────────

def add_to_pwm(pwm: np.ndarray, bg_pwm: np.ndarray,
               sequence: str, center_idx: int, motif_len: int,
               weight: float = 1.0) -> None:
    """Add one occurrence to a PWM (shape motif_len × 20) with given weight."""
    L = len(sequence)
    half = motif_len // 2
    for pos in range(motif_len):
        seq_idx = center_idx - half + pos
        if 0 <= seq_idx < L:
            aa = sequence[seq_idx]
            if aa in AA20:
                col = AA20.index(aa)
                pwm[pos, col] += weight


def pwm_to_df(pwm: np.ndarray, bg_pwm: np.ndarray, motif_len: int) -> pd.DataFrame:
    """
    Normalise PWM to frequencies and compute enrichment over background.
    Returns a DataFrame with columns: position, AA, frequency, bg_frequency, enrichment.
    """
    rows = []
    pwm_totals = pwm.sum(axis=1, keepdims=True)
    bg_totals  = bg_pwm.sum(axis=1, keepdims=True)
    pwm_freq   = pwm / np.maximum(pwm_totals, 1e-9)
    bg_freq    = bg_pwm / np.maximum(bg_totals, 1e-9)
    half = motif_len // 2
    for pos in range(motif_len):
        for ai, aa in enumerate(AA20):
            f  = float(pwm_freq[pos, ai])
            bf = float(bg_freq[pos, ai])
            rows.append({
                'position':      pos - half,   # 0-centred; −half = N-terminal end
                'AA':            aa,
                'frequency':     f,
                'bg_frequency':  bf,
                'enrichment':    f / bf if bf > 1e-9 else float('nan'),
            })
    return pd.DataFrame(rows)


def print_pwm_logo(pwm_df: pd.DataFrame, top_n: int = 3, label: str = "") -> None:
    """Print a text summary of the top enriched AAs per position."""
    print(f"\nPWM consensus ({label}):")
    print(f"  {'pos':>4}  {'top AAs (enrichment)':}")
    for pos, grp in pwm_df.groupby('position'):
        top = grp.sort_values('enrichment', ascending=False).head(top_n)
        summary = "  ".join(
            f"{r['AA']}({r['enrichment']:.2f})"
            for _, r in top.iterrows()
            if not np.isnan(r['enrichment']) and r['frequency'] > 0.01
        )
        print(f"  {pos:>4}  {summary}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.attention_json, "r") as f:
        data = json.load(f)

    SS_TYPES = ['H', 'E', 'C', 'ALL']

    # per-SS-type accumulators
    col_counter        = {s: Counter() for s in SS_TYPES}
    col_kmer_count     = {s: Counter() for s in SS_TYPES}
    col_kmer_bsum      = {s: Counter() for s in SS_TYPES}
    bg_kmer_count      = Counter()
    bg_total_windows   = 0

    # PWMs: shape (motif_len, 20)  — one for bright-attended and one for background
    pwm_bright  = {s: np.zeros((args.motif_len, 20)) for s in SS_TYPES}
    pwm_bg_all  = np.zeros((args.motif_len, 20))   # background from all sequences

    rows       = []
    n_total    = 0
    n_used     = 0
    n_skipped  = 0

    for rec in data:
        n_total += 1
        seq_id = str(rec.get("name", ""))
        seq    = str(rec.get("sequence", ""))
        L      = len(seq)

        # background k-mer and PWM from all sequences
        for kmer in iter_kmers(seq, args.kmer_len):
            bg_kmer_count[kmer] += 1
            bg_total_windows += 1
        for ci in range(L):
            add_to_pwm(pwm_bg_all, pwm_bg_all, seq, ci, args.motif_len, weight=1.0)

        if args.matrix_key not in rec:
            n_skipped += 1
            continue

        mat = np.asarray(rec[args.matrix_key], dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] != L or L == 0:
            n_skipped += 1
            continue

        ss_list = rec.get("ss_pred", [None] * L)

        # ── blob detection on col_max ─────────────────────────────────────────
        # col_max[j] = max_i A[i,j]: the single largest weight key column j ever receives.
        # This is the correct signal for "residue j is a genuine attention target"
        # because softmax rows make row-sum constant — only column concentration matters.
        work = mat.copy()
        if args.exclude_diagonal_window > 0:
            ii, jj = np.indices((L, L))
            work[np.abs(ii - jj) <= args.exclude_diagonal_window] = np.nan

        col_max = np.nanmax(work, axis=0)   # shape (L,)

        # Percentile threshold on col_max (protein-agnostic)
        cutoff_rank = int(L * (1.0 - args.min_peak_pct / 100.0))
        cutoff_rank = max(0, min(cutoff_rank, L - 1))
        threshold   = float(np.sort(col_max)[cutoff_rank])

        # Build 2-D bright mask: cell (i,j) is bright when column j passes threshold
        # so the blob naturally captures all queries that attend to a bright key column.
        bright_cols = col_max >= threshold
        mask = np.zeros((L, L), dtype=bool)
        for j in range(L):
            if bright_cols[j]:
                mask[:, j] = np.isfinite(work[:, j]) & (work[:, j] >= threshold)

        comps = connected_components(mask)
        comps = [c for c in comps if len(c) >= args.min_component_size]

        protein_col_brightness: dict = {}   # col_idx → max bright value (deduplicated)

        if not comps:
            n_used += 1
            continue

        n_used += 1

        for cid, comp in enumerate(comps, start=1):
            rr = np.array([p[0] for p in comp], dtype=int)
            cc = np.array([p[1] for p in comp], dtype=int)

            r_center = int(np.round(rr.mean()))
            c_center = int(np.round(cc.mean()))

            # SS label of the key-column centre (what is being attended to)
            col_ss = ss_list[c_center] if c_center < len(ss_list) else None
            if col_ss not in ('H', 'E', 'C'):
                col_ss = None

            col_motif = extract_motif(seq, c_center, args.motif_len)
            vals      = mat[rr, cc]

            # update counters for this SS type and for ALL
            for bucket in ([col_ss, 'ALL'] if col_ss else ['ALL']):
                col_counter[bucket][col_motif] += 1

            rows.append({
                "seq_id":               seq_id,
                "component_id":         cid,
                "component_size":       int(len(comp)),
                "col_max_threshold":    threshold,
                "row_center_idx":       r_center,
                "col_center_idx":       c_center,
                "col_ss":               col_ss,
                "col_motif":            col_motif,
                "row_span":             f"{int(rr.min())}-{int(rr.max())}",
                "col_span":             f"{int(cc.min())}-{int(cc.max())}",
                "component_mean_value": float(np.mean(vals)),
                "component_max_value":  float(np.max(vals)),
            })

            # accumulate col_max for k-mer and PWM
            for c_idx, v in zip(cc.tolist(), vals.tolist()):
                c_idx = int(c_idx)
                vv    = float(v)
                prev  = protein_col_brightness.get(c_idx)
                if prev is None or vv > prev:
                    protein_col_brightness[c_idx] = vv

        # k-mer + PWM accumulation (deduplicated per-protein)
        for c_idx, v in protein_col_brightness.items():
            kmer   = extract_motif(seq, int(c_idx), args.kmer_len)
            c_ss   = ss_list[c_idx] if c_idx < len(ss_list) else None
            if c_ss not in ('H', 'E', 'C'):
                c_ss = None
            if len(kmer) == args.kmer_len:
                for bucket in ([c_ss, 'ALL'] if c_ss else ['ALL']):
                    col_kmer_count[bucket][kmer] += 1
                    col_kmer_bsum[bucket][kmer]  += float(v)
            # PWM
            for bucket in ([c_ss, 'ALL'] if c_ss else ['ALL']):
                add_to_pwm(pwm_bright[bucket], pwm_bg_all, seq, c_idx, args.motif_len, weight=float(v))

    # ── save outputs ───────────────────────────────────────────────────────────

    # All components
    all_df = pd.DataFrame(rows)
    all_df.to_csv(os.path.join(args.output_dir, "bright_region_motifs.csv"), index=False)

    def save_counter(counter: Counter, out_name: str, key_name: str):
        total = sum(counter.values())
        recs  = [
            {key_name: k, "count": v, "fraction": (v / total if total else 0.0)}
            for k, v in counter.items()
        ]
        df = pd.DataFrame(recs).sort_values("count", ascending=False) if recs \
             else pd.DataFrame(columns=[key_name, "count", "fraction"])
        df.to_csv(os.path.join(args.output_dir, out_name), index=False)

    for ss in SS_TYPES:
        suffix   = ss.lower().replace('all', 'all')
        save_counter(col_counter[ss], f"motif_frequency_col_{suffix}.csv", "col_motif")

    # k-mer enrichment tables per SS type
    for ss in SS_TYPES:
        suffix     = ss.lower()
        total_hits = sum(col_kmer_count[ss].values())
        kmer_rows  = []
        for kmer, n_hits in col_kmer_count[ss].items():
            bsum     = float(col_kmer_bsum[ss][kmer])
            bg_cnt   = int(bg_kmer_count.get(kmer, 0))
            obs_frac = (n_hits / total_hits) if total_hits > 0 else 0.0
            bg_frac  = (bg_cnt / bg_total_windows) if bg_total_windows > 0 else 0.0
            kmer_rows.append({
                "col_kmer":                    kmer,
                "ss_type":                     ss,
                "n_unique_col_hits":            int(n_hits),
                "observed_fraction":           obs_frac,
                "bg_count":                    bg_cnt,
                "bg_fraction":                 bg_frac,
                "enrichment_over_bg":          (obs_frac / bg_frac) if bg_frac > 1e-9 else float('nan'),
                "brightness_sum":              bsum,
                "brightness_mean_per_col":     (bsum / n_hits) if n_hits > 0 else 0.0,
            })
        kmer_df = pd.DataFrame(kmer_rows)
        if not kmer_df.empty:
            kmer_df = kmer_df.sort_values("brightness_sum", ascending=False)
        kmer_df.to_csv(os.path.join(args.output_dir, f"col_kmer_brightness_{suffix}.csv"), index=False)

    # ── PWMs ──────────────────────────────────────────────────────────────────
    # Save and print for Coil (PatA) and Helix (PatB)
    for ss, label in [('C', 'Coil_PatA'), ('H', 'Helix_PatB'), ('E', 'Strand_Bg'), ('ALL', 'ALL')]:
        pwm_df = pwm_to_df(pwm_bright[ss], pwm_bg_all, args.motif_len)
        pwm_df.to_csv(os.path.join(args.output_dir, f"pwm_{label.lower()}.csv"), index=False)
        if ss in ('C', 'H'):
            print_pwm_logo(pwm_df, top_n=3, label=label)

    # ── summary ───────────────────────────────────────────────────────────────
    summary = {
        "n_records_total":            n_total,
        "n_records_used":             n_used,
        "n_records_skipped":          n_skipped,
        "matrix_key":                 args.matrix_key,
        "blob_detection":             "col_max percentile",
        "min_peak_pct":               args.min_peak_pct,
        "min_component_size":         args.min_component_size,
        "motif_len":                  args.motif_len,
        "kmer_len":                   args.kmer_len,
        "exclude_diagonal_window":    args.exclude_diagonal_window,
        "n_bright_components_total":  int(len(rows)),
        "components_by_ss":           {
            ss: int(sum(1 for r in rows if r.get("col_ss") == ss))
            for ss in ('H', 'E', 'C')
        },
    }

    with open(os.path.join(args.output_dir, "motif_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.  {len(rows)} components across {n_used} proteins.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
