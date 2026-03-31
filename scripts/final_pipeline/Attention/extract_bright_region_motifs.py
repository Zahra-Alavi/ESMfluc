#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract sequence motifs from bright regions in NxN attention/contact matrices.

For each protein record in JSON:
- read matrix from --matrix_key (default: attention_weights)
- define bright pixels by quantile threshold (or absolute threshold)
- find connected bright components (8-neighborhood)
- map each component to row/column sequence motifs around component center
- aggregate motif frequencies across proteins

Outputs:
- bright_region_motifs.csv          (all detected bright components)
- motif_frequency_pair.csv          (row_motif|col_motif counts)
- motif_frequency_row.csv           (row motif counts)
- motif_frequency_col.csv           (column motif counts)
- col_kmer_brightness.csv           (receiver-column k-mer ranking by bright-pixel intensity)
- col_kmer_background.csv           (background k-mer frequencies in all sequences)
- motif_summary.json                (overall stats)
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract and count motifs from bright matrix regions")
    p.add_argument("--attention_json", required=True, help="JSON with per-protein matrices and sequence")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--matrix_key", default="attention_weights", help="Matrix key in JSON record")
    p.add_argument("--quantile", type=float, default=0.95, help="Bright threshold quantile per protein (default 0.95)")
    p.add_argument("--absolute_threshold", type=float, default=None, help="Optional absolute threshold (overrides quantile if set)")
    p.add_argument("--min_component_size", type=int, default=4, help="Minimum bright component size in pixels")
    p.add_argument("--motif_len", type=int, default=5, help="Motif length extracted around component center")
    p.add_argument("--kmer_len", type=int, default=4, help="k-mer length for receiver-column ranking (default 4)")
    p.add_argument("--exclude_diagonal_window", type=int, default=0, help="Ignore pixels with |i-j| <= this window")
    return p.parse_args()


def extract_motif(sequence: str, center_idx: int, motif_len: int) -> str:
    if motif_len <= 0:
        return ""
    L = len(sequence)
    # Exact-length window centered on center_idx (works for odd/even motif_len).
    start = center_idx - (motif_len // 2)
    end = start + motif_len

    if start < 0:
        end += -start
        start = 0
    if end > L:
        start -= (end - L)
        end = L
    start = max(0, start)

    return sequence[start:end]


def iter_kmers(sequence: str, k: int):
    if k <= 0 or len(sequence) < k:
        return
    for i in range(0, len(sequence) - k + 1):
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.attention_json, "r") as f:
        data = json.load(f)

    pair_counter = Counter()
    row_counter = Counter()
    col_counter = Counter()
    col_kmer_count = Counter()
    col_kmer_brightness_sum = Counter()
    bg_kmer_count = Counter()
    bg_total_windows = 0
    rows = []

    n_total = 0
    n_used = 0
    n_skipped = 0

    for rec in data:
        n_total += 1
        seq_id = str(rec.get("name", ""))
        seq = str(rec.get("sequence", ""))

        # Background k-mer distribution from all sequences in the JSON.
        for kmer in iter_kmers(seq, args.kmer_len):
            bg_kmer_count[kmer] += 1
            bg_total_windows += 1

        if args.matrix_key not in rec:
            n_skipped += 1
            continue

        mat = np.asarray(rec[args.matrix_key], dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            n_skipped += 1
            continue

        L = mat.shape[0]
        if len(seq) != L or L == 0:
            n_skipped += 1
            continue

        work = mat.copy()
        if args.exclude_diagonal_window > 0:
            ii, jj = np.indices((L, L))
            work[np.abs(ii - jj) <= args.exclude_diagonal_window] = np.nan

        if args.absolute_threshold is not None:
            thr = float(args.absolute_threshold)
        else:
            finite_vals = work[np.isfinite(work)]
            if finite_vals.size == 0:
                n_skipped += 1
                continue
            thr = float(np.quantile(finite_vals, args.quantile))

        mask = np.isfinite(work) & (work >= thr)
        comps = connected_components(mask)
        comps = [c for c in comps if len(c) >= args.min_component_size]

        # Collect unique bright receiver columns per protein (deduplicate repeated rows -> same column).
        protein_col_brightness = {}

        if not comps:
            n_used += 1
            continue

        n_used += 1
        for cid, comp in enumerate(comps, start=1):
            rr = np.array([p[0] for p in comp], dtype=int)
            cc = np.array([p[1] for p in comp], dtype=int)

            r_center = int(np.round(rr.mean()))
            c_center = int(np.round(cc.mean()))

            row_motif = extract_motif(seq, r_center, args.motif_len)
            col_motif = extract_motif(seq, c_center, args.motif_len)

            pair_motif = f"{row_motif}|{col_motif}"

            vals = mat[rr, cc]

            pair_counter[pair_motif] += 1
            row_counter[row_motif] += 1
            col_counter[col_motif] += 1

            # Track unique bright columns per protein, keeping max bright value for each column.
            for c_idx, v in zip(cc.tolist(), vals.tolist()):
                c_idx = int(c_idx)
                vv = float(v)
                prev = protein_col_brightness.get(c_idx)
                if prev is None or vv > prev:
                    protein_col_brightness[c_idx] = vv

            rows.append(
                {
                    "seq_id": seq_id,
                    "component_id": cid,
                    "component_size": int(len(comp)),
                    "threshold": thr,
                    "row_center_idx": r_center,
                    "col_center_idx": c_center,
                    "row_motif": row_motif,
                    "col_motif": col_motif,
                    "pair_motif": pair_motif,
                    "row_span": f"{int(rr.min())}-{int(rr.max())}",
                    "col_span": f"{int(cc.min())}-{int(cc.max())}",
                    "component_mean_value": float(np.mean(vals)),
                    "component_max_value": float(np.max(vals)),
                }
            )

        # Receiver-column k-mer stats from unique bright columns per protein (x-axis emphasis).
        for c_idx, v in protein_col_brightness.items():
            kmer = extract_motif(seq, int(c_idx), args.kmer_len)
            if len(kmer) == args.kmer_len:
                col_kmer_count[kmer] += 1
                col_kmer_brightness_sum[kmer] += float(v)

    # Save all components
    all_df = pd.DataFrame(rows)
    all_csv = os.path.join(args.output_dir, "bright_region_motifs.csv")
    all_df.to_csv(all_csv, index=False)

    # Save frequency tables
    def save_counter(counter: Counter, out_name: str, key_name: str):
        total = sum(counter.values())
        records = [{key_name: k, "count": v, "fraction": (v / total if total else 0.0)} for k, v in counter.items()]
        if records:
            freq_df = pd.DataFrame(records).sort_values("count", ascending=False)
        else:
            freq_df = pd.DataFrame(columns=[key_name, "count", "fraction"])
        freq_df.to_csv(os.path.join(args.output_dir, out_name), index=False)

    save_counter(pair_counter, "motif_frequency_pair.csv", "pair_motif")
    save_counter(row_counter, "motif_frequency_row.csv", "row_motif")
    save_counter(col_counter, "motif_frequency_col.csv", "col_motif")

    # Receiver-column k-mer ranking by bright-pixel intensity.
    total_hits = sum(col_kmer_count.values())
    kmer_rows = []
    for kmer, n_hits in col_kmer_count.items():
        bsum = float(col_kmer_brightness_sum[kmer])
        bg_count = int(bg_kmer_count.get(kmer, 0))
        observed_fraction = (n_hits / total_hits) if total_hits > 0 else 0.0
        background_fraction = (bg_count / bg_total_windows) if bg_total_windows > 0 else 0.0

        kmer_rows.append(
            {
                "col_kmer": kmer,
                "kmer_len": int(args.kmer_len),
                "n_unique_col_hits": int(n_hits),
                "observed_fraction": observed_fraction,
                "bg_count": bg_count,
                "bg_fraction": background_fraction,
                "enrichment_over_bg": (observed_fraction / background_fraction) if background_fraction > 0 else np.nan,
                "brightness_sum": bsum,
                "brightness_mean_per_unique_col": (bsum / n_hits) if n_hits > 0 else 0.0,
                "brightness_per_bg_occurrence": (bsum / bg_count) if bg_count > 0 else np.nan,
            }
        )
    kmer_df = pd.DataFrame(kmer_rows)
    if not kmer_df.empty:
        kmer_df = kmer_df.sort_values(["brightness_sum", "n_unique_col_hits"], ascending=[False, False])
    else:
        kmer_df = pd.DataFrame(
            columns=[
                "col_kmer",
                "kmer_len",
                "n_unique_col_hits",
                "observed_fraction",
                "bg_count",
                "bg_fraction",
                "enrichment_over_bg",
                "brightness_sum",
                "brightness_mean_per_unique_col",
                "brightness_per_bg_occurrence",
            ]
        )
    kmer_csv = os.path.join(args.output_dir, "col_kmer_brightness.csv")
    kmer_df.to_csv(kmer_csv, index=False)

    # Save background k-mer frequencies.
    bg_rows = [
        {
            "col_kmer": k,
            "kmer_len": int(args.kmer_len),
            "bg_count": int(v),
            "bg_fraction": (v / bg_total_windows) if bg_total_windows > 0 else 0.0,
        }
        for k, v in bg_kmer_count.items()
    ]
    bg_df = pd.DataFrame(bg_rows)
    if not bg_df.empty:
        bg_df = bg_df.sort_values("bg_count", ascending=False)
    else:
        bg_df = pd.DataFrame(columns=["col_kmer", "kmer_len", "bg_count", "bg_fraction"])
    bg_csv = os.path.join(args.output_dir, "col_kmer_background.csv")
    bg_df.to_csv(bg_csv, index=False)

    summary = {
        "n_records_total": n_total,
        "n_records_used": n_used,
        "n_records_skipped": n_skipped,
        "matrix_key": args.matrix_key,
        "quantile": args.quantile,
        "absolute_threshold": args.absolute_threshold,
        "min_component_size": args.min_component_size,
        "motif_len": args.motif_len,
        "kmer_len": args.kmer_len,
        "exclude_diagonal_window": args.exclude_diagonal_window,
        "n_bright_components_total": int(len(rows)),
        "n_unique_pair_motifs": int(len(pair_counter)),
        "n_unique_row_motifs": int(len(row_counter)),
        "n_unique_col_motifs": int(len(col_counter)),
        "n_unique_col_kmers": int(len(col_kmer_count)),
        "n_unique_bg_kmers": int(len(bg_kmer_count)),
        "bg_total_windows": int(bg_total_windows),
        "col_kmer_counting": "unique_bright_columns_per_protein",
    }

    with open(os.path.join(args.output_dir, "motif_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done")
    print(f"Components CSV: {all_csv}")
    print(f"Pair motifs:    {os.path.join(args.output_dir, 'motif_frequency_pair.csv')}")
    print(f"Row motifs:     {os.path.join(args.output_dir, 'motif_frequency_row.csv')}")
    print(f"Col motifs:     {os.path.join(args.output_dir, 'motif_frequency_col.csv')}")
    print(f"Col k-mers:     {kmer_csv}")
    print(f"K-mer bg:       {bg_csv}")
    print(f"Summary:        {os.path.join(args.output_dir, 'motif_summary.json')}")


if __name__ == "__main__":
    main()
