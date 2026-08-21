#!/usr/bin/env python3
"""
fetch_dynamine_mdcath.py

Fetch DynaMine backbone S^2 predictions for the MMseqs2-strict, ATLAS-disjoint
mdCATH 320K sequence set, reusing the Bio2Byte msatools API client from
benchmark_vs_baselines.py.

Usage:
    python fetch_dynamine_mdcath.py \\
        --fasta results/benchmark_pegasus_mdcath/mdcath_320K_strict.fasta \\
        --output_dir results/benchmark_pegasus_mdcath

Output:
    <output_dir>/cache_dynamine_mdcath.csv   (name, res_idx, dynamine_bb)
"""

import argparse
from pathlib import Path

from benchmark_vs_baselines import parse_fasta, run_dynamine_api, _random_token


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="results/benchmark_pegasus_mdcath/mdcath_320K_strict.fasta")
    ap.add_argument("--output_dir", default="results/benchmark_pegasus_mdcath")
    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--poll_interval", type=int, default=20)
    ap.add_argument("--max_polls", type=int, default=90)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache_dynamine_mdcath.csv"

    if cache_path.exists():
        print(f"[dynamine] Cached — {cache_path} already exists. Nothing to do.")
        return

    fasta_records = parse_fasta(args.fasta)
    print(f"[dynamine] Loaded {len(fasta_records)} sequences from {args.fasta}")

    token = _random_token(10)
    print(f"[dynamine] Submitting to Bio2Byte msatools API (token: {token}) "
          f"in batches of {args.batch_size} …")
    df = run_dynamine_api(
        fasta_records,
        token=token,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        max_polls=args.max_polls,
    )
    df.to_csv(cache_path, index=False)
    n_valid = df["dynamine_bb"].notna().sum()
    n_total = len(df)
    print(f"[dynamine] Done — {n_valid}/{n_total} valid residue scores; "
          f"cached to {cache_path}")


if __name__ == "__main__":
    main()
