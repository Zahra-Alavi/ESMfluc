#!/usr/bin/env python3
"""Fetch only the small Neq members needed to disambiguate duplicate sequences.

Requires ``remotezip`` (pip install remotezip). HTTP byte ranges are used, so
the large trajectory members of each ATLAS analysis archive are not downloaded.
"""

import argparse
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-csv", type=Path, default=Path("../../data/atlas.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    atlas = pd.read_csv(args.atlas_csv)
    duplicates = atlas[atlas.duplicated("seqres", keep=False)].name.sort_values().tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in duplicates:
        member = f"{name}_Neq.tsv"
        destination = args.output_dir / member
        if destination.exists():
            continue
        url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{name}/{name}_analysis.zip"
        print(f"Fetching {member}")
        with RemoteZip(url) as archive:
            destination.write_bytes(archive.read(member))
    print(f"Wrote {len(duplicates)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()
