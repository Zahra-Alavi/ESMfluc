#!/usr/bin/env python3
"""Audit all publication-v2 signed-contribution outputs and build a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default="results/publication_comparable_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.result_root).resolve()
    run_manifest = list(csv.DictReader((root / "manifest.tsv").open(), delimiter="\t"))
    runs = [row for row in run_manifest if row["architecture"] == "bilstm_attention"]
    if len(runs) != 18:
        raise ValueError(f"Expected 18 BiLSTM runs, found {len(runs)}")

    expected_counts = {
        split: len(pd.read_csv(root / f"{split}_data.csv"))
        for split in ("train", "validation", "test")
    }
    records = []
    errors = []
    for run in runs:
        output_dir = Path(run["run_dir"]).resolve() / "all_split_signed_contributions"
        summary_path = output_dir / "extraction_summary.json"
        if not summary_path.is_file():
            errors.append(f"Missing summary: {summary_path}")
            continue
        summary_doc = json.loads(summary_path.read_text())
        summaries = summary_doc.get(
            "completed_splits", summary_doc.get("completed_splits_this_invocation", [])
        )
        by_split = {item["split"]: item for item in summaries}
        for split, expected_count in expected_counts.items():
            output = output_dir / f"{split}_signed_contributions.json.gz"
            item = by_split.get(split)
            if item is None:
                errors.append(f"Missing {split} summary: {summary_path}")
                continue
            if not output.is_file():
                errors.append(f"Missing output: {output}")
                continue
            if item["protein_count"] != expected_count:
                errors.append(
                    f"{output}: {item['protein_count']} proteins, expected {expected_count}"
                )
            records.append({
                "condition": run["condition"],
                "seed": int(run["seed"]),
                "split": split,
                "json_gz": str(output),
                "protein_count": int(item["protein_count"]),
                "residue_count": int(item["residue_count"]),
                "max_reconstruction_abs_error": float(item["max_reconstruction_abs_error"]),
                "compressed_bytes": int(output.stat().st_size),
            })

    partials = list(root.glob("runs/**/all_split_signed_contributions/.*.part-*"))
    if partials:
        errors.append(f"Found {len(partials)} partial output files")
    if len(records) != 54:
        errors.append(f"Found {len(records)} completed split records, expected 54")
    if errors:
        raise SystemExit("Audit failed:\n  " + "\n  ".join(errors))

    manifest_path = root / "all_split_signed_contributions_manifest.tsv"
    temporary = manifest_path.with_name(f".{manifest_path.name}.part-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, manifest_path)

    audit = {
        "bilstm_run_count": len(runs),
        "split_json_count": len(records),
        "expected_proteins_per_split": expected_counts,
        "total_protein_inferences": sum(row["protein_count"] for row in records),
        "total_residue_inferences": sum(row["residue_count"] for row in records),
        "total_compressed_bytes": sum(row["compressed_bytes"] for row in records),
        "max_reconstruction_abs_error": max(
            row["max_reconstruction_abs_error"] for row in records
        ),
        "partial_file_count": len(partials),
        "manifest": str(manifest_path),
    }
    audit_path = root / "all_split_signed_contributions_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
