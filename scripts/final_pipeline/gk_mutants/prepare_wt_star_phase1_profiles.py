#!/usr/bin/env python3
"""Prepare compact WT* signed-influence profiles for the locked Phase 1 code."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from weinreb_analysis_common import load_seed, read_manifest


SCHEMA_VERSION = "esmfluc.signed_contributions.v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="esm3_top28_bilstm_attn")
    parser.add_argument("--protein", default="WT_star")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_json_gz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def main() -> None:
    args = arguments()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    rows = read_manifest(args.manifest, args.condition, seeds)
    manifest_rows = []
    audit_rows = []
    reference_sequence = None

    for row in rows:
        seed = int(row["seed"])
        proteins = load_seed(row)
        if args.protein not in proteins:
            raise ValueError(f"{args.protein} is absent for seed {seed}")
        protein = proteins[args.protein]
        if reference_sequence is None:
            reference_sequence = protein.sequence
        elif protein.sequence != reference_sequence:
            raise ValueError(f"{args.protein} sequence differs across seeds")

        contribution = np.asarray(protein.contribution, dtype=np.float64)
        attention = np.asarray(protein.attention, dtype=np.float64)
        length = len(protein.sequence)
        if contribution.shape != (length, length) or attention.shape != (length, length):
            raise ValueError(
                f"seed {seed}: expected {(length, length)}, got "
                f"C={contribution.shape}, A={attention.shape}"
            )
        if not np.isfinite(contribution).all() or not np.isfinite(attention).all():
            raise ValueError(f"seed {seed}: matrices contain nonfinite values")

        influence = contribution.mean(axis=0)
        attention_mean = attention.mean(axis=0)
        if np.any(attention_mean <= 0):
            raise ValueError(f"seed {seed}: attention-column mean is not positive")
        evidence = influence / attention_mean
        decomposition_error = float(
            np.max(np.abs(contribution - attention * evidence[None, :]))
        )
        identity_error = float(
            np.max(np.abs(influence - evidence * attention_mean))
        )
        row_sum_error = float(np.max(np.abs(attention.sum(axis=1) - 1.0)))
        if decomposition_error > 5e-6 or identity_error > 5e-10:
            raise ValueError(
                f"seed {seed}: contribution decomposition failed: "
                f"matrix={decomposition_error}, profile={identity_error}"
            )

        output_path = (
            args.output_dir / "profiles" / f"seed_{seed}_wt_star_signed_profile.json.gz"
        ).resolve()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "condition": args.condition,
            "seed": seed,
            "split": "gk_wt_star",
            "protein_count": 1,
            "residue_count": length,
            "source_manifest": str(args.manifest.resolve()),
            "source_attention_json": str(Path(row["attention_json"]).resolve()),
            "source_contribution_npz": str(
                Path(row["logit_contributions_npz"]).resolve()
            ),
            "proteins": [{
                "name": args.protein,
                "sequence": protein.sequence,
                "length": length,
                "intrinsic_signed_evidence": evidence.tolist(),
                "signed_column_influence": influence.tolist(),
                "attention_column_mean": attention_mean.tolist(),
            }],
        }
        write_json_gz(output_path, payload)
        manifest_rows.append({
            "condition": args.condition,
            "seed": seed,
            "split": "gk_wt_star",
            "json_gz": str(output_path),
        })
        audit_rows.append({
            "condition": args.condition,
            "seed": seed,
            "protein": args.protein,
            "length": length,
            "attention_row_sum_max_abs_error": row_sum_error,
            "contribution_decomposition_max_abs_error": decomposition_error,
            "profile_identity_max_abs_error": identity_error,
            "passes": True,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "profiles_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "seed", "split", "json_gz"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    audit_path = args.output_dir / "profile_preparation_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    print(json.dumps({
        "profiles": len(manifest_rows),
        "manifest": str(manifest_path.resolve()),
        "audit": str(audit_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
