#!/usr/bin/env python3
"""Write audited WT* profile JSONs for all six model conditions."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from gk_v2_common import CONDITIONS, SEEDS, manifest_rows, reconstruct_evidence
from weinreb_analysis_common import load_seed


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protein", default="WT_star")
    return parser.parse_args()


def main():
    args = arguments()
    source = manifest_rows(args.manifest)
    manifest = []
    audits = []
    for condition in CONDITIONS:
        reference_sequence = None
        for seed in SEEDS:
            payloads = load_seed(source[(condition, seed)])
            protein = payloads[args.protein]
            if reference_sequence is None:
                reference_sequence = protein.sequence
            if protein.sequence != reference_sequence:
                raise ValueError(f"{condition}: WT* differs across seeds")
            c = np.asarray(protein.contribution, dtype=float)
            a = np.asarray(protein.attention, dtype=float)
            evidence, matrix_error = reconstruct_evidence(c, a)
            influence = c.mean(axis=0)
            breadth = a.mean(axis=0)
            identity_error = float(np.max(np.abs(influence - evidence * breadth)))
            row_error = float(np.max(np.abs(a.sum(axis=1) - 1.0)))
            if matrix_error > 1e-6 or identity_error > 1e-8 or row_error > 1e-5:
                raise ValueError(f"{condition} seed {seed}: profile audit failed")
            out = (
                args.output_dir / "profiles" / condition / f"seed_{seed}"
                / "wt_star_signed_profile.json.gz"
            ).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            document = {
                "schema_version": "esmfluc.signed_contributions.v1",
                "condition": condition,
                "seed": seed,
                "split": "gk_wt_star",
                "protein_count": 1,
                "residue_count": len(protein.sequence),
                "source_manifest": str(args.manifest.resolve()),
                "proteins": [{
                    "name": args.protein,
                    "sequence": protein.sequence,
                    "length": len(protein.sequence),
                    "intrinsic_signed_evidence": evidence.tolist(),
                    "signed_column_influence": influence.tolist(),
                    "attention_column_mean": breadth.tolist(),
                }],
            }
            temporary = out.with_name(f".{out.name}.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as h:
                json.dump(document, h, separators=(",", ":"), allow_nan=False)
            temporary.replace(out)
            manifest.append({
                "condition": condition, "seed": seed, "split": "gk_wt_star",
                "json_gz": str(out), "protein_count": 1,
                "residue_count": len(protein.sequence),
            })
            audits.append({
                "condition": condition, "seed": seed, "protein": args.protein,
                "length": len(protein.sequence),
                "attention_row_sum_max_abs_error": row_error,
                "contribution_decomposition_max_abs_error": matrix_error,
                "profile_identity_max_abs_error": identity_error,
                "passes": True,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows, delimiter in (
        (args.output_dir / "profiles_manifest.tsv", manifest, "\t"),
        (args.output_dir / "profile_preparation_audit.csv", audits, ","),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"profiles": len(manifest), "audit_rows": len(audits)}, indent=2))


if __name__ == "__main__":
    main()
