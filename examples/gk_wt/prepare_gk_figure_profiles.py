"""Prepare one GK sequence's audited profiles from existing exact predictions.

Writes only new, figure-specific compact profiles; never rewrites source runs.
The optional seed-mean field is calculated directly here so the general band
commands can operate without modifying any existing JSON files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from esmfluc.visualization.plot_seed_averaged_contributions import read_sequence, source_digest
from esmfluc.visualization.visualize_exact_logit_contributions import (
    iter_json_array, load_selected_seed, select_manifest_runs,
)
from esmfluc.analysis.bands.contribution_utils import reconstruct_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fasta-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protein", default="WT")
    parser.add_argument("--condition", default="esm3_top28_bilstm_attn")
    parser.add_argument("--split", default="gk_wt")
    args = parser.parse_args()
    sequence = read_sequence(args.fasta_file, args.protein)
    runs = select_manifest_runs(args.manifest, args.condition, [1, 2, 3])
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    seed_profiles = []
    audit = []
    sources = [args.manifest, args.fasta_file, Path(__file__)]
    for run in runs:
        payload = load_selected_seed(run, [args.protein], atol=1e-5, rtol=1e-5)[args.protein]
        if payload.sequence != sequence:
            raise ValueError(f"Seed {run.seed}: saved prediction differs from supplied FASTA")
        record = next(r for r in iter_json_array(run.attention_path) if r["name"] == args.protein)
        attention = np.asarray(record["attention_weights"], dtype=float)
        contribution = payload.contribution
        if attention.shape != contribution.shape or not np.isfinite(attention).all():
            raise ValueError("Invalid saved attention matrix")
        if (attention < 0).any():
            raise ValueError("Saved attention matrix has negative weights")
        evidence, matrix_error = reconstruct_evidence(contribution, attention)
        influence = contribution.mean(axis=0)
        breadth = attention.mean(axis=0)
        identity_error = float(np.max(np.abs(influence - evidence * breadth)))
        row_error = float(np.max(np.abs(attention.sum(axis=1) - 1)))
        if matrix_error > 1e-6 or identity_error > 1e-8 or row_error > 1e-5:
            raise ValueError(f"Seed {run.seed}: exact profile decomposition audit failed")
        seed_profiles.append({
            "name": args.protein, "sequence": sequence, "length": len(sequence),
            "intrinsic_signed_evidence": evidence.tolist(),
            "signed_column_influence": influence.tolist(),
            "attention_column_mean": breadth.tolist(),
        })
        audit.append({"seed": run.seed, "decomposition_max_abs_error": matrix_error,
                      "profile_identity_max_abs_error": identity_error,
                      "attention_row_sum_max_abs_error": row_error})
        sources.extend([run.attention_path, run.contribution_path])
    average = np.mean([p["signed_column_influence"] for p in seed_profiles], axis=0)
    output.mkdir(parents=True)
    manifest = []
    for run, profile in zip(runs, seed_profiles):
        # Maintain field order accepted by the compact streaming detector.
        profile = {**{k: v for k, v in profile.items() if k != "attention_column_mean"},
                   "seed_averaged_signed_column_influence": average.tolist(),
                   "attention_column_mean": profile["attention_column_mean"]}
        path = output / "profiles" / f"seed_{run.seed}" / "signed_profile.json.gz"
        path.parent.mkdir(parents=True)
        document = {
            "schema_version": "esmfluc.signed_contributions.v1",
            "condition": args.condition, "seed": run.seed, "split": args.split,
            "seed_averaged_signed_column_influence_definition": {
                "formula": "mean_seed_I_j = (I_j_seed1 + I_j_seed2 + I_j_seed3) / 3",
                "source_seeds": [1, 2, 3],
            },
            "protein_count": 1, "residue_count": len(sequence),
            "source_manifest": str(args.manifest.resolve()), "proteins": [profile],
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"), allow_nan=False)
        manifest.append({"condition": args.condition, "seed": run.seed,
                         "split": args.split, "json_gz": str(path)})
    mean_manifest = [{**manifest[0], "seed": "average", "source_seeds": "1,2,3",
                      "influence_field": "seed_averaged_signed_column_influence"}]
    for filename, rows in (("profiles_manifest.tsv", manifest),
                           ("averaged_profiles_manifest.tsv", mean_manifest)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    report = {"protein": args.protein, "sequence": sequence, "length": len(sequence),
              "condition": args.condition, "seeds": [1, 2, 3], "seed_audits": audit,
              "sources": [source_digest(p) for p in sources]}
    (output / "profile_preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protein": args.protein, "length": len(sequence), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
