#!/usr/bin/env python3
"""Map stable WT* bands to paper coordinates and label GK mutation sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from weinreb_analysis_common import mutation_sites, parse_position_map, read_fasta


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-bands", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--position-map", type=Path, required=True)
    parser.add_argument("--reference", default="WT_star")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mapped_position(position_map: dict, fasta_position: int):
    value = position_map.get(int(fasta_position))
    if value is None:
        return None, None
    return value


def main() -> None:
    args = arguments()
    fasta = read_fasta(args.fasta)
    if args.reference not in fasta:
        raise ValueError(f"Reference {args.reference!r} is absent from FASTA")
    reference = fasta[args.reference]
    position_map = parse_position_map(args.position_map, args.reference)
    bands = pd.read_csv(args.stable_bands)
    bands = bands[
        (bands["protein"].astype(str) == args.reference)
        & bands["primary_stable_band"].astype(bool)
    ].copy()
    if bands.empty:
        raise ValueError(f"No primary stable bands found for {args.reference}")
    if set(bands["sign"].astype(int)) - {-1, 1}:
        raise ValueError("Stable-band signs must be -1 or +1")

    mapped_rows = []
    for row in bands.itertuples(index=False):
        start = int(row.start_residue_1based)
        end = int(row.end_residue_1based_inclusive)
        apex = int(row.apex_residue_1based)
        start_chain, start_paper = mapped_position(position_map, start)
        end_chain, end_paper = mapped_position(position_map, end)
        apex_chain, apex_paper = mapped_position(position_map, apex)
        mapped_rows.append({
            **row._asdict(),
            "pdb_chain_at_start": start_chain,
            "paper_position_at_start": start_paper,
            "pdb_chain_at_end": end_chain,
            "paper_position_at_end": end_paper,
            "pdb_chain_at_apex": apex_chain,
            "paper_position_at_apex": apex_paper,
        })
    mapped_bands = pd.DataFrame(mapped_rows)

    mutation_rows = []
    for variant, sequence in fasta.items():
        if variant in {args.reference, "WT"}:
            continue
        if len(sequence) != len(reference):
            raise ValueError(f"{variant}: length differs from {args.reference}")
        sites = mutation_sites(sequence, reference)
        if not sites:
            continue
        for site_number, fasta_position in enumerate(sites, start=1):
            containing = mapped_bands[
                (mapped_bands["start_residue_1based"] <= fasta_position)
                & (mapped_bands["end_residue_1based_inclusive"] >= fasta_position)
            ].copy()
            distances = np.abs(
                mapped_bands["apex_residue_1based"].astype(int).to_numpy()
                - fasta_position
            )
            nearest_index = int(np.argmin(distances))
            nearest = mapped_bands.iloc[nearest_index]
            chain, paper_position = mapped_position(position_map, fasta_position)
            if containing.empty:
                location = "outside_all_stable_bands"
                band_id = None
                band_sign = None
                band_label = None
                band_rank = None
                distance_to_containing_apex = None
            else:
                if len(containing) != 1:
                    raise ValueError(
                        f"{variant} position {fasta_position} lies in "
                        f"{len(containing)} stable bands"
                    )
                band = containing.iloc[0]
                band_sign = int(band["sign"])
                location = (
                    "inside_stable_positive_band"
                    if band_sign == 1 else "inside_stable_negative_band"
                )
                band_id = band["band_id"]
                band_label = band["label"]
                band_rank = band["absolute_apex_rank_within_protein"]
                distance_to_containing_apex = abs(
                    int(band["apex_residue_1based"]) - fasta_position
                )
            mutation_rows.append({
                "variant": variant,
                "site_number_within_variant": site_number,
                "reference": args.reference,
                "fasta_position_1based": fasta_position,
                "paper_position": paper_position,
                "pdb_chain": chain,
                "reference_amino_acid": reference[fasta_position - 1],
                "mutant_amino_acid": sequence[fasta_position - 1],
                "band_location": location,
                "inside_any_stable_band": not containing.empty,
                "containing_band_sign": band_sign,
                "containing_band_label": band_label,
                "containing_band_id": band_id,
                "containing_band_absolute_apex_rank": band_rank,
                "distance_to_containing_band_apex_residues": distance_to_containing_apex,
                "nearest_stable_band_sign": int(nearest["sign"]),
                "nearest_stable_band_label": nearest["label"],
                "nearest_stable_band_id": nearest["band_id"],
                "nearest_stable_band_apex_fasta_position_1based": int(
                    nearest["apex_residue_1based"]
                ),
                "nearest_stable_band_apex_paper_position": nearest[
                    "paper_position_at_apex"
                ],
                "distance_to_nearest_stable_band_apex_residues": int(
                    distances[nearest_index]
                ),
            })

    mutation_labels = pd.DataFrame(mutation_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapped_path = args.output_dir / "stable_bands_with_paper_coordinates.csv"
    label_path = args.output_dir / "mutation_band_labels.csv"
    summary_path = args.output_dir / "mutation_band_label_summary.csv"
    audit_path = args.output_dir / "labeling_audit.json"
    mapped_bands.to_csv(mapped_path, index=False)
    mutation_labels.to_csv(label_path, index=False)
    summary = (
        mutation_labels.groupby("band_location", dropna=False)
        .agg(mutation_sites=("variant", "size"), variants=("variant", "nunique"))
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    audit = {
        "reference": args.reference,
        "reference_length": len(reference),
        "stable_band_count": int(len(mapped_bands)),
        "stable_positive_band_count": int((mapped_bands["sign"] == 1).sum()),
        "stable_negative_band_count": int((mapped_bands["sign"] == -1).sum()),
        "mutation_variant_count": int(mutation_labels["variant"].nunique()),
        "mutation_site_count": int(len(mutation_labels)),
        "mapped_mutation_site_count": int(mutation_labels["paper_position"].notna().sum()),
        "all_mutation_sites_mapped": bool(mutation_labels["paper_position"].notna().all()),
        "all_sites_have_nearest_band": bool(
            mutation_labels["nearest_stable_band_id"].notna().all()
        ),
        "source_stable_bands": str(args.stable_bands.resolve()),
        "source_position_map": str(args.position_map.resolve()),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        **audit,
        "mapped_bands": str(mapped_path.resolve()),
        "mutation_labels": str(label_path.resolve()),
        "summary": str(summary_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
