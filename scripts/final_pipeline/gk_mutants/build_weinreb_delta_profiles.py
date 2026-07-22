#!/usr/bin/env python3
"""Audit primary inputs and build seed-paired DeltaI profiles against WT_star."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from weinreb_analysis_common import (
    NEW_SEQUENCES, PRIMARY_CONDITION, PRIMARY_SEEDS, WEINREB_ANALYSIS_ROOT,
    WEINREB_FASTA, WEINREB_RESULTS_ROOT, domain_annotation,
    json_dump, kabsch_displacements, load_seed, mean_sd_se, mutation_sites,
    named_paper_sites, parse_ca, parse_position_map, read_fasta, read_manifest,
    safe_corr, sign_agreement, write_csv,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=Path, default=WEINREB_FASTA)
    p.add_argument("--manifest", type=Path, default=WEINREB_RESULTS_ROOT / "exact_contributions_v2" / "manifest.tsv")
    p.add_argument("--position-map", type=Path, default=WEINREB_RESULTS_ROOT / "position_map_1ZNX_WT.csv")
    p.add_argument("--structure", type=Path, default=WEINREB_RESULTS_ROOT / "pdb_cache" / "1ZNX_WT.pdb")
    p.add_argument("--apo-structure", type=Path, default=WEINREB_ANALYSIS_ROOT / "inputs" / "1ZNW_apo.pdb")
    p.add_argument("--condition", default=PRIMARY_CONDITION)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--output", type=Path, default=WEINREB_ANALYSIS_ROOT)
    return p.parse_args()


def main() -> None:
    args = arguments()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    fasta = read_fasta(args.fasta)
    length_set = {len(s) for s in fasta.values()}
    if len(length_set) != 1:
        raise ValueError(f"FASTA lengths differ: {sorted(length_set)}")
    length = next(iter(length_set))
    if "WT_star" not in fasta:
        raise ValueError("WT_star is absent from FASTA")

    manifest_rows = read_manifest(args.manifest, args.condition, seeds)
    seed_payloads = {int(r["seed"]): load_seed(r) for r in manifest_rows}
    matrix_audit = []
    seed_record_audit = []
    for seed, payloads in seed_payloads.items():
        observed = list(payloads)
        missing = sorted(set(fasta) - set(observed))
        extra = sorted(set(observed) - set(fasta))
        seed_record_audit.append({
            "condition": args.condition, "seed": seed,
            "expected_records": len(fasta), "observed_records": len(observed),
            "missing_records": ";".join(missing), "extra_records": ";".join(extra),
            "all_expected_present": not missing and not extra,
            "new_sequences_present": all(x in payloads for x in NEW_SEQUENCES),
        })
        for name, sequence in fasta.items():
            if name not in payloads:
                continue
            x = payloads[name]
            c, a = x.contribution, x.attention
            reconstruction = c.sum(axis=1) + x.bias
            rec_err = float(np.max(np.abs(reconstruction - x.margin)))
            prob = 1.0 / (1.0 + np.exp(-x.margin))
            probability_err = float(np.max(np.abs(prob - x.flexible_scores)))
            matrix_audit.append({
                "condition": args.condition, "seed": seed, "protein": name,
                "fasta_length": len(sequence), "json_length": len(x.sequence),
                "matrix_rows": c.shape[0], "matrix_columns": c.shape[1],
                "attention_rows": a.shape[0], "attention_columns": a.shape[1],
                "sequence_exact_match": sequence == x.sequence,
                "matrix_is_LxL": c.shape == (len(sequence), len(sequence)),
                "attention_is_LxL": a.shape == (len(sequence), len(sequence)),
                "finite": bool(np.isfinite(c).all() and np.isfinite(a).all()),
                "attention_row_sum_max_error": float(np.max(np.abs(a.sum(axis=1) - 1))),
                "reconstruction_max_abs_error": rec_err,
                "probability_max_abs_error": probability_err,
                "passes": bool(sequence == x.sequence and c.shape == (len(sequence), len(sequence))
                               and a.shape == (len(sequence), len(sequence))
                               and np.isfinite(c).all() and np.isfinite(a).all()
                               and rec_err <= 1e-5 and probability_err <= 1e-5),
            })

    if not all(r["all_expected_present"] and r["new_sequences_present"] for r in seed_record_audit):
        raise ValueError("At least one seed is missing FASTA records")
    if not all(r["passes"] for r in matrix_audit):
        raise ValueError("At least one contribution/attention matrix failed integrity checks")

    position_map = parse_position_map(args.position_map)
    closed_ca = parse_ca(args.structure)
    open_ca = parse_ca(args.apo_structure) if args.apo_structure.is_file() else {}
    displacement = kabsch_displacements(closed_ca, open_ca) if open_ca else {}
    mapping_rows = []
    for fasta_pos in range(1, length + 1):
        chain, paper_pos = position_map.get(fasta_pos, (None, None))
        xyz = closed_ca.get(paper_pos) if paper_pos is not None else None
        domain, function = domain_annotation(paper_pos)
        mapping_rows.append({
            "paper_position": paper_pos, "fasta_position": fasta_pos,
            "matrix_index": fasta_pos - 1, "wt_star_residue": fasta["WT_star"][fasta_pos - 1],
            "pdb_chain": chain, "pdb_residue": paper_pos,
            "ca_x": None if xyz is None else xyz[0],
            "ca_y": None if xyz is None else xyz[1],
            "ca_z": None if xyz is None else xyz[2],
            "open_to_closed_displacement_A": displacement.get(paper_pos),
            "domain": domain, "functional_annotation": function,
            "has_structural_coordinate": xyz is not None,
        })
    map_by_fasta = {r["fasta_position"]: r for r in mapping_rows}

    coordinate_audit = []
    for name, sequence in fasta.items():
        sites = mutation_sites(sequence, fasta["WT_star"])
        mapped_paper = [map_by_fasta[p]["paper_position"] for p in sites]
        named = named_paper_sites(name)
        coordinate_audit.append({
            "protein": name, "n_sequence_differences_vs_WT_star": len(sites),
            "fasta_positions": ";".join(map(str, sites)),
            "matrix_indices": ";".join(str(x - 1) for x in sites),
            "mapped_paper_positions": ";".join(str(x) for x in mapped_paper if x is not None),
            "name_paper_positions": ";".join(map(str, named)),
            "name_mapping_matches": (mapped_paper == named) if named else name in {"WT", "WT_star"},
            "is_native_WT_control": name == "WT",
            "is_new_sequence": name in NEW_SEQUENCES,
        })
    named_rows = [r for r in coordinate_audit if r["name_paper_positions"]]
    if not all(r["name_mapping_matches"] for r in named_rows):
        bad = [r["protein"] for r in named_rows if not r["name_mapping_matches"]]
        raise ValueError(f"Mutation-name/position-map mismatch: {bad}")

    seed_profile_rows = []
    summary_rows = []
    correlation_rows = []
    variants = [x for x in fasta if x != "WT_star"]
    all_delta = {}
    all_baseline = {}
    for seed in seeds:
        payloads = seed_payloads[seed]
        wt_i = payloads["WT_star"].contribution.mean(axis=0)
        all_baseline[seed] = wt_i
        for name in variants:
            delta = payloads[name].contribution.mean(axis=0) - wt_i
            all_delta[(name, seed)] = delta
            for j in range(length):
                mp = mapping_rows[j]
                seed_profile_rows.append({
                    "condition": args.condition, "seed": seed, "mutant": name,
                    "matrix_index": j, "fasta_position": j + 1,
                    "paper_position": mp["paper_position"], "wt_star_I": wt_i[j],
                    "mutant_I": wt_i[j] + delta[j], "delta_I": delta[j],
                })
    for name in variants:
        values = np.vstack([all_delta[(name, s)] for s in seeds])
        baselines = np.vstack([all_baseline[s] for s in seeds])
        mean, sd, se = mean_sd_se(values)
        bmean, bsd, bse = mean_sd_se(baselines)
        agree = sign_agreement(values)
        ratio = np.abs(mean) / np.maximum(sd, 1e-12)
        for j in range(length):
            mp = mapping_rows[j]
            summary_rows.append({
                "condition": args.condition, "mutant": name, "matrix_index": j,
                "fasta_position": j + 1, "paper_position": mp["paper_position"],
                "wt_star_residue": mp["wt_star_residue"], "domain": mp["domain"],
                "functional_annotation": mp["functional_annotation"],
                "wt_star_I_mean": bmean[j], "wt_star_I_sd": bsd[j], "wt_star_I_se": bse[j],
                "delta_I_mean": mean[j], "delta_I_sd": sd[j], "delta_I_se": se[j],
                "sign_agreement_n": int(agree[j]), "effect_to_seed_variation": ratio[j],
                "reproducible_2of3": bool(agree[j] >= 2),
                "strong_3of3": bool(agree[j] == len(seeds)),
            })
        for s1, s2 in itertools.combinations(seeds, 2):
            correlation_rows.append({
                "condition": args.condition, "mutant": name,
                "seed_1": s1, "seed_2": s2,
                "pearson_delta_I_profile": safe_corr(all_delta[(name, s1)], all_delta[(name, s2)]),
            })

    out = args.output
    write_csv(out / "audit" / "seed_record_audit.csv", seed_record_audit)
    write_csv(out / "audit" / "matrix_integrity_audit.csv", matrix_audit)
    write_csv(out / "audit" / "coordinate_conversion_audit.csv", coordinate_audit)
    write_csv(out / "coordinate_mapping.csv", mapping_rows)
    write_csv(out / "profiles" / "delta_I_by_seed.csv", seed_profile_rows)
    write_csv(out / "profiles" / "delta_I_summary.csv", summary_rows)
    write_csv(out / "profiles" / "seed_profile_correlations.csv", correlation_rows)
    np.savez_compressed(out / "profiles" / "delta_I_arrays.npz",
                        mutant_names=np.asarray(variants), seeds=np.asarray(seeds),
                        delta=np.stack([[all_delta[(m, s)] for s in seeds] for m in variants]),
                        wt_star=np.stack([all_baseline[s] for s in seeds]))
    json_dump(out / "audit" / "build_summary.json", {
        "condition": args.condition, "seeds": list(seeds), "records": len(fasta),
        "sequence_length": length, "new_sequences": list(NEW_SEQUENCES),
        "new_sequences_all_present": all(r["new_sequences_present"] for r in seed_record_audit),
        "all_matrices_pass": all(r["passes"] for r in matrix_audit),
        "max_reconstruction_error": max(r["reconstruction_max_abs_error"] for r in matrix_audit),
        "max_probability_error": max(r["probability_max_abs_error"] for r in matrix_audit),
        "mapped_positions": sum(r["has_structural_coordinate"] for r in mapping_rows),
        "unmapped_positions": sum(not r["has_structural_coordinate"] for r in mapping_rows),
        "coordinate_convention": "paper=PDB residue; FASTA=paper-1 where mapped; matrix=FASTA-1",
        "open_structure_available": bool(open_ca),
    })


if __name__ == "__main__":
    main()

