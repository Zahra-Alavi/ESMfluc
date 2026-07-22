#!/usr/bin/env python3
"""Map changed key bands to receivers and quantify mutant-specific query routing.

This module explicitly treats every attention row (query residue) as a routing
pattern that may change in a mutant, including queries and changed keys that are
far from the mutation site.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from weinreb_analysis_common import (
    NEW_SEQUENCES, PRIMARY_CONDITION, WEINREB_ANALYSIS_ROOT, WEINREB_FASTA,
    WEINREB_RESULTS_ROOT, load_seed, mean_sd_se, mutation_sites, parse_ca,
    parse_position_map, read_fasta, read_manifest, safe_corr, sign_agreement,
    write_csv, json_dump,
)


BINDING_SITES = {30, 88, 31, 53, 60, 33, 101, 32}
HIGH_STRAIN_SITES = {178, 179, 175, 174, 177, 173, 176, 29}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=WEINREB_RESULTS_ROOT / "exact_contributions_v2" / "manifest.tsv")
    p.add_argument("--fasta", type=Path, default=WEINREB_FASTA)
    p.add_argument("--position-map", type=Path, default=WEINREB_RESULTS_ROOT / "position_map_1ZNX_WT.csv")
    p.add_argument("--structure", type=Path, default=WEINREB_RESULTS_ROOT / "pdb_cache" / "1ZNX_WT.pdb")
    p.add_argument("--analysis", type=Path, default=WEINREB_ANALYSIS_ROOT)
    p.add_argument("--condition", default=PRIMARY_CONDITION)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--top-keys", type=int, default=5)
    p.add_argument("--permutations", type=int, default=2000)
    return p.parse_args()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-15
    p, q = np.maximum(p, eps), np.maximum(q, eps)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def point_segment_distance(x, a, b):
    ab = b - a
    t = np.clip(np.dot(x - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(x - (a + t * ab)))


def main() -> None:
    args = arguments()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    fasta = read_fasta(args.fasta)
    wt_sequence = fasta["WT_star"]
    posmap = parse_position_map(args.position_map)
    coords = parse_ca(args.structure)
    with (args.analysis / "coordinate_mapping.csv").open(encoding="utf-8", newline="") as h:
        mapping = list(csv.DictReader(h))
    for r in mapping:
        for k in ("paper_position", "fasta_position", "matrix_index"):
            r[k] = None if r[k] in {"", "None"} else int(r[k])
        r["open_to_closed_displacement_A"] = (None if r["open_to_closed_displacement_A"] in {"", "None"}
                                                   else float(r["open_to_closed_displacement_A"]))
    paper = [r["paper_position"] for r in mapping]
    length = len(mapping)
    runs = read_manifest(args.manifest, args.condition, seeds)
    loaded = {int(r["seed"]): load_seed(r) for r in runs}
    names = [n for n in loaded[seeds[0]] if n != "WT_star"]

    # Structural annotations used in receiver enrichment.
    if 75 in coords and 171 in coords:
        corridor = {p for p, xyz in coords.items()
                    if point_segment_distance(xyz, coords[75], coords[171]) <= 8.0}
    else:
        corridor = set()
    disp = np.array([r["open_to_closed_displacement_A"] for r in mapping
                     if r["open_to_closed_displacement_A"] is not None])
    displacement_cut = float(np.quantile(disp, 0.75)) if len(disp) else float("inf")

    query_seed_rows = []
    query_summary_rows = []
    top_key_rows = []
    mean_delta_a = np.empty((len(names), length, length), dtype=np.float32)
    mean_delta_c = np.empty((len(names), length, length), dtype=np.float32)
    for ni, name in enumerate(names):
        sites = mutation_sites(fasta[name], wt_sequence)
        paper_sites = [posmap[p][1] for p in sites if p in posmap]
        seed_metrics = []
        da_seeds, dc_seeds = [], []
        for seed in seeds:
            x, wt = loaded[seed][name], loaded[seed]["WT_star"]
            da = x.attention - wt.attention
            dc = x.contribution - wt.contribution
            da_seeds.append(da)
            dc_seeds.append(dc)
            metrics = []
            for i in range(length):
                distant = []
                unmapped = []
                for j in range(length):
                    seq_far = all(abs((j + 1) - s) > 15 for s in sites) if sites else False
                    if paper[j] is None or not paper_sites or paper[j] not in coords:
                        unmapped.append(abs(dc[i, j]))
                        continue
                    d3 = [np.linalg.norm(coords[paper[j]] - coords[p]) for p in paper_sites if p in coords]
                    if seq_far and d3 and min(d3) > 12:
                        distant.append(abs(dc[i, j]))
                values = {
                    "delta_query_logit_margin": float(dc[i].sum()),
                    "attention_total_variation": float(0.5 * np.abs(da[i]).sum()),
                    "attention_js_divergence": js_divergence(x.attention[i], wt.attention[i]),
                    "contribution_pattern_l1": float(np.abs(dc[i]).sum()),
                    "contribution_pattern_correlation": safe_corr(x.contribution[i], wt.contribution[i]),
                    "distant_key_mean_abs_delta_C": float(np.mean(distant)) if distant else np.nan,
                    "distant_key_integrated_abs_delta_C": float(np.sum(distant)) if distant else np.nan,
                    "distant_key_count": len(distant), "unmapped_key_count": len(unmapped),
                }
                metrics.append(values)
                query_seed_rows.append({
                    "mutant": name, "seed": seed, "query_matrix_index": i,
                    "query_fasta_position": i + 1, "query_paper_position": paper[i], **values,
                })
            seed_metrics.append(metrics)
        da_seeds = np.stack(da_seeds)
        dc_seeds = np.stack(dc_seeds)
        mean_delta_a[ni] = da_seeds.mean(axis=0)
        mean_delta_c[ni] = dc_seeds.mean(axis=0)
        for i in range(length):
            query_seq_distance = min((abs(i + 1 - s) for s in sites), default=None)
            d3 = ([float(np.linalg.norm(coords[paper[i]] - coords[p])) for p in paper_sites
                   if paper[i] in coords and p in coords] if paper[i] is not None else [])
            row = {
                "mutant": name, "query_matrix_index": i, "query_fasta_position": i + 1,
                "query_paper_position": paper[i], "query_domain": mapping[i]["domain"],
                "query_function": mapping[i]["functional_annotation"],
                "query_sequence_distance_to_mutation": query_seq_distance,
                "query_ca_distance_to_mutation_A": min(d3) if d3 else None,
                "query_is_sequence_distal_gt15": query_seq_distance is not None and query_seq_distance > 15,
                "query_is_structurally_distal_gt12A": bool(d3 and min(d3) > 12),
            }
            for metric in ("delta_query_logit_margin", "attention_total_variation",
                           "attention_js_divergence", "contribution_pattern_l1",
                           "contribution_pattern_correlation", "distant_key_mean_abs_delta_C",
                           "distant_key_integrated_abs_delta_C"):
                vals = np.array([seed_metrics[k][i][metric] for k in range(len(seeds))], float)
                valid = vals[np.isfinite(vals)]
                if len(valid):
                    m, sd, se = mean_sd_se(valid)
                    row[metric + "_mean"] = m
                    row[metric + "_sd"] = sd
                    row[metric + "_se"] = se
                else:
                    row[metric + "_mean"] = row[metric + "_sd"] = row[metric + "_se"] = None
            margin_values = np.array([seed_metrics[k][i]["delta_query_logit_margin"] for k in range(len(seeds))])
            row["query_margin_sign_agreement_n"] = int(sign_agreement(margin_values, axis=0))
            row["delta_attention_pattern_seed_correlation"] = float(np.nanmean([
                safe_corr(da_seeds[a, i], da_seeds[b, i]) for a, b in itertools.combinations(range(len(seeds)), 2)]))
            row["delta_contribution_pattern_seed_correlation"] = float(np.nanmean([
                safe_corr(dc_seeds[a, i], dc_seeds[b, i]) for a, b in itertools.combinations(range(len(seeds)), 2)]))
            query_summary_rows.append(row)

            top = np.argsort(np.abs(mean_delta_c[ni, i]))[-args.top_keys:][::-1]
            for rank, j in enumerate(top, 1):
                key_d3 = ([float(np.linalg.norm(coords[paper[j]] - coords[p])) for p in paper_sites
                           if paper[j] in coords and p in coords] if paper[j] is not None else [])
                top_key_rows.append({
                    "mutant": name, "query_matrix_index": i, "query_fasta_position": i + 1,
                    "query_paper_position": paper[i], "rank": rank, "key_matrix_index": int(j),
                    "key_fasta_position": int(j + 1), "key_paper_position": paper[j],
                    "mean_delta_C": mean_delta_c[ni, i, j],
                    "mean_delta_attention": mean_delta_a[ni, i, j],
                    "key_sequence_distance_to_mutation": min((abs(j + 1 - s) for s in sites), default=None),
                    "key_ca_distance_to_mutation_A": min(key_d3) if key_d3 else None,
                    "key_domain": mapping[j]["domain"], "key_function": mapping[j]["functional_annotation"],
                })
    write_csv(args.analysis / "receivers" / "query_routing_by_seed.csv", query_seed_rows)
    write_csv(args.analysis / "receivers" / "query_routing_summary.csv", query_summary_rows)
    write_csv(args.analysis / "receivers" / "query_top_changed_keys.csv", top_key_rows)
    np.savez_compressed(args.analysis / "receivers" / "query_pattern_arrays.npz",
                        mutant_names=np.asarray(names), mean_delta_attention=mean_delta_a,
                        mean_delta_contribution=mean_delta_c)

    # Explicit between-mutant comparison at each query. This separates a common
    # query hotspot from a mutant-specific change in which keys that query
    # consults. New hypothesis sequences are scored against the 34 paper-mutant
    # reference distribution but do not define it.
    original_indices = [i for i, n in enumerate(names) if n != "WT" and n not in NEW_SEQUENCES]
    specificity_rows = []
    summary_lookup = {(r["mutant"], int(r["query_matrix_index"])): r for r in query_summary_rows}
    for i in range(length):
        original_tv = np.array([0.5 * np.abs(mean_delta_a[k, i]).sum() for k in original_indices])
        common_tv = float(np.median(original_tv))
        tv_scale = float(1.4826 * np.median(np.abs(original_tv - common_tv)))
        for ni, name in enumerate(names):
            comparison = [k for k in original_indices if k != ni]
            typical_pattern = np.median(mean_delta_a[comparison, i], axis=0)
            residual = mean_delta_a[ni, i] - typical_pattern
            raw_tv = float(0.5 * np.abs(mean_delta_a[ni, i]).sum())
            dominant = int(np.argmax(np.abs(residual)))
            base = summary_lookup[(name, i)]
            specificity_rows.append({
                "mutant": name, "query_matrix_index": i, "query_fasta_position": i + 1,
                "query_paper_position": paper[i], "query_domain": mapping[i]["domain"],
                "query_function": mapping[i]["functional_annotation"],
                "query_sequence_distance_to_mutation": base["query_sequence_distance_to_mutation"],
                "query_ca_distance_to_mutation_A": base["query_ca_distance_to_mutation_A"],
                "mutant_attention_total_variation": raw_tv,
                "typical_mutant_attention_total_variation": common_tv,
                "mutant_specific_TV_excess": raw_tv - common_tv,
                "mutant_specific_TV_z": (raw_tv - common_tv) / max(tv_scale, 1e-12),
                "attention_change_correlation_to_other_mutants": safe_corr(mean_delta_a[ni, i], typical_pattern),
                "attention_change_residual_total_variation": float(0.5 * np.abs(residual).sum()),
                "delta_query_logit_margin": float(mean_delta_c[ni, i].sum()),
                "most_mutant_specific_key_matrix_index": dominant,
                "most_mutant_specific_key_fasta_position": dominant + 1,
                "most_mutant_specific_key_paper_position": paper[dominant],
                "most_mutant_specific_delta_attention_residual": residual[dominant],
                "comparison_reference": "leave-one-out median of 34 paper mutants",
            })
    write_csv(args.analysis / "receivers" / "query_mutant_specificity.csv", specificity_rows)

    band_path = args.analysis / "bands" / "change_bands_primary.csv"
    with band_path.open(encoding="utf-8", newline="") as h:
        bands = list(csv.DictReader(h))
    receiver_rows = []
    enrichment_rows = []
    rng = np.random.default_rng(20250719)
    for band_id, band in enumerate(bands, 1):
        name = band["mutant"]
        start, end = int(band["start_matrix_index"]), int(band["end_matrix_index"])
        seed_receiver = []
        for seed in seeds:
            x, wt = loaded[seed][name], loaded[seed]["WT_star"]
            seed_receiver.append((x.contribution[:, start:end + 1] -
                                  wt.contribution[:, start:end + 1]).sum(axis=1))
        seed_receiver = np.stack(seed_receiver)
        rmean, rsd, rse = mean_sd_se(seed_receiver)
        ragree = sign_agreement(seed_receiver)
        for i in range(length):
            pp = paper[i]
            receiver_rows.append({
                "band_id": band_id, "mutant": name, "band_start_matrix_index": start,
                "band_apex_matrix_index": int(band["apex_matrix_index"]),
                "band_end_matrix_index": end, "query_matrix_index": i,
                "query_fasta_position": i + 1, "query_paper_position": pp,
                "delta_C_from_band_mean": rmean[i], "delta_C_from_band_sd": rsd[i],
                "delta_C_from_band_se": rse[i], "sign_agreement_n": int(ragree[i]),
                "query_domain": mapping[i]["domain"], "query_function": mapping[i]["functional_annotation"],
                "is_experimental_binding_residue": pp in BINDING_SITES,
                "is_experimental_high_strain_residue": pp in HIGH_STRAIN_SITES,
                "is_C75_C171_corridor": pp in corridor,
                "is_large_open_closed_displacement": (mapping[i]["open_to_closed_displacement_A"] is not None and
                                                      mapping[i]["open_to_closed_displacement_A"] >= displacement_cut),
            })
        categories = {
            "GMP-binding domain": np.array([r["domain"] == "GMP-binding" for r in mapping]),
            "LID": np.array([r["domain"] == "LID" for r in mapping]),
            "hinges": np.array(["hinge" in r["functional_annotation"] for r in mapping]),
            "P-loop": np.array([r["functional_annotation"] == "P-loop" for r in mapping]),
            "experimental binding residues": np.array([p in BINDING_SITES for p in paper]),
            "experimental high-strain residues": np.array([p in HIGH_STRAIN_SITES for p in paper]),
            "C75-C171 structural corridor": np.array([p in corridor for p in paper]),
            "large open-to-closed displacement": np.array([
                r["open_to_closed_displacement_A"] is not None and
                r["open_to_closed_displacement_A"] >= displacement_cut for r in mapping]),
        }
        abs_effect = np.abs(rmean)
        for category, mask in categories.items():
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            observed = float(abs_effect[idx].mean())
            # Draw a matched null matrix in vectorized form. Each category
            # residue contributes one replacement matched on domain and
            # coordinate availability; sampling is with replacement.
            sampled = np.empty((args.permutations, len(idx)), dtype=int)
            fallback = np.flatnonzero(~mask)
            for column, i in enumerate(idx):
                candidates = np.array([j for j in range(length) if not mask[j]
                                       and mapping[j]["domain"] == mapping[i]["domain"]
                                       and (paper[j] is None) == (paper[i] is None)], dtype=int)
                if not len(candidates):
                    candidates = fallback
                sampled[:, column] = rng.choice(candidates, size=args.permutations, replace=True)
            null = abs_effect[sampled].mean(axis=1)
            enrichment_rows.append({
                "band_id": band_id, "mutant": name, "category": category,
                "category_n": len(idx), "observed_mean_abs_delta_C": observed,
                "matched_null_mean": float(null.mean()),
                "effect_ratio": observed / max(float(null.mean()), 1e-15),
                "permutation_p_greater": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                "permutations": args.permutations,
            })
    # Benjamini-Hochberg correction across the complete, pre-specified family
    # of band-by-category enrichment tests.
    if enrichment_rows:
        pvals = np.array([r["permutation_p_greater"] for r in enrichment_rows])
        order = np.argsort(pvals)
        adjusted = np.empty(len(pvals), float)
        running = 1.0
        for rank_index in range(len(pvals) - 1, -1, -1):
            original = order[rank_index]
            rank = rank_index + 1
            running = min(running, pvals[original] * len(pvals) / rank)
            adjusted[original] = running
        for r, q in zip(enrichment_rows, adjusted):
            r["BH_FDR_q"] = float(min(q, 1.0))
    write_csv(args.analysis / "receivers" / "band_receivers.csv", receiver_rows)
    write_csv(args.analysis / "receivers" / "receiver_enrichment_matched_permutations.csv", enrichment_rows)
    json_dump(args.analysis / "receivers" / "receiver_analysis_config.json", {
        "query_pattern_definition": "full mutant-vs-WT_star attention/contribution row change",
        "distant_key_definition": {"sequence_distance_gt": 15, "ca_distance_gt_A": 12},
        "pattern_metrics": ["attention total variation", "attention Jensen-Shannon divergence",
                            "signed received logit change", "contribution L1 change"],
        "matched_permutations": args.permutations,
        "matching_factors": ["coarse domain", "structural-coordinate availability"],
        "pulling_corridor": "residues whose CA lies within 8 A of the C75-C171 line segment",
        "large_displacement": "top quartile of aligned apo 1ZNW vs closed 1ZNX CA displacement",
    })


if __name__ == "__main__":
    main()
