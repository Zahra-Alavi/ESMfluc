#!/usr/bin/env python3
"""Analyze mutant responses to fixed, seed-stable WT* contribution bands."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gk_v2_common import (
    CONDITIONS, PAPER_MUTANTS, PRIMARY_CONDITION, SEEDS, bh_adjust,
    load_combined_runs, reconstruct_evidence, write_csv, write_json,
)
from weinreb_analysis_common import mutation_sites, parse_ca, parse_position_map, read_fasta


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--a176g-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--position-map", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--stable-bands", type=Path, required=True)
    parser.add_argument("--mean-bands", type=Path, required=True)
    parser.add_argument("--uniform-stable-bands", type=Path, required=True)
    parser.add_argument("--uniform-mean-bands", type=Path, required=True)
    parser.add_argument("--uniform-profile-similarity", type=Path, required=True)
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--legacy-hotspots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=123)
    return parser.parse_args()


def sign_agreement(values):
    values = np.asarray(values)
    return int(max(np.sum(values > 0), np.sum(values < 0)))


def interval_iou(a0, a1, b0, b1):
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union else 0.0


def fit_ols(x, y):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    return beta, residual, float(np.sum(residual * residual))


def loo_rmse(x, y):
    predicted = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        predicted[index] = x[index] @ np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
    return float(np.sqrt(np.mean((y - predicted) ** 2)))


def partial_test(y, covariates, predictor, rng, permutations):
    predictor = np.asarray(predictor, dtype=float)
    if len(np.unique(predictor)) > 2:
        predictor = (predictor - predictor.mean()) / (predictor.std() or 1.0)
    full = np.column_stack([covariates, predictor])
    beta0, residual0, sse0 = fit_ols(covariates, y)
    beta1, _residual1, sse1 = fit_ols(full, y)
    improvement = max(0.0, sse0 - sse1)
    null = np.empty(permutations, dtype=float)
    fitted0 = covariates @ beta0
    for iteration in range(permutations):
        permuted_y = fitted0 + rng.permutation(residual0)
        _, _, null0 = fit_ols(covariates, permuted_y)
        _, _, null1 = fit_ols(full, permuted_y)
        null[iteration] = max(0.0, null0 - null1)
    return {
        "coefficient": float(beta1[-1]),
        "partial_R2": float(improvement / sse0) if sse0 else None,
        "freedman_lane_permutation_p": float(
            (1 + np.sum(null >= improvement)) / (1 + permutations)
        ),
        "reduced_LOO_RMSE": loo_rmse(covariates, y),
        "full_LOO_RMSE": loo_rmse(full, y),
    }


def main():
    args = arguments()
    rng = np.random.default_rng(args.random_seed)
    fasta = read_fasta(args.fasta)
    reference = fasta["WT_star"]
    position_map = parse_position_map(args.position_map, "WT_star")
    if any(pdb_residue != fasta_position + 1 or chain != "A"
           for fasta_position, (chain, pdb_residue) in position_map.items()):
        raise ValueError("WT* position map is not the expected FASTA+1 mapping on chain A")
    ca = parse_ca(args.structure)
    bands = pd.read_csv(args.stable_bands)
    bands = bands[bands["primary_stable_band"].astype(bool)].copy()
    mean_bands = pd.read_csv(args.mean_bands)
    mean_index = mean_bands.set_index("band_id")
    for band in bands.itertuples(index=False):
        if band.band_id not in mean_index.index:
            raise ValueError(f"Stable band is absent from mean-profile catalog: {band.band_id}")
        source = mean_index.loc[band.band_id]
        for field in ("condition", "protein", "sign", "start_index_0based",
                      "apex_index_0based", "end_index_0based_inclusive"):
            if source[field] != getattr(band, field):
                raise ValueError(f"Stable/mean band mismatch for {band.band_id}: {field}")
    for condition, group in bands.groupby("condition"):
        coverage = np.zeros(len(reference), dtype=int)
        for band in group.itertuples(index=False):
            coverage[int(band.start_index_0based):
                     int(band.end_index_0based_inclusive) + 1] += 1
        if np.any(coverage > 1):
            raise ValueError(f"Overlapping stable bands in {condition}")
    primary_bands = bands[bands.condition == PRIMARY_CONDITION].copy()
    if len(primary_bands) != 11:
        raise ValueError(f"Expected 11 primary WT* bands, found {len(primary_bands)}")
    primary_bands = primary_bands.sort_values("apex_residue_1based").reset_index(drop=True)
    primary_bands["gk_band_number"] = np.arange(1, len(primary_bands) + 1)
    length = len(reference)
    band_mask = np.zeros(length, dtype=bool)
    for band in primary_bands.itertuples(index=False):
        band_mask[int(band.start_index_0based):int(band.end_index_0based_inclusive) + 1] = True
    nonband_mask = ~band_mask

    loaded = load_combined_runs(args.base_manifest, args.a176g_manifest)
    expected = set(fasta)
    for key, payloads in loaded.items():
        if set(payloads) != expected:
            raise ValueError(
                f"{key}: expected 40 FASTA records; missing={sorted(expected-set(payloads))}, "
                f"extra={sorted(set(payloads)-expected)}"
            )

    # Fixed reference-band mapping and mutation membership across conditions.
    mapped_band_rows = []
    for band in primary_bands.itertuples(index=False):
        start = int(band.start_residue_1based)
        apex = int(band.apex_residue_1based)
        end = int(band.end_residue_1based_inclusive)
        resolved = [position_map[pos][1] for pos in range(start, end + 1)
                    if pos in position_map]
        mapped_band_rows.append({
            "gk_band_number": int(band.gk_band_number), "band_id": band.band_id,
            "sign": int(band.sign), "label": band.label,
            "fasta_start_1based": start,
            "fasta_apex_1based": apex,
            "fasta_end_1based": end,
            "paper_start": start + 1,
            "paper_apex": apex + 1,
            "paper_end": end + 1,
            "pdb_resolved_start": min(resolved) if resolved else None,
            "pdb_resolved_end": max(resolved) if resolved else None,
            "pdb_resolved_residue_count": len(resolved),
            "pdb_resolved_fraction": len(resolved) / (end - start + 1),
            "apex_has_pdb_coordinates": apex in position_map,
            "seed_support_count": int(band.seed_support_count),
            "absolute_apex_rank_within_protein": int(
                band.absolute_apex_rank_within_protein
            ),
        })
    write_csv(args.output_dir / "reference_bands" / "primary_wt_star_bands.csv", mapped_band_rows)

    membership_rows = []
    condition_band_lookup = {
        condition: bands[bands.condition == condition].copy()
        for condition in CONDITIONS
    }
    for variant, sequence in fasta.items():
        if variant in {"WT", "WT_star"}:
            continue
        for fasta_position in mutation_sites(sequence, reference):
            paper_position = position_map.get(fasta_position, (None, None))[1]
            for condition in CONDITIONS:
                candidates = condition_band_lookup[condition]
                hit = candidates[
                    (candidates.start_residue_1based <= fasta_position)
                    & (candidates.end_residue_1based_inclusive >= fasta_position)
                ]
                apices = candidates.apex_residue_1based.astype(int).to_numpy()
                nearest_distance = int(np.min(np.abs(apices - fasta_position))) if len(apices) else None
                if len(hit) > 1:
                    raise ValueError(f"Overlapping bands at {condition}/{variant}/{fasta_position}")
                membership_rows.append({
                    "variant": variant, "condition": condition,
                    "is_primary_condition": condition == PRIMARY_CONDITION,
                    "fasta_position_1based": fasta_position,
                    "paper_position": paper_position,
                    "reference_amino_acid": reference[fasta_position - 1],
                    "mutant_amino_acid": sequence[fasta_position - 1],
                    "inside_stable_band": bool(len(hit)),
                    "band_sign": int(hit.iloc[0].sign) if len(hit) else None,
                    "band_id": hit.iloc[0].band_id if len(hit) else None,
                    "distance_to_nearest_stable_apex": nearest_distance,
                })
    write_csv(args.output_dir / "reference_bands" / "mutation_membership_all_conditions.csv", membership_rows)

    variant_rows = []
    band_response_rows = []
    primary_receiver_rows = []
    primary_mean_profiles = {}
    maximum_decomposition_error = 0.0
    for condition in CONDITIONS:
        names = [name for name in fasta if name != "WT_star"]
        for variant in names:
            sites = mutation_sites(fasta[variant], reference)
            seed_delta_i = []
            seed_probability_change = []
            seed_probability_abs_change = []
            seed_attention_tv = []
            seed_contribution_l1 = []
            seed_band_records = {int(b.gk_band_number): [] for b in primary_bands.itertuples()}
            for seed in SEEDS:
                wt = loaded[(condition, seed)]["WT_star"]
                mutant = loaded[(condition, seed)][variant]
                delta_c = mutant.contribution - wt.contribution
                delta_i = delta_c.mean(axis=0)
                delta_a = mutant.attention - wt.attention
                seed_delta_i.append(delta_i)
                seed_probability_change.append(
                    float(np.mean(mutant.flexible_scores - wt.flexible_scores))
                )
                seed_probability_abs_change.append(
                    float(np.mean(np.abs(mutant.flexible_scores - wt.flexible_scores)))
                )
                seed_attention_tv.append(float(np.mean(0.5 * np.abs(delta_a).sum(axis=1))))
                seed_contribution_l1.append(float(np.mean(np.abs(delta_c).sum(axis=1))))
                sm, em = reconstruct_evidence(mutant.contribution, mutant.attention)
                sw, ew = reconstruct_evidence(wt.contribution, wt.attention)
                maximum_decomposition_error = max(maximum_decomposition_error, em, ew)
                bm, bw = mutant.attention.mean(axis=0), wt.attention.mean(axis=0)
                intrinsic = 0.5 * (bm + bw) * (sm - sw)
                routing = 0.5 * (sm + sw) * (bm - bw)
                maximum_decomposition_error = max(
                    maximum_decomposition_error,
                    float(np.max(np.abs(intrinsic + routing - delta_i))),
                )
                for band in primary_bands.itertuples(index=False):
                    number = int(band.gk_band_number)
                    start, end = int(band.start_index_0based), int(band.end_index_0based_inclusive)
                    receiver = delta_c[:, start:end + 1].sum(axis=1)
                    seed_band_records[number].append({
                        "delta_i_signed": float(delta_i[start:end + 1].sum()),
                        "delta_i_absolute": float(np.abs(delta_i[start:end + 1]).sum()),
                        "intrinsic": float(intrinsic[start:end + 1].sum()),
                        "routing": float(routing[start:end + 1].sum()),
                        "receiver": receiver,
                    })
            delta_stack = np.stack(seed_delta_i)
            mean_delta = delta_stack.mean(axis=0)
            if condition == PRIMARY_CONDITION:
                primary_mean_profiles[variant] = mean_delta
            distance = np.full(length, np.inf)
            for site in sites:
                distance = np.minimum(distance, np.abs(np.arange(1, length + 1) - site))
            distal = distance > 15
            primary_membership = [
                row for row in membership_rows
                if row["variant"] == variant and row["condition"] == PRIMARY_CONDITION
            ]
            inside_primary = any(row["inside_stable_band"] for row in primary_membership)
            nearest_primary = min(
                row["distance_to_nearest_stable_apex"] for row in primary_membership
            ) if primary_membership else None
            receiver_band_values = []
            for band in primary_bands.itertuples(index=False):
                number = int(band.gk_band_number)
                records = seed_band_records[number]
                signed = np.array([r["delta_i_signed"] for r in records])
                absolute = np.array([r["delta_i_absolute"] for r in records])
                intrinsic = np.array([r["intrinsic"] for r in records])
                routing = np.array([r["routing"] for r in records])
                receivers = np.stack([r["receiver"] for r in records])
                receiver_mean = receivers.mean(axis=0)
                receiver_band_values.append(float(np.mean(np.abs(receiver_mean))))
                dominant = "intrinsic_evidence" if abs(intrinsic.mean()) > abs(routing.mean()) else "attention_routing"
                band_response_rows.append({
                    "condition": condition, "variant": variant,
                    "gk_band_number": number, "band_id": band.band_id,
                    "band_sign": int(band.sign),
                    "band_start_fasta_1based": int(band.start_residue_1based),
                    "band_apex_fasta_1based": int(band.apex_residue_1based),
                    "band_end_fasta_1based": int(band.end_residue_1based_inclusive),
                    "mutation_sites_fasta_1based": ";".join(map(str, sites)),
                    "mutation_inside_this_band": any(
                        int(band.start_residue_1based) <= site <= int(band.end_residue_1based_inclusive)
                        for site in sites
                    ),
                    "delta_I_integrated_mean": float(signed.mean()),
                    "delta_I_integrated_sd": float(signed.std(ddof=1)),
                    "delta_I_absolute_integrated_mean": float(absolute.mean()),
                    "delta_I_seed_sign_agreement": sign_agreement(signed),
                    "intrinsic_component_integrated_mean": float(intrinsic.mean()),
                    "routing_component_integrated_mean": float(routing.mean()),
                    "dominant_component": dominant,
                    "receiver_mean_absolute_delta_C": float(np.mean(np.abs(receiver_mean))),
                    "receiver_max_absolute_delta_C": float(np.max(np.abs(receiver_mean))),
                    "receiver_seed_profile_correlation_mean": float(np.nanmean([
                        np.corrcoef(receivers[a], receivers[b])[0, 1]
                        for a, b in ((0, 1), (0, 2), (1, 2))
                    ])),
                })
                if condition == PRIMARY_CONDITION:
                    for query_index in range(length):
                        query_paper = position_map.get(query_index + 1, (None, None))[1]
                        query_distances = [abs(query_index + 1 - site) for site in sites]
                        mutation_papers = [position_map[s][1] for s in sites if s in position_map]
                        distances_3d = [
                            float(np.linalg.norm(ca[query_paper] - ca[p]))
                            for p in mutation_papers
                            if query_paper in ca and p in ca
                        ] if query_paper is not None else []
                        values = receivers[:, query_index]
                        primary_receiver_rows.append({
                            "variant": variant, "gk_band_number": number,
                            "band_sign": int(band.sign),
                            "query_index_0based": query_index,
                            "query_fasta_position_1based": query_index + 1,
                            "query_paper_position": query_paper,
                            "query_sequence_distance_to_mutation": min(query_distances) if query_distances else None,
                            "query_ca_distance_to_mutation_A": min(distances_3d) if distances_3d else None,
                            "delta_C_from_fixed_band_mean": float(values.mean()),
                            "delta_C_from_fixed_band_sd": float(values.std(ddof=1)),
                            "seed_sign_agreement": sign_agreement(values),
                        })
            variant_rows.append({
                "condition": condition, "variant": variant,
                "is_paper_mutant": variant in PAPER_MUTANTS,
                "mutation_sites_fasta_1based": ";".join(map(str, sites)),
                "mutation_inside_primary_wt_band": inside_primary,
                "distance_to_nearest_primary_wt_band_apex": nearest_primary,
                "global_mean_abs_delta_I": float(np.mean(np.abs(mean_delta))),
                "fixed_band_mean_abs_delta_I": float(np.mean(np.abs(mean_delta[band_mask]))),
                "nonband_mean_abs_delta_I": float(np.mean(np.abs(mean_delta[nonband_mask]))),
                "fixed_to_nonband_abs_delta_I_ratio": float(
                    np.mean(np.abs(mean_delta[band_mask]))
                    / max(np.mean(np.abs(mean_delta[nonband_mask])), 1e-15)
                ),
                "distal_fixed_band_mean_abs_delta_I": float(np.mean(np.abs(mean_delta[band_mask & distal])))
                if np.any(band_mask & distal) else None,
                "mean_flexible_probability_change": float(np.mean(seed_probability_change)),
                "mean_absolute_flexible_probability_change": float(
                    np.mean(seed_probability_abs_change)
                ),
                "mean_query_attention_total_variation": float(np.mean(seed_attention_tv)),
                "mean_query_contribution_L1_change": float(np.mean(seed_contribution_l1)),
                "fixed_band_receiver_mean_abs_delta_C": float(np.mean(receiver_band_values)),
            })

    write_csv(args.output_dir / "fixed_band_responses" / "variant_condition_summary.csv", variant_rows)
    write_csv(args.output_dir / "fixed_band_responses" / "band_response_summary.csv", band_response_rows)
    write_csv(args.output_dir / "receivers" / "primary_fixed_band_receivers.csv", primary_receiver_rows)

    # Test whether frozen WT* bands are recurrent response hubs.
    paper_profiles = np.stack([primary_mean_profiles[name] for name in PAPER_MUTANTS])
    observed = float(np.mean(np.abs(paper_profiles[:, band_mask])))
    # For a single 207-residue protein, every possible circular shift can be
    # evaluated exactly; random resampling would only repeat these same shifts.
    null = np.array([
        float(np.mean(np.abs(paper_profiles[:, np.roll(band_mask, shift)])))
        for shift in range(length)
    ])
    hub_test = {
        "profile": "seed-averaged absolute mutant-minus-WT* I_j across 34 Weinreb mutants",
        "observed_mean_abs_delta_I_in_fixed_bands": observed,
        "nonband_mean_abs_delta_I": float(np.mean(np.abs(paper_profiles[:, nonband_mask]))),
        "fixed_to_nonband_ratio": float(observed / np.mean(np.abs(paper_profiles[:, nonband_mask]))),
        "circular_shift_count": len(null),
        "circular_shift_p_greater_or_equal": float(np.mean(null >= observed)),
        "null_mean": float(null.mean()),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
    }
    write_json(args.output_dir / "fixed_band_responses" / "fixed_band_hub_test.json", hub_test)

    # Legacy hotspot overlap is descriptive: the legacy hotspot was selected on these mutants.
    legacy = pd.read_csv(args.legacy_hotspots)
    overlap_rows = []
    for hotspot in legacy.itertuples(index=False):
        if int(hotspot.hotspot_rank) > 20:
            continue
        position = int(hotspot.fasta_position)
        hit = primary_bands[
            (primary_bands.start_residue_1based <= position)
            & (primary_bands.end_residue_1based_inclusive >= position)
        ]
        overlap_rows.append({
            "legacy_hotspot_rank": int(hotspot.hotspot_rank),
            "fasta_position": position, "paper_position": hotspot.paper_position,
            "median_abs_delta_I": hotspot.median_abs_delta_I,
            "inside_primary_wt_band": bool(len(hit)),
            "wt_band_sign": int(hit.iloc[0].sign) if len(hit) else None,
            "wt_band_id": hit.iloc[0].band_id if len(hit) else None,
            "interpretation": "post_hoc_overlap_not_independent_validation",
        })
    write_csv(args.output_dir / "fixed_band_responses" / "legacy_hotspot_overlap.csv", overlap_rows)

    # Cross-condition band overlap and response-profile robustness.
    overlap_summary = []
    for condition in CONDITIONS:
        other = condition_band_lookup[condition]
        primary_mask = {sign: np.zeros(length, bool) for sign in (-1, 1)}
        other_mask = {sign: np.zeros(length, bool) for sign in (-1, 1)}
        for row in primary_bands.itertuples(index=False):
            primary_mask[int(row.sign)][int(row.start_index_0based):int(row.end_index_0based_inclusive)+1] = True
        for row in other.itertuples(index=False):
            other_mask[int(row.sign)][int(row.start_index_0based):int(row.end_index_0based_inclusive)+1] = True
        for sign in (-1, 1):
            intersection = int(np.sum(primary_mask[sign] & other_mask[sign]))
            union = int(np.sum(primary_mask[sign] | other_mask[sign]))
            overlap_summary.append({
                "condition": condition, "sign": sign,
                "primary_band_count": int((primary_bands.sign == sign).sum()),
                "condition_band_count": int((other.sign == sign).sum()),
                "residue_mask_jaccard": intersection / union if union else None,
                "primary_residues_covered_by_condition": intersection / np.sum(primary_mask[sign]),
                "condition_residues_covered_by_primary": intersection / np.sum(other_mask[sign]) if np.sum(other_mask[sign]) else None,
            })
    write_csv(args.output_dir / "robustness" / "wt_band_overlap_across_conditions.csv", overlap_summary)
    summary_frame = pd.DataFrame(variant_rows)
    response_robustness = []
    primary_values = summary_frame[
        (summary_frame.condition == PRIMARY_CONDITION) & summary_frame.is_paper_mutant
    ].set_index("variant")
    for condition in CONDITIONS:
        values = summary_frame[
            (summary_frame.condition == condition) & summary_frame.is_paper_mutant
        ].set_index("variant").loc[list(PAPER_MUTANTS)]
        for metric in (
            "global_mean_abs_delta_I", "fixed_band_mean_abs_delta_I",
            "fixed_band_receiver_mean_abs_delta_C", "mean_query_attention_total_variation",
        ):
            response_robustness.append({
                "condition": condition, "metric": metric,
                "pearson_vs_primary": float(np.corrcoef(
                    primary_values.loc[list(PAPER_MUTANTS), metric], values[metric]
                )[0, 1]),
                "spearman_vs_primary": float(spearmanr(
                    primary_values.loc[list(PAPER_MUTANTS), metric], values[metric]
                ).statistic),
            })
    write_csv(args.output_dir / "robustness" / "response_metric_robustness.csv", response_robustness)

    # Uniform-routing control: profile similarity plus band counts/overlap.
    similarity = pd.read_csv(args.uniform_profile_similarity)
    uniform_stable = pd.read_csv(args.uniform_stable_bands)
    uniform_mean = pd.read_csv(args.uniform_mean_bands)
    uniform_rows = []
    for condition in CONDITIONS:
        observed_condition = condition_band_lookup[condition]
        uniform_condition = uniform_stable[uniform_stable.condition == condition]
        sim = similarity[similarity.condition == condition]
        uniform_rows.append({
            "condition": condition,
            "observed_stable_band_count": len(observed_condition),
            "uniform_stable_band_count": len(uniform_condition),
            "uniform_mean_band_count": int((uniform_mean.condition == condition).sum()),
            "mean_seed_averaged_profile_pearson": float(
                sim.pearson_observed_uniform.mean()
            ),
            "interpretation": "no_uniform_seed_stable_bands" if uniform_condition.empty else "uniform_bands_present",
        })
    write_csv(args.output_dir / "uniform_control" / "uniform_control_summary.csv", uniform_rows)

    # Activity analysis: a small predeclared family and position-collapsed sensitivity.
    phenotype = pd.read_csv(args.phenotypes)
    phenotype = phenotype[phenotype.mutation.isin(PAPER_MUTANTS)].copy()
    primary_metrics = summary_frame[
        (summary_frame.condition == PRIMARY_CONDITION) & summary_frame.is_paper_mutant
    ].copy()
    primary_membership = pd.DataFrame(membership_rows)
    primary_membership = primary_membership[
        (primary_membership.condition == PRIMARY_CONDITION)
        & primary_membership.variant.isin(PAPER_MUTANTS)
    ]
    member_by_variant = primary_membership.groupby("variant").agg(
        inside_stable_band=("inside_stable_band", "max"),
        distance_to_nearest_stable_apex=("distance_to_nearest_stable_apex", "min"),
        mutation_paper_position=("paper_position", "min"),
    ).reset_index()
    activity = phenotype.merge(primary_metrics, left_on="mutation", right_on="variant")
    activity = activity.merge(member_by_variant, on="variant")
    if len(activity) != 34:
        raise ValueError(f"Expected 34 activity mutants, found {len(activity)}")
    activity["activity_deviation_abs_log"] = np.abs(np.log(activity.activity_replicate_mean))
    activity["group_Binding"] = (activity.group == "Binding").astype(float)
    activity["group_High_Strain"] = (activity.group == "High-Strain").astype(float)
    binding = activity.binding_strain.astype(float).to_numpy()
    binding = (binding - binding.mean()) / (binding.std() or 1.0)
    covariates = np.column_stack([
        np.ones(len(activity)), binding,
        activity.group_Binding.to_numpy(), activity.group_High_Strain.to_numpy(),
    ])
    activity["negative_distance_to_nearest_apex"] = -activity[
        "distance_to_nearest_stable_apex"
    ].astype(float)
    predictor_names = [
        "inside_stable_band",
        "negative_distance_to_nearest_apex",
        "fixed_band_mean_abs_delta_I",
        "fixed_band_receiver_mean_abs_delta_C",
        "global_mean_abs_delta_I",
        "mean_absolute_flexible_probability_change",
    ]
    y = activity.activity_deviation_abs_log.astype(float).to_numpy()
    association_rows = []
    for predictor_name in predictor_names:
        predictor = activity[predictor_name].astype(float).to_numpy()
        result = partial_test(y, covariates, predictor, rng, args.permutations)
        rho = spearmanr(predictor, y)
        association_rows.append({
            "predictor": predictor_name,
            "outcome": "absolute_log_activity_fold_change_from_WT_star",
            "n_variants": len(activity),
            "spearman_rho": float(rho.statistic),
            "spearman_asymptotic_p_descriptive": float(rho.pvalue),
            **result,
            "LOO_RMSE_improvement": result["reduced_LOO_RMSE"] - result["full_LOO_RMSE"],
        })
    adjusted = bh_adjust([row["freedman_lane_permutation_p"] for row in association_rows])
    for row, q in zip(association_rows, adjusted):
        row["BH_FDR_q"] = float(q)
    write_csv(args.output_dir / "phenotypes" / "activity_associations.csv", association_rows)

    group_rows = []
    for group, group_frame in activity.groupby("group"):
        for inside, subset in group_frame.groupby("inside_stable_band"):
            group_rows.append({
                "group": group, "inside_stable_band": bool(inside),
                "n_variants": len(subset),
                "mean_activity": float(subset.activity_replicate_mean.mean()),
                "median_activity": float(subset.activity_replicate_mean.median()),
                "mean_absolute_log_activity_change": float(
                    subset.activity_deviation_abs_log.mean()
                ),
                "mean_fixed_band_response": float(
                    subset.fixed_band_mean_abs_delta_I.mean()
                ),
            })
    write_csv(args.output_dir / "phenotypes" / "activity_group_descriptives.csv", group_rows)
    activity.to_csv(args.output_dir / "phenotypes" / "activity_analysis_table.csv", index=False)

    # Collapse substitutions at the same paper position to avoid treating chemical
    # alternatives at one site as independent locations.
    collapsed = activity.groupby("mutation_paper_position", as_index=False).agg({
        "activity_deviation_abs_log": "mean", "binding_strain": "mean",
        "group_Binding": "first", "group_High_Strain": "first",
        **{name: "mean" for name in predictor_names},
    })
    collapsed_binding = collapsed.binding_strain.to_numpy(dtype=float)
    collapsed_binding = (
        collapsed_binding - collapsed_binding.mean()
    ) / (collapsed_binding.std() or 1.0)
    collapsed_covariates = np.column_stack([
        np.ones(len(collapsed)), collapsed_binding,
        collapsed.group_Binding.to_numpy(), collapsed.group_High_Strain.to_numpy(),
    ])
    collapsed_rows = []
    collapsed_y = collapsed.activity_deviation_abs_log.to_numpy(dtype=float)
    for predictor_name in predictor_names:
        result = partial_test(
            collapsed_y, collapsed_covariates,
            collapsed[predictor_name].to_numpy(dtype=float), rng, args.permutations,
        )
        collapsed_rows.append({
            "predictor": predictor_name, "n_unique_mutation_positions": len(collapsed),
            **result,
        })
    adjusted = bh_adjust([row["freedman_lane_permutation_p"] for row in collapsed_rows])
    for row, q in zip(collapsed_rows, adjusted):
        row["BH_FDR_q"] = float(q)
    write_csv(args.output_dir / "phenotypes" / "activity_position_collapsed_sensitivity.csv", collapsed_rows)

    # Direct model-level B1/C1 comparison; experimental activity values remain
    # qualitative because the two papers use different constructs/experiments.
    trio_names = ["A175G", "A176G", "A175G_A176G"]
    trio = summary_frame[
        (summary_frame.condition == PRIMARY_CONDITION)
        & summary_frame.variant.isin(trio_names)
    ].copy()
    trio["construct_role"] = trio.variant.map({
        "A176G": "B1", "A175G_A176G": "C1", "A175G": "single-site decomposition control",
    })
    trio["reported_2018_activity_context"] = trio.variant.map({
        "A176G": "approximately 0.1x WT", "A175G_A176G": "approximately 10x WT",
        "A175G": "not reported as B1 or C1",
    })
    (args.output_dir / "b1_c1").mkdir(parents=True, exist_ok=True)
    trio.to_csv(args.output_dir / "b1_c1" / "b1_c1_model_comparison.csv", index=False)

    # Test model-level non-additivity directly on the signed matrices. Under an
    # additive response this interaction residual is zero.
    epistasis_c = []
    epistasis_probability = []
    for seed in SEEDS:
        run = loaded[(PRIMARY_CONDITION, seed)]
        epistasis_c.append(
            run["A175G_A176G"].contribution
            - run["A175G"].contribution
            - run["A176G"].contribution
            + run["WT_star"].contribution
        )
        epistasis_probability.append(
            run["A175G_A176G"].flexible_scores
            - run["A175G"].flexible_scores
            - run["A176G"].flexible_scores
            + run["WT_star"].flexible_scores
        )
    epistasis_c = np.stack(epistasis_c)
    mean_epistasis_c = epistasis_c.mean(axis=0)
    mean_epistasis_i = mean_epistasis_c.mean(axis=0)
    epistasis_receiver = [
        float(np.mean(np.abs(mean_epistasis_c[:, int(b.start_index_0based):
                                             int(b.end_index_0based_inclusive) + 1].sum(axis=1))))
        for b in primary_bands.itertuples(index=False)
    ]
    double_row = trio[trio.variant == "A175G_A176G"].iloc[0]
    epistasis_rows = [{
        "condition": PRIMARY_CONDITION,
        "interaction": "A175G_A176G_minus_A175G_minus_A176G_plus_WT_star",
        "global_mean_abs_interaction_delta_I": float(np.mean(np.abs(mean_epistasis_i))),
        "fixed_band_mean_abs_interaction_delta_I": float(
            np.mean(np.abs(mean_epistasis_i[band_mask]))
        ),
        "nonband_mean_abs_interaction_delta_I": float(
            np.mean(np.abs(mean_epistasis_i[nonband_mask]))
        ),
        "fixed_band_interaction_fraction_of_double_mutant_response": float(
            np.mean(np.abs(mean_epistasis_i[band_mask]))
            / double_row.fixed_band_mean_abs_delta_I
        ),
        "mean_absolute_flexible_probability_interaction": float(
            np.mean(np.abs(np.stack(epistasis_probability)))
        ),
        "fixed_band_receiver_mean_abs_interaction_delta_C": float(
            np.mean(epistasis_receiver)
        ),
        "query_contribution_L1_interaction": float(
            np.mean(np.abs(epistasis_c).sum(axis=2))
        ),
    }]
    write_csv(args.output_dir / "b1_c1" / "b1_c1_epistasis.csv", epistasis_rows)

    # Structural mapping tables and a ready-to-open PyMOL script.
    structural_mutations = pd.DataFrame(membership_rows)
    structural_mutations = structural_mutations[
        structural_mutations.condition == PRIMARY_CONDITION
    ].copy()
    (args.output_dir / "structure").mkdir(parents=True, exist_ok=True)
    structural_mutations.to_csv(
        args.output_dir / "structure" / "mutation_band_structure_map.csv", index=False
    )
    pml = [
        f"load {args.structure.resolve()}, gk", "hide everything, gk",
        "show cartoon, gk", "color gray80, gk",
    ]
    for row in mapped_band_rows:
        if row["pdb_resolved_start"] is None or row["pdb_resolved_end"] is None:
            continue
        color = "tv_orange" if row["sign"] == 1 else "marine"
        selection = f"band_{row['gk_band_number']}"
        pml.extend([
            f"select {selection}, gk and chain A and resi {int(row['pdb_resolved_start'])}-{int(row['pdb_resolved_end'])}",
            f"color {color}, {selection}", f"show sticks, {selection}",
        ])
    inside_positions = sorted(set(
        structural_mutations.loc[
            structural_mutations.inside_stable_band, "paper_position"
        ].dropna().astype(int)
    ))
    outside_positions = sorted(set(
        structural_mutations.loc[
            ~structural_mutations.inside_stable_band, "paper_position"
        ].dropna().astype(int)
    ))
    if inside_positions:
        pml.extend([
            "select mutations_in_band, gk and chain A and resi " + "+".join(map(str, inside_positions)),
            "show spheres, mutations_in_band", "color magenta, mutations_in_band",
        ])
    if outside_positions:
        pml.extend([
            "select mutations_outside_band, gk and chain A and resi " + "+".join(map(str, outside_positions)),
            "show spheres, mutations_outside_band", "color green, mutations_outside_band",
        ])
    pml.extend(["bg_color white", "orient gk"])
    (args.output_dir / "structure" / "view_wt_bands_and_mutations.pml").write_text(
        "\n".join(pml) + "\n", encoding="utf-8"
    )

    write_json(args.output_dir / "analysis_complete.json", {
        "schema": "esmfluc.gk_fixed_wt_bands.v2",
        "primary_condition": PRIMARY_CONDITION,
        "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "fasta_records": len(fasta), "primary_stable_bands": len(primary_bands),
        "all_condition_stable_bands": len(bands),
        "paper_mutants": len(PAPER_MUTANTS),
        "fixed_band_variant_condition_rows": len(variant_rows),
        "fixed_band_response_rows": len(band_response_rows),
        "primary_receiver_rows": len(primary_receiver_rows),
        "maximum_decomposition_error": maximum_decomposition_error,
        "stable_bands_validated_against_mean_catalog": True,
        "hub_test": hub_test,
        "uniform_stable_band_count": len(uniform_stable),
        "activity_tests": len(association_rows),
        "position_collapsed_activity_tests": len(collapsed_rows),
    })
    if maximum_decomposition_error > 1e-6:
        raise ValueError(f"Decomposition error too large: {maximum_decomposition_error}")


if __name__ == "__main__":
    main()
