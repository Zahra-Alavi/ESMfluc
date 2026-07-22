#!/usr/bin/env python3
"""Detect reproducible DeltaI bands, distances, distal burdens, and common hotspots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from weinreb_analysis_common import (
    NEW_SEQUENCES, WEINREB_ANALYSIS_ROOT, WEINREB_FASTA, WEINREB_RESULTS_ROOT,
    domain_annotation, effect_classification, mean_sd_se,
    mutation_sites, parse_ca, parse_position_map, read_fasta, sign_agreement,
    write_csv, json_dump,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, default=WEINREB_ANALYSIS_ROOT)
    p.add_argument("--fasta", type=Path, default=WEINREB_FASTA)
    p.add_argument("--position-map", type=Path, default=WEINREB_RESULTS_ROOT / "position_map_1ZNX_WT.csv")
    p.add_argument("--structure", type=Path, default=WEINREB_RESULTS_ROOT / "pdb_cache" / "1ZNX_WT.pdb")
    return p.parse_args()


def robust_scale(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def load_mapping(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    for r in rows:
        for k in ("paper_position", "fasta_position", "matrix_index"):
            r[k] = None if r[k] in {"", "None"} else int(r[k])
    return rows


def ca_distance(coords, p1, p2):
    if p1 is None or p2 is None or p1 not in coords or p2 not in coords:
        return None
    return float(np.linalg.norm(coords[p1] - coords[p2]))


def main() -> None:
    args = arguments()
    fasta = read_fasta(args.fasta)
    wt = fasta["WT_star"]
    position_map = parse_position_map(args.position_map)
    mapping = load_mapping(args.analysis / "coordinate_mapping.csv")
    coords = parse_ca(args.structure)
    with np.load(args.analysis / "profiles" / "delta_I_arrays.npz", allow_pickle=False) as z:
        names = [str(x) for x in z["mutant_names"]]
        seeds = z["seeds"].astype(int)
        delta = np.asarray(z["delta"], float)  # mutant, seed, key
        baseline = np.asarray(z["wt_star"], float)
    length = delta.shape[-1]
    paper_by_index = [r["paper_position"] for r in mapping]

    configs = []
    for sigma in (1.0, 1.5, 2.0):
        for prominence_mult in (0.5, 1.0, 1.5):
            for min_width in (2, 3, 4):
                configs.append((sigma, prominence_mult, min_width))
    band_rows = []
    primary_rows = []
    for mi, name in enumerate(names):
        mean, sd, se = mean_sd_se(delta[mi])
        agree = sign_agreement(delta[mi])
        sites = mutation_sites(fasta[name], wt)
        paper_sites = [position_map[p][1] for p in sites if p in position_map]
        for sigma, pmult, min_width in configs:
            smooth = gaussian_filter1d(mean, sigma=sigma, mode="nearest")
            # The primary search is deliberately conservative: prominence must
            # exceed both profile-scale variation and the upper quartile of
            # across-seed noise. This avoids turning tiny threshold crossings
            # into dozens of biological bands.
            base_prominence = max(1e-4, 3.0 * robust_scale(smooth),
                                  float(np.quantile(sd, 0.75)))
            for polarity in (1, -1):
                peaks, props = find_peaks(polarity * smooth,
                                          prominence=base_prominence * pmult,
                                          width=min_width)
                if not len(peaks):
                    continue
                widths = peak_widths(polarity * smooth, peaks, rel_height=0.7)
                for k, apex in enumerate(peaks):
                    start = max(0, int(np.floor(widths[2][k])))
                    end = min(length - 1, int(np.ceil(widths[3][k])))
                    # Enforce a genuinely contiguous band of the detected sign.
                    sign_start = int(apex)
                    sign_end = int(apex)
                    while sign_start > 0 and polarity * smooth[sign_start - 1] > 0:
                        sign_start -= 1
                    while sign_end < length - 1 and polarity * smooth[sign_end + 1] > 0:
                        sign_end += 1
                    start, end = max(start, sign_start), min(end, sign_end)
                    if end - start + 1 < min_width or agree[apex] < 2:
                        continue
                    seed_integrated = delta[mi, :, start:end + 1].sum(axis=1)
                    int_mean, int_sd, int_se = mean_sd_se(seed_integrated)
                    apex_ratio = abs(float(mean[apex])) / max(float(sd[apex]), 1e-12)
                    integrated_ratio = abs(float(int_mean)) / max(float(int_sd), 1e-12)
                    if apex_ratio < 1.0 or integrated_ratio < 1.0:
                        continue
                    apex_paper = paper_by_index[apex]
                    seq_distances = [abs((apex + 1) - p) for p in sites]
                    structural = [ca_distance(coords, apex_paper, p) for p in paper_sites]
                    structural = [x for x in structural if x is not None]
                    domain, function = domain_annotation(apex_paper)
                    baseline_band = float(baseline[:, start:end + 1].mean())
                    delta_band = float(mean[start:end + 1].mean())
                    row = {
                        "mutant": name, "polarity": "positive" if polarity == 1 else "negative",
                        "start_matrix_index": start, "apex_matrix_index": int(apex), "end_matrix_index": end,
                        "start_fasta_position": start + 1, "apex_fasta_position": int(apex + 1),
                        "end_fasta_position": end + 1, "start_paper_position": paper_by_index[start],
                        "apex_paper_position": apex_paper, "end_paper_position": paper_by_index[end],
                        "apex_delta_I": mean[apex], "integrated_delta_I": int_mean,
                        "mean_band_delta_I": delta_band, "wt_star_baseline_I": baseline_band,
                        "effect_classification": effect_classification(baseline_band, delta_band),
                        "seed_integrated_sd": int_sd, "seed_integrated_se": int_se,
                        "seed_sign_agreement_n": int(max(np.sum(seed_integrated > 0), np.sum(seed_integrated < 0))),
                        "apex_sign_agreement_n": int(agree[apex]),
                        "effect_to_seed_variation": integrated_ratio,
                        "apex_effect_to_seed_variation": apex_ratio,
                        "mutation_fasta_positions": ";".join(map(str, sites)),
                        "mutation_paper_positions": ";".join(map(str, paper_sites)),
                        "minimum_sequence_distance": min(seq_distances) if seq_distances else None,
                        "minimum_ca_distance_A": min(structural) if structural else None,
                        "domain": domain, "functional_annotation": function,
                        "gaussian_sigma": sigma, "prominence_multiplier": pmult,
                        "minimum_width": min_width,
                        "primary_parameters": sigma == 1.5 and pmult == 1.0 and min_width == 3,
                    }
                    band_rows.append(row)
                    if row["primary_parameters"]:
                        primary_rows.append(row)

    write_csv(args.analysis / "bands" / "change_bands_sensitivity.csv", band_rows)
    write_csv(args.analysis / "bands" / "change_bands_primary.csv", primary_rows)

    distal_rows = []
    for mi, name in enumerate(names):
        sites = mutation_sites(fasta[name], wt)
        paper_sites = [position_map[p][1] for p in sites if p in position_map]
        mean_abs = np.abs(delta[mi].mean(axis=0))
        for seq_cut in (10, 15, 20):
            for ca_cut in (10, 12, 15):
                selected, local, unmapped = [], [], []
                for j, value in enumerate(mean_abs):
                    seq_distance = min((abs(j + 1 - p) for p in sites), default=None)
                    p = paper_by_index[j]
                    d3 = [ca_distance(coords, p, x) for x in paper_sites]
                    d3 = [x for x in d3 if x is not None]
                    if not d3:
                        unmapped.append(value)
                    elif seq_distance is not None and seq_distance > seq_cut and min(d3) > ca_cut:
                        selected.append(value)
                    else:
                        local.append(value)
                distal_rows.append({
                    "mutant": name, "sequence_threshold": seq_cut, "ca_threshold_A": ca_cut,
                    "distal_count": len(selected), "local_or_near_count": len(local),
                    "structurally_unmapped_count": len(unmapped),
                    "distal_mean_abs_delta_I": np.mean(selected) if selected else None,
                    "local_mean_abs_delta_I": np.mean(local) if local else None,
                    "unmapped_mean_abs_delta_I": np.mean(unmapped) if unmapped else None,
                    "distal_integrated_abs_delta_I": np.sum(selected) if selected else None,
                })
    write_csv(args.analysis / "bands" / "distal_burden.csv", distal_rows)

    # Common response hotspots use the 34 paper mutants only: no construct control
    # and no newly added hypothesis sequences.
    original_idx = [i for i, n in enumerate(names) if n != "WT" and n not in NEW_SEQUENCES]
    original_names = [names[i] for i in original_idx]
    mean_matrix = delta[original_idx].mean(axis=1)
    h = np.median(np.abs(mean_matrix), axis=0)
    hotspot_rows = []
    residual_rows = []
    for j in range(length):
        order = int(np.sum(h > h[j]) + 1)
        hotspot_rows.append({
            "matrix_index": j, "fasta_position": j + 1, "paper_position": paper_by_index[j],
            "median_abs_delta_I": h[j], "mean_abs_delta_I": np.mean(np.abs(mean_matrix[:, j])),
            "hotspot_rank": order, "hotspot_percentile": 1.0 - (order - 1) / length,
            "response_fraction_above_mutant_profile_median": np.mean(
                np.abs(mean_matrix[:, j]) > np.median(np.abs(mean_matrix), axis=1)),
            "domain": mapping[j]["domain"], "functional_annotation": mapping[j]["functional_annotation"],
        })
        for mi, name in enumerate(original_names):
            others = np.delete(mean_matrix[:, j], mi)
            typical = np.median(others)
            scale = robust_scale(others)
            residual_rows.append({
                "mutant": name, "matrix_index": j, "fasta_position": j + 1,
                "paper_position": paper_by_index[j], "raw_delta_I": mean_matrix[mi, j],
                "leave_one_out_typical_delta_I": typical,
                "leave_one_out_residual_delta_I": mean_matrix[mi, j] - typical,
                "leave_one_out_standardized_delta_I": (mean_matrix[mi, j] - typical) / max(scale, 1e-12),
            })
    write_csv(args.analysis / "hotspots" / "common_hotspots.csv", hotspot_rows)
    write_csv(args.analysis / "hotspots" / "position_standardized_effects.csv", residual_rows)

    # Formal rank/random-window test of the preliminary matrix-position 155-157 claim.
    target = np.arange(154, 157)  # reported matrix positions 155-157, converted to zero based
    target_score = float(np.mean(h[target]))
    windows = np.array([np.mean(h[i:i + 3]) for i in range(length - 2)])
    target_domain = mapping[target[1]]["domain"]
    matched = np.array([np.mean(h[i:i + 3]) for i in range(length - 2)
                        if mapping[i + 1]["domain"] == target_domain])
    hotspot_test = {
        "reported_matrix_positions_one_based": [155, 156, 157],
        "matrix_indices_zero_based": target.tolist(),
        "fasta_positions": (target + 1).tolist(),
        "paper_positions": [paper_by_index[i] for i in target],
        "mean_median_abs_delta_I": target_score,
        "all_contiguous_windows_empirical_p": float((1 + np.sum(windows >= target_score)) / (1 + len(windows))),
        "domain_matched_windows": int(len(matched)),
        "domain_matched_empirical_p": float((1 + np.sum(matched >= target_score)) / (1 + len(matched))),
        "all_window_rank": int(1 + np.sum(windows > target_score)),
    }
    json_dump(args.analysis / "hotspots" / "hotspot_155_157_test.json", hotspot_test)
    json_dump(args.analysis / "bands" / "band_detection_config.json", {
        "primary": {"gaussian_sigma": 1.5, "prominence_multiplier": 1.0, "minimum_width": 3},
        "base_prominence": "max(1e-4, 3*robust MAD scale of smoothed profile, upper-quartile seed SD)",
        "reproducibility_filter": "at least 2/3 seed signs at apex plus apex and integrated |mean|/SD >= 1",
        "sensitivity_grid": {"gaussian_sigma": [1.0, 1.5, 2.0],
                             "prominence_multiplier": [0.5, 1.0, 1.5],
                             "minimum_width": [2, 3, 4]},
    })


if __name__ == "__main__":
    main()
