#!/usr/bin/env python3
"""Analyze apex-centered PWMs and exact k-mers for signed bands.

Raw PWMs use every band apex. Enrichment PWMs compare bands with same-protein,
exact-Q8 control anchors whose complete analysis window does not overlap a
band. Each matched band contributes total control weight one, regardless of
the number of available control anchors. Exact k-mers are discovered in train
and evaluated unchanged in validation and test using protein-level inference.
Reduced-alphabet motifs are tested independently with the same design.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as student_t


AA = tuple("ACDEFGHIKLMNPQRSTVWY")
BAND_LABELS = {1: "flexibility_supporting", -1: "rigidity_supporting"}
REDUCED_CLASS_MEMBERS = {
    "B": "KRH",
    "N": "DE",
    "P": "STNQC",
    "H": "AVILM",
    "A": "FWY",
    "T": "GP",
}
REDUCED_CLASS_LABELS = {
    "B": "positive",
    "N": "negative",
    "P": "polar",
    "H": "hydrophobic",
    "A": "aromatic",
    "T": "turn",
}
REDUCED_AA_TO_CLASS = {
    amino_acid: code
    for code, members in REDUCED_CLASS_MEMBERS.items()
    for amino_acid in members
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument(
        "--splits", nargs="*", default=("train", "validation", "test")
    )
    parser.add_argument("--window_radius", type=int, default=10)
    parser.add_argument("--controls_per_band", type=int, default=5)
    parser.add_argument(
        "--background_exclusion_radius", type=int, default=None,
        help="Distance kept between a control anchor and every band interval; defaults to window radius.",
    )
    parser.add_argument("--minimum_bands_per_q8_pwm", type=int, default=20)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--kmer_lengths", nargs="*", type=int, default=(2, 3, 4, 5))
    parser.add_argument("--minimum_kmer_proteins", type=int, default=5)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--discovery_split", default="train")
    parser.add_argument(
        "--replication_splits", nargs="*", default=("validation", "test")
    )
    parser.add_argument("--discovery_q_threshold", type=float, default=0.05)
    parser.add_argument("--discovery_min_abs_log2_odds", type=float, default=1.0)
    parser.add_argument("--replication_q_threshold", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def fixed_window(sequence: str, anchor: int, radius: int) -> str:
    return "".join(
        sequence[index] if 0 <= index < len(sequence) else "-"
        for index in range(anchor - radius, anchor + radius + 1)
    )


def prepare_residue_lookup(
    residue: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, np.ndarray]]:
    lookup = {}
    for key, group in residue.groupby(["split", "protein"], sort=False):
        group = group.sort_values("residue_index_0based")
        indices = group.residue_index_0based.to_numpy(int)
        if not np.array_equal(indices, np.arange(len(group))):
            raise ValueError(f"{key}: residue coordinates are not contiguous")
        sequence = "".join(group.amino_acid.astype(str))
        q8 = group.q8.fillna("unknown").astype(str).to_numpy()
        lookup[(str(key[0]), str(key[1]))] = (sequence, q8)
    return lookup


def blocked_control_anchors(
    length: int, bands: pd.DataFrame, exclusion_radius: int,
) -> np.ndarray:
    blocked = np.zeros(length, dtype=bool)
    for band in bands.itertuples(index=False):
        left = max(0, int(band.start_index_0based) - exclusion_radius)
        right = min(length, int(band.end_index_0based_inclusive) + exclusion_radius + 1)
        blocked[left:right] = True
    return blocked


def build_windows(
    bands: pd.DataFrame, residue: pd.DataFrame, radius: int,
    controls_per_band: int, exclusion_radius: int, random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    residue_lookup = prepare_residue_lookup(residue)
    rng = np.random.default_rng(random_seed)
    rows = []
    coverage_rows = []
    failures = Counter()
    for context, protein_bands in bands.groupby(
        ["condition", "split", "protein"], sort=False
    ):
        condition, split, protein = map(str, context)
        annotation = residue_lookup.get((split, protein))
        if annotation is None:
            failures["missing_residue_annotations"] += len(protein_bands)
            continue
        sequence, q8 = annotation
        blocked = blocked_control_anchors(
            len(sequence), protein_bands, exclusion_radius
        )
        for band in protein_bands.itertuples(index=False):
            apex = int(band.apex_index_0based)
            sign = int(band.sign)
            apex_q8 = str(q8[apex])
            candidates = np.flatnonzero(~blocked & (q8 == apex_q8))
            if controls_per_band and len(candidates) > controls_per_band:
                candidates = np.sort(rng.choice(
                    candidates, controls_per_band, replace=False
                ))
            n_controls = len(candidates)
            if not n_controls:
                failures["no_same_protein_exact_q8_background"] += 1
            common = {
                "condition": condition, "split": split, "protein": protein,
                "matched_set_id": str(band.band_id), "band_id": str(band.band_id),
                "sign": sign, "band_type": BAND_LABELS[sign],
                "apex_q8": apex_q8, "window_radius": radius,
                "n_controls_for_band": n_controls,
            }
            rows.append({
                **common, "is_band": 1, "anchor_index_0based": apex,
                "anchor_amino_acid": sequence[apex],
                "window_sequence": fixed_window(sequence, apex, radius),
                "control_weight": 0.0,
            })
            for anchor in candidates:
                rows.append({
                    **common, "is_band": 0,
                    "anchor_index_0based": int(anchor),
                    "anchor_amino_acid": sequence[int(anchor)],
                    "window_sequence": fixed_window(sequence, int(anchor), radius),
                    "control_weight": 1.0 / n_controls,
                })
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No band windows could be constructed")
    cases = windows[windows.is_band.eq(1)]
    for key, group in cases.groupby(
        ["condition", "split", "sign", "band_type", "apex_q8"], sort=False
    ):
        matched = group.n_controls_for_band.gt(0)
        coverage_rows.append({
            **dict(zip(
                ["condition", "split", "sign", "band_type", "apex_q8"], key
            )),
            "n_bands": len(group), "n_proteins": int(group.protein.nunique()),
            "n_bands_with_background": int(matched.sum()),
            "background_match_fraction": float(matched.mean()),
            "mean_controls_per_matched_band": float(
                group.loc[matched, "n_controls_for_band"].mean()
            ) if matched.any() else np.nan,
        })
    audit = {
        "input_bands": len(bands), "band_windows": len(cases),
        "control_windows": int(windows.is_band.eq(0).sum()),
        "bands_with_background": int(cases.n_controls_for_band.gt(0).sum()),
        "bands_without_background": int(cases.n_controls_for_band.eq(0).sum()),
        "failures": dict(failures),
    }
    return windows, pd.DataFrame(coverage_rows), audit


def character_counts(
    sequences: pd.Series, offset_index: int, weights: np.ndarray,
) -> tuple[dict[str, float], float]:
    characters = sequences.str[offset_index].to_numpy()
    valid = np.isin(characters, AA)
    denominator = float(weights[valid].sum())
    counts = {
        amino_acid: float(weights[characters == amino_acid].sum())
        for amino_acid in AA
    }
    return counts, denominator


def pwm_rows_for_group(
    group: pd.DataFrame, stratum: str, q8_label: str,
    radius: int, pseudocount: float,
) -> list[dict]:
    cases = group[group.is_band.eq(1)].copy()
    matched_ids = set(cases.loc[
        cases.n_controls_for_band.gt(0), "matched_set_id"
    ].astype(str))
    matched_cases = cases[cases.matched_set_id.astype(str).isin(matched_ids)]
    controls = group[
        group.is_band.eq(0) & group.matched_set_id.astype(str).isin(matched_ids)
    ]
    rows = []
    first = group.iloc[0]
    case_weights = np.ones(len(cases), dtype=float)
    matched_case_weights = np.ones(len(matched_cases), dtype=float)
    control_weights = controls.control_weight.to_numpy(float)
    for offset_index, offset in enumerate(range(-radius, radius + 1)):
        all_counts, all_total = character_counts(
            cases.window_sequence, offset_index, case_weights
        )
        matched_counts, matched_total = character_counts(
            matched_cases.window_sequence, offset_index, matched_case_weights
        )
        control_counts, control_total = character_counts(
            controls.window_sequence, offset_index, control_weights
        )
        matched_smoothed_total = matched_total + pseudocount * len(AA)
        control_smoothed_total = control_total + pseudocount * len(AA)
        for amino_acid in AA:
            all_probability = all_counts[amino_acid] / all_total if all_total else np.nan
            matched_probability = (
                matched_counts[amino_acid] / matched_total if matched_total else np.nan
            )
            background_probability = (
                control_counts[amino_acid] / control_total if control_total else np.nan
            )
            if matched_total and control_total:
                band_smoothed = (
                    matched_counts[amino_acid] + pseudocount
                ) / matched_smoothed_total
                background_smoothed = (
                    control_counts[amino_acid] + pseudocount
                ) / control_smoothed_total
                log2_enrichment = float(np.log2(
                    band_smoothed / background_smoothed
                ))
            else:
                log2_enrichment = np.nan
            rows.append({
                "condition": first.condition, "split": first.split,
                "sign": int(first.sign), "band_type": first.band_type,
                "stratum": stratum, "apex_q8": q8_label,
                "window_radius": radius, "offset": offset,
                "amino_acid": amino_acid,
                "n_bands": len(cases),
                "n_band_proteins": int(cases.protein.nunique()),
                "n_matched_bands": len(matched_cases),
                "n_matched_band_proteins": int(matched_cases.protein.nunique()),
                "effective_background_windows": control_total,
                "all_band_count": all_counts[amino_acid],
                "all_band_probability": all_probability,
                "matched_band_count": matched_counts[amino_acid],
                "matched_band_probability": matched_probability,
                "matched_background_count": control_counts[amino_acid],
                "matched_background_probability": background_probability,
                "log2_matched_background_enrichment": log2_enrichment,
            })
    return rows


def calculate_pwms(
    windows: pd.DataFrame, radius: int, pseudocount: float,
    minimum_q8_bands: int,
) -> pd.DataFrame:
    rows = []
    for _, group in windows.groupby(
        ["condition", "split", "sign"], sort=False
    ):
        rows.extend(pwm_rows_for_group(
            group, "sign_only", "ALL", radius, pseudocount
        ))
        for q8, q8_group in group.groupby("apex_q8", sort=False):
            n_cases = int(q8_group.is_band.eq(1).sum())
            if n_cases >= minimum_q8_bands:
                rows.extend(pwm_rows_for_group(
                    q8_group, "sign_by_q8", str(q8), radius, pseudocount
                ))
    return pd.DataFrame(rows)


def bh(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return output
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index.to_numpy()[order]] = np.minimum(adjusted, 1.0)
    return output


def matched_design(data: pd.DataFrame) -> dict | None:
    case_rows = data.index[data.is_band.eq(1)].to_numpy(int)
    control_rows = data.index[data.is_band.eq(0)].to_numpy(int)
    if not len(case_rows) or not len(control_rows):
        return None
    case_sets = data.loc[case_rows, "matched_set_id"].astype(str)
    if case_sets.duplicated().any():
        raise ValueError("Matched k-mer sets must contain exactly one band row")
    set_number = {value: number for number, value in enumerate(case_sets)}
    control_set = data.loc[control_rows, "matched_set_id"].astype(str).map(set_number)
    keep = control_set.notna().to_numpy()
    control_rows = control_rows[keep]
    control_set = control_set[keep].to_numpy(int)
    sets_with_controls = np.unique(control_set)
    if not len(sets_with_controls):
        return None
    case_rows = case_rows[np.isin(np.arange(len(case_rows)), sets_with_controls)]
    old_to_new = np.full(len(set_number), -1, dtype=int)
    old_to_new[sets_with_controls] = np.arange(len(sets_with_controls))
    control_set = old_to_new[control_set]
    protein_labels, protein_set = np.unique(
        data.loc[case_rows, "protein"].astype(str).to_numpy(), return_inverse=True
    )
    row_to_case = np.full(len(data), -1, dtype=int)
    row_to_case[case_rows] = np.arange(len(case_rows))
    row_to_control_set = np.full(len(data), -1, dtype=int)
    row_to_control_set[control_rows] = control_set
    return {
        "case_rows": case_rows,
        "control_rows": control_rows,
        "control_set": control_set,
        "control_count": np.bincount(control_set, minlength=len(case_rows)),
        "protein_set": protein_set,
        "n_proteins": len(protein_labels),
        "row_to_case": row_to_case,
        "row_to_control_set": row_to_control_set,
    }


def empty_matched_result() -> dict:
    return {
        "n_proteins": 0, "n_matched_bands": 0,
        "band_frequency": np.nan, "control_frequency": np.nan,
        "conditional_odds_ratio": np.nan, "log2_odds_ratio": np.nan,
        "protein_mean_difference": np.nan,
        "ci95_low": np.nan, "ci95_high": np.nan,
        "inference_eligible": False, "protein_sign_test_p": np.nan,
    }


def matched_binary_association(
    occurrence_rows: list[int], design: dict | None,
    minimum_proteins: int,
) -> dict:
    if design is None:
        return empty_matched_result()
    case = np.zeros(len(design["case_rows"]), dtype=float)
    control_hits = np.zeros(len(case), dtype=float)
    if occurrence_rows:
        rows = np.asarray(occurrence_rows, dtype=int)
        case_numbers = design["row_to_case"][rows]
        case_numbers = case_numbers[case_numbers >= 0]
        case[case_numbers] = 1.0
        control_sets = design["row_to_control_set"][rows]
        control_sets = control_sets[control_sets >= 0]
        control_hits = np.bincount(control_sets, minlength=len(case)).astype(float)
    control = control_hits / design["control_count"]
    set_difference = case - control
    protein_count = np.bincount(
        design["protein_set"], minlength=design["n_proteins"]
    )
    protein_difference = np.bincount(
        design["protein_set"], weights=set_difference,
        minlength=design["n_proteins"],
    ) / protein_count
    n_proteins = design["n_proteins"]
    mean_difference = float(np.mean(protein_difference))
    standard_deviation = (
        float(np.std(protein_difference, ddof=1)) if n_proteins > 1 else np.nan
    )
    eligible = n_proteins >= minimum_proteins
    half_width = (
        float(student_t.ppf(0.975, n_proteins - 1)
              * standard_deviation / math.sqrt(n_proteins))
        if eligible else np.nan
    )
    nonzero = protein_difference[protein_difference != 0]
    p_value = (
        float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
        if eligible and len(nonzero) else (1.0 if eligible else np.nan)
    )
    band_success = float(case.sum())
    control_success = float(control.sum())
    total = len(case)
    odds_ratio = ((band_success + 0.5) / (total - band_success + 0.5)) / (
        (control_success + 0.5) / (total - control_success + 0.5)
    )
    return {
        "n_proteins": n_proteins, "n_matched_bands": total,
        "band_frequency": float(case.mean()),
        "control_frequency": float(control.mean()),
        "conditional_odds_ratio": float(odds_ratio),
        "log2_odds_ratio": float(np.log2(odds_ratio)),
        "protein_mean_difference": mean_difference,
        "ci95_low": mean_difference - half_width,
        "ci95_high": mean_difference + half_width,
        "inference_eligible": eligible,
        "protein_sign_test_p": p_value,
    }


def unique_kmers(sequence: str, lengths: tuple[int, ...]) -> set[str]:
    output = set()
    valid = set(AA)
    for k in lengths:
        for start in range(len(sequence) - k + 1):
            kmer = sequence[start:start + k]
            if set(kmer).issubset(valid):
                output.add(kmer)
    return output


def analysis_strata(
    data: pd.DataFrame, minimum_q8_bands: int,
):
    for (condition, sign), group in data.groupby(
        ["condition", "sign"], sort=False
    ):
        yield str(condition), int(sign), "sign_only", "ALL", group
        for q8, q8_group in group.groupby("apex_q8", sort=False):
            if int(q8_group.is_band.eq(1).sum()) >= minimum_q8_bands:
                yield str(condition), int(sign), "sign_by_q8", str(q8), q8_group


def kmer_occurrence_rows(
    data: pd.DataFrame, lengths: tuple[int, ...],
    targets: set[str] | None = None,
) -> dict[str, list[int]]:
    rows = defaultdict(list)
    for row_number, sequence in enumerate(data.window_sequence.astype(str)):
        for kmer in unique_kmers(sequence, lengths):
            if targets is None or kmer in targets:
                rows[kmer].append(row_number)
    return rows


def discover_kmers(
    windows: pd.DataFrame, discovery_split: str, lengths: tuple[int, ...],
    minimum_recurrence: int, minimum_inference: int,
    minimum_q8_bands: int, q_threshold: float,
    minimum_abs_log2_odds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = windows[windows.split.eq(discovery_split)].copy()
    rows = []
    for condition, sign, stratum, q8, raw_group in analysis_strata(
        discovery, minimum_q8_bands
    ):
        group = raw_group.reset_index(drop=True)
        design = matched_design(group)
        if design is None:
            continue
        protein_occurrence = defaultdict(set)
        for row_number in design["case_rows"]:
            protein = str(group.loc[row_number, "protein"])
            for kmer in unique_kmers(
                str(group.loc[row_number, "window_sequence"]), lengths
            ):
                protein_occurrence[kmer].add(protein)
        candidates = {
            kmer for kmer, proteins in protein_occurrence.items()
            if len(proteins) >= minimum_recurrence
        }
        occurrences = kmer_occurrence_rows(group, lengths, candidates)
        band_type = BAND_LABELS[sign]
        for kmer in sorted(candidates, key=lambda value: (len(value), value)):
            result = matched_binary_association(
                occurrences.get(kmer, []), design, minimum_inference
            )
            rows.append({
                "condition": condition, "discovery_split": discovery_split,
                "sign": sign, "band_type": band_type,
                "stratum": stratum, "apex_q8": q8,
                "window_radius": int(group.window_radius.iloc[0]),
                "k": len(kmer), "kmer": kmer,
                "band_protein_recurrence": len(protein_occurrence[kmer]),
                **result,
            })
    tests = pd.DataFrame(rows)
    if tests.empty:
        tests["protein_level_q_bh"] = pd.Series(dtype=float)
        tests["train_discovered"] = pd.Series(dtype=bool)
        return tests, tests.copy()
    tests["protein_level_q_bh"] = bh(tests.protein_sign_test_p)
    tests["train_discovered"] = (
        tests.inference_eligible.astype(bool)
        & tests.protein_level_q_bh.le(q_threshold)
        & tests.log2_odds_ratio.abs().ge(minimum_abs_log2_odds)
    )
    return tests, tests[tests.train_discovered].copy()


def evaluate_discovered_kmers(
    windows: pd.DataFrame, discovered: pd.DataFrame,
    replication_splits: tuple[str, ...], minimum_inference: int,
    q_threshold: float,
) -> pd.DataFrame:
    rows = []
    if discovered.empty:
        return pd.DataFrame(rows)
    keys = ["condition", "sign", "band_type", "stratum", "apex_q8"]
    for split in replication_splits:
        split_windows = windows[windows.split.eq(split)]
        for key, definitions in discovered.groupby(keys, sort=False):
            condition, sign, band_type, stratum, q8 = key
            group = split_windows[
                split_windows.condition.astype(str).eq(str(condition))
                & split_windows.sign.eq(int(sign))
            ]
            if stratum == "sign_by_q8":
                group = group[group.apex_q8.astype(str).eq(str(q8))]
            group = group.reset_index(drop=True)
            design = matched_design(group) if not group.empty else None
            targets = set(definitions.kmer.astype(str))
            lengths = tuple(sorted({len(kmer) for kmer in targets}))
            occurrences = (
                kmer_occurrence_rows(group, lengths, targets)
                if not group.empty else {}
            )
            for definition in definitions.itertuples(index=False):
                result = matched_binary_association(
                    occurrences.get(str(definition.kmer), []),
                    design, minimum_inference,
                )
                rows.append({
                    "condition": condition, "evaluation_split": split,
                    "sign": int(sign), "band_type": band_type,
                    "stratum": stratum, "apex_q8": q8,
                    "window_radius": int(definition.window_radius),
                    "k": int(definition.k), "kmer": str(definition.kmer),
                    "train_band_protein_recurrence": int(
                        definition.band_protein_recurrence
                    ),
                    "train_band_frequency": float(definition.band_frequency),
                    "train_control_frequency": float(definition.control_frequency),
                    "train_log2_odds_ratio": float(definition.log2_odds_ratio),
                    "train_q_value": float(definition.protein_level_q_bh),
                    **result,
                })
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["protein_level_q_bh"] = output.groupby(
        "evaluation_split", group_keys=False
    ).protein_sign_test_p.transform(bh)
    output["direction_matches_train"] = (
        np.sign(output.log2_odds_ratio) == np.sign(output.train_log2_odds_ratio)
    )
    output["replicated"] = (
        output.inference_eligible.astype(bool)
        & output.direction_matches_train.astype(bool)
        & output.protein_level_q_bh.le(q_threshold)
    )
    return output


def summarize_replication(
    discovered: pd.DataFrame, replication: pd.DataFrame,
    replication_splits: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    group_keys = ["condition", "sign", "band_type", "stratum", "apex_q8"]
    for key, definitions in discovered.groupby(group_keys, sort=False):
        common = dict(zip(group_keys, key))
        for split in replication_splits:
            evaluated = replication[
                replication.evaluation_split.eq(split)
                & replication.condition.astype(str).eq(str(common["condition"]))
                & replication.sign.eq(int(common["sign"]))
                & replication.stratum.eq(str(common["stratum"]))
                & replication.apex_q8.astype(str).eq(str(common["apex_q8"]))
            ] if not replication.empty else replication
            rows.append({
                **common, "evaluation_split": split,
                "n_train_discovered": len(definitions),
                "n_inference_eligible": int(
                    evaluated.inference_eligible.sum()
                ) if not evaluated.empty else 0,
                "n_direction_matches_train": int(
                    evaluated.direction_matches_train.sum()
                ) if not evaluated.empty else 0,
                "n_replicated": int(
                    evaluated.replicated.sum()
                ) if not evaluated.empty else 0,
            })
    return pd.DataFrame(rows)


def combine_replication_splits(
    discovered: pd.DataFrame, replication: pd.DataFrame,
    replication_splits: tuple[str, ...],
) -> pd.DataFrame:
    identifiers = [
        "condition", "sign", "band_type", "stratum", "apex_q8",
        "window_radius", "k", "kmer",
    ]
    train_columns = identifiers + [
        "band_protein_recurrence", "band_frequency", "control_frequency",
        "log2_odds_ratio", "protein_level_q_bh",
    ]
    if discovered.empty:
        return pd.DataFrame(columns=train_columns + [
            "n_replication_splits_passed", "replicated_all_splits",
        ])
    output = discovered[train_columns].copy().rename(columns={
        "band_protein_recurrence": "train_band_protein_recurrence",
        "band_frequency": "train_band_frequency",
        "control_frequency": "train_control_frequency",
        "log2_odds_ratio": "train_log2_odds_ratio",
        "protein_level_q_bh": "train_q_value",
    })
    for split in replication_splits:
        split_results = replication[
            replication.evaluation_split.eq(split)
        ][identifiers + [
            "n_proteins", "n_matched_bands", "band_frequency",
            "control_frequency", "log2_odds_ratio", "protein_sign_test_p",
            "protein_level_q_bh", "direction_matches_train", "replicated",
        ]].copy()
        split_results = split_results.rename(columns={
            column: f"{split}_{column}" for column in split_results.columns
            if column not in identifiers
        })
        output = output.merge(
            split_results, on=identifiers, how="left", validate="one_to_one"
        )
    replication_columns = [f"{split}_replicated" for split in replication_splits]
    for column in replication_columns:
        if column not in output:
            output[column] = False
        output[column] = output[column].fillna(False).astype(bool)
    output["n_replication_splits_passed"] = output[replication_columns].sum(axis=1)
    output["replicated_all_splits"] = output[replication_columns].all(axis=1)
    return output


def reduced_sequence(sequence: str) -> str:
    return "".join(REDUCED_AA_TO_CLASS.get(amino_acid, "-") for amino_acid in sequence)


def reduced_motif_name(code: str) -> str:
    return "-".join(REDUCED_CLASS_LABELS[symbol] for symbol in code)


def unique_reduced_motifs(sequence: str, lengths: tuple[int, ...]) -> set[str]:
    encoded = reduced_sequence(sequence)
    valid = set(REDUCED_CLASS_LABELS)
    output = set()
    for k in lengths:
        for start in range(len(encoded) - k + 1):
            motif = encoded[start:start + k]
            if set(motif).issubset(valid):
                output.add(motif)
    return output


def reduced_motif_occurrence_rows(
    data: pd.DataFrame, lengths: tuple[int, ...],
    targets: set[str] | None = None,
) -> dict[str, list[int]]:
    rows = defaultdict(list)
    for row_number, sequence in enumerate(data.window_sequence.astype(str)):
        for motif in unique_reduced_motifs(sequence, lengths):
            if targets is None or motif in targets:
                rows[motif].append(row_number)
    return rows


def discover_reduced_motifs(
    windows: pd.DataFrame, discovery_split: str, lengths: tuple[int, ...],
    minimum_recurrence: int, minimum_inference: int,
    minimum_q8_bands: int, q_threshold: float,
    minimum_abs_log2_odds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = windows[windows.split.eq(discovery_split)].copy()
    rows = []
    for condition, sign, stratum, q8, raw_group in analysis_strata(
        discovery, minimum_q8_bands
    ):
        group = raw_group.reset_index(drop=True)
        design = matched_design(group)
        if design is None:
            continue
        protein_occurrence = defaultdict(set)
        for row_number in design["case_rows"]:
            protein = str(group.loc[row_number, "protein"])
            for motif in unique_reduced_motifs(
                str(group.loc[row_number, "window_sequence"]), lengths
            ):
                protein_occurrence[motif].add(protein)
        candidates = {
            motif for motif, proteins in protein_occurrence.items()
            if len(proteins) >= minimum_recurrence
        }
        occurrences = reduced_motif_occurrence_rows(group, lengths, candidates)
        band_type = BAND_LABELS[sign]
        for motif in sorted(candidates, key=lambda value: (len(value), value)):
            result = matched_binary_association(
                occurrences.get(motif, []), design, minimum_inference
            )
            rows.append({
                "condition": condition, "discovery_split": discovery_split,
                "sign": sign, "band_type": band_type,
                "stratum": stratum, "apex_q8": q8,
                "window_radius": int(group.window_radius.iloc[0]),
                "k": len(motif), "motif_code": motif,
                "motif": reduced_motif_name(motif),
                "band_protein_recurrence": len(protein_occurrence[motif]),
                **result,
            })
    tests = pd.DataFrame(rows)
    if tests.empty:
        tests["protein_level_q_bh"] = pd.Series(dtype=float)
        tests["train_discovered"] = pd.Series(dtype=bool)
        return tests, tests.copy()
    tests["protein_level_q_bh"] = bh(tests.protein_sign_test_p)
    tests["train_discovered"] = (
        tests.inference_eligible.astype(bool)
        & tests.protein_level_q_bh.le(q_threshold)
        & tests.log2_odds_ratio.abs().ge(minimum_abs_log2_odds)
    )
    return tests, tests[tests.train_discovered].copy()


def evaluate_discovered_reduced_motifs(
    windows: pd.DataFrame, discovered: pd.DataFrame,
    replication_splits: tuple[str, ...], minimum_inference: int,
    q_threshold: float,
) -> pd.DataFrame:
    rows = []
    if discovered.empty:
        return pd.DataFrame(rows)
    keys = ["condition", "sign", "band_type", "stratum", "apex_q8"]
    for split in replication_splits:
        split_windows = windows[windows.split.eq(split)]
        for key, definitions in discovered.groupby(keys, sort=False):
            condition, sign, band_type, stratum, q8 = key
            group = split_windows[
                split_windows.condition.astype(str).eq(str(condition))
                & split_windows.sign.eq(int(sign))
            ]
            if stratum == "sign_by_q8":
                group = group[group.apex_q8.astype(str).eq(str(q8))]
            group = group.reset_index(drop=True)
            design = matched_design(group) if not group.empty else None
            targets = set(definitions.motif_code.astype(str))
            lengths = tuple(sorted({len(motif) for motif in targets}))
            occurrences = (
                reduced_motif_occurrence_rows(group, lengths, targets)
                if not group.empty else {}
            )
            for definition in definitions.itertuples(index=False):
                result = matched_binary_association(
                    occurrences.get(str(definition.motif_code), []),
                    design, minimum_inference,
                )
                rows.append({
                    "condition": condition, "evaluation_split": split,
                    "sign": int(sign), "band_type": band_type,
                    "stratum": stratum, "apex_q8": q8,
                    "window_radius": int(definition.window_radius),
                    "k": int(definition.k),
                    "motif_code": str(definition.motif_code),
                    "motif": str(definition.motif),
                    "train_band_protein_recurrence": int(
                        definition.band_protein_recurrence
                    ),
                    "train_band_frequency": float(definition.band_frequency),
                    "train_control_frequency": float(definition.control_frequency),
                    "train_log2_odds_ratio": float(definition.log2_odds_ratio),
                    "train_q_value": float(definition.protein_level_q_bh),
                    **result,
                })
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["protein_level_q_bh"] = output.groupby(
        "evaluation_split", group_keys=False
    ).protein_sign_test_p.transform(bh)
    output["direction_matches_train"] = (
        np.sign(output.log2_odds_ratio) == np.sign(output.train_log2_odds_ratio)
    )
    output["replicated"] = (
        output.inference_eligible.astype(bool)
        & output.direction_matches_train.astype(bool)
        & output.protein_level_q_bh.le(q_threshold)
    )
    return output


def combine_reduced_replication_splits(
    discovered: pd.DataFrame, replication: pd.DataFrame,
    replication_splits: tuple[str, ...],
) -> pd.DataFrame:
    identifiers = [
        "condition", "sign", "band_type", "stratum", "apex_q8",
        "window_radius", "k", "motif_code", "motif",
    ]
    train_columns = identifiers + [
        "band_protein_recurrence", "band_frequency", "control_frequency",
        "log2_odds_ratio", "protein_level_q_bh",
    ]
    if discovered.empty:
        return pd.DataFrame(columns=train_columns + [
            "n_replication_splits_passed", "replicated_all_splits",
        ])
    output = discovered[train_columns].copy().rename(columns={
        "band_protein_recurrence": "train_band_protein_recurrence",
        "band_frequency": "train_band_frequency",
        "control_frequency": "train_control_frequency",
        "log2_odds_ratio": "train_log2_odds_ratio",
        "protein_level_q_bh": "train_q_value",
    })
    for split in replication_splits:
        split_results = replication[
            replication.evaluation_split.eq(split)
        ][identifiers + [
            "n_proteins", "n_matched_bands", "band_frequency",
            "control_frequency", "log2_odds_ratio", "protein_sign_test_p",
            "protein_level_q_bh", "direction_matches_train", "replicated",
        ]].copy()
        split_results = split_results.rename(columns={
            column: f"{split}_{column}" for column in split_results.columns
            if column not in identifiers
        })
        output = output.merge(
            split_results, on=identifiers, how="left", validate="one_to_one"
        )
    replication_columns = [f"{split}_replicated" for split in replication_splits]
    for column in replication_columns:
        if column not in output:
            output[column] = False
        output[column] = output[column].fillna(False).astype(bool)
    output["n_replication_splits_passed"] = output[replication_columns].sum(axis=1)
    output["replicated_all_splits"] = output[replication_columns].all(axis=1)
    return output


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))


def write_matrix_tables(pwm: pd.DataFrame, root: Path) -> int:
    count = 0
    keys = [
        "condition", "split", "sign", "band_type", "stratum", "apex_q8",
        "window_radius",
    ]
    for key, group in pwm.groupby(keys, sort=False):
        condition, split, sign, band_type, stratum, q8, radius = key
        directory = root / safe_name(condition) / safe_name(split)
        directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{safe_name(band_type)}__{safe_name(stratum)}__q8_{safe_name(q8)}"
            f"__pm{int(radius)}"
        )
        for value in (
            "all_band_probability", "matched_band_probability",
            "matched_background_probability",
            "log2_matched_background_enrichment",
        ):
            matrix = group.pivot(
                index="offset", columns="amino_acid", values=value
            ).reindex(columns=AA)
            matrix.to_csv(directory / f"{stem}__{value}.csv")
        count += 1
    return count


def write_heatmaps(pwm: pd.DataFrame, root: Path) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/esmfluc_matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = 0
    keys = [
        "condition", "split", "sign", "band_type", "stratum", "apex_q8",
        "window_radius",
    ]
    for key, group in pwm.groupby(keys, sort=False):
        condition, split, sign, band_type, stratum, q8, radius = key
        offsets = sorted(group.offset.unique())
        probability = group.pivot(
            index="amino_acid", columns="offset", values="all_band_probability"
        ).reindex(index=AA, columns=offsets)
        enrichment = group.pivot(
            index="amino_acid", columns="offset",
            values="log2_matched_background_enrichment",
        ).reindex(index=AA, columns=offsets)
        finite = np.abs(enrichment.to_numpy(float))
        finite = finite[np.isfinite(finite)]
        enrichment_limit = max(1.0, float(np.quantile(finite, 0.98))) if len(finite) else 1.0
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        image0 = axes[0].imshow(
            probability.to_numpy(float), aspect="auto", cmap="viridis",
            vmin=0, vmax=max(0.15, float(np.nanmax(probability.to_numpy(float)))),
        )
        image1 = axes[1].imshow(
            enrichment.to_numpy(float), aspect="auto", cmap="coolwarm",
            vmin=-enrichment_limit, vmax=enrichment_limit,
        )
        for axis in axes:
            axis.set_yticks(np.arange(len(AA)), AA)
            axis.set_xticks(np.arange(len(offsets)), offsets)
            axis.set_xlabel("Offset from band apex")
            axis.set_ylabel("Amino acid")
            axis.axvline(offsets.index(0), color="black", linewidth=0.7)
        axes[0].set_title("Raw band PWM")
        axes[1].set_title("Log2 enrichment over matched background")
        figure.colorbar(image0, ax=axes[0], label="Probability", fraction=0.025)
        figure.colorbar(image1, ax=axes[1], label="Log2 enrichment", fraction=0.025)
        figure.suptitle(
            f"{condition} | {split} | {band_type} | {stratum} | Q8={q8}"
        )
        directory = root / safe_name(condition) / safe_name(split)
        directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{safe_name(band_type)}__{safe_name(stratum)}__q8_{safe_name(q8)}"
            f"__pm{int(radius)}.png"
        )
        figure.savefig(directory / stem, dpi=180)
        plt.close(figure)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.window_radius < 0:
        raise ValueError("--window_radius must be nonnegative")
    if args.controls_per_band < 1:
        raise ValueError("--controls_per_band must be positive")
    if args.minimum_bands_per_q8_pwm < 1:
        raise ValueError("--minimum_bands_per_q8_pwm must be positive")
    if args.pseudocount <= 0:
        raise ValueError("--pseudocount must be positive")
    if any(length < 2 for length in args.kmer_lengths):
        raise ValueError("--kmer_lengths values must be at least 2")
    if args.minimum_kmer_proteins < 1:
        raise ValueError("--minimum_kmer_proteins must be positive")
    if args.minimum_inference_proteins < 2:
        raise ValueError("--minimum_inference_proteins must be at least 2")
    if not 0 < args.discovery_q_threshold <= 1:
        raise ValueError("--discovery_q_threshold must be in (0, 1]")
    if args.discovery_min_abs_log2_odds < 0:
        raise ValueError("--discovery_min_abs_log2_odds must be nonnegative")
    if not 0 < args.replication_q_threshold <= 1:
        raise ValueError("--replication_q_threshold must be in (0, 1]")
    exclusion_radius = (
        args.window_radius
        if args.background_exclusion_radius is None
        else args.background_exclusion_radius
    )
    if exclusion_radius < 0:
        raise ValueError("--background_exclusion_radius must be nonnegative")
    bands = pd.read_csv(args.bands_csv)
    residue = pd.read_csv(
        args.residue_annotations_csv,
        usecols=[
            "split", "protein", "residue_index_0based", "amino_acid", "q8",
        ],
    )
    if args.conditions:
        bands = bands[bands.condition.isin(args.conditions)]
    bands = bands[bands.split.isin(args.splits)].copy()
    residue = residue[residue.split.isin(args.splits)].copy()
    if bands.empty:
        raise ValueError("No bands remain after condition and split filtering")
    windows, coverage, audit = build_windows(
        bands, residue, args.window_radius, args.controls_per_band,
        exclusion_radius, args.random_seed,
    )
    pwm = calculate_pwms(
        windows, args.window_radius, args.pseudocount,
        args.minimum_bands_per_q8_pwm,
    )
    kmer_lengths = tuple(sorted(set(args.kmer_lengths)))
    kmer_tests, discovered_kmers = discover_kmers(
        windows, args.discovery_split, kmer_lengths,
        args.minimum_kmer_proteins, args.minimum_inference_proteins,
        args.minimum_bands_per_q8_pwm, args.discovery_q_threshold,
        args.discovery_min_abs_log2_odds,
    )
    replication_splits = tuple(dict.fromkeys(args.replication_splits))
    kmer_replication = evaluate_discovered_kmers(
        windows, discovered_kmers, replication_splits,
        args.minimum_inference_proteins, args.replication_q_threshold,
    )
    replication_summary = summarize_replication(
        discovered_kmers, kmer_replication, replication_splits
    )
    cross_split_replication = combine_replication_splits(
        discovered_kmers, kmer_replication, replication_splits
    )
    reduced_tests, discovered_reduced = discover_reduced_motifs(
        windows, args.discovery_split, kmer_lengths,
        args.minimum_kmer_proteins, args.minimum_inference_proteins,
        args.minimum_bands_per_q8_pwm, args.discovery_q_threshold,
        args.discovery_min_abs_log2_odds,
    )
    reduced_replication = evaluate_discovered_reduced_motifs(
        windows, discovered_reduced, replication_splits,
        args.minimum_inference_proteins, args.replication_q_threshold,
    )
    reduced_replication_summary = summarize_replication(
        discovered_reduced, reduced_replication, replication_splits
    )
    reduced_cross_split = combine_reduced_replication_splits(
        discovered_reduced, reduced_replication, replication_splits
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows.to_csv(
        output / "matched_apex_windows.csv.gz", index=False, compression="gzip"
    )
    coverage.to_csv(output / "pwm_coverage_summary.csv", index=False)
    pwm.to_csv(output / "pwm_amino_acid_frequencies.csv.gz", index=False, compression="gzip")
    kmer_tests.to_csv(
        output / "kmer_enrichment_train.csv.gz", index=False, compression="gzip"
    )
    discovered_kmers.to_csv(output / "train_discovered_kmers.csv", index=False)
    kmer_replication.to_csv(
        output / "kmer_validation_test_replication.csv", index=False
    )
    replication_summary.to_csv(
        output / "kmer_replication_summary.csv", index=False
    )
    cross_split_replication.to_csv(
        output / "kmer_cross_split_replication.csv", index=False
    )
    reduced_tests.to_csv(
        output / "reduced_motif_enrichment_train.csv.gz",
        index=False, compression="gzip",
    )
    discovered_reduced.to_csv(
        output / "train_discovered_reduced_motifs.csv", index=False
    )
    reduced_replication.to_csv(
        output / "reduced_motif_validation_test_replication.csv", index=False
    )
    reduced_replication_summary.to_csv(
        output / "reduced_motif_replication_summary.csv", index=False
    )
    reduced_cross_split.to_csv(
        output / "reduced_motif_cross_split_replication.csv", index=False
    )
    pd.DataFrame([
        {
            "class_code": code, "class_name": REDUCED_CLASS_LABELS[code],
            "amino_acids": members,
        }
        for code, members in REDUCED_CLASS_MEMBERS.items()
    ]).to_csv(output / "reduced_alphabet_mapping.csv", index=False)
    matrix_count = write_matrix_tables(pwm, output / "pwm_matrices")
    plot_count = 0 if args.skip_plots else write_heatmaps(
        pwm, output / "pwm_heatmaps"
    )
    parameters = {
        "analysis": "apex-centered amino-acid PWM",
        "conditions": sorted(bands.condition.astype(str).unique()),
        "splits": sorted(bands.split.astype(str).unique()),
        "band_types": BAND_LABELS,
        "window_radius": args.window_radius,
        "background": "same-protein exact-apex-Q8 anchors with no band in the exclusion window",
        "controls_per_band": args.controls_per_band,
        "background_exclusion_radius": exclusion_radius,
        "control_weighting": "controls sum to weight one within each matched band",
        "pseudocount_per_amino_acid": args.pseudocount,
        "q8_stratification_minimum_bands": args.minimum_bands_per_q8_pwm,
        "kmer_lengths": list(kmer_lengths),
        "kmer_definition": "exact sequence present anywhere in the apex-centered window",
        "kmer_discovery_split": args.discovery_split,
        "kmer_replication_splits": list(replication_splits),
        "minimum_kmer_band_proteins": args.minimum_kmer_proteins,
        "minimum_inference_proteins": args.minimum_inference_proteins,
        "kmer_inference": "matched band-minus-mean-control effects averaged within protein; two-sided exact protein sign test; BH correction",
        "discovery_q_threshold": args.discovery_q_threshold,
        "discovery_min_abs_log2_odds": args.discovery_min_abs_log2_odds,
        "replication_q_threshold": args.replication_q_threshold,
        "replication_definition": "same direction as train and held-out BH q at or below the replication threshold",
        "reduced_alphabet": {
            REDUCED_CLASS_LABELS[code]: members
            for code, members in REDUCED_CLASS_MEMBERS.items()
        },
        "reduced_motif_definition": "exact class sequence present anywhere in the apex-centered window",
        "reduced_motif_multiple_testing": "BH correction is separate from exact amino-acid k-mers",
        "random_seed": args.random_seed,
        "audit": audit,
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        **audit, "pwm_rows": len(pwm), "pwm_groups": matrix_count,
        "heatmaps": plot_count, "kmer_tests": len(kmer_tests),
        "train_discovered_kmers": len(discovered_kmers),
        "heldout_kmer_tests": len(kmer_replication),
        "heldout_replications": int(kmer_replication.replicated.sum())
        if not kmer_replication.empty else 0,
        "replicated_in_all_heldout_splits": int(
            cross_split_replication.replicated_all_splits.sum()
        ) if not cross_split_replication.empty else 0,
        "reduced_motif_tests": len(reduced_tests),
        "train_discovered_reduced_motifs": len(discovered_reduced),
        "heldout_reduced_motif_tests": len(reduced_replication),
        "heldout_reduced_motif_replications": int(
            reduced_replication.replicated.sum()
        ) if not reduced_replication.empty else 0,
        "reduced_motifs_replicated_in_all_heldout_splits": int(
            reduced_cross_split.replicated_all_splits.sum()
        ) if not reduced_cross_split.empty else 0,
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
