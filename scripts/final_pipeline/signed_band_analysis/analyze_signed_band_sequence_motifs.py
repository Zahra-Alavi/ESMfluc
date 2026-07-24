#!/usr/bin/env python3
"""Analyze local sequence motifs associated with signed contribution bands.

Motifs are discovered on train proteins using same-protein, exact-Q8 unselected
segments as controls.  Definitions are locked before validation and test are
evaluated.  Segment-selection models are fitted on train only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


AA = tuple("ACDEFGHIKLMNPQRSTVWY")
PROPERTY_SETS = {
    "glycine": set("G"),
    "proline": set("P"),
    "charged": set("DEKR"),
    "hydrophobic": set("AVILMFWY"),
    "aromatic": set("FWY"),
    "polar": set("STNQ"),
}

BASE_FEATURES = [
    "mean_neq", "max_neq", "neq_peak_fraction", "neq_peak_excess",
    "mean_rsa", "max_rsa", "log_segment_length", "normalized_midpoint",
    "mean_torsion_change", "max_torsion_change", "q3_boundary_fraction",
    "mean_distance_to_q3_boundary", "structured_linker_fraction",
    "mean_disorder", "max_disorder",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--candidate_segments_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--window_radii", nargs="*", type=int, default=(3, 5, 10))
    parser.add_argument("--kmer_lengths", nargs="*", type=int, default=(2, 3, 4, 5))
    parser.add_argument("--max_controls_per_case", type=int, default=5)
    parser.add_argument("--minimum_discovery_proteins", type=int, default=10)
    parser.add_argument("--minimum_kmer_proteins", type=int, default=5)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--lock_q_threshold", type=float, default=0.05)
    parser.add_argument("--lock_min_abs_log2_odds", type=float, default=1.0)
    parser.add_argument("--max_locked_features", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--max_iter", type=int, default=500)
    return parser.parse_args()


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


def fixed_window(sequence: str, anchor: int, radius: int) -> str:
    return "".join(
        sequence[index] if 0 <= index < len(sequence) else "-"
        for index in range(anchor - radius, anchor + radius + 1)
    )


def audit_candidate_sequences(candidates: pd.DataFrame, residue: pd.DataFrame) -> dict:
    required = {"split", "protein", "residue_index_0based", "amino_acid"}
    missing = required - set(residue)
    if missing:
        raise ValueError(f"Residue annotations lack {sorted(missing)}")
    sequences = {}
    for key, group in residue.groupby(["split", "protein"], sort=False):
        group = group.sort_values("residue_index_0based")
        indices = group.residue_index_0based.to_numpy(int)
        if not np.array_equal(indices, np.arange(len(group))):
            raise ValueError(f"{key}: residue coordinates are not contiguous")
        sequences[(str(key[0]), str(key[1]))] = "".join(group.amino_acid.astype(str))
    unique_segments = candidates.drop_duplicates([
        "split", "protein", "start_index_0based",
        "end_index_0based_exclusive", "segment_sequence",
    ])
    missing_proteins = mismatches = 0
    for row in unique_segments.itertuples(index=False):
        sequence = sequences.get((str(row.split), str(row.protein)))
        if sequence is None:
            missing_proteins += 1
            continue
        observed = sequence[
            int(row.start_index_0based):int(row.end_index_0based_exclusive)
        ]
        if observed != str(row.segment_sequence):
            mismatches += 1
    if missing_proteins or mismatches:
        raise ValueError(
            f"Candidate sequence audit failed: missing proteins={missing_proteins}, "
            f"segment mismatches={mismatches}"
        )
    return {
        "unique_segments_checked": len(unique_segments),
        "missing_proteins": missing_proteins, "segment_mismatches": mismatches,
    }


def centered_interval(sequence: str, anchor: int, width: int) -> str:
    left = anchor - (width - 1) // 2
    return "".join(
        sequence[index] if 0 <= index < len(sequence) else "-"
        for index in range(left, left + width)
    )


def build_matched_windows(
    bands: pd.DataFrame, candidates: pd.DataFrame, radii: list[int],
    max_controls: int, random_seed: int,
) -> tuple[pd.DataFrame, dict]:
    candidate_lookup = {
        key: group.copy() for key, group in candidates.groupby(
            ["condition", "split", "protein"], sort=False
        )
    }
    rng = np.random.default_rng(random_seed)
    rows = []
    skipped = Counter()
    for band in bands.itertuples(index=False):
        context = (str(band.condition), str(band.split), str(band.protein))
        segments = candidate_lookup.get(context)
        if segments is None:
            skipped["missing_candidate_context"] += 1
            continue
        apex = int(band.apex_index_0based)
        containing = segments[
            (segments.start_index_0based <= apex)
            & (segments.end_index_0based_exclusive > apex)
        ]
        if len(containing) != 1:
            skipped["nonunique_apex_segment"] += 1
            continue
        case = containing.iloc[0]
        if bool(case.selected_both_signs):
            skipped["dual_sign_segment"] += 1
            continue
        controls = segments[
            segments.q8.astype(str).eq(str(case.q8)) & segments.clean_control.astype(bool)
        ]
        if controls.empty:
            skipped["no_exact_q8_control"] += 1
            continue
        if max_controls and len(controls) > max_controls:
            controls = controls.iloc[rng.choice(len(controls), max_controls, replace=False)]
        case_start = int(case.start_index_0based)
        case_length = int(case.segment_length)
        relative_anchor = (apex - case_start) / max(1, case_length - 1)
        common = {
            "condition": context[0], "split": context[1], "protein": context[2],
            "band_id": str(band.band_id), "matched_set_id": str(band.band_id),
            "sign": int(band.sign), "source_q8": str(case.q8),
            "band_width": int(band.band_width),
            "case_apex_index_0based": apex,
        }

        def make_row(segment: pd.Series, selected: int, anchor: int, control_number: int) -> dict:
            sequence = str(segment.segment_sequence)
            local_anchor = anchor - int(segment.start_index_0based)
            row = {
                **common, "selected": selected, "control_number": control_number,
                "segment_id": str(segment.segment_id), "q8": str(segment.q8),
                "segment_start_index_0based": int(segment.start_index_0based),
                "segment_end_index_0based_exclusive": int(segment.end_index_0based_exclusive),
                "segment_length": int(segment.segment_length),
                "segment_sequence": sequence, "anchor_index_0based": anchor,
                "anchor_within_segment_0based": local_anchor,
                "matched_band_interval_sequence": centered_interval(
                    sequence, local_anchor, int(band.band_width)
                ),
            }
            for radius in radii:
                row[f"window_pm{radius}"] = fixed_window(sequence, local_anchor, radius)
            return row

        rows.append(make_row(case, 1, apex, 0))
        for control_number, (_, control) in enumerate(controls.iterrows(), 1):
            control_length = int(control.segment_length)
            local_anchor = int(round(relative_anchor * max(0, control_length - 1)))
            anchor = int(control.start_index_0based) + local_anchor
            rows.append(make_row(control, 0, anchor, control_number))
    windows = pd.DataFrame(rows)
    audit = {
        "matched_band_sets": int(windows.loc[windows.selected.eq(1), "matched_set_id"].nunique()) if not windows.empty else 0,
        "rows": len(windows), "skipped_bands": dict(skipped),
    }
    return windows, audit


def matched_design(data: pd.DataFrame) -> dict | None:
    case_rows = data.index[data.selected.eq(1)].to_numpy(int)
    control_rows = data.index[data.selected.eq(0)].to_numpy(int)
    if not len(case_rows) or not len(control_rows):
        return None
    case_sets = data.loc[case_rows, "matched_set_id"].astype(str)
    if case_sets.duplicated().any():
        raise ValueError("Matched motif sets must contain exactly one case row")
    set_number = {value: number for number, value in enumerate(case_sets)}
    control_set = data.loc[control_rows, "matched_set_id"].astype(str).map(set_number)
    keep = control_set.notna().to_numpy()
    control_rows = control_rows[keep]
    control_set = control_set[keep].to_numpy(int)
    sets_with_controls = np.unique(control_set)
    case_keep = np.isin(np.arange(len(case_rows)), sets_with_controls)
    case_rows = case_rows[case_keep]
    if not len(case_rows):
        return None
    old_to_new = np.full(len(set_number), -1, dtype=int)
    old_to_new[sets_with_controls] = np.arange(len(sets_with_controls))
    control_set = old_to_new[control_set]
    case_proteins = data.loc[case_rows, "protein"].astype(str).to_numpy()
    protein_labels, protein_set = np.unique(case_proteins, return_inverse=True)
    return {
        "case_rows": case_rows, "control_rows": control_rows,
        "control_set": control_set,
        "control_count": np.bincount(control_set, minlength=len(case_rows)),
        "protein_set": protein_set, "n_proteins": len(protein_labels),
    }


def matched_binary_association(
    data: pd.DataFrame, values: pd.Series, minimum_proteins: int,
    design: dict | None = None,
) -> dict:
    if design is None:
        data = data.reset_index(drop=True)
        values = values.reset_index(drop=True)
        design = matched_design(data)
    if design is None:
        return {
            "n_proteins": 0, "n_matched_sets": 0, "case_frequency": np.nan,
            "control_frequency": np.nan, "conditional_odds_ratio": np.nan,
            "log2_odds_ratio": np.nan, "protein_mean_difference": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
            "inference_eligible": False, "protein_sign_p_two_sided": np.nan,
        }
    value_array = values.astype(float).to_numpy()
    case = value_array[design["case_rows"]]
    control = np.bincount(
        design["control_set"], weights=value_array[design["control_rows"]],
        minlength=len(case),
    ) / design["control_count"]
    set_difference = case - control
    protein_count = np.bincount(
        design["protein_set"], minlength=design["n_proteins"]
    )
    differences = np.bincount(
        design["protein_set"], weights=set_difference,
        minlength=design["n_proteins"],
    ) / protein_count
    n = design["n_proteins"]
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else np.nan
    eligible = n >= minimum_proteins
    half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if eligible else np.nan
    nonzero = differences[differences != 0]
    p = (
        float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
        if eligible and len(nonzero) else (1.0 if eligible else np.nan)
    )
    case_success = float(case.sum())
    control_success = float(control.sum())
    total = len(case)
    odds_ratio = ((case_success + 0.5) / (total - case_success + 0.5)) / (
        (control_success + 0.5) / (total - control_success + 0.5)
    )
    return {
        "n_proteins": n, "n_matched_sets": total,
        "case_frequency": float(case.mean()),
        "control_frequency": float(control.mean()),
        "conditional_odds_ratio": float(odds_ratio),
        "log2_odds_ratio": float(np.log2(odds_ratio)),
        "protein_mean_difference": mean, "ci95_low": mean - half,
        "ci95_high": mean + half, "inference_eligible": eligible,
        "protein_sign_p_two_sided": p,
    }


def position_specific_enrichment(
    train: pd.DataFrame, radii: list[int], minimum_proteins: int,
) -> pd.DataFrame:
    rows = []
    for (sign, q8), group in train.groupby(["sign", "source_q8"], sort=False):
        group = group.reset_index(drop=True)
        design = matched_design(group)
        for radius in radii:
            column = f"window_pm{radius}"
            for position, offset in enumerate(range(-radius, radius + 1)):
                characters = group[column].str[position]
                for amino_acid in AA:
                    result = matched_binary_association(
                        group, characters.eq(amino_acid), minimum_proteins, design
                    )
                    rows.append({
                        "sign": sign, "q8": q8, "window_radius": radius,
                        "offset": offset, "symbol_type": "amino_acid",
                        "symbol": amino_acid, **result,
                    })
                for property_name, members in PROPERTY_SETS.items():
                    result = matched_binary_association(
                        group, characters.isin(members), minimum_proteins, design
                    )
                    rows.append({
                        "sign": sign, "q8": q8, "window_radius": radius,
                        "offset": offset, "symbol_type": "property",
                        "symbol": property_name, **result,
                    })
    output = pd.DataFrame(rows)
    if not output.empty:
        output["protein_sign_q_bh"] = bh(output.protein_sign_p_two_sided)
    return output


def unique_kmers(sequence: str, k: int) -> set[str]:
    clean = sequence.replace("-", "")
    return {clean[index:index + k] for index in range(len(clean) - k + 1)}


def kmer_enrichment(
    train: pd.DataFrame, lengths: list[int], minimum_recurrence: int,
    minimum_inference: int,
) -> pd.DataFrame:
    rows = []
    sequence_units = {
        "q8_segment": "segment_sequence",
        "band_width_matched": "matched_band_interval_sequence",
    }
    for (sign, q8), group in train.groupby(["sign", "source_q8"], sort=False):
        group = group.reset_index(drop=True)
        design = matched_design(group)
        for unit, column in sequence_units.items():
            for k in lengths:
                protein_occurrence: defaultdict[str, set[str]] = defaultdict(set)
                for row in group.itertuples(index=False):
                    if int(row.selected) == 1:
                        for kmer in unique_kmers(str(getattr(row, column)), k):
                            protein_occurrence[kmer].add(str(row.protein))
                candidates = [
                    kmer for kmer, proteins in protein_occurrence.items()
                    if len(proteins) >= minimum_recurrence
                ]
                candidate_set = set(candidates)
                occurrence_rows: defaultdict[str, list[int]] = defaultdict(list)
                for row_number, sequence in enumerate(group[column].astype(str)):
                    for kmer in unique_kmers(sequence, k) & candidate_set:
                        occurrence_rows[kmer].append(row_number)
                for kmer in candidates:
                    values = np.zeros(len(group), dtype=bool)
                    values[occurrence_rows[kmer]] = True
                    result = matched_binary_association(
                        group, pd.Series(values),
                        minimum_inference, design,
                    )
                    rows.append({
                        "sign": sign, "q8": q8, "sequence_unit": unit,
                        "k": k, "kmer": kmer,
                        "case_protein_recurrence": len(protein_occurrence[kmer]),
                        **result,
                    })
    output = pd.DataFrame(rows)
    if not output.empty:
        output["protein_sign_q_bh"] = bh(output.protein_sign_p_two_sided)
    return output


def lock_definitions(position: pd.DataFrame, kmers: pd.DataFrame, args: argparse.Namespace) -> dict:
    if position.empty:
        position_locked = position.copy()
    else:
        position_locked = position[
            position.symbol_type.eq("amino_acid")
            & position.inference_eligible.astype(bool)
            & position.protein_sign_q_bh.le(args.lock_q_threshold)
            & position.log2_odds_ratio.abs().ge(args.lock_min_abs_log2_odds)
        ].copy()
    if kmers.empty:
        kmer_locked = kmers.copy()
    else:
        kmer_locked = kmers[
            kmers.inference_eligible.astype(bool)
            & kmers.protein_sign_q_bh.le(args.lock_q_threshold)
            & kmers.log2_odds_ratio.abs().ge(args.lock_min_abs_log2_odds)
            & kmers.case_protein_recurrence.ge(args.minimum_kmer_proteins)
        ].copy()
    if not position_locked.empty:
        position_locked = position_locked.sort_values(
            ["protein_sign_q_bh", "log2_odds_ratio"], ascending=[True, False]
        ).head(args.max_locked_features)
    if not kmer_locked.empty:
        kmer_locked = kmer_locked.sort_values(
            ["protein_sign_q_bh", "log2_odds_ratio"], ascending=[True, False]
        ).head(args.max_locked_features)
    position_definitions = []
    for number, row in enumerate(position_locked.itertuples(index=False), 1):
        position_definitions.append({
            "motif_id": f"pos_{number:04d}", "type": "position_specific",
            "sign": int(row.sign), "q8": str(row.q8),
            "window_radius": int(row.window_radius), "offset": int(row.offset),
            "amino_acid": str(row.symbol), "train_log2_odds_ratio": float(row.log2_odds_ratio),
            "train_q_value": float(row.protein_sign_q_bh),
        })
    kmer_definitions = []
    for number, row in enumerate(kmer_locked.itertuples(index=False), 1):
        kmer_definitions.append({
            "motif_id": f"kmer_{number:04d}", "type": "kmer",
            "sign": int(row.sign), "q8": str(row.q8),
            "sequence_unit": str(row.sequence_unit), "k": int(row.k),
            "kmer": str(row.kmer), "train_log2_odds_ratio": float(row.log2_odds_ratio),
            "train_q_value": float(row.protein_sign_q_bh),
        })
    pssm_definitions = []
    for (sign, q8, radius), group in position_locked.groupby(
        ["sign", "q8", "window_radius"], sort=False
    ):
        pssm_definitions.append({
            "motif_id": f"pssm_{len(pssm_definitions) + 1:04d}", "type": "pssm",
            "sign": int(sign), "q8": str(q8), "window_radius": int(radius),
            "weights": [
                {
                    "offset": int(row.offset), "amino_acid": str(row.symbol),
                    "weight": float(row.log2_odds_ratio),
                }
                for row in group.itertuples(index=False)
            ],
        })
    return {
        "discovery_split": "train", "validation_role": "evaluation only",
        "test_role": "final held-out evaluation",
        "discovery_aggregation": "pooled conditions with protein-level matched effects",
        "lock_q_threshold": args.lock_q_threshold,
        "lock_min_abs_log2_odds": args.lock_min_abs_log2_odds,
        "position_specific": position_definitions,
        "kmers": kmer_definitions, "pssms": pssm_definitions,
    }


def definition_values(data: pd.DataFrame, definition: dict) -> pd.Series:
    if definition["type"] == "position_specific":
        radius = int(definition["window_radius"])
        position = int(definition["offset"]) + radius
        return data[f"window_pm{radius}"].str[position].eq(definition["amino_acid"])
    column = "segment_sequence" if definition["sequence_unit"] == "q8_segment" else "matched_band_interval_sequence"
    return data[column].str.contains(definition["kmer"], regex=False)


def evaluate_locked(windows: pd.DataFrame, locked: dict, minimum_proteins: int) -> pd.DataFrame:
    rows = []
    definitions = locked["position_specific"] + locked["kmers"]
    for definition in definitions:
        for split in ("validation", "test"):
            group = windows[
                windows.split.eq(split)
                & windows.sign.eq(definition["sign"])
                & windows.source_q8.astype(str).eq(definition["q8"])
            ]
            result = matched_binary_association(
                group, definition_values(group, definition), minimum_proteins
            ) if not group.empty else matched_binary_association(
                group, pd.Series(dtype=bool), minimum_proteins
            )
            rows.append({
                "motif_id": definition["motif_id"], "motif_type": definition["type"],
                "sign": definition["sign"], "q8": definition["q8"],
                "evaluation_split": split, **result,
            })
    output = pd.DataFrame(rows)
    if not output.empty:
        output["protein_sign_q_bh"] = output.groupby(
            ["evaluation_split", "motif_type"], group_keys=False
        ).protein_sign_p_two_sided.transform(bh)
    return output


def model_family_consistency(
    windows: pd.DataFrame, locked: dict, minimum_proteins: int,
) -> pd.DataFrame:
    rows = []
    definitions = locked["position_specific"] + locked["kmers"]
    families = windows.condition.astype(str).str.extract(r"^(esm[23])", expand=False)
    for definition in definitions:
        for split in ("validation", "test"):
            for family in ("esm2", "esm3"):
                group = windows[
                    windows.split.eq(split)
                    & families.eq(family)
                    & windows.sign.eq(definition["sign"])
                    & windows.source_q8.astype(str).eq(definition["q8"])
                ]
                if group.empty:
                    continue
                result = matched_binary_association(
                    group, definition_values(group, definition), minimum_proteins
                )
                rows.append({
                    "motif_id": definition["motif_id"],
                    "motif_type": definition["type"],
                    "sign": definition["sign"], "q8": definition["q8"],
                    "evaluation_split": split, "model_family": family,
                    "train_log2_odds_ratio": definition["train_log2_odds_ratio"],
                    **result,
                })
    output = pd.DataFrame(rows)
    if not output.empty:
        output["direction_agrees_with_train"] = np.sign(output.log2_odds_ratio) == np.sign(
            output.train_log2_odds_ratio
        )
        output["protein_sign_q_bh"] = output.groupby(
            "evaluation_split", group_keys=False
        ).protein_sign_p_two_sided.transform(bh)
    return output


def pssm_score(sequence: str, definition: dict) -> float:
    if not sequence:
        return np.nan
    weights = definition["weights"]
    best = -np.inf
    for anchor in range(len(sequence)):
        score = 0.0
        for item in weights:
            index = anchor + int(item["offset"])
            if 0 <= index < len(sequence) and sequence[index] == item["amino_acid"]:
                score += float(item["weight"])
        best = max(best, score)
    return float(best)


def signed_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for sign, selected_name, opposite_name in (
        (1, "selected_positive", "selected_negative"),
        (-1, "selected_negative", "selected_positive"),
    ):
        cases = candidates[candidates[selected_name].astype(bool) & ~candidates[opposite_name].astype(bool)].copy()
        controls = candidates[candidates.clean_control.astype(bool)].copy()
        cases["selected"] = 1
        controls["selected"] = 0
        data = pd.concat([cases, controls], ignore_index=True)
        data["sign"] = sign
        counts = data.groupby(
            ["condition", "split", "protein", "q8", "selected"]
        ).size().unstack(fill_value=0)
        valid = counts[(counts.get(0, 0) > 0) & (counts.get(1, 0) > 0)].reset_index()[
            ["condition", "split", "protein", "q8"]
        ]
        outputs.append(data.merge(
            valid, on=["condition", "split", "protein", "q8"], how="inner"
        ))
    return pd.concat(outputs, ignore_index=True)


def add_motif_scores(data: pd.DataFrame, locked: dict) -> tuple[pd.DataFrame, list[str]]:
    data = data.copy()
    feature_names = []
    for definition in locked["kmers"]:
        name = definition["motif_id"]
        column = "segment_sequence"
        applies = data.sign.eq(definition["sign"]) & data.q8.astype(str).eq(definition["q8"])
        data[name] = 0.0
        data.loc[applies, name] = data.loc[applies, column].str.contains(
            definition["kmer"], regex=False
        ).astype(float)
        feature_names.append(name)
    for definition in locked["pssms"]:
        name = definition["motif_id"]
        applies = data.sign.eq(definition["sign"]) & data.q8.astype(str).eq(definition["q8"])
        data[name] = 0.0
        data.loc[applies, name] = data.loc[applies, "segment_sequence"].map(
            lambda sequence: pssm_score(str(sequence), definition)
        )
        feature_names.append(name)
    return data, feature_names


def add_weights(data: pd.DataFrame) -> pd.DataFrame:
    sizes = data.groupby(
        ["condition", "split", "protein", "q8", "sign", "selected"]
    ).size().rename("class_size").reset_index()
    data = data.merge(
        sizes, on=["condition", "split", "protein", "q8", "sign", "selected"],
        how="left", validate="many_to_one",
    )
    data["sample_weight"] = 0.5 / data.class_size
    return data


def motif_models(data: pd.DataFrame, motif_features: list[str], args: argparse.Namespace) -> pd.DataFrame:
    if not motif_features:
        return pd.DataFrame(columns=[
            "condition", "sign", "stage", "evaluation_split", "n_segments",
            "n_proteins", "weighted_auroc", "weighted_average_precision",
            "macro_within_protein_q8_auroc", "delta_auroc_from_base",
        ])
    data = add_weights(data)
    stages = {
        "00_biophysical_base": BASE_FEATURES,
        "01_motif_only": motif_features,
        "02_base_plus_locked_motifs": BASE_FEATURES + motif_features,
    }
    rows = []
    for condition in sorted(data.condition.unique()):
        for sign in (-1, 1):
            train = data[
                data.condition.eq(condition) & data.sign.eq(sign) & data.split.eq("train")
            ]
            if train.selected.nunique() < 2:
                continue
            for stage, numeric in stages.items():
                transformer = ColumnTransformer([
                    ("q8", OneHotEncoder(handle_unknown="ignore"), ["q8"]),
                    ("numeric", Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), numeric),
                ])
                model = Pipeline([
                    ("transform", transformer),
                    ("logistic", LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=args.max_iter,
                        random_state=args.random_seed,
                    )),
                ])
                model.fit(
                    train[["q8"] + numeric], train.selected,
                    logistic__sample_weight=train.sample_weight,
                )
                for split in ("validation", "test"):
                    evaluate = data[
                        data.condition.eq(condition) & data.sign.eq(sign) & data.split.eq(split)
                    ]
                    if evaluate.selected.nunique() < 2:
                        continue
                    score = model.predict_proba(evaluate[["q8"] + numeric])[:, 1]
                    scored = evaluate[["protein", "q8", "selected"]].copy()
                    scored["score"] = score
                    stratum_auc = []
                    for _, group in scored.groupby(["protein", "q8"], sort=False):
                        if group.selected.nunique() == 2:
                            stratum_auc.append(roc_auc_score(group.selected, group.score))
                    rows.append({
                        "condition": condition, "sign": sign, "stage": stage,
                        "evaluation_split": split, "n_segments": len(evaluate),
                        "n_proteins": int(evaluate.protein.nunique()),
                        "weighted_auroc": float(roc_auc_score(
                            evaluate.selected, score, sample_weight=evaluate.sample_weight
                        )),
                        "weighted_average_precision": float(average_precision_score(
                            evaluate.selected, score, sample_weight=evaluate.sample_weight
                        )),
                        "macro_within_protein_q8_auroc": float(np.mean(stratum_auc)) if stratum_auc else np.nan,
                    })
    output = pd.DataFrame(rows)
    if not output.empty:
        base = output[output.stage.eq("00_biophysical_base")][
            ["condition", "sign", "evaluation_split", "weighted_auroc"]
        ].rename(columns={"weighted_auroc": "base_auroc"})
        output = output.merge(base, on=["condition", "sign", "evaluation_split"], how="left")
        output["delta_auroc_from_base"] = output.weighted_auroc - output.base_auroc
    return output


def write_logo_tables(position: pd.DataFrame, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    if position.empty:
        return 0
    count = 0
    amino = position[position.symbol_type.eq("amino_acid")]
    for (sign, q8, radius), group in amino.groupby(
        ["sign", "q8", "window_radius"], sort=False
    ):
        table = group.pivot(index="offset", columns="symbol", values="log2_odds_ratio")
        table = table.reindex(columns=AA)
        label = "positive" if int(sign) == 1 else "negative"
        table.to_csv(directory / f"{label}__q8_{q8}__pm{radius}_log2_odds.csv")
        count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.minimum_inference_proteins < 2 or args.minimum_discovery_proteins < 2:
        raise ValueError("Protein minimums must be at least 2")
    if args.max_controls_per_case < 1:
        raise ValueError("--max_controls_per_case must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bands = pd.read_csv(args.bands_csv)
    candidates = pd.read_csv(
        args.candidate_segments_csv, keep_default_na=False, na_values=[""]
    )
    residue = pd.read_csv(args.residue_annotations_csv)
    if args.conditions:
        bands = bands[bands.condition.isin(args.conditions)]
        candidates = candidates[candidates.condition.isin(args.conditions)]
    sequence_audit = audit_candidate_sequences(candidates, residue)
    windows, matching_audit = build_matched_windows(
        bands, candidates, sorted(set(args.window_radii)),
        args.max_controls_per_case, args.random_seed,
    )
    if windows.empty:
        raise ValueError("No bands had a same-protein exact-Q8 unselected control")
    windows.to_csv(output / "matched_sequence_windows.csv.gz", index=False, compression="gzip")
    train = windows[windows.split.eq("train")].copy()
    position = position_specific_enrichment(
        train, sorted(set(args.window_radii)), args.minimum_discovery_proteins
    )
    kmers = kmer_enrichment(
        train, sorted(set(args.kmer_lengths)), args.minimum_kmer_proteins,
        args.minimum_discovery_proteins,
    )
    discovery_p = pd.concat([
        position.protein_sign_p_two_sided if not position.empty else pd.Series(dtype=float),
        kmers.protein_sign_p_two_sided if not kmers.empty else pd.Series(dtype=float),
    ], ignore_index=True)
    discovery_q = bh(discovery_p)
    if not position.empty:
        position["protein_sign_q_bh"] = discovery_q.iloc[:len(position)].to_numpy()
    if not kmers.empty:
        kmers["protein_sign_q_bh"] = discovery_q.iloc[len(position):].to_numpy()
    position.to_csv(output / "position_specific_enrichment.csv", index=False)
    kmers.to_csv(output / "kmer_enrichment_train.csv", index=False)
    locked = lock_definitions(position, kmers, args)
    (output / "locked_motif_definitions.json").write_text(json.dumps(locked, indent=2) + "\n")
    heldout = evaluate_locked(windows, locked, args.minimum_inference_proteins)
    heldout.to_csv(output / "motif_validation_test_results.csv", index=False)
    family_consistency = model_family_consistency(
        windows, locked, args.minimum_inference_proteins
    )
    family_consistency.to_csv(output / "motif_model_family_consistency.csv", index=False)
    signed_candidates = signed_candidate_rows(candidates)
    scored_candidates, motif_features = add_motif_scores(signed_candidates, locked)
    performance = motif_models(scored_candidates, motif_features, args)
    performance.to_csv(output / "motif_incremental_model_performance.csv", index=False)
    logo_tables = write_logo_tables(position, output / "sequence_logos")
    parameters = {
        "primary_background": "same-protein exact-Q8 unselected segments outside all bands",
        "case_anchor": "band apex",
        "control_anchor": "same relative position inside the control Q8 segment",
        "control_sampling": "random without matching Neq, RSA, length, or position",
        "discovery_split": "train",
        "discovery_aggregation": "pooled conditions with protein-level matched effects",
        "validation_and_test": "locked motif evaluation without rediscovery",
        "model_family_consistency": "reported separately for ESM2 and ESM3",
        "window_radii": sorted(set(args.window_radii)),
        "kmer_lengths": sorted(set(args.kmer_lengths)),
        "minimum_discovery_proteins": args.minimum_discovery_proteins,
        "minimum_kmer_proteins": args.minimum_kmer_proteins,
        "minimum_inference_proteins": args.minimum_inference_proteins,
        "inference": "two-sided exact protein sign randomization with global BH correction",
        "lock_q_threshold": args.lock_q_threshold,
        "lock_min_abs_log2_odds": args.lock_min_abs_log2_odds,
        "random_seed": args.random_seed,
        "matching_audit": matching_audit,
        "candidate_sequence_audit": sequence_audit,
        "residue_annotation_source": str(Path(args.residue_annotations_csv).resolve()),
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "matching": matching_audit,
        "position_tests": len(position), "kmer_tests": len(kmers),
        "locked_position_features": len(locked["position_specific"]),
        "locked_kmers": len(locked["kmers"]),
        "heldout_tests": len(heldout),
        "model_family_consistency_rows": len(family_consistency),
        "model_rows": len(performance),
        "logo_tables": logo_tables, "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
