#!/usr/bin/env python3
"""Phase 3A: analyze why one Q8 segment is selected as a signed band.

The analysis uses complete contiguous Q8 segments as candidate objects.  A case
contains an apex of the requested sign.  A control is a segment in the same
protein with the identical Q8 label that neither contains an apex nor overlaps
any signed-band interval.  Descriptive effects therefore compare like with
like inside proteins without matching away candidate explanatory variables.

Cumulative logistic models are trained on the publication train split and
evaluated without refitting on validation and test.  Stratum weights give the
selected and control classes equal total weight within protein/Q8 strata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


AA = tuple("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWY")
CHARGED = set("DEKR")
AROMATIC = set("FWY")

FEATURE_GROUPS = {
    "neq": ["mean_neq", "max_neq", "neq_peak_fraction", "neq_peak_excess"],
    "exposure_length_position": [
        "mean_rsa", "max_rsa", "log_segment_length", "normalized_midpoint"
    ],
    "geometry_boundaries": [
        "mean_torsion_change", "max_torsion_change",
        "q3_boundary_fraction", "mean_distance_to_q3_boundary",
        "structured_linker_fraction",
    ],
    "disorder": ["mean_disorder", "max_disorder"],
    "sequence": [
        "fraction_glycine", "fraction_proline", "fraction_hydrophobic",
        "fraction_charged", "fraction_aromatic", "sequence_entropy",
    ],
}

STAGES = {
    "00_q8_only": [],
    "01_add_neq": FEATURE_GROUPS["neq"],
    "02_add_exposure_length_position": (
        FEATURE_GROUPS["neq"] + FEATURE_GROUPS["exposure_length_position"]
    ),
    "03_add_geometry_boundaries": (
        FEATURE_GROUPS["neq"] + FEATURE_GROUPS["exposure_length_position"]
        + FEATURE_GROUPS["geometry_boundaries"]
    ),
    "04_add_disorder": (
        FEATURE_GROUPS["neq"] + FEATURE_GROUPS["exposure_length_position"]
        + FEATURE_GROUPS["geometry_boundaries"] + FEATURE_GROUPS["disorder"]
    ),
    "05_add_sequence": sum(FEATURE_GROUPS.values(), []),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--protein_summary_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument(
        "--minimum_inference_proteins", type=int, default=10,
        help="Minimum independent proteins required for CI and Wilcoxon inference.",
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    return parser.parse_args()


def bh(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return out
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[valid.index.to_numpy()[order]] = np.minimum(adjusted, 1.0)
    return out


def safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else np.nan


def safe_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else np.nan


def entropy(sequence: str) -> float:
    if not sequence:
        return np.nan
    counts = np.array([sequence.count(a) for a in AA], dtype=float)
    probabilities = counts[counts > 0] / len(sequence)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def q8_runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    if len(labels) == 0:
        return []
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(labels)]
    return [(int(a), int(b), str(labels[a])) for a, b in zip(starts, ends)]


def build_base_segments(residue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [
        "neq", "rsa", "torsion_change_from_previous", "q3_boundary",
        "distance_to_q3_boundary", "structured_linker_loop", "disorder",
        "neq_peak",
    ]
    for (split, protein), group in residue.groupby(["split", "protein"], sort=False):
        group = group.sort_values("residue_index_0based").reset_index(drop=True)
        length = len(group)
        if not np.array_equal(group.residue_index_0based.to_numpy(), np.arange(length)):
            raise ValueError(f"{split}/{protein}: non-contiguous residue coordinates")
        arrays = {
            name: pd.to_numeric(group[name], errors="coerce").to_numpy(float)
            for name in numeric
        }
        sequence = "".join(group.amino_acid.astype(str))
        for number, (start, end, q8) in enumerate(q8_runs(group.q8.astype(str).to_numpy()), 1):
            seq = sequence[start:end]
            neq_values = arrays["neq"][start:end]
            rows.append({
                "split": str(split), "protein": str(protein),
                "protein_length": length, "segment_number": number,
                "q8": q8, "start_index_0based": start,
                "end_index_0based_exclusive": end,
                "end_index_0based_inclusive": end - 1,
                "segment_length": end - start, "segment_sequence": seq,
                "mean_neq": safe_mean(neq_values), "max_neq": safe_max(neq_values),
                "neq_peak_fraction": safe_mean(arrays["neq_peak"][start:end]),
                "neq_peak_excess": safe_max(neq_values) - safe_mean(neq_values),
                "mean_rsa": safe_mean(arrays["rsa"][start:end]),
                "max_rsa": safe_max(arrays["rsa"][start:end]),
                "log_segment_length": float(np.log1p(end - start)),
                "normalized_midpoint": ((start + end - 1) / 2) / max(1, length - 1),
                "mean_torsion_change": safe_mean(arrays["torsion_change_from_previous"][start:end]),
                "max_torsion_change": safe_max(arrays["torsion_change_from_previous"][start:end]),
                "q3_boundary_fraction": safe_mean(arrays["q3_boundary"][start:end]),
                "mean_distance_to_q3_boundary": safe_mean(arrays["distance_to_q3_boundary"][start:end]),
                "structured_linker_fraction": safe_mean(arrays["structured_linker_loop"][start:end]),
                "mean_disorder": safe_mean(arrays["disorder"][start:end]),
                "max_disorder": safe_max(arrays["disorder"][start:end]),
                "fraction_glycine": seq.count("G") / len(seq),
                "fraction_proline": seq.count("P") / len(seq),
                "fraction_hydrophobic": sum(a in HYDROPHOBIC for a in seq) / len(seq),
                "fraction_charged": sum(a in CHARGED for a in seq) / len(seq),
                "fraction_aromatic": sum(a in AROMATIC for a in seq) / len(seq),
                "sequence_entropy": entropy(seq),
            })
    return pd.DataFrame(rows)


def attach_selection(
    base: pd.DataFrame, bands: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    segment_lookup = {
        key: group.copy() for key, group in base.groupby(["split", "protein"], sort=False)
    }
    band_lookup = {
        key: group for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    rows = []
    for context in summary.itertuples(index=False):
        condition, split, protein = map(str, (context.condition, context.split, context.protein))
        segments = segment_lookup[(split, protein)].copy()
        eligible_start = int(context.eligible_start_index_0based)
        eligible_end = int(context.eligible_end_index_0based_exclusive)
        segments = segments[
            (segments.end_index_0based_exclusive > eligible_start)
            & (segments.start_index_0based < eligible_end)
        ].copy()
        context_bands = band_lookup.get((condition, split, protein))
        plus_apices = np.array([], dtype=int)
        minus_apices = np.array([], dtype=int)
        intervals: list[tuple[int, int]] = []
        if context_bands is not None:
            plus_apices = context_bands.loc[context_bands.sign == 1, "apex_index_0based"].to_numpy(int)
            minus_apices = context_bands.loc[context_bands.sign == -1, "apex_index_0based"].to_numpy(int)
            intervals = list(zip(
                context_bands.start_index_0based.astype(int),
                context_bands.end_index_0based_inclusive.astype(int),
            ))
        for row in segments.to_dict("records"):
            start, end = int(row["start_index_0based"]), int(row["end_index_0based_exclusive"])
            plus_count = int(np.sum((plus_apices >= start) & (plus_apices < end)))
            minus_count = int(np.sum((minus_apices >= start) & (minus_apices < end)))
            overlaps = any(left < end and right >= start for left, right in intervals)
            row.update({
                "condition": condition,
                "eligible_start_index_0based": eligible_start,
                "eligible_end_index_0based_exclusive": eligible_end,
                "positive_apex_count": plus_count,
                "negative_apex_count": minus_count,
                "selected_positive": plus_count > 0,
                "selected_negative": minus_count > 0,
                "selected_both_signs": plus_count > 0 and minus_count > 0,
                "overlaps_any_band": overlaps,
                "clean_control": not overlaps,
                "segment_id": f"{condition}__{split}__{protein}__q8seg{int(row['segment_number']):04d}",
            })
            rows.append(row)
    return pd.DataFrame(rows)


def analysis_rows(candidates: pd.DataFrame, sign: int) -> pd.DataFrame:
    selected = "selected_positive" if sign == 1 else "selected_negative"
    opposite = "selected_negative" if sign == 1 else "selected_positive"
    cases = candidates[candidates[selected] & ~candidates[opposite]].copy()
    controls = candidates[candidates.clean_control].copy()
    cases["selected"] = 1
    controls["selected"] = 0
    data = pd.concat([cases, controls], ignore_index=True)
    data["sign"] = sign
    # Only strata that permit an actual within-protein, same-Q8 contrast.
    counts = data.groupby(["condition", "split", "protein", "q8", "sign", "selected"]).size().unstack(fill_value=0)
    valid = counts[(counts.get(0, 0) > 0) & (counts.get(1, 0) > 0)].reset_index()[
        ["condition", "split", "protein", "q8", "sign"]
    ]
    return data.merge(valid, on=["condition", "split", "protein", "q8", "sign"], how="inner")


def matched_effects(
    data: pd.DataFrame, minimum_inference_proteins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = sum(FEATURE_GROUPS.values(), [])
    keys = ["condition", "split", "protein", "sign", "q8"]
    rows = []
    for key, group in data.groupby(keys, sort=False):
        cases, controls = group[group.selected == 1], group[group.selected == 0]
        for feature in features:
            case = pd.to_numeric(cases[feature], errors="coerce")
            control = pd.to_numeric(controls[feature], errors="coerce")
            if case.notna().any() and control.notna().any():
                rows.append({
                    **dict(zip(keys, key)), "feature": feature,
                    "feature_group": next(k for k, v in FEATURE_GROUPS.items() if feature in v),
                    "n_selected_segments": len(cases), "n_control_segments": len(controls),
                    "selected_mean": float(case.mean()), "control_mean": float(control.mean()),
                    "selected_minus_control": float(case.mean() - control.mean()),
                })
    per_protein = pd.DataFrame(rows)
    summary_rows = []
    group_keys = ["condition", "split", "sign", "q8", "feature", "feature_group"]
    for key, group in per_protein.groupby(group_keys, sort=False):
        values = group.selected_minus_control.to_numpy(float)
        n = len(values)
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1)) if n > 1 else np.nan
        inference_eligible = n >= minimum_inference_proteins
        half = (
            float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
            if inference_eligible else np.nan
        )
        if inference_eligible:
            try:
                p = float(wilcoxon(
                    values, zero_method="zsplit", alternative="two-sided",
                    method="approx",
                ).pvalue)
            except ValueError:
                p = 1.0
        else:
            p = np.nan
        summary_rows.append({
            **dict(zip(group_keys, key)), "n_proteins": n,
            "n_selected_segments": int(group.n_selected_segments.sum()),
            "n_control_segments": int(group.n_control_segments.sum()),
            "selected_macro_protein_mean": float(group.selected_mean.mean()),
            "control_macro_protein_mean": float(group.control_mean.mean()),
            "selected_minus_control_macro_mean": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "inference_eligible": inference_eligible,
            "wilcoxon_p_two_sided": p,
        })
    summary = pd.DataFrame(summary_rows)
    summary["wilcoxon_q_bh"] = bh(summary.wilcoxon_p_two_sided)
    return per_protein, summary


def add_stratum_weights(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    keys = ["condition", "split", "protein", "q8", "sign", "selected"]
    sizes = data.groupby(keys).size().rename("class_size").reset_index()
    data = data.merge(sizes, on=keys, how="left", validate="many_to_one")
    data["sample_weight"] = 0.5 / data.class_size
    return data


def matched_concordance(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, int]:
    work = frame[["protein", "q8", "selected", "sample_weight"]].copy()
    work["score"] = scores
    protein_values = []
    n_strata = 0
    for _key, group in work.groupby(["protein", "q8"], sort=False):
        if group.selected.nunique() < 2:
            continue
        n_strata += 1
        protein_values.append(float(roc_auc_score(group.selected, group.score)))
    return (float(np.mean(protein_values)) if protein_values else np.nan, n_strata)


def sequential_models(data: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_stratum_weights(data)
    performance, coefficients = [], []
    for condition in sorted(data.condition.unique()):
        for sign in (-1, 1):
            train = data[(data.condition == condition) & (data.sign == sign) & (data.split == "train")]
            if train.selected.nunique() < 2:
                continue
            for stage, features in STAGES.items():
                numeric_transformer = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ])
                transformer = ColumnTransformer([
                    ("q8", OneHotEncoder(handle_unknown="ignore"), ["q8"]),
                    ("numeric", numeric_transformer, features),
                ])
                model = Pipeline([
                    ("transform", transformer),
                    ("logistic", LogisticRegression(
                        C=1.0, max_iter=args.max_iter, solver="lbfgs",
                        random_state=args.random_seed,
                    )),
                ])
                model.fit(train[["q8"] + features], train.selected, logistic__sample_weight=train.sample_weight)
                names = model.named_steps["transform"].get_feature_names_out()
                coefs = model.named_steps["logistic"].coef_[0]
                for name, value in zip(names, coefs):
                    coefficients.append({
                        "condition": condition, "sign": sign, "stage": stage,
                        "feature": str(name), "standardized_coefficient": float(value),
                    })
                for split in ("validation", "test"):
                    evaluate = data[(data.condition == condition) & (data.sign == sign) & (data.split == split)]
                    if evaluate.selected.nunique() < 2:
                        continue
                    score = model.predict_proba(evaluate[["q8"] + features])[:, 1]
                    concordance, n_strata = matched_concordance(evaluate, score)
                    performance.append({
                        "condition": condition, "sign": sign, "stage": stage,
                        "evaluation_split": split,
                        "n_segments": len(evaluate),
                        "n_selected_segments": int(evaluate.selected.sum()),
                        "n_proteins": int(evaluate.protein.nunique()),
                        "weighted_auroc": float(roc_auc_score(
                            evaluate.selected, score, sample_weight=evaluate.sample_weight
                        )),
                        "weighted_average_precision": float(average_precision_score(
                            evaluate.selected, score, sample_weight=evaluate.sample_weight
                        )),
                        "macro_within_protein_q8_concordance": concordance,
                        "n_matched_protein_q8_strata": n_strata,
                    })
    performance = pd.DataFrame(performance)
    if not performance.empty:
        performance["delta_auroc_from_previous_stage"] = performance.groupby(
            ["condition", "sign", "evaluation_split"], sort=False
        ).weighted_auroc.diff()
        performance["delta_concordance_from_previous_stage"] = performance.groupby(
            ["condition", "sign", "evaluation_split"], sort=False
        ).macro_within_protein_q8_concordance.diff()
    return performance, pd.DataFrame(coefficients)


def main() -> None:
    args = parse_args()
    bands = pd.read_csv(args.bands_csv)
    protein_summary = pd.read_csv(args.protein_summary_csv)
    residue = pd.read_csv(args.residue_annotations_csv)
    if args.conditions:
        bands = bands[bands.condition.isin(args.conditions)]
        protein_summary = protein_summary[protein_summary.condition.isin(args.conditions)]
    required = {
        "split", "protein", "residue_index_0based", "amino_acid", "q8",
        *sum(FEATURE_GROUPS.values(), []),
    }
    # Derived feature names are not direct residue columns.
    direct = {
        "neq", "rsa", "torsion_change_from_previous", "q3_boundary",
        "distance_to_q3_boundary", "structured_linker_loop", "disorder", "neq_peak",
    }
    missing = ({"split", "protein", "residue_index_0based", "amino_acid", "q8"} | direct) - set(residue)
    if missing:
        raise ValueError(f"Residue annotations lack {sorted(missing)}")
    base = build_base_segments(residue)
    candidates = attach_selection(base, bands, protein_summary)
    signed_data = pd.concat(
        [analysis_rows(candidates, -1), analysis_rows(candidates, 1)],
        ignore_index=True,
    )
    if args.minimum_inference_proteins < 2:
        raise ValueError("--minimum_inference_proteins must be at least 2")
    per_protein, feature_summary = matched_effects(
        signed_data, args.minimum_inference_proteins
    )
    performance, coefficients = sequential_models(signed_data, args)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / "q8_segment_candidates.csv.gz", index=False, compression="gzip")
    per_protein.to_csv(output / "matched_feature_effects_by_protein.csv.gz", index=False, compression="gzip")
    feature_summary.to_csv(output / "matched_feature_effect_summary.csv", index=False)
    performance.to_csv(output / "sequential_model_performance.csv", index=False)
    coefficients.to_csv(output / "sequential_model_coefficients.csv", index=False)
    parameters = {
        "phase": "3A", "candidate_object": "complete contiguous Q8 segment",
        "case": "segment contains exactly one sign class of band apex",
        "control": "same-protein, identical-Q8 segment outside every band interval",
        "dual_sign_segments": "excluded from both sign-specific case sets",
        "feature_groups": FEATURE_GROUPS, "stages": STAGES,
        "model_training": "train split only",
        "model_evaluation": ["validation", "test"],
        "weighting": "equal case/control weight within protein-Q8 strata",
        "minimum_inference_proteins": args.minimum_inference_proteins,
        "random_seed": args.random_seed,
    }
    (output / "phase3a_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "base_segments": len(base), "candidate_context_rows": len(candidates),
        "matched_analysis_rows": len(signed_data),
        "feature_summary_rows": len(feature_summary),
        "model_performance_rows": len(performance), "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
