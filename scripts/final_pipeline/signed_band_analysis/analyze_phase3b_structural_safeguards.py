#!/usr/bin/env python3
"""Phase 3B safeguards: coordinate-completeness sensitivity and clustered AUROC bootstrap.

This module consumes a Phase 3B segment-feature table and the alignment-audited
contact records used to build it. It does not redefine cases or controls.
Instead, it symmetrically filters cases and controls, reconstructs matched
protein/Q8 strata, separates annotation-derived geometry from experimental-PDB
geometry, saves fixed held-out predictions, and bootstraps test proteins.

Publication execution is intentionally gated by ``--catalog_status
final_stable``. Legacy overlapping-band inputs may be used only for development
with ``--catalog_status legacy_development --development_only`` and an output
directory beneath /tmp.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

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

from .analyze_signed_band_external_structure import (
    BASE_FEATURES,
    DOMAIN_FEATURES,
    LOCAL_ANNOTATION_FEATURES,
    NETWORK_FEATURES,
    PDB_GEOMETRY_FEATURES,
    STAGES,
    add_weights,
    bh,
    ca_curvature,
    ca_virtual_torsion,
    contact_edges,
    load_contacts,
    reconstruct_matched_strata,
    segment_coordinate_completeness,
    within_stratum_concordance,
)


COHORTS = OrderedDict([
    ("reference_80", "accepted mapping, >=80% protein coverage, >=80% segment resolution"),
    ("fully_resolved_local", "accepted mapping and complete segment C-alpha coverage"),
    ("fully_resolved_geometry", "accepted mapping and complete intended PDB geometry"),
    ("strict_graph_95", "fully resolved local segment, complete graph nodes, >=95% protein coverage"),
])

CENTRAL_FEATURES = [
    "mean_contact_degree",
    "mean_inverse_distance_weighted_degree",
    "mean_betweenness",
    "mean_closeness",
    "mean_ca_curvature_degrees",
    "mean_abs_ca_virtual_torsion_degrees",
    "ca_end_to_end_ratio",
]

CENTRAL_FEATURE_GROUP = {
    "mean_contact_degree": "contact_network",
    "mean_inverse_distance_weighted_degree": "contact_network",
    "mean_betweenness": "contact_network",
    "mean_closeness": "contact_network",
    "mean_ca_curvature_degrees": "experimental_pdb_geometry",
    "mean_abs_ca_virtual_torsion_degrees": "experimental_pdb_geometry",
    "ca_end_to_end_ratio": "experimental_pdb_geometry",
}

INCREMENTS = OrderedDict([
    (
        "delta_local_annotation",
        ("01_base_biophysics", "02_add_local_sequence_or_annotation_geometry"),
    ),
    (
        "delta_experimental_pdb_geometry",
        ("02_add_local_sequence_or_annotation_geometry", "03_add_experimental_pdb_geometry"),
    ),
    (
        "delta_contact_network",
        ("03_add_experimental_pdb_geometry", "04_add_contact_network"),
    ),
    (
        "delta_ecod_domain",
        ("04_add_contact_network", "05_add_ecod_domain"),
    ),
    (
        "delta_total_external_structure",
        ("01_base_biophysics", "05_add_ecod_domain"),
    ),
])

SEGMENT_KEY = [
    "split", "protein", "start_index_0based", "end_index_0based_exclusive"
]
STRATUM_KEY = ["condition", "split", "protein", "q8", "sign"]
PREDICTION_KEY = [
    "condition", "sign", "evaluation_split", "structural_cohort",
    "model_training_cohort", "protein", "q8", "segment_id", "selected",
]
PROVENANCE_KEY = ["condition", "split", "protein", "segment_id"]
PROVENANCE_VALUE_COLUMNS = [
    "protein_length", "q8", "start_index_0based", "end_index_0based_exclusive",
    "selected_positive", "selected_negative", "clean_control",
    *BASE_FEATURES, *LOCAL_ANNOTATION_FEATURES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external_features_csv", required=True)
    parser.add_argument("--source_candidate_table", required=True)
    parser.add_argument("--contact_json", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--catalog_status",
        required=True,
        choices=["final_stable", "legacy_development"],
        help="Explicit provenance gate required by the safeguard handoff.",
    )
    parser.add_argument("--development_only", action="store_true")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--reference_segment_resolution", type=float, default=0.80)
    parser.add_argument("--reference_protein_coverage", type=float, default=0.80)
    parser.add_argument("--strict_graph_protein_coverage", type=float, default=0.95)
    parser.add_argument("--min_mapping_identity", type=float, default=0.90)
    parser.add_argument("--min_contact_sequence_separation", type=int, default=3)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--skip_fully_resolved_refit", action="store_true")
    parser.add_argument("--bootstrap_fully_resolved_refit", action="store_true")
    parser.add_argument("--include_validation_bootstrap", action="store_true")
    args = parser.parse_args()
    for name in (
        "reference_segment_resolution", "reference_protein_coverage",
        "strict_graph_protein_coverage", "min_mapping_identity",
    ):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must lie in [0, 1]")
    if args.minimum_inference_proteins < 2:
        parser.error("--minimum_inference_proteins must be at least 2")
    if args.n_bootstrap < 1:
        parser.error("--n_bootstrap must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if args.catalog_status == "legacy_development":
        if not args.development_only:
            parser.error("legacy inputs require --development_only")
        try:
            output.relative_to(Path("/tmp"))
        except ValueError:
            parser.error("legacy development outputs must be written beneath /tmp")
    elif args.development_only:
        parser.error("--development_only is incompatible with --catalog_status final_stable")
    return args


def mapping_is_accepted(record: dict, min_identity: float, min_coverage: float) -> bool:
    alignment = record.get("alignment") or {}
    return bool(
        record.get("status") == "ok"
        and record.get("contact_map_coordinate_system") == "input_sequence_0based"
        and float(alignment.get("identity", -1)) >= min_identity
        and float(alignment.get("input_coverage", -1)) >= min_coverage
    )


def validate_candidate_provenance(
    external: pd.DataFrame, candidates: pd.DataFrame
) -> dict:
    """Require the external table to be an exact derivative of the candidate table."""
    required = set(PROVENANCE_KEY + PROVENANCE_VALUE_COLUMNS)
    for label, frame in (("external feature table", external), ("candidate table", candidates)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} lacks provenance columns {sorted(missing)}")
        if frame.duplicated(PROVENANCE_KEY).any():
            raise ValueError(f"{label} has duplicate candidate provenance keys")

    left = external[PROVENANCE_KEY + PROVENANCE_VALUE_COLUMNS].copy()
    right = candidates[PROVENANCE_KEY + PROVENANCE_VALUE_COLUMNS].copy()
    merged = left.merge(
        right,
        on=PROVENANCE_KEY,
        how="outer",
        suffixes=("_external", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    missing_from_external = int((merged["_merge"] == "right_only").sum())
    extra_in_external = int((merged["_merge"] == "left_only").sum())
    matched = merged[merged["_merge"] == "both"]
    mismatch_counts = {}
    for column in PROVENANCE_VALUE_COLUMNS:
        external_values = matched[f"{column}_external"]
        candidate_values = matched[f"{column}_candidate"]
        if pd.api.types.is_numeric_dtype(external_values) and pd.api.types.is_numeric_dtype(
            candidate_values
        ):
            equal = np.isclose(
                pd.to_numeric(external_values, errors="coerce").to_numpy(float),
                pd.to_numeric(candidate_values, errors="coerce").to_numpy(float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            equal = (
                external_values.fillna("<NA>").astype(str).to_numpy()
                == candidate_values.fillna("<NA>").astype(str).to_numpy()
            )
        mismatch_counts[column] = int((~equal).sum())

    passed = (
        missing_from_external == 0
        and extra_in_external == 0
        and not any(mismatch_counts.values())
        and len(external) == len(candidates) == len(matched)
    )
    report = {
        "external_rows": len(external),
        "candidate_rows": len(candidates),
        "matched_rows": len(matched),
        "missing_from_external": missing_from_external,
        "extra_in_external": extra_in_external,
        "mismatch_counts": mismatch_counts,
        "passed": passed,
    }
    if not passed:
        nonzero = {key: value for key, value in mismatch_counts.items() if value}
        raise ValueError(
            "External features do not match the declared Phase 3A candidate table: "
            f"missing={missing_from_external}, extra={extra_in_external}, "
            f"value_mismatches={nonzero}"
        )
    return report


def contact_degree_array(record: dict, min_sequence_separation: int) -> np.ndarray:
    length = len(record.get("sequence", ""))
    resolved = np.asarray(record.get("resolved_mask", [False] * length), dtype=bool)
    degree = np.full(length, np.nan)
    degree[resolved] = 0.0
    for left, right, _distance in contact_edges(record):
        if (
            abs(left - right) >= min_sequence_separation
            and resolved[left] and resolved[right]
        ):
            degree[left] += 1.0
            degree[right] += 1.0
    return degree


def add_resolution_flags(
    external: pd.DataFrame,
    contacts: dict[str, dict],
    *,
    min_mapping_identity: float,
    reference_protein_coverage: float,
    min_contact_sequence_separation: int,
) -> pd.DataFrame:
    """Attach feature-specific coordinate-completeness fields to segment rows."""
    required = {
        "protein", "split", "start_index_0based", "end_index_0based_exclusive",
        "protein_length", "segment_resolved_fraction",
    }
    missing = required - set(external)
    if missing:
        raise ValueError(f"External feature table lacks {sorted(missing)}")

    physical = external[SEGMENT_KEY + ["protein_length"]].drop_duplicates()
    if physical.duplicated(SEGMENT_KEY).any():
        raise ValueError("Physical segment coordinates are not unique")
    protein_cache = {}
    rows = []
    for segment in physical.itertuples(index=False):
        protein = str(segment.protein)
        start = int(segment.start_index_0based)
        end = int(segment.end_index_0based_exclusive)
        length = int(segment.protein_length)
        if protein not in protein_cache:
            record = contacts.get(protein)
            if record and len(record.get("sequence", "")) != length:
                raise ValueError(
                    f"{protein}: contact length {len(record.get('sequence', ''))} "
                    f"does not match candidate length {length}"
                )
            accepted = bool(
                record and mapping_is_accepted(
                    record, min_mapping_identity, reference_protein_coverage
                )
            )
            if record and record.get("status") == "ok":
                coordinates = record.get("ca_coordinates_angstrom")
                resolved = np.asarray(
                    record.get("resolved_mask", [False] * length), dtype=bool
                )
                curvature = ca_curvature(coordinates) if coordinates else np.full(length, np.nan)
                torsion = ca_virtual_torsion(coordinates) if coordinates else np.full(length, np.nan)
                degree = contact_degree_array(record, min_contact_sequence_separation)
                protein_coverage = float(
                    (record.get("alignment") or {}).get("input_coverage", np.nan)
                )
            else:
                coordinates = None
                resolved = np.zeros(length, dtype=bool)
                curvature = np.full(length, np.nan)
                torsion = np.full(length, np.nan)
                degree = np.full(length, np.nan)
                protein_coverage = np.nan
            protein_cache[protein] = {
                "coordinates": coordinates,
                "resolved": resolved,
                "curvature": curvature,
                "torsion": torsion,
                "degree": degree,
                "accepted": accepted,
                "protein_coverage": protein_coverage,
            }
        cached = protein_cache[protein]
        audit = segment_coordinate_completeness(
            coordinates=cached["coordinates"],
            resolved_mask=cached["resolved"],
            curvature_values=cached["curvature"],
            torsion_values=cached["torsion"],
            contact_degree_values=cached["degree"],
            start=start,
            end=end,
            mapping_accepted=cached["accepted"],
        )
        rows.append({
            "split": segment.split,
            "protein": protein,
            "start_index_0based": start,
            "end_index_0based_exclusive": end,
            "accepted_structure_mapping": cached["accepted"],
            "protein_input_coverage": cached["protein_coverage"],
            **audit,
        })
    flags = pd.DataFrame(rows)
    replace = [column for column in flags if column in external and column not in SEGMENT_KEY]
    output = external.drop(columns=replace).merge(
        flags, on=SEGMENT_KEY, how="left", validate="many_to_one"
    )
    if output.accepted_structure_mapping.isna().any():
        raise RuntimeError("Resolution flags failed to map to every segment row")
    return output


def cohort_mask(frame: pd.DataFrame, cohort: str, args) -> pd.Series:
    accepted = frame.accepted_structure_mapping.astype(bool)
    if cohort == "reference_80":
        return (
            accepted
            & (frame.protein_input_coverage >= args.reference_protein_coverage)
            & (frame.segment_resolved_fraction >= args.reference_segment_resolution)
        )
    if cohort == "fully_resolved_local":
        return accepted & frame.segment_ca_fully_resolved.astype(bool)
    if cohort == "fully_resolved_geometry":
        return accepted & frame.experimental_geometry_complete.astype(bool)
    if cohort == "strict_graph_95":
        return (
            accepted
            & frame.segment_ca_fully_resolved.astype(bool)
            & frame.network_coordinate_complete.astype(bool)
            & (frame.protein_input_coverage >= args.strict_graph_protein_coverage)
        )
    raise ValueError(f"Unknown structural cohort: {cohort}")


def unfiltered_case_control_rows(external: pd.DataFrame, sign: int) -> pd.DataFrame:
    selected_column = "selected_positive" if sign == 1 else "selected_negative"
    opposite_column = "selected_negative" if sign == 1 else "selected_positive"
    cases = external[
        external[selected_column].astype(bool)
        & ~external[opposite_column].astype(bool)
    ].copy()
    controls = external[external.clean_control.astype(bool)].copy()
    cases["selected"], controls["selected"] = 1, 0
    data = pd.concat([cases, controls], ignore_index=True)
    data["sign"] = sign
    return data


def signed_cohort_data(external: pd.DataFrame, cohort: str, args) -> pd.DataFrame:
    pieces = []
    for sign in (-1, 1):
        raw = unfiltered_case_control_rows(external, sign)
        filtered = raw[cohort_mask(raw, cohort, args)].copy()
        filtered["structural_cohort"] = cohort
        pieces.append(reconstruct_matched_strata(filtered))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def cohort_flow(external: pd.DataFrame, args) -> pd.DataFrame:
    total_lookup = (
        external.groupby(["condition", "split", "q8"]).size().rename("n_total_candidate_segments")
    )
    rows = []
    for sign in (-1, 1):
        raw = unfiltered_case_control_rows(external, sign)
        for key, group in raw.groupby(["condition", "split", "q8"], sort=False):
            base = {
                "condition": key[0], "split": key[1], "sign": sign, "q8": key[2],
                "n_total_candidate_segments": int(total_lookup.loc[key]),
                "n_selected_segments": int((group.selected == 1).sum()),
                "n_clean_control_segments": int((group.selected == 0).sum()),
                "n_case_control_proteins": int(group.protein.nunique()),
            }
            stage_masks = OrderedDict([
                ("accepted_mapping", group.accepted_structure_mapping.astype(bool)),
                ("reference_80", cohort_mask(group, "reference_80", args)),
                ("fully_resolved_local", cohort_mask(group, "fully_resolved_local", args)),
                (
                    "fully_resolved_geometry",
                    cohort_mask(group, "fully_resolved_geometry", args),
                ),
                ("strict_graph_95", cohort_mask(group, "strict_graph_95", args)),
            ])
            for stage, mask in stage_masks.items():
                subset = group[mask]
                base[f"{stage}_segments"] = len(subset)
                base[f"{stage}_selected_segments"] = int(subset.selected.sum())
                base[f"{stage}_control_segments"] = int((subset.selected == 0).sum())
                base[f"{stage}_proteins"] = int(subset.protein.nunique())
                matched = reconstruct_matched_strata(subset)
                base[f"{stage}_matched_strata"] = int(
                    matched[["protein", "q8"]].drop_duplicates().shape[0]
                )
                base[f"{stage}_matched_selected_segments"] = int(matched.selected.sum())
                base[f"{stage}_matched_control_segments"] = int(
                    (matched.selected == 0).sum()
                )
            ref_cases = base["reference_80_matched_selected_segments"]
            ref_controls = base["reference_80_matched_control_segments"]
            full_cases = base["fully_resolved_local_matched_selected_segments"]
            full_controls = base["fully_resolved_local_matched_control_segments"]
            base["selected_segments_lost_reference_to_full"] = ref_cases - full_cases
            base["control_segments_lost_reference_to_full"] = ref_controls - full_controls
            base["selected_loss_rate_reference_to_full"] = (
                (ref_cases - full_cases) / ref_cases if ref_cases else np.nan
            )
            base["control_loss_rate_reference_to_full"] = (
                (ref_controls - full_controls) / ref_controls if ref_controls else np.nan
            )
            rows.append(base)
    return pd.DataFrame(rows)


def central_matched_effects(
    cohort_data: dict[str, pd.DataFrame], minimum_proteins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keys = ["condition", "split", "protein", "sign", "q8"]
    for cohort, data in cohort_data.items():
        for key, group in data.groupby(keys, sort=False):
            cases = group[group.selected == 1]
            controls = group[group.selected == 0]
            for feature in CENTRAL_FEATURES:
                case_values = pd.to_numeric(cases[feature], errors="coerce")
                control_values = pd.to_numeric(controls[feature], errors="coerce")
                if not case_values.notna().any() or not control_values.notna().any():
                    continue
                rows.append({
                    **dict(zip(keys, key)),
                    "structural_cohort": cohort,
                    "feature": feature,
                    "feature_group": CENTRAL_FEATURE_GROUP[feature],
                    "n_selected_segments": int(case_values.notna().sum()),
                    "n_control_segments": int(control_values.notna().sum()),
                    "selected_mean": float(case_values.mean()),
                    "control_mean": float(control_values.mean()),
                    "selected_minus_control": float(
                        case_values.mean() - control_values.mean()
                    ),
                })
    per_protein = pd.DataFrame(rows)
    summary_rows = []
    summary_keys = [
        "condition", "split", "sign", "structural_cohort", "q8",
        "feature", "feature_group",
    ]
    for key, group in per_protein.groupby(summary_keys, sort=False):
        values = group.selected_minus_control.dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        inference = n >= minimum_proteins
        half = (
            float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n))
            if inference else np.nan
        )
        if inference:
            try:
                pvalue = float(
                    wilcoxon(values, zero_method="zsplit", method="approx").pvalue
                )
            except ValueError:
                pvalue = 1.0
        else:
            pvalue = np.nan
        summary_rows.append({
            **dict(zip(summary_keys, key)),
            "n_proteins": n,
            "n_selected_segments": int(group.n_selected_segments.sum()),
            "n_control_segments": int(group.n_control_segments.sum()),
            "selected_macro_protein_mean": float(group.selected_mean.mean()),
            "control_macro_protein_mean": float(group.control_mean.mean()),
            "selected_minus_control_macro_mean": mean,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "inference_eligible": inference,
            "wilcoxon_p_two_sided": pvalue,
        })
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return per_protein, summary
    summary["wilcoxon_q_bh"] = bh(summary.wilcoxon_p_two_sided)
    reference = summary[summary.structural_cohort == "reference_80"][
        ["condition", "split", "sign", "q8", "feature",
         "selected_minus_control_macro_mean"]
    ].rename(columns={
        "selected_minus_control_macro_mean": "reference_80_effect"
    })
    summary = summary.merge(
        reference,
        on=["condition", "split", "sign", "q8", "feature"],
        how="left",
        validate="many_to_one",
    )
    summary["effect_difference_from_reference"] = (
        summary.selected_minus_control_macro_mean - summary.reference_80_effect
    )
    summary["effect_direction_preserved"] = (
        np.sign(summary.selected_minus_control_macro_mean)
        == np.sign(summary.reference_80_effect)
    )
    return per_protein, summary


def build_model(features: list[str], args) -> Pipeline:
    transformer = ColumnTransformer([
        ("q8", OneHotEncoder(handle_unknown="ignore"), ["q8"]),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), features),
    ])
    return Pipeline([
        ("transform", transformer),
        ("logistic", LogisticRegression(
            C=1.0, max_iter=args.max_iter, solver="lbfgs",
            random_state=args.random_seed,
        )),
    ])


def performance_row(
    frame: pd.DataFrame, scores: np.ndarray, metadata: dict
) -> dict:
    concordance, n_strata = within_stratum_concordance(frame, scores)
    return {
        **metadata,
        "model_status": "ok",
        "n_segments": len(frame),
        "n_selected_segments": int(frame.selected.sum()),
        "n_control_segments": int((frame.selected == 0).sum()),
        "n_proteins": int(frame.protein.nunique()),
        "weighted_auroc": float(
            roc_auc_score(frame.selected, scores, sample_weight=frame.sample_weight)
        ),
        "weighted_average_precision": float(
            average_precision_score(
                frame.selected, scores, sample_weight=frame.sample_weight
            )
        ),
        "macro_within_protein_q8_concordance": concordance,
        "n_matched_protein_q8_strata": n_strata,
    }


def add_performance_increments(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return performance
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    performance = performance.copy()
    performance["_stage_order"] = performance.stage.map(stage_order)
    performance = performance.sort_values(
        [
            "model_training_cohort", "structural_cohort", "condition",
            "sign", "evaluation_split", "_stage_order",
        ]
    )
    keys = [
        "model_training_cohort", "structural_cohort", "condition",
        "sign", "evaluation_split",
    ]
    performance["delta_auroc_from_previous_stage"] = performance.groupby(
        keys, sort=False
    ).weighted_auroc.diff()
    performance["delta_concordance_from_previous_stage"] = performance.groupby(
        keys, sort=False
    ).macro_within_protein_q8_concordance.diff()
    return performance.drop(columns="_stage_order")


def fit_and_predict(
    cohort_data: dict[str, pd.DataFrame],
    args,
    *,
    source_candidate_table: str,
    source_external_features: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weighted = {
        cohort: add_weights(data.copy()) for cohort, data in cohort_data.items()
    }
    specifications = [(
        "reference_80",
        ["reference_80", "fully_resolved_local", "fully_resolved_geometry", "strict_graph_95"],
    )]
    if not args.skip_fully_resolved_refit:
        specifications.append(("fully_resolved_local", ["fully_resolved_local"]))

    prediction_rows = []
    direct_performance = []
    conditions = sorted(cohort_data["reference_80"].condition.unique())
    for training_cohort, evaluation_cohorts in specifications:
        training_data = weighted[training_cohort]
        for condition in conditions:
            for sign in (-1, 1):
                train = training_data[
                    (training_data.condition == condition)
                    & (training_data.sign == sign)
                    & (training_data.split == "train")
                ]
                if (
                    train.selected.nunique() < 2
                    or train.protein.nunique() < args.minimum_inference_proteins
                ):
                    continue
                for stage, features in STAGES.items():
                    model = build_model(features, args)
                    model.fit(
                        train[["q8"] + features],
                        train.selected,
                        logistic__sample_weight=train.sample_weight,
                    )
                    fitted_id = (
                        f"{condition}__sign{sign:+d}__train-{training_cohort}__{stage}"
                    )
                    for evaluation_cohort in evaluation_cohorts:
                        for split in ("validation", "test"):
                            evaluate = weighted[evaluation_cohort]
                            evaluate = evaluate[
                                (evaluate.condition == condition)
                                & (evaluate.sign == sign)
                                & (evaluate.split == split)
                            ].copy()
                            if evaluate.selected.nunique() < 2:
                                continue
                            scores = model.predict_proba(
                                evaluate[["q8"] + features]
                            )[:, 1]
                            metadata = {
                                "condition": condition,
                                "sign": sign,
                                "stage": stage,
                                "evaluation_split": split,
                                "structural_cohort": evaluation_cohort,
                                "model_training_cohort": training_cohort,
                                "fitted_model_identifier": fitted_id,
                                "n_train_segments": len(train),
                                "n_train_proteins": int(train.protein.nunique()),
                            }
                            direct_performance.append(
                                performance_row(evaluate, scores, metadata)
                            )
                            for row, score in zip(
                                evaluate.itertuples(index=False), scores
                            ):
                                prediction_rows.append({
                                    "condition": condition,
                                    "sign": sign,
                                    "evaluation_split": split,
                                    "protein": row.protein,
                                    "q8": row.q8,
                                    "segment_id": row.segment_id,
                                    "selected": int(row.selected),
                                    "sample_weight": float(row.sample_weight),
                                    "structural_cohort": evaluation_cohort,
                                    "model_training_cohort": training_cohort,
                                    "stage": stage,
                                    "predicted_probability": float(score),
                                    "fitted_model_identifier": fitted_id,
                                    "source_candidate_table": source_candidate_table,
                                    "source_external_features": source_external_features,
                                })
    performance = add_performance_increments(pd.DataFrame(direct_performance))
    return pd.DataFrame(prediction_rows), performance


def performance_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "condition", "sign", "stage", "evaluation_split",
        "structural_cohort", "model_training_cohort", "fitted_model_identifier",
    ]
    for key, group in predictions.groupby(keys, sort=False):
        metadata = dict(zip(keys, key))
        rows.append(
            performance_row(
                group,
                group.predicted_probability.to_numpy(float),
                metadata,
            )
        )
    return add_performance_increments(pd.DataFrame(rows))


def safe_weighted_auroc(labels, scores, weights) -> tuple[float, bool, str]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return np.nan, False, "no_positive_weight_rows"
    if len(np.unique(labels[keep])) < 2:
        return np.nan, False, "one_outcome_class"
    return float(roc_auc_score(labels[keep], scores[keep], sample_weight=weights[keep])), True, ""


def protein_multiplicity_matrix(
    proteins: Iterable[str], n_replicates: int, seed: int
) -> tuple[list[str], np.ndarray]:
    universe = sorted(set(str(protein) for protein in proteins))
    if not universe:
        raise ValueError("Cannot bootstrap an empty protein universe")
    rng = np.random.default_rng(seed)
    matrix = rng.multinomial(
        len(universe),
        np.full(len(universe), 1.0 / len(universe)),
        size=n_replicates,
    )
    return universe, matrix


def multiply_weights_by_protein(
    frame: pd.DataFrame, multiplicities: dict[str, int]
) -> np.ndarray:
    return (
        frame.sample_weight.to_numpy(float)
        * frame.protein.astype(str).map(multiplicities).fillna(0).to_numpy(float)
    )


def bootstrap_predictions(
    predictions: pd.DataFrame,
    *,
    n_replicates: int,
    random_seed: int,
    include_validation: bool,
    include_refit: bool,
    expected_conditions: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = ["test"] + (["validation"] if include_validation else [])
    allowed = (
        ((predictions.model_training_cohort == "reference_80")
         & predictions.structural_cohort.isin([
             "reference_80", "fully_resolved_local", "fully_resolved_geometry",
         ]))
        | (
            include_refit
            & (predictions.model_training_cohort == "fully_resolved_local")
            & (predictions.structural_cohort == "fully_resolved_local")
        )
    )
    work = predictions[allowed & predictions.evaluation_split.isin(splits)].copy()
    replicate_rows = []
    synchronized_rows = []
    draw_banks = {}
    draw_bank_keys = ["sign", "evaluation_split", "model_training_cohort"]
    for draw_key, draw_group in work.groupby(draw_bank_keys, sort=False):
        universe = sorted(draw_group.protein.astype(str).unique())
        stable_seed = (
            random_seed + zlib.crc32("|".join(map(str, draw_key)).encode())
        ) % (2**32)
        draw_banks[draw_key] = protein_multiplicity_matrix(
            universe, n_replicates, stable_seed
        )
    group_keys = [
        "sign", "evaluation_split", "structural_cohort", "model_training_cohort"
    ]
    for group_key, group in work.groupby(group_keys, sort=False):
        sign, split, cohort, training_cohort = group_key
        conditions = sorted(group.condition.unique())
        if expected_conditions is not None:
            expected = sorted(str(value) for value in expected_conditions)
            if conditions != expected:
                raise ValueError(
                    "Synchronized bootstrap condition mismatch for "
                    f"sign={sign}, split={split}, cohort={cohort}, "
                    f"training_cohort={training_cohort}: "
                    f"expected={expected}, observed={conditions}"
                )
        universe, multiplicity_matrix = draw_banks[
            (sign, split, training_cohort)
        ]
        universe_index = {protein: index for index, protein in enumerate(universe)}
        cohort_proteins = set(group.protein.astype(str))
        cohort_protein_codes = np.asarray(
            [universe_index[protein] for protein in cohort_proteins], dtype=int
        )
        prepared = {}
        for condition in conditions:
            condition_frame = group[group.condition == condition]
            stages = {}
            reference_keys = None
            for stage, stage_frame in condition_frame.groupby("stage", sort=False):
                stage_frame = stage_frame.sort_values(
                    ["protein", "q8", "segment_id", "selected"]
                ).reset_index(drop=True)
                row_keys = list(zip(
                    stage_frame.protein.astype(str),
                    stage_frame.q8.astype(str),
                    stage_frame.segment_id.astype(str),
                    stage_frame.selected.astype(int),
                ))
                if reference_keys is None:
                    reference_keys = row_keys
                    labels = stage_frame.selected.to_numpy(int)
                    base_weights = stage_frame.sample_weight.to_numpy(float)
                    protein_codes = stage_frame.protein.astype(str).map(
                        universe_index
                    ).to_numpy(int)
                elif row_keys != reference_keys:
                    raise ValueError(
                        f"Stage row mismatch for {condition}, sign={sign}, cohort={cohort}"
                    )
                stages[stage] = stage_frame.predicted_probability.to_numpy(float)
            prepared[condition] = {
                "labels": labels,
                "base_weights": base_weights,
                "protein_codes": protein_codes,
                "scores": stages,
                "n_selected": int(labels.sum()),
                "n_control": int((labels == 0).sum()),
                "n_proteins": int(condition_frame.protein.nunique()),
            }

        for replicate in range(n_replicates):
            multiplicities = multiplicity_matrix[replicate]
            draw_checksum = f"{zlib.crc32(multiplicities.tobytes()):08x}"
            condition_deltas = {increment: [] for increment in INCREMENTS}
            for condition in conditions:
                prepared_condition = prepared[condition]
                weights = (
                    prepared_condition["base_weights"]
                    * multiplicities[prepared_condition["protein_codes"]]
                )
                stage_aurocs = {}
                stage_validity = {}
                for stage, scores in prepared_condition["scores"].items():
                    value, valid, reason = safe_weighted_auroc(
                        prepared_condition["labels"], scores, weights
                    )
                    stage_aurocs[stage] = value
                    stage_validity[stage] = (valid, reason)
                for increment, (small_stage, large_stage) in INCREMENTS.items():
                    small_valid, small_reason = stage_validity.get(
                        small_stage, (False, "missing_stage")
                    )
                    large_valid, large_reason = stage_validity.get(
                        large_stage, (False, "missing_stage")
                    )
                    valid = small_valid and large_valid
                    reason = "" if valid else small_reason or large_reason
                    delta = (
                        stage_aurocs[large_stage] - stage_aurocs[small_stage]
                        if valid else np.nan
                    )
                    if valid:
                        condition_deltas[increment].append(delta)
                    replicate_rows.append({
                        "replicate": replicate + 1,
                        "draw_checksum": draw_checksum,
                        "condition": condition,
                        "sign": sign,
                        "evaluation_split": split,
                        "structural_cohort": cohort,
                        "model_training_cohort": training_cohort,
                        "increment": increment,
                        "smaller_stage": small_stage,
                        "expanded_stage": large_stage,
                        "smaller_stage_auroc": stage_aurocs.get(small_stage, np.nan),
                        "expanded_stage_auroc": stage_aurocs.get(large_stage, np.nan),
                        "delta_auroc": delta,
                        "valid": valid,
                        "invalid_reason": reason,
                        "n_proteins_in_universe": len(universe),
                        "n_eligible_proteins_in_cohort": len(cohort_proteins),
                        "n_unique_proteins_sampled": int((multiplicities > 0).sum()),
                        "n_unique_eligible_proteins_sampled": int(
                            (multiplicities[cohort_protein_codes] > 0).sum()
                        ),
                        "n_selected_segments": prepared_condition["n_selected"],
                        "n_control_segments": prepared_condition["n_control"],
                    })
            for increment, deltas in condition_deltas.items():
                valid = len(deltas) == len(conditions)
                synchronized_rows.append({
                    "replicate": replicate + 1,
                    "draw_checksum": draw_checksum,
                    "sign": sign,
                    "evaluation_split": split,
                    "structural_cohort": cohort,
                    "model_training_cohort": training_cohort,
                    "increment": increment,
                    "mean_delta_auroc": float(np.mean(deltas)) if valid else np.nan,
                    "valid": valid,
                    "n_valid_conditions": len(deltas),
                    "n_expected_conditions": len(conditions),
                    "n_proteins_in_universe": len(universe),
                    "n_eligible_proteins_in_cohort": len(cohort_proteins),
                })
    return pd.DataFrame(replicate_rows), pd.DataFrame(synchronized_rows)


def audit_prediction_table(
    predictions: pd.DataFrame, expected_conditions: Optional[Iterable[str]] = None
) -> dict:
    required = set(PREDICTION_KEY + [
        "stage", "predicted_probability", "sample_weight",
        "fitted_model_identifier", "source_candidate_table",
        "source_external_features",
    ])
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Prediction table lacks required columns {sorted(missing)}")
    duplicate_rows = int(predictions.duplicated(PREDICTION_KEY + ["stage"]).sum())
    invalid_probabilities = int(
        (
            ~np.isfinite(predictions.predicted_probability)
            | ~predictions.predicted_probability.between(0, 1)
        ).sum()
    )
    invalid_weights = int(
        (~np.isfinite(predictions.sample_weight) | (predictions.sample_weight <= 0)).sum()
    )
    candidate_sources = sorted(
        predictions.source_candidate_table.dropna().astype(str).unique()
    )
    external_sources = sorted(
        predictions.source_external_features.dropna().astype(str).unique()
    )
    incomplete_stage_groups = 0
    missing_condition_groups = 0
    expected = (
        sorted(str(value) for value in expected_conditions)
        if expected_conditions is not None else None
    )
    evaluation_grouping = [
        "sign", "evaluation_split", "structural_cohort", "model_training_cohort",
    ]
    for _key, group in predictions.groupby(
        evaluation_grouping + ["condition"], sort=False
    ):
        if set(group.stage.astype(str)) != set(STAGES):
            incomplete_stage_groups += 1
    for _key, group in predictions.groupby(evaluation_grouping, sort=False):
        if expected is not None and sorted(group.condition.astype(str).unique()) != expected:
            missing_condition_groups += 1
    missing_required_fixed_model_groups = []
    if expected is not None:
        observed_groups = set(
            predictions[
                predictions.model_training_cohort == "reference_80"
            ][["sign", "evaluation_split", "structural_cohort"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        required_groups = {
            (sign, split, cohort)
            for sign in (-1, 1)
            for split in ("validation", "test")
            for cohort in COHORTS
        }
        missing_required_fixed_model_groups = sorted(
            [list(group) for group in required_groups - observed_groups],
            key=str,
        )
    passed = (
        len(predictions) > 0
        and duplicate_rows == 0
        and invalid_probabilities == 0
        and invalid_weights == 0
        and len(candidate_sources) == 1
        and len(external_sources) == 1
        and incomplete_stage_groups == 0
        and missing_condition_groups == 0
        and not missing_required_fixed_model_groups
    )
    return {
        "prediction_rows": len(predictions),
        "duplicate_prediction_stage_rows": duplicate_rows,
        "invalid_probability_rows": invalid_probabilities,
        "invalid_sample_weight_rows": invalid_weights,
        "source_candidate_tables": candidate_sources,
        "source_external_feature_tables": external_sources,
        "incomplete_stage_groups": incomplete_stage_groups,
        "missing_condition_groups": missing_condition_groups,
        "missing_required_fixed_model_groups": missing_required_fixed_model_groups,
        "passed": passed,
    }


def observed_increment_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "condition", "sign", "evaluation_split",
        "structural_cohort", "model_training_cohort",
    ]
    for key, group in predictions.groupby(keys, sort=False):
        stage_aurocs = {}
        for stage, stage_frame in group.groupby("stage", sort=False):
            value, valid, _reason = safe_weighted_auroc(
                stage_frame.selected,
                stage_frame.predicted_probability,
                stage_frame.sample_weight,
            )
            if valid:
                stage_aurocs[stage] = value
        for increment, (small_stage, large_stage) in INCREMENTS.items():
            if small_stage in stage_aurocs and large_stage in stage_aurocs:
                rows.append({
                    **dict(zip(keys, key)),
                    "increment": increment,
                    "smaller_stage": small_stage,
                    "expanded_stage": large_stage,
                    "observed_delta_auroc": (
                        stage_aurocs[large_stage] - stage_aurocs[small_stage]
                    ),
                    "n_proteins": int(group.protein.nunique()),
                    "n_selected_segments": int(
                        group.drop_duplicates("segment_id").selected.sum()
                    ),
                    "n_control_segments": int(
                        (group.drop_duplicates("segment_id").selected == 0).sum()
                    ),
                })
    return pd.DataFrame(rows)


def bootstrap_summaries(
    replicates: pd.DataFrame,
    synchronized: pd.DataFrame,
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keys = [
        "condition", "sign", "evaluation_split",
        "structural_cohort", "model_training_cohort", "increment",
    ]
    for key, group in replicates.groupby(keys, sort=False):
        valid = group[group.valid.astype(bool)].delta_auroc.dropna().to_numpy(float)
        observed_row = observed
        for column, value in zip(keys, key):
            observed_row = observed_row[observed_row[column] == value]
        metadata = observed_row.iloc[0].to_dict() if len(observed_row) else {}
        rows.append({
            **dict(zip(keys, key)),
            "smaller_stage": group.smaller_stage.iloc[0],
            "expanded_stage": group.expanded_stage.iloc[0],
            "observed_delta_auroc": metadata.get("observed_delta_auroc", np.nan),
            "bootstrap_median_delta_auroc": float(np.median(valid)) if len(valid) else np.nan,
            "ci95_low": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
            "ci95_high": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
            "valid_replicates": len(valid),
            "total_replicates": len(group),
            "valid_replicate_fraction": len(valid) / len(group),
            "fraction_delta_le_zero": float(np.mean(valid <= 0)) if len(valid) else np.nan,
            "n_proteins": metadata.get("n_proteins", np.nan),
            "n_selected_segments": metadata.get("n_selected_segments", np.nan),
            "n_control_segments": metadata.get("n_control_segments", np.nan),
        })
    condition_summary = pd.DataFrame(rows)

    mean_rows = []
    mean_keys = [
        "sign", "evaluation_split", "structural_cohort",
        "model_training_cohort", "increment",
    ]
    observed_means = observed.groupby(mean_keys).observed_delta_auroc.mean()
    for key, group in synchronized.groupby(mean_keys, sort=False):
        valid = group[group.valid.astype(bool)].mean_delta_auroc.dropna().to_numpy(float)
        mean_rows.append({
            **dict(zip(mean_keys, key)),
            "observed_mean_delta_auroc": float(observed_means.loc[key]),
            "bootstrap_median_mean_delta_auroc": (
                float(np.median(valid)) if len(valid) else np.nan
            ),
            "ci95_low": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
            "ci95_high": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
            "valid_replicates": len(valid),
            "total_replicates": len(group),
            "valid_replicate_fraction": len(valid) / len(group),
            "fraction_mean_delta_le_zero": (
                float(np.mean(valid <= 0)) if len(valid) else np.nan
            ),
            "n_conditions": int(group.n_expected_conditions.max()),
            "n_proteins_in_universe": int(group.n_proteins_in_universe.max()),
        })
    return condition_summary, pd.DataFrame(mean_rows)


def prediction_reconstruction_audit(
    direct: pd.DataFrame, reconstructed: pd.DataFrame
) -> dict:
    keys = [
        "condition", "sign", "stage", "evaluation_split",
        "structural_cohort", "model_training_cohort",
    ]
    columns = [
        "weighted_auroc", "weighted_average_precision",
        "macro_within_protein_q8_concordance",
    ]
    merged = direct[keys + columns].merge(
        reconstructed[keys + columns],
        on=keys,
        suffixes=("_direct", "_reconstructed"),
        validate="one_to_one",
    )
    errors = {}
    for column in columns:
        errors[column] = float(np.nanmax(np.abs(
            merged[f"{column}_direct"] - merged[f"{column}_reconstructed"]
        )))
    return {
        "direct_rows": len(direct),
        "reconstructed_rows": len(reconstructed),
        "matched_rows": len(merged),
        "maximum_absolute_errors": errors,
        "passed": (
            len(direct) == len(reconstructed) == len(merged)
            and max(errors.values(), default=0.0) <= 1e-12
        ),
    }


def write_readme(path: Path, args) -> None:
    path.write_text(
        "# Phase 3B structural safeguards\n\n"
        "This directory contains the complete-coordinate sensitivity analysis "
        "and paired protein-clustered bootstrap requested in the structural "
        "safeguards handoff.\n\n"
        f"Catalog status: `{args.catalog_status}`.\n\n"
        "The experimental-PDB increment is stage 03 minus stage 02; stage 02 "
        "contains only local sequence/annotation geometry. Bootstrap models are "
        "held fixed and test proteins—not segments—are resampled. Strain is not "
        "conditioned on PDB-coordinate completeness.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty safeguard output directory: {output}"
        )
    external_path = Path(args.external_features_csv).expanduser().resolve()
    candidate_path = Path(args.source_candidate_table).expanduser().resolve()
    if not external_path.is_file():
        raise FileNotFoundError(external_path)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    external = pd.read_csv(external_path)
    candidates = pd.read_csv(candidate_path)
    if args.conditions:
        external = external[external.condition.isin(args.conditions)].copy()
        candidates = candidates[candidates.condition.isin(args.conditions)].copy()
    provenance_audit = validate_candidate_provenance(external, candidates)
    required = {
        "condition", "split", "protein", "protein_length", "q8", "segment_id",
        "start_index_0based", "end_index_0based_exclusive",
        "selected_positive", "selected_negative", "clean_control",
        "segment_resolved_fraction", *BASE_FEATURES, *LOCAL_ANNOTATION_FEATURES,
        *PDB_GEOMETRY_FEATURES, *NETWORK_FEATURES, *DOMAIN_FEATURES,
    }
    missing = required - set(external)
    if missing:
        raise ValueError(f"External feature table lacks {sorted(missing)}")
    for column in ("selected_positive", "selected_negative", "clean_control"):
        if external[column].isna().any():
            raise ValueError(f"{column} contains missing values")
    if external.segment_id.duplicated().any():
        raise ValueError("segment_id must be unique in the external feature table")
    expected_conditions = sorted(external.condition.astype(str).unique())
    if args.catalog_status == "final_stable":
        expected_splits = {"train", "validation", "test"}
        observed_splits = set(external.split.astype(str))
        if observed_splits != expected_splits:
            raise ValueError(
                f"Final stable execution requires splits {sorted(expected_splits)}, "
                f"found {sorted(observed_splits)}"
            )
        if external.condition.nunique() != 6:
            raise ValueError(
                "Final stable execution requires all six model conditions; "
                f"found {external.condition.nunique()}"
            )
    contacts = load_contacts(args.contact_json)
    flagged = add_resolution_flags(
        external,
        contacts,
        min_mapping_identity=args.min_mapping_identity,
        reference_protein_coverage=args.reference_protein_coverage,
        min_contact_sequence_separation=args.min_contact_sequence_separation,
    )
    cohorts = {
        cohort: signed_cohort_data(flagged, cohort, args) for cohort in COHORTS
    }
    flow = cohort_flow(flagged, args)
    per_protein, effect_summary = central_matched_effects(
        cohorts, args.minimum_inference_proteins
    )
    predictions, direct_performance = fit_and_predict(
        cohorts,
        args,
        source_candidate_table=str(candidate_path),
        source_external_features=str(external_path),
    )
    prediction_table_audit = audit_prediction_table(
        predictions,
        expected_conditions=(
            expected_conditions if args.catalog_status == "final_stable" else None
        ),
    )
    if not prediction_table_audit["passed"]:
        raise RuntimeError(
            f"Structural prediction-table audit failed: {prediction_table_audit}"
        )
    reconstructed = performance_from_predictions(predictions)
    reconstruction_audit = prediction_reconstruction_audit(
        direct_performance, reconstructed
    )
    if not reconstruction_audit["passed"]:
        raise RuntimeError("Saved predictions failed point-estimate reconstruction audit")

    bootstrap_splits = ["test"] + (
        ["validation"] if args.include_validation_bootstrap else []
    )
    bootstrap_predictions_input = predictions[
        predictions.evaluation_split.isin(bootstrap_splits)
    ]
    observed = observed_increment_table(bootstrap_predictions_input)
    replicates, synchronized = bootstrap_predictions(
        bootstrap_predictions_input,
        n_replicates=args.n_bootstrap,
        random_seed=args.random_seed,
        include_validation=args.include_validation_bootstrap,
        include_refit=args.bootstrap_fully_resolved_refit,
        expected_conditions=(
            expected_conditions if args.catalog_status == "final_stable" else None
        ),
    )
    increment_summary, synchronized_summary = bootstrap_summaries(
        replicates, synchronized, observed
    )

    resolution_audit = {
        "catalog_status": args.catalog_status,
        "candidate_table": str(candidate_path),
        "external_features": str(external_path),
        "input_rows": len(external),
        "flagged_rows": len(flagged),
        "cohort_rows": {name: len(data) for name, data in cohorts.items()},
        "cohort_proteins": {
            name: int(data.protein.nunique()) for name, data in cohorts.items()
        },
        "partially_resolved_rows_in_fully_resolved_local": int(
            (~cohorts["fully_resolved_local"].segment_ca_fully_resolved.astype(bool)).sum()
        ),
        "geometry_incomplete_rows_in_fully_resolved_geometry": int(
            (
                ~cohorts["fully_resolved_geometry"].experimental_geometry_complete.astype(bool)
            ).sum()
        ),
        "incomplete_graph_rows_in_strict_graph_95": int(
            (
                ~cohorts["strict_graph_95"].network_coordinate_complete.astype(bool)
            ).sum()
        ),
        "cohorts_with_invalid_case_control_strata": {
            name: int(
                (
                    data.groupby(STRATUM_KEY).selected.nunique() != 2
                ).sum()
            ) if len(data) else 0
            for name, data in cohorts.items()
        },
        "strain_policy": "not filtered or re-estimated by this safeguard",
        "passed": (
            len(external) == len(flagged)
            and provenance_audit["passed"]
            and not cohorts["fully_resolved_local"].empty
            and not cohorts["fully_resolved_geometry"].empty
            and bool(cohorts["fully_resolved_local"].segment_ca_fully_resolved.all())
            and bool(
                cohorts[
                    "fully_resolved_geometry"
                ].experimental_geometry_complete.all()
            )
            and all(
                (
                    data.groupby(STRATUM_KEY).selected.nunique() == 2
                ).all()
                for data in cohorts.values() if len(data)
            )
        ),
    }
    synchronized_draws_verified = bool(
        len(replicates)
        and (
            replicates.groupby(
                [
                    "sign", "evaluation_split", "model_training_cohort",
                    "replicate",
                ]
            ).draw_checksum.nunique() == 1
        ).all()
    )
    synchronized_condition_count_verified = bool(
        len(synchronized)
        and (
            synchronized.n_expected_conditions
            == (6 if args.catalog_status == "final_stable" else synchronized.n_expected_conditions)
        ).all()
    )
    geometry_bootstrap_present = bool(
        len(increment_summary)
        and (
            (increment_summary.model_training_cohort == "reference_80")
            & (increment_summary.structural_cohort == "fully_resolved_geometry")
            & (increment_summary.evaluation_split == "test")
        ).any()
    )
    bootstrap_audit = {
        "bootstrap_unit": "protein",
        "segment_resampling_used": False,
        "models_refit_within_primary_bootstrap": False,
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed,
        "bootstrap_splits": bootstrap_splits,
        "increment_definitions": {
            name: {"smaller_stage": pair[0], "expanded_stage": pair[1]}
            for name, pair in INCREMENTS.items()
        },
        "experimental_pdb_increment_is_stage03_minus_stage02": True,
        "minimum_condition_valid_replicate_fraction": (
            float(increment_summary.valid_replicate_fraction.min())
            if len(increment_summary) else np.nan
        ),
        "minimum_synchronized_valid_replicate_fraction": (
            float(synchronized_summary.valid_replicate_fraction.min())
            if len(synchronized_summary) else np.nan
        ),
        "all_condition_valid_fractions_at_least_0.95": bool(
            len(increment_summary)
            and (increment_summary.valid_replicate_fraction >= 0.95).all()
        ),
        "all_synchronized_valid_fractions_at_least_0.95": bool(
            len(synchronized_summary)
            and (synchronized_summary.valid_replicate_fraction >= 0.95).all()
        ),
        "synchronized_draws": synchronized_draws_verified,
        "synchronized_condition_count_verified": synchronized_condition_count_verified,
        "fully_resolved_geometry_bootstrap_present": geometry_bootstrap_present,
        "prediction_table_passed": prediction_table_audit["passed"],
    }
    bootstrap_audit["passed"] = bool(
        bootstrap_audit["all_condition_valid_fractions_at_least_0.95"]
        and bootstrap_audit["all_synchronized_valid_fractions_at_least_0.95"]
        and bootstrap_audit["synchronized_draws"]
        and bootstrap_audit["synchronized_condition_count_verified"]
        and bootstrap_audit["fully_resolved_geometry_bootstrap_present"]
        and bootstrap_audit["prediction_table_passed"]
    )
    if args.catalog_status == "final_stable" and not resolution_audit["passed"]:
        raise RuntimeError(f"Final resolution-sensitivity audit failed: {resolution_audit}")
    if args.catalog_status == "final_stable" and not bootstrap_audit["passed"]:
        raise RuntimeError(f"Final structural bootstrap audit failed: {bootstrap_audit}")

    # Do not leave a plausible-looking partial publication result if a final
    # audit fails. Output creation begins only after every mandatory gate passes.
    resolution_dir = output / "resolution_sensitivity"
    bootstrap_dir = output / "bootstrap"
    audit_dir = output / "audits"
    for directory in (resolution_dir, bootstrap_dir, audit_dir, output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(
        resolution_dir / "segment_resolution_flags.csv.gz",
        index=False, compression="gzip",
    )
    flow.to_csv(resolution_dir / "cohort_flow.csv", index=False)
    per_protein.to_csv(
        resolution_dir / "matched_effects_by_protein.csv.gz",
        index=False, compression="gzip",
    )
    effect_summary.to_csv(
        resolution_dir / "matched_effect_summary.csv", index=False
    )
    fixed = direct_performance[
        direct_performance.model_training_cohort == "reference_80"
    ]
    refit = direct_performance[
        direct_performance.model_training_cohort == "fully_resolved_local"
    ]
    fixed.to_csv(resolution_dir / "fixed_model_performance.csv", index=False)
    refit.to_csv(
        resolution_dir / "fully_resolved_refit_performance.csv", index=False
    )
    predictions.to_csv(
        bootstrap_dir / "structural_model_predictions.csv.gz",
        index=False, compression="gzip",
    )
    replicates.to_csv(
        bootstrap_dir / "bootstrap_replicates.csv.gz",
        index=False, compression="gzip",
    )
    increment_summary.to_csv(
        bootstrap_dir / "bootstrap_increment_summary.csv", index=False
    )
    synchronized.to_csv(
        bootstrap_dir / "synchronized_condition_mean_replicates.csv.gz",
        index=False, compression="gzip",
    )
    synchronized_summary.to_csv(
        bootstrap_dir / "synchronized_condition_mean_summary.csv", index=False
    )
    (audit_dir / "candidate_provenance_audit.json").write_text(
        json.dumps(provenance_audit, indent=2) + "\n", encoding="utf-8"
    )
    (audit_dir / "prediction_table_audit.json").write_text(
        json.dumps(prediction_table_audit, indent=2) + "\n", encoding="utf-8"
    )
    (audit_dir / "resolution_sensitivity_audit.json").write_text(
        json.dumps(resolution_audit, indent=2) + "\n", encoding="utf-8"
    )
    (audit_dir / "prediction_reconstruction_audit.json").write_text(
        json.dumps(reconstruction_audit, indent=2) + "\n", encoding="utf-8"
    )
    (audit_dir / "bootstrap_audit.json").write_text(
        json.dumps(bootstrap_audit, indent=2) + "\n", encoding="utf-8"
    )
    parameters = {
        "phase": "3B structural safeguards",
        "catalog_status": args.catalog_status,
        "development_only": args.development_only,
        "source_candidate_table": str(candidate_path),
        "source_external_features": str(external_path),
        "candidate_provenance_validated": True,
        "contact_json": [
            str(Path(path).expanduser().resolve()) for path in args.contact_json
        ],
        "cohorts": COHORTS,
        "corrected_model_stages": STAGES,
        "increments": {
            name: {"smaller_stage": pair[0], "expanded_stage": pair[1]}
            for name, pair in INCREMENTS.items()
        },
        "central_matched_features": CENTRAL_FEATURES,
        "reference_segment_resolution": args.reference_segment_resolution,
        "reference_protein_coverage": args.reference_protein_coverage,
        "strict_graph_protein_coverage": args.strict_graph_protein_coverage,
        "min_mapping_identity": args.min_mapping_identity,
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed,
        "fully_resolved_refit": not args.skip_fully_resolved_refit,
        "bootstrap_fully_resolved_refit": args.bootstrap_fully_resolved_refit,
        "bootstrapped_fixed_model_cohorts": [
            "reference_80", "fully_resolved_local", "fully_resolved_geometry",
        ],
        "strain_conditioned_on_pdb_resolution": False,
    }
    (output / "parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(output / "README.md", args)
    print(json.dumps({
        "catalog_status": args.catalog_status,
        "flagged_segment_rows": len(flagged),
        "prediction_rows": len(predictions),
        "bootstrap_replicate_rows": len(replicates),
        "minimum_condition_valid_bootstrap_fraction": bootstrap_audit[
            "minimum_condition_valid_replicate_fraction"
        ],
        "minimum_synchronized_valid_bootstrap_fraction": bootstrap_audit[
            "minimum_synchronized_valid_replicate_fraction"
        ],
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
