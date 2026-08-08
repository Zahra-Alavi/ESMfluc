#!/usr/bin/env python3
"""Audit signed-band coordinate integrity and output completeness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_bands_csv", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--enrichment_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=None,
        help=(
            "Dataset splits included in the enrichment run. The annotation "
            "files may still contain all splits; completeness expectations "
            "are restricted to the splits listed here."
        ),
    )
    return parser.parse_args()


def check(name: str, condition: bool, details, checks: list[dict]) -> None:
    checks.append({"check": name, "passed": bool(condition), "details": details})


def main() -> None:
    args = parse_args()
    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    enrichment_dir = Path(args.enrichment_dir).expanduser().resolve()
    original = pd.read_csv(args.original_bands_csv)
    annotated = pd.read_csv(annotation_dir / "signed_bands_biophysical_annotations.csv.gz")
    residue = pd.read_csv(annotation_dir / "residue_biophysical_annotations.csv.gz")
    summary = pd.read_csv(enrichment_dir / "apex_circular_shift_enrichment_summary.csv")
    null = pd.read_csv(enrichment_dir / "apex_circular_shift_null.csv.gz")
    per_protein = pd.read_csv(enrichment_dir / "apex_metrics_by_protein.csv.gz")
    contrasts = pd.read_csv(enrichment_dir / "paired_flex_vs_rigid_summary.csv")
    phase1_protein = pd.read_csv(
        enrichment_dir / "annotation_band_coverage_by_protein.csv.gz"
    )
    phase1_summary = pd.read_csv(
        enrichment_dir / "annotation_band_coverage_identifier_summary.csv"
    )
    matched_controls = pd.read_csv(
        enrichment_dir / "within_q3_matched_controls.csv.gz"
    )
    matched_cases = pd.read_csv(
        enrichment_dir / "within_q3_matched_cases.csv.gz"
    )
    match_coverage = pd.read_csv(
        enrichment_dir / "within_q3_match_coverage_summary.csv"
    )
    match_balance = pd.read_csv(enrichment_dir / "within_q3_match_balance.csv")
    matched_protein = pd.read_csv(
        enrichment_dir / "within_q3_matched_effects_by_protein.csv.gz"
    )
    matched_summary = pd.read_csv(
        enrichment_dir / "within_q3_matched_enrichment_summary.csv"
    )
    group_inference = pd.read_csv(
        enrichment_dir / "headline_union_group_inference.csv"
    )
    parameters = json.loads(
        (enrichment_dir / "biophysical_enrichment_parameters.json").read_text()
    )
    annotation_audit = json.loads((annotation_dir / "annotation_audit.json").read_text())
    protein_summary = pd.read_csv(parameters["protein_summary_csv"])
    strain_parameters = parameters.get("strain", {})
    strain_requested = bool(strain_parameters.get("requested", False))
    strain_audit = pd.DataFrame()
    if strain_requested:
        strain_audit_path = Path(strain_parameters["audit"])
        if strain_audit_path.exists():
            strain_audit = pd.read_csv(strain_audit_path)
    checks = []

    requested_splits = set(args.splits or annotated["split"].astype(str).unique())
    scoped_annotated = annotated[
        annotated["split"].astype(str).isin(requested_splits)
    ].copy()
    scoped_protein_summary = protein_summary[
        protein_summary["split"].astype(str).isin(requested_splits)
    ].copy()

    scoped_outputs = {
        "enrichment_summary": summary,
        "null": null,
        "per_protein_metrics": per_protein,
        "paired_contrasts": contrasts,
        "coverage_by_protein": phase1_protein,
        "coverage_summary": phase1_summary,
        "matched_controls": matched_controls,
        "matched_cases": matched_cases,
        "match_coverage": match_coverage,
        "match_balance": match_balance,
        "matched_effects_by_protein": matched_protein,
        "matched_summary": matched_summary,
    }
    output_split_details = {
        name: sorted(frame["split"].astype(str).unique().tolist())
        for name, frame in scoped_outputs.items()
        if "split" in frame.columns
    }
    check(
        "analysis_outputs_match_requested_splits",
        all(set(splits) == requested_splits for splits in output_split_details.values()),
        {
            "requested": sorted(requested_splits),
            "observed_by_output": output_split_details,
        },
        checks,
    )

    check("original_band_ids_unique", not original.band_id.duplicated().any(), len(original), checks)
    check("annotated_band_ids_unique", not annotated.band_id.duplicated().any(), len(annotated), checks)
    check(
        "band_row_count_preserved",
        len(original) == len(annotated),
        {"original": len(original), "annotated": len(annotated)},
        checks,
    )
    check(
        "band_id_set_preserved",
        set(original.band_id) == set(annotated.band_id),
        "exact set comparison",
        checks,
    )
    check(
        "coordinates_preserved",
        original.set_index("band_id")[[
            "apex_index_0based", "start_index_0based", "end_index_0based_inclusive"
        ]].sort_index().equals(
            annotated.set_index("band_id")[[
                "apex_index_0based", "start_index_0based", "end_index_0based_inclusive"
            ]].sort_index()
        ),
        "apex/start/end exact equality",
        checks,
    )
    check(
        "interval_counts_equal_width",
        bool((annotated.interval_annotation_residue_count == annotated.band_width).all()),
        int((annotated.interval_annotation_residue_count != annotated.band_width).sum()),
        checks,
    )
    check(
        "apex_amino_acids_present",
        bool(annotated.apex_amino_acid.notna().all()),
        int(annotated.apex_amino_acid.isna().sum()),
        checks,
    )
    residue_key = ["split", "protein", "residue_index_0based"]
    check(
        "residue_keys_unique",
        not residue.duplicated(residue_key).any(),
        int(residue.duplicated(residue_key).sum()),
        checks,
    )
    expected_proteins = {"train": 967, "validation": 208, "test": 208}
    observed_proteins = residue.groupby("split").protein.nunique().to_dict()
    check(
        "expected_split_protein_counts",
        observed_proteins == expected_proteins,
        {"observed": observed_proteins, "expected": expected_proteins},
        checks,
    )
    check(
        "all_apex_annotations_complete",
        not annotated[["apex_neq", "apex_q3", "apex_q8", "apex_rsa", "apex_disorder"]].isna().any().any(),
        annotated[["apex_neq", "apex_q3", "apex_q8", "apex_rsa", "apex_disorder"]].isna().sum().to_dict(),
        checks,
    )
    apex_reference = residue[[
        "split", "protein", "residue_index_0based", "neq", "q3", "q8"
    ]].rename(columns={
        "residue_index_0based": "apex_index_0based",
        "neq": "reference_neq",
        "q3": "reference_q3",
        "q8": "reference_q8",
    })
    apex_join = annotated[[
        "band_id", "split", "protein", "apex_index_0based", "apex_neq", "apex_q3", "apex_q8"
    ]].merge(
        apex_reference,
        on=["split", "protein", "apex_index_0based"],
        how="left",
        validate="many_to_one",
    )
    check(
        "apex_neq_rejoined_exactly",
        bool(np.allclose(apex_join.apex_neq, apex_join.reference_neq, equal_nan=True)),
        int((~np.isclose(apex_join.apex_neq, apex_join.reference_neq, equal_nan=True)).sum()),
        checks,
    )
    check(
        "apex_q3_q8_rejoined_exactly",
        bool((apex_join.apex_q3 == apex_join.reference_q3).all())
        and bool((apex_join.apex_q8 == apex_join.reference_q8).all()),
        {
            "q3_mismatches": int((apex_join.apex_q3 != apex_join.reference_q3).sum()),
            "q8_mismatches": int((apex_join.apex_q8 != apex_join.reference_q8).sum()),
        },
        checks,
    )
    check(
        "neq_flexible_label_consistent",
        bool((residue.flexible_neq_gt1.astype(bool) == (residue.neq > 1.0)).all()),
        "Neq > 1.0",
        checks,
    )
    check(
        "q3_values_valid",
        set(residue.q3.unique()).issubset(set("HEC")),
        sorted(residue.q3.unique().tolist()),
        checks,
    )
    check(
        "q8_values_valid",
        set(residue.q8.unique()).issubset(set("GHIBESTC")),
        sorted(residue.q8.unique().tolist()),
        checks,
    )

    n_permutations = int(parameters["n_block_shifts"])
    metric_count = len(parameters["metrics"])
    strata_count = scoped_annotated[
        ["condition", "split", "sign", "label"]
    ].drop_duplicates().shape[0]
    expected_summary_rows = strata_count * metric_count
    expected_null_rows = expected_summary_rows * n_permutations
    check(
        "summary_complete",
        len(summary) == expected_summary_rows,
        {"observed": len(summary), "expected": expected_summary_rows},
        checks,
    )
    check(
        "null_complete",
        len(null) == expected_null_rows,
        {"observed": len(null), "expected": expected_null_rows},
        checks,
    )
    null_counts = null.groupby(["condition", "split", "sign", "label", "metric"]).permutation.nunique()
    check(
        "every_test_has_all_permutations",
        bool((null_counts == n_permutations).all()),
        null_counts.value_counts().to_dict(),
        checks,
    )
    summary_keys = ["condition", "split", "sign", "label", "metric"]
    per_protein_long = per_protein.melt(
        id_vars=["condition", "split", "protein", "sign", "label", "n_bands"],
        value_vars=list(parameters["metrics"]),
        var_name="metric",
        value_name="value",
    )
    recomputed_observed = (
        per_protein_long.groupby(summary_keys, as_index=False).value.mean()
        .rename(columns={"value": "recomputed_observed"})
    )
    recomputed_null = (
        null.groupby(summary_keys, as_index=False).macro_protein_mean.mean()
        .rename(columns={"macro_protein_mean": "recomputed_null"})
    )
    summary_check = summary.merge(
        recomputed_observed, on=summary_keys, how="left", validate="one_to_one"
    ).merge(recomputed_null, on=summary_keys, how="left", validate="one_to_one")
    check(
        "summary_observed_and_null_means_recompute",
        bool(np.allclose(
            summary_check.observed_macro_protein_mean,
            summary_check.recomputed_observed,
            equal_nan=True,
        )) and bool(np.allclose(
            summary_check.null_mean,
            summary_check.recomputed_null,
            equal_nan=True,
        )),
        "recomputed from per-protein and per-permutation tables",
        checks,
    )
    p_q_columns = [column for column in summary.columns if column.startswith("p_")]
    p_q_columns += [column for column in summary.columns if column.endswith("_q_bh")]
    in_range = all(
        summary[column].dropna().between(0, 1).all() for column in sorted(set(p_q_columns))
    )
    check("summary_p_q_values_in_unit_interval", in_range, sorted(set(p_q_columns)), checks)
    contrast_pq = [c for c in contrasts if c.startswith("p_") or c.endswith("_q_bh")]
    check(
        "contrast_p_q_values_in_unit_interval",
        all(contrasts[c].dropna().between(0, 1).all() for c in contrast_pq),
        contrast_pq,
        checks,
    )
    check(
        "per_protein_keys_unique",
        not per_protein.duplicated(["condition", "split", "protein", "sign", "label"]).any(),
        int(per_protein.duplicated(["condition", "split", "protein", "sign", "label"]).sum()),
        checks,
    )
    check(
        "annotation_internal_audit_passed",
        bool(annotation_audit["all_band_coordinates_and_apex_amino_acids_validated"])
        and bool(annotation_audit["band_interval_counts_equal_band_width"]),
        annotation_audit.get("strain_note"),
        checks,
    )

    # Phase 1B/1C: reverse the conditioning direction (annotation -> band).
    phase1_key = [
        "condition", "split", "protein", "sign", "target", "unit_type",
        "detection_method",
    ]
    n_phase1_methods_per_sign = 9  # Q8 residue 2 + Neq residue 5 + Q8 segment 2.
    expected_phase1_rows = (
        len(scoped_protein_summary) * 2 * n_phase1_methods_per_sign
    )
    if strain_requested and not strain_audit.empty:
        ok_proteins = set(
            strain_audit.loc[strain_audit.strain_status == "ok", "protein"].astype(str)
        )
        strain_contexts = scoped_protein_summary[
            (scoped_protein_summary["split"].astype(str) == "test")
            & scoped_protein_summary["protein"].astype(str).isin(ok_proteins)
        ]
        expected_phase1_rows += len(strain_contexts) * 2 * 5
    check(
        "phase1_all_proteins_and_methods_present",
        len(phase1_protein) == expected_phase1_rows,
        {"observed": len(phase1_protein), "expected": expected_phase1_rows},
        checks,
    )
    check(
        "phase1_keys_unique",
        not phase1_protein.duplicated(phase1_key).any(),
        int(phase1_protein.duplicated(phase1_key).sum()),
        checks,
    )
    if strain_requested:
        check(
            "strain_audit_present",
            not strain_audit.empty,
            strain_parameters.get("audit"),
            checks,
        )
        invalid_statuses = (
            strain_audit.loc[
                ~strain_audit.strain_status.isin(["ok", "missing"]),
                ["protein", "strain_status"],
            ].to_dict("records")
            if not strain_audit.empty else []
        )
        check(
            "strain_files_have_valid_schema_and_exact_indices",
            len(invalid_statuses) == 0,
            invalid_statuses[:20],
            checks,
        )
        strain_metrics = [
            metric for metric in parameters["metrics"] if metric.startswith("strain_")
            or metric.startswith("distance_to_strain_")
        ]
        non_test_strain = per_protein.loc[
            per_protein["split"].astype(str) != "test", strain_metrics
        ] if strain_metrics else pd.DataFrame()
        check(
            "strain_is_test_only",
            non_test_strain.empty or bool(non_test_strain.isna().all().all()),
            {"strain_metrics": strain_metrics},
            checks,
        )
    expected_contexts = set(map(
        tuple,
        scoped_protein_summary[["condition", "split", "protein"]].to_numpy(),
    ))
    observed_contexts = set(map(tuple, phase1_protein[["condition", "split", "protein"]].drop_duplicates().to_numpy()))
    check(
        "phase1_zero_band_proteins_retained",
        observed_contexts == expected_contexts,
        {
            "expected_contexts": len(expected_contexts),
            "observed_contexts": len(observed_contexts),
        },
        checks,
    )
    phase1_counts_valid = bool(
        (phase1_protein.n_detected_target_units <= phase1_protein.n_target_units).all()
    )
    expected_coverage = (
        phase1_protein.n_detected_target_units
        / phase1_protein.n_target_units.replace(0, np.nan)
    )
    phase1_counts_valid &= bool(np.allclose(
        phase1_protein.coverage, expected_coverage, equal_nan=True
    ))
    residue_phase1 = phase1_protein[phase1_protein.unit_type == "residue"].copy()
    phase1_counts_valid &= bool((residue_phase1.tp + residue_phase1.fn == residue_phase1.n_target_units).all())
    phase1_counts_valid &= bool((residue_phase1.tp == residue_phase1.n_detected_target_units).all())
    phase1_counts_valid &= bool((residue_phase1.tp + residue_phase1.fp == residue_phase1.n_predicted_units).all())
    phase1_denominator = (
        residue_phase1.analysis_residue_count
        if "analysis_residue_count" in residue_phase1
        else residue_phase1.eligible_residue_count
    )
    phase1_counts_valid &= bool((
        residue_phase1.tp + residue_phase1.fp + residue_phase1.fn + residue_phase1.tn
        == phase1_denominator
    ).all())
    check(
        "phase1_classification_counts_and_coverage_recompute",
        phase1_counts_valid,
        "TP/FP/FN/TN, detection counts, and eligible denominators",
        checks,
    )
    phase1_group = [
        "condition", "split", "sign", "label", "target", "unit_type",
        "detection_method",
    ]
    phase1_recomputed = residue_phase1.groupby(phase1_group, as_index=False).agg(
        total_target_units=("n_target_units", "sum"),
        total_detected_target_units=("n_detected_target_units", "sum"),
        micro_tp=("tp", "sum"),
        micro_fp=("fp", "sum"),
        micro_fn=("fn", "sum"),
        micro_tn=("tn", "sum"),
    )
    phase1_summary_check = phase1_summary[phase1_summary.unit_type == "residue"].merge(
        phase1_recomputed, on=phase1_group, how="left", validate="one_to_one",
        suffixes=("", "_recomputed"),
    )
    count_names = [
        "total_target_units", "total_detected_target_units",
        "micro_tp", "micro_fp", "micro_fn", "micro_tn",
    ]
    phase1_summary_valid = all(
        np.allclose(
            phase1_summary_check[name],
            phase1_summary_check[f"{name}_recomputed"],
            equal_nan=True,
        )
        for name in count_names
    )
    check(
        "phase1_summary_counts_recompute",
        phase1_summary_valid,
        count_names,
        checks,
    )
    # For an exact-apex detector, precision P(annotation | band apex) must equal
    # the original Phase 1A apex metric. This catches a reversed numerator or label.
    direction_pairs = {
        "q8_loop_turn_bend_CTS": "q8_loop_turn_bend_CTS",
        "neq_peak": "neq_peak",
    }
    direction_rows = phase1_summary[
        (phase1_summary.unit_type == "residue")
        & (phase1_summary.detection_method == "apex_exact")
        & phase1_summary.target.isin(direction_pairs)
    ].copy()
    direction_rows["metric"] = direction_rows.target.map(direction_pairs)
    direction_check = direction_rows.merge(
        summary[[
            "condition", "split", "sign", "label", "metric",
            "observed_macro_protein_mean",
        ]],
        on=["condition", "split", "sign", "label", "metric"],
        how="left",
        validate="one_to_one",
    )
    check(
        "phase1_precision_direction_matches_apex_enrichment",
        bool(np.allclose(
            direction_check.macro_precision,
            direction_check.observed_macro_protein_mean,
            equal_nan=True,
        )),
        "P(annotation | apex) independently agrees with Phase 1A",
        checks,
    )
    probability_metric_names = (
        "coverage", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "target_prevalence",
    )
    phase1_probability_columns = [
        f"{prefix}_{metric}"
        for prefix in ("micro", "macro")
        for metric in probability_metric_names
        if f"{prefix}_{metric}" in phase1_summary.columns
    ]
    check(
        "phase1_probability_metrics_in_unit_interval",
        all(phase1_summary[c].dropna().between(0, 1).all() for c in phase1_probability_columns),
        phase1_probability_columns,
        checks,
    )

    # Phase 2: exact-Q3, within-protein, non-band matched controls.
    expected_schemes = set(
        parameters["phase2_within_q3_matching"]["selected_match_schemes"]
    )
    check(
        "phase2_nested_match_schemes_complete",
        set(match_coverage.match_scheme.unique()) == expected_schemes,
        sorted(match_coverage.match_scheme.unique().tolist()),
        checks,
    )
    match_key = ["match_scheme", "band_id", "control_rank"]
    check(
        "phase2_matched_control_keys_unique",
        not matched_controls.duplicated(match_key).any(),
        int(matched_controls.duplicated(match_key).sum()),
        checks,
    )
    reference = residue[[
        "split", "protein", "residue_index_0based", "q3",
    ]]
    control_reference = reference.rename(columns={
        "residue_index_0based": "control_index_0based", "q3": "reference_control_q3",
    })
    case_reference = reference.rename(columns={
        "residue_index_0based": "case_index_0based", "q3": "reference_case_q3",
    })
    q3_check = matched_controls.merge(
        control_reference,
        on=["split", "protein", "control_index_0based"],
        how="left",
        validate="many_to_one",
    ).merge(
        case_reference,
        on=["split", "protein", "case_index_0based"],
        how="left",
        validate="many_to_one",
    )
    check(
        "phase2_exact_q3_match_rejoins_independently",
        bool((q3_check.reference_control_q3 == q3_check.q3).all())
        and bool((q3_check.reference_case_q3 == q3_check.q3).all()),
        {
            "control_mismatches": int((q3_check.reference_control_q3 != q3_check.q3).sum()),
            "case_mismatches": int((q3_check.reference_case_q3 != q3_check.q3).sum()),
        },
        checks,
    )
    # Reconstruct excluded masks from the original band intervals and verify
    # that no selected control is itself inside a band of either sign.
    band_groups = {
        key: group for key, group in annotated.groupby(
            ["condition", "split", "protein"], sort=False
        )
    }
    length_lookup = {
        (str(split), str(protein)): int(length)
        for (split, protein), length in residue.groupby(["split", "protein"]).size().items()
    }
    phase2_parameters = parameters["phase2_within_q3_matching"]
    terminal_exclusion = int(
        phase2_parameters.get("terminal_exclusion_residues_per_end", 0)
    )
    case_keys = pd.MultiIndex.from_frame(
        matched_cases[["split", "protein"]].astype(str)
    )
    control_keys = pd.MultiIndex.from_frame(
        matched_controls[["split", "protein"]].astype(str)
    )
    case_lengths = case_keys.map(length_lookup).to_numpy(dtype=int)
    control_lengths = control_keys.map(length_lookup).to_numpy(dtype=int)
    matched_nonterminal = bool(
        matched_cases.case_index_0based.ge(terminal_exclusion).all()
        and matched_cases.case_index_0based.lt(
            case_lengths - terminal_exclusion
        ).all()
        and matched_controls.case_index_0based.ge(terminal_exclusion).all()
        and matched_controls.case_index_0based.lt(
            control_lengths - terminal_exclusion
        ).all()
        and matched_controls.control_index_0based.ge(terminal_exclusion).all()
        and matched_controls.control_index_0based.lt(
            control_lengths - terminal_exclusion
        ).all()
    )
    check(
        "phase2_matched_cases_and_controls_nonterminal",
        matched_nonterminal,
        {"terminal_exclusion_residues_per_end": terminal_exclusion},
        checks,
    )
    controls_inside_bands = 0
    for key, group in matched_controls.groupby(
        ["condition", "split", "protein"], sort=False
    ):
        condition, split, protein = key
        length = int(length_lookup[(split, protein)])
        excluded = np.zeros(length, dtype=bool)
        for band in band_groups[key].itertuples(index=False):
            left = max(0, int(band.start_index_0based) - int(parameters["phase2_within_q3_matching"]["non_band_buffer_residues"]))
            right = min(
                length,
                int(band.end_index_0based_inclusive) + 1
                + int(parameters["phase2_within_q3_matching"]["non_band_buffer_residues"]),
            )
            excluded[left:right] = True
        controls_inside_bands += int(np.sum(
            excluded[group.control_index_0based.astype(int).to_numpy()]
        ))
    check(
        "phase2_controls_outside_all_band_intervals",
        controls_inside_bands == 0,
        controls_inside_bands,
        checks,
    )
    calipers = phase2_parameters["calipers"]
    neq_rows = matched_controls.match_scheme != "q3_only"
    neq_valid = (
        (matched_controls.loc[neq_rows, "control_neq"] - matched_controls.loc[neq_rows, "case_neq"]).abs()
        <= float(calipers["neq"]) + 1e-12
    ).all()
    rsa_rows = matched_controls.match_scheme.str.contains("rsa")
    rsa_valid = (
        (matched_controls.loc[rsa_rows, "control_rsa"] - matched_controls.loc[rsa_rows, "case_rsa"]).abs()
        <= float(calipers["rsa"]) + 1e-12
    ).all()
    position_rows = matched_controls.match_scheme.str.endswith("position")
    position_valid = (
        (
            matched_controls.loc[position_rows, "control_normalized_position"]
            - matched_controls.loc[position_rows, "case_normalized_position"]
        ).abs()
        <= float(calipers["normalized_sequence_position"]) + 1e-12
    ).all()
    check(
        "phase2_all_matching_calipers_respected",
        bool(neq_valid and rsa_valid and position_valid),
        {"neq": bool(neq_valid), "rsa": bool(rsa_valid), "position": bool(position_valid)},
        checks,
    )
    capped_rows = matched_controls.match_scheme != "q3_only"
    capped_ranks = matched_controls.loc[capped_rows, "control_rank"]
    capped_ranks_valid = capped_ranks.between(
        1, int(phase2_parameters["controls_per_apex_maximum"])
    ).all()
    check(
        "phase2_caliper_scheme_control_ranks_within_configured_maximum",
        bool(capped_ranks_valid),
        int(capped_ranks.max()) if len(capped_ranks) else None,
        checks,
    )
    q3_only_cases = matched_cases[matched_cases.match_scheme == "q3_only"].copy()
    if "q3_only" in expected_schemes:
        residue_group_lookup = {
            key: group.sort_values("residue_index_0based").reset_index(drop=True)
            for key, group in residue.groupby(["split", "protein"], sort=False)
        }
        eligibility_lookup = protein_summary.set_index(
            ["condition", "split", "protein"]
        )[["eligible_start_index_0based", "eligible_end_index_0based_exclusive"]]
        expected_control_values: dict[tuple[str, str, str, str], tuple] = {}
        for key in q3_only_cases[[
            "condition", "split", "protein"
        ]].drop_duplicates().itertuples(index=False, name=None):
            condition, split, protein = map(str, key)
            annotation = residue_group_lookup[(split, protein)]
            q3_values = annotation.q3.astype(str).to_numpy()
            neq_values = pd.to_numeric(annotation.neq, errors="coerce").to_numpy(float)
            rsa_values = pd.to_numeric(annotation.rsa, errors="coerce").to_numpy(float)
            length = len(q3_values)
            interval = eligibility_lookup.loc[(condition, split, protein)]
            eligible = np.zeros(length, dtype=bool)
            eligible[
                int(interval.eligible_start_index_0based):
                int(interval.eligible_end_index_0based_exclusive)
            ] = True
            if terminal_exclusion:
                eligible[:terminal_exclusion] = False
                eligible[-terminal_exclusion:] = False
            positions = np.arange(length)
            start = int(interval.eligible_start_index_0based)
            end = int(interval.eligible_end_index_0based_exclusive)
            normalized_position = (positions - start) / max(1, end - start - 1)
            excluded = np.zeros(length, dtype=bool)
            for band in band_groups[(condition, split, protein)].itertuples(index=False):
                left = max(
                    0,
                    int(band.start_index_0based)
                    - int(phase2_parameters["non_band_buffer_residues"]),
                )
                right = min(
                    length,
                    int(band.end_index_0based_inclusive) + 1
                    + int(phase2_parameters["non_band_buffer_residues"]),
                )
                excluded[left:right] = True
            control_mask = eligible & ~excluded
            for q3_label in np.unique(q3_values):
                selected = control_mask & (q3_values == q3_label)
                selected_count = int(np.sum(selected))
                expected_control_values[(condition, split, protein, q3_label)] = (
                    selected_count,
                    float(np.mean(neq_values[selected])) if selected_count else np.nan,
                    float(np.mean(rsa_values[selected])) if selected_count else np.nan,
                    (
                        float(np.mean(normalized_position[selected]))
                        if selected_count else np.nan
                    ),
                )
        expected_rows = [
            expected_control_values[(
                str(row.condition), str(row.split), str(row.protein), str(row.q3)
            )]
            for row in q3_only_cases.itertuples(index=False)
        ]
        expected_array = np.asarray(expected_rows, dtype=float)
        count_valid = np.array_equal(
            q3_only_cases.n_controls.to_numpy(int), expected_array[:, 0].astype(int)
        )
        means_valid = all((
            np.allclose(q3_only_cases.control_mean_neq, expected_array[:, 1]),
            np.allclose(q3_only_cases.control_mean_rsa, expected_array[:, 2]),
            np.allclose(
                q3_only_cases.control_mean_normalized_position,
                expected_array[:, 3],
            ),
        ))
        expected_matched_cases = int(
            match_coverage.loc[
                match_coverage.match_scheme == "q3_only", "n_matched_cases"
            ].sum()
        )
        row_count_valid = len(q3_only_cases) == expected_matched_cases
        complete_q3_only = bool(count_valid and means_valid and row_count_valid)
        incomplete_q3_only = int(np.sum(
            q3_only_cases.n_controls.to_numpy(int) != expected_array[:, 0].astype(int)
        ))
    else:
        complete_q3_only = True
        incomplete_q3_only = 0
    check(
        "phase2_q3_only_uses_all_eligible_same_q3_controls",
        complete_q3_only,
        {
            "incomplete_band_sets": incomplete_q3_only,
            "matched_case_rows": len(q3_only_cases),
        },
        checks,
    )
    match_coverage_valid = bool((match_coverage.n_matched_cases <= match_coverage.n_cases).all())
    match_coverage_valid &= bool(np.allclose(
        match_coverage.match_rate,
        match_coverage.n_matched_cases / match_coverage.n_cases,
        equal_nan=True,
    ))
    match_coverage_valid &= bool((match_coverage.n_controls_selected >= match_coverage.n_matched_cases).all())
    check(
        "phase2_match_coverage_recomputes",
        match_coverage_valid,
        "matched cases, selected controls, and rates",
        checks,
    )
    matched_key = [
        "match_scheme", "condition", "split", "protein", "sign", "label",
        "q3", "metric",
    ]
    check(
        "phase2_per_protein_effect_keys_unique",
        not matched_protein.duplicated(matched_key).any(),
        int(matched_protein.duplicated(matched_key).sum()),
        checks,
    )
    matched_group = [
        "match_scheme", "condition", "split", "sign", "label", "q3", "metric",
    ]
    matched_recomputed = matched_protein.groupby(matched_group, as_index=False).agg(
        recomputed_case=("case_mean", "mean"),
        recomputed_control=("matched_control_mean", "mean"),
        recomputed_difference=("case_minus_control", "mean"),
    )
    matched_summary_check = matched_summary.merge(
        matched_recomputed, on=matched_group, how="left", validate="one_to_one"
    )
    matched_summary_valid = bool(np.allclose(
        matched_summary_check.case_macro_protein_mean,
        matched_summary_check.recomputed_case,
        equal_nan=True,
    )) and bool(np.allclose(
        matched_summary_check.matched_control_macro_protein_mean,
        matched_summary_check.recomputed_control,
        equal_nan=True,
    )) and bool(np.allclose(
        matched_summary_check.case_minus_control_macro_mean,
        matched_summary_check.recomputed_difference,
        equal_nan=True,
    ))
    check(
        "phase2_summary_effects_recompute",
        matched_summary_valid,
        "equal-protein case, control, and paired difference means",
        checks,
    )
    matched_pq = [
        c for c in matched_summary.columns if c.startswith("p_") or c.endswith("_q_bh")
    ]
    check(
        "phase2_p_q_values_in_unit_interval",
        all(matched_summary[c].dropna().between(0, 1).all() for c in matched_pq),
        matched_pq,
        checks,
    )
    primary_valid = (
        {"primary_p_two_sided", "primary_q_bh"}.issubset(group_inference.columns)
        and len(group_inference) == 48
        and set(
            map(tuple, group_inference[["match_scheme", "metric"]].drop_duplicates().to_numpy())
        ) == {
            ("q3_neq", "rsa"),
            ("q3_neq_rsa", "torsion_change_from_previous"),
            ("q3_neq_rsa", "distance_to_q3_boundary"),
            ("q3_neq_rsa", "strain_ensemble_mean"),
        }
        and group_inference["primary_p_two_sided"].dropna().between(0, 1).all()
        and group_inference["primary_q_bh"].dropna().between(0, 1).all()
        and all(
            "primary_p_two_sided" not in frame.columns
            and "primary_q_bh" not in frame.columns
            for frame in (summary, contrasts, matched_summary)
        )
    )
    check(
        "one_two_sided_primary_pvalue_family",
        bool(primary_valid),
        "48 union-group headline tests; other tables are diagnostic",
        checks,
    )
    forbidden = (
        (matched_summary.metric.eq("neq") & matched_summary.match_scheme.ne("q3_only"))
        | (
            matched_summary.metric.eq("rsa")
            & matched_summary.match_scheme.str.contains("rsa")
        )
        | (
            matched_summary.metric.eq("normalized_position")
            & matched_summary.match_scheme.str.endswith("position")
        )
    )
    check(
        "matched_covariates_absent_from_inferential_outcomes",
        not bool(forbidden.any()),
        int(forbidden.sum()),
        checks,
    )
    expected_matched_on = (
        ((match_balance.covariate == "neq") & (match_balance.match_scheme != "q3_only"))
        | ((match_balance.covariate == "rsa") & match_balance.match_scheme.str.contains("rsa"))
        | (
            (match_balance.covariate == "normalized_position")
            & match_balance.match_scheme.str.endswith("position")
        )
    )
    check(
        "phase2_balance_labels_match_schemes",
        bool((match_balance.matched_on_covariate.astype(bool) == expected_matched_on).all()),
        int((match_balance.matched_on_covariate.astype(bool) != expected_matched_on).sum()),
        checks,
    )

    failures = [item for item in checks if not item["passed"]]
    report = {
        "passed": len(failures) == 0,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "analysis_scope": {"splits": sorted(requested_splits)},
        "checks": checks,
        "strain_status": (
            strain_audit.strain_status.value_counts().to_dict()
            if strain_requested and not strain_audit.empty
            else annotation_audit.get("strain_note")
        ),
        "row_counts": {
            "original_bands": len(original),
            "annotated_bands": len(annotated),
            "residue_annotations": len(residue),
            "per_protein_metrics": len(per_protein),
            "enrichment_summary": len(summary),
            "null": len(null),
            "paired_contrasts": len(contrasts),
            "phase1_by_protein": len(phase1_protein),
            "phase1_summary": len(phase1_summary),
            "matched_controls": len(matched_controls),
            "matched_cases": len(matched_cases),
            "match_coverage": len(match_coverage),
            "match_balance": len(match_balance),
            "matched_effects_by_protein": len(matched_protein),
            "matched_summary": len(matched_summary),
            "headline_union_group_inference": len(group_inference),
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "checks": report["n_checks"],
        "failures": report["n_failures"],
        "report": str(output),
    }, indent=2))
    if failures:
        for failure in failures:
            print(json.dumps(failure, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
