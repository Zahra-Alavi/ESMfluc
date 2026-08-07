#!/usr/bin/env python3
"""End-to-end integrity audit for the fixed-WT* GK v2 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gk_v2_common import CONDITIONS


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--a176g-summary", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--position-map", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = arguments()
    root = args.analysis
    required = [
        "analysis_complete.json",
        "phase1_wt_star/reproducibility_interval_iou05/stable_signed_bands.csv",
        "uniform_control/reproducibility_interval_iou05/stable_signed_bands.csv",
        "reference_bands/primary_wt_star_bands.csv",
        "reference_bands/mutation_membership_all_conditions.csv",
        "fixed_band_responses/variant_condition_summary.csv",
        "fixed_band_responses/band_response_summary.csv",
        "fixed_band_responses/fixed_band_hub_test.json",
        "receivers/primary_fixed_band_receivers.csv",
        "phenotypes/activity_analysis_table.csv",
        "phenotypes/activity_associations.csv",
        "phenotypes/activity_position_collapsed_sensitivity.csv",
        "robustness/wt_band_overlap_across_conditions.csv",
        "robustness/response_metric_robustness.csv",
        "uniform_control/uniform_control_summary.csv",
        "b1_c1/b1_c1_model_comparison.csv",
        "b1_c1/b1_c1_epistasis.csv",
        "structure/mutation_band_structure_map.csv",
        "structure/view_wt_bands_and_mutations.pml",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GK v2 outputs: {missing}")
    complete = json.loads((root / "analysis_complete.json").read_text())
    extraction = json.loads(args.a176g_summary.read_text())
    bands = pd.read_csv(root / required[1])
    uniform = pd.read_csv(root / required[2])
    references = pd.read_csv(root / "reference_bands/primary_wt_star_bands.csv")
    membership = pd.read_csv(root / "reference_bands/mutation_membership_all_conditions.csv")
    variants = pd.read_csv(root / "fixed_band_responses/variant_condition_summary.csv")
    responses = pd.read_csv(root / "fixed_band_responses/band_response_summary.csv")
    receivers = pd.read_csv(root / "receivers/primary_fixed_band_receivers.csv")
    activity = pd.read_csv(root / "phenotypes/activity_analysis_table.csv")
    associations = pd.read_csv(root / "phenotypes/activity_associations.csv")
    robustness = pd.read_csv(root / "robustness/wt_band_overlap_across_conditions.csv")
    b1c1 = pd.read_csv(root / "b1_c1/b1_c1_model_comparison.csv")
    epistasis = pd.read_csv(root / "b1_c1/b1_c1_epistasis.csv")
    profile_audit = pd.read_csv(root / "phase1_wt_star/profile_preparation_audit.csv")
    params = json.loads((root / "phase1_wt_star/per_seed/signed_band_parameters.json").read_text())
    checks = {
        "a176g_18_runs_complete": extraction.get("completed") == 18 and not extraction.get("failures"),
        "forty_sequence_analysis": complete.get("fasta_records") == 40,
        "six_conditions": set(bands.condition) == set(CONDITIONS),
        "eighteen_wt_profiles_pass": len(profile_audit) == 18 and profile_audit.passes.astype(bool).all(),
        "locked_raw_detector": params.get("detector_version") == "phase1_raw_amplitude_half_intensity_v1",
        "locked_two_mad": params.get("amplitude_mad") == 2.0,
        "locked_half_intensity": params.get("support_method") == "half_intensity_merge",
        "primary_has_11_bands": len(references) == 11,
        "all_conditions_have_stable_bands": bands.groupby("condition").size().reindex(CONDITIONS).notna().all(),
        "uniform_has_no_stable_bands": len(uniform) == 0,
        "mutation_membership_complete": len(membership) == 39 * 6,
        "all_mutations_mapped": membership.paper_position.notna().all(),
        "variant_condition_rows_complete": len(variants) == 39 * 6,
        "band_response_rows_complete": len(responses) == 39 * 6 * 11,
        "receiver_rows_complete": len(receivers) == 39 * 11 * 207,
        "activity_34_complete": len(activity) == 34,
        "six_predeclared_activity_tests": len(associations) == 6,
        "robustness_six_conditions_two_signs": len(robustness) == 12,
        "b1_c1_sequence_identity_correct": set(b1c1.variant) == {"A175G", "A176G", "A175G_A176G"},
        "b1_c1_epistasis_complete": len(epistasis) == 1,
        "decomposition_passes": complete.get("maximum_decomposition_error", 1) <= 1e-6,
        "stable_bands_match_mean_catalog": complete.get(
            "stable_bands_validated_against_mean_catalog"
        ) is True,
    }
    # Pandas reductions can return numpy.bool_, which json cannot serialize.
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "schema": "esmfluc.gk_fixed_wt_bands.audit.v2",
        "passed": not missing and all(checks.values()),
        "checks": checks, "missing_outputs": missing,
        "counts": {
            "all_condition_stable_bands": len(bands),
            "uniform_stable_bands": len(uniform),
            "primary_bands": len(references),
            "mutation_membership_rows": len(membership),
            "variant_condition_rows": len(variants),
            "band_response_rows": len(responses),
            "receiver_rows": len(receivers),
            "activity_rows": len(activity),
        },
        "source_sha256": {
            "fasta": sha256(args.fasta),
            "position_map": sha256(args.position_map),
            "a176g_extraction_summary": sha256(args.a176g_summary),
        },
    }
    audit_path = root / "audit/final_integrity.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise ValueError("GK v2 final integrity audit failed")


if __name__ == "__main__":
    main()
