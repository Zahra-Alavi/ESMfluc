#!/usr/bin/env python3
"""End-to-end integrity audit for the Weinreb mutation-network analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from weinreb_analysis_common import NEW_SEQUENCES, WEINREB_ANALYSIS_ROOT, json_dump


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, default=WEINREB_ANALYSIS_ROOT)
    return p.parse_args()


def rows(path):
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = arguments()
    a = args.analysis
    required = [
        "audit/seed_record_audit.csv", "audit/matrix_integrity_audit.csv",
        "audit/coordinate_conversion_audit.csv", "coordinate_mapping.csv",
        "profiles/delta_I_by_seed.csv", "profiles/delta_I_summary.csv",
        "bands/change_bands_primary.csv", "bands/distal_burden.csv",
        "hotspots/common_hotspots.csv", "hotspots/hotspot_155_157_test.json",
        "decomposition/components_summary.csv", "decomposition/integrity.json",
        "receivers/band_receivers.csv", "receivers/query_routing_summary.csv",
        "receivers/query_mutant_specificity.csv",
        "phenotypes/activity_replicates.csv", "phenotypes/activity_associations.csv",
        "mechanics/rheology_raw_measurements.csv", "mechanics/mechanical_subset_summary.csv",
        "robustness/fixed_band_replication.csv", "robustness/replication_summary.json",
    ]
    missing = [p for p in required if not (a / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")
    build = json.loads((a / "audit/build_summary.json").read_text())
    decomposition = json.loads((a / "decomposition/integrity.json").read_text())
    seed_audit = rows(a / "audit/seed_record_audit.csv")
    matrix_audit = rows(a / "audit/matrix_integrity_audit.csv")
    bands = rows(a / "bands/change_bands_primary.csv")
    receivers = rows(a / "receivers/band_receivers.csv")
    queries = rows(a / "receivers/query_routing_summary.csv")
    query_specificity = rows(a / "receivers/query_mutant_specificity.csv")
    predictors = rows(a / "phenotypes/esm_predictors.csv")
    phenotype = rows(a / "phenotypes/paper_annotations_joined.csv")
    robustness = rows(a / "robustness/fixed_band_replication.csv")

    checks = {
        "three_primary_seeds": len(seed_audit) == 3,
        "39_records_each_seed": all(int(r["observed_records"]) == 39 for r in seed_audit),
        "new_sequences_each_seed": all(r["new_sequences_present"] == "True" for r in seed_audit),
        "all_117_primary_matrices_pass": len(matrix_audit) == 117 and all(r["passes"] == "True" for r in matrix_audit),
        "all_primary_bands_reproducible_2of3": all(int(r["apex_sign_agreement_n"]) >= 2 for r in bands),
        "each_band_has_207_receivers": len(receivers) == len(bands) * 207,
        "all_38_nonreference_variants_have_207_queries": len(queries) == 38 * 207,
        "between_mutant_query_specificity_complete": len(query_specificity) == 38 * 207,
        "three_new_sequences_have_query_analysis": all(any(r["mutant"] == n for r in queries) for n in NEW_SEQUENCES),
        "new_sequences_excluded_from_phenotypes": all(not any(r["mutation"] == n for r in phenotype) for n in NEW_SEQUENCES),
        "decomposition_passes": bool(decomposition["passes"]),
        "robustness_includes_six_conditions": len({r["condition"] for r in robustness}) == 6,
    }
    source_files = [a / "inputs/supplementary_data_1.xlsx", a / "inputs/extended_data_table1.csv",
                    a / "inputs/1ZNW_apo.pdb", a / "inputs/extended_data_table1.jpg"]
    source_hashes = {p.name: sha256(p) for p in source_files if p.is_file()}
    result = {
        "passes": all(checks.values()) and not missing,
        "checks": checks, "missing_outputs": missing,
        "counts": {"primary_bands": len(bands), "receiver_rows": len(receivers),
                   "query_summary_rows": len(queries), "predictor_rows": len(predictors),
                   "phenotype_rows": len(phenotype), "robustness_rows": len(robustness)},
        "numerical_integrity": {"max_matrix_reconstruction_error": build["max_reconstruction_error"],
                                "max_decomposition_error": decomposition["maximum_decomposition_reconstruction_error"]},
        "source_sha256": source_hashes,
    }
    json_dump(a / "audit/final_integrity.json", result)
    if not result["passes"]:
        raise ValueError(f"Final integrity audit failed: {checks}")


if __name__ == "__main__":
    main()
