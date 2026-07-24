#!/usr/bin/env python3
"""Independently audit Phase 4 upgraded feature and receiver outputs.

This script is intended to run after feature construction and again after the
receiver aggregation. It does not regenerate scientific features; it checks
identity alignment, missing-data semantics, geometric/path invariants, and
fixed-cohort model reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument(
        "--receiver_output_dir",
        nargs="*",
        default=None,
        help="Optional one or more upgraded per-condition receiver output dirs",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--maximum_failures_to_store", type=int, default=200)
    return parser.parse_args()


class Audit:
    def __init__(self, maximum_failures: int):
        self.checks = 0
        self.failures: list[dict] = []
        self.maximum_failures = maximum_failures

    def check(self, passed: bool, check: str, context: str, detail: str = "") -> None:
        self.checks += 1
        if not passed and len(self.failures) < self.maximum_failures:
            self.failures.append({
                "check": check,
                "context": context,
                "detail": detail,
            })


def all_nan(values: np.ndarray) -> bool:
    return bool(np.isnan(np.asarray(values, dtype=float)).all())


def audit_feature_partition(
    path: Path,
    expected: pd.DataFrame,
    minimum_band_fraction: float,
    audit: Audit,
) -> int:
    context = str(path)
    if not path.exists():
        audit.check(False, "feature_partition_exists", context)
        return 0
    with np.load(path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    audit.check(
        str(data.get("schema_version", [""])[0])
        == "esmfluc.phase4_structural_water.v1",
        "schema_version",
        context,
    )
    expected_ids = expected.sort_values("band_id").band_id.astype(str).to_numpy()
    audit.check(
        np.array_equal(data["band_id"].astype(str), expected_ids),
        "band_identity_and_order",
        context,
    )
    length = int(data["protein_length"][0])
    n_bands = len(expected_ids)
    pair_names = data["pair_feature_names"].astype(str)
    band_names = data["band_feature_names"].astype(str)
    for name in pair_names:
        audit.check(
            data[name].shape == (n_bands, length),
            "pair_feature_shape",
            f"{context}:{name}",
            f"observed={data[name].shape}, expected={(n_bands, length)}",
        )
    for name in band_names:
        audit.check(
            data[name].shape == (n_bands,),
            "band_feature_shape",
            f"{context}:{name}",
        )
    if not {"query_resolved", "pair_structure_eligible"} <= set(data):
        audit.check(False, "required_eligibility_features", context)
        return n_bands * length

    resolved = data["query_resolved"].astype(float)
    eligible = data["pair_structure_eligible"].astype(float)
    fractions = data["band_resolved_fraction"].astype(float)
    expected_eligible = (
        (resolved == 1)
        & (fractions[:, None] >= minimum_band_fraction)
    )
    audit.check(
        np.array_equal(eligible == 1, expected_eligible),
        "structure_eligibility_definition",
        context,
    )
    unresolved = resolved == 0
    non_indicator_features = [
        name for name in pair_names
        if name not in {
            "query_resolved", "pair_structure_eligible",
            "water_structure_eligible",
        }
    ]
    for name in non_indicator_features:
        audit.check(
            all_nan(data[name][unresolved]),
            "unresolved_queries_are_missing",
            f"{context}:{name}",
        )

    if {"direct_ca_contact", "query_to_band_contact_count"} <= set(data):
        direct = data["direct_ca_contact"].astype(float)
        count = data["query_to_band_contact_count"].astype(float)
        valid = np.isfinite(direct) & np.isfinite(count)
        audit.check(
            np.array_equal(direct[valid] == 1, count[valid] >= 1),
            "direct_ca_contact_matches_count",
            context,
        )
        band_sizes = np.broadcast_to(
            data["band_resolved_residue_count"][:, None],
            count.shape,
        )
        audit.check(
            bool((count[valid] <= band_sizes[valid]).all())
            if valid.any() else True,
            "contact_count_not_above_resolved_band_size",
            context,
        )
    if {
        "minimum_ca_distance_angstrom",
        "shortest_ordinary_contact_path_edges",
    } <= set(data):
        ordered = expected.sort_values("band_id").reset_index(drop=True)
        minimum_ca = data["minimum_ca_distance_angstrom"].astype(float)
        ordinary_path = data["shortest_ordinary_contact_path_edges"].astype(float)
        for band_number, band in enumerate(ordered.itertuples(index=False)):
            inside = np.zeros(length, dtype=bool)
            inside[
                int(band.start_index_0based):
                int(band.end_index_0based_inclusive) + 1
            ] = True
            valid_inside = inside & (eligible[band_number] == 1)
            audit.check(
                bool(np.allclose(minimum_ca[band_number, valid_inside], 0.0))
                if valid_inside.any() else True,
                "resolved_inside_band_ca_distance_is_zero",
                f"{context}:band={band.band_id}",
            )
            audit.check(
                bool(np.allclose(ordinary_path[band_number, valid_inside], 0.0))
                if valid_inside.any() else True,
                "resolved_inside_band_ordinary_path_is_zero",
                f"{context}:band={band.band_id}",
            )
    for prefix in ("ordinary", "nonlocal"):
        exists_name = f"{prefix}_contact_path_exists"
        length_name = f"shortest_{prefix}_contact_path_edges"
        if {exists_name, length_name} <= set(data):
            exists = data[exists_name].astype(float)
            path_length = data[length_name].astype(float)
            valid = np.isfinite(exists)
            audit.check(
                all_nan(path_length[valid & (exists == 0)]),
                "absent_contact_path_length_is_missing",
                f"{context}:{prefix}",
            )
            audit.check(
                bool(np.isfinite(path_length[valid & (exists == 1)]).all()),
                "present_contact_path_has_length",
                f"{context}:{prefix}",
            )

    if {"direct_putative_polar_contact", "direct_putative_polar_contact_count"} <= set(data):
        direct = data["direct_putative_polar_contact"].astype(float)
        count = data["direct_putative_polar_contact_count"].astype(float)
        valid = np.isfinite(direct) & np.isfinite(count)
        audit.check(
            np.array_equal(direct[valid] == 1, count[valid] >= 1),
            "direct_polar_contact_matches_count",
            context,
        )

    required_water = {
        "water_structure_eligible", "shared_water_count", "one_water_bridge",
        "water_path_exists", "shortest_water_path_water_count",
        "water_path_at_most_1", "water_path_at_most_2",
        "water_path_at_most_3",
    }
    if required_water <= set(data):
        water_eligible = data["water_structure_eligible"].astype(float) == 1
        shared = data["shared_water_count"].astype(float)
        bridge = data["one_water_bridge"].astype(float)
        path_exists = data["water_path_exists"].astype(float)
        shortest = data["shortest_water_path_water_count"].astype(float)
        valid = water_eligible & np.isfinite(shared) & np.isfinite(bridge)
        audit.check(
            np.array_equal(shared[valid] >= 1, bridge[valid] == 1),
            "one_water_bridge_matches_shared_water",
            context,
        )
        valid_path = water_eligible & np.isfinite(path_exists)
        audit.check(
            all_nan(shortest[valid_path & (path_exists == 0)]),
            "absent_water_path_length_is_missing",
            context,
        )
        audit.check(
            bool(np.isfinite(shortest[valid_path & (path_exists == 1)]).all()),
            "present_water_path_has_length",
            context,
        )
        audit.check(
            bool(
                (
                    shortest[
                        valid_path
                        & (path_exists == 1)
                    ] >= 1
                ).all()
            ),
            "water_path_uses_at_least_one_water",
            context,
        )
        for maximum in (1, 2, 3):
            indicator = data[f"water_path_at_most_{maximum}"].astype(float)
            valid = water_eligible & np.isfinite(indicator)
            expected_indicator = (
                np.isfinite(shortest[valid]) & (shortest[valid] <= maximum)
            )
            audit.check(
                np.array_equal(indicator[valid] == 1, expected_indicator),
                "water_path_threshold_indicator",
                f"{context}:at_most_{maximum}",
            )
        for name in (
            "shared_water_fraction_of_query_contacts",
            "shared_water_fraction_of_band_contacts",
        ):
            if name in data:
                values = data[name].astype(float)
                valid = water_eligible & np.isfinite(values)
                audit.check(
                    bool(((values[valid] >= 0) & (values[valid] <= 1)).all()),
                    "water_fraction_in_unit_interval",
                    f"{context}:{name}",
                )
    return n_bands * length


def audit_receiver_output(path: Path, audit: Audit) -> None:
    context = str(path)
    required = {
        "receiver_model_performance.csv",
        "receiver_feature_summary.csv",
        "receiver_structural_water_ablation_performance.csv",
        "parameters.json",
    }
    for name in required:
        audit.check((path / name).exists(), "receiver_output_exists", f"{context}:{name}")
    performance_path = path / "receiver_model_performance.csv"
    parameters_path = path / "parameters.json"
    if not performance_path.exists() or not parameters_path.exists():
        return
    performance = pd.read_csv(performance_path)
    expected_columns = {
        "model_cohort", "receiver_scope", "stage", "evaluation_split",
        "auroc", "n_proteins",
    }
    audit.check(
        expected_columns <= set(performance),
        "receiver_performance_schema",
        context,
        f"missing={sorted(expected_columns - set(performance))}",
    )
    if expected_columns <= set(performance):
        audit.check(
            set(performance.evaluation_split.astype(str)) <= {"validation", "test"},
            "held_out_evaluation_only",
            context,
        )
        water = performance[
            performance.model_cohort.astype(str).eq("water_eligible_fixed")
        ]
        if not water.empty:
            stages = set(water.stage.astype(str))
            audit.check(
                {
                    "00_distance_q8",
                    "01_add_query_biophysics",
                    "06_add_crystallographic_water_network",
                } <= stages,
                "water_cohort_has_fixed_comparison_stages",
                context,
            )
            counts = water.groupby(
                [
                    "condition", "sign", "receiver_scope",
                    "evaluation_split",
                ],
                dropna=False,
            ).n_pairs.nunique()
            audit.check(
                bool((counts == 1).all()),
                "fixed_cohort_pair_count_constant_across_stages",
                context,
            )
    parameters = json.loads(parameters_path.read_text())
    audit.check(
        bool(parameters.get("pairwise_structure_source")),
        "receiver_parameters_record_feature_source",
        context,
    )


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    parameters = json.loads((feature_dir / "parameters.json").read_text())
    minimum_band_fraction = float(
        parameters["quality"]["min_band_resolved_fraction"]
    )
    bands = pd.read_csv(args.bands_csv).sort_values(
        ["condition", "split", "protein", "band_id"]
    )
    manifest_path = feature_dir / "pair_feature_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    audit = Audit(args.maximum_failures_to_store)
    expected_lookup = {
        key: group
        for key, group in bands.groupby(
            ["condition", "split", "protein"], sort=False
        )
    }
    manifest_keys = set()
    total_pairs = 0
    for row in manifest.itertuples(index=False):
        key = (str(row.condition), str(row.split), str(row.protein))
        manifest_keys.add(key)
        expected = expected_lookup.get(key)
        audit.check(expected is not None, "manifest_context_in_band_table", str(key))
        if expected is not None:
            total_pairs += audit_feature_partition(
                Path(row.feature_npz),
                expected,
                minimum_band_fraction,
                audit,
            )
    accepted_proteins = pd.read_csv(
        feature_dir / "structure_water_input_audit.csv"
    )
    accepted = set(
        accepted_proteins.loc[
            accepted_proteins.mapping_accepted.fillna(False).astype(bool),
            "protein",
        ].astype(str)
    )
    expected_feature_keys = {
        key for key in expected_lookup if key[2] in accepted
    }
    audit.check(
        manifest_keys == expected_feature_keys,
        "feature_manifest_complete_for_accepted_mappings",
        str(feature_dir),
        (
            f"missing={len(expected_feature_keys - manifest_keys)}, "
            f"unexpected={len(manifest_keys - expected_feature_keys)}"
        ),
    )
    for receiver in args.receiver_output_dir or []:
        audit_receiver_output(Path(receiver).expanduser().resolve(), audit)
    result = {
        "passed": not audit.failures,
        "checks": audit.checks,
        "failure_count_stored": len(audit.failures),
        "failures": audit.failures,
        "feature_partitions": len(manifest),
        "band_query_pairs_in_feature_store": total_pairs,
        "receiver_output_dirs_audited": len(args.receiver_output_dir or []),
    }
    destination = Path(args.output_json).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    if audit.failures:
        raise SystemExit(
            f"Phase 4 upgraded audit failed: {len(audit.failures)} stored failures"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
