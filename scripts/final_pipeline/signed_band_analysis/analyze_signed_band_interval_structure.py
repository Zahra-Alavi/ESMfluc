#!/usr/bin/env python3
"""Structural sensitivity analysis using the detected band intervals.

Each stable band is compared with same-protein non-band intervals having the
same width and the same apex-to-boundary offsets. Matching follows the four
nested Phase 2 schemes. Inference uses protein-level sign flips and protein
bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from signed_band_analysis.analyze_signed_band_apex_structure import (
        MATCH_MODELS,
        STRUCTURE_FEATURES,
        band_masks,
        bh,
        prepare_inputs,
    )
    from signed_band_analysis.analyze_signed_band_biophysical_enrichment import (
        bootstrap_mean_ci,
        sign_flip_test,
    )
except ModuleNotFoundError:  # supports direct execution from this directory
    from analyze_signed_band_apex_structure import (  # type: ignore
        MATCH_MODELS,
        STRUCTURE_FEATURES,
        band_masks,
        bh,
        prepare_inputs,
    )
    from analyze_signed_band_biophysical_enrichment import (  # type: ignore
        bootstrap_mean_ci,
        sign_flip_test,
    )


COHORT_THRESHOLDS = {
    "resolved_80": 0.80,
    "fully_resolved": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable_bands_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument("--contact_json", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--detection_threshold_R_p", type=float, default=2.0)
    parser.add_argument("--max_controls_per_apex", type=int, default=5)
    parser.add_argument("--position_caliper", type=float, default=0.25)
    parser.add_argument("--neq_caliper", type=float, default=0.25)
    parser.add_argument("--rsa_caliper", type=float, default=0.15)
    parser.add_argument("--min_mapping_identity", type=float, default=0.90)
    parser.add_argument("--min_input_coverage", type=float, default=0.80)
    parser.add_argument("--min_contact_sequence_separation", type=int, default=3)
    parser.add_argument("--betweenness_samples", type=int, default=64)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--n_sign_flips", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=123)
    args = parser.parse_args()
    for name in (
        "detection_threshold_R_p", "position_caliper", "neq_caliper", "rsa_caliper",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_controls_per_apex < 1:
        parser.error("--max_controls_per_apex must be positive")
    if args.minimum_inference_proteins < 2:
        parser.error("--minimum_inference_proteins must be at least 2")
    if args.n_sign_flips < 1 or args.n_bootstrap < 1:
        parser.error("--n_sign_flips and --n_bootstrap must be positive")
    return args


def prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values)]


def interval_total(prefix: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return prefix[ends] - prefix[starts]


def make_protein_arrays(residues: pd.DataFrame) -> dict[str, dict]:
    output = {}
    for protein, group in residues.groupby("protein", sort=False):
        group = group.sort_values("residue_index_0based").reset_index(drop=True)
        length = int(group.protein_length_structure.iloc[0])
        if len(group) != length or not np.array_equal(
            group.residue_index_0based.to_numpy(int), np.arange(length)
        ):
            raise ValueError(f"{protein}: structure residue table is not input-index complete")
        resolved = group.structure_resolved.fillna(False).to_numpy(bool)
        record = {
            "length": length,
            "q3": group.q3.astype(str).to_numpy(),
            "neq": pd.to_numeric(group.neq, errors="coerce").to_numpy(float),
            "rsa": pd.to_numeric(group.rsa, errors="coerce").to_numpy(float),
            "normalized_position": group.normalized_position.to_numpy(float),
            "resolved_prefix": prefix_sum(resolved.astype(float)),
            "feature_prefix": {},
        }
        for feature in STRUCTURE_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(float)
            finite = np.isfinite(values)
            record["feature_prefix"][feature] = (
                prefix_sum(np.where(finite, values, 0.0)),
                prefix_sum(finite.astype(float)),
            )
        output[str(protein)] = record
    return output


def interval_feature_means(
    record: dict, starts: np.ndarray, ends: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {}
    for feature, (total_prefix, count_prefix) in record["feature_prefix"].items():
        totals = interval_total(total_prefix, starts, ends)
        counts = interval_total(count_prefix, starts, ends)
        output[feature] = np.divide(
            totals, counts, out=np.full(len(starts), np.nan), where=counts > 0
        )
    return output


def band_case_row(band, record: dict) -> dict:
    start = int(band.start_index_0based)
    end = int(band.end_index_0based_inclusive) + 1
    starts, ends = np.array([start]), np.array([end])
    resolved = float(interval_total(record["resolved_prefix"], starts, ends)[0] / (end - start))
    features = interval_feature_means(record, starts, ends)
    row = {
        "band_id": str(band.band_id), "condition": str(band.condition),
        "split": str(band.split), "protein": str(band.protein),
        "sign": int(band.sign), "start_index_0based": start,
        "end_index_0based_inclusive": end - 1,
        "apex_index_0based": int(band.apex_index_0based),
        "band_width": end - start, "resolved_fraction": resolved,
    }
    for feature in STRUCTURE_FEATURES:
        row[f"interval_mean_{feature}"] = float(features[feature][0])
    return row


def candidate_intervals(
    band, record: dict, band_mask: np.ndarray, scheme_name: str,
    minimum_resolved_fraction: float, args,
) -> pd.DataFrame:
    settings = MATCH_MODELS[scheme_name]
    apex = int(band.apex_index_0based)
    left_offset = apex - int(band.start_index_0based)
    right_offset = int(band.end_index_0based_inclusive) - apex
    eligible_start = int(band.eligible_start_index_0based)
    eligible_end = int(band.eligible_end_index_0based_exclusive)
    centers = np.arange(eligible_start + left_offset, eligible_end - right_offset)
    starts = centers - left_offset
    ends = centers + right_offset + 1
    if len(centers) == 0:
        return pd.DataFrame()

    mask_prefix = prefix_sum(band_mask.astype(float))
    valid = interval_total(mask_prefix, starts, ends) == 0
    widths = ends - starts
    resolved_fraction = interval_total(record["resolved_prefix"], starts, ends) / widths
    valid &= resolved_fraction >= minimum_resolved_fraction - 1e-12
    valid &= record["q3"][centers] == str(record["q3"][apex])

    case_neq = float(record["neq"][apex])
    case_rsa = float(record["rsa"][apex])
    case_position = float(record["normalized_position"][apex])
    delta_neq = record["neq"][centers] - case_neq
    delta_rsa = record["rsa"][centers] - case_rsa
    delta_position = record["normalized_position"][centers] - case_position
    distance_squared = np.zeros(len(centers), dtype=float)
    if settings["use_neq"]:
        valid &= np.isfinite(delta_neq) & (np.abs(delta_neq) <= args.neq_caliper)
        distance_squared += (delta_neq / args.neq_caliper) ** 2
    if settings["use_rsa"]:
        valid &= np.isfinite(delta_rsa) & (np.abs(delta_rsa) <= args.rsa_caliper)
        distance_squared += (delta_rsa / args.rsa_caliper) ** 2
    if settings["use_position"]:
        valid &= np.abs(delta_position) <= args.position_caliper
        distance_squared += (delta_position / args.position_caliper) ** 2

    selected = np.flatnonzero(valid)
    if len(selected) == 0:
        return pd.DataFrame()
    order = np.lexsort((centers[selected], np.sqrt(distance_squared[selected])))
    if not settings["use_all_controls"]:
        order = order[: args.max_controls_per_apex]
    selected = selected[order]
    feature_means = interval_feature_means(record, starts[selected], ends[selected])
    result = pd.DataFrame({
        "control_center_index_0based": centers[selected],
        "control_start_index_0based": starts[selected],
        "control_end_index_0based_inclusive": ends[selected] - 1,
        "control_resolved_fraction": resolved_fraction[selected],
        "match_distance": np.sqrt(distance_squared[selected]),
        "delta_neq_control_minus_apex": delta_neq[selected],
        "delta_rsa_control_minus_apex": delta_rsa[selected],
        "delta_position_control_minus_apex": delta_position[selected],
    })
    for feature in STRUCTURE_FEATURES:
        result[f"control_interval_mean_{feature}"] = feature_means[feature]
    return result


def build_matches(
    bands: pd.DataFrame, residues: pd.DataFrame, masks: dict, args,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrays = make_protein_arrays(residues)
    case_rows, control_rows, coverage_rows = [], [], []
    for band in bands.itertuples(index=False):
        protein = str(band.protein)
        record = arrays.get(protein)
        if record is None:
            continue
        case = band_case_row(band, record)
        case_rows.append(case)
        for cohort, threshold in COHORT_THRESHOLDS.items():
            for scheme in MATCH_MODELS:
                selected = pd.DataFrame()
                if case["resolved_fraction"] >= threshold - 1e-12:
                    selected = candidate_intervals(
                        band, record, masks[(str(band.condition), protein)],
                        scheme, threshold, args,
                    )
                coverage_rows.append({
                    "band_id": str(band.band_id), "condition": str(band.condition),
                    "split": str(band.split), "protein": protein, "sign": int(band.sign),
                    "cohort": cohort, "match_scheme": scheme,
                    "case_eligible": case["resolved_fraction"] >= threshold - 1e-12,
                    "n_controls": len(selected), "matched": len(selected) > 0,
                })
                for rank, control in enumerate(selected.itertuples(index=False), 1):
                    row = {
                        "band_id": str(band.band_id), "condition": str(band.condition),
                        "split": str(band.split), "protein": protein,
                        "sign": int(band.sign), "cohort": cohort,
                        "match_scheme": scheme, "control_rank": rank,
                    }
                    row.update(control._asdict())
                    control_rows.append(row)
    return pd.DataFrame(case_rows), pd.DataFrame(control_rows), pd.DataFrame(coverage_rows)


def matched_effects(
    cases: pd.DataFrame, controls: pd.DataFrame, args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if controls.empty:
        return pd.DataFrame(), pd.DataFrame()
    indexed = cases.set_index("band_id")
    rows = []
    for keys, group in controls.groupby(["band_id", "cohort", "match_scheme"], sort=False):
        band_id, cohort, scheme = keys
        case = indexed.loc[band_id]
        for feature in STRUCTURE_FEATURES:
            case_value = case[f"interval_mean_{feature}"]
            values = pd.to_numeric(
                group[f"control_interval_mean_{feature}"], errors="coerce"
            ).dropna()
            if not np.isfinite(case_value) or values.empty:
                continue
            rows.append({
                "band_id": band_id, "condition": case.condition,
                "split": case.split, "protein": case.protein, "sign": int(case.sign),
                "cohort": cohort, "match_scheme": scheme, "feature": feature,
                "band_mean": float(case_value), "control_mean": float(values.mean()),
                "band_minus_control": float(case_value - values.mean()),
            })
    by_case = pd.DataFrame(rows)
    by_protein = by_case.groupby(
        ["condition", "split", "protein", "sign", "cohort", "match_scheme", "feature"],
        as_index=False,
    ).agg(
        n_bands=("band_id", "nunique"),
        band_mean=("band_mean", "mean"), control_mean=("control_mean", "mean"),
        band_minus_control=("band_minus_control", "mean"),
    )
    rng = np.random.default_rng(args.random_seed)
    summary_rows = []
    for keys, group in by_protein.groupby(
        ["condition", "split", "sign", "cohort", "match_scheme", "feature"],
        sort=False,
    ):
        values = group.band_minus_control.dropna().to_numpy(float)
        n = len(values)
        lower, upper = bootstrap_mean_ci(values, args.n_bootstrap, rng)
        observed = float(np.mean(values)) if n else np.nan
        z = p_upper = p_lower = p_two = np.nan
        if n >= args.minimum_inference_proteins:
            observed, z, p_upper, p_lower, p_two = sign_flip_test(
                values, args.n_sign_flips, rng
            )
        summary_rows.append({
            **dict(zip(
                ["condition", "split", "sign", "cohort", "match_scheme", "feature"],
                keys,
            )),
            "n_proteins": n, "n_bands": int(group.n_bands.sum()),
            "mean_band_minus_control": observed,
            "bootstrap_ci95_low": lower, "bootstrap_ci95_high": upper,
            "sign_flip_z": z, "sign_flip_p_upper": p_upper,
            "sign_flip_p_lower": p_lower, "sign_flip_p_two_sided": p_two,
            "inference_eligible": n >= args.minimum_inference_proteins,
        })
    summary = pd.DataFrame(summary_rows)
    summary["sign_flip_q_bh_global"] = bh(summary.sign_flip_p_two_sided)
    summary["sign_flip_q_bh_within_cohort_scheme_sign"] = summary.groupby(
        ["cohort", "match_scheme", "sign"], group_keys=False
    ).sign_flip_p_two_sided.apply(bh)
    return by_protein, summary


def audit_run(
    bands: pd.DataFrame, cases: pd.DataFrame, controls: pd.DataFrame,
    coverage: pd.DataFrame, masks: dict, args,
) -> dict:
    checks = {
        "one_case_row_per_mapped_band": cases.band_id.is_unique,
        "control_intervals_preserve_width": True,
        "control_intervals_outside_all_bands": True,
        "q3_matching_inherited_from_phase2": list(MATCH_MODELS) == list(PHASE2_SCHEME_ORDER),
        "position_used_only_in_final_scheme": all(
            settings["use_position"] == (name == "q3_neq_rsa_position")
            for name, settings in MATCH_MODELS.items()
        ),
    }
    case_lookup = bands.set_index("band_id")
    for row in controls.itertuples(index=False):
        band = case_lookup.loc[row.band_id]
        expected = int(band.end_index_0based_inclusive) - int(band.start_index_0based) + 1
        observed = int(row.control_end_index_0based_inclusive) - int(row.control_start_index_0based) + 1
        checks["control_intervals_preserve_width"] &= observed == expected
        mask = masks[(str(row.condition), str(row.protein))]
        checks["control_intervals_outside_all_bands"] &= not mask[
            int(row.control_start_index_0based): int(row.control_end_index_0based_inclusive) + 1
        ].any()
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "input_bands": len(bands), "mapped_band_intervals": len(cases),
            "control_interval_rows": len(controls),
            "matched_band_contexts": int(coverage.matched.sum()),
        },
    }


PHASE2_SCHEME_ORDER = (
    "q3_only", "q3_neq", "q3_neq_rsa", "q3_neq_rsa_position",
)


def main() -> None:
    args = parse_args()
    print("Loading bands, annotations, and experimental structures...", file=sys.stderr)
    bands, residues, mapping_audit = prepare_inputs(args)
    masks = band_masks(bands)
    print("Matching actual bands to same-width non-band intervals...", file=sys.stderr)
    cases, controls, coverage = build_matches(bands, residues, masks, args)
    print("Calculating protein-level interval effects...", file=sys.stderr)
    by_protein, summary = matched_effects(cases, controls, args)
    audit = audit_run(bands, cases, controls, coverage, masks, args)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping_audit.to_csv(output / "structure_mapping_audit.csv", index=False)
    cases.to_csv(output / "band_interval_structure_features.csv.gz", index=False, compression="gzip")
    controls.to_csv(output / "matched_nonband_intervals.csv.gz", index=False, compression="gzip")
    coverage.to_csv(output / "band_interval_match_coverage.csv", index=False)
    by_protein.to_csv(
        output / "matched_band_interval_effects_by_protein.csv.gz",
        index=False, compression="gzip",
    )
    summary.to_csv(output / "matched_band_interval_effect_summary.csv", index=False)
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    parameters = {
        "analysis_unit": "actual seed-stable contribution-band interval",
        "control_interval": (
            "same protein, same width and apex-to-boundary offsets, outside every band"
        ),
        "matching_schemes": MATCH_MODELS,
        "cohorts": COHORT_THRESHOLDS,
        "calipers": {
            "neq": args.neq_caliper, "rsa": args.rsa_caliper,
            "normalized_position": args.position_caliper,
        },
        "n_sign_flips": args.n_sign_flips, "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed, "split": args.split,
        "stable_bands_csv": str(Path(args.stable_bands_csv).expanduser().resolve()),
        "residue_annotations_csv": str(Path(args.residue_annotations_csv).expanduser().resolve()),
        "contact_json": [str(Path(path).expanduser().resolve()) for path in args.contact_json],
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "mapped_band_intervals": len(cases), "control_interval_rows": len(controls),
        "effect_summary_rows": len(summary), "audit_status": audit["status"],
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
