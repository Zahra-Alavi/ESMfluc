#!/usr/bin/env python3
"""Band-centric sensitivity analysis using experimental protein structures.

The analysis unit is a stable band apex, not a NetSurfP Q8 segment. Stable
test-set apex indices are mapped to experimental PDB residues through the
alignment-audited, input-sequence-indexed contact files produced by
Attention/build_contact_maps_from_pdb.py.

Two complementary questions are answered separately for positive and negative
bands:

1. Does an apex occupy a different experimental structural environment than
   matched, same-protein residues outside every detected band?
2. Among apices, does standardized importance R_p vary with an experimental
   structural feature X? The fitted response is
       R = log2(R_p / detection_threshold) ~ X + adjustments.

The matched comparison uses the same four nested control schemes as Phase 2:
Q3 only; Q3 plus N_eq; Q3 plus N_eq and RSA; and a final sensitivity that also
matches normalized sequence position.
Q8 is never used to define, match, or model an object.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

try:
    from signed_band_analysis.analyze_signed_band_external_structure import (
        audit_contacts,
        ca_curvature,
        ca_virtual_torsion,
        contact_edges,
        graph_features,
        load_contacts,
    )
    from signed_band_analysis.analyze_signed_band_biophysical_enrichment import (
        MATCH_SCHEMES as PHASE2_MATCH_SCHEMES,
        add_primary_pvalue_family,
        bootstrap_mean_ci,
        deterministic_tie_values,
        sign_flip_test,
    )
except ModuleNotFoundError:  # supports direct execution from this directory
    from analyze_signed_band_external_structure import (  # type: ignore
        audit_contacts,
        ca_curvature,
        ca_virtual_torsion,
        contact_edges,
        graph_features,
        load_contacts,
    )
    from analyze_signed_band_biophysical_enrichment import (  # type: ignore
        MATCH_SCHEMES as PHASE2_MATCH_SCHEMES,
        add_primary_pvalue_family,
        bootstrap_mean_ci,
        deterministic_tie_values,
        sign_flip_test,
    )


STRUCTURE_FEATURES = [
    "ca_curvature_degrees",
    "abs_ca_virtual_torsion_degrees",
    "contact_degree",
    "ca_packing_index",
    "mean_contact_distance_angstrom",
    "betweenness",
    "closeness",
    "participation_coefficient",
    "community_boundary",
]
COMPLETE_STRUCTURE_FEATURES = [
    feature for feature in STRUCTURE_FEATURES
    if feature != "mean_contact_distance_angstrom"
]

MATCH_MODELS = {
    name: dict(settings) for name, settings in PHASE2_MATCH_SCHEMES.items()
}

IMPORTANCE_MODELS = {
    "A1_q3_stability": ["strict_3_of_3", "q3"],
    "A2_add_neq": ["strict_3_of_3", "q3", "neq"],
    "A3_add_rsa": ["strict_3_of_3", "q3", "neq", "rsa"],
    "A4_add_position": [
        "strict_3_of_3", "q3", "neq", "rsa", "normalized_position",
    ],
}

HEADLINE_NETWORK_FEATURES = (
    "contact_degree",
    "ca_packing_index",
    "betweenness",
    "closeness",
    "participation_coefficient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable_bands_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument(
        "--contact_json", nargs="+", required=True,
        help="Alignment-audited contact JSON/JSON.GZ files with C-alpha coordinates",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--detection_threshold_R_p", type=float, default=2.0)
    parser.add_argument("--max_controls_per_apex", type=int, default=5)
    parser.add_argument("--position_caliper", type=float, default=0.25)
    parser.add_argument("--neq_caliper", type=float, default=0.25)
    parser.add_argument("--rsa_caliper", type=float, default=0.15)
    parser.add_argument(
        "--terminal_exclusion", type=int, default=2,
        help=(
            "Exclude this many residues at each protein terminus from both "
            "matched apices and controls (default: 2)."
        ),
    )
    parser.add_argument("--min_mapping_identity", type=float, default=0.90)
    parser.add_argument("--min_input_coverage", type=float, default=0.80)
    parser.add_argument("--min_contact_sequence_separation", type=int, default=3)
    parser.add_argument("--betweenness_samples", type=int, default=64)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--n_sign_flips", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument(
        "--group_manifest_csv",
        default="data_splits/atlas_grouped_v1/split_manifest_grouped_v1.csv",
        help="Manifest containing name, split, and union_group_id.",
    )
    args = parser.parse_args()
    if args.detection_threshold_R_p <= 0:
        parser.error("--detection_threshold_R_p must be positive")
    if args.max_controls_per_apex < 1:
        parser.error("--max_controls_per_apex must be positive")
    for name in ("position_caliper", "neq_caliper", "rsa_caliper"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.minimum_inference_proteins < 2:
        parser.error("--minimum_inference_proteins must be at least 2")
    if args.n_sign_flips < 1 or args.n_bootstrap < 1:
        parser.error("--n_sign_flips and --n_bootstrap must be positive")
    if args.terminal_exclusion < 0:
        parser.error("--terminal_exclusion must be nonnegative")
    return args


def bh(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return output
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index.to_numpy()[order]] = np.minimum(adjusted, 1.0)
    return output


def read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser())


def require_columns(table: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(table)
    if missing:
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")


def contact_mean_distance(record: dict, graph_node: dict, args) -> np.ndarray:
    length = len(record["sequence"])
    totals = np.zeros(length, dtype=float)
    counts = np.zeros(length, dtype=int)
    resolved = np.asarray(graph_node["resolved"], dtype=bool)
    for i, j, distance in contact_edges(record):
        if (
            abs(i - j) < args.min_contact_sequence_separation
            or not resolved[i] or not resolved[j] or not np.isfinite(distance)
        ):
            continue
        totals[i] += distance
        totals[j] += distance
        counts[i] += 1
        counts[j] += 1
    output = np.full(length, np.nan)
    output[counts > 0] = totals[counts > 0] / counts[counts > 0]
    return output


def structure_residue_table(
    contacts: dict[str, dict], accepted: set[str], args,
) -> pd.DataFrame:
    rows = []
    for protein in sorted(accepted):
        record = contacts[protein]
        coordinates = record.get("ca_coordinates_angstrom")
        if coordinates is None:
            raise ValueError(
                f"{protein}: contact record lacks ca_coordinates_angstrom; rebuild "
                "with --include_ca_coordinates"
            )
        length = len(record["sequence"])
        if len(coordinates) != length:
            raise ValueError(f"{protein}: coordinate array length does not match sequence")
        node, _ = graph_features(record, args)
        curvature = ca_curvature(coordinates)
        torsion = ca_virtual_torsion(coordinates)
        mean_distance = contact_mean_distance(record, node, args)
        mappings = {
            int(item["input_index_0based"]): item
            for item in record.get("residue_mapping", [])
        }
        for index in range(length):
            mapping = mappings.get(index, {})
            coordinate = coordinates[index]
            row = {
                "protein": protein,
                "residue_index_0based": index,
                "protein_length_structure": length,
                "structure_resolved": bool(node["resolved"][index]),
                "pdb_id": record.get("pdb_id"),
                "pdb_chain_id": mapping.get("pdb_chain_id"),
                "pdb_resid": mapping.get("pdb_resid"),
                "pdb_insertion_code": mapping.get("pdb_insertion_code"),
                "pdb_residue_label": mapping.get("pdb_residue_label"),
                "ca_x": coordinate[0] if coordinate is not None else np.nan,
                "ca_y": coordinate[1] if coordinate is not None else np.nan,
                "ca_z": coordinate[2] if coordinate is not None else np.nan,
                "ca_curvature_degrees": curvature[index],
                "abs_ca_virtual_torsion_degrees": torsion[index],
                "contact_degree": node["contact_degree"][index],
                "ca_packing_index": node["inverse_distance_weighted_degree"][index],
                "mean_contact_distance_angstrom": mean_distance[index],
                "betweenness": node["betweenness"][index],
                "closeness": node["closeness"][index],
                "participation_coefficient": node["participation_coefficient"][index],
                "community_boundary": node["community_boundary"][index],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_inputs(args):
    bands = read_table(args.stable_bands_csv)
    require_columns(
        bands,
        [
            "condition", "split", "protein", "sign", "band_id",
            "apex_index_0based", "start_index_0based",
            "end_index_0based_inclusive", "protein_length",
            "apex_standardized_magnitude_R_p", "stability_class",
            "seed_support_count", "eligible_start_index_0based",
            "eligible_end_index_0based_exclusive",
        ],
        "stable bands",
    )
    bands = bands[bands.split.astype(str).eq(args.split)].copy()
    if args.conditions:
        bands = bands[bands.condition.isin(args.conditions)].copy()
    if bands.empty:
        raise ValueError(f"No stable bands remain for split={args.split!r}")
    if not bands.sign.isin([-1, 1]).all():
        raise ValueError("Band signs must be -1 or +1")
    if bands.band_id.duplicated().any():
        raise ValueError("band_id is not unique in the selected stable-band table")
    if (
        (bands.apex_index_0based < bands.start_index_0based)
        | (bands.apex_index_0based > bands.end_index_0based_inclusive)
    ).any():
        raise ValueError("At least one apex lies outside its band interval")
    if (bands.apex_standardized_magnitude_R_p < args.detection_threshold_R_p - 1e-9).any():
        raise ValueError("Stable table contains an apex below the stated R_p threshold")

    annotations = read_table(args.residue_annotations_csv)
    require_columns(
        annotations,
        [
            "split", "protein", "residue_index_0based", "protein_length",
            "q3", "neq", "rsa",
        ],
        "residue annotations",
    )
    annotations = annotations[annotations.split.astype(str).eq(args.split)].copy()
    annotations = annotations[
        annotations.protein.astype(str).isin(bands.protein.astype(str).unique())
    ].copy()
    if annotations.duplicated(["protein", "residue_index_0based"]).any():
        raise ValueError("Residue annotations are not unique by protein and residue index")

    protein_splits = bands[["protein", "split"]].drop_duplicates()
    if protein_splits.protein.duplicated().any():
        raise ValueError("A selected protein occurs in multiple splits")
    protein_split = dict(zip(protein_splits.protein.astype(str), protein_splits.split.astype(str)))
    contacts = load_contacts(args.contact_json)
    mapping_audit = audit_contacts(contacts, protein_split, args)
    accepted = set(
        mapping_audit.loc[
            mapping_audit.mapping_accepted.astype(bool)
            & mapping_audit.protein.astype(str).isin(protein_split),
            "protein",
        ].astype(str)
    )
    residues = structure_residue_table(contacts, accepted, args)
    residues = residues.merge(
        annotations[
            ["protein", "residue_index_0based", "q3", "neq", "rsa"]
        ],
        on=["protein", "residue_index_0based"],
        how="left",
        validate="one_to_one",
    )
    residues["normalized_position"] = (
        residues.residue_index_0based
        / np.maximum(residues.protein_length_structure - 1, 1)
    )
    # Mean contact distance is correctly undefined for a resolved residue with
    # zero nonlocal contacts; do not let that exclude unpacked residues.
    residues["complete_structure"] = (
        residues[COMPLETE_STRUCTURE_FEATURES].notna().all(axis=1)
    )
    return bands, residues, mapping_audit


def band_masks(bands: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    masks = {}
    for (condition, protein), group in bands.groupby(["condition", "protein"], sort=False):
        lengths = group.protein_length.astype(int).unique()
        if len(lengths) != 1:
            raise ValueError(f"{condition}/{protein}: inconsistent protein length")
        mask = np.zeros(int(lengths[0]), dtype=bool)
        for row in group.itertuples(index=False):
            mask[int(row.start_index_0based): int(row.end_index_0based_inclusive) + 1] = True
        masks[(str(condition), str(protein))] = mask
    return masks


def build_apex_table(
    bands: pd.DataFrame, residues: pd.DataFrame, args,
) -> pd.DataFrame:
    apex = bands.merge(
        residues,
        left_on=["protein", "apex_index_0based"],
        right_on=["protein", "residue_index_0based"],
        how="left",
        validate="many_to_one",
        indicator="structure_residue_merge",
    )
    apex["R_p"] = pd.to_numeric(
        apex.apex_standardized_magnitude_R_p, errors="coerce"
    )
    apex["R"] = np.log2(apex.R_p / args.detection_threshold_R_p)
    apex["strict_3_of_3"] = (
        apex.seed_support_count.astype(int).eq(3)
        | apex.stability_class.astype(str).eq("strict_3_of_3")
    ).astype(float)
    apex["sign_label"] = np.where(apex.sign.eq(1), "positive", "negative")
    return apex


def candidate_control_pool(
    case, residues_by_protein: dict[str, pd.DataFrame], masks, model: str, args,
) -> pd.DataFrame:
    pool = residues_by_protein.get(str(case.protein))
    if pool is None:
        return pd.DataFrame()
    pool = pool.copy()
    mask = masks[(str(case.condition), str(case.protein))]
    indices = pool.residue_index_0based.astype(int).to_numpy()
    valid_index = (indices >= 0) & (indices < len(mask))
    pool = pool.loc[
        valid_index
        & ~mask[np.clip(indices, 0, len(mask) - 1)]
        & pool.structure_resolved.astype(bool).to_numpy()
        & (indices >= int(case.eligible_start_index_0based))
        & (indices < int(case.eligible_end_index_0based_exclusive))
        & (indices >= int(args.terminal_exclusion))
        & (indices < len(mask) - int(args.terminal_exclusion))
    ].copy()
    settings = MATCH_MODELS[model]
    if pd.isna(case.q3):
        return pool.iloc[0:0]
    pool = pool[pool.q3.astype(str).eq(str(case.q3))].copy()
    if pool.empty:
        return pool
    pool["delta_position"] = pool.normalized_position - float(case.normalized_position)
    pool["abs_delta_position"] = pool.delta_position.abs()
    pool["delta_neq"] = pd.to_numeric(pool.neq, errors="coerce") - float(case.neq)
    pool["delta_rsa"] = pd.to_numeric(pool.rsa, errors="coerce") - float(case.rsa)
    distance_squared = np.zeros(len(pool), dtype=float)
    if settings["use_neq"]:
        if not np.isfinite(case.neq):
            return pool.iloc[0:0]
        keep = pool.delta_neq.abs().le(args.neq_caliper)
        pool = pool[keep].copy()
        distance_squared = (
            pool.delta_neq.to_numpy(float) / args.neq_caliper
        ) ** 2
    else:
        distance_squared = np.zeros(len(pool), dtype=float)
    if settings["use_rsa"]:
        if not np.isfinite(case.rsa):
            return pool.iloc[0:0]
        keep = pool.delta_rsa.abs().le(args.rsa_caliper)
        pool = pool[keep].copy()
        distance_squared = distance_squared[keep.to_numpy()] + (
            pool.delta_rsa.to_numpy(float) / args.rsa_caliper
        ) ** 2
    if settings["use_position"]:
        if not np.isfinite(case.normalized_position):
            return pool.iloc[0:0]
        keep = pool.abs_delta_position.le(args.position_caliper)
        pool = pool[keep].copy()
        distance_squared = distance_squared[keep.to_numpy()] + (
            pool.delta_position.to_numpy(float) / args.position_caliper
        ) ** 2
    pool["match_distance"] = np.sqrt(distance_squared)
    pool["tie_break"] = deterministic_tie_values(
        random_seed=args.random_seed,
        condition=str(case.condition),
        split=str(case.split),
        protein=str(case.protein),
        band_id=str(case.band_id),
        match_scheme=model,
        candidate_indices=pool.residue_index_0based.to_numpy(int),
    )
    ordered = pool.sort_values(["match_distance", "tie_break"], kind="mergesort")
    return ordered if settings["use_all_controls"] else ordered.head(
        args.max_controls_per_apex
    )


def match_controls(apex: pd.DataFrame, residues: pd.DataFrame, masks, args) -> pd.DataFrame:
    residue_groups = {
        str(protein): group.reset_index(drop=True)
        for protein, group in residues.groupby("protein", sort=False)
    }
    rows = []
    for case in apex.itertuples(index=False):
        if case.structure_residue_merge != "both" or not bool(case.structure_resolved):
            continue
        if (
            int(case.apex_index_0based) < int(args.terminal_exclusion)
            or int(case.apex_index_0based)
            >= int(case.protein_length) - int(args.terminal_exclusion)
        ):
            continue
        for model in MATCH_MODELS:
            selected = candidate_control_pool(case, residue_groups, masks, model, args)
            for rank, control in enumerate(selected.itertuples(index=False), start=1):
                row = {
                    "band_id": case.band_id, "condition": case.condition,
                    "split": case.split, "protein": case.protein,
                    "sign": int(case.sign), "sign_label": case.sign_label,
                    "apex_index_0based": int(case.apex_index_0based),
                    "protein_length": int(case.protein_length),
                    "match_model": model, "control_rank": rank,
                    "control_index_0based": int(control.residue_index_0based),
                    "match_distance": float(control.match_distance),
                    "delta_position_control_minus_apex": float(control.delta_position),
                    "delta_neq_control_minus_apex": float(control.delta_neq)
                    if np.isfinite(control.delta_neq) else np.nan,
                    "delta_rsa_control_minus_apex": float(control.delta_rsa)
                    if np.isfinite(control.delta_rsa) else np.nan,
                    "apex_q3": case.q3, "control_q3": control.q3,
                    "control_pdb_id": control.pdb_id,
                    "control_pdb_chain_id": control.pdb_chain_id,
                    "control_pdb_resid": control.pdb_resid,
                    "control_pdb_insertion_code": control.pdb_insertion_code,
                    "control_pdb_residue_label": control.pdb_residue_label,
                    "control_ca_x": control.ca_x,
                    "control_ca_y": control.ca_y,
                    "control_ca_z": control.ca_z,
                }
                for feature in STRUCTURE_FEATURES:
                    row[f"control_{feature}"] = getattr(control, feature)
                rows.append(row)
    return pd.DataFrame(rows)


def match_coverage_balance(apex: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, sign), cases in apex.groupby(["condition", "sign"], sort=False):
        for model in MATCH_MODELS:
            selected = controls[
                controls.condition.eq(condition)
                & controls.sign.eq(sign)
                & controls.match_model.eq(model)
            ] if not controls.empty else controls
            matched_ids = set(selected.band_id) if not selected.empty else set()
            rows.append({
                "condition": condition, "sign": int(sign), "match_model": model,
                "n_apices": len(cases),
                "n_structure_resolved_apices": int(cases.structure_resolved.fillna(False).sum()),
                "n_matched_apices": int(cases.band_id.isin(matched_ids).sum()),
                "matched_fraction_of_all_apices": float(cases.band_id.isin(matched_ids).mean()),
                "n_controls": len(selected),
                "mean_controls_per_matched_apex": (
                    float(selected.groupby("band_id").size().mean())
                    if not selected.empty else np.nan
                ),
                "mean_abs_position_difference": (
                    float(selected.delta_position_control_minus_apex.abs().mean())
                    if not selected.empty else np.nan
                ),
                "mean_abs_neq_difference": (
                    float(selected.delta_neq_control_minus_apex.abs().mean())
                    if not selected.empty else np.nan
                ),
                "mean_abs_rsa_difference": (
                    float(selected.delta_rsa_control_minus_apex.abs().mean())
                    if not selected.empty else np.nan
                ),
                "q3_exact_fraction": (
                    float((selected.apex_q3.astype(str) == selected.control_q3.astype(str)).mean())
                    if not selected.empty else np.nan
                ),
            })
    return pd.DataFrame(rows)


def matched_effects(
    apex: pd.DataFrame, controls: pd.DataFrame, args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if controls.empty:
        return pd.DataFrame(), pd.DataFrame()
    apex_indexed = apex.set_index("band_id")
    case_rows = []
    for (band_id, model), group in controls.groupby(["band_id", "match_model"], sort=False):
        case = apex_indexed.loc[band_id]
        for feature in STRUCTURE_FEATURES:
            apex_value = pd.to_numeric(pd.Series([case[feature]]), errors="coerce").iloc[0]
            control_values = pd.to_numeric(group[f"control_{feature}"], errors="coerce").dropna()
            if not np.isfinite(apex_value) or control_values.empty:
                continue
            case_rows.append({
                "band_id": band_id, "condition": case.condition, "split": case.split,
                "protein": case.protein, "sign": int(case.sign),
                "match_model": model, "feature": feature,
                "apex_value": float(apex_value),
                "control_mean": float(control_values.mean()),
                "apex_minus_control": float(apex_value - control_values.mean()),
                "n_feature_available_controls": len(control_values),
            })
    by_case = pd.DataFrame(case_rows)
    if by_case.empty:
        return by_case, pd.DataFrame()
    by_protein = by_case.groupby(
        ["condition", "split", "protein", "sign", "match_model", "feature"],
        as_index=False,
    ).agg(
        n_apices=("band_id", "nunique"),
        apex_mean=("apex_value", "mean"),
        control_mean=("control_mean", "mean"),
        apex_minus_control=("apex_minus_control", "mean"),
    )
    rows = []
    rng = np.random.default_rng(args.random_seed)
    for keys, group in by_protein.groupby(
        ["condition", "split", "sign", "match_model", "feature"], sort=False,
    ):
        values = group.apex_minus_control.dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        lower, upper = bootstrap_mean_ci(values, args.n_bootstrap, rng)
        observed = mean
        sign_flip_z = p_upper = p_lower = p_two = np.nan
        if n >= args.minimum_inference_proteins:
            observed, sign_flip_z, p_upper, p_lower, p_two = sign_flip_test(
                values, args.n_sign_flips, rng
            )
        rows.append({
            **dict(zip(["condition", "split", "sign", "match_model", "feature"], keys)),
            "n_proteins": n, "n_apices": int(group.n_apices.sum()),
            "mean_apex_minus_control": observed,
            "ci95_low": lower, "ci95_high": upper,
            "bootstrap_ci95_low": lower, "bootstrap_ci95_high": upper,
            "median_apex_minus_control": float(np.median(values)) if n else np.nan,
            "sign_flip_z": sign_flip_z,
            "sign_flip_p_upper": p_upper,
            "sign_flip_p_lower": p_lower,
            "sign_flip_p_two_sided": p_two,
            "inference_eligible": n >= args.minimum_inference_proteins,
        })
    summary = pd.DataFrame(rows)
    summary["sign_flip_q_bh_global"] = bh(summary.sign_flip_p_two_sided)
    summary["sign_flip_q_bh_within_model_sign"] = summary.groupby(
        ["match_model", "sign"], group_keys=False
    ).sign_flip_p_two_sided.apply(bh)
    return by_protein, summary


def union_group_network_inference(
    by_protein: pd.DataFrame, args,
) -> pd.DataFrame:
    """Equal-union-group sensitivity intervals for strict network effects."""
    if by_protein.empty:
        return pd.DataFrame()
    manifest = pd.read_csv(args.group_manifest_csv)
    required = {"name", "split", "union_group_id"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Group manifest lacks required columns: {sorted(missing)}")
    mapping = manifest[["name", "split", "union_group_id"]].rename(
        columns={"name": "protein"}
    )
    if mapping.duplicated(["protein", "split"]).any():
        raise ValueError("Group manifest has duplicate protein/split keys")
    selected = by_protein[
        by_protein.match_model.eq("q3_neq")
        & by_protein.feature.isin(HEADLINE_NETWORK_FEATURES)
    ].copy()
    selected = selected.merge(
        mapping, on=["protein", "split"], how="left", validate="many_to_one"
    )
    if selected.union_group_id.isna().any():
        raise ValueError("At least one apex-structure protein lacks a union group")
    keys = ["condition", "split", "sign", "match_model", "feature"]
    group_effects = selected.groupby(
        [*keys, "union_group_id"], as_index=False
    ).agg(
        group_effect=("apex_minus_control", "mean"),
        proteins_in_group=("protein", "nunique"),
    )
    rng = np.random.default_rng(args.random_seed + 4001)
    rows = []
    for key, frame in group_effects.groupby(keys, sort=False):
        values = frame.group_effect.to_numpy(float)
        low, high = bootstrap_mean_ci(values, args.n_bootstrap, rng)
        z = p_two = np.nan
        if len(values) >= args.minimum_inference_proteins:
            _, z, _, _, p_two = sign_flip_test(values, args.n_sign_flips, rng)
        rows.append({
            **dict(zip(keys, key)),
            "n_union_groups": int(len(values)),
            "n_proteins": int(frame.proteins_in_group.sum()),
            "union_group_mean_effect": float(np.mean(values)),
            "union_group_ci95_low": low,
            "union_group_ci95_high": high,
            "union_group_sign_flip_z": z,
            "union_group_p_two_sided": p_two,
        })
    return add_primary_pvalue_family(
        pd.DataFrame(rows), source="union_group_p_two_sided"
    )


def design_matrix(group: pd.DataFrame, feature: str, adjustments: list[str]):
    required = ["R", "protein", feature] + adjustments
    data = group[required].copy()
    for column in ["R", feature, "normalized_position", "strict_3_of_3", "neq", "rsa"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if data.empty:
        return data, None, None, None

    feature_sd = float(data[feature].std(ddof=0))
    if not np.isfinite(feature_sd) or feature_sd <= 0:
        return data, None, None, None
    design = pd.DataFrame(index=data.index)
    for adjustment in adjustments:
        if adjustment == "q3":
            dummies = pd.get_dummies(data.q3.astype(str), prefix="q3", drop_first=True)
            if not dummies.empty:
                for column in dummies:
                    design[column] = dummies[column].astype(float)
        else:
            values = data[adjustment].to_numpy(float)
            sd = float(np.std(values))
            if sd > 0:
                design[adjustment] = (values - np.mean(values)) / sd
    design["X"] = (
        data[feature].to_numpy(float) - data[feature].mean()
    ) / feature_sd
    response = data.R.astype(float)

    # Absorb protein fixed effects by within-protein demeaning. This is
    # coefficient-equivalent to adding an intercept and one indicator per
    # protein, while avoiding repeated inversion of hundreds of dummy columns.
    proteins = data.protein.astype(str)
    within_design = design - design.groupby(proteins).transform("mean")
    within_response = response - response.groupby(proteins).transform("mean")
    varying = within_design.std(ddof=0).gt(1e-12)
    if not varying.get("X", False):
        return data, None, None, None
    within_design = within_design.loc[:, varying]
    return (
        data, within_design.to_numpy(float),
        within_response.to_numpy(float), within_design.columns.tolist(),
    )


def clustered_ols(data, matrix, response, names, feature):
    n, k = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    proteins = data.protein.astype(str).to_numpy()
    groups = np.unique(proteins)
    effective_k = k + len(groups)  # includes absorbed protein fixed effects
    if rank < k or len(groups) < 2 or n <= effective_k:
        return None
    inverse = np.linalg.inv(matrix.T @ matrix)
    beta = inverse @ matrix.T @ response
    residual = response - matrix @ beta
    meat = np.zeros((k, k))
    for protein in groups:
        selected = proteins == protein
        score = matrix[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (
        (len(groups) / (len(groups) - 1))
        * ((n - 1) / (n - effective_k))
    )
    covariance = correction * inverse @ meat @ inverse
    index = names.index("X")
    se = math.sqrt(max(float(covariance[index, index]), 0.0))
    coefficient = float(beta[index])
    pvalue = (
        float(2 * student_t.sf(abs(coefficient / se), len(groups) - 1))
        if se > 0 else np.nan
    )
    critical = float(student_t.ppf(0.975, len(groups) - 1))
    total_ss = float(np.sum((response - np.mean(response)) ** 2))
    r_squared = 1 - float(np.sum(residual ** 2)) / total_ss if total_ss > 0 else np.nan
    return {
        "n_apices": n, "n_proteins": len(groups),
        "n_parameters_including_protein_fixed_effects": effective_k,
        "design_rank": rank, "X_sd_raw": float(data[feature].std(ddof=0)),
        "X_mean_raw": float(data[feature].mean()),
        "coefficient_R_per_1sd_X": coefficient,
        "cluster_se": se, "ci95_low": coefficient - critical * se,
        "ci95_high": coefficient + critical * se,
        "cluster_p_two_sided": pvalue, "within_protein_r_squared": r_squared,
    }


def importance_associations(apex: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    cohorts = {
        "feature_available": pd.Series(True, index=apex.index),
        "complete_structure": apex.complete_structure.fillna(False),
        "strict_3_of_3": apex.strict_3_of_3.eq(1),
    }
    for cohort, cohort_mask in cohorts.items():
        selected = apex[cohort_mask].copy()
        for (condition, sign), group in selected.groupby(["condition", "sign"], sort=False):
            for model, adjustments in IMPORTANCE_MODELS.items():
                for feature in STRUCTURE_FEATURES:
                    data, matrix, response, names = design_matrix(group, feature, adjustments)
                    base = {
                        "condition": condition, "split": args.split, "sign": int(sign),
                        "cohort": cohort, "importance_model": model,
                        "feature_X": feature,
                        "response_R": "log2(R_p / detection_threshold_R_p)",
                    }
                    if matrix is None:
                        rows.append({
                            **base, "n_apices": len(data), "n_proteins": data.protein.nunique(),
                            "fit_status": "insufficient_or_invariant",
                        })
                        continue
                    fit = clustered_ols(data, matrix, response, names, feature)
                    if fit is None or fit["n_proteins"] < args.minimum_inference_proteins:
                        rows.append({
                            **base, "n_apices": len(data), "n_proteins": data.protein.nunique(),
                            "fit_status": "insufficient_proteins_or_singular",
                        })
                        continue
                    rows.append({**base, **fit, "fit_status": "ok"})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["cluster_q_bh_global"] = bh(result.get("cluster_p_two_sided"))
        result["cluster_q_bh_within_model_sign"] = result.groupby(
            ["cohort", "importance_model", "sign"], group_keys=False
        ).cluster_p_two_sided.apply(bh)
    return result


def feature_coverage(apex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in apex.groupby(["condition", "split", "sign"], sort=False):
        for feature in ["structure_resolved", "complete_structure", *STRUCTURE_FEATURES]:
            if feature in ("structure_resolved", "complete_structure"):
                available = group[feature].fillna(False).astype(bool)
            else:
                available = pd.to_numeric(group[feature], errors="coerce").notna()
            rows.append({
                **dict(zip(["condition", "split", "sign"], keys)),
                "feature": feature, "n_apices": len(group),
                "n_available": int(available.sum()),
                "available_fraction": float(available.mean()),
            })
    return pd.DataFrame(rows)


def integrity_audit(
    bands: pd.DataFrame, apex: pd.DataFrame, controls: pd.DataFrame,
    masks, mapping_audit: pd.DataFrame, args,
) -> dict:
    checks = {}
    checks["selected_split_only"] = bool(bands.split.astype(str).eq(args.split).all())
    checks["band_ids_unique"] = bool(bands.band_id.is_unique)
    checks["one_apex_row_per_band"] = bool(
        len(apex) == len(bands) and apex.band_id.is_unique
    )
    checks["R_formula_exact"] = bool(np.allclose(
        apex.R.to_numpy(float),
        np.log2(apex.R_p.to_numpy(float) / args.detection_threshold_R_p),
        equal_nan=False,
    ))
    checks["accepted_maps_input_indexed"] = bool(
        mapping_audit.loc[
            mapping_audit.mapping_accepted.astype(bool),
            "mapping_accepted",
        ].all()
    )
    if controls.empty:
        checks.update({
            "controls_same_protein": False, "controls_outside_all_bands": False,
            "q3_exact_where_required": False,
        })
    else:
        case = apex.set_index("band_id")
        checks["controls_same_protein"] = bool(all(
            str(row.protein) == str(case.loc[row.band_id, "protein"])
            for row in controls.itertuples(index=False)
        ))
        checks["controls_outside_all_bands"] = bool(all(
            not masks[(str(row.condition), str(row.protein))][int(row.control_index_0based)]
            for row in controls.itertuples(index=False)
        ))
        checks["controls_differ_from_apex"] = bool(
            (controls.control_index_0based != controls.apex_index_0based).all()
        )
        checks["matched_cases_and_controls_nonterminal"] = bool(
            (
                controls.apex_index_0based.ge(args.terminal_exclusion)
                & controls.apex_index_0based.lt(
                    controls.protein_length - args.terminal_exclusion
                )
                & controls.control_index_0based.ge(args.terminal_exclusion)
                & controls.control_index_0based.lt(
                    controls.protein_length - args.terminal_exclusion
                )
            ).all()
        )
        position = controls[controls.match_model.eq("q3_neq_rsa_position")]
        checks["position_caliper_where_required"] = bool(
            position.delta_position_control_minus_apex.abs().le(
                args.position_caliper + 1e-12
            ).all()
        )
        checks["q3_exact_for_all_controls"] = bool(
            (controls.apex_q3.astype(str) == controls.control_q3.astype(str)).all()
        )
        neq = controls[controls.match_model.isin([
            "q3_neq", "q3_neq_rsa", "q3_neq_rsa_position",
        ])]
        checks["neq_caliper_where_required"] = bool(
            neq.delta_neq_control_minus_apex.abs().le(args.neq_caliper + 1e-12).all()
        )
        rsa = controls[controls.match_model.isin([
            "q3_neq_rsa", "q3_neq_rsa_position",
        ])]
        checks["rsa_caliper_where_required"] = bool(
            rsa.delta_rsa_control_minus_apex.abs().le(args.rsa_caliper + 1e-12).all()
        )
        capped = controls[~controls.match_model.eq("q3_only")]
        checks["max_controls_respected"] = bool(
            capped.empty or capped.groupby(["band_id", "match_model"]).size().max()
            <= args.max_controls_per_apex
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "stable_bands": len(bands), "apex_rows": len(apex),
            "structure_resolved_apices": int(apex.structure_resolved.fillna(False).sum()),
            "matched_control_rows": len(controls),
            "accepted_structure_mappings": int(mapping_audit.mapping_accepted.sum()),
        },
    }


def main() -> None:
    args = parse_args()
    print("Loading and validating bands, annotations, and PDB mappings...", file=sys.stderr)
    bands, residues, mapping_audit = prepare_inputs(args)
    print("Mapping stable apices and selecting matched non-band controls...", file=sys.stderr)
    masks = band_masks(bands)
    apex = build_apex_table(bands, residues, args)
    controls = match_controls(apex, residues, masks, args)
    print("Summarizing matched structural effects...", file=sys.stderr)
    coverage_balance = match_coverage_balance(apex, controls)
    by_protein, effect_summary = matched_effects(apex, controls, args)
    group_inference = union_group_network_inference(by_protein, args)
    print("Fitting R ~ X nested fixed-effect models...", file=sys.stderr)
    associations = importance_associations(apex, args)
    coverage = feature_coverage(apex)
    audit = integrity_audit(bands, apex, controls, masks, mapping_audit, args)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping_audit.to_csv(output / "structure_mapping_audit.csv", index=False)
    apex.to_csv(
        output / "stable_apex_structure_features.csv.gz",
        index=False, compression="gzip",
    )
    controls.to_csv(
        output / "matched_nonband_controls.csv.gz",
        index=False, compression="gzip",
    )
    by_protein.to_csv(
        output / "matched_apex_control_effects_by_protein.csv.gz",
        index=False, compression="gzip",
    )
    effect_summary.to_csv(
        output / "matched_apex_control_effect_summary.csv", index=False,
    )
    group_inference.to_csv(
        output / "headline_union_group_network_inference.csv", index=False,
    )
    coverage_balance.to_csv(output / "match_coverage_balance.csv", index=False)
    associations.to_csv(output / "importance_R_structure_X_associations.csv", index=False)
    coverage.to_csv(output / "structural_feature_coverage.csv", index=False)
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    parameters = {
        "analysis_unit": "stable test-set band apex",
        "q8_policy": "Q8 is not used to define, match, or model objects",
        "response_R": "log2(R_p / detection_threshold_R_p)",
        "structural_predictor_symbol": "X",
        "structure_features": STRUCTURE_FEATURES,
        "complete_structure_features": COMPLETE_STRUCTURE_FEATURES,
        "matching_models": MATCH_MODELS,
        "importance_models": IMPORTANCE_MODELS,
        "control_definition": (
            "same-protein structurally resolved residue outside every positive "
            "and negative band interval for that condition"
        ),
        "reuse_policy": "controls may be reused across apices; inference is aggregated by protein",
        "split": args.split,
        "stable_bands_csv": str(Path(args.stable_bands_csv).expanduser().resolve()),
        "residue_annotations_csv": str(
            Path(args.residue_annotations_csv).expanduser().resolve()
        ),
        "contact_json": [
            str(Path(path).expanduser().resolve()) for path in args.contact_json
        ],
        "detection_threshold_R_p": args.detection_threshold_R_p,
        "max_controls_per_apex": args.max_controls_per_apex,
        "terminal_exclusion_residues_per_end": args.terminal_exclusion,
        "calipers": {
            "normalized_position": args.position_caliper,
            "neq": args.neq_caliper, "rsa": args.rsa_caliper,
        },
        "inference": (
            "matched effects are averaged within protein, tested with protein-level "
            "sign flips, and assigned protein-bootstrap confidence intervals; "
            "importance models use protein fixed effects and protein-clustered "
            "standard errors. The sole primary structural family is the 60 "
            "Q3+Neq-matched union-group tests (6 conditions x 2 signs x 5 "
            "headline network features), with one BH correction across that "
            "table. Q3+Neq+RSA and position-matched results are sensitivity "
            "analyses; other p-values are diagnostic or exploratory."
        ),
        "group_manifest_csv": str(Path(args.group_manifest_csv).expanduser().resolve()),
        "headline_union_group_features": list(HEADLINE_NETWORK_FEATURES),
        "minimum_inference_proteins": args.minimum_inference_proteins,
        "n_sign_flips": args.n_sign_flips,
        "n_bootstrap": args.n_bootstrap,
        "random_seed": args.random_seed,
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "stable_apices": len(apex),
        "structure_resolved_apices": int(apex.structure_resolved.fillna(False).sum()),
        "matched_control_rows": len(controls),
        "matched_effect_summary_rows": len(effect_summary),
        "headline_union_group_rows": len(group_inference),
        "importance_association_rows": len(associations),
        "audit_status": audit["status"],
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
