#!/usr/bin/env python3
"""Analyze signed-band biophysical localization, coverage, and enrichment.

Primary null: circularly shift the complete same-sign apex pattern within each
protein/condition.  This preserves protein, band count, relative spacing, and
the model condition while breaking alignment to residue annotations.  Protein
means are then averaged with equal protein weight.

A complementary paired sign analysis compares flexibility- versus
rigidity-supporting bands within the same protein using protein-level sign-flip
permutations.

Phase 1B/1C reverses the conditioning direction and measures how many Q8
loop/turn/bend residues or segments and Neq peaks are recovered by band apices
or intervals.  Proteins with zero bands are retained in these denominators.

Phase 2 compares each apex with non-band residues from the same protein.  Its
least-adjusted scheme matches Q3 only and uses every eligible same-Q3 control;
three nested schemes additionally match Neq, RSA, and normalized sequence
position and retain up to a configurable number of nearest controls.  The
outcomes include Q8 subtype, Neq, RSA, position, and geometric annotations.

Optional MDStrainMapper results are accepted for the test split only.  Strain
files are index/length audited before use, terminal residues are excluded, and
missing proteins remain explicitly documented rather than being interpreted as
zero strain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "neq": ("continuous", "flexibility-supporting higher"),
    "flexible_neq_gt1": ("binary", "flexibility-supporting higher"),
    "neq_top10_within_protein": ("binary", "flexibility-supporting higher"),
    "neq_peak": ("binary", "flexibility-supporting higher"),
    "distance_to_neq_peak": ("distance", "flexibility-supporting lower"),
    "distance_to_neq_top10": ("distance", "flexibility-supporting lower"),
    "q3_is_H": ("binary", "two-sided"),
    "q3_is_E": ("binary", "two-sided"),
    "q3_is_C": ("binary", "two-sided"),
    "q8_is_T": ("binary", "upper"),
    "q8_is_S": ("binary", "upper"),
    "q8_turn_or_bend_TS": ("binary", "upper"),
    "q8_loop_turn_bend_CTS": ("binary", "upper"),
    "structured_linker_loop": ("binary", "upper"),
    "q3_boundary_within2": ("binary", "upper"),
    "q8_boundary_within2": ("binary", "upper"),
    "distance_to_q3_boundary": ("distance", "lower"),
    "distance_to_q8_boundary": ("distance", "lower"),
    "rsa": ("continuous", "two-sided"),
    "exposed_rsa_ge025": ("binary", "two-sided"),
    "disorder": ("continuous", "upper"),
    "disorder_ge05": ("binary", "upper"),
    "interface": ("binary", "two-sided"),
    "torsion_change_from_previous": ("continuous", "upper"),
}

STRAIN_METRICS = {
    "strain_ensemble_mean": ("continuous", "two-sided"),
    "strain_ensemble_std": ("continuous", "two-sided"),
    "strain_abs_gradient_from_previous": ("continuous", "two-sided"),
    "strain_top_quantile_within_protein": (
        "binary", "flexibility-supporting higher"
    ),
    "distance_to_strain_top_quantile": (
        "distance", "flexibility-supporting lower"
    ),
}

Q8_LABELS = tuple("GHIBESTC")
MATCHED_OUTCOMES = {
    **{f"q8_is_{label}": ("binary", f"Q8 {label}") for label in Q8_LABELS},
    "q8_turn_or_bend_TS": ("binary", "Q8 turn or bend (T/S)"),
    "q8_loop_turn_bend_CTS": ("binary", "Q8 loop/turn/bend (C/T/S)"),
    "structured_linker_loop": ("binary", "structured linker/loop"),
    "neq": ("continuous", "Neq"),
    "rsa": ("continuous", "relative solvent accessibility"),
    "neq_peak": ("binary", "Neq peak"),
    "distance_to_neq_peak": ("distance", "distance to Neq peak"),
    "q3_boundary_within2": ("binary", "within two residues of Q3 boundary"),
    "q8_boundary_within2": ("binary", "within two residues of Q8 boundary"),
    "distance_to_q3_boundary": ("distance", "distance to Q3 boundary"),
    "distance_to_q8_boundary": ("distance", "distance to Q8 boundary"),
    "torsion_change_from_previous": ("continuous", "backbone torsion change"),
    "disorder": ("continuous", "disorder score"),
    "disorder_ge05": ("binary", "disorder score >= 0.5"),
}

STRAIN_MATCHED_OUTCOMES = {
    "strain_ensemble_mean": ("continuous", "MDStrainMapper ensemble mean"),
    "strain_ensemble_std": ("continuous", "MDStrainMapper ensemble standard deviation"),
    "strain_abs_gradient_from_previous": (
        "continuous", "absolute change in strain from the previous residue"
    ),
    "strain_top_quantile_within_protein": (
        "binary", "within-protein high-strain residue"
    ),
    "distance_to_strain_top_quantile": (
        "distance", "distance to a within-protein high-strain residue"
    ),
}

DERIVED_MATCHED_OUTCOMES = {
    "normalized_position": ("continuous", "normalized sequence position"),
}
ALL_MATCHED_OUTCOMES = {**MATCHED_OUTCOMES, **DERIVED_MATCHED_OUTCOMES}

MATCH_SCHEMES = {
    "q3_only": {
        "matched_covariates": ["Q3"],
        "use_neq": False,
        "use_rsa": False,
        "use_position": False,
        "use_all_controls": True,
    },
    "q3_neq": {
        "matched_covariates": ["Q3", "Neq"],
        "use_neq": True,
        "use_rsa": False,
        "use_position": False,
        "use_all_controls": False,
    },
    "q3_neq_rsa": {
        "matched_covariates": ["Q3", "Neq", "RSA"],
        "use_neq": True,
        "use_rsa": True,
        "use_position": False,
        "use_all_controls": False,
    },
    "q3_neq_rsa_position": {
        "matched_covariates": ["Q3", "Neq", "RSA", "normalized sequence position"],
        "use_neq": True,
        "use_rsa": True,
        "use_position": True,
        "use_all_controls": False,
    },
}

SIGN_LABELS = {-1: "rigidity_supporting", 1: "flexibility_supporting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated_bands_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument(
        "--protein_summary_csv",
        required=True,
        help=(
            "signed_band_protein_summary.csv from the same band call. Required "
            "so coverage includes proteins with zero bands."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--strain_root",
        default=None,
        help=(
            "Optional local MDStrainMapper test-result root containing "
            "<protein>/strain_summary.csv. Strain is never loaded for train "
            "or validation proteins."
        ),
    )
    parser.add_argument(
        "--strain_terminal_exclusion",
        type=int,
        default=10,
        help="Exclude this many residues at each terminus from strain analyses.",
    )
    parser.add_argument(
        "--strain_high_quantile",
        type=float,
        default=0.90,
        help="Within-protein quantile defining high strain (default: 0.90).",
    )
    parser.add_argument("--n_block_shifts", type=int, default=1000)
    parser.add_argument("--n_sign_flips", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--match_controls_per_apex", type=int, default=5)
    parser.add_argument("--match_band_buffer", type=int, default=0)
    parser.add_argument("--match_neq_caliper", type=float, default=0.25)
    parser.add_argument("--match_rsa_caliper", type=float, default=0.15)
    parser.add_argument("--match_position_caliper", type=float, default=0.25)
    parser.add_argument(
        "--match_schemes",
        nargs="+",
        choices=tuple(MATCH_SCHEMES),
        default=list(MATCH_SCHEMES),
        help=(
            "Phase 2 schemes to run. q3_only uses every eligible same-Q3 "
            "non-band residue; stricter schemes retain at most "
            "--match_controls_per_apex nearest controls."
        ),
    )
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    return parser.parse_args()


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return output
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    output.loc[valid.index.to_numpy()[order]] = adjusted
    return output


def empirical_p(observed: float, null: np.ndarray, tail: str) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return np.nan
    if tail == "upper":
        extreme = np.sum(null >= observed)
    elif tail == "lower":
        extreme = np.sum(null <= observed)
    elif tail == "two-sided":
        center = float(np.mean(null))
        extreme = np.sum(np.abs(null - center) >= abs(observed - center))
    else:
        raise ValueError(tail)
    return float((extreme + 1) / (len(null) + 1))


def z_score(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) < 2 or not np.isfinite(observed):
        return np.nan
    sd = float(np.std(null, ddof=1))
    return (observed - float(np.mean(null))) / sd if sd > 0 else np.nan


def load_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bands = pd.read_csv(args.annotated_bands_csv)
    residue = pd.read_csv(args.residue_annotations_csv)
    protein_summary = pd.read_csv(args.protein_summary_csv)
    required_bands = {
        "band_id", "condition", "split", "protein", "sign", "label",
        "apex_index_0based", "eligible_start_index_0based",
        "eligible_end_index_0based_exclusive", "start_index_0based",
        "end_index_0based_inclusive",
    }
    required_residue = {
        "split", "protein", "protein_length", "residue_index_0based", "q3", "q8",
        *METRICS, *MATCHED_OUTCOMES,
    }
    required_summary = {
        "condition", "split", "protein", "protein_length",
        "eligible_start_index_0based", "eligible_end_index_0based_exclusive",
        "n_flexibility_supporting_bands", "n_rigidity_supporting_bands",
    }
    missing_bands = required_bands - set(bands.columns)
    missing_residue = required_residue - set(residue.columns)
    missing_summary = required_summary - set(protein_summary.columns)
    if missing_bands:
        raise ValueError(f"Annotated bands lack: {sorted(missing_bands)}")
    if missing_residue:
        raise ValueError(f"Residue annotations lack: {sorted(missing_residue)}")
    if missing_summary:
        raise ValueError(f"Protein summary lacks: {sorted(missing_summary)}")
    if args.splits:
        bands = bands[bands["split"].isin(args.splits)]
        residue = residue[residue["split"].isin(args.splits)]
        protein_summary = protein_summary[protein_summary["split"].isin(args.splits)]
    if args.conditions:
        bands = bands[bands["condition"].isin(args.conditions)]
        protein_summary = protein_summary[
            protein_summary["condition"].isin(args.conditions)
        ]
    if bands.empty:
        raise ValueError("No bands selected")
    if bands["band_id"].duplicated().any():
        raise ValueError("Duplicate band IDs")
    summary_keys = ["condition", "split", "protein"]
    if protein_summary.duplicated(summary_keys).any():
        raise ValueError("Duplicate condition/split/protein rows in protein summary")
    band_keys = bands[summary_keys].drop_duplicates()
    missing_contexts = band_keys.merge(
        protein_summary[summary_keys], on=summary_keys, how="left", indicator=True
    )
    missing_contexts = missing_contexts[missing_contexts["_merge"] != "both"]
    if not missing_contexts.empty:
        raise ValueError(
            "Band contexts missing from protein summary: "
            f"{missing_contexts[summary_keys].head(10).to_dict('records')}"
        )
    return (
        bands.reset_index(drop=True),
        residue.reset_index(drop=True),
        protein_summary.reset_index(drop=True),
    )


def attach_test_strain(
    residue: pd.DataFrame,
    strain_root: str | None,
    terminal_exclusion: int,
    high_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach strictly indexed MDStrainMapper values to test residues only."""
    output = residue.copy()
    for metric in STRAIN_METRICS:
        output[metric] = np.nan
    if not strain_root:
        return output, pd.DataFrame(columns=[
            "split", "protein", "strain_status", "strain_source",
            "protein_length", "n_file_rows", "n_finite_before_terminal_exclusion",
            "n_finite_after_terminal_exclusion", "terminal_exclusion",
            "high_quantile", "high_threshold", "n_high_strain_residues",
        ])

    root = Path(strain_root).expanduser()
    if not root.is_dir():
        raise ValueError(
            f"--strain_root must be an existing local directory, got {root}. "
            "Stage the remote directory locally or run this analysis on that host."
        )

    audit_rows: list[dict] = []
    test = output[output["split"].astype(str) == "test"]
    for protein, group in test.groupby("protein", sort=False):
        protein = str(protein)
        ordered = group.sort_values("residue_index_0based")
        length = len(ordered)
        coordinates = pd.to_numeric(
            ordered["residue_index_0based"], errors="coerce"
        ).to_numpy()
        if not np.array_equal(coordinates, np.arange(length)):
            raise ValueError(f"test/{protein}: non-contiguous residue annotations")

        path = root / protein / "strain_summary.csv"
        audit = {
            "split": "test",
            "protein": protein,
            "strain_status": "missing",
            "strain_source": str(path.resolve()),
            "protein_length": length,
            "n_file_rows": 0,
            "n_finite_before_terminal_exclusion": 0,
            "n_finite_after_terminal_exclusion": 0,
            "terminal_exclusion": terminal_exclusion,
            "high_quantile": high_quantile,
            "high_threshold": np.nan,
            "n_high_strain_residues": 0,
        }
        if not path.exists():
            audit_rows.append(audit)
            continue

        data = pd.read_csv(path)
        audit["n_file_rows"] = len(data)
        required = {"residue", "ensemble_mean", "ensemble_std"}
        missing = required - set(data.columns)
        if missing:
            audit["strain_status"] = "invalid_schema"
            audit["error"] = f"missing columns: {sorted(missing)}"
            audit_rows.append(audit)
            continue
        data = data.sort_values("residue").reset_index(drop=True)
        file_coordinates = pd.to_numeric(data["residue"], errors="coerce").to_numpy()
        if len(data) != length or not np.array_equal(
            file_coordinates, np.arange(1, length + 1)
        ):
            audit["strain_status"] = "index_or_length_mismatch"
            audit["error"] = (
                "expected exactly one row for residues 1..protein_length"
            )
            audit_rows.append(audit)
            continue

        strain = pd.to_numeric(data["ensemble_mean"], errors="coerce").to_numpy(float)
        strain_std = pd.to_numeric(data["ensemble_std"], errors="coerce").to_numpy(float)
        audit["n_finite_before_terminal_exclusion"] = int(np.isfinite(strain).sum())
        if 2 * terminal_exclusion >= length:
            audit["strain_status"] = "too_short_after_terminal_exclusion"
            audit["error"] = "terminal exclusion removes every residue"
            audit_rows.append(audit)
            continue
        if terminal_exclusion:
            strain[:terminal_exclusion] = np.nan
            strain[-terminal_exclusion:] = np.nan
            strain_std[:terminal_exclusion] = np.nan
            strain_std[-terminal_exclusion:] = np.nan

        valid = np.isfinite(strain)
        audit["n_finite_after_terminal_exclusion"] = int(valid.sum())
        if not valid.any():
            audit["strain_status"] = "no_finite_interior_values"
            audit_rows.append(audit)
            continue

        gradient = np.full(length, np.nan)
        adjacent = valid[1:] & valid[:-1]
        adjacent_indices = np.flatnonzero(adjacent) + 1
        gradient[adjacent_indices] = np.abs(np.diff(strain)[adjacent])
        threshold = float(np.nanquantile(strain, high_quantile))
        high = np.full(length, np.nan)
        high[valid] = (strain[valid] >= threshold).astype(float)
        high_indices = np.flatnonzero(high == 1)
        distance = np.full(length, np.nan)
        valid_indices = np.flatnonzero(valid)
        if len(high_indices):
            distance[valid_indices] = np.min(
                np.abs(valid_indices[:, None] - high_indices[None, :]), axis=1
            )

        values = {
            "strain_ensemble_mean": strain,
            "strain_ensemble_std": strain_std,
            "strain_abs_gradient_from_previous": gradient,
            "strain_top_quantile_within_protein": high,
            "distance_to_strain_top_quantile": distance,
        }
        for name, array in values.items():
            output.loc[ordered.index, name] = array
        audit.update({
            "strain_status": "ok",
            "high_threshold": threshold,
            "n_high_strain_residues": int(len(high_indices)),
        })
        audit_rows.append(audit)

    return output, pd.DataFrame(audit_rows)


def feature_matrix(frame: pd.DataFrame, metric_names: list[str]) -> np.ndarray:
    return frame[metric_names].astype(float).to_numpy()


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = np.mean(values[draws], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def sign_flip_test(
    differences: np.ndarray,
    n_flips: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    observed = float(np.mean(differences))
    null_parts = []
    remaining = n_flips
    while remaining:
        size = min(1000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(differences)))
        null_parts.append(np.mean(signs * differences[None, :], axis=1))
        remaining -= size
    null = np.concatenate(null_parts)
    return (
        observed,
        z_score(observed, null),
        empirical_p(observed, null, "upper"),
        empirical_p(observed, null, "lower"),
        empirical_p(observed, null, "two-sided"),
    )


def expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    return np.convolve(
        mask.astype(np.int8), np.ones(2 * radius + 1, dtype=np.int8), mode="same"
    ) > 0


def true_runs(mask: np.ndarray, start: int, end: int) -> list[np.ndarray]:
    """Return contiguous true runs intersected with [start, end)."""
    positions = np.flatnonzero(np.asarray(mask, dtype=bool)[start:end]) + start
    if len(positions) == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1) + 1
    return [part for part in np.split(positions, breaks) if len(part)]


def classification_counts(
    target: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(target & predicted))
    fp = int(np.sum(~target & predicted))
    fn = int(np.sum(target & ~predicted))
    tn = int(np.sum(~target & ~predicted))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    balanced = (
        (recall + specificity) / 2
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )
    denominator = np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": mcc,
    }


def phase1_coverage_and_identifier(
    bands: pd.DataFrame,
    residue: pd.DataFrame,
    protein_summary: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure annotation->band coverage and fixed-band identifier quality."""
    residue_lookup = {
        (split, protein): group.sort_values("residue_index_0based").reset_index(drop=True)
        for (split, protein), group in residue.groupby(["split", "protein"], sort=False)
    }
    band_lookup = {
        key: group
        for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    protein_rows: list[dict] = []
    for context in protein_summary.itertuples(index=False):
        condition = str(context.condition)
        split = str(context.split)
        protein = str(context.protein)
        annotation = residue_lookup.get((split, protein))
        if annotation is None:
            raise ValueError(f"{condition}/{split}/{protein}: missing residue annotations")
        length = len(annotation)
        if length != int(context.protein_length):
            raise ValueError(f"{condition}/{split}/{protein}: protein length mismatch")
        start = int(context.eligible_start_index_0based)
        end = int(context.eligible_end_index_0based_exclusive)
        if not 0 <= start < end <= length:
            raise ValueError(f"{condition}/{split}/{protein}: invalid eligible interval")
        eligible = np.zeros(length, dtype=bool)
        eligible[start:end] = True
        protein_bands = band_lookup.get((condition, split, protein))
        if protein_bands is None:
            protein_bands = bands.iloc[0:0]

        q8_cts = annotation["q8"].astype(str).isin(list("CTS")).to_numpy()
        neq_peak = annotation["neq_peak"].astype(bool).to_numpy()
        targets = {
            "q8_loop_turn_bend_CTS": (q8_cts, np.ones(length, dtype=bool)),
            "neq_peak": (neq_peak, np.ones(length, dtype=bool)),
        }
        if "strain_top_quantile_within_protein" in annotation:
            strain_high = pd.to_numeric(
                annotation["strain_top_quantile_within_protein"], errors="coerce"
            ).to_numpy(dtype=float)
            strain_support = np.isfinite(strain_high)
            if np.any(strain_support):
                targets["strain_top_quantile_within_protein"] = (
                    strain_high == 1,
                    strain_support,
                )
        for sign, label in SIGN_LABELS.items():
            sign_bands = protein_bands[protein_bands["sign"].astype(int) == sign]
            apex = np.zeros(length, dtype=bool)
            interval = np.zeros(length, dtype=bool)
            if not sign_bands.empty:
                apex[sign_bands["apex_index_0based"].astype(int).to_numpy()] = True
                for band in sign_bands.itertuples(index=False):
                    interval[
                        int(band.start_index_0based):
                        int(band.end_index_0based_inclusive) + 1
                    ] = True
            apex &= eligible
            interval &= eligible
            methods_by_target = {
                "q8_loop_turn_bend_CTS": {
                    "apex_exact": apex,
                    "band_interval": interval,
                },
                "neq_peak": {
                    "apex_exact": apex,
                    "apex_within1": expand_mask(apex, 1) & eligible,
                    "apex_within2": expand_mask(apex, 2) & eligible,
                    "apex_within5": expand_mask(apex, 5) & eligible,
                    "band_interval": interval,
                },
            }
            if "strain_top_quantile_within_protein" in targets:
                methods_by_target["strain_top_quantile_within_protein"] = {
                    "apex_exact": apex,
                    "apex_within1": expand_mask(apex, 1) & eligible,
                    "apex_within2": expand_mask(apex, 2) & eligible,
                    "apex_within5": expand_mask(apex, 5) & eligible,
                    "band_interval": interval,
                }
            base = {
                "condition": condition,
                "split": split,
                "protein": protein,
                "sign": sign,
                "label": label,
                "eligible_residue_count": int(np.sum(eligible)),
                "n_bands": int(len(sign_bands)),
            }
            for target_name, (target_full, target_support) in targets.items():
                analysis_mask = eligible & target_support
                if not np.any(analysis_mask):
                    continue
                target = target_full[analysis_mask]
                for method, predicted_full in methods_by_target[target_name].items():
                    predicted = predicted_full[analysis_mask]
                    scores = classification_counts(target, predicted)
                    protein_rows.append({
                        **base,
                        "analysis_residue_count": int(np.sum(analysis_mask)),
                        "target": target_name,
                        "unit_type": "residue",
                        "detection_method": method,
                        "n_target_units": int(np.sum(target)),
                        "n_detected_target_units": int(np.sum(target & predicted)),
                        "coverage": scores["recall"],
                        "n_predicted_units": int(np.sum(predicted)),
                        "target_prevalence": float(np.mean(target)),
                        **scores,
                    })

            # Segment-level recovery complements residue-level Q8 recovery.
            for method, detector in (("apex_any", apex), ("band_interval_any", interval)):
                segments = true_runs(q8_cts, start, end)
                hits = int(sum(bool(np.any(detector[segment])) for segment in segments))
                protein_rows.append({
                    **base,
                    "analysis_residue_count": int(np.sum(eligible)),
                    "target": "q8_loop_turn_bend_CTS",
                    "unit_type": "segment",
                    "detection_method": method,
                    "n_target_units": int(len(segments)),
                    "n_detected_target_units": hits,
                    "coverage": hits / len(segments) if segments else np.nan,
                    "n_predicted_units": np.nan,
                    "target_prevalence": np.nan,
                    **{name: np.nan for name in (
                        "tp", "fp", "fn", "tn", "precision", "recall",
                        "specificity", "f1", "balanced_accuracy", "mcc",
                    )},
                })

    per_protein = pd.DataFrame(protein_rows)
    summary_rows: list[dict] = []
    group_columns = [
        "condition", "split", "sign", "label", "target", "unit_type",
        "detection_method",
    ]
    macro_metrics = [
        "coverage", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "mcc", "target_prevalence",
    ]
    for key, group in per_protein.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, key))
        row["n_proteins_total"] = int(group["protein"].nunique())
        row["n_proteins_with_target"] = int(np.sum(group["n_target_units"] > 0))
        row["n_proteins_with_prediction"] = int(
            np.sum(pd.to_numeric(group["n_predicted_units"], errors="coerce").fillna(0) > 0)
        )
        row["total_target_units"] = int(group["n_target_units"].sum())
        row["total_detected_target_units"] = int(group["n_detected_target_units"].sum())
        row["micro_coverage"] = (
            row["total_detected_target_units"] / row["total_target_units"]
            if row["total_target_units"] else np.nan
        )
        for metric in macro_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"macro_{metric}"] = float(np.mean(finite)) if len(finite) else np.nan
            lower, upper = bootstrap_mean_ci(finite, n_bootstrap, rng)
            row[f"macro_{metric}_ci95_low"] = lower
            row[f"macro_{metric}_ci95_high"] = upper
        if str(row["unit_type"]) == "residue":
            tp, fp, fn, tn = (int(group[name].sum()) for name in ("tp", "fp", "fn", "tn"))
            micro_precision = tp / (tp + fp) if tp + fp else np.nan
            micro_recall = tp / (tp + fn) if tp + fn else np.nan
            micro_specificity = tn / (tn + fp) if tn + fp else np.nan
            micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
            denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            row.update({
                "micro_tp": tp, "micro_fp": fp, "micro_fn": fn, "micro_tn": tn,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_specificity": micro_specificity,
                "micro_f1": micro_f1,
                "micro_balanced_accuracy": (
                    (micro_recall + micro_specificity) / 2
                    if np.isfinite(micro_recall) and np.isfinite(micro_specificity)
                    else np.nan
                ),
                "micro_mcc": (tp * tn - fp * fn) / denominator if denominator else np.nan,
            })
        summary_rows.append(row)
    return per_protein, pd.DataFrame(summary_rows)


def matched_outcome_values(frame: pd.DataFrame, metric: str) -> np.ndarray:
    if metric.startswith("q8_is_"):
        label = metric.rsplit("_", 1)[-1]
        return (frame["q8"].astype(str).to_numpy() == label).astype(float)
    return pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)


def phase2_within_q3_matching(
    bands: pd.DataFrame,
    residue: pd.DataFrame,
    protein_summary: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    matched_outcome_definitions: dict[str, tuple[str, str]],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    """Match every apex to same-protein non-band residues with identical Q3."""
    residue_lookup = {
        (split, protein): group.sort_values("residue_index_0based").reset_index(drop=True)
        for (split, protein), group in residue.groupby(["split", "protein"], sort=False)
    }
    band_lookup = {
        key: group.sort_values("apex_index_0based")
        for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    outcome_names = list(matched_outcome_definitions)
    match_rows: list[dict] = []
    case_rows: list[dict] = []
    match_tallies: dict[tuple, dict[str, object]] = {}
    # Per-protein sums preserve equal protein weight in the inferential stage.
    effect_accumulator: dict[tuple, dict[str, np.ndarray | int]] = {}

    for context in protein_summary.itertuples(index=False):
        condition = str(context.condition)
        split = str(context.split)
        protein = str(context.protein)
        protein_bands = band_lookup.get((condition, split, protein))
        if protein_bands is None or protein_bands.empty:
            continue
        annotation = residue_lookup[(split, protein)]
        length = len(annotation)
        start = int(context.eligible_start_index_0based)
        end = int(context.eligible_end_index_0based_exclusive)
        span = end - start
        eligible = np.zeros(length, dtype=bool)
        eligible[start:end] = True
        excluded = np.zeros(length, dtype=bool)
        for band in protein_bands.itertuples(index=False):
            left = max(0, int(band.start_index_0based) - args.match_band_buffer)
            right = min(
                length,
                int(band.end_index_0based_inclusive) + 1 + args.match_band_buffer,
            )
            excluded[left:right] = True
        candidate_base = eligible & ~excluded
        positions = np.arange(length)
        normalized_position = (positions - start) / max(1, span - 1)
        neq = pd.to_numeric(annotation["neq"], errors="coerce").to_numpy(dtype=float)
        rsa = pd.to_numeric(annotation["rsa"], errors="coerce").to_numpy(dtype=float)
        q3 = annotation["q3"].astype(str).to_numpy()
        q8 = annotation["q8"].astype(str).to_numpy()
        direct_outcomes = [
            metric for metric in outcome_names if metric != "normalized_position"
        ]
        outcome_arrays = {
            metric: matched_outcome_values(annotation, metric)
            for metric in direct_outcomes
        }
        outcome_arrays["normalized_position"] = normalized_position.astype(float)

        match_schemes = {
            name: MATCH_SCHEMES[name] for name in args.match_schemes
        }
        for band in protein_bands.itertuples(index=False):
            case_index = int(band.apex_index_0based)
            sign = int(band.sign)
            label = str(band.label)
            q3_label = str(q3[case_index])
            for match_scheme, scheme in match_schemes.items():
                use_neq = bool(scheme["use_neq"])
                use_rsa = bool(scheme["use_rsa"])
                use_position = bool(scheme["use_position"])
                use_all_controls = bool(scheme["use_all_controls"])
                tally_key = (match_scheme, condition, split, sign, label, q3_label)
                tally = match_tallies.setdefault(
                    tally_key,
                    {
                        "n_cases": 0, "n_matched_cases": 0,
                        "n_controls": 0, "proteins": set(),
                    },
                )
                tally["n_cases"] = int(tally["n_cases"]) + 1
                cast_proteins = tally["proteins"]
                assert isinstance(cast_proteins, set)
                cast_proteins.add(protein)

                candidate = candidate_base & (q3 == q3_label)
                distance_squared = np.zeros(length, dtype=float)
                if use_neq:
                    candidate &= np.isfinite(neq)
                    candidate &= np.abs(neq - neq[case_index]) <= args.match_neq_caliper
                    distance_squared += (
                        (neq - neq[case_index]) / args.match_neq_caliper
                    ) ** 2
                if use_rsa:
                    candidate &= np.isfinite(rsa)
                    candidate &= np.abs(rsa - rsa[case_index]) <= args.match_rsa_caliper
                    distance_squared += (
                        (rsa - rsa[case_index]) / args.match_rsa_caliper
                    ) ** 2
                if use_position:
                    candidate &= (
                        np.abs(normalized_position - normalized_position[case_index])
                        <= args.match_position_caliper
                    )
                    distance_squared += (
                        (normalized_position - normalized_position[case_index])
                        / args.match_position_caliper
                    ) ** 2
                candidate_indices = np.flatnonzero(candidate)
                if len(candidate_indices) == 0:
                    continue
                candidate_distances = np.sqrt(distance_squared[candidate_indices])
                order = np.lexsort((candidate_indices, candidate_distances))
                if use_all_controls:
                    selected = candidate_indices[order]
                    selected_distances = candidate_distances[order]
                else:
                    selection_order = order[: args.match_controls_per_apex]
                    selected = candidate_indices[selection_order]
                    selected_distances = candidate_distances[selection_order]
                tally["n_matched_cases"] = int(tally["n_matched_cases"]) + 1
                tally["n_controls"] = int(tally["n_controls"]) + len(selected)

                case_base = {
                    "match_scheme": match_scheme,
                    "band_id": str(band.band_id),
                    "condition": condition,
                    "split": split,
                    "protein": protein,
                    "sign": sign,
                    "label": label,
                    "q3": q3_label,
                    "case_index_0based": case_index,
                    "case_residue_1based": case_index + 1,
                    "case_q8": str(q8[case_index]),
                    "n_controls": int(len(selected)),
                    "case_neq": float(neq[case_index]),
                    "control_mean_neq": float(np.mean(neq[selected])),
                    "case_rsa": float(rsa[case_index]),
                    "control_mean_rsa": float(np.mean(rsa[selected])),
                    "case_normalized_position": float(normalized_position[case_index]),
                    "control_mean_normalized_position": float(
                        np.mean(normalized_position[selected])
                    ),
                }
                case_rows.append(case_base)
                # q3_only can contain dozens or hundreds of controls per apex.
                # Its complete set is represented losslessly by the case-level
                # count/means below and used directly in the effect accumulator;
                # individual rows are retained for the capped caliper schemes.
                if not use_all_controls:
                    for rank, control_index in enumerate(selected, start=1):
                        match_rows.append({
                            **{key: case_base[key] for key in (
                                "match_scheme", "band_id", "condition", "split",
                                "protein", "sign", "label", "q3", "case_index_0based",
                                "case_residue_1based", "case_q8",
                            )},
                            "control_rank": rank,
                            "control_index_0based": int(control_index),
                            "control_residue_1based": int(control_index + 1),
                            "control_q8": str(q8[control_index]),
                            "match_distance": float(selected_distances[rank - 1]),
                            "case_neq": float(neq[case_index]),
                            "control_neq": float(neq[control_index]),
                            "case_rsa": float(rsa[case_index]),
                            "control_rsa": float(rsa[control_index]),
                            "case_normalized_position": float(
                                normalized_position[case_index]
                            ),
                            "control_normalized_position": float(normalized_position[control_index]),
                        })

                effect_key = (
                    match_scheme, condition, split, protein, sign, label, q3_label
                )
                accumulator = effect_accumulator.setdefault(effect_key, {
                    "case_sum": np.zeros(len(outcome_names), dtype=float),
                    "control_sum": np.zeros(len(outcome_names), dtype=float),
                    "difference_sum": np.zeros(len(outcome_names), dtype=float),
                    "counts": np.zeros(len(outcome_names), dtype=np.int32),
                })
                for metric_index, metric in enumerate(outcome_names):
                    values = outcome_arrays[metric]
                    case_value = values[case_index]
                    control_values = values[selected]
                    finite_controls = control_values[np.isfinite(control_values)]
                    if not np.isfinite(case_value) or len(finite_controls) == 0:
                        continue
                    control_mean = float(np.mean(finite_controls))
                    accumulator["case_sum"][metric_index] += case_value
                    accumulator["control_sum"][metric_index] += control_mean
                    accumulator["difference_sum"][metric_index] += case_value - control_mean
                    accumulator["counts"][metric_index] += 1

    matches = pd.DataFrame(match_rows)
    matched_cases = pd.DataFrame(case_rows)

    tally_rows = []
    for key, values in match_tallies.items():
        match_scheme, condition, split, sign, label, q3_label = key
        n_cases = int(values["n_cases"])
        n_matched = int(values["n_matched_cases"])
        tally_rows.append({
            "match_scheme": match_scheme,
            "condition": condition,
            "split": split,
            "sign": sign,
            "label": label,
            "q3": q3_label,
            "n_proteins_with_cases": len(values["proteins"]),
            "n_cases": n_cases,
            "n_matched_cases": n_matched,
            "match_rate": n_matched / n_cases if n_cases else np.nan,
            "n_controls_selected": int(values["n_controls"]),
            "mean_controls_per_matched_case": (
                int(values["n_controls"]) / n_matched if n_matched else np.nan
            ),
        })
    match_coverage = pd.DataFrame(tally_rows)

    balance_rows = []
    calipers = {
        "neq": args.match_neq_caliper,
        "rsa": args.match_rsa_caliper,
        "normalized_position": args.match_position_caliper,
    }
    balance_groups = [
        "match_scheme", "condition", "split", "sign", "label", "q3"
    ]
    for key, group in matched_cases.groupby(balance_groups, sort=False):
        match_scheme = str(key[0])
        for covariate, caliper in calipers.items():
            case_values = group[f"case_{covariate}"].to_numpy(dtype=float)
            control_values = group[f"control_mean_{covariate}"].to_numpy(dtype=float)
            difference = case_values - control_values
            pooled_sd = np.sqrt(
                (np.var(case_values, ddof=1) + np.var(control_values, ddof=1)) / 2
            ) if len(group) > 1 else np.nan
            balance_rows.append({
                **dict(zip(balance_groups, key)),
                "covariate": covariate,
                "matched_on_covariate": bool(
                    (covariate == "neq" and match_scheme != "q3_only")
                    or (covariate == "rsa" and "rsa" in match_scheme)
                    or (
                        covariate == "normalized_position"
                        and match_scheme.endswith("position")
                    )
                ),
                "n_matched_cases": len(group),
                "case_mean": float(np.mean(case_values)),
                "matched_control_mean": float(np.mean(control_values)),
                "paired_mean_difference": float(np.mean(difference)),
                "paired_mean_absolute_difference": float(np.mean(np.abs(difference))),
                "standardized_mean_difference": (
                    float(np.mean(difference) / pooled_sd)
                    if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
                ),
                "mean_absolute_difference_over_caliper": float(
                    np.mean(np.abs(difference)) / caliper
                ),
                "caliper": caliper,
            })
    balance = pd.DataFrame(balance_rows)

    per_protein_rows = []
    for key, accumulator in effect_accumulator.items():
        match_scheme, condition, split, protein, sign, label, q3_label = key
        for metric_index, metric in enumerate(outcome_names):
            count = int(accumulator["counts"][metric_index])
            if count == 0:
                continue
            per_protein_rows.append({
                "match_scheme": match_scheme,
                "condition": condition,
                "split": split,
                "protein": protein,
                "sign": sign,
                "label": label,
                "q3": q3_label,
                "metric": metric,
                "metric_kind": matched_outcome_definitions[metric][0],
                "metric_description": matched_outcome_definitions[metric][1],
                "n_matched_apices": count,
                "case_mean": float(accumulator["case_sum"][metric_index] / count),
                "matched_control_mean": float(
                    accumulator["control_sum"][metric_index] / count
                ),
                "case_minus_control": float(
                    accumulator["difference_sum"][metric_index] / count
                ),
            })
    per_protein = pd.DataFrame(per_protein_rows)

    summary_rows = []
    summary_groups = [
        "match_scheme", "condition", "split", "sign", "label", "q3", "metric"
    ]
    for key, group in per_protein.groupby(summary_groups, sort=False):
        differences = group["case_minus_control"].to_numpy(dtype=float)
        observed, statistic, p_upper, p_lower, p_two = sign_flip_test(
            differences, args.n_sign_flips, rng
        )
        lower, upper = bootstrap_mean_ci(differences, args.n_bootstrap, rng)
        summary_rows.append({
            **dict(zip(summary_groups, key)),
            "metric_kind": str(group["metric_kind"].iloc[0]),
            "metric_description": str(group["metric_description"].iloc[0]),
            "n_proteins": int(group["protein"].nunique()),
            "n_matched_apices": int(group["n_matched_apices"].sum()),
            "case_macro_protein_mean": float(group["case_mean"].mean()),
            "matched_control_macro_protein_mean": float(
                group["matched_control_mean"].mean()
            ),
            "case_minus_control_macro_mean": observed,
            "case_minus_control_ci95_low": lower,
            "case_minus_control_ci95_high": upper,
            "sign_flip_z": statistic,
            "p_upper": p_upper,
            "p_lower": p_lower,
            "p_two_sided": p_two,
        })
    enrichment = pd.DataFrame(summary_rows)
    if not enrichment.empty:
        for column in ("p_upper", "p_lower", "p_two_sided"):
            enrichment[f"{column}_q_bh"] = benjamini_hochberg(enrichment[column])
    return matches, matched_cases, match_coverage, balance, per_protein, enrichment


def main() -> None:
    args = parse_args()
    if args.n_block_shifts < 1 or args.n_sign_flips < 1 or args.n_bootstrap < 1:
        raise ValueError("Permutation and bootstrap counts must be positive")
    if args.match_controls_per_apex < 1 or args.match_band_buffer < 0:
        raise ValueError("Invalid matched-control count or band buffer")
    if min(
        args.match_neq_caliper,
        args.match_rsa_caliper,
        args.match_position_caliper,
    ) <= 0:
        raise ValueError("All matching calipers must be positive")
    if args.strain_terminal_exclusion < 0:
        raise ValueError("--strain_terminal_exclusion must be nonnegative")
    if not 0 < args.strain_high_quantile < 1:
        raise ValueError("--strain_high_quantile must lie strictly between 0 and 1")
    bands, residue, protein_summary = load_inputs(args)
    residue, strain_audit = attach_test_strain(
        residue,
        args.strain_root,
        args.strain_terminal_exclusion,
        args.strain_high_quantile,
    )
    metric_definitions = dict(METRICS)
    matched_outcome_definitions = dict(ALL_MATCHED_OUTCOMES)
    if args.strain_root:
        metric_definitions.update(STRAIN_METRICS)
        matched_outcome_definitions.update(STRAIN_MATCHED_OUTCOMES)
    rng = np.random.default_rng(args.random_seed)
    metric_names = list(metric_definitions)
    metric_count = len(metric_names)
    strain_metric_indices = [
        index for index, metric in enumerate(metric_names) if metric in STRAIN_METRICS
    ]
    residue_lookup = {
        (split, protein): group.sort_values("residue_index_0based").reset_index(drop=True)
        for (split, protein), group in residue.groupby(["split", "protein"], sort=False)
    }

    group_columns = ["condition", "split", "sign", "label"]
    group_keys = list(bands.groupby(group_columns, sort=False).groups)
    observed_by_group = {key: [[] for _ in metric_names] for key in group_keys}
    null_sums = {
        key: np.zeros((args.n_block_shifts, metric_count), dtype=float)
        for key in group_keys
    }
    null_counts = {
        key: np.zeros((args.n_block_shifts, metric_count), dtype=np.int32)
        for key in group_keys
    }
    protein_rows = []

    protein_group_columns = ["condition", "split", "protein", "sign", "label"]
    for group_key, group in bands.groupby(protein_group_columns, sort=False):
        condition, split, protein, sign, label = group_key
        aggregate_key = (condition, split, sign, label)
        annotation = residue_lookup[(split, protein)]
        features = feature_matrix(annotation, metric_names)
        eligible_starts = group["eligible_start_index_0based"].astype(int).unique()
        eligible_ends = group["eligible_end_index_0based_exclusive"].astype(int).unique()
        if len(eligible_starts) != 1 or len(eligible_ends) != 1:
            raise ValueError(f"{group_key}: inconsistent eligible intervals")
        eligible_start = int(eligible_starts[0])
        eligible_end = int(eligible_ends[0])
        span = eligible_end - eligible_start
        if span <= 0:
            raise ValueError(f"{group_key}: empty eligible interval")
        apices = group["apex_index_0based"].astype(int).to_numpy()
        if np.any(apices < eligible_start) or np.any(apices >= eligible_end):
            raise ValueError(f"{group_key}: apex outside eligible interval")
        original_values = features[apices]
        original_valid = np.isfinite(original_values)
        original_counts = np.sum(original_valid, axis=0)
        protein_mean = np.divide(
            np.nansum(original_values, axis=0),
            original_counts,
            out=np.full(metric_count, np.nan, dtype=float),
            where=original_counts > 0,
        )
        row = {
            "condition": condition,
            "split": split,
            "protein": protein,
            "sign": int(sign),
            "label": label,
            "n_bands": len(group),
        }
        for index, metric in enumerate(metric_names):
            value = protein_mean[index]
            row[metric] = value
            if np.isfinite(value):
                observed_by_group[aggregate_key][index].append(float(value))
        protein_rows.append(row)

        offsets = rng.integers(0, span, size=args.n_block_shifts)
        shifted = eligible_start + (
            (apices[None, :] - eligible_start + offsets[:, None]) % span
        )
        shifted_values = features[shifted]
        valid = np.isfinite(shifted_values)
        sums = np.nansum(shifted_values, axis=1)
        counts = np.sum(valid, axis=1)
        shifted_means = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan),
            where=counts > 0,
        )
        # Strain excludes terminal residues. Shift the finite-apex subset in
        # rank space over the finite strain support so null patterns never gain
        # or lose observations merely by landing in an excluded terminus.
        if strain_metric_indices:
            anchor_index = metric_names.index("strain_ensemble_mean")
            strain_support = np.flatnonzero(
                np.isfinite(features[:, anchor_index])
                & (np.arange(len(features)) >= eligible_start)
                & (np.arange(len(features)) < eligible_end)
            )
            if len(strain_support):
                support_rank = np.full(len(features), -1, dtype=int)
                support_rank[strain_support] = np.arange(len(strain_support))
                apex_ranks = support_rank[apices]
                apex_ranks = apex_ranks[apex_ranks >= 0]
            else:
                apex_ranks = np.array([], dtype=int)
            if len(apex_ranks):
                strain_offsets = rng.integers(
                    0, len(strain_support), size=args.n_block_shifts
                )
                shifted_strain_positions = strain_support[
                    (apex_ranks[None, :] + strain_offsets[:, None])
                    % len(strain_support)
                ]
                for metric_index in strain_metric_indices:
                    values = features[shifted_strain_positions, metric_index]
                    valid_values = np.isfinite(values)
                    value_counts = np.sum(valid_values, axis=1)
                    shifted_means[:, metric_index] = np.divide(
                        np.nansum(values, axis=1),
                        value_counts,
                        out=np.full(args.n_block_shifts, np.nan),
                        where=value_counts > 0,
                    )
            else:
                shifted_means[:, strain_metric_indices] = np.nan
        valid_means = np.isfinite(shifted_means)
        null_sums[aggregate_key] += np.nan_to_num(shifted_means, nan=0.0)
        null_counts[aggregate_key] += valid_means.astype(np.int32)

    per_protein = pd.DataFrame(protein_rows)
    summary_rows = []
    null_rows = []
    for key in group_keys:
        condition, split, sign, label = key
        macro_null = np.divide(
            null_sums[key],
            null_counts[key],
            out=np.full_like(null_sums[key], np.nan),
            where=null_counts[key] > 0,
        )
        for metric_index, metric in enumerate(metric_names):
            values = np.asarray(observed_by_group[key][metric_index], dtype=float)
            observed = float(np.mean(values)) if len(values) else np.nan
            null = macro_null[:, metric_index]
            finite_null = null[np.isfinite(null)]
            null_mean = float(np.mean(finite_null)) if len(finite_null) else np.nan
            kind, hypothesis = metric_definitions[metric]
            row = {
                "condition": condition,
                "split": split,
                "sign": int(sign),
                "label": label,
                "metric": metric,
                "metric_kind": kind,
                "directional_hypothesis": hypothesis,
                "n_proteins": len(values),
                "observed_macro_protein_mean": observed,
                "null_mean": null_mean,
                "delta": observed - null_mean,
                "ratio_observed_to_null": (
                    observed / null_mean if np.isfinite(null_mean) and null_mean != 0 else np.nan
                ),
                "null_std": (
                    float(np.std(finite_null, ddof=1))
                    if len(finite_null) > 1 else np.nan
                ),
                "null_z": z_score(observed, null),
                "p_upper": empirical_p(observed, null, "upper"),
                "p_lower": empirical_p(observed, null, "lower"),
                "p_two_sided": empirical_p(observed, null, "two-sided"),
            }
            summary_rows.append(row)
            for permutation, value in enumerate(null, start=1):
                null_rows.append({
                    "condition": condition,
                    "split": split,
                    "sign": int(sign),
                    "label": label,
                    "metric": metric,
                    "permutation": permutation,
                    "macro_protein_mean": value,
                    "n_contributing_proteins": int(null_counts[key][permutation - 1, metric_index]),
                })

    summary = pd.DataFrame(summary_rows)
    for column in ("p_upper", "p_lower", "p_two_sided"):
        summary[f"{column}_q_bh"] = benjamini_hochberg(summary[column])

    # Direct flexibility-supporting minus rigidity-supporting paired contrast.
    contrast_rows = []
    for (condition, split), group in per_protein.groupby(["condition", "split"], sort=False):
        for metric in metric_names:
            pivot = group.pivot_table(index="protein", columns="sign", values=metric, aggfunc="first")
            if -1 not in pivot or 1 not in pivot:
                continue
            differences = (pivot[1] - pivot[-1]).dropna().to_numpy(dtype=float)
            if len(differences) == 0:
                continue
            observed = float(np.mean(differences))
            batch = 1000
            null_parts = []
            remaining = args.n_sign_flips
            while remaining:
                size = min(batch, remaining)
                signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(differences)))
                null_parts.append(np.mean(signs * differences[None, :], axis=1))
                remaining -= size
            null = np.concatenate(null_parts)
            kind, hypothesis = metric_definitions[metric]
            contrast_rows.append({
                "condition": condition,
                "split": split,
                "metric": metric,
                "metric_kind": kind,
                "directional_hypothesis": hypothesis,
                "n_paired_proteins": len(differences),
                "flex_minus_rigid_mean": observed,
                "flex_minus_rigid_median": float(np.median(differences)),
                "sign_flip_null_mean": float(np.mean(null)),
                "sign_flip_null_std": float(np.std(null, ddof=1)),
                "sign_flip_z": z_score(observed, null),
                "p_upper": empirical_p(observed, null, "upper"),
                "p_lower": empirical_p(observed, null, "lower"),
                "p_two_sided": empirical_p(observed, null, "two-sided"),
            })
    contrasts = pd.DataFrame(contrast_rows)
    for column in ("p_upper", "p_lower", "p_two_sided"):
        contrasts[f"{column}_q_bh"] = benjamini_hochberg(contrasts[column])

    phase1_per_protein, phase1_summary = phase1_coverage_and_identifier(
        bands,
        residue,
        protein_summary,
        args.n_bootstrap,
        np.random.default_rng(args.random_seed + 1001),
    )
    (
        matched_controls,
        matched_cases,
        match_coverage,
        match_balance,
        matched_per_protein,
        matched_summary,
    ) = phase2_within_q3_matching(
        bands,
        residue,
        protein_summary,
        args,
        np.random.default_rng(args.random_seed + 2001),
        matched_outcome_definitions,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_protein_path = output_dir / "apex_metrics_by_protein.csv.gz"
    null_path = output_dir / "apex_circular_shift_null.csv.gz"
    summary_path = output_dir / "apex_circular_shift_enrichment_summary.csv"
    contrast_path = output_dir / "paired_flex_vs_rigid_summary.csv"
    phase1_per_protein_path = output_dir / "annotation_band_coverage_by_protein.csv.gz"
    phase1_summary_path = output_dir / "annotation_band_coverage_identifier_summary.csv"
    matched_controls_path = output_dir / "within_q3_matched_controls.csv.gz"
    matched_cases_path = output_dir / "within_q3_matched_cases.csv.gz"
    match_coverage_path = output_dir / "within_q3_match_coverage_summary.csv"
    match_balance_path = output_dir / "within_q3_match_balance.csv"
    matched_per_protein_path = output_dir / "within_q3_matched_effects_by_protein.csv.gz"
    matched_summary_path = output_dir / "within_q3_matched_enrichment_summary.csv"
    strain_audit_path = output_dir / "strain_input_audit.csv"
    per_protein.to_csv(per_protein_path, index=False, compression="gzip")
    pd.DataFrame(null_rows).to_csv(null_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    phase1_per_protein.to_csv(
        phase1_per_protein_path, index=False, compression="gzip"
    )
    phase1_summary.to_csv(phase1_summary_path, index=False)
    matched_controls.to_csv(matched_controls_path, index=False, compression="gzip")
    matched_cases.to_csv(matched_cases_path, index=False, compression="gzip")
    match_coverage.to_csv(match_coverage_path, index=False)
    match_balance.to_csv(match_balance_path, index=False)
    matched_per_protein.to_csv(
        matched_per_protein_path, index=False, compression="gzip"
    )
    matched_summary.to_csv(matched_summary_path, index=False)
    if args.strain_root:
        strain_audit.to_csv(strain_audit_path, index=False)
    parameters = {
        "annotated_bands_csv": str(Path(args.annotated_bands_csv).expanduser().resolve()),
        "residue_annotations_csv": str(Path(args.residue_annotations_csv).expanduser().resolve()),
        "protein_summary_csv": str(Path(args.protein_summary_csv).expanduser().resolve()),
        "metrics": metric_definitions,
        "strain": {
            "requested": bool(args.strain_root),
            "root": str(Path(args.strain_root).expanduser().resolve()) if args.strain_root else None,
            "scope": "test split only",
            "file_layout": "<strain_root>/<protein>/strain_summary.csv",
            "required_columns": ["residue", "ensemble_mean", "ensemble_std"],
            "mapping_requirement": "exactly one row for residues 1..protein_length",
            "terminal_exclusion": args.strain_terminal_exclusion,
            "high_quantile": args.strain_high_quantile,
            "available_proteins": (
                int((strain_audit["strain_status"] == "ok").sum())
                if not strain_audit.empty else 0
            ),
            "audit": str(strain_audit_path) if args.strain_root else None,
        },
        "primary_estimand": (
            "Mean of within-protein mean apex feature values; proteins receive equal weight."
        ),
        "block_shift_null": (
            "One circular offset per protein/condition/sign/permutation, applied to the "
            "complete apex pattern within its eligible interval. For test-only strain, "
            "the finite-apex subset is circularly shifted in rank space over finite "
            "nonterminal strain support so terminal missingness is preserved."
        ),
        "paired_sign_estimand": (
            "Within-protein mean(flexibility-supporting apices) minus "
            "mean(rigidity-supporting apices), tested by protein-level sign flips."
        ),
        "n_block_shifts": args.n_block_shifts,
        "n_sign_flips": args.n_sign_flips,
        "random_seed": args.random_seed,
        "phase1_coverage_identifier": {
            "estimand": (
                "Annotation-to-band coverage and fixed band-mask identifier metrics; "
                "macro estimates average proteins equally and micro estimates pool counts."
            ),
            "targets": [
                "q8_loop_turn_bend_CTS",
                "neq_peak",
                *(
                    ["strain_top_quantile_within_protein"]
                    if args.strain_root else []
                ),
            ],
            "eligible_interval_note": (
                "Residue and segment denominators are restricted to positions eligible "
                "to be band apices. Band intervals are truncated to the same support."
            ),
            "n_bootstrap": args.n_bootstrap,
        },
        "phase2_within_q3_matching": {
            "estimand": (
                "Within-protein mean band-apex minus mean matched non-band control "
                "outcome, followed by equal-protein aggregation."
            ),
            "exact_match": "Q3",
            "nested_match_schemes": {
                name: MATCH_SCHEMES[name]["matched_covariates"]
                for name in args.match_schemes
            },
            "selected_match_schemes": args.match_schemes,
            "q3_only_control_selection": (
                "Every eligible same-protein, same-Q3 residue outside all band "
                "intervals (plus the configured buffer)."
            ),
            "caliper_scheme_control_selection": (
                "Up to controls_per_apex_maximum nearest eligible controls."
            ),
            "control_storage": (
                "q3_only is stored one row per matched apex in "
                "within_q3_matched_cases.csv.gz with the complete control count "
                "and control means; capped schemes also store individual controls "
                "in within_q3_matched_controls.csv.gz."
            ),
            "controls_per_apex_maximum": args.match_controls_per_apex,
            "non_band_buffer_residues": args.match_band_buffer,
            "calipers": {
                "neq": args.match_neq_caliper,
                "rsa": args.match_rsa_caliper,
                "normalized_sequence_position": args.match_position_caliper,
            },
            "control_reuse": "Allowed across apex matched sets, not within a set.",
            "outcomes": matched_outcome_definitions,
            "inference": (
                "Protein-level sign flips and protein bootstrap confidence intervals."
            ),
        },
        "outputs": {
            "per_protein": str(per_protein_path),
            "block_shift_null": str(null_path),
            "block_shift_summary": str(summary_path),
            "paired_sign_summary": str(contrast_path),
            "phase1_by_protein": str(phase1_per_protein_path),
            "phase1_summary": str(phase1_summary_path),
            "within_q3_matched_controls": str(matched_controls_path),
            "within_q3_matched_cases": str(matched_cases_path),
            "within_q3_match_coverage": str(match_coverage_path),
            "within_q3_match_balance": str(match_balance_path),
            "within_q3_matched_effects_by_protein": str(matched_per_protein_path),
            "within_q3_matched_summary": str(matched_summary_path),
            "strain_input_audit": str(strain_audit_path) if args.strain_root else None,
        },
    }
    parameters_path = output_dir / "biophysical_enrichment_parameters.json"
    parameters_path.write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "per_protein_rows": len(per_protein),
        "summary_rows": len(summary),
        "contrast_rows": len(contrasts),
        "block_shift_null_rows": len(null_rows),
        "phase1_per_protein_rows": len(phase1_per_protein),
        "phase1_summary_rows": len(phase1_summary),
        "matched_control_rows": len(matched_controls),
        "matched_case_rows": len(matched_cases),
        "matched_summary_rows": len(matched_summary),
        "strain_proteins_ok": (
            int((strain_audit["strain_status"] == "ok").sum())
            if not strain_audit.empty else 0
        ),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
