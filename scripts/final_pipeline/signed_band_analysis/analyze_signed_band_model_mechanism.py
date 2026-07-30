#!/usr/bin/env python3
"""Phase 3C: decompose averaged signed bands into evidence and consultation.

Bands were called from arithmetic mean_seed(I_j), while each seed obeys the
exact identity I_j = s_j * B_j, with B_j the attention-column mean.  The script
therefore performs comparisons inside each seed and averages effects only
afterward.  Log-magnitude differences decompose exactly as

  delta log|I| = delta log|s| + delta log(B).

Controls are eligible residues in the same protein with identical Q8 that lie
outside every positive and negative band interval.  Extraction is checkpointed
one manifest row at a time because the source JSON files contain large LxL
matrices that must be streamed past but are not materialized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

from .extract_signed_contribution_bands import iter_profiles


METRICS = {
    "directional_evidence": "sign * intrinsic signed evidence s_j",
    "attention_breadth": "attention column mean B_j",
    "directional_influence": "sign * I_j",
    "log_abs_evidence": "log absolute intrinsic evidence",
    "log_attention_breadth": "log attention column mean",
    "log_abs_influence": "log absolute signed influence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_tsv", required=True)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--protein_summary_csv", required=True)
    parser.add_argument("--residue_annotations_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
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


def cache_path(cache_dir: Path, condition: str, seed: int, split: str) -> Path:
    return cache_dir / f"{condition}__seed{seed}__{split}.csv.gz"


def prepare_lookups(
    bands: pd.DataFrame, protein_summary: pd.DataFrame, residue: pd.DataFrame
) -> tuple[dict, dict, dict]:
    band_lookup = {
        key: group.sort_values("apex_index_0based")
        for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    summary_lookup = protein_summary.set_index(
        ["condition", "split", "protein"]
    )[["eligible_start_index_0based", "eligible_end_index_0based_exclusive"]]
    q8_lookup = {
        key: group.sort_values("residue_index_0based").q8.astype(str).to_numpy()
        for key, group in residue.groupby(["split", "protein"], sort=False)
    }
    return band_lookup, summary_lookup, q8_lookup


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def extract_manifest_row(
    row,
    output_path: Path,
    band_lookup: dict,
    summary_lookup: pd.DataFrame,
    q8_lookup: dict,
) -> dict:
    condition, split, seed = str(row.condition), str(row.split), int(row.seed)
    source = Path(row.json_gz).expanduser().resolve()
    metadata, profiles = iter_profiles(source)
    if str(metadata["condition"]) != condition or str(metadata["split"]) != split:
        raise ValueError(f"Manifest/header mismatch for {source}")
    if int(metadata["seed"]) != seed:
        raise ValueError(f"Manifest/header seed mismatch for {source}")
    rows = []
    maximum_identity_error = 0.0
    clamp_count = 0
    tiny = np.finfo(np.float64).tiny
    for profile in profiles:
        protein = str(profile["name"])
        context = (condition, split, protein)
        protein_bands = band_lookup.get(context)
        if protein_bands is None or protein_bands.empty:
            continue
        q8 = q8_lookup[(split, protein)]
        length = len(q8)
        if length != int(profile["length"]):
            raise ValueError(f"{context}: contribution/annotation length mismatch")
        interval = summary_lookup.loc[context]
        start = int(interval.eligible_start_index_0based)
        end = int(interval.eligible_end_index_0based_exclusive)
        eligible = np.zeros(length, dtype=bool)
        eligible[start:end] = True
        excluded = np.zeros(length, dtype=bool)
        for band in protein_bands.itertuples(index=False):
            excluded[int(band.start_index_0based):int(band.end_index_0based_inclusive) + 1] = True
        candidate_base = eligible & ~excluded
        evidence = np.asarray(profile["intrinsic_signed_evidence"], dtype=np.float64)
        influence = np.asarray(profile["signed_column_influence"], dtype=np.float64)
        breadth = np.asarray(profile["attention_column_mean"], dtype=np.float64)
        identity_error = float(np.max(np.abs(influence - evidence * breadth)))
        maximum_identity_error = max(maximum_identity_error, identity_error)
        if identity_error > 5e-6:
            raise ValueError(f"{context}: I=s*B error {identity_error}")
        abs_evidence = np.abs(evidence)
        abs_influence = np.abs(influence)
        clamp_count += int(np.sum(abs_evidence == 0) + np.sum(abs_influence == 0) + np.sum(breadth <= 0))
        log_abs_evidence = np.log(np.maximum(abs_evidence, tiny))
        log_breadth = np.log(np.maximum(breadth, tiny))
        log_abs_influence = np.log(np.maximum(abs_influence, tiny))
        for band in protein_bands.itertuples(index=False):
            apex, sign = int(band.apex_index_0based), int(band.sign)
            q8_label = str(q8[apex])
            controls = np.flatnonzero(candidate_base & (q8 == q8_label))
            base = {
                "band_id": str(band.band_id), "condition": condition,
                "seed": seed, "split": split, "protein": protein,
                "sign": sign, "label": str(band.label),
                "apex_index_0based": apex, "q8": q8_label,
                "n_same_protein_q8_controls": len(controls),
                "seed_signed_influence": float(influence[apex]),
                "seed_direction_agrees_with_band": bool(sign * influence[apex] > 0),
                "identity_abs_error": float(abs(influence[apex] - evidence[apex] * breadth[apex])),
            }
            arrays = {
                "directional_evidence": sign * evidence,
                "attention_breadth": breadth,
                "directional_influence": sign * influence,
                "log_abs_evidence": log_abs_evidence,
                "log_attention_breadth": log_breadth,
                "log_abs_influence": log_abs_influence,
            }
            for metric, values in arrays.items():
                case_value = float(values[apex])
                control_mean = finite_mean(values[controls]) if len(controls) else np.nan
                base[f"case_{metric}"] = case_value
                base[f"control_mean_{metric}"] = control_mean
                base[f"delta_{metric}"] = case_value - control_mean
            base["delta_log_decomposition_error"] = (
                base["delta_log_abs_influence"]
                - base["delta_log_abs_evidence"]
                - base["delta_log_attention_breadth"]
            )
            rows.append(base)
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, compression="gzip")
    return {
        "condition": condition, "seed": seed, "split": split,
        "rows": len(frame), "source": str(source), "cache": str(output_path),
        "maximum_identity_abs_error": maximum_identity_error,
        "log_clamp_count": clamp_count,
    }


def mechanism_label(evidence: float, consultation: float, total: float) -> str:
    if not np.isfinite(total):
        return "unclassified"
    if total <= 0:
        return "not_magnitude_enriched"
    if evidence > 0 and consultation <= 0:
        return "evidence_dominated"
    if consultation > 0 and evidence <= 0:
        return "consultation_dominated"
    if evidence <= 0 and consultation <= 0:
        return "unclassified"
    evidence_share = evidence / (evidence + consultation)
    if evidence_share >= 2 / 3:
        return "evidence_dominated"
    if evidence_share <= 1 / 3:
        return "consultation_dominated"
    return "combined"


def aggregate(cache_paths: list[Path], output_dir: Path, minimum_proteins: int) -> dict:
    per_seed = pd.concat([pd.read_csv(path) for path in cache_paths], ignore_index=True)
    if per_seed.duplicated(["band_id", "seed"]).any():
        raise ValueError("Duplicate band/seed rows in cache")
    counts = per_seed.groupby("band_id").seed.nunique()
    if not (counts == 3).all():
        raise ValueError(f"Bands without exactly three seeds: {int((counts != 3).sum())}")
    value_columns = [
        column for column in per_seed.columns
        if column.startswith("case_") or column.startswith("control_mean_")
        or column.startswith("delta_")
    ]
    first_columns = [
        "condition", "split", "protein", "sign", "label",
        "apex_index_0based", "q8", "n_same_protein_q8_controls",
    ]
    aggregations = {column: "mean" for column in value_columns}
    aggregations.update({column: "first" for column in first_columns})
    aggregations.update({
        "seed_direction_agrees_with_band": "mean",
        "seed_signed_influence": "mean", "identity_abs_error": "max",
    })
    bands = per_seed.groupby("band_id", as_index=False).agg(aggregations)
    bands = bands.rename(columns={
        "seed_direction_agrees_with_band": "seed_direction_agreement_fraction",
        "seed_signed_influence": "mean_seed_signed_influence",
        "identity_abs_error": "maximum_identity_abs_error",
    })
    bands["log_decomposition_reconstruction_error"] = (
        bands.delta_log_abs_influence
        - bands.delta_log_abs_evidence
        - bands.delta_log_attention_breadth
    )
    bands["evidence_share_of_positive_log_enrichment"] = np.where(
        (bands.delta_log_abs_evidence > 0) & (bands.delta_log_attention_breadth > 0),
        bands.delta_log_abs_evidence
        / (bands.delta_log_abs_evidence + bands.delta_log_attention_breadth),
        np.nan,
    )
    bands["mechanism_class"] = [
        mechanism_label(e, a, total)
        for e, a, total in zip(
            bands.delta_log_abs_evidence,
            bands.delta_log_attention_breadth,
            bands.delta_log_abs_influence,
        )
    ]

    # Matched effect estimands must use one common support.  Bands without an
    # exact-Q8 control remain in the per-band classification table as
    # unclassified, but cannot contribute a case-only value to a matched mean.
    matched_bands = bands[bands.n_same_protein_q8_controls > 0].copy()
    long_rows = []
    for metric in METRICS:
        for row in matched_bands.itertuples(index=False):
            long_rows.append({
                "condition": row.condition, "split": row.split,
                "protein": row.protein, "sign": row.sign, "label": row.label,
                "q8": row.q8, "metric": metric,
                "case_value": getattr(row, f"case_{metric}"),
                "control_mean": getattr(row, f"control_mean_{metric}"),
                "case_minus_control": getattr(row, f"delta_{metric}"),
            })
    long = pd.DataFrame(long_rows)
    protein = long.groupby(
        ["condition", "split", "protein", "sign", "label", "metric"], as_index=False
    ).agg(
        n_bands=("case_minus_control", "size"),
        case_mean=("case_value", "mean"), control_mean=("control_mean", "mean"),
        case_minus_control=("case_minus_control", "mean"),
    )
    summary_rows = []
    for key, group in protein.groupby(
        ["condition", "split", "sign", "label", "metric"], sort=False
    ):
        values = group.case_minus_control.dropna().to_numpy(float)
        n = len(values); mean = float(np.mean(values)) if n else np.nan
        eligible = n >= minimum_proteins
        sd = float(np.std(values, ddof=1)) if eligible else np.nan
        half = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n)) if eligible else np.nan
        if eligible:
            try:
                p = float(wilcoxon(values, zero_method="zsplit", method="approx").pvalue)
            except ValueError:
                p = 1.0
        else:
            p = np.nan
        summary_rows.append({
            **dict(zip(["condition", "split", "sign", "label", "metric"], key)),
            "metric_description": METRICS[str(key[-1])],
            "n_proteins": n, "n_bands": int(group.n_bands.sum()),
            "case_macro_protein_mean": float(group.case_mean.mean()),
            "control_macro_protein_mean": float(group.control_mean.mean()),
            "case_minus_control_macro_mean": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "inference_eligible": eligible, "wilcoxon_p_two_sided": p,
        })
    summary = pd.DataFrame(summary_rows)
    summary["wilcoxon_q_bh"] = bh(summary.wilcoxon_p_two_sided)

    class_counts = bands.groupby(
        ["condition", "split", "sign", "label", "mechanism_class"], as_index=False
    ).agg(n_bands=("band_id", "size"), n_proteins=("protein", "nunique"))
    class_counts["band_fraction"] = class_counts.n_bands / class_counts.groupby(
        ["condition", "split", "sign"]
    ).n_bands.transform("sum")
    protein_classes = bands.groupby(
        ["condition", "split", "protein", "sign", "mechanism_class"]
    ).size().rename("count").reset_index()
    protein_classes["fraction"] = protein_classes["count"] / protein_classes.groupby(
        ["condition", "split", "protein", "sign"]
    )["count"].transform("sum")
    macro = protein_classes.groupby(
        ["condition", "split", "sign", "mechanism_class"], as_index=False
    ).agg(macro_protein_fraction=("fraction", "mean"))
    class_counts = class_counts.merge(
        macro, on=["condition", "split", "sign", "mechanism_class"], how="left"
    )

    per_seed.to_csv(output_dir / "mechanism_by_band_and_seed.csv.gz", index=False, compression="gzip")
    bands.to_csv(output_dir / "mechanism_by_band_seed_averaged.csv.gz", index=False, compression="gzip")
    protein.to_csv(output_dir / "mechanism_effects_by_protein.csv.gz", index=False, compression="gzip")
    summary.to_csv(output_dir / "mechanism_effect_summary.csv", index=False)
    class_counts.to_csv(output_dir / "mechanism_class_summary.csv", index=False)
    return {
        "per_seed_rows": len(per_seed), "band_rows": len(bands),
        "matched_band_rows": len(matched_bands),
        "unmatched_band_rows": int(len(bands) - len(matched_bands)),
        "summary_rows": len(summary), "maximum_log_reconstruction_error": float(
            bands.log_decomposition_reconstruction_error.abs().max()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.extract_only and args.aggregate_only:
        raise ValueError("Choose at most one of --extract_only and --aggregate_only")
    manifest = pd.read_csv(args.manifest_tsv, sep="\t")
    manifest.seed = manifest.seed.astype(int)
    if args.conditions:
        manifest = manifest[manifest.condition.isin(args.conditions)]
    if args.splits:
        manifest = manifest[manifest.split.isin(args.splits)]
    if args.seeds:
        manifest = manifest[manifest.seed.isin(args.seeds)]
    if manifest.empty:
        raise ValueError("No manifest rows selected")
    output = Path(args.output_dir).expanduser().resolve()
    cache_dir = output / "per_seed_split_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    if not args.aggregate_only:
        bands = pd.read_csv(args.bands_csv)
        protein_summary = pd.read_csv(args.protein_summary_csv)
        residue = pd.read_csv(args.residue_annotations_csv)
        selected_conditions = set(manifest.condition.astype(str))
        selected_splits = set(manifest.split.astype(str))
        bands = bands[bands.condition.isin(selected_conditions) & bands.split.isin(selected_splits)]
        protein_summary = protein_summary[
            protein_summary.condition.isin(selected_conditions)
            & protein_summary.split.isin(selected_splits)
        ]
        residue = residue[residue.split.isin(selected_splits)]
        lookups = prepare_lookups(bands, protein_summary, residue)
        for row in manifest.itertuples(index=False):
            path = cache_path(cache_dir, str(row.condition), int(row.seed), str(row.split))
            if path.exists() and not args.overwrite_cache:
                reports.append({"cache": str(path), "status": "already_present"})
                continue
            report = extract_manifest_row(row, path, *lookups)
            report["status"] = "written"
            reports.append(report)
            print(json.dumps(report), flush=True)
    aggregate_report = None
    if not args.extract_only:
        expected = [
            cache_path(cache_dir, str(row.condition), int(row.seed), str(row.split))
            for row in manifest.itertuples(index=False)
        ]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot aggregate: {len(missing)} cache shards are missing; first={missing[0]}"
            )
        aggregate_report = aggregate(expected, output, args.minimum_inference_proteins)
        parameters = {
            "phase": "3C", "identity": "I_j = s_j * attention_column_mean_j",
            "ensemble_rule": "compute within seed, then average three seed effects",
            "control": "same protein, exact Q8, eligible, outside every band interval",
            "matched_effect_support": (
                "Case, control, and difference summaries all exclude bands with "
                "zero exact-Q8 controls; those bands remain unclassified per-band."
            ),
            "classification": {
                "basis": "delta log|I| = delta log|s| + delta log(attention breadth)",
                "dominance_cutoffs": [1 / 3, 2 / 3],
            },
            "minimum_inference_proteins": args.minimum_inference_proteins,
            "metrics": METRICS,
            "selected_conditions": sorted(
                manifest.condition.astype(str).unique().tolist()
            ),
            "selected_splits": sorted(
                manifest.split.astype(str).unique().tolist()
            ),
            "selected_seeds": sorted(
                manifest.seed.astype(int).unique().tolist()
            ),
        }
        (output / "phase3c_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "cache_reports": reports, "aggregate": aggregate_report,
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
