#!/usr/bin/env python3
"""Apply frozen ESM3-top28 bands/hotspot hypotheses to all other conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from weinreb_analysis_common import (
    WEINREB_ANALYSIS_ROOT, WEINREB_RESULTS_ROOT, json_dump, load_seed,
    mean_sd_se, safe_corr, sign_agreement, write_csv,
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=WEINREB_RESULTS_ROOT / "exact_contributions_v2" / "manifest.tsv")
    p.add_argument("--analysis", type=Path, default=WEINREB_ANALYSIS_ROOT)
    p.add_argument("--primary-condition", default="esm3_top28_bilstm_attn")
    return p.parse_args()


def main():
    args = arguments()
    with args.manifest.open(encoding="utf-8", newline="") as h:
        manifest = list(csv.DictReader(h, delimiter="\t"))
    conditions = sorted({r["condition"] for r in manifest})
    condition_delta = {}
    names = None
    for condition in conditions:
        rows = sorted([r for r in manifest if r["condition"] == condition], key=lambda r: int(r["seed"]))
        if [int(r["seed"]) for r in rows] != [1, 2, 3]:
            continue
        seed_data = []
        for row in rows:
            payload = load_seed(row)
            if names is None:
                names = [n for n in payload if n != "WT_star"]
            wt = payload["WT_star"].contribution.mean(axis=0)
            seed_data.append(np.stack([payload[n].contribution.mean(axis=0) - wt for n in names]))
        condition_delta[condition] = np.stack(seed_data, axis=1)  # mutant, seed, key
    primary = condition_delta[args.primary_condition]

    comparison_rows = []
    for condition, values in condition_delta.items():
        x, y = primary.mean(axis=1).ravel(), values.mean(axis=1).ravel()
        comparison_rows.append({
            "condition": condition, "is_primary": condition == args.primary_condition,
            "mutant_residue_profile_pearson_vs_primary": safe_corr(x, y),
            "mean_absolute_delta_I": float(np.mean(np.abs(values.mean(axis=1)))),
            "hotspot_155_157_median_abs_delta_I": float(np.median(np.abs(values.mean(axis=1)[:, 154:157]))),
        })
    write_csv(args.analysis / "robustness" / "condition_profile_comparison.csv", comparison_rows)

    with (args.analysis / "bands" / "change_bands_primary.csv").open(encoding="utf-8", newline="") as h:
        bands = list(csv.DictReader(h))
    band_rows = []
    for band_id, band in enumerate(bands, 1):
        mi = names.index(band["mutant"])
        start, end = int(band["start_matrix_index"]), int(band["end_matrix_index"])
        primary_effect = primary[mi, :, start:end + 1].sum(axis=1).mean()
        for condition, values in condition_delta.items():
            seed_effect = values[mi, :, start:end + 1].sum(axis=1)
            mean, sd, se = mean_sd_se(seed_effect)
            agreement = int(sign_agreement(seed_effect, axis=0))
            band_rows.append({
                "band_id": band_id, "mutant": band["mutant"], "condition": condition,
                "start_matrix_index": start, "end_matrix_index": end,
                "integrated_delta_I_mean": mean, "integrated_delta_I_sd": sd,
                "integrated_delta_I_se": se, "sign_agreement_n": agreement,
                "same_sign_as_primary": bool(np.sign(mean) == np.sign(primary_effect)),
                "replicates_2of3_and_primary_sign": bool(agreement >= 2 and np.sign(mean) == np.sign(primary_effect)),
            })
    write_csv(args.analysis / "robustness" / "fixed_band_replication.csv", band_rows)
    summary = {}
    for condition in conditions:
        rows = [r for r in band_rows if r["condition"] == condition]
        summary[condition] = {
            "bands_tested": len(rows),
            "same_sign_fraction": float(np.mean([r["same_sign_as_primary"] for r in rows])) if rows else None,
            "replicated_2of3_fraction": float(np.mean([r["replicates_2of3_and_primary_sign"] for r in rows])) if rows else None,
        }
    json_dump(args.analysis / "robustness" / "replication_summary.json", {
        "primary_condition": args.primary_condition,
        "hypotheses_frozen_from": "bands/change_bands_primary.csv and matrix positions 155-157",
        "conditions": summary,
    })


if __name__ == "__main__":
    main()

