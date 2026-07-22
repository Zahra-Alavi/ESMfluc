#!/usr/bin/env python3
"""Decompose DeltaI into intrinsic signed evidence and attention routing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from weinreb_analysis_common import (
    PRIMARY_CONDITION, PRIMARY_SEEDS, WEINREB_ANALYSIS_ROOT,
    WEINREB_RESULTS_ROOT, load_seed, mean_sd_se, read_manifest,
    reconstruct_signed_evidence, sign_agreement, write_csv, json_dump,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=WEINREB_RESULTS_ROOT / "exact_contributions_v2" / "manifest.tsv")
    p.add_argument("--analysis", type=Path, default=WEINREB_ANALYSIS_ROOT)
    p.add_argument("--condition", default=PRIMARY_CONDITION)
    p.add_argument("--seeds", default="1,2,3")
    return p.parse_args()


def main() -> None:
    args = arguments()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    runs = read_manifest(args.manifest, args.condition, seeds)
    loaded = {int(r["seed"]): load_seed(r) for r in runs}
    names = [n for n in loaded[seeds[0]] if n != "WT_star"]
    by_seed = {}
    recovery_errors = []
    reconstruction_errors = []
    seed_rows = []
    for seed in seeds:
        wt = loaded[seed]["WT_star"]
        sw, ew = reconstruct_signed_evidence(wt.contribution, wt.attention)
        recovery_errors.append(ew)
        aw = wt.attention.mean(axis=0)
        for name in names:
            x = loaded[seed][name]
            sm, em = reconstruct_signed_evidence(x.contribution, x.attention)
            recovery_errors.append(em)
            am = x.attention.mean(axis=0)
            intrinsic = 0.5 * (am + aw) * (sm - sw)
            routing = 0.5 * (sm + sw) * (am - aw)
            direct = x.contribution.mean(axis=0) - wt.contribution.mean(axis=0)
            err = float(np.max(np.abs(intrinsic + routing - direct)))
            reconstruction_errors.append(err)
            by_seed[(name, seed)] = (intrinsic, routing, direct, sm - sw, am - aw)
            for j in range(len(direct)):
                seed_rows.append({
                    "condition": args.condition, "seed": seed, "mutant": name,
                    "matrix_index": j, "fasta_position": j + 1,
                    "delta_I": direct[j], "intrinsic_evidence_component": intrinsic[j],
                    "attention_routing_component": routing[j], "delta_signed_evidence": sm[j] - sw[j],
                    "delta_mean_attention_received": am[j] - aw[j],
                    "component_reconstruction_error": intrinsic[j] + routing[j] - direct[j],
                })
    write_csv(args.analysis / "decomposition" / "components_by_seed.csv", seed_rows)

    summary_rows = []
    for name in names:
        arrays = [np.stack([by_seed[(name, s)][k] for s in seeds]) for k in range(5)]
        stats = [mean_sd_se(x) for x in arrays]
        agree_intrinsic = sign_agreement(arrays[0])
        agree_routing = sign_agreement(arrays[1])
        for j in range(arrays[0].shape[1]):
            direct_mean = stats[2][0][j]
            im, isd, ise = (x[j] for x in stats[0])
            rm, rsd, rse = (x[j] for x in stats[1])
            summary_rows.append({
                "mutant": name, "matrix_index": j, "fasta_position": j + 1,
                "delta_I_mean": direct_mean,
                "intrinsic_evidence_mean": im, "intrinsic_evidence_sd": isd,
                "intrinsic_evidence_se": ise, "intrinsic_sign_agreement_n": int(agree_intrinsic[j]),
                "attention_routing_mean": rm, "attention_routing_sd": rsd,
                "attention_routing_se": rse, "routing_sign_agreement_n": int(agree_routing[j]),
                "delta_signed_evidence_mean": stats[3][0][j],
                "delta_mean_attention_received_mean": stats[4][0][j],
                "dominant_component": "intrinsic evidence" if abs(im) > abs(rm) else "attention routing",
                "intrinsic_fraction_abs": abs(im) / max(abs(im) + abs(rm), 1e-15),
            })
    write_csv(args.analysis / "decomposition" / "components_summary.csv", summary_rows)

    band_path = args.analysis / "bands" / "change_bands_primary.csv"
    band_components = []
    if band_path.is_file():
        with band_path.open(encoding="utf-8", newline="") as h:
            bands = list(csv.DictReader(h))
        for band_id, band in enumerate(bands, 1):
            name = band["mutant"]
            start, end = int(band["start_matrix_index"]), int(band["end_matrix_index"])
            i_values = np.array([by_seed[(name, s)][0][start:end + 1].sum() for s in seeds])
            r_values = np.array([by_seed[(name, s)][1][start:end + 1].sum() for s in seeds])
            d_values = np.array([by_seed[(name, s)][2][start:end + 1].sum() for s in seeds])
            im, isd, ise = mean_sd_se(i_values)
            rm, rsd, rse = mean_sd_se(r_values)
            dm, dsd, dse = mean_sd_se(d_values)
            band_components.append({
                "band_id": band_id, "mutant": name, "start_matrix_index": start,
                "apex_matrix_index": int(band["apex_matrix_index"]), "end_matrix_index": end,
                "delta_I_integrated_mean": dm, "delta_I_integrated_sd": dsd,
                "intrinsic_integrated_mean": im, "intrinsic_integrated_sd": isd,
                "intrinsic_integrated_se": ise, "routing_integrated_mean": rm,
                "routing_integrated_sd": rsd, "routing_integrated_se": rse,
                "dominant_component": "intrinsic evidence" if abs(im) > abs(rm) else "attention routing",
                "reconstruction_error": im + rm - dm,
            })
        write_csv(args.analysis / "decomposition" / "band_components.csv", band_components)

    evidence_tolerance = 1e-7  # float32 C combined with decimal-serialized attention
    decomposition_tolerance = 1e-8
    json_dump(args.analysis / "decomposition" / "integrity.json", {
        "condition": args.condition, "seeds": list(seeds),
        "maximum_C_equals_A_times_s_error": max(recovery_errors),
        "maximum_decomposition_reconstruction_error": max(reconstruction_errors),
        "C_equals_A_times_s_tolerance": evidence_tolerance,
        "decomposition_tolerance": decomposition_tolerance,
        "tolerance_rationale": "C is float32 and attention was decimal-serialized; the algebraic decomposition is checked separately",
        "passes": max(recovery_errors) <= evidence_tolerance and max(reconstruction_errors) <= decomposition_tolerance,
    })
    if max(recovery_errors) > evidence_tolerance or max(reconstruction_errors) > decomposition_tolerance:
        raise ValueError("Evidence/routing decomposition failed numerical integrity")


if __name__ == "__main__":
    main()
