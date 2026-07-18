#!/usr/bin/env python3
"""
Test whether attention hubs and vertical-band peaks are closer to Neq peaks than
matched random residues.

Selections tested per protein/run:
  - top received-attention residues within each row mode
  - vertical-band peak residues within each row mode
  - Neq-peak-overlapping hubs within each row mode
  - low_mode_1 coil hubs, specifically

Matched null:
  Random residues are drawn from the same protein and matched by secondary
  structure and row mode, so any signal is not just "coils are flexible" or
  "low_mode_1 is coil-rich".
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_attention_morphology_biology import detect_bands
from analyze_attention_row_modes import (
    analyze_attention_modes,
    default_analysis_dir,
    resolve_existing_path,
    resolve_manifest_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare attention hubs/bands to Neq peaks with matched random controls."
    )
    parser.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    parser.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Defaults to result_root/manifest.tsv. "
             "Use manifest_attention_sources.tsv for the 30-attention source view.",
    )
    parser.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    parser.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id and q3 columns.")
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_hubs_vs_neq_peaks.")
    parser.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional condition subset.")
    parser.add_argument("--peak_quantile", type=float, default=0.90,
                        help="Neq values >= this within-protein quantile are peak candidates.")
    parser.add_argument("--peak_window", type=int, default=2,
                        help="Residue is near a peak if distance <= this window.")
    parser.add_argument("--top_frac", type=float, default=0.10,
                        help="Top received-attention fraction within each mode.")
    parser.add_argument("--high_entropy_quantile", type=float, default=0.67)
    parser.add_argument("--min_low_rows", type=int, default=8)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--band_quantile", type=float, default=0.90)
    parser.add_argument("--band_z", type=float, default=1.0)
    parser.add_argument("--min_band_width", type=int, default=2)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--permutation_seed", type=int, default=123)
    parser.add_argument("--save_selected_residues", action="store_true")
    return parser.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(int(label), str(label))


def load_neq_by_name(test_csv):
    df = pd.read_csv(test_csv)
    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]) if "name" in df.columns else str(row["sequence"])
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[name] = {"sequence": row["sequence"], "neq": neq, "flexible": classify_neq(neq)}
    return out


def load_ss_map(ss_csv):
    df = pd.read_csv(ss_csv)
    columns = {c.strip(): c for c in df.columns}
    id_col = columns.get("id")
    q3_col = columns.get("q3")
    if id_col is None or q3_col is None:
        raise ValueError(f"{ss_csv} must contain id and q3 columns.")
    ss_map = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col]).lstrip(">")
        ss_map.setdefault(seq_id, []).append(str(row[q3_col]).strip())
    return ss_map


def neq_peak_mask(neq, quantile):
    neq = np.asarray(neq, dtype=float)
    if len(neq) == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(np.quantile(neq, quantile))
    high = neq >= threshold
    local = np.zeros(len(neq), dtype=bool)
    for i in range(len(neq)):
        left = neq[i - 1] if i > 0 else -np.inf
        right = neq[i + 1] if i + 1 < len(neq) else -np.inf
        local[i] = neq[i] >= left and neq[i] >= right
    peaks = high & local
    if peaks.sum() == 0:
        peaks = high
    return peaks


def nearest_peak_distances(indices, peak_mask):
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    peak_positions = np.where(peak_mask)[0]
    if len(indices) == 0:
        return np.asarray([], dtype=float)
    if len(peak_positions) == 0:
        return np.full(len(indices), np.nan)
    return np.asarray([np.min(np.abs(peak_positions - i)) for i in indices], dtype=float)


def top_indices(values, frac):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    k = max(1, int(np.ceil(len(values) * frac)))
    return np.argsort(values)[-k:].tolist()


def build_selected_sets(received, modes, ss, bands, peak_mask, top_frac):
    selected = {}
    n = len(received)
    for label in [0, 1, 2]:
        mode = mode_name(label)
        idx = np.where(modes == label)[0]
        if len(idx):
            top_local = idx[top_indices(received[idx], top_frac)]
            selected[f"top_received__{mode}"] = set(int(i) for i in top_local)
            selected[f"neq_peak_overlapping_hubs__{mode}"] = set(
                int(i) for i in top_local if peak_mask[int(i)]
            )

    band_peaks_by_mode = {mode_name(label): set() for label in [0, 1, 2]}
    for start, end in bands:
        region = np.arange(start, end + 1)
        peak = int(region[np.argmax(received[region])])
        band_peaks_by_mode[mode_name(int(modes[peak]))].add(peak)
    for mode, idx in band_peaks_by_mode.items():
        selected[f"vertical_band_peaks__{mode}"] = idx

    low1_coil = set(
        int(i) for i in selected.get("top_received__low_mode_1", set())
        if i < n and ss[i] == "C"
    )
    selected["low_mode_1_coil_hubs"] = low1_coil
    return selected


def matched_sample(selected_indices, modes, ss, rng):
    selected_indices = list(selected_indices)
    n = len(modes)
    sampled = []
    for idx in selected_indices:
        stratum = np.where((modes == modes[idx]) & (ss == ss[idx]))[0]
        stratum = stratum[stratum != idx]
        if len(stratum) == 0:
            stratum = np.arange(n)
        sampled.append(int(rng.choice(stratum)))
    return sampled


def summarize_selection(selection_name, indices, peak_mask, neq, modes, ss, rng, n_permutations, peak_window):
    indices = sorted(set(int(i) for i in indices))
    if not indices:
        return None, []
    distances = nearest_peak_distances(indices, peak_mask)
    near_peak = distances <= peak_window
    observed = {
        "selection": selection_name,
        "n_selected": len(indices),
        "mean_distance_to_peak": float(np.nanmean(distances)),
        "median_distance_to_peak": float(np.nanmedian(distances)),
        "fraction_near_peak": float(np.nanmean(near_peak)),
        "fraction_exact_peak": float(np.mean(peak_mask[indices])),
        "mean_neq": float(np.mean(neq[indices])),
        "mean_flexible": float(np.mean(neq[indices] > 1.0)),
    }

    null_rows = []
    for perm in range(1, n_permutations + 1):
        sample = matched_sample(indices, modes, ss, rng)
        d = nearest_peak_distances(sample, peak_mask)
        null_rows.append({
            "selection": selection_name,
            "permutation": perm,
            "mean_distance_to_peak": float(np.nanmean(d)),
            "fraction_near_peak": float(np.nanmean(d <= peak_window)),
            "fraction_exact_peak": float(np.mean(peak_mask[sample])),
            "mean_neq": float(np.mean(neq[sample])),
            "mean_flexible": float(np.mean(neq[sample] > 1.0)),
        })
    return observed, null_rows


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parents[2]
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest TSV: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_hubs_vs_neq_peaks", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()
    neq_by_name = load_neq_by_name(args.test_csv)
    ss_map = load_ss_map(args.ss_csv)
    rng = np.random.default_rng(args.permutation_seed)

    observed_rows = []
    null_rows = []
    selected_rows = []

    for run in manifest.itertuples(index=False):
        attention_path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            print(f"[skip] missing {attention_path}")
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            protein = record["name"]
            if protein not in neq_by_name or protein not in ss_map:
                continue
            sequence = record["sequence"]
            n = len(sequence)
            neq = neq_by_name[protein]["neq"][:n]
            ss = np.asarray(ss_map[protein][:n])
            if len(neq) != n or len(ss) != n:
                continue

            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            received = attn.sum(axis=0)
            modes, _, _, _, _, _ = analyze_attention_modes(
                attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed
            )
            bands, _, _, _ = detect_bands(
                received,
                quantile=args.band_quantile,
                z_thresh=args.band_z,
                min_width=args.min_band_width,
                smooth_window=args.smooth_window,
            )
            peak_mask = neq_peak_mask(neq, args.peak_quantile)
            selected = build_selected_sets(received, modes, ss, bands, peak_mask, args.top_frac)

            for selection_name, indices in selected.items():
                obs, null = summarize_selection(
                    selection_name, indices, peak_mask, neq, modes, ss, rng,
                    args.n_permutations, args.peak_window
                )
                if obs is None:
                    continue
                obs.update({
                    "condition": run.condition,
                    "seed": int(run.seed),
                    "protein": protein,
                    "n_residues": n,
                    "n_neq_peaks": int(peak_mask.sum()),
                })
                observed_rows.append(obs)
                for row in null:
                    row.update({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                    })
                    null_rows.append(row)
                if args.save_selected_residues:
                    distances = nearest_peak_distances(indices, peak_mask)
                    for idx, dist in zip(sorted(indices), distances):
                        selected_rows.append({
                            "condition": run.condition,
                            "seed": int(run.seed),
                            "protein": protein,
                            "selection": selection_name,
                            "position_1based": idx + 1,
                            "aa": sequence[idx],
                            "ss": ss[idx],
                            "mode": mode_name(modes[idx]),
                            "neq": float(neq[idx]),
                            "is_neq_peak": bool(peak_mask[idx]),
                            "distance_to_nearest_peak": float(dist),
                            "received_attention": float(received[idx]),
                        })

    observed = pd.DataFrame(observed_rows)
    null = pd.DataFrame(null_rows)
    if observed.empty:
        raise RuntimeError("No selections were analyzed.")

    p_rows = []
    for key, obs_sub in observed.groupby(["condition", "selection"]):
        condition, selection = key
        null_sub = null[(null["condition"] == condition) & (null["selection"] == selection)]
        for metric, direction in [
            ("mean_distance_to_peak", "lower"),
            ("fraction_near_peak", "higher"),
            ("fraction_exact_peak", "higher"),
            ("mean_neq", "higher"),
            ("mean_flexible", "higher"),
        ]:
            obs_val = float(obs_sub[metric].mean())
            null_vals = null_sub.groupby("permutation")[metric].mean().dropna().to_numpy()
            if len(null_vals) == 0:
                continue
            if direction == "lower":
                pval = (1 + np.sum(null_vals <= obs_val)) / (len(null_vals) + 1)
            else:
                pval = (1 + np.sum(null_vals >= obs_val)) / (len(null_vals) + 1)
            p_rows.append({
                "condition": condition,
                "selection": selection,
                "metric": metric,
                "observed_mean": obs_val,
                "null_mean": float(np.mean(null_vals)),
                "observed_minus_null": obs_val - float(np.mean(null_vals)),
                "empirical_p": float(pval),
            })

    summary = (
        observed.groupby(["condition", "selection"])
        .agg({
            "n_selected": "mean",
            "mean_distance_to_peak": ["mean", "std"],
            "fraction_near_peak": ["mean", "std"],
            "fraction_exact_peak": ["mean", "std"],
            "mean_neq": ["mean", "std"],
            "mean_flexible": ["mean", "std"],
        })
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()

    observed.to_csv(output_dir / "hubs_bands_peak_proximity_by_run.csv", index=False)
    null.to_csv(output_dir / "matched_random_peak_null.csv", index=False)
    summary.to_csv(output_dir / "hubs_bands_peak_proximity_summary.csv", index=False)
    pd.DataFrame(p_rows).to_csv(output_dir / "matched_random_peak_empirical_pvalues.csv", index=False)
    if args.save_selected_residues:
        pd.DataFrame(selected_rows).to_csv(output_dir / "selected_hub_band_residues.csv", index=False)

    lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Observed selection rows: {len(observed)}",
        f"Null rows: {len(null)}",
        f"Peak quantile: {args.peak_quantile}",
        f"Peak window: {args.peak_window}",
        "",
        "Key outputs:",
        "  hubs_bands_peak_proximity_summary.csv",
        "  matched_random_peak_empirical_pvalues.csv",
        "  hubs_bands_peak_proximity_by_run.csv",
    ]
    (output_dir / "hubs_vs_neq_peaks_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
