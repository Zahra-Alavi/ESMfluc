#!/usr/bin/env python3
"""
Distance enrichment from attention-selected residues to structural/dynamic targets.

This generalizes the earlier Neq-peak proximity analysis.  For each protein/run,
the script selects attention hubs or vertical-band peaks, then asks whether those
selected residues are closer than matched random residues to:
  - Q8 turn/bend/coil labels
  - Q3 secondary-structure boundaries
  - high-Neq peaks
  - high ATLAS strain residues, if --atlas_root is provided
  - high PB entropy / PB transition residues, if --atlas_root is provided

Distances are sequence distances in residues.  Matched backgrounds are drawn from
the same protein, with optional matching on Q3, Q8, and row-mode label.
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

from analyze_attention_morphology_biology import detect_bands
from analyze_attention_row_modes import (
    analyze_attention_modes,
    default_analysis_dir,
    resolve_existing_path,
)
from analyze_low_mode_control_points import load_q3_q8_map
from analyze_attention_hubs_vs_neq_peaks import neq_peak_mask


def parse_args():
    p = argparse.ArgumentParser(description="Sequence-distance enrichment for attention hubs/bands.")
    p.add_argument("--result_root", required=True)
    p.add_argument("--manifest_tsv", default=None)
    p.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    p.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id/q3/q8.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pipeline_dir", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--atlas_root", default=None, help="Optional atlas_strain_analysis directory with strain/PB files.")
    p.add_argument("--replicates", nargs="+", default=["R1", "R2", "R3"])
    p.add_argument("--top_fracs", nargs="+", type=float, default=[0.05, 0.10])
    p.add_argument("--distance_windows", nargs="+", type=int, default=[0, 1, 2, 3, 5, 10])
    p.add_argument("--attention_scores", nargs="+", default=["received_attention", "in_degree_top1"])
    p.add_argument(
        "--selection_types",
        nargs="+",
        default=["top_hubs", "vertical_band_peaks"],
        choices=["top_hubs", "vertical_band_peaks"],
    )
    p.add_argument("--pool_modes", nargs="+", default=["all", "low_mode_1", "low_mode_2"])
    p.add_argument("--background_per_selected", type=int, default=5)
    p.add_argument(
        "--background_scope",
        choices=["eligible", "same_q3", "same_q3_q8", "same_q3_mode", "same_q3_q8_mode"],
        default="same_q3_q8_mode",
    )
    p.add_argument("--peak_quantile", type=float, default=0.90)
    p.add_argument("--high_strain_quantile", type=float, default=0.90)
    p.add_argument("--high_pb_entropy_quantile", type=float, default=0.90)
    p.add_argument("--high_pb_transition_quantile", type=float, default=0.90)
    p.add_argument("--terminal_exclusion", type=int, default=10)
    p.add_argument("--terminal_exclusion_fraction", type=float, default=0.0)
    p.add_argument("--exclude_z_from_entropy", action="store_true", default=True)
    p.add_argument("--include_z_in_entropy", action="store_false", dest="exclude_z_from_entropy")
    p.add_argument("--high_entropy_quantile", type=float, default=0.67)
    p.add_argument("--min_low_rows", type=int, default=8)
    p.add_argument("--kmeans_seed", type=int, default=0)
    p.add_argument("--band_quantile", type=float, default=0.90)
    p.add_argument("--band_z", type=float, default=1.0)
    p.add_argument("--min_band_width", type=int, default=2)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--max_records", type=int, default=0)
    return p.parse_args()


def resolve_manifest_arg(result_root, manifest_tsv):
    if manifest_tsv is None:
        return (result_root / "manifest.tsv").resolve()
    raw = Path(manifest_tsv).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, result_root / raw, result_root.parent / raw, raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_run_path(path_value, result_root, pipeline_dir):
    path = resolve_existing_path(path_value, result_root, pipeline_dir)
    if path.exists():
        return path
    raw = Path(str(path_value))
    parts = raw.parts
    candidates = []
    if "runs" in parts:
        suffix = Path(*parts[parts.index("runs"):])
        candidates.extend([result_root / suffix, result_root.parent / suffix])
    if len(parts) >= 3:
        candidates.append(result_root / Path(*parts[-3:]))
        candidates.append(result_root.parent / Path(*parts[-3:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def load_neq_by_name(test_csv):
    df = pd.read_csv(test_csv)
    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]) if "name" in df.columns else str(row["sequence"])
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[name] = {"sequence": str(row["sequence"]), "neq": neq}
    return out


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(int(label), str(label))


def terminal_mask(n, count, fraction):
    exclusion = max(int(count), int(np.ceil(float(fraction) * n)))
    mask = np.ones(n, dtype=bool)
    if exclusion > 0:
        mask[:exclusion] = False
        mask[max(0, n - exclusion):] = False
    return mask, exclusion


def attention_metrics(attn):
    n = attn.shape[0]
    work = np.asarray(attn, dtype=float).copy()
    np.fill_diagonal(work, np.nan)
    received = np.nansum(work, axis=0)
    max_received = np.nanmax(work, axis=0)
    in_degree = np.zeros(n, dtype=float)
    weighted = np.zeros(n, dtype=float)
    for i in range(n):
        row = work[i]
        if not np.any(np.isfinite(row)):
            continue
        j = int(np.nanargmax(row))
        w = float(row[j])
        if np.isfinite(w):
            in_degree[j] += 1.0
            weighted[j] += w
    return {
        "received_attention": received,
        "max_received_attention": max_received,
        "in_degree_top1": in_degree,
        "weighted_in_degree_top1": weighted,
    }


def aligned_ss(ss_map, protein, n):
    entry = ss_map.get(protein)
    if entry is None:
        return np.full(n, "", dtype=object), np.full(n, "", dtype=object)
    q3 = np.asarray(entry["q3"][:n], dtype=object)
    q8 = np.asarray(entry["q8"][:n], dtype=object)
    if len(q3) != n:
        q3 = np.full(n, "", dtype=object)
    if len(q8) != n:
        q8 = np.full(n, "", dtype=object)
    return q3, q8


def high_mask(values, eligible, quantile):
    values = np.asarray(values, dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    eligible = np.asarray(eligible, dtype=int)
    valid = eligible[np.isfinite(values[eligible])]
    if len(valid) == 0:
        return mask
    threshold = float(np.quantile(values[valid], quantile))
    mask[valid] = values[valid] >= threshold
    return mask


def q3_boundary_mask(q3, window):
    q3 = np.asarray(q3, dtype=object)
    n = len(q3)
    mask = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if q3[i] != "" and q3[i - 1] != "" and q3[i] != q3[i - 1]:
            start = max(0, i - window)
            end = min(n, i + window + 1)
            mask[start:end] = True
    return mask


def nearest_distances(indices, target_mask):
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    target = np.where(np.asarray(target_mask, dtype=bool))[0]
    if len(indices) == 0:
        return np.asarray([], dtype=float)
    if len(target) == 0:
        return np.full(len(indices), np.nan)
    return np.asarray([np.min(np.abs(target - i)) for i in indices], dtype=float)


def top_indices(score, eligible, frac):
    score = np.asarray(score, dtype=float)
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    valid = eligible[np.isfinite(score[eligible])]
    if len(valid) == 0:
        return np.asarray([], dtype=int)
    k = max(1, int(np.ceil(float(frac) * len(valid))))
    return np.asarray([int(i) for i in valid[np.argsort(score[valid])[::-1]][:k]], dtype=int)


def pool_indices(modes, eligible, pool):
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    if pool == "all":
        return eligible
    label = {"low_mode_1": 1, "low_mode_2": 2, "diffuse_high_entropy": 0}.get(pool)
    if label is None:
        raise ValueError(f"Unknown pool mode: {pool}")
    return eligible[modes[eligible] == label]


def vertical_band_peak_indices(received, bands, modes, eligible, pool):
    eligible_set = set(int(i) for i in eligible)
    peaks = []
    for start, end in bands:
        region = np.asarray([i for i in range(start, end + 1) if i in eligible_set], dtype=int)
        if len(region) == 0:
            continue
        peak = int(region[np.nanargmax(received[region])])
        peaks.append(peak)
    if not peaks:
        return np.asarray([], dtype=int)
    peaks = np.asarray(sorted(set(peaks)), dtype=int)
    if pool != "all":
        peaks = pool_indices(modes, peaks, pool)
    return peaks


def matched_background(selected, eligible, q3, q8, modes, rng, n_per_selected, scope):
    selected = sorted(set(int(i) for i in selected))
    selected_set = set(selected)
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    sampled = []
    if not selected or len(eligible) == 0:
        return sampled
    for idx in selected:
        candidates = eligible
        if scope in {"same_q3", "same_q3_q8", "same_q3_mode", "same_q3_q8_mode"} and q3[idx] != "":
            candidates = candidates[q3[candidates] == q3[idx]]
        if scope in {"same_q3_q8", "same_q3_q8_mode"} and q8[idx] != "":
            candidates = candidates[q8[candidates] == q8[idx]]
        if scope in {"same_q3_mode", "same_q3_q8_mode"}:
            candidates = candidates[modes[candidates] == modes[idx]]
        candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([c for c in eligible if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = eligible
        sampled.extend(int(x) for x in rng.choice(candidates, size=n_per_selected, replace=len(candidates) < n_per_selected))
    return np.asarray(sampled, dtype=int)


def summarize_distances(indices, target_masks, windows):
    out = {"n_residues": int(len(indices))}
    for target_name, mask in target_masks.items():
        dist = nearest_distances(indices, mask)
        finite = dist[np.isfinite(dist)]
        out[f"{target_name}__n_targets"] = int(np.asarray(mask, dtype=bool).sum())
        out[f"{target_name}__mean_distance"] = float(np.mean(finite)) if len(finite) else np.nan
        out[f"{target_name}__median_distance"] = float(np.median(finite)) if len(finite) else np.nan
        out[f"{target_name}__fraction_exact"] = float(np.mean(dist == 0)) if len(dist) else np.nan
        for window in windows:
            out[f"{target_name}__fraction_within_{int(window)}"] = float(np.mean(dist <= window)) if len(dist) else np.nan
    return out


def decode_pb_array(arr):
    if arr.dtype.kind == "S":
        return np.char.decode(arr.astype("S1"), "ascii")
    return arr.astype(str)


def entropy_bits(values, exclude_z=True):
    values = np.asarray(values, dtype=str)
    if exclude_z:
        values = values[values != "Z"]
    if len(values) == 0:
        return np.nan
    _, counts = np.unique(values, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return -float(np.sum(probs * np.log2(probs + 1e-12)))


def transition_rate(states):
    states = np.asarray(states, dtype=str)
    if len(states) < 2:
        return np.nan
    valid = (states[:-1] != "Z") & (states[1:] != "Z")
    if not np.any(valid):
        return np.nan
    return float(np.mean(states[:-1][valid] != states[1:][valid]))


def load_strain_summary(atlas_root, protein, n):
    path = Path(atlas_root) / protein / "strain_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "residue" not in df.columns or "ensemble_mean" not in df.columns:
        return None
    strain = np.full(n, np.nan)
    for _, row in df.iterrows():
        idx = int(row["residue"]) - 1
        if 0 <= idx < n:
            strain[idx] = float(row["ensemble_mean"])
    return strain


def load_pb_summary(atlas_root, protein, n, replicates, exclude_z=True):
    if h5py is None:
        raise RuntimeError("h5py is required for PB distance targets.")
    protein_dir = Path(atlas_root) / protein
    arrays = []
    for rep in replicates:
        path = protein_dir / f"pb_states_{rep}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            if "pb_states" not in handle:
                continue
            states = decode_pb_array(handle["pb_states"][()])
        if states.ndim == 2 and states.shape[1] >= n:
            arrays.append(states[:, :n])
    if not arrays:
        return None
    all_states = np.concatenate(arrays, axis=0)
    entropy = np.full(n, np.nan)
    transition = np.full(n, np.nan)
    for i in range(n):
        entropy[i] = entropy_bits(all_states[:, i], exclude_z)
        rep_rates = [transition_rate(states[:, i]) for states in arrays]
        transition[i] = float(np.nanmean(rep_rates)) if np.any(np.isfinite(rep_rates)) else np.nan
    return {"pb_entropy_bits": entropy, "pb_transition_rate": transition}


def target_masks_for_protein(q3, q8, neq, eligible, args, strain=None, pb=None):
    masks = {
        "q8_turn_T": q8 == "T",
        "q8_bend_S": q8 == "S",
        "q8_coil_C": q8 == "C",
        "q8_loop_turn_bend_CTS": np.isin(q8, ["C", "T", "S"]),
        "q3_boundary_w1": q3_boundary_mask(q3, 1),
        "q3_boundary_w2": q3_boundary_mask(q3, 2),
        "high_neq_peak": neq_peak_mask(neq, args.peak_quantile),
        "high_neq_quantile": high_mask(neq, eligible, args.peak_quantile),
    }
    if strain is not None:
        masks["high_strain_quantile"] = high_mask(strain, eligible, args.high_strain_quantile)
    if pb is not None:
        masks["high_pb_entropy_quantile"] = high_mask(pb["pb_entropy_bits"], eligible, args.high_pb_entropy_quantile)
        masks["high_pb_transition_quantile"] = high_mask(pb["pb_transition_rate"], eligible, args.high_pb_transition_quantile)
    return masks


def flatten_summary(row_prefix, obs, bg):
    row = dict(row_prefix)
    row.update({f"obs_{k}": v for k, v in obs.items()})
    row.update({f"bg_{k}": v for k, v in bg.items()})
    for key, value in obs.items():
        if key == "n_residues" or key.endswith("__n_targets"):
            continue
        bg_key = key
        if bg_key in bg:
            try:
                row[f"delta_{key}"] = float(value) - float(bg[bg_key])
            except Exception:
                row[f"delta_{key}"] = np.nan
    return row


def aggregate_distance_summary(df):
    if df.empty:
        return pd.DataFrame()
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    rows = []
    group_cols = ["condition", "attention_score", "selection_type", "pool", "top_frac"]
    for key, group in df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key))
        row["n_units"] = len(group)
        row["n_proteins"] = group["protein"].nunique()
        for col in delta_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            if vals.notna().any():
                row[f"{col}__mean"] = float(vals.mean())
                row[f"{col}__median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    manifest_path = resolve_manifest_arg(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_attention_distance_enrichment", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()

    ss_map = load_q3_q8_map(args.ss_csv)
    neq_map = load_neq_by_name(args.test_csv)
    rng = np.random.default_rng(args.random_seed)
    atlas_root = Path(args.atlas_root).expanduser().resolve() if args.atlas_root else None
    strain_cache = {}
    pb_cache = {}
    detail_rows = []
    selected_rows = []
    missing = []
    processed = 0

    for run in manifest.itertuples(index=False):
        run_dict = run._asdict()
        attention_path = resolve_run_path(run_dict["attention_json"], result_root, pipeline_dir)
        if not attention_path.exists():
            missing.append(str(attention_path))
            continue
        print(f"[load] {run_dict.get('condition')} seed={run_dict.get('seed')}: {attention_path}", flush=True)
        records = json.loads(attention_path.read_text())
        for record in records:
            if args.max_records and processed >= args.max_records:
                break
            protein = str(record["name"])
            if protein not in neq_map:
                continue
            sequence = str(record["sequence"])
            n = len(sequence)
            neq = neq_map[protein]["neq"][:n]
            if len(neq) != n:
                continue
            q3, q8 = aligned_ss(ss_map, protein, n)
            nonterminal, terminal_exclusion_used = terminal_mask(n, args.terminal_exclusion, args.terminal_exclusion_fraction)
            eligible = np.where(nonterminal)[0]

            strain = None
            pb = None
            if atlas_root is not None:
                if protein not in strain_cache:
                    strain_cache[protein] = load_strain_summary(atlas_root, protein, n)
                strain = strain_cache[protein]
                if protein not in pb_cache:
                    pb_cache[protein] = load_pb_summary(
                        atlas_root, protein, n, args.replicates, args.exclude_z_from_entropy
                    )
                pb = pb_cache[protein]

            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            modes, row_ent, *_ = analyze_attention_modes(
                attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed
            )
            metrics = attention_metrics(attn)
            received = metrics["received_attention"]
            bands, _, _, _ = detect_bands(
                received,
                quantile=args.band_quantile,
                z_thresh=args.band_z,
                min_width=args.min_band_width,
                smooth_window=args.smooth_window,
            )
            target_masks = target_masks_for_protein(q3, q8, neq, eligible, args, strain=strain, pb=pb)

            for score_name, score in metrics.items():
                if score_name not in args.attention_scores:
                    continue
                for pool in args.pool_modes:
                    pool_idx = pool_indices(modes, eligible, pool)
                    if len(pool_idx) == 0:
                        continue
                    for top_frac in args.top_fracs:
                        selections = {}
                        if "top_hubs" in args.selection_types:
                            selections["top_hubs"] = top_indices(score, pool_idx, top_frac)
                        if "vertical_band_peaks" in args.selection_types:
                            selections["vertical_band_peaks"] = vertical_band_peak_indices(
                                received, bands, modes, pool_idx, pool
                            )
                        for selection_type, selected in selections.items():
                            selected = np.asarray(sorted(set(int(i) for i in selected)), dtype=int)
                            if len(selected) == 0:
                                continue
                            bg = matched_background(
                                selected,
                                pool_idx,
                                q3,
                                q8,
                                modes,
                                rng,
                                args.background_per_selected,
                                args.background_scope,
                            )
                            obs = summarize_distances(selected, target_masks, args.distance_windows)
                            bg_summary = summarize_distances(bg, target_masks, args.distance_windows)
                            prefix = {
                                "condition": run_dict.get("condition", ""),
                                "seed": int(run_dict.get("seed", -1)),
                                "protein": protein,
                                "attention_score": score_name,
                                "selection_type": selection_type,
                                "pool": pool,
                                "top_frac": float(top_frac),
                                "terminal_exclusion_used": terminal_exclusion_used,
                                "background_scope": args.background_scope,
                            }
                            detail_rows.append(flatten_summary(prefix, obs, bg_summary))
                            for idx in selected:
                                selected_rows.append({
                                    **prefix,
                                    "position_1based": int(idx) + 1,
                                    "aa": sequence[int(idx)],
                                    "q3": q3[int(idx)],
                                    "q8": q8[int(idx)],
                                    "mode": mode_name(int(modes[int(idx)])),
                                    "mode_label": int(modes[int(idx)]),
                                    "neq": float(neq[int(idx)]),
                                    "score": float(score[int(idx)]) if np.isfinite(score[int(idx)]) else np.nan,
                                    "row_entropy": float(row_ent[int(idx)]),
                                    **{
                                        f"distance_to_{name}": float(nearest_distances([idx], mask)[0])
                                        if len(nearest_distances([idx], mask)) else np.nan
                                        for name, mask in target_masks.items()
                                    },
                                })
            processed += 1
        if args.max_records and processed >= args.max_records:
            break

    detail_df = pd.DataFrame(detail_rows)
    selected_df = pd.DataFrame(selected_rows)
    summary_df = aggregate_distance_summary(detail_df)
    detail_df.to_csv(output_dir / "attention_distance_enrichment_by_run.csv", index=False)
    selected_df.to_csv(output_dir / "attention_distance_selected_residues.csv", index=False)
    summary_df.to_csv(output_dir / "attention_distance_enrichment_summary.csv", index=False)
    if missing:
        (output_dir / "missing_inputs.txt").write_text("\n".join(missing) + "\n")

    lines = [
        f"Result root: {result_root}",
        f"Manifest: {manifest_path}",
        f"Output dir: {output_dir}",
        f"Runs in manifest after filtering: {len(manifest)}",
        f"Protein/run records analyzed: {processed}",
        f"Distance enrichment rows: {len(detail_df)}",
        f"Selected residue rows: {len(selected_df)}",
        f"Atlas root: {atlas_root if atlas_root else 'not provided'}",
        f"Background scope: {args.background_scope}",
        f"Distance windows: {args.distance_windows}",
        "",
        "Key outputs:",
        "  attention_distance_enrichment_by_run.csv",
        "  attention_distance_enrichment_summary.csv",
        "  attention_distance_selected_residues.csv",
    ]
    (output_dir / "attention_distance_enrichment_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
