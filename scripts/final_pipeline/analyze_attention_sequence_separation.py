#!/usr/bin/env python3
"""
Analyze sequence-separation structure in row-mode attention maps.

Questions this answers:
  - Do low_mode_1 and low_mode_2 attend locally or long-range?
  - Are vertical-band/key hubs receiving attention from farther sequence
    distances than matched residues from the same protein/mode/SS class?
  - Are the beehive blocks dominated by near-diagonal or off-diagonal mass?
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


SEPARATION_BINS = [
    ("self", 0, 0),
    ("local_1_4", 1, 4),
    ("short_5_12", 5, 12),
    ("medium_13_24", 13, 24),
    ("long_25_48", 25, 48),
    ("very_long_gt48", 49, None),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Measure sequence-separation structure in attention maps.")
    parser.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    parser.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Defaults to result_root/manifest.tsv. "
             "Use manifest_attention_sources.tsv for the 30-attention source view.",
    )
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_sequence_separation.")
    parser.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    parser.add_argument("--test_csv", default=None, help="Optional CSV with name, sequence, neq.")
    parser.add_argument("--ss_csv", default=None, help="Optional NetSurfP CSV with id and q3 columns.")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument(
        "--row_mode_assignments",
        default=None,
        help="Residue assignment CSV from analyze_attention_row_modes.py. "
             "Default: result_root/analysis_row_modes/row_mode_assignments_by_residue.csv if present.",
    )
    parser.add_argument("--top_frac", type=float, default=0.10)
    parser.add_argument("--high_entropy_quantile", type=float, default=0.67)
    parser.add_argument("--min_low_rows", type=int, default=8)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--band_quantile", type=float, default=0.90)
    parser.add_argument("--band_z", type=float, default=1.0)
    parser.add_argument("--min_band_width", type=int, default=2)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--n_null_per_protein", type=int, default=200,
                        help="Matched random key-residue samples per protein/selection.")
    parser.add_argument("--random_seed", type=int, default=123)
    return parser.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(int(label), str(label))


def load_neq_by_name(test_csv):
    if test_csv is None or not Path(test_csv).expanduser().exists():
        return {}
    df = pd.read_csv(Path(test_csv).expanduser())
    out = {}
    for _, row in df.iterrows():
        if "name" not in df.columns:
            continue
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[str(row["name"])] = {"neq": neq, "flexible": classify_neq(neq)}
    return out


def load_ss_map(ss_csv):
    if ss_csv is None or not Path(ss_csv).expanduser().exists():
        return {}
    df = pd.read_csv(Path(ss_csv).expanduser())
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


def resolve_attention_path(path_value, result_root, pipeline_dir):
    path = resolve_existing_path(path_value, result_root, pipeline_dir)
    if path.exists():
        return path

    raw = Path(str(path_value))
    parts = raw.parts
    candidates = []
    if "runs" in parts:
        runs_idx = parts.index("runs")
        suffix = Path(*parts[runs_idx:])
        candidates.extend([
            result_root / suffix,
            result_root.parent / suffix,
        ])
    if len(parts) >= 3:
        candidates.append(result_root / Path(*parts[-3:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def separation_matrix(n):
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :])


def summarize_weighted_distances(weights, dist):
    weights = np.asarray(weights, dtype=float)
    dist = np.asarray(dist, dtype=int)
    total = float(np.sum(weights))
    if total <= 0:
        out = {
            "attention_mass": 0.0,
            "mean_abs_separation": np.nan,
            "median_abs_separation": np.nan,
            "normalized_mean_separation": np.nan,
        }
        for name, _, _ in SEPARATION_BINS:
            out[f"mass_{name}"] = np.nan
        return out

    flat_w = weights.ravel()
    flat_d = dist.ravel().astype(int, copy=False)
    max_d = int(np.max(flat_d)) if len(flat_d) else 0
    mass_by_dist = np.bincount(flat_d, weights=flat_w, minlength=max_d + 1)
    dist_values = np.arange(len(mass_by_dist), dtype=float)
    cum = np.cumsum(mass_by_dist)
    median = float(np.searchsorted(cum, total / 2.0))
    mean = float(np.sum(mass_by_dist * dist_values) / total)
    out = {
        "attention_mass": total,
        "mean_abs_separation": mean,
        "median_abs_separation": median,
        "normalized_mean_separation": mean / max(float(max_d), 1.0),
    }
    for name, low, high in SEPARATION_BINS:
        if high is None:
            mass = np.sum(mass_by_dist[low:])
        else:
            mass = np.sum(mass_by_dist[low:min(high, max_d) + 1]) if low <= max_d else 0.0
        out[f"mass_{name}"] = float(mass / total)
    return out


def top_indices(values, frac):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    k = max(1, int(np.ceil(len(values) * frac)))
    return np.argsort(values)[-k:].tolist()


def build_selected_keys(received, modes, bands, top_frac):
    selected = {}
    for label in [0, 1, 2]:
        idx = np.where(modes == label)[0]
        if len(idx):
            selected[f"top_received__{mode_name(label)}"] = set(int(i) for i in idx[top_indices(received[idx], top_frac)])

    band_peaks = {mode_name(label): set() for label in [0, 1, 2]}
    for start, end in bands:
        region = np.arange(start, end + 1)
        peak = int(region[np.argmax(received[region])])
        band_peaks[mode_name(int(modes[peak]))].add(peak)
    for mode, peaks in band_peaks.items():
        selected[f"vertical_band_peaks__{mode}"] = peaks
    return selected


def summarize_key_set(attn, key_indices, dist=None):
    key_indices = sorted(set(int(i) for i in key_indices))
    n = attn.shape[0]
    if not key_indices:
        return None
    if dist is None:
        dist = separation_matrix(n)
    dist = dist[:, key_indices]
    weights = attn[:, key_indices]
    out = summarize_weighted_distances(weights, dist)
    out["n_keys"] = len(key_indices)
    out["mean_received_attention"] = float(attn[:, key_indices].sum(axis=0).mean())
    return out


def matched_keys(key_indices, modes, ss, rng):
    n = len(modes)
    key_indices = sorted(set(int(i) for i in key_indices))
    if not key_indices:
        return []
    selected = set(key_indices)
    sampled = []
    for idx in key_indices:
        same_mode = modes == modes[idx]
        if ss is not None and len(ss) == n:
            candidates = np.where(same_mode & (ss == ss[idx]))[0]
        else:
            candidates = np.where(same_mode)[0]
        candidates = np.asarray([c for c in candidates if int(c) not in selected], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([c for c in range(n) if int(c) not in selected], dtype=int)
        if len(candidates) == 0:
            candidates = np.arange(n)
        sampled.append(int(rng.choice(candidates)))
    return sampled


def load_assignment_maps(path, conditions=None):
    path = Path(path)
    if not path.exists():
        return {}
    print(f"[load] row-mode assignments: {path}")
    cols = [
        "condition", "seed", "protein", "position_1based", "mode_label",
        "row_entropy", "ss", "flexible",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    if conditions:
        df = df[df["condition"].isin(conditions)].copy()
    maps = {}
    for key, group in df.groupby(["condition", "seed", "protein"], sort=False):
        group = group.sort_values("position_1based")
        maps[(key[0], int(key[1]), key[2])] = {
            "modes": group["mode_label"].to_numpy(dtype=int),
            "entropy": group["row_entropy"].to_numpy(dtype=float),
            "ss": group["ss"].astype(str).to_numpy(),
            "flexible": group["flexible"].to_numpy(dtype=float),
        }
    return maps


def summarize_record(record, run, args, ss_map, neq_by_name, rng, assignment_maps):
    protein = record["name"]
    sequence = record["sequence"]
    n = len(sequence)
    attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
    dist = separation_matrix(n)
    received = attn.sum(axis=0)
    cached = assignment_maps.get((run.condition, int(run.seed), protein))
    if cached is not None and len(cached["modes"]) >= n:
        modes = cached["modes"][:n]
        ent = cached["entropy"][:n]
    else:
        modes, ent, _, _, _, _ = analyze_attention_modes(
            attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed
        )
    bands, _, _, _ = detect_bands(
        received,
        quantile=args.band_quantile,
        z_thresh=args.band_z,
        min_width=args.min_band_width,
        smooth_window=args.smooth_window,
    )
    ss = cached["ss"][:n] if cached is not None and len(cached["ss"]) >= n else ss_map.get(protein)
    if ss is not None:
        ss = np.asarray(ss[:n])
        if len(ss) != n:
            ss = None
    neq_entry = neq_by_name.get(protein)
    flexible = (
        cached["flexible"][:n]
        if cached is not None and len(cached["flexible"]) >= n
        else neq_entry["flexible"][:n] if neq_entry is not None and len(neq_entry["flexible"]) >= n
        else None
    )

    protein_rows = []
    block_rows = []
    selection_rows = []
    null_rows = []

    base = {
        "condition": run.condition,
        "seed": int(run.seed),
        "protein": protein,
        "n_residues": n,
    }
    whole = summarize_weighted_distances(attn, dist)
    protein_rows.append({**base, "query_mode": "all", **whole})

    for q_label in [0, 1, 2]:
        q_idx = np.where(modes == q_label)[0]
        if len(q_idx) == 0:
            continue
        row = summarize_weighted_distances(attn[q_idx, :], dist[q_idx, :])
        protein_rows.append({
            **base,
            "query_mode": mode_name(q_label),
            "query_mode_fraction": float(len(q_idx) / n),
            "query_mode_mean_entropy": float(np.mean(ent[q_idx])),
            **row,
        })

        for k_label in [0, 1, 2]:
            k_idx = np.where(modes == k_label)[0]
            if len(k_idx) == 0:
                continue
            block = summarize_weighted_distances(attn[np.ix_(q_idx, k_idx)], dist[np.ix_(q_idx, k_idx)])
            block_rows.append({
                **base,
                "query_mode": mode_name(q_label),
                "key_mode": mode_name(k_label),
                "query_mode_size": len(q_idx),
                "key_mode_size": len(k_idx),
                **block,
            })

    selected = build_selected_keys(received, modes, bands, args.top_frac)
    for selection, keys in selected.items():
        summary = summarize_key_set(attn, keys, dist)
        if summary is None:
            continue
        key_list = sorted(keys)
        selection_rows.append({
            **base,
            "selection": selection,
            "key_mode": mode_name(modes[key_list[0]]) if key_list else "",
            "mean_key_entropy": float(np.mean(ent[key_list])) if key_list else np.nan,
            "mean_key_flexible": float(np.mean(flexible[key_list])) if flexible is not None and key_list else np.nan,
            "mean_key_is_coil": float(np.mean(ss[key_list] == "C")) if ss is not None and key_list else np.nan,
            "mean_key_is_helix": float(np.mean(ss[key_list] == "H")) if ss is not None and key_list else np.nan,
            "mean_key_is_strand": float(np.mean(ss[key_list] == "E")) if ss is not None and key_list else np.nan,
            **summary,
        })

        for perm in range(1, args.n_null_per_protein + 1):
            sampled = matched_keys(keys, modes, ss, rng)
            null_summary = summarize_key_set(attn, sampled, dist)
            if null_summary is None:
                continue
            null_rows.append({
                **base,
                "selection": selection,
                "permutation": perm,
                **null_summary,
            })

    return protein_rows, block_rows, selection_rows, null_rows


def aggregate(df, group_cols):
    value_cols = [
        "attention_mass",
        "mean_abs_separation",
        "median_abs_separation",
        "normalized_mean_separation",
        "mass_self",
        "mass_local_1_4",
        "mass_short_5_12",
        "mass_medium_13_24",
        "mass_long_25_48",
        "mass_very_long_gt48",
    ]
    cols = [c for c in value_cols if c in df.columns]
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols)[cols].agg(["mean", "std", "count"])
    out.columns = ["_".join(c).strip("_") for c in out.columns]
    return out.reset_index()


def empirical_selection_pvalues(selection_df, null_df):
    rows = []
    if selection_df.empty or null_df.empty:
        return pd.DataFrame()
    metrics = ["mean_abs_separation", "mass_very_long_gt48", "mass_long_25_48", "mass_local_1_4"]
    null_grouped = null_df.groupby(["condition", "selection"])
    obs_grouped = selection_df.groupby(["condition", "selection"])
    for key, obs_group in obs_grouped:
        if key not in null_grouped.groups:
            continue
        null_group = null_grouped.get_group(key)
        row = {"condition": key[0], "selection": key[1], "n_observed": len(obs_group), "n_null": len(null_group)}
        for metric in metrics:
            obs = float(obs_group[metric].mean())
            null_values = null_group.groupby(["protein", "seed", "permutation"])[metric].mean().to_numpy()
            row[f"observed_{metric}"] = obs
            row[f"null_mean_{metric}"] = float(np.mean(null_values))
            row[f"delta_{metric}"] = obs - float(np.mean(null_values))
            row[f"p_greater_{metric}"] = float((np.sum(null_values >= obs) + 1) / (len(null_values) + 1))
            row[f"p_less_{metric}"] = float((np.sum(null_values <= obs) + 1) / (len(null_values) + 1))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest TSV: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_sequence_separation", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()
    ss_map = load_ss_map(args.ss_csv)
    neq_by_name = load_neq_by_name(args.test_csv)
    assignment_path = (
        Path(args.row_mode_assignments).expanduser()
        if args.row_mode_assignments
        else default_analysis_dir(result_root, "analysis_row_modes", manifest_path) / "row_mode_assignments_by_residue.csv"
    )
    assignment_maps = load_assignment_maps(assignment_path, set(args.conditions or []))
    rng = np.random.default_rng(args.random_seed)

    protein_rows = []
    block_rows = []
    selection_rows = []
    null_rows = []

    for run in manifest.itertuples(index=False):
        attention_path = resolve_attention_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            print(f"[skip] missing {attention_path}")
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            p_rows, b_rows, s_rows, n_rows = summarize_record(
                record, run, args, ss_map, neq_by_name, rng, assignment_maps
            )
            protein_rows.extend(p_rows)
            block_rows.extend(b_rows)
            selection_rows.extend(s_rows)
            null_rows.extend(n_rows)

    protein_df = pd.DataFrame(protein_rows)
    block_df = pd.DataFrame(block_rows)
    selection_df = pd.DataFrame(selection_rows)
    null_df = pd.DataFrame(null_rows)

    protein_df.to_csv(output_dir / "sequence_separation_by_protein_mode.csv", index=False)
    block_df.to_csv(output_dir / "sequence_separation_mode_blocks.csv", index=False)
    selection_df.to_csv(output_dir / "sequence_separation_selected_hubs.csv", index=False)
    null_df.to_csv(output_dir / "sequence_separation_selected_hubs_matched_null.csv", index=False)

    aggregate(protein_df, ["condition", "query_mode"]).to_csv(
        output_dir / "sequence_separation_by_condition_mode.csv", index=False
    )
    aggregate(block_df, ["condition", "query_mode", "key_mode"]).to_csv(
        output_dir / "sequence_separation_blocks_by_condition.csv", index=False
    )
    aggregate(selection_df, ["condition", "selection"]).to_csv(
        output_dir / "sequence_separation_selected_hubs_by_condition.csv", index=False
    )
    empirical_selection_pvalues(selection_df, null_df).to_csv(
        output_dir / "sequence_separation_selected_hubs_null_pvalues.csv", index=False
    )

    lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Protein/mode rows: {len(protein_df)}",
        f"Mode-block rows: {len(block_df)}",
        f"Selected hub rows: {len(selection_df)}",
        f"Matched-null rows: {len(null_df)}",
        "",
        "Key outputs:",
        "  sequence_separation_by_condition_mode.csv",
        "  sequence_separation_blocks_by_condition.csv",
        "  sequence_separation_selected_hubs_by_condition.csv",
        "  sequence_separation_selected_hubs_null_pvalues.csv",
    ]
    (output_dir / "sequence_separation_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
