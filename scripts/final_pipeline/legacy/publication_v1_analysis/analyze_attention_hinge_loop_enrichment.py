#!/usr/bin/env python3
"""
Analyze hinge/loop enrichment in low_mode_1 attention hubs.

This script asks whether low_mode_1 attention-selected residues are enriched
for turn/bend/coil Q8 states, secondary-structure boundaries, and structured
linker loops after matching controls by protein, row mode, and Q3 class.

It also builds PWM/logo files restricted to hinge-like low_mode_1 hubs.
"""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_attention_hubs_vs_neq_peaks import neq_peak_mask
from analyze_attention_morphology_biology import detect_bands
from analyze_attention_row_modes import (
    analyze_attention_modes,
    default_analysis_dir,
    resolve_existing_path,
    resolve_manifest_path,
)
from build_row_mode_attention_pwms import AA20, add_pwm, extract_window, pwm_to_df, sanitize


def parse_args():
    parser = argparse.ArgumentParser(description="Q8 hinge/loop enrichment for low_mode_1 attention hubs.")
    parser.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    parser.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Defaults to result_root/manifest.tsv. "
             "Use manifest_attention_sources.tsv for the 30-attention source view.",
    )
    parser.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    parser.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id, q3, and q8 columns.")
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_hinge_loop_enrichment.")
    parser.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument(
        "--row_mode_assignments",
        default=None,
        help="Residue assignment CSV from analyze_attention_row_modes.py. "
             "Default: result_root/analysis_row_modes/row_mode_assignments_by_residue.csv if present.",
    )
    parser.add_argument("--top_frac", type=float, default=0.10)
    parser.add_argument("--peak_quantile", type=float, default=0.90)
    parser.add_argument("--high_entropy_quantile", type=float, default=0.67)
    parser.add_argument("--min_low_rows", type=int, default=8)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--band_quantile", type=float, default=0.90)
    parser.add_argument("--band_z", type=float, default=1.0)
    parser.add_argument("--min_band_width", type=int, default=2)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--boundary_window", type=int, default=2,
                        help="Residues within this distance of a Q3 transition are SS-boundary residues.")
    parser.add_argument("--min_loop_len", type=int, default=2)
    parser.add_argument("--max_loop_len", type=int, default=20)
    parser.add_argument("--flank_window", type=int, default=3,
                        help="Structured linker loop must have H/E within this many residues on both sides.")
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--permutation_seed", type=int, default=123)
    parser.add_argument("--motif_len", type=int, default=11)
    parser.add_argument("--background_per_selected", type=int, default=5)
    parser.add_argument("--min_instances_for_logo", type=int, default=20)
    return parser.parse_args()


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(int(label), str(label))


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def load_neq_by_name(test_csv):
    df = pd.read_csv(test_csv)
    out = {}
    for _, row in df.iterrows():
        if "name" not in df.columns:
            continue
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[str(row["name"])] = {
            "sequence": str(row["sequence"]),
            "neq": neq,
            "flexible": classify_neq(neq),
        }
    return out


def load_q3_q8_map(ss_csv):
    df = pd.read_csv(ss_csv)
    columns = {c.strip(): c for c in df.columns}
    id_col = columns.get("id")
    q3_col = columns.get("q3")
    q8_col = columns.get("q8")
    if id_col is None or q3_col is None or q8_col is None:
        raise ValueError(f"{ss_csv} must contain id, q3, and q8 columns.")
    out = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col]).lstrip(">")
        out.setdefault(seq_id, {"q3": [], "q8": []})
        out[seq_id]["q3"].append(str(row[q3_col]).strip())
        out[seq_id]["q8"].append(str(row[q8_col]).strip())
    return out


def resolve_attention_path(path_value, result_root, pipeline_dir):
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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def top_indices(values, frac):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    k = max(1, int(np.ceil(len(values) * frac)))
    return np.argsort(values)[-k:].tolist()


def q3_boundary_mask(q3, window):
    q3 = np.asarray(q3)
    n = len(q3)
    transition_positions = []
    for i in range(n - 1):
        if q3[i] != q3[i + 1]:
            transition_positions.extend([i, i + 1])
    mask = np.zeros(n, dtype=bool)
    for pos in transition_positions:
        start = max(0, pos - window)
        end = min(n, pos + window + 1)
        mask[start:end] = True
    return mask


def structured_linker_loop_mask(q3, q8, min_len, max_len, flank_window):
    q3 = np.asarray(q3)
    q8 = np.asarray(q8)
    n = len(q8)
    loop_like = np.isin(q8, ["C", "T", "S"])
    structured = np.isin(q3, ["H", "E"])
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not loop_like[i]:
            i += 1
            continue
        start = i
        while i < n and loop_like[i]:
            i += 1
        end = i - 1
        length = end - start + 1
        if length < min_len or length > max_len:
            continue
        left_start = max(0, start - flank_window)
        right_end = min(n, end + flank_window + 1)
        left_structured = bool(np.any(structured[left_start:start]))
        right_structured = bool(np.any(structured[end + 1:right_end]))
        if left_structured and right_structured:
            mask[start:end + 1] = True
    return mask


def build_selected_sets(received, modes, bands, peak_mask, top_frac):
    selected = {}
    low_idx = np.where(modes == 1)[0]
    if len(low_idx):
        top_low = set(int(i) for i in low_idx[top_indices(received[low_idx], top_frac)])
        selected["top_received__low_mode_1"] = top_low
        selected["neq_peak_overlapping_hubs__low_mode_1"] = set(i for i in top_low if peak_mask[i])

    band_peaks = set()
    for start, end in bands:
        region = np.arange(start, end + 1)
        peak = int(region[np.argmax(received[region])])
        if modes[peak] == 1:
            band_peaks.add(peak)
    selected["vertical_band_peaks__low_mode_1"] = band_peaks
    return selected


def matched_sample(indices, modes, q3, rng):
    selected = sorted(set(int(i) for i in indices))
    selected_set = set(selected)
    n = len(modes)
    sampled = []
    for idx in selected:
        candidates = np.where((modes == modes[idx]) & (q3 == q3[idx]))[0]
        candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.where(modes == modes[idx])[0]
            candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.arange(n)
        sampled.append(int(rng.choice(candidates)))
    return sampled


def summarize_indices(indices, q3, q8, neq, peak_mask, boundary, linker, hinge_like):
    indices = sorted(set(int(i) for i in indices))
    if not indices:
        return None
    idx = np.asarray(indices, dtype=int)
    out = {
        "n_selected": len(idx),
        "fraction_q8_turn_T": float(np.mean(q8[idx] == "T")),
        "fraction_q8_bend_S": float(np.mean(q8[idx] == "S")),
        "fraction_q8_coil_C": float(np.mean(q8[idx] == "C")),
        "fraction_q8_turn_or_bend_TS": float(np.mean(np.isin(q8[idx], ["T", "S"]))),
        "fraction_q8_loop_turn_bend_CTS": float(np.mean(np.isin(q8[idx], ["C", "T", "S"]))),
        "fraction_q3_coil_C": float(np.mean(q3[idx] == "C")),
        "fraction_ss_boundary": float(np.mean(boundary[idx])),
        "fraction_structured_linker_loop": float(np.mean(linker[idx])),
        "fraction_hinge_like": float(np.mean(hinge_like[idx])),
        "mean_neq": float(np.mean(neq[idx])),
        "fraction_flexible": float(np.mean(neq[idx] > 1.0)),
        "fraction_neq_peak": float(np.mean(peak_mask[idx])),
    }
    return out


def empirical_pvalues(observed_df, null_df):
    if observed_df.empty or null_df.empty:
        return pd.DataFrame()
    metrics = [
        "fraction_q8_turn_T",
        "fraction_q8_bend_S",
        "fraction_q8_coil_C",
        "fraction_q8_turn_or_bend_TS",
        "fraction_q8_loop_turn_bend_CTS",
        "fraction_ss_boundary",
        "fraction_structured_linker_loop",
        "fraction_hinge_like",
        "mean_neq",
        "fraction_flexible",
        "fraction_neq_peak",
    ]
    rows = []
    for key, obs_group in observed_df.groupby(["condition", "selection"]):
        null_group = null_df[(null_df["condition"] == key[0]) & (null_df["selection"] == key[1])]
        if null_group.empty:
            continue
        row = {"condition": key[0], "selection": key[1], "n_observed_rows": len(obs_group), "n_null_rows": len(null_group)}
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


def matched_background_for_pwm(selected, modes, q3, rng, n_per_selected):
    bg = []
    selected = sorted(set(int(i) for i in selected))
    selected_set = set(selected)
    n = len(modes)
    for idx in selected:
        candidates = np.where((modes == modes[idx]) & (q3 == q3[idx]))[0]
        candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([c for c in range(n) if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.arange(n)
        replace = len(candidates) < n_per_selected
        bg.extend(int(x) for x in rng.choice(candidates, size=n_per_selected, replace=replace))
    return bg


def load_assignment_maps(path, conditions=None):
    path = Path(path)
    if not path.exists():
        return {}
    print(f"[load] row-mode assignments: {path}")
    cols = ["condition", "seed", "protein", "position_1based", "mode_label"]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    if conditions:
        df = df[df["condition"].isin(conditions)].copy()
    maps = {}
    for key, group in df.groupby(["condition", "seed", "protein"], sort=False):
        group = group.sort_values("position_1based")
        maps[(key[0], int(key[1]), key[2])] = group["mode_label"].to_numpy(dtype=int)
    return maps


def try_plot_logo(pwm_df, title, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import logomaker
    except Exception as exc:
        print(f"[logo skip] {out_path.name}: {exc}")
        return False

    mat = pwm_df.pivot(index="position", columns="AA", values="log2_enrichment").fillna(0.0)
    mat = mat.clip(lower=0.0)
    if mat.to_numpy().max() <= 0:
        return False
    colors = {
        "G": "#888888", "P": "#e67e00",
        "A": "#222222", "V": "#222222", "L": "#222222", "I": "#222222", "M": "#222222",
        "F": "#7b2d8b", "W": "#7b2d8b", "Y": "#7b2d8b",
        "S": "#2ca02c", "T": "#2ca02c", "C": "#2ca02c", "N": "#2ca02c", "Q": "#2ca02c",
        "D": "#d62728", "E": "#d62728",
        "K": "#1f77b4", "R": "#1f77b4", "H": "#1f77b4",
    }
    fig, ax = plt.subplots(figsize=(max(7, len(mat) * 0.7), 3.2))
    logomaker.Logo(mat, ax=ax, color_scheme={aa: colors.get(aa, "#333333") for aa in mat.columns})
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("log2 enrichment")
    ax.set_xlabel("Position relative to hub")
    ax.set_xticks(range(len(mat.index)))
    ax.set_xticklabels([f"{int(x):+d}" for x in mat.index])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    args = parse_args()
    if args.motif_len % 2 != 1:
        raise ValueError("--motif_len should be odd.")

    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parents[2]
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest TSV: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_hinge_loop_enrichment", manifest_path)
    )
    pwm_dir = output_dir / "pwms"
    logo_dir = output_dir / "logos"
    output_dir.mkdir(parents=True, exist_ok=True)
    pwm_dir.mkdir(exist_ok=True)
    logo_dir.mkdir(exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()
    neq_by_name = load_neq_by_name(args.test_csv)
    ss_map = load_q3_q8_map(args.ss_csv)
    assignment_path = (
        Path(args.row_mode_assignments).expanduser()
        if args.row_mode_assignments
        else default_analysis_dir(result_root, "analysis_row_modes", manifest_path) / "row_mode_assignments_by_residue.csv"
    )
    assignment_maps = load_assignment_maps(assignment_path, set(args.conditions or []))
    rng = np.random.default_rng(args.permutation_seed)

    observed_rows = []
    null_rows = []
    residue_rows = []
    obs_pwms = defaultdict(lambda: np.zeros((args.motif_len, len(AA20))))
    bg_pwms = defaultdict(lambda: np.zeros((args.motif_len, len(AA20))))
    pwm_instance_rows = []

    for run in manifest.itertuples(index=False):
        attention_path = resolve_attention_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            print(f"[skip] missing {attention_path}")
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            protein = record["name"]
            if protein not in neq_by_name or protein not in ss_map:
                continue
            seq = record["sequence"]
            n = len(seq)
            neq = neq_by_name[protein]["neq"][:n]
            q3 = np.asarray(ss_map[protein]["q3"][:n])
            q8 = np.asarray(ss_map[protein]["q8"][:n])
            if len(neq) != n or len(q3) != n or len(q8) != n:
                continue

            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            received = attn.sum(axis=0)
            cached_modes = assignment_maps.get((run.condition, int(run.seed), protein))
            if cached_modes is not None and len(cached_modes) >= n:
                modes = cached_modes[:n]
            else:
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
            boundary = q3_boundary_mask(q3, args.boundary_window)
            linker = structured_linker_loop_mask(
                q3, q8, args.min_loop_len, args.max_loop_len, args.flank_window
            )
            hinge_like = np.isin(q8, ["C", "T", "S"]) & (boundary | linker)
            selected = build_selected_sets(received, modes, bands, peak_mask, args.top_frac)

            for selection, indices in selected.items():
                if not indices:
                    continue
                obs = summarize_indices(indices, q3, q8, neq, peak_mask, boundary, linker, hinge_like)
                if obs is None:
                    continue
                obs.update({"condition": run.condition, "seed": int(run.seed), "protein": protein, "selection": selection})
                observed_rows.append(obs)

                for perm in range(1, args.n_permutations + 1):
                    sample = matched_sample(indices, modes, q3, rng)
                    null = summarize_indices(sample, q3, q8, neq, peak_mask, boundary, linker, hinge_like)
                    if null is None:
                        continue
                    null.update({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                        "selection": selection,
                        "permutation": perm,
                    })
                    null_rows.append(null)

                hinge_indices = sorted(int(i) for i in indices if hinge_like[int(i)])
                if hinge_indices:
                    key = f"{run.condition}__{selection}__hinge_like"
                    bg_idx = matched_background_for_pwm(hinge_indices, modes, q3, rng, args.background_per_selected)
                    for idx in hinge_indices:
                        add_pwm(obs_pwms[key], seq, idx, weight=float(received[idx]))
                        pwm_instance_rows.append({
                            "condition": run.condition,
                            "seed": int(run.seed),
                            "protein": protein,
                            "selection": selection,
                            "position_1based": idx + 1,
                            "aa": seq[idx],
                            "q3": q3[idx],
                            "q8": q8[idx],
                            "neq": float(neq[idx]),
                            "is_neq_peak": bool(peak_mask[idx]),
                            "is_ss_boundary": bool(boundary[idx]),
                            "is_structured_linker_loop": bool(linker[idx]),
                            "received_attention": float(received[idx]),
                            "motif": extract_window(seq, idx, args.motif_len),
                            "background": False,
                        })
                    for idx in bg_idx:
                        add_pwm(bg_pwms[key], seq, idx, weight=1.0)
                        pwm_instance_rows.append({
                            "condition": run.condition,
                            "seed": int(run.seed),
                            "protein": protein,
                            "selection": selection,
                            "position_1based": idx + 1,
                            "aa": seq[idx],
                            "q3": q3[idx],
                            "q8": q8[idx],
                            "neq": float(neq[idx]),
                            "is_neq_peak": bool(peak_mask[idx]),
                            "is_ss_boundary": bool(boundary[idx]),
                            "is_structured_linker_loop": bool(linker[idx]),
                            "received_attention": float(received[idx]),
                            "motif": extract_window(seq, idx, args.motif_len),
                            "background": True,
                        })

                for idx in sorted(indices):
                    residue_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                        "selection": selection,
                        "position_1based": idx + 1,
                        "aa": seq[idx],
                        "q3": q3[idx],
                        "q8": q8[idx],
                        "mode": mode_name(modes[idx]),
                        "neq": float(neq[idx]),
                        "is_neq_peak": bool(peak_mask[idx]),
                        "is_ss_boundary": bool(boundary[idx]),
                        "is_structured_linker_loop": bool(linker[idx]),
                        "is_hinge_like": bool(hinge_like[idx]),
                        "received_attention": float(received[idx]),
                    })

    observed_df = pd.DataFrame(observed_rows)
    null_df = pd.DataFrame(null_rows)
    residue_df = pd.DataFrame(residue_rows)
    pwm_instances = pd.DataFrame(pwm_instance_rows)

    observed_df.to_csv(output_dir / "hinge_loop_enrichment_by_run.csv", index=False)
    null_df.to_csv(output_dir / "hinge_loop_enrichment_matched_null.csv", index=False)
    residue_df.to_csv(output_dir / "low_mode_1_hinge_loop_selected_residues.csv", index=False)
    pwm_instances.to_csv(output_dir / "hinge_like_pwm_instances.csv", index=False)

    pvals = empirical_pvalues(observed_df, null_df)
    pvals.to_csv(output_dir / "hinge_loop_enrichment_matched_pvalues.csv", index=False)

    if not observed_df.empty:
        metrics = [c for c in observed_df.columns if c.startswith("fraction_") or c == "mean_neq"]
        summary = observed_df.groupby(["condition", "selection"])[metrics].agg(["mean", "std", "count"])
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]
        summary.reset_index().to_csv(output_dir / "hinge_loop_enrichment_summary.csv", index=False)

    pwm_summary_rows = []
    for key, obs_pwm in sorted(obs_pwms.items()):
        n_obs = int((~pwm_instances["background"] & ((pwm_instances["condition"] + "__" + pwm_instances["selection"] + "__hinge_like") == key)).sum())
        n_bg = int((pwm_instances["background"] & ((pwm_instances["condition"] + "__" + pwm_instances["selection"] + "__hinge_like") == key)).sum())
        condition, selection_with_suffix = key.split("__", 1)
        suffix = "__hinge_like"
        selection = (
            selection_with_suffix[:-len(suffix)]
            if selection_with_suffix.endswith(suffix)
            else selection_with_suffix
        )
        written = n_obs >= args.min_instances_for_logo
        pwm_summary_rows.append({
            "condition": condition,
            "selection": selection,
            "subset": "hinge_like",
            "n_observed_instances": n_obs,
            "n_background_instances": n_bg,
            "pwm_written": written,
        })
        if not written:
            continue
        pwm_df = pwm_to_df(obs_pwm, bg_pwms[key])
        safe = sanitize(key)
        pwm_path = pwm_dir / f"pwm__{safe}.csv"
        logo_path = logo_dir / f"logo__{safe}.png"
        pwm_df.to_csv(pwm_path, index=False)
        try_plot_logo(pwm_df, f"{condition} {selection} hinge-like", logo_path)
    pd.DataFrame(pwm_summary_rows).to_csv(output_dir / "hinge_like_pwm_summary.csv", index=False)

    lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Observed selection rows: {len(observed_df)}",
        f"Matched null rows: {len(null_df)}",
        f"Selected residue rows: {len(residue_df)}",
        f"Hinge-like PWM rows: {len(pwm_instances)}",
        "",
        "Key outputs:",
        "  hinge_loop_enrichment_summary.csv",
        "  hinge_loop_enrichment_matched_pvalues.csv",
        "  low_mode_1_hinge_loop_selected_residues.csv",
        "  hinge_like_pwm_instances.csv",
        "  pwms/pwm__*.csv",
        "  logos/logo__*.png",
    ]
    (output_dir / "hinge_loop_enrichment_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
