#!/usr/bin/env python3
"""
Compare ESM2 and ESM3 BiLSTM attention maps.

For each freeze condition (frozen/top4/top28), the script averages attention
over seeds, computes ESM3 - ESM2 BiLSTM attention differences, and writes:
  - one interactive HTML page per selected protein with four rows:
      ESM2 backbone, ESM2 BiLSTM, ESM3 BiLSTM, ESM3-ESM2 BiLSTM difference
  - per-residue differential received-attention tables
  - top differential-residue enrichment summaries against same-Q3 controls
"""

import argparse
import ast
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyze_attention_row_modes import resolve_existing_path, resolve_manifest_path
from analyze_low_mode_control_points import load_q3_q8_map


FREEZE_LEVELS = ["frozen", "top4", "top28"]
ESM2_BACKBONE = {
    "frozen": "esm2_frozen_backbone_attention",
    "top4": "esm2_top4_backbone_attention",
    "top28": "esm2_top28_backbone_attention",
}
ESM2_BILSTM = {
    "frozen": "esm2_frozen_bilstm_attention",
    "top4": "esm2_top4_bilstm_attention",
    "top28": "esm2_top28_bilstm_attention",
}
ESM3_BILSTM = {
    "frozen": "esm3_frozen_bilstm_attention",
    "top4": "esm3_top4_bilstm_attention",
    "top28": "esm3_top28_bilstm_attention",
}


def parse_args():
    p = argparse.ArgumentParser(description="ESM2-vs-ESM3 differential attention analysis.")
    p.add_argument("--result_root", required=True)
    p.add_argument("--manifest_tsv", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pipeline_dir", default=None)
    p.add_argument("--test_csv", default=None, help="Optional CSV with name, sequence, neq.")
    p.add_argument("--ss_csv", default=None, help="Optional NetSurfP CSV with id/q3/q8.")
    p.add_argument("--n_proteins", type=int, default=5)
    p.add_argument("--proteins", nargs="*", default=None)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--seed_mode", choices=["average", "single"], default="average")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scale_scope", choices=["global", "protein"], default="global")
    p.add_argument("--scale_quantile", type=float, default=0.995)
    p.add_argument("--diff_quantile", type=float, default=0.995)
    p.add_argument("--top_fracs", nargs="+", type=float, default=[0.05, 0.10])
    p.add_argument("--tick_step", type=int, default=25)
    p.add_argument("--subplot_size", type=int, default=285)
    p.add_argument("--colorscale", default="Viridis")
    p.add_argument("--diff_colorscale", default="RdBu")
    p.add_argument("--include_plotlyjs", choices=["cdn", "inline"], default="cdn")
    return p.parse_args()


def load_records(path):
    records = json.loads(Path(path).read_text())
    return {str(r["name"]): r for r in records if "name" in r and "attention_weights" in r}


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


def resolve_manifest_arg(result_root, manifest_tsv):
    if manifest_tsv is None:
        return (result_root / "manifest_attention_sources.tsv").resolve()
    raw = Path(manifest_tsv).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            result_root / raw,
            result_root.parent / raw,
            raw,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def required_conditions():
    out = []
    for level in FREEZE_LEVELS:
        out.extend([ESM2_BACKBONE[level], ESM2_BILSTM[level], ESM3_BILSTM[level]])
    return out


def load_grouped(manifest, result_root, pipeline_dir, seed_mode, seed):
    grouped = {}
    for condition in required_conditions():
        sub = manifest[manifest["condition"] == condition].copy()
        if seed_mode == "single":
            sub = sub[sub["seed"].astype(int) == int(seed)]
        if sub.empty:
            raise ValueError(f"No manifest rows found for condition={condition!r}")
        grouped[condition] = []
        for run in sub.itertuples(index=False):
            path = resolve_run_path(run.attention_json, result_root, pipeline_dir)
            grouped[condition].append({
                "condition": condition,
                "seed": int(run.seed),
                "path": path,
                "records": load_records(path),
            })
    return grouped


def common_proteins(grouped):
    common = None
    for runs in grouped.values():
        cond = None
        for run in runs:
            proteins = set(run["records"])
            cond = proteins if cond is None else cond & proteins
        common = cond if common is None else common & cond
    return sorted(common or [])


def choose_proteins(available, explicit, n, random_seed):
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f"Requested proteins absent from at least one condition: {missing}")
        return list(explicit)
    rng = np.random.default_rng(random_seed)
    if len(available) < n:
        raise ValueError(f"Only {len(available)} common proteins; requested {n}.")
    return sorted(rng.choice(available, size=n, replace=False).tolist())


def averaged_matrix(grouped, condition, protein):
    matrices = []
    sequence = None
    seeds = []
    for run in grouped[condition]:
        record = run["records"].get(protein)
        if record is None:
            continue
        matrix = np.asarray(record["attention_weights"], dtype=float)
        seq = str(record.get("sequence", ""))
        n = len(seq) if seq else matrix.shape[0]
        matrix = matrix[:n, :n]
        if sequence is None:
            sequence = seq if seq else "X" * matrix.shape[0]
        if len(sequence) != matrix.shape[0]:
            raise ValueError(f"Length mismatch for {condition} {protein}.")
        matrices.append(matrix)
        seeds.append(run["seed"])
    if not matrices:
        raise ValueError(f"No matrices for {condition} {protein}")
    if len({m.shape for m in matrices}) != 1:
        raise ValueError(f"Matrix shape mismatch for {condition} {protein}")
    return np.mean(np.stack(matrices, axis=0), axis=0), sequence, seeds


def compute_zmax(matrices, quantile):
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices if np.any(np.isfinite(m))])
    if len(vals) == 0:
        return 1.0
    return float(np.max(vals) if quantile >= 1.0 else np.quantile(vals, quantile))


def compute_diff_absmax(diffs, quantile):
    vals = np.concatenate([np.abs(m[np.isfinite(m)]).ravel() for m in diffs if np.any(np.isfinite(m))])
    if len(vals) == 0:
        return 1.0
    return float(np.max(vals) if quantile >= 1.0 else np.quantile(vals, quantile))


def ticks(sequence, step):
    positions = list(range(0, len(sequence), max(1, step)))
    labels = [f"{i + 1}-{sequence[i]}" for i in positions]
    return positions, labels


def hover_text(matrix, sequence, title, seeds, is_diff=False):
    seeds_text = ",".join(str(s) for s in seeds)
    label = "Difference" if is_diff else "Attention"
    rows = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            row.append(
                f"{title}<br>Seeds: {seeds_text}<br>"
                f"Query: {i + 1}-{sequence[i]}<br>Key: {j + 1}-{sequence[j]}<br>"
                f"{label}: {matrix[i, j]:.6g}"
            )
        rows.append(row)
    return rows


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def load_neq_map(test_csv):
    if not test_csv:
        return {}
    path = Path(test_csv).expanduser()
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    if "name" not in df.columns or "neq" not in df.columns:
        return out
    for _, row in df.iterrows():
        out[str(row["name"])] = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
    return out


def aligned_annotations(protein, n, ss_map, neq_map):
    q3 = np.asarray([""] * n, dtype=object)
    q8 = np.asarray([""] * n, dtype=object)
    neq = np.full(n, np.nan)
    entry = ss_map.get(protein) if ss_map else None
    if entry is not None and len(entry["q3"]) >= n:
        q3 = np.asarray(entry["q3"][:n], dtype=object)
        q8 = np.asarray(entry["q8"][:n], dtype=object)
    if protein in neq_map and len(neq_map[protein]) >= n:
        neq = np.asarray(neq_map[protein][:n], dtype=float)
    return q3, q8, neq


def write_protein_page(protein, matrices, sequence, seeds_by_condition, zmax, diff_absmax, output_html, args):
    subplot_titles = []
    for row_name in ["ESM2 backbone", "ESM2 BiLSTM", "ESM3 BiLSTM", "ESM3 - ESM2 BiLSTM"]:
        for level in FREEZE_LEVELS:
            subplot_titles.append(f"{row_name} {level}")
    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.035,
        vertical_spacing=0.055,
    )
    tick_positions, tick_labels = ticks(sequence, args.tick_step)

    row_specs = [
        ("esm2_backbone", ESM2_BACKBONE, False),
        ("esm2_bilstm", ESM2_BILSTM, False),
        ("esm3_bilstm", ESM3_BILSTM, False),
        ("diff", None, True),
    ]
    for r, (row_name, condition_map, is_diff) in enumerate(row_specs, start=1):
        for c, level in enumerate(FREEZE_LEVELS, start=1):
            if is_diff:
                matrix = matrices[f"diff_{level}"]
                title = f"ESM3 - ESM2 BiLSTM {level}"
                seeds = sorted(set(seeds_by_condition[ESM2_BILSTM[level]]) | set(seeds_by_condition[ESM3_BILSTM[level]]))
                trace = go.Heatmap(
                    z=matrix,
                    text=hover_text(matrix, sequence, title, seeds, is_diff=True),
                    hoverinfo="text",
                    colorscale=args.diff_colorscale,
                    zmin=-diff_absmax,
                    zmax=diff_absmax,
                    coloraxis="coloraxis2",
                )
            else:
                condition = condition_map[level]
                matrix = matrices[condition]
                title = subplot_titles[(r - 1) * 3 + (c - 1)]
                trace = go.Heatmap(
                    z=matrix,
                    text=hover_text(matrix, sequence, title, seeds_by_condition[condition]),
                    hoverinfo="text",
                    colorscale=args.colorscale,
                    zmin=0.0,
                    zmax=zmax,
                    coloraxis="coloraxis",
                )
            fig.add_trace(trace, row=r, col=c)
            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_positions,
                ticktext=tick_labels,
                title_text="Key Residue" if r == 4 else None,
                row=r,
                col=c,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=tick_positions,
                ticktext=tick_labels if c == 1 else None,
                title_text="Query Residue" if c == 1 else None,
                autorange="reversed",
                row=r,
                col=c,
            )
    fig.update_layout(
        title=f"{protein}: seed-{'averaged' if args.seed_mode == 'average' else args.seed} ESM2/ESM3 attention differences",
        width=args.subplot_size * 3 + 260,
        height=args.subplot_size * 4 + 190,
        coloraxis=dict(colorscale=args.colorscale, cmin=0.0, cmax=zmax, colorbar=dict(title="Attention", x=1.01, len=0.44, y=0.76)),
        coloraxis2=dict(colorscale=args.diff_colorscale, cmin=-diff_absmax, cmax=diff_absmax, colorbar=dict(title="ESM3-ESM2", x=1.01, len=0.34, y=0.22)),
        margin=dict(l=70, r=150, t=95, b=70),
    )

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(protein)} ESM2 ESM3 Differential Attention</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .meta{color:#444;font-size:13px;line-height:1.5;}</style>",
        "</head><body>",
        f"<h1>{html.escape(protein)} ESM2/ESM3 Differential Attention</h1>",
        "<div class='meta'>",
        f"Seed mode: {html.escape(args.seed_mode)}<br>",
        f"Attention zmax={zmax:g}; difference +/-{diff_absmax:g}<br>",
        "Rows: ESM2 backbone, ESM2 BiLSTM, ESM3 BiLSTM, ESM3-minus-ESM2 BiLSTM.",
        "</div>",
        fig.to_html(full_html=False, include_plotlyjs=args.include_plotlyjs),
        "</body></html>",
    ]
    Path(output_html).write_text("\n".join(parts))


def write_index(protein_pages, output_html, selected_path):
    links = "\n".join(
        f"<li><a href='{html.escape(Path(page).name)}'>{html.escape(protein)}</a></li>"
        for protein, page in protein_pages
    )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>ESM2 ESM3 Differential Attention Index</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .meta{color:#444;font-size:13px;line-height:1.5;}</style>",
        "</head><body>",
        "<h1>ESM2/ESM3 Differential Attention Index</h1>",
        f"<div class='meta'>Selected proteins file: {html.escape(str(selected_path))}</div>",
        f"<ul>{links}</ul>",
        "</body></html>",
    ]
    Path(output_html).write_text("\n".join(parts))


def main():
    args = parse_args()
    if args.seed_mode == "single" and args.seed is None:
        raise ValueError("--seed is required with --seed_mode single.")
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    manifest_path = resolve_manifest_arg(result_root, args.manifest_tsv)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_root / "analysis_esm2_esm3_differential_attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    grouped = load_grouped(manifest, result_root, pipeline_dir, args.seed_mode, args.seed)
    available = common_proteins(grouped)
    proteins = choose_proteins(available, args.proteins, args.n_proteins, args.random_seed)
    selected_path = output_dir / "selected_proteins.txt"
    selected_path.write_text("\n".join(proteins) + "\n")

    ss_map = load_q3_q8_map(args.ss_csv) if args.ss_csv else {}
    neq_map = load_neq_map(args.test_csv)

    all_attention_mats = []
    all_diff_mats = []
    protein_cache = {}
    residue_rows = []

    for protein in proteins:
        matrices = {}
        seeds_by_condition = {}
        sequence = None
        for condition in required_conditions():
            matrix, seq, seeds = averaged_matrix(grouped, condition, protein)
            matrices[condition] = matrix
            seeds_by_condition[condition] = seeds
            all_attention_mats.append(matrix)
            if sequence is None:
                sequence = seq
            elif sequence != seq:
                raise ValueError(f"Sequence mismatch for {protein}")
        for level in FREEZE_LEVELS:
            diff = matrices[ESM3_BILSTM[level]] - matrices[ESM2_BILSTM[level]]
            matrices[f"diff_{level}"] = diff
            all_diff_mats.append(diff)

            esm2_received = matrices[ESM2_BILSTM[level]].sum(axis=0)
            esm3_received = matrices[ESM3_BILSTM[level]].sum(axis=0)
            diff_received = esm3_received - esm2_received
            q3, q8, neq = aligned_annotations(protein, len(sequence), ss_map, neq_map)
            for i in range(len(sequence)):
                residue_rows.append({
                    "protein": protein,
                    "freeze_level": level,
                    "position": i + 1,
                    "aa": sequence[i],
                    "q3": q3[i],
                    "q8": q8[i],
                    "neq": neq[i],
                    "esm2_received_attention": esm2_received[i],
                    "esm3_received_attention": esm3_received[i],
                    "diff_received_attention": diff_received[i],
                    "abs_diff_received_attention": abs(diff_received[i]),
                    "esm3_minus_esm2_positive": diff_received[i] > 0,
                })
        protein_cache[protein] = (matrices, sequence, seeds_by_condition)

    zmax_global = compute_zmax(all_attention_mats, args.scale_quantile)
    diff_absmax_global = compute_diff_absmax(all_diff_mats, args.diff_quantile)

    protein_pages = []
    for protein, (matrices, sequence, seeds_by_condition) in protein_cache.items():
        if args.scale_scope == "protein":
            zmax = compute_zmax([matrices[c] for c in required_conditions()], args.scale_quantile)
            diff_absmax = compute_diff_absmax([matrices[f"diff_{l}"] for l in FREEZE_LEVELS], args.diff_quantile)
        else:
            zmax = zmax_global
            diff_absmax = diff_absmax_global
        page = output_dir / f"{safe_name(protein)}_esm2_esm3_diff_grid.html"
        write_protein_page(protein, matrices, sequence, seeds_by_condition, zmax, diff_absmax, page, args)
        protein_pages.append((protein, page))

    residue_df = pd.DataFrame(residue_rows)
    residue_df.to_csv(output_dir / "esm2_esm3_differential_received_attention_by_residue.csv", index=False)

    enrich_rows = []
    rng = np.random.default_rng(args.random_seed)
    for (protein, level), sub in residue_df.groupby(["protein", "freeze_level"], dropna=False):
        for direction, col in [("esm3_higher", "diff_received_attention"), ("esm2_higher", "diff_received_attention")]:
            values = sub[col].to_numpy(dtype=float)
            score = values if direction == "esm3_higher" else -values
            for frac in args.top_fracs:
                k = max(1, int(np.ceil(frac * len(sub))))
                selected_idx = np.argsort(-score)[:k]
                selected = sub.iloc[selected_idx]
                control_rows = []
                for _, srow in selected.iterrows():
                    candidates = sub[
                        (sub["q3"] == srow["q3"])
                        & (~sub["position"].isin(selected["position"]))
                    ]
                    if len(candidates) == 0:
                        candidates = sub[~sub["position"].isin(selected["position"])]
                    if len(candidates) > 0:
                        control_rows.append(candidates.iloc[int(rng.integers(0, len(candidates)))])
                control = pd.DataFrame(control_rows)
                row = {
                    "protein": protein,
                    "freeze_level": level,
                    "direction": direction,
                    "top_frac": frac,
                    "n_selected": len(selected),
                    "selected_mean_neq": float(selected["neq"].mean()),
                    "control_mean_neq": float(control["neq"].mean()) if len(control) else np.nan,
                    "selected_loop_C_fraction": float((selected["q3"] == "C").mean()),
                    "control_loop_C_fraction": float((control["q3"] == "C").mean()) if len(control) else np.nan,
                }
                for label in sorted(set(selected["q8"].dropna().astype(str)) | set(control["q8"].dropna().astype(str) if len(control) else [])):
                    if label:
                        row[f"selected_q8_{label}_fraction"] = float((selected["q8"] == label).mean())
                        row[f"control_q8_{label}_fraction"] = float((control["q8"] == label).mean()) if len(control) else np.nan
                enrich_rows.append(row)
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(output_dir / "esm2_esm3_top_differential_residue_enrichment.csv", index=False)
    if not enrich_df.empty:
        summary = (
            enrich_df
            .groupby(["freeze_level", "direction", "top_frac"], dropna=False)
            .agg(
                proteins=("protein", "nunique"),
                selected_mean_neq=("selected_mean_neq", "mean"),
                control_mean_neq=("control_mean_neq", "mean"),
                selected_loop_C_fraction=("selected_loop_C_fraction", "mean"),
                control_loop_C_fraction=("control_loop_C_fraction", "mean"),
            )
            .reset_index()
        )
        summary["delta_mean_neq"] = summary["selected_mean_neq"] - summary["control_mean_neq"]
        summary["delta_loop_C_fraction"] = summary["selected_loop_C_fraction"] - summary["control_loop_C_fraction"]
        summary.to_csv(output_dir / "esm2_esm3_top_differential_residue_summary.csv", index=False)

    index = output_dir / "index.html"
    write_index(protein_pages, index, selected_path)
    summary_txt = [
        f"Manifest: {manifest_path}",
        f"Selected proteins: {', '.join(proteins)}",
        f"Index HTML: {index}",
        f"Per-residue differential table: {output_dir / 'esm2_esm3_differential_received_attention_by_residue.csv'}",
        f"Top differential enrichment: {output_dir / 'esm2_esm3_top_differential_residue_enrichment.csv'}",
        f"Global attention zmax: {zmax_global:g}",
        f"Global difference absmax: {diff_absmax_global:g}",
    ]
    (output_dir / "esm2_esm3_differential_summary.txt").write_text("\n".join(summary_txt) + "\n")
    print("\n".join(summary_txt))


if __name__ == "__main__":
    main()
