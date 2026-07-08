#!/usr/bin/env python3
"""
Plot seed-averaged attention heatmaps for selected proteins across attention sources.

Default layout:
  columns: frozen, top4, top28
  row 1: ESM2 backbone attention
  row 2: ESM2 BiLSTM attention
  row 3: ESM3 BiLSTM attention

The script writes one interactive Plotly HTML page containing one 3x3 heatmap
grid per protein.  All heatmaps share one color scale by default.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyze_attention_row_modes import resolve_existing_path, resolve_manifest_path


GRID = [
    [
        ("esm2_frozen_backbone_attention", "ESM2 backbone frozen"),
        ("esm2_top4_backbone_attention", "ESM2 backbone top4"),
        ("esm2_top28_backbone_attention", "ESM2 backbone top28"),
    ],
    [
        ("esm2_frozen_bilstm_attention", "ESM2 BiLSTM frozen"),
        ("esm2_top4_bilstm_attention", "ESM2 BiLSTM top4"),
        ("esm2_top28_bilstm_attention", "ESM2 BiLSTM top28"),
    ],
    [
        ("esm3_frozen_bilstm_attention", "ESM3 BiLSTM frozen"),
        ("esm3_top4_bilstm_attention", "ESM3 BiLSTM top4"),
        ("esm3_top28_bilstm_attention", "ESM3 BiLSTM top28"),
    ],
]


def parse_args():
    p = argparse.ArgumentParser(description="Plot comparable attention-source grids for random proteins.")
    p.add_argument("--result_root", required=True, help="Result root containing manifest_attention_sources.tsv.")
    p.add_argument("--manifest_tsv", default=None, help="Default: result_root/manifest_attention_sources.tsv.")
    p.add_argument("--output_html", default=None, help="Default: result_root/attention_source_grid_random.html.")
    p.add_argument("--output_dir", default=None, help="Directory for selected_proteins.txt if --output_html is omitted.")
    p.add_argument("--pipeline_dir", default=None, help="Directory for resolving relative manifest paths.")
    p.add_argument("--n_proteins", type=int, default=5)
    p.add_argument("--proteins", nargs="*", default=None, help="Explicit protein IDs. Overrides random sampling.")
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--seed_mode", choices=["average", "single"], default="average")
    p.add_argument("--seed", type=int, default=None, help="Required when --seed_mode single.")
    p.add_argument(
        "--scale_scope",
        choices=["global", "protein"],
        default="global",
        help="Use one scale for all proteins or one shared scale within each protein grid.",
    )
    p.add_argument(
        "--scale_quantile",
        type=float,
        default=0.995,
        help="Upper color limit quantile. Use 1.0 for true max. Default clips top 0.5%%.",
    )
    p.add_argument("--zmin", type=float, default=0.0)
    p.add_argument("--colorscale", default="Viridis")
    p.add_argument("--tick_step", type=int, default=25)
    p.add_argument("--subplot_size", type=int, default=285)
    p.add_argument("--include_plotlyjs", choices=["cdn", "inline"], default="cdn")
    return p.parse_args()


def condition_names():
    return [condition for row in GRID for condition, _ in row]


def load_records(path):
    records = json.loads(Path(path).read_text())
    out = {}
    for record in records:
        if "name" in record and "attention_weights" in record:
            out[str(record["name"])] = record
    return out


def load_condition_records(manifest, result_root, pipeline_dir, required_conditions, seed_mode, seed):
    grouped = {}
    for condition in required_conditions:
        sub = manifest[manifest["condition"] == condition].copy()
        if seed_mode == "single":
            sub = sub[sub["seed"].astype(int) == int(seed)]
        if sub.empty:
            raise ValueError(f"No manifest rows found for condition={condition!r}.")
        grouped[condition] = []
        for run in sub.itertuples(index=False):
            path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
            if not path.exists():
                raise FileNotFoundError(f"Missing attention JSON for {condition}: {path}")
            grouped[condition].append({
                "seed": int(run.seed),
                "path": path,
                "records": load_records(path),
            })
    return grouped


def common_proteins(grouped):
    common = None
    for runs in grouped.values():
        cond_set = None
        for run in runs:
            proteins = set(run["records"].keys())
            cond_set = proteins if cond_set is None else cond_set & proteins
        common = cond_set if common is None else common & cond_set
    return sorted(common or [])


def choose_proteins(available, explicit, n, random_seed):
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f"Requested proteins not present in every condition: {missing}")
        return list(explicit)
    rng = np.random.default_rng(random_seed)
    if len(available) < n:
        raise ValueError(f"Only {len(available)} proteins are common across conditions; requested {n}.")
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
            raise ValueError(f"Length mismatch for {condition} {protein} seed={run['seed']}.")
        matrices.append(matrix)
        seeds.append(run["seed"])
    if not matrices:
        raise ValueError(f"No matrices found for {condition} {protein}.")
    if len({m.shape for m in matrices}) != 1:
        raise ValueError(f"Matrix shape mismatch for {condition} {protein}: {[m.shape for m in matrices]}")
    return np.mean(np.stack(matrices, axis=0), axis=0), sequence, seeds


def compute_zmax(matrices, quantile):
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices if np.any(np.isfinite(m))])
    if len(vals) == 0:
        return 1.0
    if quantile >= 1.0:
        return float(np.max(vals))
    return float(np.quantile(vals, quantile))


def ticks(sequence, step):
    n = len(sequence)
    positions = list(range(0, n, max(1, step)))
    labels = [f"{i + 1}-{sequence[i]}" for i in positions]
    return positions, labels


def hover_text(matrix, sequence, title, seeds):
    seeds_text = ",".join(str(s) for s in seeds)
    text = []
    for i in range(matrix.shape[0]):
        row = []
        for j in range(matrix.shape[1]):
            row.append(
                f"{title}<br>"
                f"Seeds: {seeds_text}<br>"
                f"Query: {i + 1}-{sequence[i]}<br>"
                f"Key: {j + 1}-{sequence[j]}<br>"
                f"Attention: {matrix[i, j]:.6g}"
            )
        text.append(row)
    return text


def figure_for_protein(protein, grouped, scale_scope, global_zmax, args):
    matrices = {}
    sequence = None
    all_mats = []
    seeds_by_condition = {}
    for row in GRID:
        for condition, title in row:
            matrix, seq, seeds = averaged_matrix(grouped, condition, protein)
            matrices[condition] = matrix
            seeds_by_condition[condition] = seeds
            all_mats.append(matrix)
            if sequence is None:
                sequence = seq
            elif sequence != seq:
                raise ValueError(f"Sequence mismatch for {protein} in condition {condition}.")

    zmax = global_zmax if scale_scope == "global" else compute_zmax(all_mats, args.scale_quantile)
    subplot_titles = [title for row in GRID for _, title in row]
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.035,
        vertical_spacing=0.065,
    )
    tick_positions, tick_labels = ticks(sequence, args.tick_step)

    for r, row in enumerate(GRID, start=1):
        for c, (condition, title) in enumerate(row, start=1):
            matrix = matrices[condition]
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    text=hover_text(matrix, sequence, title, seeds_by_condition[condition]),
                    hoverinfo="text",
                    colorscale=args.colorscale,
                    zmin=args.zmin,
                    zmax=zmax,
                    coloraxis="coloraxis",
                ),
                row=r,
                col=c,
            )
            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_positions,
                ticktext=tick_labels,
                title_text="Key Residue" if r == 3 else None,
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
        title=f"{protein}: seed-{'averaged' if args.seed_mode == 'average' else args.seed} attention sources",
        width=args.subplot_size * 3 + 220,
        height=args.subplot_size * 3 + 180,
        coloraxis=dict(
            colorscale=args.colorscale,
            cmin=args.zmin,
            cmax=zmax,
            colorbar=dict(title="Attention", len=0.85),
        ),
        margin=dict(l=70, r=120, t=95, b=70),
    )
    return fig


def write_html_page(figures, proteins, output_html, include_plotlyjs, selected_path, zmax, args):
    include_js = include_plotlyjs == "inline"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Attention Source Grid</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} "
        "h1{font-size:22px;} h2{font-size:18px;margin-top:36px;} "
        ".meta{color:#444;font-size:13px;line-height:1.5;} "
        ".figure{margin-bottom:42px;}</style>",
        "</head><body>",
        "<h1>Attention Source Grid</h1>",
        "<div class='meta'>",
        f"Proteins: {html.escape(', '.join(proteins))}<br>",
        f"Seed mode: {html.escape(args.seed_mode)}<br>",
        f"Scale scope: {html.escape(args.scale_scope)}; zmin={args.zmin:g}; zmax={zmax:g}; quantile={args.scale_quantile:g}<br>",
        f"Selected proteins file: {html.escape(str(selected_path))}",
        "</div>",
    ]
    for idx, (protein, fig) in enumerate(zip(proteins, figures)):
        parts.append(f"<h2>{html.escape(protein)}</h2>")
        parts.append("<div class='figure'>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if idx == 0 and not include_js else include_js)))
        parts.append("</div>")
    parts.append("</body></html>")
    Path(output_html).write_text("\n".join(parts))


def main():
    args = parse_args()
    if args.seed_mode == "single" and args.seed is None:
        raise ValueError("--seed is required when --seed_mode single.")
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv or (result_root / "manifest_attention_sources.tsv"))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else output_dir / "attention_source_grid_random.html"

    manifest = pd.read_csv(manifest_path, sep="\t")
    required_conditions = condition_names()
    grouped = load_condition_records(manifest, result_root, pipeline_dir, required_conditions, args.seed_mode, args.seed)
    available = common_proteins(grouped)
    proteins = choose_proteins(available, args.proteins, args.n_proteins, args.random_seed)

    selected_path = output_html.with_suffix(".selected_proteins.txt")
    selected_path.write_text("\n".join(proteins) + "\n")

    all_mats = []
    if args.scale_scope == "global":
        for protein in proteins:
            for condition in required_conditions:
                matrix, _, _ = averaged_matrix(grouped, condition, protein)
                all_mats.append(matrix)
        global_zmax = compute_zmax(all_mats, args.scale_quantile)
    else:
        global_zmax = np.nan

    figures = [
        figure_for_protein(protein, grouped, args.scale_scope, global_zmax, args)
        for protein in proteins
    ]
    write_html_page(figures, proteins, output_html, args.include_plotlyjs, selected_path, global_zmax, args)

    summary = [
        f"Manifest: {manifest_path}",
        f"Output HTML: {output_html}",
        f"Selected proteins: {', '.join(proteins)}",
        f"Selected protein list: {selected_path}",
        f"Seed mode: {args.seed_mode}",
        f"Scale scope: {args.scale_scope}",
        f"zmax: {global_zmax if np.isfinite(global_zmax) else 'per-protein'}",
        "Layout rows: ESM2 backbone, ESM2 BiLSTM, ESM3 BiLSTM",
        "Layout columns: frozen, top4, top28",
    ]
    summary_path = output_html.with_suffix(".summary.txt")
    summary_path.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
