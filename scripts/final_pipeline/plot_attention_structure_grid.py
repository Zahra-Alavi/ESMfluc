#!/usr/bin/env python3
"""
Plot seed-averaged attention heatmaps next to an interactive protein structure.

The attention grid matches plot_attention_source_grid.py:
  columns: frozen, top4, top28
  row 1: ESM2 backbone attention
  row 2: ESM2 BiLSTM attention
  row 3: ESM3 BiLSTM attention

Each protein gets a standalone HTML page with embedded PDB text when a matching
structure is found or downloaded. Plotly and 3Dmol.js are loaded from CDNs by
default.
"""

import argparse
import ast
import html
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from plot_attention_source_grid import (
    GRID,
    averaged_matrix,
    choose_proteins,
    common_proteins,
    compute_zmax,
    condition_names,
    safe_name,
    ticks,
)


PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb}.pdb"


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot attention-source grids with a linked 3D protein structure viewer."
    )
    p.add_argument("--result_root", required=True, help="Result root containing the attention-source manifest.")
    p.add_argument("--manifest_tsv", default=None, help="Default: result_root/manifest_attention_sources.tsv.")
    p.add_argument("--output_dir", default=None, help="Default: result_root/attention_structure_grids.")
    p.add_argument("--output_html", default=None, help="Index HTML path. Default: output_dir/index.html.")
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
    p.add_argument(
        "--structure_dir",
        default=None,
        help="Directory containing/caching PDB files. Default: result_root/pdb_cache when present, otherwise output_dir/pdb_cache.",
    )
    p.add_argument(
        "--structure_pattern",
        default="{protein}.pdb",
        help="PDB filename pattern. Supports {protein}, {pdb}, and {chain}. Default: {protein}.pdb",
    )
    p.add_argument(
        "--download_pdb",
        choices=["auto", "always", "never"],
        default="auto",
        help="Download PDB files from RCSB when needed. auto downloads missing files; always refreshes; never disables downloads.",
    )
    p.add_argument(
        "--pdb_download_url",
        default=PDB_DOWNLOAD_URL,
        help="Download URL template. Supports {pdb}, {pdb_lower}, {protein}, and {chain}. Default: RCSB PDB endpoint.",
    )
    p.add_argument("--download_timeout", type=float, default=30.0, help="PDB download timeout in seconds.")
    p.add_argument(
        "--position_map_csv",
        default=None,
        help="Optional CSV with columns: protein,seq_pos,pdb_chain,pdb_resi.",
    )
    p.add_argument(
        "--test_csv",
        default=None,
        help="Optional CSV with name, sequence, neq columns. Default: result_root/test_data_with_names.csv if present.",
    )
    p.add_argument(
        "--click_attention_threshold_fraction",
        type=float,
        default=0.50,
        help="On heatmap click, select query residues with attention to the clicked key >= this fraction of the heatmap zmax. Default: 0.50.",
    )
    p.add_argument(
        "--click_attention_threshold_abs",
        type=float,
        default=None,
        help="Absolute attention threshold for click-linked query residue selection. Overrides --click_attention_threshold_fraction.",
    )
    p.add_argument(
        "--click_max_query_residues",
        type=int,
        default=80,
        help="Maximum number of query residues highlighted on structure for a clicked key. Highest attention values are kept. Default: 80.",
    )
    return p.parse_args()


def corrected_results_sibling_candidates(raw, pipeline_dir, leaf_is_file=False):
    parts = raw.parts
    if "results" not in parts or len(parts) < 2:
        return []
    results_index = parts.index("results")
    if leaf_is_file and len(parts) >= 3:
        sibling = Path(*parts[: results_index + 1]) / parts[-2] / parts[-1]
    else:
        sibling = Path(*parts[: results_index + 1]) / parts[-1]
    return [Path.cwd() / sibling, pipeline_dir / sibling, sibling]


def resolve_manifest_arg(result_root, pipeline_dir, manifest_tsv):
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
        candidates.extend(corrected_results_sibling_candidates(raw, pipeline_dir, leaf_is_file=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_result_root_arg(result_root_arg, pipeline_dir):
    raw = Path(result_root_arg).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, pipeline_dir / raw, raw])

    candidates.extend(corrected_results_sibling_candidates(raw, pipeline_dir, leaf_is_file=False))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return candidates[0].resolve()


def resolve_attention_path(path_value, result_root, pipeline_dir):
    path = Path(str(path_value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            result_root / path,
            result_root.parent / path,
            pipeline_dir / path,
            Path.cwd() / path,
            path,
        ])
    parts = path.parts
    if "runs" in parts:
        suffix = Path(*parts[parts.index("runs"):])
        candidates.extend([result_root / suffix, result_root.parent / suffix])
    if len(parts) >= 3:
        candidates.append(result_root / Path(*parts[-3:]))
        candidates.append(result_root.parent / Path(*parts[-3:]))
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].expanduser()


def load_condition_records_robust(manifest, result_root, pipeline_dir, required_conditions, seed_mode, seed):
    grouped = {}
    for condition in required_conditions:
        sub = manifest[manifest["condition"] == condition].copy()
        if seed_mode == "single":
            sub = sub[sub["seed"].astype(int) == int(seed)]
        if sub.empty:
            raise ValueError(f"No manifest rows found for condition={condition!r}.")
        grouped[condition] = []
        for run in sub.itertuples(index=False):
            path = resolve_attention_path(run.attention_json, result_root, pipeline_dir)
            if not path.exists():
                raise FileNotFoundError(f"Missing attention JSON for {condition}: {path}")
            grouped[condition].append({
                "seed": int(run.seed),
                "path": path,
                "records": json.loads(path.read_text()),
            })
            grouped[condition][-1]["records"] = {
                str(record["name"]): record
                for record in grouped[condition][-1]["records"]
                if "name" in record and "attention_weights" in record
            }
    return grouped


def sequence_position_customdata(n):
    return [[(i + 1, j + 1) for j in range(n)] for i in range(n)]


def parse_listlike(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def resolve_test_csv_arg(result_root, pipeline_dir, test_csv):
    if test_csv is not None:
        raw = Path(test_csv).expanduser()
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([Path.cwd() / raw, result_root / raw, result_root.parent / raw, pipeline_dir / raw, raw])
            candidates.extend(corrected_results_sibling_candidates(raw, pipeline_dir, leaf_is_file=True))
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    candidates = [
        result_root / "test_data_with_names.csv",
        result_root.parent / "test_data_with_names.csv",
        pipeline_dir / "data/test_data_with_names.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_real_neq_map(path):
    if path is None:
        return {}, "No test CSV found; real Neq omitted from hover text."
    df = pd.read_csv(path)
    required = {"name", "sequence", "neq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test CSV missing required columns for real Neq: {sorted(missing)}")
    out = {}
    for row in df.itertuples(index=False):
        name = str(row.name)
        neq = np.asarray(parse_listlike(row.neq), dtype=float)
        out[name] = {"sequence": str(row.sequence), "neq": neq}
    return out, f"Real Neq loaded from: {path}"


def neq_for_protein(protein, sequence, neq_map):
    entry = neq_map.get(protein)
    if not entry:
        return None
    neq = np.asarray(entry["neq"], dtype=float)
    if len(neq) < len(sequence):
        return None
    if entry.get("sequence") and str(entry["sequence"])[: len(sequence)] != sequence:
        return None
    return neq[: len(sequence)]


def hover_text(matrix, sequence, title, seeds, real_neq):
    seeds_text = ",".join(str(s) for s in seeds)
    text = []
    for i in range(matrix.shape[0]):
        row = []
        query_neq = real_neq[i] if real_neq is not None and i < len(real_neq) else np.nan
        query_neq_text = f"{query_neq:.6g}" if np.isfinite(query_neq) else "NA"
        for j in range(matrix.shape[1]):
            key_neq = real_neq[j] if real_neq is not None and j < len(real_neq) else np.nan
            key_neq_text = f"{key_neq:.6g}" if np.isfinite(key_neq) else "NA"
            row.append(
                f"{title}<br>"
                f"Seeds: {seeds_text}<br>"
                f"Query: {i + 1}-{sequence[i]}<br>"
                f"Query real Neq: {query_neq_text}<br>"
                f"Key: {j + 1}-{sequence[j]}<br>"
                f"Key real Neq: {key_neq_text}<br>"
                f"Attention: {matrix[i, j]:.6g}"
            )
        text.append(row)
    return text


def figure_for_protein(protein, grouped, scale_scope, global_zmax, real_neq, args):
    matrices = {}
    sequence = None
    all_mats = []
    seeds_by_condition = {}
    for row in GRID:
        for condition, _title in row:
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
    customdata = sequence_position_customdata(len(sequence))

    for r, row in enumerate(GRID, start=1):
        for c, (condition, title) in enumerate(row, start=1):
            matrix = matrices[condition]
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    text=hover_text(matrix, sequence, title, seeds_by_condition[condition], real_neq),
                    hoverinfo="text",
                    customdata=customdata,
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
    return fig, sequence, zmax


def split_protein_id(protein):
    parts = str(protein).rsplit("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return str(protein), ""


def load_position_map(path):
    if path is None:
        return {}, "No position map CSV provided; assuming protein IDs like 3d7a_B, chain from suffix, seq_pos == pdb_resi."
    df = pd.read_csv(path)
    required = {"protein", "seq_pos", "pdb_chain", "pdb_resi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Position map CSV missing required columns: {sorted(missing)}")
    mapping = {}
    for row in df.itertuples(index=False):
        protein = str(row.protein)
        seq_pos = int(row.seq_pos)
        mapping.setdefault(protein, {})[seq_pos] = {
            "chain": "" if pd.isna(row.pdb_chain) else str(row.pdb_chain),
            "resi": str(row.pdb_resi),
        }
    return mapping, f"Using position map CSV: {Path(path).expanduser().resolve()}"


def default_position_map(protein, sequence):
    _pdb_id, chain = split_protein_id(protein)
    return {i: {"chain": chain, "resi": str(i)} for i in range(1, len(sequence) + 1)}


def position_map_for_protein(protein, sequence, explicit_maps):
    if protein in explicit_maps:
        return explicit_maps[protein], "position_map_csv"
    return default_position_map(protein, sequence), "direct_seq_pos_to_pdb_resi"


def structure_candidates(structure_dir, structure_pattern, protein):
    pdb_id, chain = split_protein_id(protein)
    candidates = []
    for protein_value in [protein, pdb_id]:
        try:
            name = structure_pattern.format(protein=protein_value, pdb=pdb_id, chain=chain)
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder in --structure_pattern: {exc}") from exc
        candidates.append(Path(structure_dir) / name)
    out = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def find_structure(structure_dir, structure_pattern, protein):
    candidates = structure_candidates(structure_dir, structure_pattern, protein)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(), candidates
    return None, candidates


def pdb_download_url(url_template, protein):
    pdb_id, chain = split_protein_id(protein)
    return url_template.format(
        pdb=pdb_id.upper(),
        pdb_lower=pdb_id.lower(),
        protein=str(protein),
        chain=chain,
    )


def download_pdb(protein, destination, url_template, timeout):
    url = pdb_download_url(url_template, protein)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ESMfluc-attention-structure-grid/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        pdb_text = response.read().decode("utf-8", errors="replace")
    if "ATOM  " not in pdb_text and "HETATM" not in pdb_text:
        raise ValueError(f"Downloaded file from {url} does not look like a PDB coordinate file.")
    destination.write_text(pdb_text)
    return destination.resolve(), url


def find_or_download_structure(structure_dir, structure_pattern, protein, args):
    candidates = structure_candidates(structure_dir, structure_pattern, protein)
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    if existing is not None and args.download_pdb != "always":
        return existing.resolve(), candidates, "local"
    if args.download_pdb == "never":
        return None, candidates, "missing; downloads disabled"

    destination = candidates[0]
    try:
        path, url = download_pdb(protein, destination, args.pdb_download_url, args.download_timeout)
        return path, candidates, f"downloaded from {url}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        if existing is not None:
            return existing.resolve(), candidates, f"download failed; used cached local file ({exc})"
        return None, candidates, f"download failed ({exc})"


def plotly_script_tag(include_plotlyjs):
    if include_plotlyjs == "cdn":
        return "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
    import plotly.offline.offline as offline

    return f"<script>{offline.get_plotlyjs()}</script>"


def viewer_script(protein, sequence, pdb_text, position_map, has_structure, zmax, args):
    return f"""
<script>
const PDB_TEXT = {json.dumps(pdb_text)};
const SEQUENCE = {json.dumps(sequence)};
const POSITION_MAP = {json.dumps(position_map)};
const HAS_STRUCTURE = {json.dumps(bool(has_structure))};
const HEATMAP_ZMAX = {json.dumps(float(zmax) if np.isfinite(zmax) else None)};
const CLICK_THRESHOLD_ABS = {json.dumps(args.click_attention_threshold_abs)};
const CLICK_THRESHOLD_FRACTION = {json.dumps(args.click_attention_threshold_fraction)};
const CLICK_MAX_QUERY_RESIDUES = {json.dumps(args.click_max_query_residues)};
let viewer = null;
let lockedSelection = false;

function residueSelection(seqPos) {{
  const entry = POSITION_MAP[String(seqPos)] || POSITION_MAP[seqPos];
  if (!entry) return null;
  const selection = {{}};
  if (entry.chain) selection.chain = entry.chain;
  const resiText = String(entry.resi);
  const resiNum = Number(resiText);
  selection.resi = Number.isFinite(resiNum) && String(resiNum) === resiText ? resiNum : resiText;
  return selection;
}}

function baseStyle() {{
  viewer.setStyle({{}}, {{cartoon: {{color: "lightgray", opacity: 0.82}}}});
}}

function addResidueStyle(selection, color, sphereRadius, stickRadius) {{
  if (!selection) return false;
  viewer.addStyle(selection, {{cartoon: {{color: color, opacity: 1.0}}}});
  viewer.addStyle(selection, {{stick: {{color: color, radius: stickRadius}}}});
  viewer.addStyle(selection, {{sphere: {{color: color, radius: sphereRadius, opacity: 0.95}}}});
  return true;
}}

function renderSequenceStrip() {{
  const strip = document.getElementById("sequence-strip");
  if (!strip || strip.children.length) return;
  for (let i = 0; i < SEQUENCE.length; i++) {{
    const span = document.createElement("span");
    span.className = "seq-residue";
    span.dataset.pos = String(i + 1);
    span.title = String(i + 1) + "-" + SEQUENCE[i];
    span.textContent = SEQUENCE[i];
    strip.appendChild(span);
  }}
}}

function highlightSequence(queryPos, keyPos) {{
  document.querySelectorAll(".seq-residue").forEach(function(el) {{
    el.classList.remove("query-highlight", "key-highlight", "attending-highlight");
  }});
  const queryEl = document.querySelector('.seq-residue[data-pos="' + queryPos + '"]');
  const keyEl = document.querySelector('.seq-residue[data-pos="' + keyPos + '"]');
  if (queryEl) queryEl.classList.add("query-highlight");
  if (keyEl) keyEl.classList.add("key-highlight");
}}

function highlightSequenceGroup(keyPos, attendingPositions, clickedQueryPos) {{
  document.querySelectorAll(".seq-residue").forEach(function(el) {{
    el.classList.remove("query-highlight", "key-highlight", "attending-highlight");
  }});
  attendingPositions.forEach(function(pos) {{
    const el = document.querySelector('.seq-residue[data-pos="' + pos + '"]');
    if (el) el.classList.add("attending-highlight");
  }});
  const queryEl = document.querySelector('.seq-residue[data-pos="' + clickedQueryPos + '"]');
  const keyEl = document.querySelector('.seq-residue[data-pos="' + keyPos + '"]');
  if (queryEl) queryEl.classList.add("query-highlight");
  if (keyEl) keyEl.classList.add("key-highlight");
}}

function highlightResidues(queryPos, keyPos) {{
  if (viewer) {{
    baseStyle();
    const querySelection = residueSelection(queryPos);
    const keySelection = residueSelection(keyPos);
    addResidueStyle(querySelection, "orange", 0.42, 0.16);
    addResidueStyle(keySelection, "cyan", 0.72, 0.32);
    viewer.render();
  }}
  highlightSequence(queryPos, keyPos);
  const status = document.getElementById("hover-status");
  if (status) {{
    status.textContent = "Query residue " + queryPos + " highlighted orange; key residue " + keyPos + " highlighted cyan.";
  }}
}}

function clickThreshold() {{
  if (CLICK_THRESHOLD_ABS !== null && CLICK_THRESHOLD_ABS !== undefined) {{
    return Number(CLICK_THRESHOLD_ABS);
  }}
  if (HEATMAP_ZMAX !== null && HEATMAP_ZMAX !== undefined && Number.isFinite(Number(HEATMAP_ZMAX))) {{
    return Number(CLICK_THRESHOLD_FRACTION) * Number(HEATMAP_ZMAX);
  }}
  return 0;
}}

function queryResiduesForClickedKey(point) {{
  const trace = ATTENTION_FIG.data[point.curveNumber];
  if (!trace || !trace.z || !point.customdata || point.customdata.length < 2) return [];
  const keyPos = Number(point.customdata[1]);
  const keyIdx = keyPos - 1;
  const threshold = clickThreshold();
  const rows = [];
  for (let i = 0; i < trace.z.length; i++) {{
    const row = trace.z[i];
    if (!row || keyIdx < 0 || keyIdx >= row.length) continue;
    const value = Number(row[keyIdx]);
    if (Number.isFinite(value) && value >= threshold) {{
      rows.push({{pos: i + 1, value: value}});
    }}
  }}
  rows.sort(function(a, b) {{ return b.value - a.value; }});
  return rows.slice(0, Math.max(1, Number(CLICK_MAX_QUERY_RESIDUES) || rows.length));
}}

function highlightKeyColumn(point) {{
  lockedSelection = true;
  const queryPos = Number(point.customdata[0]);
  const keyPos = Number(point.customdata[1]);
  let attending = queryResiduesForClickedKey(point);
  if (!attending.length) attending = [{{pos: queryPos, value: Number(point.z)}}];
  const attendingPositions = attending.map(function(x) {{ return x.pos; }});
  if (viewer) {{
    baseStyle();
    attendingPositions.forEach(function(pos) {{
      if (pos !== keyPos) {{
        addResidueStyle(residueSelection(pos), "#ffd94a", 0.30, 0.12);
      }}
    }});
    addResidueStyle(residueSelection(queryPos), "orange", 0.44, 0.17);
    addResidueStyle(residueSelection(keyPos), "cyan", 0.82, 0.36);
    viewer.render();
  }}
  highlightSequenceGroup(keyPos, attendingPositions, queryPos);
  const status = document.getElementById("hover-status");
  if (status) {{
    status.textContent = "Clicked key residue " + keyPos + ": highlighted " + attendingPositions.length +
      " query residues with attention >= " + clickThreshold().toPrecision(4) +
      ". Key is cyan; clicked query is orange; other attending residues are yellow.";
  }}
}}

function initStructureViewer() {{
  const container = document.getElementById("structure-viewer");
  if (!HAS_STRUCTURE || !PDB_TEXT) return;
  viewer = $3Dmol.createViewer(container, {{backgroundColor: "white"}});
  viewer.addModel(PDB_TEXT, "pdb");
  baseStyle();
  viewer.zoomTo();
  viewer.render();
}}

function initHoverBridge() {{
  const plot = document.getElementById("attention-plot");
  if (!plot || !plot.on) return;
  plot.on("plotly_hover", function(eventData) {{
    if (lockedSelection) return;
    if (!eventData.points || !eventData.points.length) return;
    const point = eventData.points[0];
    if (!point.customdata || point.customdata.length < 2) return;
    highlightResidues(point.customdata[0], point.customdata[1]);
  }});
  plot.on("plotly_click", function(eventData) {{
    if (!eventData.points || !eventData.points.length) return;
    const point = eventData.points[0];
    if (!point.customdata || point.customdata.length < 2) return;
    highlightKeyColumn(point);
  }});
  plot.on("plotly_doubleclick", function() {{
    lockedSelection = false;
    if (viewer) {{
      baseStyle();
      viewer.render();
    }}
    document.querySelectorAll(".seq-residue").forEach(function(el) {{
      el.classList.remove("query-highlight", "key-highlight", "attending-highlight");
    }});
    const status = document.getElementById("hover-status");
    if (status) status.textContent = "Selection cleared. Hover over a heatmap cell to highlight query and key residues.";
  }});
}}

document.addEventListener("DOMContentLoaded", function() {{
  renderSequenceStrip();
  initStructureViewer();
  initHoverBridge();
}});
</script>
"""


def sequence_strip_html(sequence):
    spans = []
    for i, aa in enumerate(sequence, start=1):
        label = f"{i}-{aa}"
        spans.append(
            "<span class='seq-residue' "
            f"data-pos='{i}' title='{html.escape(label)}'>{html.escape(aa)}</span>"
        )
    return "".join(spans)


def write_protein_page(fig, protein, sequence, output_html, pdb_path, pdb_text, position_map, map_source, zmax, args):
    plot_json = json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder)
    has_structure = bool(pdb_text)
    if args.click_attention_threshold_abs is not None:
        click_threshold_text = f"attention >= {args.click_attention_threshold_abs:g} absolute"
    else:
        click_threshold_text = f"attention >= {args.click_attention_threshold_fraction:g} x heatmap zmax"
    missing_message = ""
    if not has_structure:
        missing_message = (
            "<div class='missing'>No matching PDB structure was found. "
            "The attention grid is still usable, but hover-linked 3D highlighting is disabled.</div>"
        )
    structure_line = str(pdb_path) if pdb_path else "missing"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(protein)} Attention Structure Grid</title>",
        plotly_script_tag(args.include_plotlyjs),
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:0;color:#111;}",
        "header{padding:18px 24px 12px;border-bottom:1px solid #ddd;}",
        "h1{font-size:22px;margin:0 0 8px;}",
        ".meta{color:#444;font-size:13px;line-height:1.45;}",
        ".layout{display:flex;align-items:flex-start;gap:18px;padding:18px 24px 28px;}",
        ".plot-panel{min-width:0;overflow:auto;}",
        ".structure-panel{width:420px;min-width:360px;position:sticky;top:16px;}",
        ".sequence-label{font-weight:bold;margin:0 0 6px;}",
        "#sequence-strip{font-family:Consolas,Menlo,monospace;font-size:12px;line-height:1.75;word-break:break-all;border:1px solid #ccc;border-bottom:0;padding:8px;background:#fafafa;max-height:132px;overflow:auto;}",
        ".seq-residue{display:inline-block;min-width:1ch;padding:0 2px;border-radius:3px;color:#333;}",
        ".seq-residue.query-highlight{background:orange;color:#111;}",
        ".seq-residue.key-highlight{background:cyan;color:#111;}",
        ".seq-residue.attending-highlight{background:#ffe66b;color:#111;}",
        "#structure-viewer{width:100%;height:560px;border:1px solid #ccc;background:#fff;}",
        ".viewer-title{font-weight:bold;margin:0 0 8px;}",
        ".viewer-note,.missing,#hover-status{font-size:13px;line-height:1.45;color:#444;margin-top:8px;}",
        ".missing{padding:12px;border:1px solid #d6a400;background:#fff7d6;color:#4d3a00;}",
        "@media(max-width:1100px){.layout{flex-direction:column}.structure-panel{position:static;width:100%;min-width:0}#structure-viewer{height:480px}}",
        "</style></head><body>",
        "<header>",
        f"<h1>{html.escape(protein)} Attention Structure Grid</h1>",
        "<div class='meta'>",
        f"Seed mode: {html.escape(args.seed_mode)}<br>",
        f"Scale scope: {html.escape(args.scale_scope)}; zmin={args.zmin:g}; zmax={zmax:g}; quantile={args.scale_quantile:g}<br>",
        "Rows: ESM2 backbone, ESM2 BiLSTM, ESM3 BiLSTM<br>",
        "Columns: frozen, top4, top28<br>",
        f"Structure: {html.escape(structure_line)}<br>",
        f"Residue mapping: {html.escape(map_source)}<br>",
        f"Click threshold: {html.escape(click_threshold_text)}",
        "</div></header>",
        "<main class='layout'>",
        "<section class='plot-panel'><div id='attention-plot'></div></section>",
        "<aside class='structure-panel'>",
        "<div class='sequence-label'>Sequence</div>",
        f"<div id='sequence-strip'>{sequence_strip_html(sequence)}</div>",
        "<div class='viewer-title'>3D structure</div>",
        "<div id='structure-viewer'></div>",
        missing_message,
        "<div id='hover-status'>Hover over a heatmap cell to highlight query and key residues. Click a cell to lock the key column and all above-threshold attending query residues. Double-click the heatmap to clear.</div>",
        "<div class='viewer-note'>Hover: key/x-axis is cyan, query/y-axis is orange. Click: key is cyan, clicked query is orange, other above-threshold attending residues are yellow.</div>",
        "</aside></main>",
        f"<script>const ATTENTION_FIG = {plot_json};",
        "Plotly.newPlot('attention-plot', ATTENTION_FIG.data, ATTENTION_FIG.layout, {responsive: true});</script>",
        viewer_script(protein, sequence, pdb_text, position_map, has_structure, zmax, args),
        "</body></html>",
    ]
    Path(output_html).write_text("\n".join(parts))


def write_index_page(protein_pages, output_html, selected_path, summary_path, zmax, args):
    rows = []
    for protein, page, pdb_path, structure_source in protein_pages:
        rel = Path(page).name
        structure = f"structure found ({structure_source})" if pdb_path else f"structure missing ({structure_source})"
        rows.append(f"<li><a href='{html.escape(rel)}'>{html.escape(protein)}</a> - {html.escape(structure)}</li>")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Attention Structure Grid Index</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} "
        "h1{font-size:22px;} .meta{color:#444;font-size:13px;line-height:1.5;} "
        "li{margin:8px 0;}</style>",
        "</head><body>",
        "<h1>Attention Structure Grid Index</h1>",
        "<div class='meta'>",
        f"Seed mode: {html.escape(args.seed_mode)}<br>",
        f"Scale scope: {html.escape(args.scale_scope)}; zmin={args.zmin:g}; zmax={zmax:g}; quantile={args.scale_quantile:g}<br>",
        f"Selected proteins file: {html.escape(str(selected_path))}<br>",
        f"Summary file: {html.escape(str(summary_path))}",
        "</div>",
        "<ul>",
        "\n".join(rows),
        "</ul>",
        "</body></html>",
    ]
    Path(output_html).write_text("\n".join(parts))


def main():
    args = parse_args()
    if args.seed_mode == "single" and args.seed is None:
        raise ValueError("--seed is required when --seed_mode single.")
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    result_root = resolve_result_root_arg(args.result_root, pipeline_dir)
    manifest_path = resolve_manifest_arg(result_root, pipeline_dir, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_root / "attention_structure_grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else output_dir / "index.html"
    selected_path = output_dir / "selected_proteins.txt"
    summary_path = output_dir / "summary.txt"

    manifest = pd.read_csv(manifest_path, sep="\t")
    required_conditions = condition_names()
    grouped = load_condition_records_robust(manifest, result_root, pipeline_dir, required_conditions, args.seed_mode, args.seed)
    available = common_proteins(grouped)
    proteins = choose_proteins(available, args.proteins, args.n_proteins, args.random_seed)
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

    test_csv_path = resolve_test_csv_arg(result_root, pipeline_dir, args.test_csv)
    neq_map, neq_summary = load_real_neq_map(test_csv_path)
    position_maps, mapping_summary = load_position_map(args.position_map_csv)
    if args.structure_dir:
        structure_dir = Path(args.structure_dir).expanduser()
    else:
        result_pdb_cache = result_root / "pdb_cache"
        structure_dir = result_pdb_cache if result_pdb_cache.exists() else output_dir / "pdb_cache"
    structure_dir.mkdir(parents=True, exist_ok=True)
    protein_pages = []
    structure_notes = []
    map_notes = []
    neq_notes = []

    for protein in proteins:
        _, sequence0, _ = averaged_matrix(grouped, required_conditions[0], protein)
        real_neq = neq_for_protein(protein, sequence0, neq_map)
        fig, sequence, protein_zmax = figure_for_protein(protein, grouped, args.scale_scope, global_zmax, real_neq, args)
        zmax = global_zmax if args.scale_scope == "global" else protein_zmax
        pos_map, map_source = position_map_for_protein(protein, sequence, position_maps)
        pdb_path, candidates, structure_source = find_or_download_structure(
            structure_dir, args.structure_pattern, protein, args
        )
        pdb_text = pdb_path.read_text() if pdb_path else ""
        page = output_dir / f"{safe_name(protein)}_attention_structure_grid.html"
        write_protein_page(fig, protein, sequence, page, pdb_path, pdb_text, pos_map, map_source, zmax, args)
        protein_pages.append((protein, page, pdb_path, structure_source))
        structure_notes.append(
            f"  {protein}: {pdb_path if pdb_path else 'missing'}; {structure_source} "
            f"(tried: {', '.join(str(c) for c in candidates)})"
        )
        map_notes.append(f"  {protein}: {map_source}")
        neq_notes.append(f"  {protein}: {'real Neq loaded' if real_neq is not None else 'real Neq unavailable or sequence mismatch'}")

    display_zmax = global_zmax if args.scale_scope == "global" else "per-protein"
    summary = [
        f"Manifest: {manifest_path}",
        f"Output index HTML: {output_html}",
        f"Output dir: {output_dir}",
        f"Selected proteins: {', '.join(proteins)}",
        f"Selected protein list: {selected_path}",
        f"Summary: {summary_path}",
        f"Structure dir: {structure_dir}",
        f"Structure pattern: {args.structure_pattern}",
        f"PDB download mode: {args.download_pdb}",
        f"PDB download URL template: {args.pdb_download_url}",
        f"Real Neq: {neq_summary}",
        f"Position mapping: {mapping_summary}",
        f"Click attention threshold absolute: {args.click_attention_threshold_abs}",
        f"Click attention threshold fraction of zmax: {args.click_attention_threshold_fraction}",
        f"Click maximum highlighted query residues: {args.click_max_query_residues}",
        "Default mapping assumption: without --position_map_csv, protein IDs are assumed to look like 3d7a_B; "
        "the chain is the suffix after the final underscore and seq_pos maps directly to pdb_resi.",
        "Protein pages:",
        *[f"  {protein}: {page}" for protein, page, _pdb_path, _structure_source in protein_pages],
        "Structures:",
        *structure_notes,
        "Residue mapping sources:",
        *map_notes,
        "Real Neq sources:",
        *neq_notes,
        f"Seed mode: {args.seed_mode}",
        f"Scale scope: {args.scale_scope}",
        f"zmax: {display_zmax}",
        "Layout rows: ESM2 backbone, ESM2 BiLSTM, ESM3 BiLSTM",
        "Layout columns: frozen, top4, top28",
        "Hover behavior: query/y-axis residue is orange; key/x-axis residue is larger cyan.",
    ]
    summary_path.write_text("\n".join(summary) + "\n")
    write_index_page(protein_pages, output_html, selected_path, summary_path, display_zmax, args)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
