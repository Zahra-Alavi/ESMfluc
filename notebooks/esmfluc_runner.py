"""Sequence-only ESMfluc inference and display using the deposited model source.

This adapter does not retrain models or replace the study band detector.
It is embedded in ESMfluc.ipynb so the notebook does not need a GitHub token.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib.metadata
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd

MANIFEST_SHA256 = "04dcf87f23d58bfe43cac44e6c20e1be5b3e36148992de459406df89c4e65579"
METADATA_SHA256 = "229c9b1482d990409a41d06ba044ca3832a7c8b2766b811610d49a4b35634454"
SOURCES = (
    "esmfluc/training/models.py", "esmfluc/inference/get_attn.py",
    "esmfluc/analysis/bands/extract_signed_contribution_bands.py",
    "esmfluc/analysis/bands/analyze_signed_band_seed_reproducibility.py",
)
RED, BLUE = "#c95c4b", "#347fa0"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sequence(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) > 1 or (headers and not lines[0].startswith(">")):
        raise ValueError("Provide exactly one FASTA record or one plain sequence.")
    name = headers[0][1:].strip()[:100] if headers else "query"
    sequence = "".join("".join(line.split()) for line in lines if not line.startswith(">")).upper()
    if not 3 <= len(sequence) <= 1024:
        raise ValueError("Provide 3–1,024 residues; sequences are never truncated or chunked.")
    invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWYX")
    if invalid:
        raise ValueError(f"Unsupported sequence characters: {sorted(invalid)}. Use the 20 standard amino acids or X.")
    return name or "query", sequence


def prepare_source(bundle, workspace):
    """Verify the pinned deposit and extract only the necessary source files."""
    bundle, workspace = Path(bundle), Path(workspace)
    if sha256_file(bundle / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("The model manifest differs from this notebook's pinned release.")
    if sha256_file(bundle / "metadata.zip") != METADATA_SHA256:
        raise ValueError("Metadata archive checksum mismatch.")
    manifest = json.loads((bundle / "manifest.json").read_text())
    root = workspace / "source"
    with zipfile.ZipFile(bundle / "metadata.zip") as archive:
        names = list(SOURCES) + ["release/checkpoint_bundle.py"]
        names += [n for n in archive.namelist() if n.startswith("upstream/")]
        for name in names:
            raw = archive.read(name)
            expected = manifest["source_code"]["file_sha256"].get(name)
            if name.startswith("upstream/"):
                _, family, filename = name.split("/", 2)
                expected = manifest["observed_upstream_cache"][family]["files"][filename]
            if expected is None or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Archived source/config checksum mismatch: {name}")
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive filename")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    for parent in {parent for name in SOURCES for parent in (root / name).parents
                   if parent != root and parent.is_relative_to(root)}:
        (parent / "__init__.py").write_text('"""Archived ESMfluc inference package."""\n')
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "release"))
    # A notebook run must not silently use a previously imported local module.
    for name, module in list(sys.modules.items()):
        if name.startswith("esmfluc.") and getattr(module, "__file__", None):
            relative = str(Path(module.__file__).resolve()).replace("\\", "/")
            for source in SOURCES:
                if relative.endswith(source) and sha256_file(Path(module.__file__)) != manifest["source_code"]["file_sha256"][source]:
                    raise ValueError(f"An incompatible module was already imported: {name}")
    return manifest, root


def make_model(training_args, source):
    import torch
    from esmfluc.inference.get_attn import ESM3Wrapper, EsmSequenceTokenizer, ESM3_AVAILABLE
    from esmfluc.training.models import BiLSTMWithSelfAttentionModel, ESMLinearTokenClassifier
    if training_args["esm_model"].startswith("esm3"):
        if not ESM3_AVAILABLE:
            raise RuntimeError("Install esm==3.2.1.post1 in the isolated Python 3.11 environment.")
        from esm.models.esm3 import ESM3
        tokenizer = EsmSequenceTokenizer()
        def unavailable_decoder(*args, **kwargs):
            raise RuntimeError("This notebook supports sequence-only ESMfluc inference, not ESM3 generation.")
        # Sequence-only forward uses only the sequence tokenizer. Constructing
        # the full tokenizer collection would trigger an unnecessary HF download.
        with torch.device("meta"):
            backbone = ESM3Wrapper(ESM3(
                d_model=1536, n_heads=24, v_heads=256, n_layers=48,
                structure_encoder_fn=unavailable_decoder,
                structure_decoder_fn=unavailable_decoder,
                function_decoder_fn=unavailable_decoder,
                tokenizers=SimpleNamespace(sequence=tokenizer)))
    else:
        from transformers import EsmConfig, EsmModel, EsmTokenizer
        folder = source / "upstream/esm2"
        config = EsmConfig.from_pretrained(folder, local_files_only=True)
        tokenizer = EsmTokenizer.from_pretrained(folder, local_files_only=True)
        with torch.device("meta"):
            backbone = EsmModel(config)
    with torch.device("meta"):
        if training_args["architecture"] == "esm_linear":
            model = ESMLinearTokenClassifier(backbone, num_classes=2)
        else:
            model = BiLSTMWithSelfAttentionModel(
                backbone, training_args["hidden_size"], training_args["num_layers"],
                num_classes=2, dropout=training_args["dropout"],
                bidirectional=training_args["bidirectional"])
    return model, tokenizer


def materialize_buffers(model):
    """Initialize deterministic nonpersistent buffers absent from state_dict."""
    import torch
    for module in model.modules():
        if type(module).__name__ == "RotaryEmbedding" and module.inv_freq.is_meta:
            module.device = torch.device("cpu")
            module.reset_parameters()
    for name, buffer in list(model.named_buffers()):
        if buffer.is_meta and name.endswith("embeddings.position_ids"):
            parent, attribute = name.rsplit(".", 1)
            model.get_submodule(parent)._buffers[attribute] = torch.arange(
                buffer.shape[-1], dtype=buffer.dtype).reshape(buffer.shape)
    if any(b.is_meta for _, b in model.named_buffers()):
        raise RuntimeError("Unsupported nonpersistent model buffers; refusing an incomplete model.")


def detect_bands(influence, sequence, condition, seed):
    from esmfluc.analysis.bands.extract_signed_contribution_bands import (
        detect_raw_mad_amplitude_apices, build_apex_rows,
    )
    n = len(sequence)
    calls = detect_raw_mad_amplitude_apices(influence, 0, n, 1, n - 1, 2.0)
    context = dict(condition=condition, seed=seed, split="input", json_path="notebook",
                   influence_field="signed_column_influence", protein="query",
                   protein_length=n, eligible_start=1, eligible_end=n - 1, scale_tolerance=2)
    missing = np.full(n, np.nan)
    rows, _ = build_apex_rows(influence, missing, missing, sequence, calls, context,
                              support_method="half_intensity_merge")
    return rows


def stable_bands(mean_rows, seed_rows):
    from esmfluc.analysis.bands.analyze_signed_band_seed_reproducibility import interval_iou_matches
    if not mean_rows:
        return []
    mean_frame = pd.DataFrame(mean_rows)
    support = {i: [] for i in mean_frame.index}
    for seed, rows in seed_rows.items():
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        for sign in (-1, 1):
            matched = interval_iou_matches(mean_frame[mean_frame.sign == sign], frame[frame.sign == sign], .5)
            for index in matched:
                support[index].append(seed)
    return [dict(row, seed_support_count=len(support[i]), supporting_seeds=support[i])
            for i, row in enumerate(mean_rows) if len(support[i]) >= 2]


def public_bands(rows):
    return [{"sign": int(r["sign"]),
             "class_support": "multi-PB" if r["sign"] > 0 else "single-PB",
             "peak_position": int(r["apex_residue_1based"]),
             "peak_amino_acid": r["apex_amino_acid"],
             "start": int(r["start_residue_1based"]),
             "end": int(r["end_residue_1based_inclusive"]),
             "peak_influence": float(r["apex_signed_column_influence"]),
             "seed_support_count": r.get("seed_support_count"),
             "supporting_seeds": r.get("supporting_seeds", [])} for r in rows]


def mean_predictions(probabilities):
    mean = np.asarray(probabilities, dtype=np.float64).mean(axis=0)
    return mean, (mean[:, 1] > .5).astype(int)


LINKED_VIEWER_JS = r"""
const data = JSON.parse(document.getElementById('viewer-data').textContent);
const plot = document.getElementById('contribution-plot');
const thresholdInput = document.getElementById('threshold');
const sourceInput = document.getElementById('source-position');
const status = document.getElementById('selection-status');
const sequenceButtons = Array.from(document.querySelectorAll('.seq-residue'));
const bandShapes = (plot.layout.shapes || []).slice();
let selectedKey = null;
let viewer = null;

function receiversForKey(matrix, key, threshold) {
  if (!Number.isInteger(key) || key < 1 || key > data.sequence.length ||
      !Number.isFinite(threshold) || threshold < 0) return [];
  return matrix.map((row, i) => ({position: i + 1, value: Number(row[key - 1])}))
    .filter(r => Number.isFinite(r.value) && r.value !== 0 && Math.abs(r.value) >= threshold)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.position - b.position);
}

function baseStructure() {
  if (!viewer) return;
  viewer.removeAllLabels();
  viewer.setStyle({}, {cartoon: {color: '#d4e6ce'}});
  for (const b of data.structure.bands) {
    const color = b.sign > 0 ? '#c95c4b' : '#347fa0';
    const peakColor = b.sign > 0 ? '#650010' : '#00264d';
    viewer.setStyle({serial: b.atoms}, {cartoon: {color: color}});
    if (b.peak !== null) {
      viewer.addStyle({serial: [b.peak]}, {sphere: {radius: 0.5, color: peakColor}});
      viewer.addLabel(b.label, {fontColor: peakColor, backgroundOpacity: 0, fontSize: 13},
                      {serial: [b.peak]});
    }
  }
}

function selectSource(key) {
  if (!data.has_contributions) return;
  const threshold = thresholdInput.value.trim() === '' ? NaN : Number(thresholdInput.value);
  if (!Number.isInteger(key) || key < 1 || key > data.sequence.length) {
    status.textContent = 'Choose a source position within the input sequence.';
    return;
  }
  if (!Number.isFinite(threshold) || threshold < 0) {
    status.textContent = 'Enter a finite, nonnegative contribution threshold.';
    return;
  }
  selectedKey = key;
  sourceInput.value = key;
  // The full, unclipped matrix is used, not the displayed color intensity.
  const receivers = receiversForKey(plot.data[0].z, key, threshold);
  const byPosition = new Map(receivers.map(r => [r.position, r.value]));
  for (const el of sequenceButtons) {
    const position = Number(el.dataset.pos);
    el.classList.toggle('selected-source', position === key);
    el.classList.toggle('selected-receiver', byPosition.has(position));
    el.title = el.dataset.description + (byPosition.has(position)
      ? '; contribution from source ' + key + ': ' + byPosition.get(position).toPrecision(6) : '');
  }
  Plotly.relayout(plot, {shapes: bandShapes.concat([{
    type: 'rect', xref: 'x', yref: 'y domain', x0: key - .5, x1: key + .5,
    y0: 0, y1: 1, fillcolor: 'rgba(0,127,134,0.12)', line: {color: '#007f86', width: 2}
  }])});
  const body = document.getElementById('receiver-rows');
  body.replaceChildren();
  for (const r of receivers) {
    const tr = document.createElement('tr');
    const mapped = data.structure && data.structure.mapping[r.position - 1];
    const pdbLabel = mapped && mapped.resolved
      ? mapped.pdb_chain + ':' + mapped.structure_amino_acid + mapped.pdb_residue + mapped.insertion_code
      : data.structure ? 'unresolved' : 'not supplied';
    for (const value of [data.sequence[r.position - 1] + r.position, r.value.toPrecision(6), pdbLabel]) {
      const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  document.getElementById('receiver-count').textContent = receivers.length;
  let text = 'Source ' + data.sequence[key - 1] + key + ': ' + receivers.length +
    ' receivers with |C(i,j)| ≥ ' + threshold.toPrecision(5) + '.';
  if (viewer) {
    baseStructure();
    const atoms = data.structure.atom_serials;
    const receiverAtoms = receivers.flatMap(r => atoms[String(r.position)] || []);
    if (receiverAtoms.length) viewer.setStyle({serial: receiverAtoms}, {cartoon: {color: '#ffcf4d'}});
    const sourceAtoms = atoms[String(key)] || [];
    if (sourceAtoms.length) viewer.setStyle({serial: sourceAtoms}, {cartoon: {color: '#007f86'}});
    const ca = data.structure.ca_serials[String(key)];
    if (ca !== undefined) viewer.addStyle({serial: [ca]}, {sphere: {radius: .6, color: '#007f86'}});
    viewer.render();
  }
  if (data.structure) {
    const mappedCount = receivers.filter(r => data.structure.mapping[r.position - 1].resolved).length;
    text += ' Structure: ' + mappedCount + '/' + receivers.length + ' receivers have mapped coordinates.';
    if (!data.structure.mapping[key - 1].resolved) text += ' The source has no resolved coordinates.';
  }
  status.textContent = text;
}

if (data.structure && typeof $3Dmol !== 'undefined') {
  try {
    viewer = $3Dmol.createViewer('structure-viewer', {backgroundColor: 'white'});
    viewer.addModel(data.structure.pdb, 'pdb'); baseStructure(); viewer.zoomTo(); viewer.render();
  } catch (error) {
    viewer = null;
    document.getElementById('structure-note').textContent =
      'The 3D viewer could not start: ' + String(error.message || error) +
      '. Sequence selection and numerical results remain available.';
  }
} else if (data.structure) {
  document.getElementById('structure-note').textContent =
    'The 3D viewer requires access to 3Dmol.org. Sequence selection remains available.';
}
if (data.has_contributions) {
  for (const el of sequenceButtons) el.addEventListener('click', () => selectSource(Number(el.dataset.pos)));
  document.getElementById('select-source').addEventListener('click', () => selectSource(Number(sourceInput.value)));
  sourceInput.addEventListener('keydown', e => {if (e.key === 'Enter') selectSource(Number(sourceInput.value));});
  thresholdInput.addEventListener('input', () => {if (selectedKey !== null) selectSource(selectedKey);});
  plot.on('plotly_click', event => {
    if (!event.points || !event.points.length) return;
    const point = event.points[0];
    const trace = plot.data[point.curveNumber];
    if (trace && trace.meta && trace.meta.role === 'source') selectSource(Number(point.x));
  });
  document.getElementById('clear-selection').addEventListener('click', () => {
    selectedKey = null;
    for (const el of sequenceButtons) {
      el.classList.remove('selected-source', 'selected-receiver'); el.title = el.dataset.description;
    }
    document.getElementById('receiver-rows').replaceChildren();
    document.getElementById('receiver-count').textContent = '0';
    Plotly.relayout(plot, {shapes: bandShapes}); baseStructure(); if (viewer) viewer.render();
    status.textContent = 'Selection cleared. Click a heatmap column or a sequence residue to select its source.';
  });
}
"""


def write_linked_report(output, sequence, probabilities, matrix, influence, bands, stable, structure=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(sequence)
    positions = np.arange(1, n + 1)
    label = "Seed-stable bands" if stable else "Single-seed candidate bands (stability not assessed)"
    if matrix is None:
        fig = go.Figure(go.Scatter(x=positions, y=probabilities[:, 1], mode="lines"))
        fig.update_layout(template="plotly_white", title="Multi-PB probability — linear baseline",
                          xaxis_title="Sequence position", yaxis_title="Probability")
        limit = 0.0
    else:
        matrix = np.asarray(matrix)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise ValueError("Viewer contribution matrix must be finite and match the full input sequence")
        limit = max(float(np.quantile(np.abs(matrix), .99)), 1e-12)
        fig = make_subplots(rows=3, cols=1, row_heights=[.60, .25, .15], vertical_spacing=.08,
                        subplot_titles=("Contribution matrix", f"Signed influence — {label}", "Predicted multi-PB probability"))
        fig.add_trace(go.Heatmap(z=matrix, x=positions, y=positions, meta=dict(role="source"),
            zmin=-limit, zmax=limit, colorscale="RdBu_r", colorbar=dict(title="Contribution", len=.5, y=.75),
            hovertemplate="Key %{x} → query %{y}<br>Contribution %{z:.5g}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=positions, y=influence, mode="lines", line=dict(color="#444"),
            meta=dict(role="source"), hovertemplate="Key %{x}<br>Influence %{y:.5g}<extra></extra>", showlegend=False), row=2, col=1)
        for band in bands:
            color = RED if band["sign"] > 0 else BLUE
            fig.add_vrect(x0=band["start"]-.5, x1=band["end"]+.5, fillcolor=color,
                          opacity=.15, line_width=0, row=2, col=1)
            p = band["peak_position"]
            fig.add_trace(go.Scatter(x=[p], y=[influence[p-1]], mode="markers", meta=dict(role="source"),
                marker=dict(color=color, size=7), showlegend=False,
                text=[f"{band['peak_amino_acid']}{p}; {band['class_support']} support"],
                hovertemplate="%{text}<br>Influence %{y:.5g}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=positions, y=probabilities[:, 1], mode="lines",
                                line=dict(color="#666"), showlegend=False), row=3, col=1)
        fig.update_yaxes(autorange="reversed", title="Query residue i", row=1, col=1)
        fig.update_xaxes(title="Key residue j", row=1, col=1)
        fig.update_yaxes(zeroline=True, zerolinecolor="#888", title="Influence", row=2, col=1)
        fig.update_yaxes(range=[0, 1], title="P(multi-PB)", row=3, col=1)
        fig.update_xaxes(title="Sequence position", row=3, col=1)
    fig.update_layout(template="plotly_white", height=1050 if matrix is not None else 400,
                      margin=dict(l=65, r=65, t=55, b=45))
    signs = np.zeros(n, dtype=int)
    for band in bands:
        signs[band["start"]-1:band["end"]] = band["sign"]
    sequence_html = []
    for p, aa in enumerate(sequence, 1):
        cls = " positive-band" if signs[p-1] > 0 else " negative-band" if signs[p-1] < 0 else ""
        description = f"{aa}{p}"
        if structure and not structure["mapping"][p-1]["resolved"]:
            cls += " unresolved"
            description += "; no resolved structure coordinates"
        sequence_html.append(f'<button type="button" class="seq-residue{cls}" data-pos="{p}" '
            f'data-description="{description}" title="{description}" '
            f'{"disabled" if matrix is None else ""}><small>{p}</small>{aa}</button>')
    payload = json.dumps(dict(sequence=sequence, has_contributions=matrix is not None,
                              structure=structure), allow_nan=False).replace("<", "\\u003c")
    controls = f'''<section {'hidden' if matrix is None else ''}>
<label>Source position j <input id="source-position" type="number" min="1" max="{n}" value="1"></label>
<button id="select-source" type="button">Select source</button>
<label>Minimum |contribution| <input id="threshold" type="number" min="0" step="any" value="{limit*.5:.12g}"></label>
<button id="clear-selection" type="button">Clear selection</button>
<p>Click a heatmap column or a sequence residue. Cyan marks the source; gold marks receivers meeting the threshold.
Selection uses individual source columns, not whole bands. The threshold is a display control, not a statistical test.</p>
<p id="selection-status" role="status" aria-live="polite">No source selected.</p>
<details><summary>Receiving residues (<span id="receiver-count">0</span>)</summary>
<div class="receiver-table"><table><thead><tr><th>Input residue</th><th>Signed contribution</th><th>Structure residue</th></tr></thead>
<tbody id="receiver-rows"></tbody></table></div></details></section>'''
    structure_html = '''<aside><div id="structure-viewer"></div><p id="structure-note">
Structure labels use PDB numbering. Only mapped coordinates are highlighted; missing residues remain visible in the sequence.</p></aside>''' if structure else ''
    structure_script = '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>' if structure else ''
    fragment = fig.to_html(full_html=False, include_plotlyjs=True, div_id="contribution-plot", post_script=LINKED_VIEWER_JS)
    caption = ("Red contributions support multi-PB predictions; blue contributions support single-PB predictions. "
               "Band colors in the sequence describe evidence supplied to queries, not the source residue's predicted class. "
               "Colors saturate at the 99th percentile; selection uses full, unclipped contributions. "
               "These are model contributions, not causal biological interactions.") if matrix is not None else (
               "The linear baseline provides probabilities but has no task-specific contribution matrix or bands.")
    page = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESMfluc: sequence and residue contributions</title><style>
body{font:14px Arial,sans-serif;color:#222;margin:16px}button,input{font:inherit}section{margin-bottom:14px}
label{display:inline-block;margin:4px 12px 4px 0}input{width:110px}button{cursor:pointer}
.sequence{display:flex;flex-wrap:wrap;gap:2px;max-height:220px;overflow:auto;padding:5px;border:1px solid #ddd}
.seq-residue{width:27px;border:1px solid transparent;background:#f4f4f4;padding:2px;color:#222}
.seq-residue small{display:block;font:9px Arial,sans-serif}.positive-band{background:#f4d9d4}.negative-band{background:#d7e8f0}
.unresolved{border-bottom:2px dotted #777}.selected-receiver{background:#ffcf4d;color:#111}
.selected-source{background:#007f86!important;color:white!important;border:2px solid #003d41}
.layout{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(300px,1fr);gap:16px}
.plots{grid-column:1;grid-row:1;min-width:0}aside{grid-column:2;grid-row:1}
.sequence-only{grid-template-columns:minmax(0,1fr)}#structure-viewer{height:600px;position:relative;width:100%}
.receiver-table{max-height:230px;overflow:auto}table{border-collapse:collapse}th,td{padding:4px 12px;border-bottom:1px solid #ddd;text-align:left}
@media(max-width:850px){.layout{display:block}#structure-viewer{height:450px}}
</style></head><body>'''+controls+'<div class="sequence">'+''.join(sequence_html)+'</div><p>'+caption+'</p>'+structure_script+(
        '<script id="viewer-data" type="application/json">'+payload+'</script><div class="layout'+('' if structure else ' sequence-only')+'">'+
        structure_html+'<div class="plots">'+fragment+'</div></div></body></html>')
    (Path(output) / "report.html").write_text(page)


def render_report(output, sequence, probabilities, matrix, influence, bands, stable):
    write_linked_report(output, sequence, probabilities, matrix, influence, bands, stable)
    if matrix is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(sequence)
    positions = np.arange(1, n + 1)
    label = "Seed-stable bands" if stable else "Single-seed candidate bands (stability not assessed)"
    limit = max(float(np.quantile(np.abs(matrix), .99)), 1e-12)
    static, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [5, 2, 1]}, constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper",
                            extent=(.5, n+.5, n+.5, .5), rasterized=True)
    static.colorbar(image, ax=axes[0], label="Contribution")
    axes[0].set(xlabel="Key residue j", ylabel="Query residue i")
    axes[1].plot(positions, influence, color="#444", lw=1)
    axes[1].axhline(0, color="#888", lw=.7)
    for band in bands:
        color = RED if band["sign"] > 0 else BLUE
        axes[1].axvspan(band["start"]-.5, band["end"]+.5, color=color, alpha=.15)
        p = band["peak_position"]
        axes[1].scatter(p, influence[p-1], color=color, s=15, zorder=3)
    axes[1].set(ylabel="Signed influence", title=label, xlim=(.5, n+.5))
    axes[2].plot(positions, probabilities[:, 1], color="#666", lw=1)
    axes[2].set(xlabel="Sequence position", ylabel="P(multi-PB)", ylim=(0, 1), xlim=(.5, n+.5))
    for suffix in ("pdf", "svg"):
        static.savefig(output / f"overview.{suffix}", facecolor="white")
    plt.close(static)


def run(bundle, output, fasta, condition, seeds, device):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name, sequence = parse_sequence(Path(fasta).read_text())
    if sorted(seeds) not in ([1], [1, 2, 3]):
        raise ValueError("Choose seed 1 preview or all three seeds 1, 2, 3.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU is available. In Colab choose Runtime → Change runtime type → GPU.")
    torch.set_num_threads(4)
    manifest, source = prepare_source(Path(bundle), output / "runtime")
    from checkpoint_bundle import load_bundle_state, model_entry
    from esmfluc.inference.get_attn import run_model_bilstm_attn
    entries = [model_entry(manifest, condition, seed) for seed in seeds]
    required = {manifest["backbones"][entries[0]["backbone"]]["file"]:
                manifest["backbones"][entries[0]["backbone"]]["sha256"]}
    required.update({entry["file"]: entry["sha256"] for entry in entries})
    for filename, digest in required.items():
        print(f"Checking {filename}", flush=True)
        if sha256_file(Path(bundle) / filename) != digest:
            raise ValueError(f"Weight checksum mismatch: {filename}")
    arrays, per_seed, predictions, margins, biases = {}, {}, [], [], []
    matrices, attentions, profiles = [], [], []
    for entry in entries:
        seed = entry["seed"]
        print(f"Running {condition}, seed {seed}", flush=True)
        model, tokenizer = make_model(entry["training_args"], source)
        # Files were verified above. Strict loading still checks tensor names/shapes.
        state = load_bundle_state(Path(bundle), condition, seed, verify=False)
        model.load_state_dict(state, strict=True, assign=True)
        del state
        materialize_buffers(model)
        model.to(device).eval()
        if hasattr(model, "attention"):
            attention, tokens, _, _, probs, contribution = run_model_bilstm_attn(
                model, tokenizer, sequence, device, return_logit_contributions=True)
            if len(tokens) != len(sequence):
                raise ValueError("Token/residue alignment failed")
            probabilities = probs.numpy()
            matrix = contribution["matrix"]
            influence = matrix.astype(np.float64).mean(axis=0)
            matrices.append(matrix.astype(np.float64))
            attentions.append(attention.astype(np.float64))
            profiles.append(influence)
            margins.append(contribution["margin"])
            biases.append(contribution["bias"])
            per_seed[seed] = detect_bands(influence, sequence, condition, seed)
            arrays.update({f"seed_{seed}_C": matrix, f"seed_{seed}_A": attention,
                           f"seed_{seed}_s": contribution["intrinsic_signed_evidence"],
                           f"seed_{seed}_I": influence})
        else:
            encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                logits, _ = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            probabilities = logits.softmax(-1)[0].cpu().numpy()
        if probabilities.shape != (len(sequence), 2) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid residue probability output")
        predictions.append(probabilities)
        arrays[f"seed_{seed}_probabilities"] = probabilities
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_prob, labels = mean_predictions(predictions)
    matrix = np.mean(matrices, axis=0) if matrices else None
    influence = matrix.mean(axis=0) if matrix is not None else None
    rows = detect_bands(influence, sequence, condition, "mean") if matrix is not None else []
    stable = len(seeds) == 3
    if stable:
        rows = stable_bands(rows, per_seed)
    bands = public_bands(rows)
    table = pd.DataFrame({"position": np.arange(1, len(sequence)+1), "amino_acid": list(sequence),
                          "multi_pb_probability": mean_prob[:, 1], "predicted_class": labels,
                          "class_name": np.where(labels == 1, "multi-PB", "single-PB")})
    arrays["mean_probabilities"] = mean_prob
    if matrix is not None:
        if not np.allclose(matrix.sum(axis=1) + np.mean(biases), np.mean(margins, axis=0), atol=1e-5, rtol=1e-5):
            raise ValueError("Ensemble margin reconstruction failed")
        arrays.update(mean_C=matrix, mean_A=np.mean(attentions, axis=0), mean_I=influence)
        table["mean_logit_margin"] = np.mean(margins, axis=0)
        table["signed_influence"] = influence
        table["influence_seed_sd"] = np.std(profiles, axis=0, ddof=1) if stable else np.nan
    table.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(bands, columns=["sign", "class_support", "peak_position", "peak_amino_acid",
        "start", "end", "peak_influence", "seed_support_count", "supporting_seeds"]).to_csv(output / "bands.csv", index=False)
    np.savez_compressed(output / "contributions.npz", **arrays)
    summary = dict(name=name, sequence=sequence, condition=condition, seeds=seeds, device=device,
        band_mode="seed-stable" if stable and matrix is not None else "candidate" if matrix is not None else "not available",
        bands=bands, n_bands=len(bands), source_commit=manifest["source_code"]["commit"],
        manifest_sha256=MANIFEST_SHA256, weights_sha256=required,
        band_detection=manifest["band_detection"], classes=manifest["prediction"]["classes"],
        environment={p: importlib.metadata.version(p) for p in
                     ["torch", "numpy", "pandas", "scipy", "transformers", "safetensors", "plotly", "biopython"]
                     + (["esm"] if condition.startswith("esm3") else [])},
        notes=["Mean probabilities define ensemble class predictions.",
               "Mean contributions are averaged after inference; mean(A*s) is not generally mean(A)*mean(s).",
               "Bands describe model evidence, not causal residue interactions."])
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_report(output, sequence, mean_prob, matrix, influence, bands, stable)
    archive_results(output)
    print(f"Finished: {len(sequence)} residues, {len(bands)} bands. Results: {output}", flush=True)


def archive_results(output):
    output = Path(output)
    files = [p for p in output.iterdir() if p.is_file() and p.suffix != ".zip"]
    with zipfile.ZipFile(output / "esmfluc_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)


def map_structure(output, pdb_path, chain_id):
    """Map to the first PDB model by sequence alignment, never residue offset."""
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.Align import PairwiseAligner
    from Bio.Data.PDBData import protein_letters_3to1_extended
    output = Path(output)
    summary = json.loads((output / "run.json").read_text())
    sequence, bands = summary["sequence"], summary["bands"]
    model = next(PDBParser(QUIET=True).get_structure("input", str(pdb_path)).get_models())
    available = [c.id for c in model if any("CA" in r for r in c)]
    if not chain_id and len(available) == 1:
        chain_id = available[0]
    if chain_id not in available:
        raise ValueError(f"Choose a protein chain from {available}")
    chain = model[chain_id]
    residues = [r for r in chain if "CA" in r and r.get_resname().upper() in protein_letters_3to1_extended]
    structure_seq = "".join(protein_letters_3to1_extended[r.get_resname().upper()] for r in residues)
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                              open_gap_score=-8, extend_gap_score=-.5)
    alignment = aligner.align(sequence, structure_seq)[0]
    mapping = {}
    for a, b in zip(alignment.aligned[0], alignment.aligned[1]):
        mapping.update(zip(range(int(a[0]), int(a[1])), range(int(b[0]), int(b[1]))))
    comparable = [(i, j) for i, j in mapping.items() if structure_seq[j] != "X"]
    identity = sum(sequence[i] == structure_seq[j] for i, j in comparable) / max(1, len(comparable))
    coverage = len(mapping) / len(sequence)
    if identity < .9 or coverage < .8:
        raise ValueError(f"Structure mapping rejected: identity={identity:.1%}, coverage={coverage:.1%}; require 90%/80%.")
    class ChosenChain(Select):
        def accept_model(self, m): return m.id == model.id
        def accept_chain(self, c): return c.id == chain_id
        def accept_residue(self, r): return r.id in {rr.id for rr in residues}
    writer = PDBIO()
    writer.set_structure(model)
    buffer = io.StringIO()
    writer.save(buffer, select=ChosenChain())
    pdb_text = buffer.getvalue()
    displayed = next(PDBParser(QUIET=True).get_structure("display", io.StringIO(pdb_text)).get_models())[chain_id]
    rows, atom_serials, ca_serials = [], {}, {}
    for i, amino_acid in enumerate(sequence):
        if i not in mapping:
            rows.append(dict(position=i+1, amino_acid=amino_acid, resolved=False, pdb_chain=None,
                             pdb_residue=None, insertion_code=None, structure_amino_acid=None,
                             sequence_match=None))
            continue
        residue = residues[mapping[i]]
        normalized = displayed[residue.id]
        atom_serials[i+1] = [int(a.serial_number) for a in normalized.get_unpacked_list()]
        ca_serials[i+1] = int(normalized["CA"].serial_number)
        rows.append(dict(position=i+1, amino_acid=amino_acid, resolved=True, pdb_chain=chain_id,
            pdb_residue=int(residue.id[1]), insertion_code=residue.id[2].strip(),
            structure_amino_acid=structure_seq[mapping[i]], sequence_match=amino_acid == structure_seq[mapping[i]]))
    selections = []
    for band in bands:
        positions = range(band["start"], band["end"]+1)
        peak = band["peak_position"]
        mapped = rows[peak-1]
        selections.append(dict(sign=band["sign"], atoms=[a for p in positions for a in atom_serials.get(p, [])],
            peak=ca_serials.get(peak), label=(f"{mapped['structure_amino_acid']}{mapped['pdb_residue']}{mapped['insertion_code']}"
                                           if mapped["resolved"] else None)))
    payload = json.dumps(dict(pdb=pdb_text, bands=selections)).replace("<", "\\u003c")
    page = """<!doctype html><html><meta charset="utf-8"><body>
<div id="protein" style="width:100%;height:620px;position:relative"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script><script>
const data=PAYLOAD;const v=$3Dmol.createViewer('protein',{backgroundColor:'white'});
v.addModel(data.pdb,'pdb');v.setStyle({},{cartoon:{color:'#d4e6ce'}});
for(const b of data.bands){const c=b.sign>0?'#c95c4b':'#347fa0';const p=b.sign>0?'#650010':'#00264d';
v.setStyle({serial:b.atoms},{cartoon:{color:c}});
if(b.peak!==null){v.addStyle({serial:[b.peak]},{sphere:{radius:0.5,color:p}});
v.addLabel(b.label,{fontColor:p,backgroundOpacity:0,fontSize:13},{serial:[b.peak]});}}
v.zoomTo();v.render();</script>
<p>Red: multi-PB-supporting bands; blue: single-PB-supporting bands. Dark spheres mark resolved peaks.
Labels use PDB numbering. Unresolved residues are not invented. Rotate to inspect hidden peaks.</p></body></html>""".replace("PAYLOAD", payload)
    (output / "structure.html").write_text(page)
    pd.DataFrame(rows).to_csv(output / "structure_mapping.csv", index=False)
    (output / "structure_mapping.json").write_text(json.dumps(dict(identity=identity, input_coverage=coverage,
        chain=chain_id, structure_sha256=sha256_file(pdb_path), model="first", mapping=rows), indent=2))
    with np.load(output / "contributions.npz", allow_pickle=False) as arrays:
        write_linked_report(output, sequence, arrays["mean_probabilities"],
            arrays["mean_C"] if "mean_C" in arrays else None,
            arrays["mean_I"] if "mean_I" in arrays else None, bands,
            summary.get("band_mode") == "seed-stable",
            structure=dict(pdb=pdb_text, bands=selections, mapping=rows,
                           atom_serials=atom_serials, ca_serials=ca_serials))
    archive_results(output)
    print(f"Mapped chain {chain_id}: {identity:.1%} identity, {coverage:.1%} resolved coverage", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    inference = sub.add_parser("predict")
    inference.add_argument("--bundle", required=True, type=Path)
    inference.add_argument("--output", required=True, type=Path)
    inference.add_argument("--fasta", required=True, type=Path)
    inference.add_argument("--condition", default="esm3_top28_bilstm_attn")
    inference.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    inference.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    structure = sub.add_parser("map")
    structure.add_argument("--output", required=True, type=Path)
    structure.add_argument("--pdb", required=True, type=Path)
    structure.add_argument("--chain", default="")
    validate = sub.add_parser("validate")
    validate.add_argument("--fasta", required=True, type=Path)
    args = p.parse_args()
    if args.command == "predict":
        run(args.bundle, args.output, args.fasta, args.condition, args.seeds, args.device)
    elif args.command == "map":
        map_structure(args.output, args.pdb, args.chain)
    else:
        _, sequence = parse_sequence(args.fasta.read_text())
        print(f"Valid input: {len(sequence)} residues")


if __name__ == "__main__":
    main()
