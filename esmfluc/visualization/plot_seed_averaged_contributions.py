"""Export an audited seed-mean contribution heatmap and saved signed bands.

This is a read-only consumer of completed inference and band-detection outputs:
it does not run a model, redetect peaks, or modify any source data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

from esmfluc.visualization.visualize_exact_logit_contributions import (
    load_selected_seed,
    resolve_input_path,
    select_manifest_runs,
)


def read_table(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_sequence(path: Path, protein: str) -> str:
    sequences = {}
    name = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in sequences:
                    raise ValueError(f"Duplicate FASTA identifier: {name}")
                sequences[name] = ""
            elif name is None:
                raise ValueError("FASTA sequence precedes its header")
            else:
                sequences[name] += line
    sequence = sequences.get(protein, "")
    if not sequence:
        raise ValueError(f"No sequence for {protein!r} in {path}")
    return sequence


def assert_profile(actual: np.ndarray, saved: object, description: str) -> float:
    saved = np.asarray(saved, dtype=np.float64)
    if saved.shape != actual.shape or not np.isfinite(saved).all():
        raise ValueError(f"Invalid saved profile: {description}")
    error = float(np.max(np.abs(actual - saved)))
    if not np.allclose(actual, saved, atol=1e-10, rtol=1e-8):
        raise ValueError(f"{description} disagrees with contribution matrices: {error}")
    return error


def validate_bands(rows: list[dict], influence: np.ndarray, sequence: str) -> list[dict]:
    """Check saved coordinates, signs, peak values and stability; never redetect."""
    if not rows:
        raise ValueError("No saved bands match the requested protein/condition/split")
    length = len(sequence)
    intervals = []
    for row in rows:
        start = int(row["start_residue_1based"])
        end = int(row["end_residue_1based_inclusive"])
        peak = int(row["apex_residue_1based"])
        sign = int(row["sign"])
        if int(row["protein_length"]) != length:
            raise ValueError("Saved band protein length differs from FASTA")
        if not 1 <= start <= peak <= end <= length or sign not in (-1, 1):
            raise ValueError("Invalid saved band coordinates or sign")
        for column, expected in (
            ("start_index_0based", start - 1),
            ("end_index_0based_inclusive", end - 1),
            ("apex_index_0based", peak - 1),
        ):
            if int(row[column]) != expected:
                raise ValueError(f"Inconsistent 0-/1-based coordinates: {column}")
        if row["apex_amino_acid"] != sequence[peak - 1]:
            raise ValueError("Saved peak amino acid disagrees with FASTA")
        if not np.all(sign * influence[start - 1:end] > 0):
            raise ValueError("Saved interval contains opposite-sign or zero influence")
        if not np.isclose(
            influence[peak - 1], float(row["apex_signed_column_influence"]),
            atol=1e-10, rtol=1e-8,
        ):
            raise ValueError("Saved band peak disagrees with seed-mean influence")
        if row["primary_stable_band"].lower() != "true":
            raise ValueError("Band table contains a non-primary-stable band")
        if int(row["seed_support_count"]) < 2:
            raise ValueError("Saved band has fewer than two supporting seeds")
        if float(row["matching_min_interval_iou"]) != 0.5:
            raise ValueError("Saved matching threshold is not the locked IoU=0.5")
        if float(row["minimum_seed_interval_iou"]) < 0.5 - 1e-12:
            raise ValueError("Saved supporting match falls below IoU=0.5")
        intervals.append((start, end))
    ordered = sorted(intervals)
    if any(a[1] >= b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Saved band intervals overlap")
    return sorted(rows, key=lambda row: int(row["start_residue_1based"]))


def source_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def render(matrix: np.ndarray, influence: np.ndarray, bands: list[dict],
           prefix: Path, title: str, quantile: float) -> dict:
    length = len(influence)
    limit = float(np.quantile(np.abs(matrix), quantile))
    if limit <= 0:
        limit = float(np.max(np.abs(matrix))) or 1.0
    clipped = float(np.mean(np.abs(matrix) > limit))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    figure_width, figure_height = 7.1, 8.9
    profile_height_inches = 1.55
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    # Fixed axes keep both panels' residue coordinates physically aligned.
    # The heatmap's physical height equals its width, giving square cells.
    left, width = 0.15, 0.64
    heat_height = figure_width * width / figure_height
    heat_bottom = 3.43 / figure_height
    heat = fig.add_axes((left, heat_bottom, width, heat_height))
    color_ax = fig.add_axes((0.815, heat_bottom, 0.022, heat_height))
    profile = fig.add_axes((left, 1.56 / figure_height, width,
                            profile_height_inches / figure_height), sharex=heat)
    image = heat.imshow(
        matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
        origin="upper", interpolation="nearest", aspect="auto",
        extent=(0.5, length + 0.5, length + 0.5, 0.5),
    )
    heat.set_ylabel("Query residue $i$")
    heat.tick_params(axis="x", labelbottom=False)
    fig.colorbar(image, cax=color_ax, extend="both" if clipped else "neither").set_label(
        "Mean signed contribution $\\overline{C}_{ij}$"
    )
    x = np.arange(1, length + 1)
    positive, negative = "#b2182b", "#2166ac"
    profile.axhline(0, color="0.65", linewidth=0.7)
    profile.plot(x, influence, color="0.25", linewidth=1.1, zorder=3)
    profile.fill_between(x, influence, 0, where=influence >= 0, color=positive, alpha=0.10)
    profile.fill_between(x, influence, 0, where=influence < 0, color=negative, alpha=0.10)
    for row in bands:
        start = int(row["start_residue_1based"])
        end = int(row["end_residue_1based_inclusive"])
        peak = int(row["apex_residue_1based"])
        color = positive if int(row["sign"]) > 0 else negative
        profile.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.16, linewidth=0)
        profile.scatter(peak, influence[peak - 1], color=color, edgecolors="white",
                        linewidths=0.5, s=22, zorder=4)
    ticks = [1] + list(range(25, length + 1, 25))
    if ticks[-1] != length:
        if len(ticks) > 1 and length - ticks[-1] < 12:
            ticks.pop()
        ticks.append(length)
    heat.set_yticks(ticks)
    profile.set_xticks(ticks)
    profile.set_xlim(0.5, length + 0.5)
    profile.set_xlabel("Key residue $j$ (sequence position)")
    profile.set_ylabel("Signed influence $\\overline{I}_j$")
    profile.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    for label, ax in (("a", heat), ("b", profile)):
        ax.text(-0.20, 1.07, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.suptitle(title, y=0.965, fontsize=12)
    fig.legend(handles=[
        Patch(facecolor=positive, alpha=0.35, label="Positive band: multi-PB support"),
        Patch(facecolor=negative, alpha=0.35, label="Negative band: single-PB support"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=1, frameon=False, fontsize=9)
    fig.text(0.5, 0.024, "Shading: saved seed-stable intervals; dots: saved primary peaks",
             ha="center", fontsize=8, color="0.35")
    if clipped:
        fig.text(0.5, 0.010, f"Heatmap color limits: {quantile * 100:g}th percentile of absolute contributions",
                 ha="center", fontsize=8, color="0.35")
    for extension in ("pdf", "svg"):
        output = Path(str(prefix) + "." + extension)
        fig.savefig(output, format=extension, dpi=600, metadata={"Creator": __name__})
        if extension == "svg":
            if ET.parse(output).getroot().tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError("SVG export does not contain an SVG root element")
        elif not output.read_bytes().startswith(b"%PDF-"):
            raise ValueError("PDF export has an invalid file signature")
    preview = Path(str(prefix) + ".png")
    fig.savefig(preview, dpi=180)
    plt.close(fig)
    return {"color_limit": limit, "color_quantile": quantile,
            "fraction_of_cells_color_clipped": clipped, "preview": str(preview),
            "profile_panel_height_inches": profile_height_inches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--fasta-file", type=Path, required=True)
    parser.add_argument("--profiles-manifest-tsv", type=Path, required=True)
    parser.add_argument("--bands-csv", type=Path, required=True)
    parser.add_argument("--condition", default="esm3_top28_bilstm_attn")
    parser.add_argument("--protein", required=True, help="Exact saved FASTA identifier.")
    parser.add_argument("--split", default="gk_wt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this prefix's generated figure and audit files; source data are never modified")
    parser.add_argument("--title", default=None)
    parser.add_argument("--color-quantile", type=float, default=0.99)
    args = parser.parse_args()
    if sorted(args.seeds) != [1, 2, 3]:
        parser.error("This figure requires the saved three-seed analysis (1 2 3)")
    if not 0 < args.color_quantile <= 1:
        parser.error("--color-quantile must lie in (0, 1]")
    sequence = read_sequence(args.fasta_file, args.protein)
    runs = select_manifest_runs(args.manifest_tsv, args.condition, args.seeds)
    source_paths = [Path(__file__), args.manifest_tsv, args.fasta_file,
                    args.profiles_manifest_tsv, args.bands_csv]
    matrices = []
    seed_checks = []
    profile_rows = read_table(args.profiles_manifest_tsv, "\t")
    documents = []
    for run in runs:
        payload = load_selected_seed(run, [args.protein], atol=1e-5, rtol=1e-5)[args.protein]
        if payload.sequence != sequence:
            raise ValueError(f"Seed {run.seed} sequence differs from FASTA")
        matrices.append(payload.contribution)
        selected = [r for r in profile_rows if r["condition"] == args.condition
                    and r["seed"] == str(run.seed) and r["split"] == args.split]
        if len(selected) != 1:
            raise ValueError(f"Expected one saved profile row for seed {run.seed}")
        path = resolve_input_path(selected[0]["json_gz"], args.profiles_manifest_tsv)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
        if (document["condition"], int(document["seed"]), document["split"]) != (
            args.condition, run.seed, args.split
        ):
            raise ValueError("Saved profile metadata differs from manifest")
        definition = document["seed_averaged_signed_column_influence_definition"]
        if sorted(definition["source_seeds"]) != [1, 2, 3]:
            raise ValueError("Saved mean profile was not calculated over seeds 1, 2, 3")
        records = [p for p in document["proteins"] if p["name"] == args.protein]
        if len(records) != 1 or records[0]["sequence"] != sequence:
            raise ValueError("Saved profile identifier/sequence mismatch")
        record = records[0]
        error = assert_profile(payload.contribution.mean(axis=0), record["signed_column_influence"],
                               f"Seed {run.seed} influence")
        seed_checks.append({"seed": run.seed, "influence_max_abs_error": error,
                            "logit_reconstruction_and_probability_audit": "passed by load_selected_seed"})
        documents.append(record)
        source_paths.extend([run.attention_path, run.contribution_path, path])
    matrix = np.mean(np.stack(matrices), axis=0, dtype=np.float64)
    influence = matrix.mean(axis=0)
    mean_errors = [assert_profile(influence, p["seed_averaged_signed_column_influence"],
                                  "Saved seed-mean influence") for p in documents]
    bands = [r for r in read_table(args.bands_csv) if r["condition"] == args.condition
             and r["protein"] == args.protein and r["split"] == args.split]
    if any(r["seed"] != "average" for r in bands):
        raise ValueError("Band intervals were not detected on the seed-mean profile")
    bands = validate_bands(bands, influence, sequence)
    prefix = args.output_prefix.resolve()
    outputs = [Path(str(prefix) + suffix) for suffix in (
        ".pdf", ".svg", ".png", "_audit.json", "_influence.csv", "_bands.csv",
    )]
    if any(p.exists() for p in outputs) and not args.overwrite:
        raise FileExistsError("Output already exists; choose a new --output-prefix or pass --overwrite")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    display_protein = "Guanylate kinase" if args.protein == "WT" else args.protein
    model, setting = args.condition.replace("_bilstm_attn", "").split("_", 1)
    display_condition = model.upper() + " " + setting.replace("top", "top-")
    plot_info = render(matrix, influence, bands, prefix,
                       args.title or f"{display_protein} | {display_condition} | mean of three seeds",
                       args.color_quantile)
    with Path(str(prefix) + "_influence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_position_1based", "amino_acid", "seed_mean_signed_influence"])
        writer.writerows((j, aa, value) for j, (aa, value) in enumerate(zip(sequence, influence), 1))
    with Path(str(prefix) + "_bands.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bands[0]))
        writer.writeheader()
        writer.writerows(bands)
    audit = {
        "protein": args.protein, "sequence": sequence, "length": len(sequence),
        "condition": args.condition, "split": args.split, "seeds": [run.seed for run in runs],
        "matrix_definition": "elementwise arithmetic mean of the three saved exact C matrices",
        "influence_definition": "mean_query(mean_seed(C_ij)); unsmoothed",
        "seed_checks": seed_checks, "saved_mean_profile_max_abs_error": max(mean_errors),
        "band_count": len(bands),
        "positive_band_count": sum(int(r["sign"]) > 0 for r in bands),
        "negative_band_count": sum(int(r["sign"]) < 0 for r in bands),
        "band_definition": "saved primary seed-stable intervals, >=2/3 seeds at IoU>=0.5; no redetection",
        "coordinate_convention": "input sequence positions, 1-based inclusive; not PDB numbering",
        "mean_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "sources": [source_digest(p) for p in dict.fromkeys(source_paths)],
        "outputs": [str(p) for p in outputs], **plot_info,
    }
    Path(str(prefix) + "_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in (
        "protein", "condition", "length", "band_count", "positive_band_count",
        "negative_band_count", "saved_mean_profile_max_abs_error", "outputs",
    )}, indent=2))


if __name__ == "__main__":
    main()
