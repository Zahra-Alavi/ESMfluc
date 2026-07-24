#!/usr/bin/env python3
"""Plot Phase 2 signed-band composition, effects, and inverse selection.

Band apices are compared with all eligible residues in the same protein that
have exact-matching Q3 and lie outside every positive and negative band.  The
script visualizes Q3/Q8 composition, Neq/RSA/torsion distributions, matched
within-protein effects, and the inverse probability that annotated residue
classes are positive apices, negative apices, band flanks, or non-band sites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_ENRICHMENT_DIR = (
    "results/publication_comparable_v2/"
    "analysis_seed_averaged_band_biophysics/enrichment"
)
DEFAULT_OUTPUT_DIR = (
    "results/publication_comparable_v2/"
    "analysis_seed_averaged_band_biophysics/phase2_figures"
)

PARAMETERS_FILE = "biophysical_enrichment_parameters.json"

EXPECTED_CONDITION_ORDER = (
    "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn",
    "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn",
    "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
)
CONDITION_LABELS = {
    "esm2_frozen_bilstm_attn": "ESM2 frozen",
    "esm2_top4_bilstm_attn": "ESM2 top4",
    "esm2_top28_bilstm_attn": "ESM2 top28",
    "esm3_frozen_bilstm_attn": "ESM3 frozen",
    "esm3_top4_bilstm_attn": "ESM3 top4",
    "esm3_top28_bilstm_attn": "ESM3 top28",
}
CONDITION_SHORT = {
    "esm2_frozen_bilstm_attn": "2-Fr",
    "esm2_top4_bilstm_attn": "2-T4",
    "esm2_top28_bilstm_attn": "2-T28",
    "esm3_frozen_bilstm_attn": "3-Fr",
    "esm3_top4_bilstm_attn": "3-T4",
    "esm3_top28_bilstm_attn": "3-T28",
}
CONDITION_COLORS = {
    "esm2_frozen_bilstm_attn": "#7f2704",
    "esm2_top4_bilstm_attn": "#d94801",
    "esm2_top28_bilstm_attn": "#fd8d3c",
    "esm3_frozen_bilstm_attn": "#084594",
    "esm3_top4_bilstm_attn": "#2171b5",
    "esm3_top28_bilstm_attn": "#6baed6",
}

Q8_ORDER = tuple("GHIBETSC")
Q8_COLORS = {
    "G": "#f46d43",
    "H": "#b2182b",
    "I": "#fdae61",
    "B": "#74add1",
    "E": "#2166ac",
    "T": "#762a83",
    "S": "#af8dc3",
    "C": "#bdbdbd",
}

PRIMARY_STRATA = (
    ("flexibility_supporting", "C", "+ · Q3 C", "#b2182b"),
    ("rigidity_supporting", "H", "− · Q3 H", "#2166ac"),
    ("rigidity_supporting", "E", "− · Q3 E", "#053061"),
)

CATEGORY_ORDER = (
    "positive_apex",
    "negative_apex",
    "band_interval_non_apex",
    "non_band",
)
CATEGORY_LABELS = {
    "positive_apex": "+ apex",
    "negative_apex": "− apex",
    "band_interval_non_apex": "Band interval, not apex",
    "non_band": "Outside every band",
}
CATEGORY_COLORS = {
    "positive_apex": "#b2182b",
    "negative_apex": "#2166ac",
    "band_interval_non_apex": "#b2abd2",
    "non_band": "#d9d9d9",
}

FEATURE_BINS = {
    "neq": (
        np.array([-np.inf, 1.0000001, 1.5, 2.5, 4.0, np.inf]),
        ("1.0", "1–1.5", "1.5–2.5", "2.5–4", ">4"),
    ),
    "rsa": (
        np.array([-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf]),
        ("<0.10", "0.10–0.25", "0.25–0.50", "0.50–0.75", ">0.75"),
    ),
    "torsion_change_from_previous": (
        np.array([-np.inf, 5.0, 15.0, 30.0, 60.0, 120.0, 180.0, np.inf]),
        ("<5", "5–15", "15–30", "30–60", "60–120", "120–180", ">180"),
    ),
    "normalized_position": (
        np.array([-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf]),
        ("0–20%", "20–40%", "40–60%", "60–80%", "80–100%"),
    ),
}

ALL_METRIC_ORDER = (
    "q8_is_G", "q8_is_H", "q8_is_I", "q8_is_B", "q8_is_E",
    "q8_is_T", "q8_is_S", "q8_is_C",
    "q8_turn_or_bend_TS", "q8_loop_turn_bend_CTS",
    "structured_linker_loop",
    "neq_peak", "distance_to_neq_peak",
    "q3_boundary_within2", "distance_to_q3_boundary",
    "q8_boundary_within2", "distance_to_q8_boundary",
    "torsion_change_from_previous", "disorder", "disorder_ge05",
)

METRIC_LABELS = {
    "q8_is_G": "Q8 G",
    "q8_is_H": "Q8 H",
    "q8_is_I": "Q8 I",
    "q8_is_B": "Q8 B",
    "q8_is_E": "Q8 E",
    "q8_is_T": "Q8 T",
    "q8_is_S": "Q8 S",
    "q8_is_C": "Q8 C",
    "q8_turn_or_bend_TS": "Q8 T/S",
    "q8_loop_turn_bend_CTS": "Q8 C/T/S",
    "structured_linker_loop": "Structured linker",
    "neq_peak": "Neq peak",
    "distance_to_neq_peak": "Distance to Neq peak",
    "q3_boundary_within2": "Within 2 aa of Q3 boundary",
    "distance_to_q3_boundary": "Distance to Q3 boundary",
    "q8_boundary_within2": "Within 2 aa of Q8 boundary",
    "distance_to_q8_boundary": "Distance to Q8 boundary",
    "torsion_change_from_previous": "Torsional change",
    "disorder": "Disorder score",
    "disorder_ge05": "Disorder >= 0.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrichment_dir", default=DEFAULT_ENRICHMENT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition subset; default uses every available condition.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"),
        default=["png", "pdf"],
    )
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    return args


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def ordered_conditions(values: Iterable[str], requested: list[str] | None) -> list[str]:
    available = set(map(str, values))
    if requested:
        missing = set(requested) - available
        if missing:
            raise ValueError(f"Requested conditions are unavailable: {sorted(missing)}")
        return list(dict.fromkeys(requested))
    ordered = [value for value in EXPECTED_CONDITION_ORDER if value in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def load_annotation_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Path]]:
    """Load the band calls and residue table needed for Q3-only comparisons."""
    source = Path(args.enrichment_dir).expanduser().resolve()
    parameters_path = source / PARAMETERS_FILE
    if not parameters_path.exists():
        raise FileNotFoundError(f"Missing analysis parameters: {parameters_path}")
    parameters = json.loads(parameters_path.read_text())
    paths = {
        "parameters": parameters_path,
        "bands": Path(parameters["annotated_bands_csv"]).expanduser().resolve(),
        "residue": Path(parameters["residue_annotations_csv"]).expanduser().resolve(),
        "protein_summary": Path(parameters["protein_summary_csv"]).expanduser().resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing band annotation inputs: {missing}")
    bands = pd.read_csv(paths["bands"])
    residue = pd.read_csv(paths["residue"])
    protein_summary = pd.read_csv(paths["protein_summary"])
    require_columns(bands, {
        "condition", "split", "protein", "sign", "apex_index_0based",
        "start_index_0based", "end_index_0based_inclusive",
    }, paths["bands"].name)
    require_columns(residue, {
        "split", "protein", "residue_index_0based", "q3", "q8", "neq", "rsa",
        "torsion_change_from_previous", "structured_linker_loop",
        "q3_boundary_within2", "q8_boundary_within2", "neq_peak", "disorder",
    }, paths["residue"].name)
    require_columns(protein_summary, {
        "condition", "split", "protein", "eligible_start_index_0based",
        "eligible_end_index_0based_exclusive",
    }, paths["protein_summary"].name)
    bands = bands[bands["split"].eq(args.split)].copy()
    residue = residue[residue["split"].eq(args.split)].copy()
    protein_summary = protein_summary[protein_summary["split"].eq(args.split)].copy()
    conditions = ordered_conditions(bands["condition"].unique(), args.conditions)
    bands = bands[bands["condition"].isin(conditions)].copy()
    protein_summary = protein_summary[protein_summary["condition"].isin(conditions)].copy()
    return bands, residue, protein_summary, conditions, paths


def build_residue_categories(
    bands: pd.DataFrame,
    residue: pd.DataFrame,
    protein_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assign every eligible condition-residue to an exhaustive band category."""
    residue_lookup = {
        protein: group.sort_values("residue_index_0based").reset_index(drop=True)
        for protein, group in residue.groupby("protein", sort=False)
    }
    band_lookup = {
        key: group for key, group in bands.groupby(["condition", "protein"], sort=False)
    }
    frames = []
    for context in protein_summary.itertuples(index=False):
        condition = str(context.condition)
        protein = str(context.protein)
        annotation = residue_lookup[protein]
        length = len(annotation)
        start = int(context.eligible_start_index_0based)
        end = int(context.eligible_end_index_0based_exclusive)
        protein_bands = band_lookup.get((condition, protein), bands.iloc[0:0])
        positive_apex = np.zeros(length, dtype=bool)
        negative_apex = np.zeros(length, dtype=bool)
        covered = np.zeros(length, dtype=bool)
        for band in protein_bands.itertuples(index=False):
            apex = int(band.apex_index_0based)
            if int(band.sign) > 0:
                positive_apex[apex] = True
            else:
                negative_apex[apex] = True
            covered[
                int(band.start_index_0based):
                int(band.end_index_0based_inclusive) + 1
            ] = True
        if np.any(positive_apex & negative_apex):
            raise ValueError(f"{condition}/{protein}: a residue is both a + and - apex")
        indices = np.arange(start, end)
        category = np.full(len(indices), "non_band", dtype=object)
        category[covered[indices]] = "band_interval_non_apex"
        category[negative_apex[indices]] = "negative_apex"
        category[positive_apex[indices]] = "positive_apex"
        selected = annotation.iloc[indices].copy()
        selected.insert(0, "condition", condition)
        selected["category"] = category
        selected["normalized_position"] = (
            (indices - start) / max(1, end - start - 1)
        )
        frames.append(selected)
    categorized = pd.concat(frames, ignore_index=True)
    if categorized["category"].isna().any():
        raise ValueError("At least one eligible residue lacks a band category")
    return categorized


def build_q3_only_comparisons(
    categorized: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare apices with all same-protein/Q3 residues outside every band."""
    q8_rows = []
    histogram_rows = []
    effect_rows = []
    effect_metrics = (
        "q8_is_S", "q8_is_H", "q8_is_G", "q8_is_E",
        "structured_linker_loop", "q3_boundary_within2",
        "neq", "rsa", "normalized_position", "torsion_change_from_previous",
    )
    for (condition, protein), group in categorized.groupby(
        ["condition", "protein"], sort=False
    ):
        for label, q3, display, _color in PRIMARY_STRATA:
            apex_category = (
                "positive_apex" if label == "flexibility_supporting"
                else "negative_apex"
            )
            cases = group[group["category"].eq(apex_category) & group["q3"].eq(q3)]
            controls = group[group["category"].eq("non_band") & group["q3"].eq(q3)]
            if cases.empty:
                continue
            base = {
                "condition": condition,
                "protein": protein,
                "label": label,
                "q3": q3,
                "stratum": display,
                "n_cases": len(cases),
                "n_controls": len(controls),
                "has_controls": not controls.empty,
            }
            if controls.empty:
                effect_rows.append({**base, "metric": "__availability__", "difference": np.nan})
                continue
            for role, frame in (("apex", cases), ("control", controls)):
                q8_fraction = frame["q8"].value_counts(normalize=True)
                for q8 in Q8_ORDER:
                    q8_rows.append({
                        **base, "role": role, "q8": q8,
                        "fraction": float(q8_fraction.get(q8, 0.0)),
                    })
                for feature, (bins, labels) in FEATURE_BINS.items():
                    values = pd.to_numeric(frame[feature], errors="coerce").dropna()
                    counts = pd.cut(
                        values, bins=bins, labels=labels, include_lowest=True,
                        ordered=True,
                    ).value_counts(normalize=True, sort=False)
                    for bin_label in labels:
                        histogram_rows.append({
                            **base, "role": role, "feature": feature,
                            "bin": bin_label,
                            "fraction": float(counts.get(bin_label, 0.0)),
                        })
            metric_values = {
                "q8_is_S": (cases["q8"].eq("S"), controls["q8"].eq("S")),
                "q8_is_H": (cases["q8"].eq("H"), controls["q8"].eq("H")),
                "q8_is_G": (cases["q8"].eq("G"), controls["q8"].eq("G")),
                "q8_is_E": (cases["q8"].eq("E"), controls["q8"].eq("E")),
            }
            for metric in effect_metrics:
                if metric in metric_values:
                    case_values, control_values = metric_values[metric]
                else:
                    case_values, control_values = cases[metric], controls[metric]
                case_numeric = pd.to_numeric(case_values, errors="coerce").dropna()
                control_numeric = pd.to_numeric(control_values, errors="coerce").dropna()
                difference = (
                    float(case_numeric.mean() - control_numeric.mean())
                    if len(case_numeric) and len(control_numeric) else np.nan
                )
                effect_rows.append({**base, "metric": metric, "difference": difference})
    return pd.DataFrame(q8_rows), pd.DataFrame(histogram_rows), pd.DataFrame(effect_rows)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[str]:
    outputs = []
    for extension in dict.fromkeys(formats):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def stratum_frame(
    frame: pd.DataFrame,
    label: str,
    q3: str,
    scheme: str | None = None,
) -> pd.DataFrame:
    selected = frame[frame["label"].eq(label) & frame["q3"].eq(q3)]
    if scheme is not None:
        selected = selected[selected["match_scheme"].eq(scheme)]
    return selected


def panel_letter(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.08, 1.06, letter, transform=axis.transAxes, fontsize=14,
        fontweight="bold", va="top",
    )


def draw_q8_composition(
    axis: plt.Axes,
    summary: pd.DataFrame,
    scheme: str,
) -> None:
    q8_metrics = [f"q8_is_{label}" for label in Q8_ORDER]
    rows: list[tuple[str, str, str]] = []
    for label, q3, display, _color in PRIMARY_STRATA:
        rows.extend(((label, q3, f"{display} · apex"), (label, q3, f"{display} · control")))
    y_positions = np.array([5.2, 4.4, 3.0, 2.2, 0.8, 0.0])
    for y, (label, q3, row_label) in zip(y_positions, rows):
        group = stratum_frame(summary, label, q3, scheme)
        values = []
        is_case = row_label.endswith("apex")
        column = (
            "case_macro_protein_mean" if is_case
            else "matched_control_macro_protein_mean"
        )
        for metric in q8_metrics:
            metric_values = group.loc[group["metric"].eq(metric), column]
            values.append(float(metric_values.mean()) if len(metric_values) else 0.0)
        total = sum(values)
        if total > 0:
            values = [value / total for value in values]
        left = 0.0
        for q8, value in zip(Q8_ORDER, values):
            axis.barh(
                y, 100 * value, left=100 * left, height=0.62,
                color=Q8_COLORS[q8], edgecolor="white", linewidth=0.4,
            )
            left += value
    axis.set_yticks(y_positions)
    axis.set_yticklabels([row[2] for row in rows])
    axis.set_xlim(0, 100)
    axis.set_xlabel("Q8 composition (%)")
    axis.set_title(f"Matched Q8 composition ({FULL_SCHEME_LABELS[scheme]})")
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=Q8_COLORS[label], label=label)
        for label in Q8_ORDER
    ]
    axis.legend(
        handles=handles, title="Q8", ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, -0.34), frameon=False, columnspacing=0.9,
        handlelength=1.0,
    )


def draw_binary_forest(
    axis: plt.Axes,
    summary: pd.DataFrame,
    conditions: list[str],
    scheme: str,
    q_threshold: float,
) -> None:
    definitions = (
        ("flexibility_supporting", "C", "q8_is_S", "Flex C · Q8 S"),
        ("flexibility_supporting", "C", "structured_linker_loop", "Flex C · structured linker"),
        ("flexibility_supporting", "C", "q3_boundary_within2", "Flex C · near Q3 boundary"),
        ("rigidity_supporting", "H", "q8_is_H", "Rigid H · Q8 H"),
        ("rigidity_supporting", "H", "q8_is_G", "Rigid H · Q8 G"),
        ("rigidity_supporting", "H", "q3_boundary_within2", "Rigid H · near Q3 boundary"),
        ("rigidity_supporting", "E", "q8_is_E", "Rigid E · Q8 E"),
        ("rigidity_supporting", "E", "q3_boundary_within2", "Rigid E · near Q3 boundary"),
    )
    y_base = np.arange(len(definitions))[::-1].astype(float)
    offsets = np.linspace(-0.22, 0.22, len(conditions))
    all_limits = []
    for condition_index, condition in enumerate(conditions):
        for row_index, (label, q3, metric, _display) in enumerate(definitions):
            row = summary[
                summary["match_scheme"].eq(scheme)
                & summary["condition"].eq(condition)
                & summary["label"].eq(label)
                & summary["q3"].eq(q3)
                & summary["metric"].eq(metric)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            estimate = 100 * float(row.case_minus_control_macro_mean)
            low = 100 * float(row.case_minus_control_ci95_low)
            high = 100 * float(row.case_minus_control_ci95_high)
            significant = float(row.p_two_sided_q_bh) < q_threshold
            color = CONDITION_COLORS.get(condition, plt.cm.tab10(condition_index % 10))
            axis.errorbar(
                estimate, y_base[row_index] + offsets[condition_index],
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt="o", markersize=4.3, color=color, ecolor=color,
                elinewidth=0.8, capsize=1.5,
                markerfacecolor=color if significant else "white",
                markeredgewidth=0.9, zorder=3,
            )
            all_limits.extend((low, high))
    axis.axvline(0, color="#333333", linewidth=0.9)
    axis.set_yticks(y_base)
    axis.set_yticklabels([definition[3] for definition in definitions])
    axis.set_xlabel("Apex − matched control (percentage points)")
    axis.set_title("Strict matched binary effects; filled markers: BH q < 0.05")
    if all_limits:
        left, right = min(all_limits), max(all_limits)
        margin = max(3.0, 0.08 * (right - left))
        axis.set_xlim(left - margin, right + margin)
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            color=CONDITION_COLORS.get(condition, plt.cm.tab10(index % 10)),
            label=CONDITION_LABELS.get(condition, condition),
        )
        for index, condition in enumerate(conditions)
    ]
    axis.legend(
        handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(0, -0.38),
        frameon=False, columnspacing=0.9,
    )


def make_matching_diagnostics(
    coverage: pd.DataFrame,
    balance: pd.DataFrame,
    args: argparse.Namespace,
) -> plt.Figure:
    figure, (coverage_axis, balance_axis) = plt.subplots(
        1, 2, figsize=(18, 9), constrained_layout=True,
        gridspec_kw={"width_ratios": (0.9, 1.35)},
    )
    x = np.arange(len(SCHEMES))
    width = 0.23
    for stratum_index, (label, q3, display, color) in enumerate(PRIMARY_STRATA):
        values = []
        for scheme in SCHEMES:
            group = coverage[
                coverage["match_scheme"].eq(scheme)
                & coverage["label"].eq(label)
                & coverage["q3"].eq(q3)
            ]
            rate = group["n_matched_cases"].sum() / group["n_cases"].sum()
            values.append(100 * rate)
        positions = x + (stratum_index - 1) * width
        bars = coverage_axis.bar(
            positions, values, width=width, color=color, label=display,
            edgecolor="white",
        )
        coverage_axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    coverage_axis.set_xticks(x)
    coverage_axis.set_xticklabels([SCHEME_LABELS[scheme] for scheme in SCHEMES])
    coverage_axis.set_ylabel("Matched cases (%)")
    coverage_axis.set_ylim(0, 105)
    coverage_axis.set_title("Matching coverage")
    coverage_axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    coverage_axis.legend(frameon=False, loc="lower left")

    balance_rows = []
    for label, q3, display, color in PRIMARY_STRATA:
        for scheme in SCHEMES:
            group = balance[
                balance["match_scheme"].eq(scheme)
                & balance["label"].eq(label)
                & balance["q3"].eq(q3)
                & balance["matched_on_covariate"].astype(bool)
            ]
            for covariate, covariate_group in group.groupby("covariate", sort=False):
                values = covariate_group["standardized_mean_difference"].abs()
                balance_rows.append({
                    "display": display,
                    "color": color,
                    "scheme": scheme,
                    "covariate": str(covariate),
                    "median": float(values.median()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                })
    balance_table = pd.DataFrame(balance_rows)
    balance_table["row_label"] = (
        balance_table["display"] + " · "
        + balance_table["scheme"].map(FULL_SCHEME_LABELS) + " · "
        + balance_table["covariate"].replace({"normalized_position": "position"})
    )
    balance_table = balance_table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(balance_table))
    for index, row in balance_table.iterrows():
        balance_axis.plot(
            [row.minimum, row.maximum], [index, index], color=row.color,
            linewidth=1.0, alpha=0.75,
        )
        balance_axis.scatter(
            row["median"], index, color=row.color, s=25, zorder=3,
        )
    balance_axis.axvline(0.1, color="#555555", linestyle="--", linewidth=1)
    balance_axis.set_yticks(y)
    balance_axis.set_yticklabels(balance_table["row_label"])
    balance_axis.set_xlabel("Absolute standardized mean difference")
    balance_axis.set_title("Post-match covariate balance")
    balance_axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    balance_axis.text(
        0.99, 0.01, "Point: condition median; line: condition range",
        transform=balance_axis.transAxes, ha="right", va="bottom", fontsize=7.5,
    )
    panel_letter(coverage_axis, "A")
    panel_letter(balance_axis, "B")
    figure.suptitle(
        f"Phase 2 matching diagnostics — {args.split} split",
        fontsize=15, fontweight="bold",
    )
    return figure


def draw_violin_panel(
    axis: plt.Axes,
    protein: pd.DataFrame,
    conditions: list[str],
    label: str,
    q3: str,
    metric: str,
    title: str,
    multiplier: float,
    ylabel: str,
) -> None:
    data = []
    kept_conditions = []
    for condition in conditions:
        values = protein[
            protein["condition"].eq(condition)
            & protein["label"].eq(label)
            & protein["q3"].eq(q3)
            & protein["metric"].eq(metric)
        ]["case_minus_control"].dropna().to_numpy(dtype=float) * multiplier
        if len(values):
            data.append(values)
            kept_conditions.append(condition)
    positions = np.arange(1, len(data) + 1)
    if not data:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    violin = axis.violinplot(
        data, positions=positions, widths=0.8, showmeans=False,
        showmedians=True, showextrema=False,
    )
    for body, condition in zip(violin["bodies"], kept_conditions):
        body.set_facecolor(CONDITION_COLORS.get(condition, "#777777"))
        body.set_edgecolor("white")
        body.set_alpha(0.75)
    violin["cmedians"].set_color("#111111")
    violin["cmedians"].set_linewidth(1.2)
    for position, values in zip(positions, data):
        q1, q3_value = np.quantile(values, [0.25, 0.75])
        axis.plot([position, position], [q1, q3_value], color="#222222", linewidth=2)
        axis.scatter(position, np.mean(values), color="white", edgecolor="#111111", s=18, zorder=4)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels([CONDITION_SHORT.get(c, c) for c in kept_conditions], rotation=30)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)


def make_per_protein_figure(
    protein: pd.DataFrame,
    conditions: list[str],
    args: argparse.Namespace,
) -> plt.Figure:
    strict = protein[protein["match_scheme"].eq(args.match_scheme)]
    definitions = (
        ("flexibility_supporting", "C", "structured_linker_loop", "Flex C: structured linker", 100.0, "Apex − control (pp)"),
        ("flexibility_supporting", "C", "torsion_change_from_previous", "Flex C: torsional change", 1.0, "Apex − control (degrees)"),
        ("rigidity_supporting", "H", "q3_boundary_within2", "Rigid H: near Q3 boundary", 100.0, "Apex − control (pp)"),
        ("rigidity_supporting", "H", "torsion_change_from_previous", "Rigid H: torsional change", 1.0, "Apex − control (degrees)"),
        ("rigidity_supporting", "E", "q3_boundary_within2", "Rigid E: near Q3 boundary", 100.0, "Apex − control (pp)"),
        ("rigidity_supporting", "E", "torsion_change_from_previous", "Rigid E: torsional change", 1.0, "Apex − control (degrees)"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    for axis, definition, letter in zip(axes.flat, definitions, "ABCDEF"):
        draw_violin_panel(axis, strict, conditions, *definition)
        panel_letter(axis, letter)
    figure.suptitle(
        f"Per-protein strict matched effects — {args.split} split\n"
        "Violin: distribution; black bar: IQR; white point: mean; black line: median",
        fontsize=15, fontweight="bold",
    )
    return figure


def make_effect_heatmap(
    summary: pd.DataFrame,
    protein: pd.DataFrame,
    conditions: list[str],
    args: argparse.Namespace,
) -> plt.Figure:
    strict = protein[protein["match_scheme"].eq(args.match_scheme)].copy()
    group_columns = ["condition", "label", "q3", "metric"]
    standardized = strict.groupby(group_columns, as_index=False).agg(
        mean_difference=("case_minus_control", "mean"),
        difference_sd=("case_minus_control", "std"),
    )
    standardized["paired_standardized_effect"] = (
        standardized["mean_difference"] / standardized["difference_sd"]
    )
    strict_summary = summary[summary["match_scheme"].eq(args.match_scheme)]
    matrix = np.full((len(ALL_METRIC_ORDER), len(PRIMARY_STRATA)), np.nan)
    significance = np.zeros_like(matrix, dtype=int)
    denominators = np.zeros_like(matrix, dtype=int)
    for column_index, (label, q3, _display, _color) in enumerate(PRIMARY_STRATA):
        for row_index, metric in enumerate(ALL_METRIC_ORDER):
            values = standardized[
                standardized["label"].eq(label)
                & standardized["q3"].eq(q3)
                & standardized["metric"].eq(metric)
            ]["paired_standardized_effect"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                matrix[row_index, column_index] = float(values.mean())
            tests = strict_summary[
                strict_summary["label"].eq(label)
                & strict_summary["q3"].eq(q3)
                & strict_summary["metric"].eq(metric)
            ]["p_two_sided_q_bh"].dropna()
            significance[row_index, column_index] = int(np.sum(tests < args.q_threshold))
            denominators[row_index, column_index] = len(tests)
    finite = np.abs(matrix[np.isfinite(matrix)])
    color_limit = max(0.5, float(np.quantile(finite, 0.95))) if len(finite) else 1.0
    figure, axis = plt.subplots(
        figsize=(10, 13), constrained_layout=True,
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = axis.imshow(
        matrix, aspect="auto", cmap=cmap, vmin=-color_limit, vmax=color_limit,
    )
    axis.set_xticks(np.arange(len(PRIMARY_STRATA)))
    axis.set_xticklabels([item[2] for item in PRIMARY_STRATA], rotation=20, ha="right")
    axis.set_yticks(np.arange(len(ALL_METRIC_ORDER)))
    axis.set_yticklabels([METRIC_LABELS[metric] for metric in ALL_METRIC_ORDER])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not np.isfinite(matrix[row, column]):
                label = "NA"
            else:
                label = (
                    f"{matrix[row, column]:+.2f}\n"
                    f"{significance[row, column]}/{denominators[row, column]}"
                )
            text_color = "white" if (
                np.isfinite(matrix[row, column])
                and abs(matrix[row, column]) > 0.58 * color_limit
            ) else "#222222"
            axis.text(column, row, label, ha="center", va="center", fontsize=7, color=text_color)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Mean paired standardized effect across conditions")
    axis.set_title(
        f"All strict matched outcomes — {args.split} split\n"
        "Cell: standardized effect; second line: conditions with global BH q < 0.05",
        fontsize=13, fontweight="bold",
    )
    axis.tick_params(length=0)
    return figure


def draw_stacked_composition(
    axis: plt.Axes,
    table: pd.DataFrame,
    row_column: str,
    value_column: str,
    component_column: str,
    row_order: list[str],
    component_order: list[str],
    colors: dict[str, str],
    row_labels: dict[str, str] | None = None,
    xlabel: str = "Composition (%)",
) -> None:
    y = np.arange(len(row_order))[::-1]
    for position, row_value in zip(y, row_order):
        group = table[table[row_column].eq(row_value)]
        left = 0.0
        for component in component_order:
            selected = group.loc[group[component_column].eq(component), value_column]
            value = 100 * (float(selected.iloc[0]) if len(selected) else 0.0)
            axis.barh(
                position, value, left=left, height=0.68,
                color=colors[component], edgecolor="white", linewidth=0.5,
            )
            left += value
    axis.set_yticks(y)
    axis.set_yticklabels([
        row_labels.get(value, value) if row_labels else value for value in row_order
    ])
    axis.set_xlim(0, 100)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)


def make_q3_q8_composition_figure(
    categorized: pd.DataFrame,
    q8_composition: pd.DataFrame,
    args: argparse.Namespace,
) -> plt.Figure:
    figure, (q3_axis, q8_axis) = plt.subplots(
        1, 2, figsize=(18, 8), constrained_layout=True,
        gridspec_kw={"width_ratios": (0.78, 1.22)},
    )
    source_masks = {
        "positive_apex": categorized["category"].eq("positive_apex"),
        "negative_apex": categorized["category"].eq("negative_apex"),
        "non_band": categorized["category"].eq("non_band"),
        "all_eligible": np.ones(len(categorized), dtype=bool),
    }
    q3_rows = []
    for source, mask in source_masks.items():
        fractions = categorized.loc[mask, "q3"].value_counts(normalize=True)
        for q3 in "HEC":
            q3_rows.append({
                "source": source, "q3": q3,
                "fraction": float(fractions.get(q3, 0.0)),
            })
    q3_table = pd.DataFrame(q3_rows)
    draw_stacked_composition(
        q3_axis, q3_table, "source", "fraction", "q3",
        ["positive_apex", "negative_apex", "non_band", "all_eligible"],
        list("HEC"),
        {"H": "#b2182b", "E": "#2166ac", "C": "#bdbdbd"},
        {
            "positive_apex": "+ band apices",
            "negative_apex": "− band apices",
            "non_band": "Outside every band",
            "all_eligible": "All eligible residues",
        },
        "Q3 composition (%)",
    )
    q3_axis.set_title("Q3 composition of band and background residues")
    q3_axis.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=color, label=label)
            for label, color in (("H", "#b2182b"), ("E", "#2166ac"), ("C", "#bdbdbd"))
        ],
        title="Q3", ncol=3, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
    )

    q8_mean = (
        q8_composition.groupby(["stratum", "role", "q8"], as_index=False)
        .fraction.mean()
    )
    q8_mean["row"] = q8_mean["stratum"] + " · " + q8_mean["role"]
    q8_order = []
    for _label, _q3, display, _color in PRIMARY_STRATA:
        q8_order.extend((f"{display} · apex", f"{display} · control"))
    draw_stacked_composition(
        q8_axis, q8_mean, "row", "fraction", "q8", q8_order,
        list(Q8_ORDER), Q8_COLORS,
        xlabel="Q8 composition (%)",
    )
    q8_axis.set_title(
        "Q8 composition after exact-Q3, within-protein comparison\n"
        "Controls: all same-Q3 residues outside every band"
    )
    q8_axis.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=Q8_COLORS[q8], label=q8)
            for q8 in Q8_ORDER
        ],
        title="Q8", ncol=8, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, -0.23), columnspacing=0.8,
    )
    panel_letter(q3_axis, "A")
    panel_letter(q8_axis, "B")
    figure.suptitle(
        f"Signed-band secondary-structure composition — {args.split} split",
        fontsize=15, fontweight="bold",
    )
    return figure


def make_q3_only_continuous_figure(
    histograms: pd.DataFrame,
    args: argparse.Namespace,
) -> plt.Figure:
    features = (
        ("neq", "Neq composition"),
        ("rsa", "RSA composition"),
        ("torsion_change_from_previous", "Torsional-change composition (degrees)"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    for row_index, (label, q3, display, color) in enumerate(PRIMARY_STRATA):
        for column_index, (feature, title) in enumerate(features):
            axis = axes[row_index, column_index]
            group = histograms[
                histograms["label"].eq(label)
                & histograms["q3"].eq(q3)
                & histograms["feature"].eq(feature)
            ]
            means = group.groupby(["role", "bin"]).fraction.mean()
            labels = list(FEATURE_BINS[feature][1])
            x = np.arange(len(labels))
            width = 0.39
            apex_values = np.array([
                means.get(("apex", bin_label), 0.0) for bin_label in labels
            ]) * 100
            control_values = np.array([
                means.get(("control", bin_label), 0.0) for bin_label in labels
            ]) * 100
            axis.bar(
                x - width / 2, apex_values, width, color=color,
                edgecolor="white", label="Band apex",
            )
            axis.bar(
                x + width / 2, control_values, width, color="#bdbdbd",
                edgecolor="white", label="Same-Q3 non-band control",
            )
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=32, ha="right")
            axis.set_ylabel("Residues in bin (%)")
            axis.set_title(f"{display}\n{title}")
            axis.grid(axis="y", color="#dddddd", linewidth=0.5)
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=8)
    for axis, letter in zip(axes.flat, "ABCDEFGHI"):
        panel_letter(axis, letter)
    figure.suptitle(
        f"Exact-Q3 apex versus all same-Q3 non-band residues — {args.split} split\n"
        "Bars average within-protein compositions across model conditions",
        fontsize=15, fontweight="bold",
    )
    return figure


def bootstrap_effect_summary(
    effects: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    selected = effects[effects["metric"].ne("__availability__")].dropna(
        subset=["difference"]
    )
    columns = ["condition", "label", "q3", "stratum", "metric"]
    for key, group in selected.groupby(columns, sort=False):
        values = group["difference"].to_numpy(dtype=float)
        estimate = float(np.mean(values))
        if len(values) == 1:
            low = high = estimate
        else:
            draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            bootstrap = np.mean(values[draws], axis=1)
            low, high = map(float, np.quantile(bootstrap, [0.025, 0.975]))
        rows.append({
            **dict(zip(columns, key)),
            "n_proteins": len(values), "effect": estimate,
            "ci95_low": low, "ci95_high": high,
        })
    return pd.DataFrame(rows)


def draw_condition_effects(
    axis: plt.Axes,
    effect_summary: pd.DataFrame,
    definitions: list[tuple[str, str, str, str]],
    conditions: list[str],
    multiplier: float,
    xlabel: str,
    title: str,
) -> None:
    y = np.arange(len(definitions))[::-1].astype(float)
    offsets = np.linspace(-0.22, 0.22, len(conditions))
    limits = []
    for condition_index, condition in enumerate(conditions):
        for row_index, (label, q3, metric, _display) in enumerate(definitions):
            row = effect_summary[
                effect_summary["condition"].eq(condition)
                & effect_summary["label"].eq(label)
                & effect_summary["q3"].eq(q3)
                & effect_summary["metric"].eq(metric)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            value = multiplier * float(row.effect)
            low = multiplier * float(row.ci95_low)
            high = multiplier * float(row.ci95_high)
            color = CONDITION_COLORS.get(condition, plt.cm.tab10(condition_index % 10))
            axis.errorbar(
                value, y[row_index] + offsets[condition_index],
                xerr=np.array([[value - low], [high - value]]),
                fmt="o", color=color, ecolor=color, markersize=4,
                elinewidth=0.8, capsize=1.5,
            )
            limits.extend((low, high))
    axis.axvline(0, color="#333333", linewidth=0.9)
    axis.set_yticks(y)
    axis.set_yticklabels([item[3] for item in definitions])
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    if limits:
        margin = max(0.03 * (max(limits) - min(limits)), 0.01)
        axis.set_xlim(min(limits) - margin, max(limits) + margin)


def make_q3_only_effect_figure(
    effects: pd.DataFrame,
    conditions: list[str],
    args: argparse.Namespace,
) -> plt.Figure:
    summary = bootstrap_effect_summary(effects)
    figure = plt.figure(figsize=(19, 12), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.35, 1, 1))
    binary_axis = figure.add_subplot(grid[:, 0])
    continuous_axes = (
        figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[1, 2]),
    )
    binary_definitions = [
        ("flexibility_supporting", "C", "q8_is_S", "+ C · Q8 S"),
        ("flexibility_supporting", "C", "structured_linker_loop", "+ C · structured linker"),
        ("flexibility_supporting", "C", "q3_boundary_within2", "+ C · near Q3 boundary"),
        ("rigidity_supporting", "H", "q8_is_H", "− H · Q8 H"),
        ("rigidity_supporting", "H", "q8_is_G", "− H · Q8 G"),
        ("rigidity_supporting", "H", "q3_boundary_within2", "− H · near Q3 boundary"),
        ("rigidity_supporting", "E", "q8_is_E", "− E · Q8 E"),
        ("rigidity_supporting", "E", "q3_boundary_within2", "− E · near Q3 boundary"),
    ]
    draw_condition_effects(
        binary_axis, summary, binary_definitions, conditions, 100.0,
        "Apex − same-Q3 non-band control (percentage points)",
        "Binary outcomes",
    )
    continuous = (
        ("neq", "Neq", "Apex − control (Neq units)"),
        ("rsa", "RSA", "Apex − control (RSA units)"),
        ("normalized_position", "Normalized sequence position", "Apex − control"),
        ("torsion_change_from_previous", "Torsional change", "Apex − control (degrees)"),
    )
    for axis, (metric, title, xlabel) in zip(continuous_axes, continuous):
        definitions = [
            (label, q3, metric, display)
            for label, q3, display, _color in PRIMARY_STRATA
        ]
        draw_condition_effects(
            axis, summary, definitions, conditions, 1.0, xlabel, title,
        )
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            color=CONDITION_COLORS.get(condition, plt.cm.tab10(index % 10)),
            label=CONDITION_LABELS.get(condition, condition),
        )
        for index, condition in enumerate(conditions)
    ]
    binary_axis.legend(
        handles=handles, ncol=2, frameon=False, loc="lower left",
        bbox_to_anchor=(0, -0.13),
    )
    for axis, letter in zip((binary_axis, *continuous_axes), "ABCDE"):
        panel_letter(axis, letter)
    figure.suptitle(
        f"Exact-Q3 within-protein effects — {args.split} split\n"
        "Controls are all same-Q3 residues outside every band; bars are protein-bootstrap 95% CIs",
        fontsize=15, fontweight="bold",
    )
    return figure


def inverse_composition_table(
    categorized: pd.DataFrame,
    feature: str,
    order: list[str],
) -> pd.DataFrame:
    counts = (
        categorized.groupby([feature, "category"], observed=False).size()
        .rename("count").reset_index()
    )
    counts["fraction"] = counts["count"] / counts.groupby(feature)["count"].transform("sum")
    counts[feature] = counts[feature].astype(str)
    return counts[counts[feature].isin(order)]


def draw_inverse_stacks(
    axis: plt.Axes,
    table: pd.DataFrame,
    feature: str,
    order: list[str],
    title: str,
) -> None:
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    positive = np.zeros(len(order))
    negative = np.zeros(len(order))
    for category in CATEGORY_ORDER:
        values = np.array([
            float(table.loc[
                table[feature].eq(label) & table["category"].eq(category),
                "fraction",
            ].iloc[0])
            if np.any(table[feature].eq(label) & table["category"].eq(category))
            else 0.0
            for label in order
        ]) * 100
        axis.bar(
            x, values, bottom=bottom, color=CATEGORY_COLORS[category],
            edgecolor="white", linewidth=0.45, label=CATEGORY_LABELS[category],
        )
        if category == "positive_apex":
            positive = values
        elif category == "negative_apex":
            negative = values
        bottom += values
    for position, plus, minus in zip(x, positive, negative):
        axis.text(
            position, 101.0, f"+{plus:.1f} / −{minus:.1f}",
            ha="center", va="bottom", fontsize=6.5, rotation=35,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=32, ha="right")
    axis.set_ylim(0, 111)
    axis.set_ylabel("Residues in category (%)")
    axis.set_title(title)
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)


def make_inverse_selection_figure(
    categorized: pd.DataFrame,
    args: argparse.Namespace,
) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    q3_order = list("HEC")
    q8_order = list(Q8_ORDER)
    draw_inverse_stacks(
        axes[0, 0], inverse_composition_table(categorized, "q3", q3_order),
        "q3", q3_order, "Within each Q3: band-selection composition",
    )
    draw_inverse_stacks(
        axes[0, 1], inverse_composition_table(categorized, "q8", q8_order),
        "q8", q8_order, "Within each Q8: band-selection composition",
    )
    binned = categorized.copy()
    panels = (
        ("neq", axes[0, 2], "Within each Neq bin"),
        ("rsa", axes[1, 0], "Within each RSA bin"),
        ("torsion_change_from_previous", axes[1, 1], "Within each torsion-change bin"),
        ("normalized_position", axes[1, 2], "Within each sequence-position bin"),
    )
    for feature, axis, title in panels:
        bins, labels = FEATURE_BINS[feature]
        column = f"{feature}_bin"
        binned[column] = pd.cut(
            pd.to_numeric(binned[feature], errors="coerce"), bins=bins,
            labels=labels, include_lowest=True, ordered=True,
        )
        table = inverse_composition_table(binned.dropna(subset=[column]), column, list(labels))
        draw_inverse_stacks(axis, table, column, list(labels), title)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[category],
                      label=CATEGORY_LABELS[category])
        for category in CATEGORY_ORDER
    ]
    figure.legend(
        handles=handles, ncol=4, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
    )
    for axis, letter in zip(axes.flat, "ABCDEF"):
        panel_letter(axis, letter)
    figure.suptitle(
        f"Inverse question: which residues become band apices? — {args.split} split\n"
        "Each bar sums to 100%; labels above bars give + apex % / − apex %",
        fontsize=15, fontweight="bold",
    )
    return figure


def main() -> None:
    args = parse_args()
    configure_style()
    bands, residue, protein_summary, conditions, paths = load_annotation_inputs(args)
    categorized = build_residue_categories(bands, residue, protein_summary)
    q8_composition, histograms, effects = build_q3_only_comparisons(categorized)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_suffix = f"{args.split}_q3_only"
    outputs: dict[str, list[str]] = {}
    outputs["q3_q8_composition"] = save_figure(
        make_q3_q8_composition_figure(categorized, q8_composition, args),
        output_dir, f"phase2_q3_q8_composition_{stem_suffix}",
        args.formats, args.dpi,
    )
    outputs["continuous_compositions"] = save_figure(
        make_q3_only_continuous_figure(histograms, args),
        output_dir, f"phase2_continuous_compositions_{stem_suffix}",
        args.formats, args.dpi,
    )
    outputs["q3_only_effects"] = save_figure(
        make_q3_only_effect_figure(effects, conditions, args),
        output_dir, f"phase2_effects_{stem_suffix}", args.formats, args.dpi,
    )
    outputs["inverse_selection"] = save_figure(
        make_inverse_selection_figure(categorized, args),
        output_dir, f"phase2_inverse_selection_{stem_suffix}",
        args.formats, args.dpi,
    )
    availability = effects[[
        "condition", "protein", "label", "q3", "stratum", "n_cases",
        "n_controls", "has_controls",
    ]].drop_duplicates()
    control_summary = availability.groupby(
        ["condition", "label", "q3", "stratum"], as_index=False
    ).agg(
        n_proteins_with_cases=("protein", "nunique"),
        n_cases=("n_cases", "sum"),
        n_proteins_with_controls=("has_controls", "sum"),
        n_cases_with_controls=(
            "n_cases", lambda values: int(values[availability.loc[values.index, "has_controls"]].sum())
        ),
        minimum_controls_per_protein=("n_controls", "min"),
        median_controls_per_protein=("n_controls", "median"),
        mean_controls_per_protein=("n_controls", "mean"),
        maximum_controls_per_protein=("n_controls", "max"),
    )
    control_summary["case_coverage"] = (
        control_summary["n_cases_with_controls"] / control_summary["n_cases"]
    )
    control_summary_path = output_dir / f"phase2_q3_only_control_availability_{args.split}.csv"
    control_summary.to_csv(control_summary_path, index=False)
    manifest = {
        "description": "Q3-only signed-band composition and selection figures",
        "split": args.split,
        "comparison": (
            "Every apex is compared with all eligible residues in the same protein "
            "with exact Q3 and outside every positive and negative band interval."
        ),
        "conditions": conditions,
        "condition_aggregation_note": (
            "Compositions and effects across model conditions are robustness summaries; "
            "conditions using the same proteins are not independent biological replicates."
        ),
        "category_definition": (
            "+ apex and - apex take priority; band_interval_non_apex contains all "
            "other residues covered by at least one band; non_band lies outside every band."
        ),
        "inputs": {name: str(path) for name, path in paths.items()},
        "outputs": outputs,
        "q3_only_control_availability": str(control_summary_path),
    }
    manifest_path = output_dir / f"phase2_figure_manifest_{stem_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "split": args.split,
        "comparison": "exact Q3; all same-protein non-band controls",
        "conditions": conditions,
        "figures": outputs,
        "control_availability": str(control_summary_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
