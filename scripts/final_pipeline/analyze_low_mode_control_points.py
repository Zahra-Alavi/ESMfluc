#!/usr/bin/env python3
"""
Test whether low_mode_1 attention hubs are conformational control points.

The analysis has two parts:
  1. Within low_mode_1, rank key residues by received attention and compare
     top/bottom quantiles against same-protein, same-SS matched backgrounds.
  2. Extract mode-specific bright attention blobs and motif windows around
     key residues, then summarize exact k-mers, reduced-alphabet patterns,
     and AA/class PWM enrichments.
"""

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_attention_row_modes import (
    analyze_attention_modes,
    default_analysis_dir,
    mode_name,
    resolve_existing_path,
    resolve_manifest_path,
)


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
REDUCED_ALPHABET = {
    "G": "G",  # glycine is kept separate because it is hinge/turn-special.
    "P": "P",  # proline is kept separate because it breaks regular SS.
    "A": "s", "S": "s",  # small/flexible, excluding G because it is explicit.
    "T": "p", "N": "p", "Q": "p",
    "D": "c", "E": "c", "K": "c", "R": "c", "H": "c",
    "V": "h", "L": "h", "I": "h", "M": "h", "F": "h", "W": "h", "Y": "h",
    "C": "x",
}
REDUCED_CLASS_LABELS = {
    "G": "glycine",
    "P": "proline",
    "s": "small_flexible_AS",
    "p": "polar_STNQ",
    "c": "charged_DEKRH",
    "h": "hydrophobic_AVLIMFWY",
    "x": "cysteine_other",
    "-": "padding",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Low-mode attention hub quantile, control-point, blob, and motif analysis."
    )
    p.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    p.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Defaults to result_root/manifest.tsv. "
             "Use manifest_attention_sources.tsv for all 30 attention sources.",
    )
    p.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    p.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id, q3, q8, and q8 probabilities.")
    p.add_argument("--output_dir", default=None, help="Default: result_root/analysis_low_mode_control_points[_suffix].")
    p.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument(
        "--row_mode_assignments",
        default=None,
        help="Residue assignment CSV from analyze_attention_row_modes.py. "
             "Default follows --manifest_tsv output dir if present.",
    )
    p.add_argument(
        "--pb_entropy_csv",
        default=None,
        help="Optional per-residue PB/protein-block entropy CSV with repeated id rows.",
    )
    p.add_argument("--pb_entropy_col", default=None, help="PB entropy column. Auto-detected if omitted.")
    p.add_argument("--pb_id_col", default="id", help="ID column in --pb_entropy_csv. Default: id.")
    p.add_argument(
        "--site_csv",
        default=None,
        help="Optional residue annotation CSV for functional/allosteric/control sites.",
    )
    p.add_argument("--site_id_col", default="id", help="Protein ID column in --site_csv.")
    p.add_argument("--site_pos_col", default="position", help="Residue position column in --site_csv.")
    p.add_argument("--site_label_col", default="label", help="Annotation label column in --site_csv.")
    p.add_argument("--site_position_base", type=int, choices=[0, 1], default=1)
    p.add_argument("--top_fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--bottom_frac", type=float, default=0.50)
    p.add_argument("--mode_label", type=int, default=1, help="Mode label to rank for quantile enrichment. Default: 1.")
    p.add_argument("--blob_mode_labels", nargs="+", type=int, default=[1, 2])
    p.add_argument("--high_neq_quantile", type=float, default=0.90)
    p.add_argument("--high_q8_entropy_quantile", type=float, default=0.90)
    p.add_argument("--high_pb_entropy_quantile", type=float, default=0.90)
    p.add_argument("--boundary_window", type=int, default=1)
    p.add_argument("--min_linker_len", type=int, default=1)
    p.add_argument("--max_linker_len", type=int, default=12)
    p.add_argument("--flank_window", type=int, default=3)
    p.add_argument("--background_per_selected", type=int, default=5)
    p.add_argument("--motif_len", type=int, default=11)
    p.add_argument("--kmer_len", type=int, default=5)
    p.add_argument("--blob_top_frac", type=float, default=0.10)
    p.add_argument("--blob_min_component_size", type=int, default=4)
    p.add_argument("--exclude_diagonal_window", type=int, default=0)
    p.add_argument("--background_scope", choices=["same_mode", "all_modes"], default="same_mode")
    p.add_argument("--high_entropy_quantile", type=float, default=0.67)
    p.add_argument("--min_low_rows", type=int, default=8)
    p.add_argument("--kmeans_seed", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--max_records", type=int, default=0, help="Debug limit across manifest records. 0 means all.")
    return p.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([float(v) > threshold for v in values], dtype=bool)


def load_neq_by_name(test_csv):
    df = pd.read_csv(test_csv)
    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]) if "name" in df.columns else str(row["sequence"])
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[name] = {"sequence": str(row["sequence"]), "neq": neq, "flexible": classify_neq(neq)}
    return out


def stripped_columns(df):
    return {c.strip(): c for c in df.columns}


def load_q3_q8_map(ss_csv):
    df = pd.read_csv(ss_csv)
    columns = stripped_columns(df)
    id_col = columns.get("id")
    q3_col = columns.get("q3")
    q8_col = columns.get("q8")
    if id_col is None or q3_col is None or q8_col is None:
        raise ValueError(f"{ss_csv} must contain id, q3, and q8 columns.")
    q8_prob_cols = [
        columns[c] for c in ["p[q8_G]", "p[q8_H]", "p[q8_I]", "p[q8_B]", "p[q8_E]", "p[q8_S]", "p[q8_T]", "p[q8_C]"]
        if c in columns
    ]
    out = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col]).lstrip(">")
        out.setdefault(seq_id, {"q3": [], "q8": [], "q8_entropy": []})
        out[seq_id]["q3"].append(str(row[q3_col]).strip())
        out[seq_id]["q8"].append(str(row[q8_col]).strip())
        if q8_prob_cols:
            probs = np.asarray([float(row[c]) for c in q8_prob_cols], dtype=float)
            probs = probs / max(float(probs.sum()), 1e-12)
            ent = -float(np.sum(probs * np.log(probs + 1e-12))) / np.log(len(probs))
            out[seq_id]["q8_entropy"].append(ent)
        else:
            out[seq_id]["q8_entropy"].append(np.nan)
    return out


def load_pb_entropy_map(path, id_col="id", entropy_col=None):
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    columns = stripped_columns(df)
    raw_id = columns.get(id_col.strip(), id_col)
    if raw_id not in df.columns:
        raise ValueError(f"{path} must contain id column {id_col!r}.")
    if entropy_col is None:
        for candidate in ["pb_entropy", "PB_entropy", "protein_block_entropy", "protein_block_ent", "pb_ent"]:
            if candidate in columns:
                entropy_col = columns[candidate]
                break
    elif entropy_col.strip() in columns:
        entropy_col = columns[entropy_col.strip()]
    if entropy_col is None or entropy_col not in df.columns:
        raise ValueError(f"Could not find PB entropy column in {path}. Pass --pb_entropy_col.")
    out = {}
    for _, row in df.iterrows():
        seq_id = str(row[raw_id]).lstrip(">")
        out.setdefault(seq_id, []).append(float(row[entropy_col]))
    return out


FUNCTIONAL_TERMS = ("functional", "active", "binding", "catalytic", "ligand", "site")
ALLOSTERIC_TERMS = ("allosteric", "communication", "communicator", "connector", "network", "pathway")


def load_site_annotation_map(path, id_col="id", pos_col="position", label_col="label", position_base=1):
    out = defaultdict(lambda: {"annotated": set(), "functional": set(), "allosteric": set()})
    if not path:
        return out
    path = Path(path).expanduser()
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Missing --site_id_col {id_col!r} in {path}.")
    if pos_col not in df.columns:
        raise ValueError(f"Missing --site_pos_col {pos_col!r} in {path}.")
    has_label = label_col in df.columns
    for _, row in df.iterrows():
        protein = str(row[id_col]).lstrip(">")
        try:
            idx = int(row[pos_col]) - int(position_base)
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        label = str(row[label_col]).lower() if has_label else "annotated"
        out[protein]["annotated"].add(idx)
        if any(term in label for term in FUNCTIONAL_TERMS):
            out[protein]["functional"].add(idx)
        if any(term in label for term in ALLOSTERIC_TERMS):
            out[protein]["allosteric"].add(idx)
    return out


def site_feature_arrays(site_map, protein, n):
    annotated = np.zeros(n, dtype=bool)
    functional = np.zeros(n, dtype=bool)
    allosteric = np.zeros(n, dtype=bool)
    entry = site_map.get(protein, {})
    for idx in entry.get("annotated", set()):
        if 0 <= int(idx) < n:
            annotated[int(idx)] = True
    for idx in entry.get("functional", set()):
        if 0 <= int(idx) < n:
            functional[int(idx)] = True
    for idx in entry.get("allosteric", set()):
        if 0 <= int(idx) < n:
            allosteric[int(idx)] = True
    return {
        "is_annotated_control_site": annotated,
        "is_functional_site": functional,
        "is_allosteric_connector": allosteric,
        "is_functional_or_allosteric": functional | allosteric,
    }


def load_assignment_maps(path, conditions=None):
    path = Path(path)
    if not path.exists():
        return {}
    print(f"[load] row-mode assignments: {path}")
    cols = ["condition", "seed", "protein", "position_1based", "mode_label", "row_entropy"]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    if conditions:
        df = df[df["condition"].isin(conditions)].copy()
    maps = {}
    for key, group in df.groupby(["condition", "seed", "protein"], sort=False):
        group = group.sort_values("position_1based")
        maps[(key[0], int(key[1]), key[2])] = {
            "modes": group["mode_label"].to_numpy(dtype=int),
            "row_entropy": group["row_entropy"].to_numpy(dtype=float),
        }
    return maps


def extract_window(sequence, center_idx, length):
    half = length // 2
    chars = []
    for offset in range(-half, half + 1):
        idx = center_idx + offset
        chars.append(sequence[idx] if 0 <= idx < len(sequence) else "-")
    return "".join(chars)


def reduced_pattern(text):
    return "".join(REDUCED_ALPHABET.get(aa, "-") for aa in text)


def top_indices(values, frac, largest=True):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.asarray([], dtype=int)
    k = max(1, int(np.ceil(len(values) * frac)))
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    return order[:k]


def transition_label(a, b):
    pair = {str(a), str(b)}
    if pair == {"C", "H"}:
        return "C_H"
    if pair == {"C", "E"}:
        return "C_E"
    if pair == {"H", "E"}:
        return "H_E"
    return None


def boundary_masks(q3, window):
    q3 = np.asarray(q3)
    n = len(q3)
    masks = {
        "boundary_any": np.zeros(n, dtype=bool),
        "boundary_C_H": np.zeros(n, dtype=bool),
        "boundary_C_E": np.zeros(n, dtype=bool),
        "boundary_H_E": np.zeros(n, dtype=bool),
    }
    for i in range(n - 1):
        if q3[i] == q3[i + 1]:
            continue
        label = transition_label(q3[i], q3[i + 1])
        start = max(0, i - window)
        end = min(n, i + 2 + window)
        masks["boundary_any"][start:end] = True
        if label:
            masks[f"boundary_{label}"][start:end] = True
    return masks


def short_structured_linker_mask(q3, min_len, max_len, flank_window):
    q3 = np.asarray(q3)
    n = len(q3)
    structured = np.isin(q3, ["H", "E"])
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if q3[i] != "C":
            i += 1
            continue
        start = i
        while i < n and q3[i] == "C":
            i += 1
        end = i - 1
        length = end - start + 1
        if length < min_len or length > max_len:
            continue
        left = structured[max(0, start - flank_window):start]
        right = structured[end + 1:min(n, end + flank_window + 1)]
        if bool(np.any(left)) and bool(np.any(right)):
            mask[start:end + 1] = True
    return mask


def high_mask(values, quantile):
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=bool)
    valid = np.isfinite(values)
    if not np.any(valid):
        return out
    threshold = float(np.quantile(values[valid], quantile))
    out[valid] = values[valid] >= threshold
    return out


def feature_arrays(q3, q8, neq, q8_entropy, pb_entropy, args):
    q3 = np.asarray(q3)
    q8 = np.asarray(q8)
    neq = np.asarray(neq, dtype=float)
    q8_entropy = np.asarray(q8_entropy, dtype=float)
    pb_entropy = np.asarray(pb_entropy, dtype=float)
    boundaries = boundary_masks(q3, args.boundary_window)
    linker = short_structured_linker_mask(q3, args.min_linker_len, args.max_linker_len, args.flank_window)
    out = {
        "q3": q3,
        "q8": q8,
        "neq": neq,
        "q8_entropy": q8_entropy,
        "pb_entropy": pb_entropy,
        "is_q8_turn_T": q8 == "T",
        "is_q8_bend_S": q8 == "S",
        "is_q8_coil_C": q8 == "C",
        "is_q8_loop_CTS": np.isin(q8, ["C", "T", "S"]),
        "is_q3_coil_C": q3 == "C",
        "is_short_structured_linker": linker,
        "is_flexible_neq_gt1": neq > 1.0,
        "is_high_neq": high_mask(neq, args.high_neq_quantile),
        "is_high_q8_entropy": high_mask(q8_entropy, args.high_q8_entropy_quantile),
        "is_high_pb_entropy": high_mask(pb_entropy, args.high_pb_entropy_quantile),
    }
    out.update({f"is_{name}": value for name, value in boundaries.items()})
    return out


SUMMARY_METRICS = [
    "is_q8_turn_T",
    "is_q8_bend_S",
    "is_q8_coil_C",
    "is_q8_loop_CTS",
    "is_q3_coil_C",
    "is_boundary_any",
    "is_boundary_C_H",
    "is_boundary_C_E",
    "is_boundary_H_E",
    "is_short_structured_linker",
    "is_flexible_neq_gt1",
    "is_high_neq",
    "is_high_q8_entropy",
    "is_high_pb_entropy",
    "is_annotated_control_site",
    "is_functional_site",
    "is_allosteric_connector",
    "is_functional_or_allosteric",
]
VALUE_METRICS = ["neq", "q8_entropy", "pb_entropy", "received_attention"]


def summarize_indices(indices, features, received):
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if len(indices) == 0:
        return None
    row = {"n_residues": int(len(indices))}
    for metric in SUMMARY_METRICS:
        values = features.get(metric)
        clean_metric = metric[3:] if metric.startswith("is_") else metric
        row[f"fraction_{clean_metric}"] = float(np.mean(values[indices])) if values is not None else np.nan
    for metric in VALUE_METRICS:
        values = received if metric == "received_attention" else features.get(metric)
        if values is None:
            row[f"mean_{metric}"] = np.nan
            continue
        vals = np.asarray(values, dtype=float)[indices]
        row[f"mean_{metric}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
    return row


def matched_background(selected, eligible, q3, rng, n_per_selected):
    selected = sorted(set(int(i) for i in selected))
    selected_set = set(selected)
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    sampled = []
    if len(selected) == 0 or len(eligible) == 0:
        return sampled
    for idx in selected:
        candidates = eligible[q3[eligible] == q3[idx]]
        candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([c for c in eligible if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = eligible
        replace = len(candidates) < n_per_selected
        sampled.extend(int(x) for x in rng.choice(candidates, size=n_per_selected, replace=replace))
    return sampled


def residue_rows(indices, run, protein, selection, background, seq, features, received, modes, row_entropy):
    rows = []
    for idx in sorted(int(i) for i in indices):
        rows.append({
            "condition": run.condition,
            "seed": int(run.seed),
            "protein": protein,
            "selection": selection,
            "background": bool(background),
            "position_1based": idx + 1,
            "aa": seq[idx],
            "mode": mode_name(int(modes[idx])),
            "mode_label": int(modes[idx]),
            "q3": features["q3"][idx],
            "q8": features["q8"][idx],
            "neq": float(features["neq"][idx]),
            "q8_entropy": float(features["q8_entropy"][idx]) if np.isfinite(features["q8_entropy"][idx]) else np.nan,
            "pb_entropy": float(features["pb_entropy"][idx]) if np.isfinite(features["pb_entropy"][idx]) else np.nan,
            "row_entropy": float(row_entropy[idx]) if row_entropy is not None and idx < len(row_entropy) else np.nan,
            "received_attention": float(received[idx]),
            **{metric: bool(features[metric][idx]) for metric in SUMMARY_METRICS if metric in features},
        })
    return rows


def motif_rows(indices, run, protein, source, mode_label, background, seq, features, received, motif_len, kmer_len, weight_values=None):
    rows = []
    for idx in sorted(set(int(i) for i in indices)):
        motif = extract_window(seq, idx, motif_len)
        kmer = extract_window(seq, idx, kmer_len)
        rows.append({
            "condition": run.condition,
            "seed": int(run.seed),
            "protein": protein,
            "source": source,
            "mode": mode_name(int(mode_label)),
            "mode_label": int(mode_label),
            "background": bool(background),
            "position_1based": idx + 1,
            "aa": seq[idx],
            "q3": features["q3"][idx],
            "q8": features["q8"][idx],
            "neq": float(features["neq"][idx]),
            "received_attention": float(received[idx]),
            "weight": float(weight_values[idx]) if weight_values is not None else float(received[idx]),
            "motif": motif,
            "kmer": kmer,
            "reduced_motif": reduced_pattern(motif),
            "reduced_kmer": reduced_pattern(kmer),
        })
    return rows


def neighbors8(r, c, n):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                yield rr, cc


def connected_components(mask):
    n = mask.shape[0]
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(n):
        for c in range(n):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors8(rr, cc, n):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def detect_mode_blobs(attn, modes, mode_label, top_frac, min_component_size, exclude_diagonal_window):
    n = attn.shape[0]
    work = np.asarray(attn, dtype=float).copy()
    if exclude_diagonal_window > 0:
        ii, jj = np.indices((n, n))
        work[np.abs(ii - jj) <= exclude_diagonal_window] = np.nan
    key_idx = np.where(modes == mode_label)[0]
    if len(key_idx) == 0:
        return [], np.full(n, np.nan)
    col_max = np.nanmax(work, axis=0)
    valid_values = col_max[key_idx]
    valid_values = valid_values[np.isfinite(valid_values)]
    if len(valid_values) == 0:
        return [], col_max
    threshold = float(np.quantile(valid_values, max(0.0, 1.0 - top_frac)))
    bright_cols = set(int(i) for i in key_idx if np.isfinite(col_max[i]) and col_max[i] >= threshold)
    mask = np.zeros((n, n), dtype=bool)
    for j in bright_cols:
        mask[:, j] = np.isfinite(work[:, j]) & (work[:, j] >= threshold)
    comps = [c for c in connected_components(mask) if len(c) >= min_component_size]
    rows = []
    for cid, comp in enumerate(comps, start=1):
        rr = np.asarray([p[0] for p in comp], dtype=int)
        cc = np.asarray([p[1] for p in comp], dtype=int)
        cols = sorted(set(int(c) for c in cc))
        peak_col = int(max(cols, key=lambda c: col_max[c] if np.isfinite(col_max[c]) else -np.inf))
        if modes[peak_col] != mode_label:
            continue
        vals = attn[rr, cc]
        rows.append({
            "component_id": cid,
            "mode_label": int(mode_label),
            "mode": mode_name(int(mode_label)),
            "component_size": int(len(comp)),
            "row_span": f"{int(rr.min())}-{int(rr.max())}",
            "col_span": f"{int(cc.min())}-{int(cc.max())}",
            "row_center_idx": int(np.round(rr.mean())),
            "col_center_idx": int(np.round(cc.mean())),
            "peak_key_idx": peak_col,
            "threshold": threshold,
            "component_mean_value": float(np.mean(vals)),
            "component_max_value": float(np.max(vals)),
            "key_col_max": float(col_max[peak_col]),
        })
    return rows, col_max


def aggregate_quantile_summaries(summary_df):
    if summary_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in summary_df.columns if c.startswith("obs_")]
    rows = []
    for key, group in summary_df.groupby(["condition", "selection"], sort=False):
        row = {"condition": key[0], "selection": key[1], "n_units": len(group)}
        for obs_col in metric_cols:
            metric = obs_col[len("obs_"):]
            bg_col = f"bg_{metric}"
            if bg_col not in group.columns:
                continue
            obs = group[obs_col].astype(float)
            bg = group[bg_col].astype(float)
            valid = obs.notna() & bg.notna()
            if not valid.any():
                continue
            obs_v = obs[valid].to_numpy()
            bg_v = bg[valid].to_numpy()
            row[f"obs_{metric}_mean"] = float(np.mean(obs_v))
            row[f"bg_{metric}_mean"] = float(np.mean(bg_v))
            row[f"delta_{metric}_mean"] = float(np.mean(obs_v - bg_v))
            denom = float(np.mean(bg_v))
            row[f"enrichment_{metric}"] = float(np.mean(obs_v) / denom) if abs(denom) > 1e-12 else np.nan
            row[f"p_obs_greater_{metric}"] = float((np.sum(bg_v >= obs_v) + 1) / (len(obs_v) + 1))
            row[f"p_obs_less_{metric}"] = float((np.sum(bg_v <= obs_v) + 1) / (len(obs_v) + 1))
        rows.append(row)
    return pd.DataFrame(rows)


def motif_count_enrichment(rows, value_col, output_col):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    group_cols = ["condition", "mode", "source", value_col]
    obs = df[~df["background"]].copy()
    bg = df[df["background"]].copy()
    obs_counts = (
        obs.groupby(group_cols)
        .agg(
            observed_count=(value_col, "size"),
            observed_unique_proteins=("protein", "nunique"),
            observed_unique_seeds=("seed", "nunique"),
            observed_mean_weight=("weight", "mean"),
        )
        .reset_index()
    )
    bg_counts = bg.groupby(group_cols).size().rename("background_count").reset_index()
    obs_totals = obs.groupby(["condition", "mode", "source"]).size().rename("observed_total").reset_index()
    bg_totals = bg.groupby(["condition", "mode", "source"]).size().rename("background_total").reset_index()
    out = obs_counts.merge(bg_counts, on=group_cols, how="left")
    out = out.merge(obs_totals, on=["condition", "mode", "source"], how="left")
    out = out.merge(bg_totals, on=["condition", "mode", "source"], how="left")
    out["background_count"] = out["background_count"].fillna(0).astype(int)
    out["background_total"] = out["background_total"].fillna(0).astype(int)
    out["observed_fraction"] = out["observed_count"] / out["observed_total"].clip(lower=1)
    out["background_fraction"] = out["background_count"] / out["background_total"].clip(lower=1)
    out["enrichment"] = (out["observed_fraction"] + 1e-9) / (out["background_fraction"] + 1e-9)
    out["log2_enrichment"] = np.log2(out["enrichment"])
    out = out.rename(columns={value_col: output_col})
    return out.sort_values(["condition", "mode", "source", "log2_enrichment", "observed_count"], ascending=[True, True, True, False, False])


def pwm_enrichment(rows, motif_col, alphabet, symbol_col):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out_rows = []
    for key, group in df.groupby(["condition", "mode", "source"], sort=False):
        obs = group[~group["background"]]
        bg = group[group["background"]]
        if obs.empty:
            continue
        motif_len = len(str(obs.iloc[0][motif_col]))
        half = motif_len // 2
        obs_counts = np.zeros((motif_len, len(alphabet)), dtype=float)
        bg_counts = np.zeros((motif_len, len(alphabet)), dtype=float)
        index = {a: i for i, a in enumerate(alphabet)}
        for _, row in obs.iterrows():
            motif = str(row[motif_col])
            weight = float(row.get("weight", 1.0))
            for pos, sym in enumerate(motif[:motif_len]):
                if sym in index:
                    obs_counts[pos, index[sym]] += weight
        for _, row in bg.iterrows():
            motif = str(row[motif_col])
            for pos, sym in enumerate(motif[:motif_len]):
                if sym in index:
                    bg_counts[pos, index[sym]] += 1.0
        obs_freq = obs_counts / np.maximum(obs_counts.sum(axis=1, keepdims=True), 1e-9)
        bg_freq = bg_counts / np.maximum(bg_counts.sum(axis=1, keepdims=True), 1e-9)
        for pos in range(motif_len):
            for sym in alphabet:
                ai = index[sym]
                obs_f = float(obs_freq[pos, ai])
                bg_f = float(bg_freq[pos, ai])
                out_rows.append({
                    "condition": key[0],
                    "mode": key[1],
                    "source": key[2],
                    "position": pos - half,
                    symbol_col: sym,
                    "frequency": obs_f,
                    "bg_frequency": bg_f,
                    "enrichment": obs_f / bg_f if bg_f > 1e-12 else np.nan,
                    "log2_enrichment": np.log2((obs_f + 1e-6) / (bg_f + 1e-6)),
                })
    return pd.DataFrame(out_rows)


def main():
    args = parse_args()
    if args.motif_len % 2 != 1 or args.kmer_len % 2 != 1:
        raise ValueError("--motif_len and --kmer_len must be odd.")

    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest TSV: {manifest_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_low_mode_control_points", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()

    assignment_path = (
        Path(args.row_mode_assignments).expanduser()
        if args.row_mode_assignments
        else default_analysis_dir(result_root, "analysis_row_modes", manifest_path) / "row_mode_assignments_by_residue.csv"
    )
    assignment_maps = load_assignment_maps(assignment_path, set(args.conditions or []))
    neq_by_name = load_neq_by_name(args.test_csv)
    ss_map = load_q3_q8_map(args.ss_csv)
    pb_map = load_pb_entropy_map(args.pb_entropy_csv, args.pb_id_col, args.pb_entropy_col)
    site_map = load_site_annotation_map(
        args.site_csv,
        args.site_id_col,
        args.site_pos_col,
        args.site_label_col,
        args.site_position_base,
    )
    rng = np.random.default_rng(args.random_seed)

    residue_out = []
    bg_residue_out = []
    summary_rows = []
    motif_out = []
    blob_rows = []
    missing = []
    processed_records = 0

    for run in manifest.itertuples(index=False):
        attention_path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            missing.append(f"{run.condition}\t{run.seed}\t{attention_path}")
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            if args.max_records and processed_records >= args.max_records:
                break
            protein = record["name"]
            if protein not in neq_by_name or protein not in ss_map:
                continue
            seq = str(record["sequence"])
            n = len(seq)
            q3 = np.asarray(ss_map[protein]["q3"][:n])
            q8 = np.asarray(ss_map[protein]["q8"][:n])
            q8_entropy = np.asarray(ss_map[protein]["q8_entropy"][:n], dtype=float)
            neq = neq_by_name[protein]["neq"][:n]
            pb_entropy = np.asarray(pb_map.get(protein, [np.nan] * n)[:n], dtype=float)
            if len(q3) != n or len(q8) != n or len(neq) != n:
                continue
            if len(pb_entropy) != n:
                pb_entropy = np.full(n, np.nan)

            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            received = attn.sum(axis=0)
            cached = assignment_maps.get((run.condition, int(run.seed), protein))
            if cached is not None and len(cached["modes"]) >= n:
                modes = cached["modes"][:n]
                row_entropy = cached["row_entropy"][:n]
            else:
                modes, row_entropy, _, _, _, _ = analyze_attention_modes(
                    attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed
                )

            features = feature_arrays(q3, q8, neq, q8_entropy, pb_entropy, args)
            features.update(site_feature_arrays(site_map, protein, n))
            low_idx = np.where(modes == args.mode_label)[0]
            if len(low_idx) == 0:
                processed_records += 1
                continue

            eligible_bg = low_idx if args.background_scope == "same_mode" else np.arange(n)
            selection_sets = {}
            for frac in args.top_fracs:
                local = top_indices(received[low_idx], frac, largest=True)
                selection_sets[f"top_{int(round(frac * 100))}pct"] = low_idx[local]
            local = top_indices(received[low_idx], args.bottom_frac, largest=False)
            selection_sets[f"bottom_{int(round(args.bottom_frac * 100))}pct"] = low_idx[local]

            for selection, selected in selection_sets.items():
                selected = sorted(set(int(i) for i in selected))
                if not selected:
                    continue
                bg = matched_background(selected, eligible_bg, q3, rng, args.background_per_selected)
                obs_summary = summarize_indices(selected, features, received)
                bg_summary = summarize_indices(bg, features, received)
                if obs_summary and bg_summary:
                    row = {
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                        "selection": selection,
                    }
                    row.update({f"obs_{k}": v for k, v in obs_summary.items()})
                    row.update({f"bg_{k}": v for k, v in bg_summary.items()})
                    summary_rows.append(row)
                residue_out.extend(residue_rows(selected, run, protein, selection, False, seq, features, received, modes, row_entropy))
                bg_residue_out.extend(residue_rows(bg, run, protein, selection, True, seq, features, received, modes, row_entropy))
                motif_out.extend(motif_rows(
                    selected, run, protein, f"quantile_{selection}", args.mode_label,
                    False, seq, features, received, args.motif_len, args.kmer_len
                ))
                motif_out.extend(motif_rows(
                    bg, run, protein, f"quantile_{selection}", args.mode_label,
                    True, seq, features, received, args.motif_len, args.kmer_len
                ))

            for blob_mode in args.blob_mode_labels:
                blobs, col_max = detect_mode_blobs(
                    attn, modes, blob_mode, args.blob_top_frac,
                    args.blob_min_component_size, args.exclude_diagonal_window
                )
                key_positions = []
                for blob in blobs:
                    idx = int(blob["peak_key_idx"])
                    key_positions.append(idx)
                    blob_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                        "position_1based": idx + 1,
                        "aa": seq[idx],
                        "q3": q3[idx],
                        "q8": q8[idx],
                        "received_attention": float(received[idx]),
                        "col_max": float(col_max[idx]) if np.isfinite(col_max[idx]) else np.nan,
                        **blob,
                    })
                key_positions = sorted(set(key_positions))
                if key_positions:
                    blob_bg = matched_background(
                        key_positions,
                        np.where(modes == blob_mode)[0] if args.background_scope == "same_mode" else np.arange(n),
                        q3,
                        rng,
                        args.background_per_selected,
                    )
                    source = f"blob_colmax_top_{int(round(args.blob_top_frac * 100))}pct"
                    motif_out.extend(motif_rows(
                        key_positions, run, protein, source, blob_mode, False,
                        seq, features, received, args.motif_len, args.kmer_len, weight_values=col_max
                    ))
                    motif_out.extend(motif_rows(
                        blob_bg, run, protein, source, blob_mode, True,
                        seq, features, received, args.motif_len, args.kmer_len
                    ))

            processed_records += 1
        if args.max_records and processed_records >= args.max_records:
            break

    residue_df = pd.DataFrame(residue_out)
    bg_residue_df = pd.DataFrame(bg_residue_out)
    summary_df = pd.DataFrame(summary_rows)
    motif_df = pd.DataFrame(motif_out)
    blob_df = pd.DataFrame(blob_rows)

    residue_df.to_csv(output_dir / "low_mode1_quantile_residues.csv", index=False)
    bg_residue_df.to_csv(output_dir / "low_mode1_quantile_matched_background_residues.csv", index=False)
    summary_df.to_csv(output_dir / "low_mode1_quantile_enrichment_by_run.csv", index=False)
    aggregate_quantile_summaries(summary_df).to_csv(output_dir / "low_mode1_quantile_enrichment_by_condition.csv", index=False)
    blob_df.to_csv(output_dir / "mode_attention_blobs.csv", index=False)
    motif_df.to_csv(output_dir / "mode_attention_motif_instances.csv", index=False)

    motif_count_enrichment(motif_out, "kmer", "kmer").to_csv(output_dir / "exact_kmer_recurrence_enrichment.csv", index=False)
    motif_count_enrichment(motif_out, "reduced_kmer", "reduced_kmer").to_csv(
        output_dir / "reduced_alphabet_kmer_recurrence_enrichment.csv", index=False
    )
    pwm_enrichment(motif_out, "motif", AA20 + ["-"], "AA").to_csv(output_dir / "aa_pwm_enrichment.csv", index=False)
    pwm_enrichment(motif_out, "reduced_motif", list(REDUCED_CLASS_LABELS), "class_code").to_csv(
        output_dir / "reduced_class_pwm_enrichment.csv", index=False
    )

    if missing:
        (output_dir / "missing_attention_files.txt").write_text("\n".join(missing) + "\n")

    summary_lines = [
        f"Result root: {result_root}",
        f"Manifest: {manifest_path}",
        f"Output dir: {output_dir}",
        f"Processed protein/run records: {processed_records}",
        f"Quantile residue rows: {len(residue_df)}",
        f"Matched background residue rows: {len(bg_residue_df)}",
        f"Blob rows: {len(blob_df)}",
        f"Motif rows: {len(motif_df)}",
        f"PB entropy available: {bool(pb_map)}",
        f"Site annotations available: {bool(site_map)}",
        f"Reduced alphabet: {json.dumps(REDUCED_CLASS_LABELS, sort_keys=True)}",
        "",
        "Key outputs:",
        "  low_mode1_quantile_enrichment_by_condition.csv",
        "  low_mode1_quantile_residues.csv",
        "  mode_attention_blobs.csv",
        "  mode_attention_motif_instances.csv",
        "  exact_kmer_recurrence_enrichment.csv",
        "  reduced_alphabet_kmer_recurrence_enrichment.csv",
        "  aa_pwm_enrichment.csv",
        "  reduced_class_pwm_enrichment.csv",
    ]
    (output_dir / "low_mode_control_point_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
