#!/usr/bin/env python3
"""
Analyze attention maps as sparse directed residue graphs and annotate hubs.

Each attention matrix is treated as a directed weighted graph:
  query residue i -> key residue j, edge weight = attention[i, j]

The main biological object is the key-side hub: residues with high incoming
top-k attention.  The script writes per-residue hub metrics, same-SS matched
hub enrichment tests, and seed-stability summaries.
"""

import argparse
import itertools
import json
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
from analyze_low_mode_control_points import (
    SUMMARY_METRICS,
    VALUE_METRICS,
    feature_arrays,
    load_assignment_maps,
    load_neq_by_name,
    load_pb_entropy_map,
    load_q3_q8_map,
    load_site_annotation_map,
    matched_background,
    site_feature_arrays,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Publication-style attention graph hub analysis."
    )
    p.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    p.add_argument(
        "--manifest_tsv",
        default=None,
        help="Manifest TSV to analyze. Use manifest_attention_sources.tsv for the 30-source view.",
    )
    p.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    p.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id, q3, q8, and q8 probabilities.")
    p.add_argument("--output_dir", default=None, help="Default: result_root/analysis_attention_graph_hubs[_suffix].")
    p.add_argument("--pipeline_dir", default=None, help="Directory for resolving relative manifest paths.")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument(
        "--row_mode_assignments",
        default=None,
        help="Optional row_mode_assignments_by_residue.csv. If absent, modes are recomputed per record.",
    )
    p.add_argument("--pb_entropy_csv", default=None)
    p.add_argument("--pb_entropy_col", default=None)
    p.add_argument("--pb_id_col", default="id")
    p.add_argument("--site_csv", default=None)
    p.add_argument("--site_id_col", default="id")
    p.add_argument("--site_pos_col", default="position")
    p.add_argument("--site_label_col", default="label")
    p.add_argument("--site_position_base", type=int, choices=[0, 1], default=1)
    p.add_argument(
        "--topk_values",
        nargs="+",
        type=int,
        default=[1, 2, 5],
        help="Outgoing top-k graph definitions to compute. Default: 1 2 5.",
    )
    p.add_argument(
        "--exclude_diagonal_window",
        type=int,
        default=0,
        help="Exclude edges with |i-j| <= window before top-k. Default 0 excludes self-edges. Use -1 to include self.",
    )
    p.add_argument("--top_fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument(
        "--hub_scores",
        nargs="+",
        default=[
            "received_attention",
            "in_degree_top1",
            "weighted_in_degree_top1",
            "pagerank_top1",
            "in_degree_top2",
            "weighted_in_degree_top2",
            "in_degree_top5",
            "weighted_in_degree_top5",
        ],
        help="Residue metric columns used to define top hubs.",
    )
    p.add_argument(
        "--pool_mode_labels",
        nargs="+",
        type=int,
        default=[-1, 1, 2],
        help="Pools for top-hub selection: -1 means all residues, 1/2 are low modes. Default: -1 1 2.",
    )
    p.add_argument("--background_per_selected", type=int, default=5)
    p.add_argument("--high_neq_quantile", type=float, default=0.90)
    p.add_argument("--high_q8_entropy_quantile", type=float, default=0.90)
    p.add_argument("--high_pb_entropy_quantile", type=float, default=0.90)
    p.add_argument("--boundary_window", type=int, default=1)
    p.add_argument("--min_linker_len", type=int, default=1)
    p.add_argument("--max_linker_len", type=int, default=12)
    p.add_argument("--flank_window", type=int, default=3)
    p.add_argument("--high_entropy_quantile", type=float, default=0.67)
    p.add_argument("--min_low_rows", type=int, default=8)
    p.add_argument("--kmeans_seed", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--top_hubs_per_protein", type=int, default=25)
    p.add_argument("--max_records", type=int, default=0, help="Debug limit across manifest records. 0 means all.")
    return p.parse_args()


def row_entropy(attn):
    eps = 1e-12
    n = attn.shape[1]
    if n <= 1:
        return np.zeros(attn.shape[0], dtype=float)
    row_sums = attn.sum(axis=1, keepdims=True)
    probs = np.divide(attn, row_sums, out=np.zeros_like(attn), where=row_sums > 0)
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return ent / np.log(n)


def q8_entry(ss_map, protein, n):
    entry = ss_map.get(protein)
    if entry is None:
        return None
    q3 = np.asarray(entry["q3"][:n])
    q8 = np.asarray(entry["q8"][:n])
    q8_entropy = np.asarray(entry["q8_entropy"][:n], dtype=float)
    if len(q3) != n or len(q8) != n:
        return None
    return q3, q8, q8_entropy


def pb_array(pb_map, protein, n):
    values = pb_map.get(protein)
    if values is None:
        return np.full(n, np.nan)
    arr = np.asarray(values[:n], dtype=float)
    if len(arr) != n:
        return np.full(n, np.nan)
    return arr


def select_topk_indices(row, k):
    valid = np.where(np.isfinite(row))[0]
    if len(valid) == 0:
        return np.asarray([], dtype=int)
    k = min(int(k), len(valid))
    if k <= 0:
        return np.asarray([], dtype=int)
    local = np.argpartition(-row[valid], k - 1)[:k]
    return valid[local]


def pagerank_from_adjacency(adj, damping=0.85, max_iter=100, tol=1e-10):
    n = adj.shape[0]
    if n == 0:
        return np.asarray([])
    row_sums = adj.sum(axis=1, keepdims=True)
    trans = np.divide(adj, row_sums, out=np.zeros_like(adj), where=row_sums > 0)
    dangling = np.where(row_sums[:, 0] <= 0)[0]
    rank = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - damping) / n)
    for _ in range(max_iter):
        new_rank = teleport + damping * trans.T.dot(rank)
        if len(dangling):
            new_rank += damping * rank[dangling].sum() / n
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank
    total = rank.sum()
    return rank / total if total > 0 else rank


def graph_hub_metrics(attn, modes, topk_values, exclude_diagonal_window):
    n = attn.shape[0]
    work = np.asarray(attn, dtype=float).copy()
    if exclude_diagonal_window >= 0:
        ii, jj = np.indices((n, n))
        work[np.abs(ii - jj) <= exclude_diagonal_window] = np.nan

    metrics = {
        "received_attention": np.nansum(work, axis=0),
        "max_received_attention": np.nanmax(work, axis=0),
    }
    metrics["row_entropy"] = row_entropy(attn)

    for k in topk_values:
        adj = np.zeros((n, n), dtype=float)
        source_counts = {
            0: np.zeros(n, dtype=int),
            1: np.zeros(n, dtype=int),
            2: np.zeros(n, dtype=int),
        }
        source_weight = {
            0: np.zeros(n, dtype=float),
            1: np.zeros(n, dtype=float),
            2: np.zeros(n, dtype=float),
        }
        same_mode_counts = np.zeros(n, dtype=int)
        same_mode_weight = np.zeros(n, dtype=float)

        for i in range(n):
            selected = select_topk_indices(work[i], k)
            for j in selected:
                weight = float(work[i, j])
                if not np.isfinite(weight):
                    continue
                adj[i, j] = weight
                src_mode = int(modes[i]) if i < len(modes) else -1
                tgt_mode = int(modes[j]) if j < len(modes) else -2
                if src_mode in source_counts:
                    source_counts[src_mode][j] += 1
                    source_weight[src_mode][j] += weight
                if src_mode == tgt_mode:
                    same_mode_counts[j] += 1
                    same_mode_weight[j] += weight

        indeg = (adj > 0).sum(axis=0).astype(float)
        windeg = adj.sum(axis=0)
        metrics[f"in_degree_top{k}"] = indeg
        metrics[f"weighted_in_degree_top{k}"] = windeg
        metrics[f"mean_in_weight_top{k}"] = np.divide(windeg, indeg, out=np.full(n, np.nan), where=indeg > 0)
        metrics[f"pagerank_top{k}"] = pagerank_from_adjacency(adj)
        metrics[f"same_mode_in_fraction_top{k}"] = np.divide(
            same_mode_counts, indeg, out=np.full(n, np.nan), where=indeg > 0
        )
        metrics[f"same_mode_weight_fraction_top{k}"] = np.divide(
            same_mode_weight, windeg, out=np.full(n, np.nan), where=windeg > 0
        )
        for label in [0, 1, 2]:
            name = mode_name(label)
            metrics[f"in_from_{name}_top{k}"] = source_counts[label].astype(float)
            metrics[f"weighted_in_from_{name}_top{k}"] = source_weight[label]

    return metrics


def rank_desc(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, -np.inf))[::-1]
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    ranks[~np.isfinite(values)] = np.nan
    return ranks


def build_residue_rows(run_dict, protein, sequence, features, modes, metrics):
    n = len(sequence)
    rank_cols = {
        f"rank_{name}": rank_desc(values)
        for name, values in metrics.items()
        if name.startswith(("received_attention", "in_degree_", "weighted_in_degree_", "pagerank_"))
    }
    rows = []
    for i, aa in enumerate(sequence):
        row = {
            "condition": run_dict.get("condition", ""),
            "seed": int(run_dict.get("seed", -1)),
            "architecture": run_dict.get("architecture", ""),
            "esm_model": run_dict.get("esm_model", ""),
            "attention_kind": run_dict.get("attention_kind", ""),
            "source_condition": run_dict.get("source_condition", ""),
            "protein": protein,
            "n_residues": n,
            "position_1based": i + 1,
            "aa": aa,
            "mode_label": int(modes[i]),
            "mode": mode_name(int(modes[i])),
            "q3": features["q3"][i],
            "q8": features["q8"][i],
            "neq": float(features["neq"][i]),
            "q8_entropy": float(features["q8_entropy"][i]) if np.isfinite(features["q8_entropy"][i]) else np.nan,
            "pb_entropy": float(features["pb_entropy"][i]) if np.isfinite(features["pb_entropy"][i]) else np.nan,
        }
        for metric in SUMMARY_METRICS:
            if metric in features:
                row[metric] = bool(features[metric][i])
        for name, values in metrics.items():
            row[name] = float(values[i]) if np.isfinite(values[i]) else np.nan
        for name, values in rank_cols.items():
            row[name] = float(values[i]) if np.isfinite(values[i]) else np.nan
        rows.append(row)
    return rows


def summarize_indices(indices, features, hub_values, received_values):
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if len(indices) == 0:
        return None
    row = {"n_residues": int(len(indices))}
    for metric in SUMMARY_METRICS:
        values = features.get(metric)
        clean_metric = metric[3:] if metric.startswith("is_") else metric
        row[f"fraction_{clean_metric}"] = float(np.mean(values[indices])) if values is not None else np.nan
    for metric in VALUE_METRICS:
        values = received_values if metric == "received_attention" else features.get(metric)
        if values is None:
            row[f"mean_{metric}"] = np.nan
            continue
        vals = np.asarray(values, dtype=float)[indices]
        row[f"mean_{metric}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
    vals = np.asarray(hub_values, dtype=float)[indices]
    row["mean_hub_score"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
    return row


def top_indices(values, frac, eligible):
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    if len(eligible) == 0:
        return []
    values = np.asarray(values, dtype=float)
    usable = eligible[np.isfinite(values[eligible])]
    if len(usable) == 0:
        return []
    k = max(1, int(np.ceil(len(usable) * frac)))
    order = usable[np.argsort(values[usable])[::-1]]
    return [int(i) for i in order[:k]]


def pool_name(label):
    return "all_residues" if int(label) == -1 else mode_name(int(label))


def aggregate_enrichment(summary_df):
    if summary_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in summary_df.columns if c.startswith("obs_")]
    group_cols = ["condition", "hub_score", "pool", "selection"]
    rows = []
    for key, group in summary_df.groupby(group_cols, sort=False):
        row = {col: value for col, value in zip(group_cols, key)}
        row["n_units"] = len(group)
        for obs_col in metric_cols:
            metric = obs_col[len("obs_"):]
            bg_col = f"bg_{metric}"
            obs = group[obs_col].astype(float)
            bg = group[bg_col].astype(float) if bg_col in group.columns else pd.Series(np.nan, index=group.index)
            row[f"obs_{metric}_mean"] = float(np.nanmean(obs)) if np.any(np.isfinite(obs)) else np.nan
            row[f"bg_{metric}_mean"] = float(np.nanmean(bg)) if np.any(np.isfinite(bg)) else np.nan
            delta = obs - bg
            row[f"delta_{metric}_mean"] = float(np.nanmean(delta)) if np.any(np.isfinite(delta)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hub_stability(residue_df, hub_scores, pool_mode_labels, top_fracs):
    rows = []
    if residue_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = residue_df.groupby(["condition", "protein"], sort=False)
    for (condition, protein), group in grouped:
        seeds = sorted(group["seed"].dropna().astype(int).unique())
        if len(seeds) < 2:
            continue
        for score in hub_scores:
            if score not in group.columns:
                continue
            for pool_label in pool_mode_labels:
                pool = pool_name(pool_label)
                for frac in top_fracs:
                    selected = {}
                    for seed in seeds:
                        sub = group[group["seed"].astype(int) == seed]
                        if int(pool_label) != -1:
                            sub = sub[sub["mode_label"].astype(int) == int(pool_label)]
                        if sub.empty:
                            continue
                        k = max(1, int(np.ceil(len(sub) * float(frac))))
                        top = sub.sort_values(score, ascending=False).head(k)
                        selected[seed] = set(top["position_1based"].astype(int).tolist())
                    if len(selected) < 2:
                        continue
                    vals = []
                    for a, b in itertools.combinations(selected, 2):
                        union = selected[a] | selected[b]
                        vals.append(len(selected[a] & selected[b]) / len(union) if union else np.nan)
                    rows.append({
                        "condition": condition,
                        "protein": protein,
                        "hub_score": score,
                        "pool": pool,
                        "selection": f"top_{int(round(frac * 100))}pct",
                        "n_seed_pairs": len(vals),
                        "top_hub_jaccard_mean": float(np.nanmean(vals)) if vals else np.nan,
                    })
    by_protein = pd.DataFrame(rows)
    if by_protein.empty:
        return by_protein, pd.DataFrame()
    by_condition = (
        by_protein
        .groupby(["condition", "hub_score", "pool", "selection"], sort=False)["top_hub_jaccard_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "top_hub_jaccard_mean", "std": "top_hub_jaccard_std", "count": "n_proteins"})
    )
    return by_protein, by_condition


def default_row_mode_assignment_path(result_root, manifest_path):
    analysis_dir = default_analysis_dir(result_root, "analysis_row_modes", manifest_path)
    return analysis_dir / "row_mode_assignments_by_residue.csv"


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
        else default_analysis_dir(result_root, "analysis_attention_graph_hubs", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()

    neq_map = load_neq_by_name(args.test_csv)
    ss_map = load_q3_q8_map(args.ss_csv)
    pb_map = load_pb_entropy_map(args.pb_entropy_csv, args.pb_id_col, args.pb_entropy_col)
    site_map = load_site_annotation_map(
        args.site_csv,
        args.site_id_col,
        args.site_pos_col,
        args.site_label_col,
        args.site_position_base,
    )

    assignment_path = Path(args.row_mode_assignments).expanduser() if args.row_mode_assignments else default_row_mode_assignment_path(result_root, manifest_path)
    assignment_maps = load_assignment_maps(assignment_path, args.conditions) if assignment_path.exists() else {}

    rng = np.random.default_rng(args.random_seed)
    residue_rows = []
    enrichment_rows = []
    missing = []
    skipped = []
    processed_records = 0

    for run in manifest.itertuples(index=False):
        run_dict = run._asdict()
        attention_path = resolve_existing_path(run_dict["attention_json"], result_root, pipeline_dir)
        if not attention_path.exists():
            missing.append(str(attention_path))
            continue
        print(f"[load] {run_dict.get('condition')} seed={run_dict.get('seed')}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            if args.max_records and processed_records >= args.max_records:
                break
            protein = str(record["name"])
            sequence = str(record["sequence"])
            n = len(sequence)
            if protein not in neq_map:
                skipped.append(f"{run_dict.get('condition')} seed={run_dict.get('seed')} {protein}: missing Neq")
                continue
            ss_entry = q8_entry(ss_map, protein, n)
            if ss_entry is None:
                skipped.append(f"{run_dict.get('condition')} seed={run_dict.get('seed')} {protein}: missing/length-mismatched SS")
                continue
            neq_entry = neq_map[protein]
            neq = np.asarray(neq_entry["neq"][:n], dtype=float)
            if len(neq) != n:
                skipped.append(f"{run_dict.get('condition')} seed={run_dict.get('seed')} {protein}: length-mismatched Neq")
                continue

            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            q3, q8, q8_entropy = ss_entry
            features = feature_arrays(q3, q8, neq, q8_entropy, pb_array(pb_map, protein, n), args)
            features.update(site_feature_arrays(site_map, protein, n))

            key = (run_dict.get("condition"), int(run_dict.get("seed")), protein)
            assignment = assignment_maps.get(key)
            if assignment is not None and len(assignment["modes"]) >= n:
                modes = assignment["modes"][:n]
                ent = assignment["row_entropy"][:n]
            else:
                modes, ent, *_ = analyze_attention_modes(
                    attn,
                    args.high_entropy_quantile,
                    args.min_low_rows,
                    args.kmeans_seed,
                )
            metrics = graph_hub_metrics(attn, modes, args.topk_values, args.exclude_diagonal_window)
            metrics["row_entropy"] = ent

            residue_rows.extend(build_residue_rows(run_dict, protein, sequence, features, modes, metrics))

            for hub_score in args.hub_scores:
                if hub_score not in metrics:
                    continue
                values = metrics[hub_score]
                for pool_label in args.pool_mode_labels:
                    if int(pool_label) == -1:
                        eligible = np.arange(n)
                    else:
                        eligible = np.where(modes == int(pool_label))[0]
                    if len(eligible) == 0:
                        continue
                    for frac in args.top_fracs:
                        selected = top_indices(values, frac, eligible)
                        if not selected:
                            continue
                        bg = matched_background(selected, eligible, q3, rng, args.background_per_selected)
                        obs_summary = summarize_indices(selected, features, values, metrics["received_attention"])
                        bg_summary = summarize_indices(bg, features, values, metrics["received_attention"])
                        if obs_summary is None or bg_summary is None:
                            continue
                        row = {
                            "condition": run_dict.get("condition", ""),
                            "seed": int(run_dict.get("seed", -1)),
                            "protein": protein,
                            "hub_score": hub_score,
                            "pool": pool_name(pool_label),
                            "pool_mode_label": int(pool_label),
                            "selection": f"top_{int(round(frac * 100))}pct",
                        }
                        row.update({f"obs_{k}": v for k, v in obs_summary.items()})
                        row.update({f"bg_{k}": v for k, v in bg_summary.items()})
                        enrichment_rows.append(row)
            processed_records += 1
        if args.max_records and processed_records >= args.max_records:
            break

    residue_df = pd.DataFrame(residue_rows)
    enrichment_df = pd.DataFrame(enrichment_rows)
    enrichment_by_condition = aggregate_enrichment(enrichment_df)
    stability_by_protein, stability_by_condition = hub_stability(
        residue_df,
        [score for score in args.hub_scores if score in residue_df.columns],
        args.pool_mode_labels,
        args.top_fracs,
    )

    top_hub_rows = []
    if not residue_df.empty:
        group_cols = ["condition", "seed", "protein"]
        for score in args.hub_scores:
            if score not in residue_df.columns:
                continue
            cols = [
                "condition", "seed", "protein", "position_1based", "aa", "mode", "mode_label",
                "q3", "q8", "neq", "q8_entropy", "pb_entropy", score,
            ]
            cols += [c for c in SUMMARY_METRICS if c in residue_df.columns]
            top = (
                residue_df.sort_values(group_cols + [score], ascending=[True, True, True, False])
                .groupby(group_cols, as_index=False)
                .head(args.top_hubs_per_protein)
            )
            top = top[[c for c in cols if c in top.columns]].copy()
            top.insert(3, "hub_score", score)
            top_hub_rows.append(top)
    top_hubs = pd.concat(top_hub_rows, ignore_index=True) if top_hub_rows else pd.DataFrame()

    residue_df.to_csv(output_dir / "attention_graph_hub_residue_metrics.csv", index=False)
    enrichment_df.to_csv(output_dir / "attention_graph_hub_enrichment_by_run.csv", index=False)
    enrichment_by_condition.to_csv(output_dir / "attention_graph_hub_enrichment_by_condition.csv", index=False)
    stability_by_protein.to_csv(output_dir / "attention_graph_hub_stability_by_protein.csv", index=False)
    stability_by_condition.to_csv(output_dir / "attention_graph_hub_stability_by_condition.csv", index=False)
    top_hubs.to_csv(output_dir / "attention_graph_top_hubs.csv", index=False)

    if missing:
        (output_dir / "missing_attention_files.txt").write_text("\n".join(missing) + "\n")
    if skipped:
        (output_dir / "skipped_records.txt").write_text("\n".join(skipped) + "\n")

    summary_lines = [
        f"Result root: {result_root}",
        f"Manifest: {manifest_path}",
        f"Output dir: {output_dir}",
        f"Runs in manifest after filtering: {len(manifest)}",
        f"Protein/run attention records analyzed: {processed_records}",
        f"Residue rows: {len(residue_df)}",
        f"Hub enrichment rows: {len(enrichment_df)}",
        f"Top-k graph definitions: {args.topk_values}",
        f"Diagonal exclusion window: {args.exclude_diagonal_window}",
        f"Hub scores: {', '.join(args.hub_scores)}",
        f"Hub pools: {', '.join(pool_name(x) for x in args.pool_mode_labels)}",
        f"Row-mode assignments loaded: {len(assignment_maps)}",
        f"PB entropy available: {bool(pb_map)}",
        f"Site annotations available: {bool(site_map)}",
        "",
        "Key outputs:",
        "  attention_graph_hub_residue_metrics.csv",
        "  attention_graph_top_hubs.csv",
        "  attention_graph_hub_enrichment_by_run.csv",
        "  attention_graph_hub_enrichment_by_condition.csv",
        "  attention_graph_hub_stability_by_protein.csv",
        "  attention_graph_hub_stability_by_condition.csv",
    ]
    (output_dir / "attention_graph_hub_analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
