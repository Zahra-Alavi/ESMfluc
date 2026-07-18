#!/usr/bin/env python3
"""
Relate received attention to ATLAS local strain while controlling terminal artifacts.

The strain data are expected under:
  atlas_strain_analysis/<protein>/strain_summary.csv

strain_summary.csv must contain:
  residue,R1_mean,R2_mean,R3_mean,ensemble_mean,ensemble_std

The analysis excludes terminal residues before correlations and matched
background tests, because local strain is inflated at unconstrained chain ends.
"""

import argparse
import ast
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


def parse_args():
    p = argparse.ArgumentParser(description="Attention-vs-ATLAS-local-strain analysis.")
    p.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    p.add_argument("--manifest_tsv", default=None, help="Use manifest_attention_sources.tsv for the 30-source view.")
    p.add_argument("--atlas_root", required=True, help="atlas_strain_analysis directory.")
    p.add_argument("--output_dir", default=None, help="Default: result_root/analysis_attention_strain[_suffix].")
    p.add_argument("--pipeline_dir", default=None, help="Directory for resolving relative manifest paths.")
    p.add_argument("--test_csv", default=None, help="Optional CSV with name, sequence, neq.")
    p.add_argument("--ss_csv", default=None, help="Optional NetSurfP CSV with id and q3/q8 labels.")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--terminal_exclusion", type=int, default=10, help="Drop this many residues at each terminus.")
    p.add_argument(
        "--terminal_exclusion_fraction",
        type=float,
        default=0.0,
        help="Also drop this fraction of residues at each terminus. Effective exclusion is max(count, fraction*n).",
    )
    p.add_argument("--top_fracs", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument(
        "--attention_scores",
        nargs="+",
        default=["received_attention", "max_received_attention", "in_degree_top1", "weighted_in_degree_top1"],
    )
    p.add_argument("--background_per_selected", type=int, default=5)
    p.add_argument("--background_scope", choices=["same_q3", "same_q3_mode", "eligible"], default="same_q3_mode")
    p.add_argument("--high_strain_quantile", type=float, default=0.90)
    p.add_argument("--high_entropy_quantile", type=float, default=0.67)
    p.add_argument("--min_low_rows", type=int, default=8)
    p.add_argument("--kmeans_seed", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--max_records", type=int, default=0)
    return p.parse_args()


def stripped_columns(df):
    return {c.strip(): c for c in df.columns}


def load_neq_map(test_csv):
    if not test_csv or not Path(test_csv).expanduser().exists():
        return {}
    df = pd.read_csv(Path(test_csv).expanduser())
    out = {}
    if "name" not in df.columns or "neq" not in df.columns:
        return out
    for _, row in df.iterrows():
        out[str(row["name"])] = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
    return out


def load_ss_map(ss_csv):
    if not ss_csv or not Path(ss_csv).expanduser().exists():
        return {}
    df = pd.read_csv(Path(ss_csv).expanduser())
    columns = stripped_columns(df)
    id_col = columns.get("id")
    q3_col = columns.get("q3")
    q8_col = columns.get("q8")
    if id_col is None or q3_col is None:
        raise ValueError(f"{ss_csv} must contain id and q3 columns.")
    out = {}
    for _, row in df.iterrows():
        protein = str(row[id_col]).lstrip(">")
        out.setdefault(protein, {"q3": [], "q8": []})
        out[protein]["q3"].append(str(row[q3_col]).strip())
        out[protein]["q8"].append(str(row[q8_col]).strip() if q8_col is not None else "")
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


def terminal_mask(n, count, fraction):
    exclusion = max(int(count), int(np.ceil(float(fraction) * n)))
    mask = np.ones(n, dtype=bool)
    if exclusion > 0:
        mask[:exclusion] = False
        mask[max(0, n - exclusion):] = False
    return mask, exclusion


def load_strain_summary(atlas_root, protein, n):
    path = Path(atlas_root) / protein / "strain_summary.csv"
    if not path.exists():
        return None, str(path)
    df = pd.read_csv(path)
    required = {"residue", "ensemble_mean", "ensemble_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.sort_values("residue")
    strain = np.full(n, np.nan)
    strain_std = np.full(n, np.nan)
    for _, row in df.iterrows():
        idx = int(row["residue"]) - 1
        if 0 <= idx < n:
            strain[idx] = float(row["ensemble_mean"])
            strain_std[idx] = float(row["ensemble_std"])
    return {"strain_mean": strain, "strain_std": strain_std}, str(path)


def attention_metrics(attn):
    n = attn.shape[0]
    work = np.asarray(attn, dtype=float).copy()
    np.fill_diagonal(work, np.nan)
    received = np.nansum(work, axis=0)
    max_received = np.nanmax(work, axis=0)
    in_degree = np.zeros(n, dtype=float)
    weighted = np.zeros(n, dtype=float)
    for i in range(n):
        row = work[i]
        if not np.any(np.isfinite(row)):
            continue
        j = int(np.nanargmax(row))
        w = float(row[j])
        if np.isfinite(w):
            in_degree[j] += 1.0
            weighted[j] += w
    return {
        "received_attention": received,
        "max_received_attention": max_received,
        "in_degree_top1": in_degree,
        "weighted_in_degree_top1": weighted,
    }


def get_labels(ss_map, protein, n):
    entry = ss_map.get(protein)
    if entry is None:
        return np.full(n, "", dtype=object), np.full(n, "", dtype=object)
    q3 = np.asarray(entry["q3"][:n], dtype=object)
    q8 = np.asarray(entry["q8"][:n], dtype=object)
    if len(q3) != n:
        q3 = np.full(n, "", dtype=object)
    if len(q8) != n:
        q8 = np.full(n, "", dtype=object)
    return q3, q8


def spearman(x, y):
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    return float(df["x"].corr(df["y"], method="spearman"))


def high_mask(values, eligible, quantile):
    values = np.asarray(values, dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    eligible = np.asarray(eligible, dtype=int)
    valid = eligible[np.isfinite(values[eligible])]
    if len(valid) == 0:
        return mask
    threshold = float(np.quantile(values[valid], quantile))
    mask[valid] = values[valid] >= threshold
    return mask


def top_indices(values, eligible, frac):
    values = np.asarray(values, dtype=float)
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    valid = eligible[np.isfinite(values[eligible])]
    if len(valid) == 0:
        return []
    k = max(1, int(np.ceil(len(valid) * float(frac))))
    return [int(i) for i in valid[np.argsort(values[valid])[::-1]][:k]]


def matched_background(selected, eligible, q3, modes, rng, n_per_selected, scope):
    selected = sorted(set(int(i) for i in selected))
    selected_set = set(selected)
    eligible = np.asarray(sorted(set(int(i) for i in eligible)), dtype=int)
    sampled = []
    if not selected or len(eligible) == 0:
        return sampled
    for idx in selected:
        candidates = eligible
        if scope in {"same_q3", "same_q3_mode"} and q3[idx] != "":
            candidates = candidates[q3[candidates] == q3[idx]]
        if scope == "same_q3_mode":
            candidates = candidates[modes[candidates] == modes[idx]]
        candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([c for c in eligible if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = eligible
        sampled.extend(int(x) for x in rng.choice(candidates, size=n_per_selected, replace=len(candidates) < n_per_selected))
    return sampled


def summarize(indices, score, strain, strain_std, high_strain):
    indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if len(indices) == 0:
        return None
    out = {"n_residues": int(len(indices))}
    for name, values in [
        ("score", score),
        ("strain", strain),
        ("strain_std", strain_std),
    ]:
        vals = np.asarray(values, dtype=float)[indices]
        out[f"mean_{name}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
        out[f"median_{name}"] = float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan
    out["fraction_high_strain"] = float(np.mean(high_strain[indices]))
    return out


def aggregate_summary(df):
    if df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in df.columns if c.startswith("obs_")]
    rows = []
    for key, group in df.groupby(["condition", "attention_score", "pool", "selection"], sort=False):
        row = dict(zip(["condition", "attention_score", "pool", "selection"], key))
        row["n_units"] = len(group)
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
            row[f"obs_{metric}_mean"] = float(obs[valid].mean())
            row[f"bg_{metric}_mean"] = float(bg[valid].mean())
            row[f"delta_{metric}_mean"] = float((obs[valid] - bg[valid]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parents[2]
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_analysis_dir(result_root, "analysis_attention_strain", manifest_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()

    ss_map = load_ss_map(args.ss_csv)
    neq_map = load_neq_map(args.test_csv)
    rng = np.random.default_rng(args.random_seed)

    per_residue = []
    corr_rows = []
    enrich_rows = []
    missing = []
    processed = 0

    for run in manifest.itertuples(index=False):
        run_dict = run._asdict()
        attention_path = resolve_attention_path(run_dict["attention_json"], result_root, pipeline_dir)
        if not attention_path.exists():
            missing.append(str(attention_path))
            continue
        print(f"[load] {run_dict.get('condition')} seed={run_dict.get('seed')}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            if args.max_records and processed >= args.max_records:
                break
            protein = str(record["name"])
            sequence = str(record["sequence"])
            n = len(sequence)
            strain_data, strain_path = load_strain_summary(args.atlas_root, protein, n)
            if strain_data is None:
                missing.append(strain_path)
                continue
            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            modes, row_ent, *_ = analyze_attention_modes(attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed)
            q3, q8 = get_labels(ss_map, protein, n)
            neq = neq_map.get(protein, np.full(n, np.nan))[:n]
            if len(neq) != n:
                neq = np.full(n, np.nan)
            metrics = attention_metrics(attn)
            strain = strain_data["strain_mean"]
            strain_std = strain_data["strain_std"]
            nonterminal, exclusion = terminal_mask(n, args.terminal_exclusion, args.terminal_exclusion_fraction)
            valid = nonterminal & np.isfinite(strain)
            eligible = np.where(valid)[0]
            high_strain = high_mask(strain, eligible, args.high_strain_quantile)

            for score_name, score in metrics.items():
                if score_name not in args.attention_scores:
                    continue
                corr_rows.append({
                    "condition": run_dict.get("condition", ""),
                    "seed": int(run_dict.get("seed", -1)),
                    "protein": protein,
                    "attention_score": score_name,
                    "n_residues": n,
                    "n_eligible_nonterminal": int(valid.sum()),
                    "terminal_exclusion_used": exclusion,
                    "spearman_attention_vs_strain": spearman(score[valid], strain[valid]),
                    "spearman_attention_vs_strain_std": spearman(score[valid], strain_std[valid]),
                    "spearman_attention_vs_neq": spearman(score[valid], neq[valid]),
                    "spearman_strain_vs_neq": spearman(strain[valid], neq[valid]),
                })
                pools = {"all_nonterminal": eligible}
                for label in [1, 2]:
                    pools[mode_name(label)] = np.asarray([i for i in eligible if modes[i] == label], dtype=int)
                for pool, pool_idx in pools.items():
                    if len(pool_idx) == 0:
                        continue
                    for frac in args.top_fracs:
                        selected = top_indices(score, pool_idx, frac)
                        bg = matched_background(
                            selected, pool_idx, q3, modes, rng,
                            args.background_per_selected, args.background_scope,
                        )
                        obs = summarize(selected, score, strain, strain_std, high_strain)
                        bgs = summarize(bg, score, strain, strain_std, high_strain)
                        if obs is None or bgs is None:
                            continue
                        row = {
                            "condition": run_dict.get("condition", ""),
                            "seed": int(run_dict.get("seed", -1)),
                            "protein": protein,
                            "attention_score": score_name,
                            "pool": pool,
                            "selection": f"top_{int(round(frac * 100))}pct",
                        }
                        row.update({f"obs_{k}": v for k, v in obs.items()})
                        row.update({f"bg_{k}": v for k, v in bgs.items()})
                        enrich_rows.append(row)

            for i, aa in enumerate(sequence):
                if not valid[i]:
                    continue
                row = {
                    "condition": run_dict.get("condition", ""),
                    "seed": int(run_dict.get("seed", -1)),
                    "protein": protein,
                    "position_1based": i + 1,
                    "aa": aa,
                    "mode": mode_name(int(modes[i])),
                    "mode_label": int(modes[i]),
                    "q3": q3[i],
                    "q8": q8[i],
                    "neq": float(neq[i]) if np.isfinite(neq[i]) else np.nan,
                    "row_entropy": float(row_ent[i]),
                    "strain_mean": float(strain[i]),
                    "strain_std": float(strain_std[i]) if np.isfinite(strain_std[i]) else np.nan,
                    "is_high_strain": bool(high_strain[i]),
                    "terminal_exclusion_used": exclusion,
                }
                row.update({k: float(v[i]) if np.isfinite(v[i]) else np.nan for k, v in metrics.items()})
                per_residue.append(row)
            processed += 1
        if args.max_records and processed >= args.max_records:
            break

    per_residue_df = pd.DataFrame(per_residue)
    corr_df = pd.DataFrame(corr_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    by_condition = (
        corr_df.groupby(["condition", "attention_score"], sort=False)
        .agg(
            n_units=("protein", "size"),
            spearman_attention_vs_strain_mean=("spearman_attention_vs_strain", "mean"),
            spearman_attention_vs_strain_std=("spearman_attention_vs_strain", "std"),
            spearman_attention_vs_neq_mean=("spearman_attention_vs_neq", "mean"),
            spearman_strain_vs_neq_mean=("spearman_strain_vs_neq", "mean"),
        )
        .reset_index()
        if not corr_df.empty else pd.DataFrame()
    )
    enrich_by_condition = aggregate_summary(enrich_df)

    per_residue_df.to_csv(output_dir / "attention_strain_per_residue.csv", index=False)
    corr_df.to_csv(output_dir / "attention_strain_correlations_by_run.csv", index=False)
    by_condition.to_csv(output_dir / "attention_strain_correlations_by_condition.csv", index=False)
    enrich_df.to_csv(output_dir / "attention_strain_top_attention_enrichment_by_run.csv", index=False)
    enrich_by_condition.to_csv(output_dir / "attention_strain_top_attention_enrichment_by_condition.csv", index=False)
    if missing:
        (output_dir / "missing_inputs.txt").write_text("\n".join(missing) + "\n")

    lines = [
        f"Result root: {result_root}",
        f"Atlas root: {Path(args.atlas_root).expanduser().resolve()}",
        f"Manifest: {manifest_path}",
        f"Output dir: {output_dir}",
        f"Runs in manifest after filtering: {len(manifest)}",
        f"Protein/run records analyzed: {processed}",
        f"Per-residue rows after terminal exclusion: {len(per_residue_df)}",
        f"Correlation rows: {len(corr_df)}",
        f"Enrichment rows: {len(enrich_df)}",
        f"Terminal exclusion count: {args.terminal_exclusion}",
        f"Terminal exclusion fraction: {args.terminal_exclusion_fraction}",
        f"Background scope: {args.background_scope}",
        "",
        "Key outputs:",
        "  attention_strain_per_residue.csv",
        "  attention_strain_correlations_by_run.csv",
        "  attention_strain_correlations_by_condition.csv",
        "  attention_strain_top_attention_enrichment_by_run.csv",
        "  attention_strain_top_attention_enrichment_by_condition.csv",
    ]
    (output_dir / "attention_strain_analysis_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
