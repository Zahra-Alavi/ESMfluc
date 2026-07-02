#!/usr/bin/env python3
"""
Test whether row-mode assignments predict flexibility after controlling for
secondary structure.

Core question:
  Are row modes merely a re-labeling of C/H/E, or do they add information about
  Neq/flexibility within secondary-structure classes?

The script reconstructs row modes from attention.json files, builds a
residue-level table, then compares protein-held-out logistic models:
  1. SS only
  2. SS + row mode
  3. SS + row mode + SS:mode interactions

It also computes within-SS flexibility rates by mode and an optional matched
null that shuffles row modes within each protein x SS stratum.

Example:
  python analyze_row_modes_flexibility_control_ss.py \
    --result_root results/publication_comparable_v1 \
    --test_csv ../../data/test_data_with_names.csv \
    --ss_csv ../../data/test_data_nsp3.csv \
    --n_permutations 50
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from analyze_attention_row_modes import analyze_attention_modes, resolve_existing_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask whether row modes predict Neq/flexibility after controlling for C/H/E."
    )
    parser.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    parser.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    parser.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id and q3 columns.")
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_row_modes_controlled_ss.")
    parser.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional subset of conditions.")
    parser.add_argument(
        "--row_mode_assignments",
        default=None,
        help="Optional residue assignment CSV from analyze_attention_row_modes.py. "
             "Default: result_root/analysis_row_modes/row_mode_assignments_by_residue.csv if present.",
    )
    parser.add_argument("--high_entropy_quantile", type=float, default=0.67)
    parser.add_argument("--min_low_rows", type=int, default=8)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5, help="Protein-grouped CV folds. Default: 5.")
    parser.add_argument("--n_permutations", type=int, default=50, help="Matched null permutations. Default: 50.")
    parser.add_argument("--permutation_seed", type=int, default=123)
    parser.add_argument("--max_rows_per_condition", type=int, default=0,
                        help="Optional residue subsample per condition for speed. 0 means use all.")
    parser.add_argument("--save_residue_table", action="store_true",
                        help="Save full residue-level table. It can be large.")
    return parser.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def load_neq_by_name(test_csv):
    df = pd.read_csv(test_csv)
    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]) if "name" in df.columns else str(row["sequence"])
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[name] = {
            "sequence": row["sequence"],
            "neq": neq,
            "flexible": classify_neq(neq),
        }
    return out


def load_ss_map(ss_csv):
    df = pd.read_csv(ss_csv)
    columns = {c.strip(): c for c in df.columns}
    id_col = columns.get("id")
    q3_col = columns.get("q3")
    if id_col is None or q3_col is None:
        raise ValueError(f"{ss_csv} must contain id and q3 columns.")
    ss_map = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_col]).lstrip(">")
        ss_map.setdefault(seq_id, []).append(str(row[q3_col]).strip())
    return ss_map


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(int(label), str(label))


def build_residue_table(args, result_root, pipeline_dir):
    assignment_path = (
        Path(args.row_mode_assignments)
        if args.row_mode_assignments
        else result_root / "analysis_row_modes" / "row_mode_assignments_by_residue.csv"
    )
    if assignment_path.exists():
        print(f"[load] row-mode assignments: {assignment_path}")
        df = pd.read_csv(assignment_path)
        required = {
            "condition", "seed", "protein", "position_1based", "aa",
            "ss", "mode_label", "mode", "row_entropy", "neq", "flexible",
        }
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"{assignment_path} is missing required columns: {sorted(missing_cols)}"
            )
        if args.conditions:
            df = df[df["condition"].isin(args.conditions)].copy()
        df = df[df["ss"].isin(["C", "H", "E"])].copy()
        df["seed"] = df["seed"].astype(int)
        df["mode_label"] = df["mode_label"].astype(int)
        df["neq"] = df["neq"].astype(float)
        df["flexible"] = df["flexible"].astype(int)
        return df

    manifest = pd.read_csv(result_root / "manifest.tsv", sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()

    neq_by_name = load_neq_by_name(args.test_csv)
    ss_map = load_ss_map(args.ss_csv)
    rows = []
    missing = []

    for run in manifest.itertuples(index=False):
        attention_path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            missing.append(str(attention_path))
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            protein = record["name"]
            if protein not in neq_by_name or protein not in ss_map:
                continue
            sequence = record["sequence"]
            n = len(sequence)
            neq = neq_by_name[protein]["neq"][:n]
            flexible = neq_by_name[protein]["flexible"][:n]
            ss = np.asarray(ss_map[protein][:n])
            if len(neq) != n or len(ss) != n:
                continue
            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            modes, ent, _, _, _, _ = analyze_attention_modes(
                attn,
                args.high_entropy_quantile,
                args.min_low_rows,
                args.kmeans_seed,
            )
            for i in range(n):
                if ss[i] not in {"C", "H", "E"}:
                    continue
                rows.append({
                    "condition": run.condition,
                    "seed": int(run.seed),
                    "protein": protein,
                    "position_1based": i + 1,
                    "aa": sequence[i],
                    "ss": ss[i],
                    "mode_label": int(modes[i]),
                    "mode": mode_name(modes[i]),
                    "row_entropy": float(ent[i]),
                    "neq": float(neq[i]),
                    "flexible": int(flexible[i]),
                })

    if missing:
        print(f"[warn] Missing {len(missing)} attention files.")
    if not rows:
        raise RuntimeError("No residue rows built. Check paths and inputs.")
    return pd.DataFrame(rows)


def make_features(df, feature_set):
    if feature_set == "ss":
        cols = ["ss"]
    elif feature_set == "ss_mode":
        cols = ["ss", "mode"]
    elif feature_set == "ss_mode_interaction":
        tmp = df.copy()
        tmp["ss_mode"] = tmp["ss"] + "__" + tmp["mode"]
        cols = ["ss", "mode", "ss_mode"]
        df = tmp
    else:
        raise ValueError(feature_set)

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    x = encoder.fit_transform(df[cols])
    return x, encoder.get_feature_names_out(cols)


def evaluate_feature_set(df, feature_set, n_splits):
    y = df["flexible"].to_numpy(dtype=int)
    groups = df["protein"].to_numpy()
    x, feature_names = make_features(df, feature_set)

    unique_groups = np.unique(groups)
    splits = min(n_splits, len(unique_groups))
    if splits < 2:
        raise ValueError("Need at least 2 proteins for grouped CV.")
    cv = GroupKFold(n_splits=splits)

    rows = []
    coef_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        model.fit(x[train_idx], y[train_idx])
        prob = model.predict_proba(x[test_idx])[:, 1]
        rows.append({
            "feature_set": feature_set,
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_test_proteins": int(len(np.unique(groups[test_idx]))),
            "auroc": roc_auc_score(y[test_idx], prob),
            "auprc": average_precision_score(y[test_idx], prob),
            "log_loss": log_loss(y[test_idx], prob, labels=[0, 1]),
            "brier": brier_score_loss(y[test_idx], prob),
        })
        for name, coef in zip(feature_names, model.coef_[0]):
            coef_rows.append({
                "feature_set": feature_set,
                "fold": fold,
                "feature": name,
                "coef": float(coef),
            })
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def compare_models(cv_metrics):
    wide = cv_metrics.pivot_table(index="fold", columns="feature_set", values=["auroc", "auprc", "log_loss", "brier"])
    rows = []
    comparisons = [
        ("ss_mode", "ss"),
        ("ss_mode_interaction", "ss"),
        ("ss_mode_interaction", "ss_mode"),
    ]
    for richer, simpler in comparisons:
        row = {"comparison": f"{richer} minus {simpler}"}
        for metric in ["auroc", "auprc"]:
            if (metric, richer) in wide and (metric, simpler) in wide:
                diff = wide[(metric, richer)] - wide[(metric, simpler)]
                row[f"delta_{metric}_mean"] = float(diff.mean())
                row[f"delta_{metric}_std"] = float(diff.std())
        for metric in ["log_loss", "brier"]:
            if (metric, richer) in wide and (metric, simpler) in wide:
                diff = wide[(metric, simpler)] - wide[(metric, richer)]
                row[f"improvement_{metric}_mean"] = float(diff.mean())
                row[f"improvement_{metric}_std"] = float(diff.std())
        rows.append(row)
    return pd.DataFrame(rows)


def within_ss_mode_table(df):
    rows = []
    for (condition, ss), sub in df.groupby(["condition", "ss"]):
        ss_rate = sub["flexible"].mean()
        for mode, mode_sub in sub.groupby("mode"):
            rows.append({
                "condition": condition,
                "ss": ss,
                "mode": mode,
                "n": int(len(mode_sub)),
                "flexible_rate": float(mode_sub["flexible"].mean()),
                "ss_background_flexible_rate": float(ss_rate),
                "delta_vs_ss_background": float(mode_sub["flexible"].mean() - ss_rate),
                "ratio_vs_ss_background": float(mode_sub["flexible"].mean() / ss_rate) if ss_rate > 0 else np.nan,
                "mean_neq": float(mode_sub["neq"].mean()),
                "mean_row_entropy": float(mode_sub["row_entropy"].mean()),
            })
    return pd.DataFrame(rows)


def shuffle_modes_within_protein_ss(df, rng):
    shuffled = df.copy()
    shuffled["mode"] = shuffled["mode"].to_numpy()
    shuffled["mode_label"] = shuffled["mode_label"].to_numpy()
    for _, idx in shuffled.groupby(["protein", "ss"]).groups.items():
        idx = np.asarray(list(idx))
        mode_vals = shuffled.loc[idx, "mode"].to_numpy().copy()
        label_vals = shuffled.loc[idx, "mode_label"].to_numpy().copy()
        perm = rng.permutation(len(idx))
        shuffled.loc[idx, "mode"] = mode_vals[perm]
        shuffled.loc[idx, "mode_label"] = label_vals[perm]
    return shuffled


def run_condition_analysis(df, condition, args, rng):
    cond_df = df[df["condition"] == condition].copy()
    if args.max_rows_per_condition and len(cond_df) > args.max_rows_per_condition:
        cond_df = cond_df.sample(args.max_rows_per_condition, random_state=args.permutation_seed)

    cv_parts = []
    coef_parts = []
    for feature_set in ["ss", "ss_mode", "ss_mode_interaction"]:
        cv, coefs = evaluate_feature_set(cond_df, feature_set, args.n_splits)
        cv_parts.append(cv)
        coef_parts.append(coefs)
    cv_metrics = pd.concat(cv_parts, ignore_index=True)
    coef_df = pd.concat(coef_parts, ignore_index=True)
    comparison = compare_models(cv_metrics)
    comparison.insert(0, "condition", condition)
    cv_metrics.insert(0, "condition", condition)
    coef_df.insert(0, "condition", condition)

    null_rows = []
    observed = comparison.set_index("comparison").to_dict(orient="index")
    for perm_idx in range(1, args.n_permutations + 1):
        shuffled = shuffle_modes_within_protein_ss(cond_df, rng)
        null_cv_parts = []
        for feature_set in ["ss", "ss_mode", "ss_mode_interaction"]:
            cv, _ = evaluate_feature_set(shuffled, feature_set, args.n_splits)
            null_cv_parts.append(cv)
        null_comparison = compare_models(pd.concat(null_cv_parts, ignore_index=True))
        for row in null_comparison.to_dict(orient="records"):
            row["condition"] = condition
            row["permutation"] = perm_idx
            null_rows.append(row)

    null_df = pd.DataFrame(null_rows)
    p_rows = []
    if not null_df.empty:
        for comp_name, obs in observed.items():
            sub = null_df[null_df["comparison"] == comp_name]
            row = {"condition": condition, "comparison": comp_name}
            for metric in [
                "delta_auroc_mean",
                "delta_auprc_mean",
                "improvement_log_loss_mean",
                "improvement_brier_mean",
            ]:
                if metric in obs and metric in sub:
                    obs_val = obs[metric]
                    null_vals = sub[metric].dropna().to_numpy()
                    if len(null_vals):
                        row[f"observed_{metric}"] = obs_val
                        row[f"null_{metric}_mean"] = float(np.mean(null_vals))
                        row[f"empirical_p_greater_{metric}"] = float((1 + np.sum(null_vals >= obs_val)) / (len(null_vals) + 1))
            p_rows.append(row)
    p_df = pd.DataFrame(p_rows)
    return cv_metrics, comparison, coef_df, null_df, p_df


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_root / "analysis_row_modes_controlled_ss"
    output_dir.mkdir(parents=True, exist_ok=True)

    residues = build_residue_table(args, result_root, pipeline_dir)
    conditions = sorted(residues["condition"].unique())
    rng = np.random.default_rng(args.permutation_seed)

    within_ss = within_ss_mode_table(residues)
    all_cv = []
    all_comparisons = []
    all_coefs = []
    all_nulls = []
    all_p = []

    for condition in conditions:
        print(f"[analyze] {condition}")
        cv, comparison, coefs, nulls, pvals = run_condition_analysis(residues, condition, args, rng)
        all_cv.append(cv)
        all_comparisons.append(comparison)
        all_coefs.append(coefs)
        all_nulls.append(nulls)
        all_p.append(pvals)

    cv_df = pd.concat(all_cv, ignore_index=True)
    comparison_df = pd.concat(all_comparisons, ignore_index=True)
    coef_df = pd.concat(all_coefs, ignore_index=True)
    null_df = pd.concat(all_nulls, ignore_index=True) if all_nulls else pd.DataFrame()
    p_df = pd.concat(all_p, ignore_index=True) if all_p else pd.DataFrame()

    if args.save_residue_table:
        residues.to_csv(output_dir / "row_mode_residue_table.csv", index=False)
    else:
        # Compact audit sample without writing the full large table.
        residues.head(1000).to_csv(output_dir / "row_mode_residue_table_head1000.csv", index=False)

    within_ss.to_csv(output_dir / "within_ss_flexibility_by_mode.csv", index=False)
    cv_df.to_csv(output_dir / "controlled_ss_cv_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "controlled_ss_model_comparisons.csv", index=False)
    coef_df.to_csv(output_dir / "controlled_ss_logistic_coefficients.csv", index=False)
    null_df.to_csv(output_dir / "matched_mode_shuffle_null.csv", index=False)
    p_df.to_csv(output_dir / "matched_mode_shuffle_empirical_pvalues.csv", index=False)

    summary_lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Residue rows analyzed: {len(residues)}",
        f"Conditions: {', '.join(conditions)}",
        f"Grouped CV folds: {args.n_splits}",
        f"Matched null permutations: {args.n_permutations}",
        "",
        "Key outputs:",
        "  within_ss_flexibility_by_mode.csv",
        "  controlled_ss_cv_metrics.csv",
        "  controlled_ss_model_comparisons.csv",
        "  matched_mode_shuffle_empirical_pvalues.csv",
        "  controlled_ss_logistic_coefficients.csv",
    ]
    (output_dir / "controlled_ss_analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
