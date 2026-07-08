#!/usr/bin/env python3
"""
Causal ablation tests for attention hubs in ESMfluc BiLSTM-attention models.

For each model run/protein, the script:
  1. Defines hub residues from the saved BiLSTM attention matrix.
  2. Builds matched random residue controls from the same Q3/Q8/Neq-bin/mode
     stratum when possible.
  3. Re-runs inference with selected residues intervened on:
       - zero_states: zero selected BiLSTM residue states before self-attention
       - remove_attention_columns: remove selected key columns from attention
         softmax and renormalize
  4. Reports performance drops relative to unperturbed inference.

It also writes a residualized Neq target table. Training residual-Neq models is
a separate step; if you later pass their attention manifest with
--residual_manifest_tsv, this script compares whether original hubs recur in
the residual-trained attention maps.
"""

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from analyze_attention_row_modes import (
    analyze_attention_modes,
    default_analysis_dir,
    resolve_existing_path,
    resolve_manifest_path,
)
from analyze_low_mode_control_points import load_q3_q8_map
from data_utils import create_classification_func
from models import BiLSTMWithSelfAttentionModel, run_masked_lstm
from train import load_esm_model, load_esm_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Ablate attention hubs at inference.")
    p.add_argument("--result_root", required=True)
    p.add_argument("--manifest_tsv", default=None)
    p.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    p.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id/q3/q8.")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pipeline_dir", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--neq_thresholds", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bidirectional", type=int, choices=[0, 1], default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top_fracs", nargs="+", type=float, default=[0.05, 0.10])
    p.add_argument("--pool_modes", nargs="+", default=["all", "low_mode_1", "low_mode_2"])
    p.add_argument(
        "--interventions",
        nargs="+",
        default=["zero_states", "remove_attention_columns"],
        choices=["zero_states", "remove_attention_columns"],
    )
    p.add_argument("--controls_per_protein", type=int, default=5)
    p.add_argument("--neq_bins", type=int, default=5)
    p.add_argument("--high_entropy_quantile", type=float, default=0.67)
    p.add_argument("--min_low_rows", type=int, default=8)
    p.add_argument("--kmeans_seed", type=int, default=0)
    p.add_argument("--random_seed", type=int, default=123)
    p.add_argument("--max_proteins", type=int, default=0)
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--residual_manifest_tsv", default=None)
    p.add_argument("--pb_entropy_csv", default=None)
    p.add_argument("--pb_entropy_col", default=None)
    p.add_argument("--pb_id_col", default="id")
    return p.parse_args()


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def load_test_rows(test_csv, classify):
    df = pd.read_csv(test_csv)
    rows = {}
    for idx, row in df.iterrows():
        name = str(row["name"]) if "name" in df.columns else str(idx)
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        sequence = str(row["sequence"])
        rows[name] = {
            "name": name,
            "sequence": sequence,
            "neq": neq,
            "labels": np.asarray([classify(v) for v in neq], dtype=int),
        }
    return rows


def load_attention_records(path):
    records = json.loads(Path(path).read_text())
    out = {}
    for record in records:
        if "name" in record and "attention_weights" in record:
            out[str(record["name"])] = record
    return out


def infer_bilstm_params(checkpoint):
    hidden_size = None
    num_layers = 0
    for key, value in checkpoint.items():
        if key.startswith("lstm.weight_ih_l") and "_reverse" not in key:
            layer_idx = int(key.split("lstm.weight_ih_l", 1)[1])
            num_layers = max(num_layers, layer_idx + 1)
            if layer_idx == 0:
                hidden_size = int(value.shape[0] // 4)
    if hidden_size is None or num_layers <= 0:
        raise ValueError("Could not infer BiLSTM hidden_size/num_layers from checkpoint.")
    return hidden_size, num_layers


def build_model(run, checkpoint_path, device, args):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_size, num_layers = infer_bilstm_params(checkpoint)
    embedding_model, _ = load_esm_model(str(run.esm_model), device=str(device))
    model = BiLSTMWithSelfAttentionModel(
        embedding_model=embedding_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def tokenize_sequence(tokenizer, sequence, device):
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    if "input_ids" in enc:
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
    else:
        input_ids = enc["sequence_tokens"].to(device)
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    return input_ids, attn_mask


def self_attention_with_removed_columns(attention_layer, h, attention_mask, remove_idx):
    q = attention_layer.query(h)
    k = attention_layer.key(h)
    v = attention_layer.value(h)
    scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
    if attention_mask is not None:
        valid_mask = attention_mask[:, None, :].bool()
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    if len(remove_idx):
        scores[:, :, torch.as_tensor(remove_idx, dtype=torch.long, device=scores.device)] = torch.finfo(scores.dtype).min
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.bmm(attn, v)
    return ctx, attn


def predict_with_intervention(model, tokenizer, sequence, device, intervention=None, indices=None):
    indices = [] if indices is None else list(map(int, indices))
    input_ids, attention_mask = tokenize_sequence(tokenizer, sequence, device)
    with torch.no_grad():
        emb = model.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = run_masked_lstm(model.lstm, emb, attention_mask)
        if intervention == "zero_states" and indices:
            h = h.clone()
            h[:, indices, :] = 0.0
        if intervention == "remove_attention_columns":
            ctx, attn = self_attention_with_removed_columns(model.attention, h, attention_mask, indices)
        else:
            ctx, attn = model.attention(h, attention_mask, return_weights=True)
        logits = model.fc(model.dropout(ctx))
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        preds = probs.argmax(axis=-1)
    return preds, probs, attn[0].detach().cpu().numpy()


def residue_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def neq_bin_array(neq, n_bins):
    neq = np.asarray(neq, dtype=float)
    if len(np.unique(neq)) <= 1:
        return np.zeros(len(neq), dtype=int)
    ranks = pd.Series(neq).rank(method="first").to_numpy()
    bins = np.floor((ranks - 1) / len(neq) * n_bins).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def pool_mask(mode_labels, pool_name):
    if pool_name == "all":
        return np.ones(len(mode_labels), dtype=bool)
    if pool_name == "low_mode_1":
        return mode_labels == 1
    if pool_name == "low_mode_2":
        return mode_labels == 2
    raise ValueError(f"Unknown pool mode: {pool_name}")


def select_top_hubs(attn, modes, pool_name, top_frac):
    received = np.asarray(attn, dtype=float).sum(axis=0)
    eligible = pool_mask(modes, pool_name)
    idx = np.where(eligible)[0]
    if len(idx) == 0:
        return np.asarray([], dtype=int), received
    k = max(1, int(np.ceil(float(top_frac) * len(idx))))
    order = idx[np.argsort(-received[idx])]
    return np.sort(order[:k]), received


def matched_random_indices(selected, modes, q3, q8, neq_bins, pool_name, rng, controls_per_protein):
    selected = np.asarray(selected, dtype=int)
    all_idx = np.arange(len(modes))
    eligible_pool = pool_mask(modes, pool_name)
    selected_set = set(selected.tolist())
    controls = []
    for _ in range(controls_per_protein):
        chosen = []
        used = set()
        for idx in selected:
            strict = (
                eligible_pool
                & (q3 == q3[idx])
                & (q8 == q8[idx])
                & (neq_bins == neq_bins[idx])
                & np.asarray([j not in selected_set and j not in used for j in all_idx])
            )
            relaxed = (
                eligible_pool
                & (q3 == q3[idx])
                & (neq_bins == neq_bins[idx])
                & np.asarray([j not in selected_set and j not in used for j in all_idx])
            )
            fallback = eligible_pool & np.asarray([j not in selected_set and j not in used for j in all_idx])
            candidates = np.where(strict)[0]
            if len(candidates) == 0:
                candidates = np.where(relaxed)[0]
            if len(candidates) == 0:
                candidates = np.where(fallback)[0]
            if len(candidates) == 0:
                continue
            pick = int(rng.choice(candidates))
            chosen.append(pick)
            used.add(pick)
        controls.append(np.sort(np.asarray(chosen, dtype=int)))
    return controls


def aligned_q_arrays(ss_map, protein, n):
    entry = ss_map.get(protein)
    if entry is None:
        return np.asarray([""] * n), np.asarray([""] * n)
    q3 = np.asarray(entry["q3"][:n], dtype=object)
    q8 = np.asarray(entry["q8"][:n], dtype=object)
    if len(q3) != n or len(q8) != n:
        return np.asarray([""] * n), np.asarray([""] * n)
    return q3, q8


def stripped_columns(df):
    return {c.strip(): c for c in df.columns}


def load_pb_entropy_map(path, id_col="id", entropy_col=None):
    if not path:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = stripped_columns(df)
    raw_id = cols.get(id_col.strip(), id_col)
    if entropy_col is None:
        for candidate in ["pb_entropy", "PB_entropy", "protein_block_entropy", "entropy_bits"]:
            if candidate in cols:
                entropy_col = cols[candidate]
                break
    elif entropy_col.strip() in cols:
        entropy_col = cols[entropy_col.strip()]
    if entropy_col is None or entropy_col not in df.columns:
        raise ValueError(f"Could not find PB entropy column in {path}.")
    out = {}
    for _, row in df.iterrows():
        out.setdefault(str(row[raw_id]).lstrip(">"), []).append(float(row[entropy_col]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def residualize_targets(test_rows, ss_map, pb_map):
    rows = []
    design_rows = []
    y = []
    for protein, entry in test_rows.items():
        n = len(entry["sequence"])
        q3, q8 = aligned_q_arrays(ss_map, protein, n)
        pb = pb_map.get(protein, np.full(n, np.nan))
        if len(pb) != n:
            pb = np.full(n, np.nan)
        for i in range(n):
            design_rows.append({"q3": q3[i], "q8": q8[i], "pb_entropy": pb[i]})
            y.append(float(entry["neq"][i]))
            rows.append({"name": protein, "position": i + 1, "sequence": entry["sequence"], "neq": float(entry["neq"][i])})
    design = pd.DataFrame(design_rows)
    x = pd.get_dummies(design[["q3", "q8"]].astype(str), dummy_na=True)
    pb = design["pb_entropy"].astype(float)
    x["pb_entropy"] = pb.fillna(float(pb.mean(skipna=True)) if pb.notna().any() else 0.0)
    x.insert(0, "intercept", 1.0)
    xmat = x.to_numpy(dtype=float)
    yvec = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(xmat, yvec, rcond=None)
    pred = xmat.dot(coef)
    residual = yvec - pred
    out = pd.DataFrame(rows)
    out["neq_pred_from_q3_q8_pb_entropy"] = pred
    out["neq_residual"] = residual
    return out


def compare_residual_manifest(original_selections, residual_manifest, result_root, pipeline_dir, args):
    if not residual_manifest:
        return pd.DataFrame()
    path = resolve_manifest_path(result_root, residual_manifest)
    if not path.exists():
        raise FileNotFoundError(f"Missing residual manifest: {path}")
    manifest = pd.read_csv(path, sep="\t")
    rows = []
    for run in manifest.itertuples(index=False):
        attn_path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        records = load_attention_records(attn_path)
        for key, selected in original_selections.items():
            condition, seed, protein, pool, frac = key
            if str(seed) != str(run.seed) or protein not in records:
                continue
            attn = np.asarray(records[protein]["attention_weights"], dtype=float)
            n = attn.shape[0]
            labels, *_ = analyze_attention_modes(attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed)
            residual_selected, received = select_top_hubs(attn, labels, pool, frac)
            a = set(map(int, selected))
            b = set(map(int, residual_selected))
            union = a | b
            rows.append({
                "original_condition": condition,
                "residual_condition": run.condition,
                "seed": seed,
                "protein": protein,
                "pool": pool,
                "top_frac": frac,
                "original_n": len(a),
                "residual_n": len(b),
                "jaccard": len(a & b) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    manifest_path = resolve_manifest_path(result_root, args.manifest_tsv or (result_root / "manifest.tsv"))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_analysis_dir(result_root, "analysis_attention_hub_ablation", manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    rng = np.random.default_rng(args.random_seed)

    classify = create_classification_func(args.num_classes, args.neq_thresholds)
    test_rows = load_test_rows(args.test_csv, classify)
    ss_map = load_q3_q8_map(args.ss_csv)
    pb_map = load_pb_entropy_map(args.pb_entropy_csv, args.pb_id_col, args.pb_entropy_col)
    residual_targets = residualize_targets(test_rows, ss_map, pb_map)
    residual_targets.to_csv(output_dir / "neq_residual_targets_q3_q8_pb.csv", index=False)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()
    manifest = manifest[manifest["architecture"].astype(str) == "bilstm_attention"].copy()
    if args.max_runs:
        manifest = manifest.head(args.max_runs).copy()

    per_protein = []
    selected_rows = []
    original_selections = {}
    for run_idx, run in enumerate(manifest.itertuples(index=False), start=1):
        print(f"[{run_idx}/{len(manifest)}] {run.condition} seed={run.seed}", flush=True)
        checkpoint = resolve_run_path(run.checkpoint, result_root, pipeline_dir)
        attn_json = resolve_run_path(run.attention_json, result_root, pipeline_dir)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if not attn_json.exists():
            raise FileNotFoundError(f"Missing attention JSON: {attn_json}")

        model = build_model(run, checkpoint, device, args)
        tokenizer = load_esm_tokenizer(str(run.esm_model))
        records = load_attention_records(attn_json)
        proteins = [p for p in records if p in test_rows]
        if args.max_proteins:
            proteins = proteins[:args.max_proteins]

        for protein in proteins:
            entry = test_rows[protein]
            y_true = entry["labels"]
            sequence = entry["sequence"]
            record = records[protein]
            attn = np.asarray(record["attention_weights"], dtype=float)[:len(sequence), :len(sequence)]
            n = len(sequence)
            if len(y_true) != n:
                continue
            modes, row_ent, *_ = analyze_attention_modes(attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed)
            q3, q8 = aligned_q_arrays(ss_map, protein, n)
            neq_bins = neq_bin_array(entry["neq"], args.neq_bins)
            baseline_pred, _, _ = predict_with_intervention(model, tokenizer, sequence, device)
            baseline = residue_metrics(y_true, baseline_pred[:n])

            for pool in args.pool_modes:
                for frac in args.top_fracs:
                    selected, received = select_top_hubs(attn, modes, pool, frac)
                    if len(selected) == 0:
                        continue
                    original_selections[(run.condition, run.seed, protein, pool, float(frac))] = selected
                    controls = matched_random_indices(selected, modes, q3, q8, neq_bins, pool, rng, args.controls_per_protein)

                    for idx in selected:
                        selected_rows.append({
                            "condition": run.condition,
                            "seed": run.seed,
                            "protein": protein,
                            "pool": pool,
                            "top_frac": frac,
                            "selection_type": "hub",
                            "position": int(idx) + 1,
                            "aa": sequence[int(idx)],
                            "q3": q3[int(idx)],
                            "q8": q8[int(idx)],
                            "neq": float(entry["neq"][int(idx)]),
                            "neq_bin": int(neq_bins[int(idx)]),
                            "mode_label": int(modes[int(idx)]),
                            "received_attention": float(received[int(idx)]),
                        })

                    selections = [("hub", 0, selected)] + [
                        ("matched_random", c_idx + 1, control) for c_idx, control in enumerate(controls)
                    ]
                    for selection_type, control_id, indices in selections:
                        for intervention in args.interventions:
                            pred, _, _ = predict_with_intervention(
                                model, tokenizer, sequence, device, intervention=intervention, indices=indices
                            )
                            metrics = residue_metrics(y_true, pred[:n])
                            row = {
                                "condition": run.condition,
                                "seed": run.seed,
                                "protein": protein,
                                "pool": pool,
                                "top_frac": frac,
                                "selection_type": selection_type,
                                "control_id": control_id,
                                "intervention": intervention,
                                "n_masked": int(len(indices)),
                            }
                            for key, value in baseline.items():
                                row[f"baseline_{key}"] = value
                                row[f"perturbed_{key}"] = metrics[key]
                                row[f"drop_{key}"] = value - metrics[key]
                            per_protein.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_protein_df = pd.DataFrame(per_protein)
    selected_df = pd.DataFrame(selected_rows)
    per_protein_df.to_csv(output_dir / "hub_ablation_per_protein.csv", index=False)
    selected_df.to_csv(output_dir / "hub_ablation_selected_residues.csv", index=False)

    if not per_protein_df.empty:
        summary = (
            per_protein_df
            .groupby(["condition", "pool", "top_frac", "selection_type", "intervention"], dropna=False)
            .agg(
                proteins=("protein", "nunique"),
                runs=("seed", "nunique"),
                mean_n_masked=("n_masked", "mean"),
                mean_drop_accuracy=("drop_accuracy", "mean"),
                median_drop_accuracy=("drop_accuracy", "median"),
                mean_drop_macro_f1=("drop_macro_f1", "mean"),
                median_drop_macro_f1=("drop_macro_f1", "median"),
                mean_drop_weighted_f1=("drop_weighted_f1", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / "hub_ablation_summary_by_condition.csv", index=False)

        paired = []
        keys = ["condition", "seed", "protein", "pool", "top_frac", "intervention"]
        hub_rows = per_protein_df[per_protein_df["selection_type"] == "hub"]
        ctrl_rows = per_protein_df[per_protein_df["selection_type"] == "matched_random"]
        ctrl_mean = ctrl_rows.groupby(keys, dropna=False)[["drop_accuracy", "drop_macro_f1", "drop_weighted_f1"]].mean().reset_index()
        merged = hub_rows.merge(ctrl_mean, on=keys, suffixes=("_hub", "_control_mean"))
        for _, row in merged.iterrows():
            paired.append({
                **{k: row[k] for k in keys},
                "hub_minus_control_drop_accuracy": row["drop_accuracy_hub"] - row["drop_accuracy_control_mean"],
                "hub_minus_control_drop_macro_f1": row["drop_macro_f1_hub"] - row["drop_macro_f1_control_mean"],
                "hub_minus_control_drop_weighted_f1": row["drop_weighted_f1_hub"] - row["drop_weighted_f1_control_mean"],
            })
        paired_df = pd.DataFrame(paired)
        paired_df.to_csv(output_dir / "hub_vs_matched_control_effects.csv", index=False)
        if not paired_df.empty:
            paired_summary = (
                paired_df
                .groupby(["condition", "pool", "top_frac", "intervention"], dropna=False)
                .agg(
                    proteins=("protein", "nunique"),
                    mean_hub_minus_control_drop_accuracy=("hub_minus_control_drop_accuracy", "mean"),
                    median_hub_minus_control_drop_accuracy=("hub_minus_control_drop_accuracy", "median"),
                    mean_hub_minus_control_drop_macro_f1=("hub_minus_control_drop_macro_f1", "mean"),
                    median_hub_minus_control_drop_macro_f1=("hub_minus_control_drop_macro_f1", "median"),
                )
                .reset_index()
            )
            paired_summary.to_csv(output_dir / "hub_vs_matched_control_summary.csv", index=False)

    residual_overlap = compare_residual_manifest(
        original_selections, args.residual_manifest_tsv, result_root, pipeline_dir, args
    )
    if not residual_overlap.empty:
        residual_overlap.to_csv(output_dir / "residual_trained_hub_overlap.csv", index=False)

    summary_txt = [
        f"Manifest: {manifest_path}",
        f"Runs analyzed: {len(manifest)}",
        f"Per-protein ablation rows: {len(per_protein_df)}",
        f"Selected hub residue rows: {len(selected_df)}",
        f"Residualized target table: {output_dir / 'neq_residual_targets_q3_q8_pb.csv'}",
        "Interventions:",
        "  zero_states: selected BiLSTM states are set to zero before self-attention.",
        "  remove_attention_columns: selected key columns are removed from the attention softmax.",
        "Controls: random residues matched by Q3, Q8 when possible, Neq quantile bin, and row-mode pool.",
        "Residual-Neq models are not trained here; pass --residual_manifest_tsv after training them to compare hub recurrence.",
    ]
    (output_dir / "hub_ablation_summary.txt").write_text("\n".join(summary_txt) + "\n")
    print("\n".join(summary_txt))


if __name__ == "__main__":
    main()
