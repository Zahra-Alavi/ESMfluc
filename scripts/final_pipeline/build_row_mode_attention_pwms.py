#!/usr/bin/env python3
"""
Build row-mode-aware PWMs and k-mer enrichment tables for attention-selected
residues.

Selections:
  - top received-attention residues within each row mode
  - vertical-band peak residues within each row mode
  - Neq-peak-overlapping hubs within each row mode

Background:
  For every selected residue, sample residues from the same protein and same
  secondary-structure class. This prevents a coil-rich attention mode from
  producing a trivial "coil amino acid composition" motif.
"""

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_attention_hubs_vs_neq_peaks import neq_peak_mask
from analyze_attention_morphology_biology import detect_bands
from analyze_attention_row_modes import analyze_attention_modes, resolve_existing_path


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    parser = argparse.ArgumentParser(description="Build PWMs for row-mode attention hubs/bands.")
    parser.add_argument("--result_root", required=True, help="Result root containing manifest.tsv.")
    parser.add_argument("--test_csv", required=True, help="CSV with name, sequence, neq.")
    parser.add_argument("--ss_csv", required=True, help="NetSurfP CSV with id and q3 columns.")
    parser.add_argument("--output_dir", default=None, help="Default: result_root/analysis_row_mode_pwms.")
    parser.add_argument("--pipeline_dir", default=None, help="Directory for resolving manifest paths.")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional condition subset.")
    parser.add_argument("--motif_len", type=int, default=11, help="Odd motif/PWM window length.")
    parser.add_argument("--kmer_len", type=int, default=5, help="Centered k-mer length.")
    parser.add_argument("--top_frac", type=float, default=0.10)
    parser.add_argument("--peak_quantile", type=float, default=0.90)
    parser.add_argument("--high_entropy_quantile", type=float, default=0.67)
    parser.add_argument("--min_low_rows", type=int, default=8)
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--band_quantile", type=float, default=0.90)
    parser.add_argument("--band_z", type=float, default=1.0)
    parser.add_argument("--min_band_width", type=int, default=2)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--background_per_selected", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--min_instances_for_pwm", type=int, default=20)
    return parser.parse_args()


def classify_neq(values, threshold=1.0):
    return np.asarray([0 if float(v) <= threshold else 1 for v in values], dtype=int)


def mode_name(label):
    return {0: "diffuse_high_entropy", 1: "low_mode_1", 2: "low_mode_2"}.get(int(label), str(label))


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def load_neq_by_name(test_csv):
    df = pd.read_csv(test_csv)
    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]) if "name" in df.columns else str(row["sequence"])
        neq = np.asarray(ast.literal_eval(row["neq"]), dtype=float)
        out[name] = {"sequence": row["sequence"], "neq": neq, "flexible": classify_neq(neq)}
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


def extract_window(sequence, center_idx, length):
    if length <= 0:
        return ""
    half = length // 2
    chars = []
    for offset in range(-half, half + 1):
        idx = center_idx + offset
        chars.append(sequence[idx] if 0 <= idx < len(sequence) else "-")
    return "".join(chars)


def add_pwm(pwm, sequence, center_idx, weight=1.0):
    motif_len = pwm.shape[0]
    half = motif_len // 2
    for pos, offset in enumerate(range(-half, half + 1)):
        idx = center_idx + offset
        if 0 <= idx < len(sequence):
            aa = sequence[idx]
            if aa in AA20:
                pwm[pos, AA20.index(aa)] += weight


def top_indices(values, frac):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    k = max(1, int(np.ceil(len(values) * frac)))
    return np.argsort(values)[-k:].tolist()


def build_selected_sets(received, modes, bands, peak_mask, top_frac):
    selected = {}
    for label in [0, 1, 2]:
        mode = mode_name(label)
        idx = np.where(modes == label)[0]
        if len(idx):
            top_local = idx[top_indices(received[idx], top_frac)]
            top_set = set(int(i) for i in top_local)
            selected[f"top_received__{mode}"] = top_set
            selected[f"neq_peak_overlapping_hubs__{mode}"] = set(i for i in top_set if peak_mask[i])

    band_peaks = {mode_name(label): set() for label in [0, 1, 2]}
    for start, end in bands:
        region = np.arange(start, end + 1)
        peak = int(region[np.argmax(received[region])])
        band_peaks[mode_name(int(modes[peak]))].add(peak)
    for mode, idx in band_peaks.items():
        selected[f"vertical_band_peaks__{mode}"] = idx
    return selected


def matched_background_indices(selected_indices, ss, rng, n_per_selected):
    selected_indices = sorted(set(int(i) for i in selected_indices))
    selected_set = set(selected_indices)
    bg = []
    all_idx = np.arange(len(ss))
    for idx in selected_indices:
        candidates = np.where(ss == ss[idx])[0]
        candidates = np.asarray([c for c in candidates if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([c for c in all_idx if int(c) not in selected_set], dtype=int)
        if len(candidates) == 0:
            candidates = all_idx
        replace = len(candidates) < n_per_selected
        bg.extend(int(x) for x in rng.choice(candidates, size=n_per_selected, replace=replace))
    return bg


def pwm_to_df(obs_pwm, bg_pwm):
    rows = []
    obs_total = obs_pwm.sum(axis=1, keepdims=True)
    bg_total = bg_pwm.sum(axis=1, keepdims=True)
    obs_freq = obs_pwm / np.maximum(obs_total, 1e-9)
    bg_freq = bg_pwm / np.maximum(bg_total, 1e-9)
    half = obs_pwm.shape[0] // 2
    for pos in range(obs_pwm.shape[0]):
        for ai, aa in enumerate(AA20):
            obs = float(obs_freq[pos, ai])
            bg = float(bg_freq[pos, ai])
            rows.append({
                "position": pos - half,
                "AA": aa,
                "frequency": obs,
                "bg_frequency": bg,
                "enrichment": obs / bg if bg > 1e-9 else np.nan,
                "log2_enrichment": np.log2((obs + 1e-6) / (bg + 1e-6)),
            })
    return pd.DataFrame(rows)


def kmer_enrichment(obs_counter, bg_counter):
    obs_total = sum(obs_counter.values())
    bg_total = sum(bg_counter.values())
    rows = []
    for kmer in sorted(set(obs_counter) | set(bg_counter)):
        obs = obs_counter.get(kmer, 0)
        bg = bg_counter.get(kmer, 0)
        obs_frac = obs / obs_total if obs_total else 0.0
        bg_frac = bg / bg_total if bg_total else 0.0
        rows.append({
            "kmer": kmer,
            "observed_count": int(obs),
            "background_count": int(bg),
            "observed_fraction": obs_frac,
            "background_fraction": bg_frac,
            "enrichment": obs_frac / bg_frac if bg_frac > 0 else np.nan,
            "log2_enrichment": np.log2((obs_frac + 1e-9) / (bg_frac + 1e-9)),
        })
    return pd.DataFrame(rows).sort_values(["log2_enrichment", "observed_count"], ascending=[False, False])


def main():
    args = parse_args()
    if args.motif_len % 2 != 1 or args.kmer_len % 2 != 1:
        raise ValueError("--motif_len and --kmer_len should be odd so windows are centered.")

    result_root = Path(args.result_root).expanduser().resolve()
    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve() if args.pipeline_dir else Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_root / "analysis_row_mode_pwms"
    pwm_dir = output_dir / "pwms"
    kmer_dir = output_dir / "kmers"
    output_dir.mkdir(parents=True, exist_ok=True)
    pwm_dir.mkdir(exist_ok=True)
    kmer_dir.mkdir(exist_ok=True)

    manifest = pd.read_csv(result_root / "manifest.tsv", sep="\t")
    if args.conditions:
        manifest = manifest[manifest["condition"].isin(args.conditions)].copy()
    neq_by_name = load_neq_by_name(args.test_csv)
    ss_map = load_ss_map(args.ss_csv)
    rng = np.random.default_rng(args.random_seed)

    obs_pwms = defaultdict(lambda: np.zeros((args.motif_len, len(AA20))))
    bg_pwms = defaultdict(lambda: np.zeros((args.motif_len, len(AA20))))
    obs_kmers = defaultdict(Counter)
    bg_kmers = defaultdict(Counter)
    instance_rows = []
    bg_rows = []

    for run in manifest.itertuples(index=False):
        attention_path = resolve_existing_path(run.attention_json, result_root, pipeline_dir)
        if not attention_path.exists():
            print(f"[skip] missing {attention_path}")
            continue
        print(f"[load] {run.condition} seed={run.seed}: {attention_path}")
        records = json.loads(attention_path.read_text())
        for record in records:
            protein = record["name"]
            if protein not in neq_by_name or protein not in ss_map:
                continue
            seq = record["sequence"]
            n = len(seq)
            neq = neq_by_name[protein]["neq"][:n]
            ss = np.asarray(ss_map[protein][:n])
            if len(neq) != n or len(ss) != n:
                continue
            attn = np.asarray(record["attention_weights"], dtype=float)[:n, :n]
            received = attn.sum(axis=0)
            modes, _, _, _, _, _ = analyze_attention_modes(
                attn, args.high_entropy_quantile, args.min_low_rows, args.kmeans_seed
            )
            bands, _, _, _ = detect_bands(
                received,
                quantile=args.band_quantile,
                z_thresh=args.band_z,
                min_width=args.min_band_width,
                smooth_window=args.smooth_window,
            )
            peak_mask = neq_peak_mask(neq, args.peak_quantile)
            selected_sets = build_selected_sets(received, modes, bands, peak_mask, args.top_frac)

            for selection, idx_set in selected_sets.items():
                if not idx_set:
                    continue
                key = f"{run.condition}__{selection}"
                bg_idx = matched_background_indices(idx_set, ss, rng, args.background_per_selected)
                for idx in sorted(idx_set):
                    motif = extract_window(seq, idx, args.motif_len)
                    kmer = extract_window(seq, idx, args.kmer_len)
                    add_pwm(obs_pwms[key], seq, idx, weight=float(received[idx]))
                    if "-" not in kmer:
                        obs_kmers[key][kmer] += 1
                    instance_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                        "selection": selection,
                        "position_1based": idx + 1,
                        "aa": seq[idx],
                        "ss": ss[idx],
                        "mode": mode_name(modes[idx]),
                        "neq": float(neq[idx]),
                        "is_neq_peak": bool(peak_mask[idx]),
                        "received_attention": float(received[idx]),
                        "motif": motif,
                        "kmer": kmer,
                    })
                for idx in bg_idx:
                    motif = extract_window(seq, idx, args.motif_len)
                    kmer = extract_window(seq, idx, args.kmer_len)
                    add_pwm(bg_pwms[key], seq, idx, weight=1.0)
                    if "-" not in kmer:
                        bg_kmers[key][kmer] += 1
                    bg_rows.append({
                        "condition": run.condition,
                        "seed": int(run.seed),
                        "protein": protein,
                        "selection": selection,
                        "position_1based": idx + 1,
                        "aa": seq[idx],
                        "ss": ss[idx],
                        "mode": mode_name(modes[idx]),
                        "neq": float(neq[idx]),
                        "is_neq_peak": bool(peak_mask[idx]),
                        "background_for": key,
                        "motif": motif,
                        "kmer": kmer,
                    })

    instance_df = pd.DataFrame(instance_rows)
    bg_df = pd.DataFrame(bg_rows)
    instance_df.to_csv(output_dir / "attention_motif_instances.csv", index=False)
    bg_df.to_csv(output_dir / "matched_background_motif_instances.csv", index=False)

    summary_rows = []
    for key in sorted(obs_pwms):
        n_obs = int((instance_df["condition"] + "__" + instance_df["selection"] == key).sum()) if not instance_df.empty else 0
        n_bg = int((bg_df["background_for"] == key).sum()) if not bg_df.empty else 0
        condition, selection = key.split("__", 1)
        summary_rows.append({
            "condition": condition,
            "selection": selection,
            "n_observed_instances": n_obs,
            "n_background_instances": n_bg,
            "pwm_written": n_obs >= args.min_instances_for_pwm,
        })
        if n_obs < args.min_instances_for_pwm:
            continue
        safe = sanitize(key)
        pwm_df = pwm_to_df(obs_pwms[key], bg_pwms[key])
        pwm_df.to_csv(pwm_dir / f"pwm__{safe}.csv", index=False)
        kmer_df = kmer_enrichment(obs_kmers[key], bg_kmers[key])
        kmer_df.to_csv(kmer_dir / f"kmer_enrichment__{safe}.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(output_dir / "pwm_kmer_summary.csv", index=False)

    lines = [
        f"Result root: {result_root}",
        f"Output dir: {output_dir}",
        f"Observed motif instances: {len(instance_df)}",
        f"Background motif instances: {len(bg_df)}",
        f"Selections with PWM files: {sum(r['pwm_written'] for r in summary_rows)}",
        "",
        "Key outputs:",
        "  attention_motif_instances.csv",
        "  matched_background_motif_instances.csv",
        "  pwm_kmer_summary.csv",
        "  pwms/pwm__*.csv",
        "  kmers/kmer_enrichment__*.csv",
    ]
    (output_dir / "row_mode_pwm_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
