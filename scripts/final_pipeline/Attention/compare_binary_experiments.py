#!/usr/bin/env python3
"""
compare_binary_experiments.py

Comprehensive comparison of the four binary-classification attention patterns:
  - esm2_binary_frozen
  - esm2_binary_unfrozen
  - esm3_binary_frozen
  - esm3_binary_unfrozen

Analyses
--------
1.  Attention statistics  (mean, entropy, sparsity, col_max) per model
2.  Within-SS Neq differentiation — Kruskal-Wallis test across clusters
    within each SS class (H / E / C), to test whether clusters capture
    dynamics *beyond* SS.
3.  Q8 enrichment per cluster × model  (obs/exp vs all residues)
4.  Per-cluster Neq / RSA / Disorder / ASA means
5.  Pairwise differential peaks — for all 6 model pairs, classify residues
    as A-specific / B-specific / Shared / Background, then report Q8
    enrichment and Neq/RSA/disorder means per category
6.  Sequence-separation profiles of top-5% attention pairs  (PNG)
7.  Per-cluster PWM sequence logos  (logomaker letter-height logos)

All text results are written to  analysis_report.txt  in the output dir.
Numerical results are also saved as CSV files.

Usage
-----
python compare_binary_experiments.py \\
    --exp_root    /path/to/results \\
    --neq_csv     /path/to/data/test_data_with_names.csv \\
    --nsp3_csv    /path/to/data/test_data_nsp3.csv \\
    --output_dir  /path/to/results/binary_comparison

Or call programmatically:
    from compare_binary_experiments import run_comparison
    run_comparison(exp_root, neq_csv, nsp3_csv, output_dir)
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

try:
    import logomaker
    _LOGOMAKER = True
except ImportError:
    _LOGOMAKER = False
    print("[warning] logomaker not available; PWM logos will use bar-chart fallback")

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
BINARY_EXPS = [
    "esm2_binary_frozen",
    "esm2_binary_unfrozen",
    "esm3_binary_frozen",
    "esm3_binary_unfrozen",
]
AAS         = list("ACDEFGHIKLMNPQRSTVWY")
SS3_LBLS    = ["H", "E", "C"]
Q8_LBLS     = ["H", "B", "E", "G", "I", "T", "S", "C"]
N_CLUSTERS  = 3
CLUSTER_NAMES = ["PatA", "PatB", "Bg"]
TOP_PEAK_PCT  = 0.10   # top 10 % col_max for motif extraction
MOTIF_HALF    = 3      # ±3 → 7-mer window
DIFF_TOP_PCT  = 25.0   # percentile for "peak" in differential analysis
DIFF_BOT_PCT  = 25.0   # percentile for "non-peak" in differential analysis

# Logomaker colour scheme (physicochemical groups)
AA_COLORS_LM = {
    "G": "#888888",
    "A": "#222222", "V": "#222222", "L": "#222222", "I": "#222222", "M": "#222222",
    "P": "#e67e00",
    "F": "#7b2d8b", "W": "#7b2d8b", "Y": "#7b2d8b",
    "S": "#2ca02c", "T": "#2ca02c", "C": "#2ca02c", "N": "#2ca02c", "Q": "#2ca02c",
    "D": "#d62728", "E": "#d62728",
    "K": "#1f77b4", "R": "#1f77b4", "H": "#1f77b4",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as fh:
        recs = json.load(fh)
    if isinstance(recs, list):
        return {r["name"]: r for r in recs}
    return recs


def build_ref_map(neq_csv_path, nsp3_csv_path):
    """Load neq + NetSurfP-3 annotations into a per-protein dict."""
    neq_df = pd.read_csv(neq_csv_path)
    ref = {}
    for _, row in neq_df.iterrows():
        name = str(row["name"]).strip()
        seq  = str(row["sequence"]).strip()
        try:
            neq_list = json.loads(str(row["neq"]))
        except Exception:
            import ast
            neq_list = ast.literal_eval(str(row["neq"]))
        ref[name] = {
            "name": name, "sequence": seq, "neq_real": neq_list,
            "ss_pred": [], "q8": [], "rsa": [], "asa": [], "disorder": [],
        }

    nsp3_df = pd.read_csv(nsp3_csv_path)
    nsp3_df.columns = [c.strip() for c in nsp3_df.columns]
    nsp3_df["_name"] = nsp3_df["id"].astype(str).str.lstrip(">")
    for name, grp in nsp3_df.groupby("_name", sort=False):
        name = name.strip()
        if name not in ref:
            continue
        grp = grp.sort_values("n")
        ref[name]["ss_pred"]  = grp["q3"].tolist()
        ref[name]["q8"]       = grp["q8"].tolist() if "q8" in grp.columns else []
        ref[name]["rsa"]      = grp["rsa"].astype(float).tolist()
        ref[name]["asa"]      = grp["asa"].astype(float).tolist() if "asa" in grp.columns else []
        ref[name]["disorder"] = grp["disorder"].astype(float).tolist()
    return ref


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_attention(A_np, seed=42):
    """K=3 clustering on L2-normalised row+col features, ranked by energy."""
    N = A_np.shape[0]
    if N < N_CLUSTERS + 1:
        return np.zeros(N, dtype=int)
    feats = normalize(np.concatenate([A_np, A_np.T], axis=1), norm="l2")
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
    raw = km.fit_predict(feats)
    energy = (A_np.sum(axis=1) + A_np.sum(axis=0)) / (2 * N)
    order  = np.argsort([energy[raw == c].mean() if (raw == c).any() else 0.0
                         for c in range(N_CLUSTERS)])[::-1]
    remap  = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in raw])


# ── Statistical helpers ───────────────────────────────────────────────────────

def row_entropy(row):
    out = 0.0
    for p in row:
        if p > 1e-12:
            out -= p * math.log2(p)
    return out


def attn_stats(A_np):
    N = A_np.shape[0]
    entropies = [row_entropy(A_np[i]) for i in range(N)]
    col_maxes = A_np.max(axis=0)
    return {
        "mean":          float(A_np.mean()),
        "max":           float(A_np.max()),
        "diag_mean":     float(np.diag(A_np).mean()),
        "off_diag_mean": float((A_np.sum() - np.trace(A_np)) / max(1, N * N - N)),
        "sparsity":      float((A_np < 0.01).mean()),
        "entropy_mean":  float(np.mean(entropies)),
        "entropy_std":   float(np.std(entropies)),
        "col_max_mean":  float(col_maxes.mean()),
        "col_max_std":   float(col_maxes.std()),
    }


def col_max_ranks(A_np):
    """Per-residue percentile rank (0–100) of column maximum."""
    N = A_np.shape[0]
    col_maxes = A_np.max(axis=0)
    order = np.argsort(col_maxes)
    ranks = np.empty(N)
    ranks[order] = np.arange(N) / max(1, N - 1) * 100.0
    return ranks


def kw_test(groups):
    """Kruskal-Wallis test. Returns (H, p, eta2)."""
    clean = [np.array(g) for g in groups if len(g) >= 2]
    if len(clean) < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        H, p = kruskal(*clean)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    k = len(clean)
    n = sum(len(g) for g in clean)
    eta2 = (H - k + 1) / max(1, n - k)
    return float(H), float(p), float(eta2)


def enrichment_vs_bg(cat_counts, bg_counts, labels):
    """Obs / expected enrichment for each label."""
    cat_total = sum(cat_counts.get(l, 0) for l in labels)
    bg_total  = sum(bg_counts.get(l, 0) for l in labels)
    result = {}
    for l in labels:
        obs = cat_counts.get(l, 0) / cat_total if cat_total > 0 else 0.0
        exp = bg_counts.get(l, 0) / bg_total  if bg_total  > 0 else 0.0
        result[l] = obs / exp if exp > 1e-9 else float("nan")
    return result


# ── PWM ───────────────────────────────────────────────────────────────────────

def build_pwm(motifs, half=MOTIF_HALF):
    W = 2 * half + 1
    counts = np.zeros((W, 20))
    for _, seg in motifs:
        pad_l = half - (len(seg) // 2)
        for pos, aa in enumerate(seg):
            col = pad_l + pos
            if 0 <= col < W and aa in AAS:
                counts[col, AAS.index(aa)] += 1
    row_sums = counts.sum(axis=1, keepdims=True).clip(1)
    return counts / row_sums


def save_pwm_logo(pwm, title, path):
    """Logomaker-based PWM logo: letter height ∝ enrichment above uniform background."""
    W = pwm.shape[0]
    bg = 1.0 / len(AAS)
    enr = pwm / (bg + 1e-12)
    heights_pos = np.maximum(0, enr - 1)
    heights_neg = np.maximum(0, 1 - enr)
    positions   = list(range(-(W // 2), W - (W // 2)))

    df_pos = pd.DataFrame(heights_pos, columns=AAS, index=positions)
    df_neg = pd.DataFrame(heights_neg, columns=AAS, index=positions)

    fig, axes = plt.subplots(2, 1, figsize=(max(6, W * 1.0), 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_pos, ax_neg = axes

    if _LOGOMAKER and df_pos.values.max() > 0:
        try:
            logomaker.Logo(df_pos, ax=ax_pos, color_scheme=AA_COLORS_LM, show_spines=False)
        except Exception:
            ax_pos.bar(range(W), df_pos.max(axis=1), color="steelblue", alpha=0.7)
    else:
        ax_pos.bar(range(W), df_pos.max(axis=1), color="steelblue", alpha=0.7)
    ax_pos.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax_pos.set_ylabel("Enrichment − 1", fontsize=9)
    ax_pos.set_title(title, fontsize=10, fontweight="bold")
    ax_pos.set_xticks(range(W))
    ax_pos.set_xticklabels([f"{p:+d}" for p in positions], fontsize=8)

    if _LOGOMAKER and df_neg.values.max() > 0:
        try:
            logomaker.Logo(df_neg, ax=ax_neg, color_scheme=AA_COLORS_LM, show_spines=False)
            ax_neg.invert_yaxis()
        except Exception:
            pass
    ax_neg.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax_neg.set_ylabel("Depletion", fontsize=9)
    ax_neg.set_xticks(range(W))
    ax_neg.set_xticklabels([f"{p:+d}" for p in positions], fontsize=8)
    ax_neg.set_xlabel("Position relative to peak residue", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sep_profile(seps_dict, out_path, top_pct):
    """2×2 sequence-separation histograms for all 4 models."""
    models = list(seps_dict.keys())
    all_seps = [seps_dict[m] for m in models if seps_dict[m]]
    if not all_seps:
        return
    max_sep = max(max(s) for s in all_seps if s)
    bins = np.arange(0, min(max_sep + 2, 202), 2)
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    n = len(models)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    for ax, m, c in zip(axes, models, colors):
        seps = np.array(seps_dict[m])
        counts, edges = np.histogram(seps, bins=bins, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(centres, counts, width=2, color=c, alpha=0.8, edgecolor="none")
        ax.axvline(4, color="grey", lw=1, ls="--", label="α-helix (4)")
        ax.axvline(2, color="green", lw=1, ls=":", label="β-strand (2)")
        ax.set_xlabel("Sequence separation |i − j|")
        ax.set_ylabel("Density")
        ax.set_title(m, fontsize=9)
        local  = (seps <= 5).mean() * 100
        medium = ((seps > 5) & (seps <= 20)).mean() * 100
        long_  = (seps > 20).mean() * 100
        ax.text(0.97, 0.95,
                f"≤5: {local:.0f}%\n6-20: {medium:.0f}%\n>20: {long_:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    for ax in axes[len(models):]:
        ax.set_visible(False)

    fig.suptitle(f"Sequence separation — top-{top_pct:.0f}% attention pairs", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main comparison logic ─────────────────────────────────────────────────────

def run_comparison(exp_root, neq_csv, nsp3_csv, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "analysis_report.txt"
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg)
        log_f.write(msg + "\n")

    log("=" * 72)
    log("Binary Experiments Comparison: ESM2/3 × frozen/unfrozen")
    log("=" * 72)

    # ── Load data ─────────────────────────────────────────────────────────────
    log("\nBuilding reference map …")
    ref_map = build_ref_map(neq_csv, nsp3_csv)
    log(f"  Reference proteins: {len(ref_map)}")

    exp_root = Path(exp_root)
    attn_data = {}
    for exp in BINARY_EXPS:
        path = exp_root / exp / "bilstm_attn.json"
        if path.exists():
            attn_data[exp] = load_json(path)
            log(f"  Loaded {exp}: {len(attn_data[exp])} proteins")
        else:
            log(f"  [WARN] {path} not found, skipping")

    models = list(attn_data.keys())
    if len(models) < 2:
        log("[ERROR] Need at least 2 models. Aborting.")
        log_f.close()
        return

    common = sorted(
        set.intersection(*[set(attn_data[m]) for m in models]) & set(ref_map)
    )
    log(f"\n  Common proteins across all loaded models: {len(common)}\n")

    # ── Per-protein accumulation ───────────────────────────────────────────────
    stat_accum   = {m: [] for m in models}
    seps_all     = {m: [] for m in models}
    cluster_store = {m: {} for m in models}

    # Flat residue lists per model
    per_res = {
        m: {"ss3": [], "q8": [], "neq": [], "rsa": [], "asa": [], "dis": [], "cluster": []}
        for m in models
    }
    motifs_store = {m: {c: [] for c in range(N_CLUSTERS)} for m in models}

    log("Processing proteins …")
    for idx, name in enumerate(common):
        if (idx + 1) % 50 == 0 or (idx + 1) == len(common):
            log(f"  [{idx+1}/{len(common)}]")

        ref  = ref_map[name]
        seq  = ref["sequence"]
        N    = len(seq)
        neq_real = ref["neq_real"][:N]
        ss3      = ref.get("ss_pred",  [])[:N]
        q8       = ref.get("q8",       [])[:N]
        rsa      = ref.get("rsa",      [])[:N]
        asa      = ref.get("asa",      [])[:N]
        disorder = ref.get("disorder", [])[:N]

        for m in models:
            A_np = np.array(attn_data[m][name]["attention_weights"], dtype=np.float32)
            L    = min(A_np.shape[0], N)
            A_np = A_np[:L, :L]

            stat_accum[m].append(attn_stats(A_np))

            # Sequence separation (top_sep_pct = 5 %)
            mask = ~np.eye(L, dtype=bool)
            vals = A_np[mask]
            if len(vals) > 0:
                thresh = np.percentile(vals, 95.0)
                rows, cols = np.where((A_np >= thresh) & mask)
                seps_all[m].extend(np.abs(rows - cols).tolist())

            labels = cluster_attention(A_np)
            cluster_store[m][name] = labels

            col_max = A_np.max(axis=0)
            peak_thresh = np.percentile(col_max, (1 - TOP_PEAK_PCT) * 100)

            for i in range(L):
                c = int(labels[i])
                per_res[m]["cluster"].append(c)
                per_res[m]["ss3"].append(ss3[i] if i < len(ss3) else "")
                per_res[m]["q8"].append(q8[i] if i < len(q8) else "")
                per_res[m]["neq"].append(float(neq_real[i]) if i < len(neq_real) else float("nan"))
                per_res[m]["rsa"].append(float(rsa[i]) if i < len(rsa) else float("nan"))
                per_res[m]["asa"].append(float(asa[i]) if i < len(asa) else float("nan"))
                per_res[m]["dis"].append(float(disorder[i]) if i < len(disorder) else float("nan"))

                if col_max[i] >= peak_thresh:
                    start = max(0, i - MOTIF_HALF)
                    end   = min(L, i + MOTIF_HALF + 1)
                    seg   = seq[start:end]
                    if len(seg) >= 3:
                        motifs_store[m][c].append((i, seg))

    # Convert lists to arrays
    for m in models:
        for k in per_res[m]:
            per_res[m][k] = np.array(per_res[m][k])

    log(f"\nDone processing {len(common)} proteins.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Attention Statistics
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 1: Attention Statistics")
    log("=" * 72)

    STAT_KEYS = ["mean", "max", "diag_mean", "off_diag_mean", "sparsity",
                 "entropy_mean", "entropy_std", "col_max_mean", "col_max_std"]
    stat_means = {m: {k: float(np.mean([s[k] for s in stat_accum[m]])) for k in STAT_KEYS}
                  for m in models}

    header = f"  {'Metric':<22}" + "".join(f"  {m[:22]:>22}" for m in models)
    log(header)
    log("  " + "-" * (22 + len(models) * 24))
    for k in STAT_KEYS:
        line = f"  {k:<22}" + "".join(f"  {stat_means[m][k]:>22.5f}" for m in models)
        log(line)

    stat_df = pd.DataFrame([{"model": m, **stat_means[m]} for m in models])
    stat_df.to_csv(out_dir / "attention_stats.csv", index=False)
    log(f"\n  Saved → attention_stats.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Within-SS Neq Differentiation (Kruskal-Wallis)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 2: Within-SS Neq Differentiation (Kruskal-Wallis)")
    log("  Tests whether attention clusters separate Neq *within* each SS class.")
    log("  High eta2 within SS = clusters capture dynamics beyond secondary structure.")
    log("  ESM3-unfrozen should show higher eta2 than ESM2-frozen if backbone")
    log("  fine-tuning adds dynamics information beyond SS.")
    log("=" * 72)

    kw_rows = []
    log(f"\n  {'Model':<28} {'SS':>4} {'H':>9} {'p':>9} {'eta2':>7} "
        f"{'n_PatA':>8} {'n_PatB':>8} {'n_Bg':>8}")
    log("  " + "-" * 85)

    for m in models:
        cl  = per_res[m]["cluster"]
        ss  = per_res[m]["ss3"]
        neq = per_res[m]["neq"]
        for ss_class in SS3_LBLS:
            mask = ss == ss_class
            groups = []
            ns = []
            for c in range(N_CLUSTERS):
                g = neq[mask & (cl == c)]
                g = g[~np.isnan(g)]
                groups.append(g)
                ns.append(len(g))
            H, p, eta2 = kw_test(groups)
            log(f"  {m:<28} {ss_class:>4} {H:>9.3f} {p:>9.4f} {eta2:>7.4f} "
                f"{ns[0]:>8} {ns[1]:>8} {ns[2]:>8}")
            kw_rows.append({"model": m, "ss_class": ss_class,
                            "H_stat": H, "p_val": p, "eta2": eta2,
                            "n_PatA": ns[0], "n_PatB": ns[1], "n_Bg": ns[2]})

    pd.DataFrame(kw_rows).to_csv(out_dir / "within_ss_neq.csv", index=False)
    log(f"\n  Saved → within_ss_neq.csv")

    log("\n  Mean Neq per cluster within each SS class:")
    mean_neq_rows = []
    for m in models:
        log(f"\n  [{m}]")
        cl  = per_res[m]["cluster"]
        ss  = per_res[m]["ss3"]
        neq = per_res[m]["neq"]
        log(f"    {'SS':>4}" + "".join(f"  {cn:>9}" for cn in CLUSTER_NAMES))
        for ss_class in SS3_LBLS:
            mask = ss == ss_class
            means_str = []
            for c in range(N_CLUSTERS):
                g = neq[mask & (cl == c)]
                g = g[~np.isnan(g)]
                v = f"{np.mean(g):.4f}" if len(g) > 0 else "   N/A"
                means_str.append(v)
                mean_neq_rows.append({"model": m, "ss_class": ss_class,
                                      "cluster": CLUSTER_NAMES[c],
                                      "mean_neq": np.mean(g) if len(g) > 0 else float("nan"),
                                      "n": len(g)})
            log(f"    {ss_class:>4}" + "".join(f"  {v:>9}" for v in means_str))

    pd.DataFrame(mean_neq_rows).to_csv(out_dir / "mean_neq_within_ss.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Q8 Enrichment per Cluster × Model
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 3: Q8 Enrichment per Cluster per Model  (obs/exp vs background)")
    log("=" * 72)

    q8_rows = []
    for m in models:
        log(f"\n  [{m}]")
        cl = per_res[m]["cluster"]
        q8 = per_res[m]["q8"]
        bg_counts = {q: int((q8 == q).sum()) for q in Q8_LBLS}
        log(f"    {'Cluster':<8}" + "".join(f"  {q:>7}" for q in Q8_LBLS))
        log("    " + "-" * (8 + len(Q8_LBLS) * 9))
        for c, cname in enumerate(CLUSTER_NAMES):
            mask = cl == c
            cat_counts = {q: int((q8[mask] == q).sum()) for q in Q8_LBLS}
            enr = enrichment_vs_bg(cat_counts, bg_counts, Q8_LBLS)
            vals = "".join(f"  {enr[q]:>7.3f}" for q in Q8_LBLS)
            log(f"    {cname:<8}" + vals)
            row = {"model": m, "cluster": cname}
            row.update({f"q8_enr_{q}": enr[q] for q in Q8_LBLS})
            q8_rows.append(row)

    pd.DataFrame(q8_rows).to_csv(out_dir / "q8_enrichment_per_cluster.csv", index=False)
    log(f"\n  Saved → q8_enrichment_per_cluster.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Per-Cluster Neq / RSA / Disorder / ASA Means
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 4: Per-Cluster Neq / RSA / Disorder / ASA Means")
    log("=" * 72)

    cluster_mean_rows = []
    for m in models:
        log(f"\n  [{m}]")
        cl  = per_res[m]["cluster"]
        neq = per_res[m]["neq"]
        rsa = per_res[m]["rsa"]
        dis = per_res[m]["dis"]
        asa = per_res[m]["asa"]
        log(f"    {'Cluster':<8}  {'mean_Neq':>10}  {'mean_RSA':>10}  "
            f"{'mean_Dis':>10}  {'mean_ASA':>10}  {'n':>8}")
        log("    " + "-" * 65)
        for c, cname in enumerate(CLUSTER_NAMES):
            mask = cl == c
            mn  = float(np.nanmean(neq[mask]))
            mr  = float(np.nanmean(rsa[mask]))
            md  = float(np.nanmean(dis[mask]))
            ma  = float(np.nanmean(asa[mask]))
            n   = int(mask.sum())
            log(f"    {cname:<8}  {mn:>10.4f}  {mr:>10.4f}  "
                f"{md:>10.4f}  {ma:>10.4f}  {n:>8}")
            cluster_mean_rows.append({"model": m, "cluster": cname,
                                      "mean_neq": mn, "mean_rsa": mr,
                                      "mean_disorder": md, "mean_asa": ma, "n": n})

    pd.DataFrame(cluster_mean_rows).to_csv(out_dir / "cluster_means.csv", index=False)
    log(f"\n  Saved → cluster_means.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: Pairwise Differential Peaks
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 5: Pairwise Differential Peaks")
    log(f"  Peak = top {DIFF_TOP_PCT:.0f}% col_max per protein")
    log(f"  Non-peak = bottom {DIFF_BOT_PCT:.0f}% col_max per protein")
    log("  Categories: A-specific | B-specific | Shared | Background")
    log("=" * 72)

    DIFF_CATS = ["A_specific", "B_specific", "Shared", "Background"]
    diff_rows = []

    for ma, mb in combinations(models, 2):
        pair_label = (f"{ma.replace('_binary','').replace('esm','ESM')} vs "
                      f"{mb.replace('_binary','').replace('esm','ESM')}")
        log(f"\n  ── {pair_label} ──")

        q8_counts = {c: {q: 0 for q in Q8_LBLS} for c in DIFF_CATS}
        neq_vals  = {c: [] for c in DIFF_CATS}
        rsa_vals  = {c: [] for c in DIFF_CATS}
        dis_vals  = {c: [] for c in DIFF_CATS}

        pair_common = sorted(set(attn_data[ma]) & set(attn_data[mb]) & set(ref_map))
        for name in pair_common:
            ref  = ref_map[name]
            seq  = ref["sequence"]
            N    = len(seq)
            neq_r = ref["neq_real"][:N]
            q8_l  = ref.get("q8",       [])[:N]
            rsa_l = ref.get("rsa",      [])[:N]
            dis_l = ref.get("disorder", [])[:N]

            Aa = np.array(attn_data[ma][name]["attention_weights"], dtype=np.float32)
            Ab = np.array(attn_data[mb][name]["attention_weights"], dtype=np.float32)
            L  = min(Aa.shape[0], Ab.shape[0], N)
            Aa, Ab = Aa[:L, :L], Ab[:L, :L]

            ra = col_max_ranks(Aa)
            rb = col_max_ranks(Ab)
            TOP = 100.0 - DIFF_TOP_PCT
            BOT = DIFF_BOT_PCT

            for i in range(L):
                peak_a = ra[i] >= TOP
                peak_b = rb[i] >= TOP
                low_a  = ra[i] <= BOT
                low_b  = rb[i] <= BOT
                if   peak_a and low_b:  cat = "A_specific"
                elif peak_b and low_a:  cat = "B_specific"
                elif peak_a and peak_b: cat = "Shared"
                else:                   cat = "Background"

                if i < len(q8_l) and q8_l[i] in Q8_LBLS:
                    q8_counts[cat][q8_l[i]] += 1
                if i < len(neq_r) and neq_r[i] is not None:
                    neq_vals[cat].append(float(neq_r[i]))
                if i < len(rsa_l) and rsa_l[i] is not None:
                    rsa_vals[cat].append(float(rsa_l[i]))
                if i < len(dis_l) and dis_l[i] is not None:
                    dis_vals[cat].append(float(dis_l[i]))

        bg_q8 = q8_counts["Background"]
        log(f"\n    Q8 Enrichment (obs/exp vs Background):")
        log(f"    {'Category':<14}" + "".join(f"  {q:>7}" for q in Q8_LBLS))
        log("    " + "-" * (14 + len(Q8_LBLS) * 9))
        for cat in ["A_specific", "B_specific", "Shared"]:
            enr = enrichment_vs_bg(q8_counts[cat], bg_q8, Q8_LBLS)
            log(f"    {cat:<14}" + "".join(f"  {enr[q]:>7.3f}" for q in Q8_LBLS))

        log(f"\n    Mean Neq / RSA / Disorder per category:")
        log(f"    {'Category':<14}  {'mean_Neq':>10}  {'mean_RSA':>10}  "
            f"{'mean_Dis':>10}  {'n':>8}")
        log("    " + "-" * 58)
        for cat in DIFF_CATS:
            mn = float(np.mean(neq_vals[cat])) if neq_vals[cat] else float("nan")
            mr = float(np.mean(rsa_vals[cat])) if rsa_vals[cat] else float("nan")
            md = float(np.mean(dis_vals[cat])) if dis_vals[cat] else float("nan")
            n  = sum(q8_counts[cat].values())
            log(f"    {cat:<14}  {mn:>10.4f}  {mr:>10.4f}  {md:>10.4f}  {n:>8}")

            enr = enrichment_vs_bg(q8_counts[cat], bg_q8, Q8_LBLS)
            row = {"model_a": ma, "model_b": mb, "pair": pair_label,
                   "category": cat, "mean_neq": mn, "mean_rsa": mr,
                   "mean_disorder": md, "n_residues": n}
            row.update({f"q8_enr_{q}": enr.get(q, float("nan")) for q in Q8_LBLS})
            diff_rows.append(row)

    pd.DataFrame(diff_rows).to_csv(out_dir / "differential_peaks.csv", index=False)
    log(f"\n  Saved → differential_peaks.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Sequence-Separation Profiles
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 6: Sequence-Separation Profiles (top-5% attention pairs)")
    log("=" * 72)

    sep_path = out_dir / "separation_profile.png"
    save_sep_profile(seps_all, str(sep_path), top_pct=5.0)
    log(f"  Saved → separation_profile.png")

    for m in models:
        seps = np.array(seps_all[m])
        local  = (seps <= 5).mean() * 100
        medium = ((seps > 5) & (seps <= 20)).mean() * 100
        long_  = (seps > 20).mean() * 100
        log(f"  {m}: local(≤5)={local:.1f}%  medium(6-20)={medium:.1f}%  "
            f"long(>20)={long_:.1f}%  median={np.median(seps):.0f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: Per-Cluster PWM Sequence Logos
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 72)
    log("SECTION 7: Per-Cluster PWM Sequence Logos")
    log(f"  Top {TOP_PEAK_PCT*100:.0f}% col_max positions, ±{MOTIF_HALF} window")
    log("=" * 72)

    logo_dir = out_dir / "pwm_logos"
    logo_dir.mkdir(exist_ok=True)

    for m in models:
        m_short = m.replace("_binary", "")
        log(f"\n  [{m}]")
        for c, cname in enumerate(CLUSTER_NAMES):
            mots = motifs_store[m][c]
            if len(mots) < 10:
                log(f"    {cname}: only {len(mots)} motifs, skipping")
                continue
            pwm = build_pwm(mots)
            logo_path = logo_dir / f"{m_short}_{cname}_pwm.png"
            save_pwm_logo(pwm, f"{m_short} – {cname}  (n={len(mots)})", str(logo_path))
            log(f"    {cname}: {len(mots)} motifs → {logo_path.name}")

    log("\n" + "=" * 72)
    log(f"Analysis complete.  Results in: {out_dir}")
    log("=" * 72)
    log_f.close()
    print(f"\nFull report saved to: {log_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare 4 binary ESM attention experiments")
    p.add_argument("--exp_root",   required=True,
                   help="Directory with one sub-folder per experiment")
    p.add_argument("--neq_csv",    required=True,
                   help="test_data_with_names.csv (name, sequence, neq)")
    p.add_argument("--nsp3_csv",   required=True,
                   help="test_data_nsp3.csv (id, seq, n, rsa, asa, q3, q8, …, disorder)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(
        exp_root=args.exp_root,
        neq_csv=args.neq_csv,
        nsp3_csv=args.nsp3_csv,
        output_dir=args.output_dir,
    )
