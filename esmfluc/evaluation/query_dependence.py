"""How query-dependent is the contribution matrix C_ij = A_ij s_j? (test set)

Per protein:
  col_mean_r2   fraction of variance in C explained by replacing every row with the
                column mean I_j (the reviewer's rank-1 approximation 1 I^T)
  svd1_frac     fraction of ||C||^2 in the best rank-1 approximation (top singular value)
  attn_entropy  mean normalized row entropy of A (1 = uniform attention)
  attn_key_cv   median over keys of the coefficient of variation of A_ij across queries
  margin_sd     SD of the logit margin m_i across query residues; all of this comes from
                the query-dependent part of C, since a column-constant C gives constant m_i

Usage: python esmfluc/evaluation/query_dependence.py RUNS_DIR OUT_CSV [conditions...] [--seeds 1,2,3]
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
seeds = [1, 2, 3]
if "--seeds" in args:
    i = args.index("--seeds"); seeds = [int(x) for x in args[i + 1].split(",")]; del args[i:i + 2]
runs_dir, out_csv, conds = Path(args[0]), args[1], args[2:]
if not conds:
    conds = sorted(p.name for p in runs_dir.iterdir() if p.name.endswith("bilstm_attn"))

rows = []
for cond in conds:
    for seed in seeds:
        path = runs_dir / cond / f"seed_{seed}" / "all_split_signed_contributions" / "test_signed_contributions.json.gz"
        data = json.load(gzip.open(path))
        for p in data["proteins"] if "proteins" in data else data["records"]:
            C = np.asarray(p["contribution_matrix"], float)
            A = np.asarray(p["attention_matrix"], float)
            L = len(C)
            I = C.mean(axis=0)
            tot = ((C - C.mean()) ** 2).sum()
            col_mean_r2 = 1 - ((C - I) ** 2).sum() / tot if tot > 0 else np.nan
            sv = np.linalg.svd(C, compute_uv=False)
            H = -(A * np.log(np.clip(A, 1e-12, None))).sum(axis=1) / np.log(L)
            cv = A.std(axis=0) / A.mean(axis=0)
            m = np.asarray(p["flex_minus_rigid_logit_margin"], float)
            rows.append(dict(condition=cond, seed=seed, protein=p["name"], length=L,
                             col_mean_r2=col_mean_r2, svd1_frac=sv[0] ** 2 / (sv ** 2).sum(),
                             attn_entropy=H.mean(), attn_key_cv=np.median(cv), margin_sd=m.std()))
        del data
        print(cond, seed, "done", flush=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\ncondition: median [IQR] across proteins (seeds pooled)")
for cond in conds:
    rs = [r for r in rows if r["condition"] == cond]
    out = []
    for k in ("col_mean_r2", "svd1_frac", "attn_entropy", "attn_key_cv", "margin_sd"):
        v = np.array([r[k] for r in rs], float)
        q = np.nanpercentile(v, [25, 50, 75])
        out.append(f"{k} {q[1]:.3f} [{q[0]:.3f}-{q[2]:.3f}]")
    print(cond, "|", " | ".join(out))
