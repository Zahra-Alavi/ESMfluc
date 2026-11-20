"""ATLAS test metrics with and without the first/last two residues, plus class prevalence.

Usage: python esmfluc/evaluation/atlas_chain_end_metrics.py RUNS_DIR TEST_CSV OUT_CSV
"""
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

runs_dir, test_csv, out_csv = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
csv.field_size_limit(10**9)
neq = {r["name"]: np.array(ast.literal_eval(r["neq"]), float) for r in csv.DictReader(open(test_csv))}

rows = []
for cond in sorted(p.name for p in runs_dir.iterdir() if p.is_dir()):
    for seed in (1, 2, 3):
        recs = json.load(open(runs_dir / cond / f"seed_{seed}" / "attention.json"))
        y_all, s_all, n_all, y_in, s_in, n_in = [], [], [], [], [], []
        for r in recs:
            n = neq[r["name"]]
            s = np.asarray(r["flexible_scores"], float)
            assert len(n) == len(s), r["name"]
            y_all.append(n > 1); s_all.append(s); n_all.append(n)
            keep = slice(2, len(n) - 2)
            y_in.append(n[keep] > 1); s_in.append(s[keep]); n_in.append(n[keep])
        for label, y, s, n in (("all", y_all, s_all, n_all), ("ends_excluded", y_in, s_in, n_in)):
            y, s, n = np.concatenate(y), np.concatenate(s), np.concatenate(n)
            rows.append(dict(condition=cond, seed=seed, positions=label, n_residues=len(y),
                             multi_pb_prevalence=y.mean(), auroc=roc_auc_score(y, s),
                             auprc=average_precision_score(y, s), spearman=spearmanr(s, n).correlation))
        print(cond, seed, "done", flush=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# mean +- SD across seeds
summary = {}
for r in rows:
    summary.setdefault((r["condition"], r["positions"]), []).append(r)
print("\ncondition positions n prevalence AUROC AUPRC Spearman (mean +- SD over seeds)")
for (c, p), rs in sorted(summary.items()):
    m = lambda k: f"{np.mean([x[k] for x in rs]):.3f}+-{np.std([x[k] for x in rs], ddof=1):.3f}"
    print(c, p, rs[0]["n_residues"], f"{rs[0]['multi_pb_prevalence']:.3f}", m("auroc"), m("auprc"), m("spearman"))
