#!/usr/bin/env python3
"""Join paper phenotypes and analyze activity and the mechanical subset."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy.stats import spearmanr

from weinreb_analysis_common import (
    PRIMARY_CONDITION, WEINREB_ANALYSIS_ROOT, WEINREB_RESULTS_ROOT,
    load_seed, read_manifest, safe_corr, write_csv, json_dump,
)


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
MECHANICAL_GROUPS = {
    "High-Strain": ["E173N", "P29V", "A175T", "D179S"],
    "Control": ["G62S", "V120A", "WT_star"],
    "Binding": ["S30Q", "G33S", "R60K"],
}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, default=WEINREB_ANALYSIS_ROOT)
    p.add_argument("--manifest", type=Path, default=WEINREB_RESULTS_ROOT / "exact_contributions_v2" / "manifest.tsv")
    p.add_argument("--condition", default=PRIMARY_CONDITION)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--permutations", type=int, default=5000)
    return p.parse_args()


def _col(ref: str) -> int:
    result = 0
    for char in re.match(r"[A-Z]+", ref).group():
        result = result * 26 + ord(char) - 64
    return result - 1


def read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    """Read cached-value XLSX cells without adding an openpyxl dependency."""
    with ZipFile(path) as z:
        shared = []
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        relroot = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rels = {x.attrib["Id"]: x.attrib["Target"] for x in relroot}
        rid_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        result = {}
        sheets = wb.find(NS + "sheets")
        for sheet in sheets:
            target = rels[sheet.attrib[rid_key]].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            root = ET.fromstring(z.read(target))
            rows = []
            for row in root.iter(NS + "row"):
                values = {}
                for cell in row.findall(NS + "c"):
                    v = cell.find(NS + "v")
                    value = "" if v is None else (v.text or "")
                    if cell.attrib.get("t") == "s" and value:
                        value = shared[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(NS + "t"))
                    values[_col(cell.attrib["r"])] = value
                if values:
                    rows.append([values.get(i, "") for i in range(max(values) + 1)])
            result[sheet.attrib["name"]] = rows
    return result


def rows_after_header(rows, header_text):
    hi = next(i for i, row in enumerate(rows) if row and row[0] == header_text)
    header = rows[hi]
    output = []
    for values in rows[hi + 1:]:
        values = values + [""] * (len(header) - len(values))
        output.append(dict(zip(header, values)))
    return output


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def ols_fit(x, y, weights=None):
    if weights is None:
        weights = np.ones(len(y))
    sw = np.sqrt(weights)
    beta = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]
    pred = x @ beta
    sse = float(np.sum(weights * (y - pred) ** 2))
    return beta, pred, sse


def residualize(v, covars, weights):
    return v - ols_fit(covars, v, weights)[1]


def loo_rmse(x, y, weights):
    pred = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        pred[i] = x[i] @ ols_fit(x[keep], y[keep], weights[keep])[0]
    return float(np.sqrt(np.mean((y - pred) ** 2))), float(safe_corr(y, pred)), pred


def main() -> None:
    args = arguments()
    rng = np.random.default_rng(20250719)
    seeds = tuple(int(x) for x in args.seeds.split(","))
    xlsx = args.analysis / "inputs" / "supplementary_data_1.xlsx"
    sheets = read_xlsx(xlsx)

    activity_raw = rows_after_header(sheets["Activity"], "Mutation")
    activity_rows = [{"mutation": r["Mutation"].replace("WT*", "WT_star"),
                      "activity": f(r["Activity"]), "replicate_index": i + 1}
                     for i, r in enumerate(activity_raw)]
    write_csv(args.analysis / "phenotypes" / "activity_replicates.csv", activity_rows)

    nr_raw = rows_after_header(sheets["NR - Measurements"], "Group")
    rheology_raw = []
    for i, r in enumerate(nr_raw):
        rheology_raw.append({
            "group": r["Group"], "mutation": r["Mutation"].replace("WT*", "WT_star"),
            "frequency_Hz": f(r["Hz"]), "amplitude": f(r["Amplitude"]),
            "phase": f(r["Phase"]), "measurement_day": r["Day"], "row_index": i + 1,
        })
    write_csv(args.analysis / "mechanics" / "rheology_raw_measurements.csv", rheology_raw)
    day_raw = rows_after_header(sheets["NR - Day"], "Mutation")
    day_rows = [{k.lower().replace(" ", "_"): (r[k].replace("WT*", "WT_star") if k == "Mutation" else r[k])
                 for k in r} for r in day_raw]
    write_csv(args.analysis / "mechanics" / "rheology_day_averages.csv", day_rows)
    mutation_raw = rows_after_header(sheets["NR - Mutation"], "Mutation")
    mutation_rows = [{k.lower().replace(" ", "_"): (r[k].replace("WT*", "WT_star") if k == "Mutation" else r[k])
                      for k in r} for r in mutation_raw]
    write_csv(args.analysis / "mechanics" / "rheology_mutation_averages.csv", mutation_rows)

    with (args.analysis / "inputs" / "extended_data_table1.csv").open(encoding="utf-8", newline="") as h:
        annotations = list(csv.DictReader(h))
    ann = {r["mutation"]: r for r in annotations}
    by_mut = {}
    for r in activity_rows:
        by_mut.setdefault(r["mutation"], []).append(r["activity"])
    phenotype_rows = []
    for name, r in ann.items():
        values = np.asarray(by_mut.get(name, []), float)
        phenotype_rows.append({**r,
            "activity_replicate_mean": float(values.mean()) if len(values) else None,
            "activity_replicate_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "activity_replicate_se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "activity_replicates_observed": len(values),
            "below_50pct_WT_star": bool(len(values) and values.mean() < 0.5),
        })
    write_csv(args.analysis / "phenotypes" / "paper_annotations_joined.csv", phenotype_rows)

    with np.load(args.analysis / "profiles" / "delta_I_arrays.npz", allow_pickle=False) as z:
        names = [str(x) for x in z["mutant_names"]]
        delta = np.asarray(z["delta"], float).mean(axis=1)
        baseline = np.asarray(z["wt_star"], float).mean(axis=0)
    with (args.analysis / "coordinate_mapping.csv").open(encoding="utf-8", newline="") as h:
        mapping = list(csv.DictReader(h))
    paper = np.array([int(r["paper_position"]) if r["paper_position"] else -1 for r in mapping])
    domain = np.array([r["domain"] for r in mapping])
    function = np.array([r["functional_annotation"] for r in mapping])
    name_index = {n: i for i, n in enumerate(names)}
    with (args.analysis / "bands" / "distal_burden.csv").open(encoding="utf-8", newline="") as h:
        distal = list(csv.DictReader(h))
    distal_primary = {r["mutant"]: f(r["distal_mean_abs_delta_I"]) for r in distal
                      if r["sequence_threshold"] == "15" and r["ca_threshold_A"] == "12"}
    with np.load(args.analysis / "receivers" / "query_pattern_arrays.npz", allow_pickle=False) as z:
        qnames = [str(x) for x in z["mutant_names"]]
        mean_dc = np.asarray(z["mean_delta_contribution"], float)
    qindex = {n: i for i, n in enumerate(qnames)}
    binding_mask = np.isin(paper, [30, 88, 31, 53, 60, 33, 101, 32])
    high_mask = np.isin(paper, [178, 179, 175, 174, 177, 173, 176, 29])

    loaded = {int(r["seed"]): load_seed(r) for r in read_manifest(args.manifest, args.condition, seeds)}
    predictor_rows = []
    for name in names:
        d = delta[name_index[name]]
        flex_mask = baseline > 0
        neq_delta = np.mean([loaded[s][name].neq_preds.mean() - loaded[s]["WT_star"].neq_preds.mean()
                             for s in seeds])
        predictor_rows.append({
            "mutation": name,
            "distal_abs_delta_I_burden": distal_primary.get(name),
            "signed_LID_delta_I": float(d[domain == "LID"].mean()),
            "signed_hinge_Ploop_delta_I": float(d[np.char.find(function.astype(str), "hinge") >= 0].mean()
                                                   + d[function == "P-loop"].mean()),
            "loss_WT_star_flexibility_band_strength": float(-d[flex_mask].mean()),
            "gain_of_rigidity": float(np.maximum(-d, 0).mean()),
            "hotspot_155_157_signed_delta_I": float(d[154:157].mean()),
            "hotspot_155_157_abs_delta_I": float(np.abs(d[154:157]).mean()),
            "change_received_by_binding_residues": float(np.abs(mean_dc[qindex[name], binding_mask]).sum(axis=1).mean()),
            "overlap_experimental_high_strain": float(np.abs(d[high_mask]).mean()),
            "predicted_Neq_fraction_change": float(neq_delta),
        })
    write_csv(args.analysis / "phenotypes" / "esm_predictors.csv", predictor_rows)
    predictors = {r["mutation"]: r for r in predictor_rows}

    analysis_names = [n for n in names if n in ann and n != "WT_star" and len(by_mut.get(n, []))]
    y = np.array([np.mean(by_mut[n]) for n in analysis_names])
    se = np.array([f(ann[n]["activity_se"]) for n in analysis_names])
    finite_se = se[np.isfinite(se) & (se > 0)]
    floor = np.median(finite_se) if len(finite_se) else 0.05
    weights = 1.0 / np.maximum(np.where(np.isfinite(se), se, floor), floor / 2) ** 2
    binding_strain = np.array([f(ann[n]["binding_strain"]) for n in analysis_names])
    group_binding = np.array([ann[n]["group"] == "Binding" for n in analysis_names], float)
    group_high = np.array([ann[n]["group"] == "High-Strain" for n in analysis_names], float)
    covars = np.column_stack([np.ones(len(y)), (binding_strain - binding_strain.mean()) / binding_strain.std(),
                             group_binding, group_high])
    predictor_keys = [k for k in predictor_rows[0] if k != "mutation"]
    neq_all = np.array([f(predictors[n]["predicted_Neq_fraction_change"]) for n in analysis_names])
    association_rows = []
    for key in predictor_keys:
        metric = np.array([f(predictors[n][key]) for n in analysis_names])
        keep = np.isfinite(metric) & np.isfinite(y) & np.isfinite(binding_strain)
        yy, mm, ww, cc = y[keep], metric[keep], weights[keep], covars[keep]
        mmz = (mm - mm.mean()) / (mm.std() or 1)
        full = np.column_stack([cc, mmz])
        beta, pred, sse_full = ols_fit(full, yy, ww)
        _, _, sse_reduced = ols_fit(cc, yy, ww)
        partial_r2 = (sse_reduced - sse_full) / sse_reduced if sse_reduced else np.nan
        ry, rm = residualize(yy, cc, ww), residualize(mmz, cc, ww)
        observed = abs(safe_corr(ry, rm))
        null = np.array([abs(safe_corr(ry, rng.permutation(rm))) for _ in range(args.permutations)])
        rho, rho_p = spearmanr(mm, yy)
        base_rmse, base_r, _ = loo_rmse(cc, yy, ww)
        full_rmse, full_r, _ = loo_rmse(full, yy, ww)
        beyond_neq = {}
        if key != "predicted_Neq_fraction_change":
            nn = neq_all[keep]
            nnz = (nn - nn.mean()) / (nn.std() or 1)
            cc_neq = np.column_stack([cc, nnz])
            full_neq = np.column_stack([cc_neq, mmz])
            beta_neq, _, sse_full_neq = ols_fit(full_neq, yy, ww)
            _, _, sse_reduced_neq = ols_fit(cc_neq, yy, ww)
            ry_neq, rm_neq = residualize(yy, cc_neq, ww), residualize(mmz, cc_neq, ww)
            observed_neq = abs(safe_corr(ry_neq, rm_neq))
            null_neq = np.array([abs(safe_corr(ry_neq, rng.permutation(rm_neq)))
                                 for _ in range(args.permutations)])
            neq_rmse, neq_r, _ = loo_rmse(cc_neq, yy, ww)
            full_neq_rmse, full_neq_r, _ = loo_rmse(full_neq, yy, ww)
            beyond_neq = {
                "effect_beyond_Neq": beta_neq[-1],
                "partial_R2_beyond_Neq": ((sse_reduced_neq - sse_full_neq) / sse_reduced_neq
                                           if sse_reduced_neq else np.nan),
                "permutation_p_beyond_Neq": (1 + np.sum(null_neq >= observed_neq)) / (1 + len(null_neq)),
                "Neq_baseline_LOO_RMSE": neq_rmse,
                "topology_plus_Neq_LOO_RMSE": full_neq_rmse,
                "LOO_RMSE_improvement_beyond_Neq": neq_rmse - full_neq_rmse,
                "Neq_baseline_LOO_correlation": neq_r,
                "topology_plus_Neq_LOO_correlation": full_neq_r,
            }
        else:
            beyond_neq = {k: None for k in (
                "effect_beyond_Neq", "partial_R2_beyond_Neq", "permutation_p_beyond_Neq",
                "Neq_baseline_LOO_RMSE", "topology_plus_Neq_LOO_RMSE",
                "LOO_RMSE_improvement_beyond_Neq", "Neq_baseline_LOO_correlation",
                "topology_plus_Neq_LOO_correlation")}
        association_rows.append({
            "predictor": key, "n": len(yy), "spearman_rho": rho, "spearman_asymptotic_p": rho_p,
            "adjusted_standardized_effect": beta[-1], "partial_R2": partial_r2,
            "covariate_adjusted_permutation_p": (1 + np.sum(null >= observed)) / (1 + len(null)),
            "baseline_LOO_RMSE": base_rmse, "full_LOO_RMSE": full_rmse,
            "LOO_RMSE_improvement": base_rmse - full_rmse,
            "baseline_LOO_correlation": base_r, "full_LOO_correlation": full_r,
            **beyond_neq,
        })
    for p_key, q_key in (("covariate_adjusted_permutation_p", "BH_FDR_q"),
                         ("permutation_p_beyond_Neq", "BH_FDR_q_beyond_Neq")):
        valid = [(i, float(r[p_key])) for i, r in enumerate(association_rows)
                 if r.get(p_key) not in {None, ""}]
        if valid:
            pvals = np.array([x[1] for x in valid])
            order = np.argsort(pvals)
            adjusted = np.empty(len(pvals), float)
            running = 1.0
            for rank_index in range(len(pvals) - 1, -1, -1):
                original = order[rank_index]
                running = min(running, pvals[original] * len(pvals) / (rank_index + 1))
                adjusted[original] = running
            for local_i, (row_i, _) in enumerate(valid):
                association_rows[row_i][q_key] = float(min(adjusted[local_i], 1.0))
        for row in association_rows:
            row.setdefault(q_key, None)
    write_csv(args.analysis / "phenotypes" / "activity_associations.csv", association_rows)

    pair_rows = []
    for a, b in [("P29A", "P29V"), ("S30G", "S30Q"), ("E173H", "E173N"), ("L174F", "L174G")]:
        row = {"substitution_1": a, "substitution_2": b,
               "activity_difference_2_minus_1": np.mean(by_mut[b]) - np.mean(by_mut[a])}
        for key in predictor_keys:
            row[key + "_difference_2_minus_1"] = f(predictors[b][key]) - f(predictors[a][key])
        pair_rows.append(row)
    write_csv(args.analysis / "phenotypes" / "same_position_chemical_controls.csv", pair_rows)

    # Mechanics: preserve hierarchy above, then summarize each curve relative to WT_star.
    curve = {}
    for r in mutation_rows:
        name = r["mutation"]
        hz = f(r["hz"])
        curve[(name, hz)] = (f(r["phase_mean"]), f(r["amplitude_mean"]))
    frequencies = sorted({hz for (_, hz) in curve})
    mechanical_rows = []
    for group, members in MECHANICAL_GROUPS.items():
        for name in members:
            phase_diff, amp_logratio = [], []
            for hz in frequencies:
                if (name, hz) not in curve or ("WT_star", hz) not in curve:
                    continue
                phase_diff.append((curve[(name, hz)][0] - curve[("WT_star", hz)][0]) ** 2)
                amp_logratio.append(np.log(curve[(name, hz)][1] / curve[("WT_star", hz)][1]) ** 2)
            mechanical_rows.append({
                "mutation": name, "pre_specified_group": group,
                "frequencies_observed": len(phase_diff),
                "phase_curve_RMSD_vs_WT_star": np.sqrt(np.mean(phase_diff)) if phase_diff else None,
                "amplitude_log_curve_RMSD_vs_WT_star": np.sqrt(np.mean(amp_logratio)) if amp_logratio else None,
                "combined_curve_distance": (np.sqrt(np.mean(phase_diff)) + np.sqrt(np.mean(amp_logratio)))
                                           if phase_diff and amp_logratio else None,
                "activity_mean": np.mean(by_mut[name]) if name in by_mut else None,
                "distal_abs_delta_I_burden": predictors.get(name, {}).get("distal_abs_delta_I_burden"),
            })
    write_csv(args.analysis / "mechanics" / "mechanical_subset_summary.csv", mechanical_rows)

    mechanical_tests = []
    non_wt = [r for r in mechanical_rows if r["mutation"] != "WT_star" and r["combined_curve_distance"] is not None]
    for outcome in ("combined_curve_distance", "distal_abs_delta_I_burden"):
        vals = np.array([f(r[outcome]) for r in non_wt])
        labels = np.array([r["pre_specified_group"] == "High-Strain" for r in non_wt])
        keep = np.isfinite(vals)
        vals, labels = vals[keep], labels[keep]
        observed = vals[labels].mean() - vals[~labels].mean()
        # Exact enumeration over all label assignments of the same size.
        null = []
        import itertools
        for selected in itertools.combinations(range(len(vals)), int(labels.sum())):
            mask = np.zeros(len(vals), bool); mask[list(selected)] = True
            null.append(vals[mask].mean() - vals[~mask].mean())
        null = np.asarray(null)
        mechanical_tests.append({
            "outcome": outcome, "contrast": "High-Strain minus Binding/Control",
            "n": len(vals), "effect_difference": observed,
            "exact_two_sided_permutation_p": (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)),
            "exact_assignments": len(null),
        })
    write_csv(args.analysis / "mechanics" / "mechanical_subset_exact_tests.csv", mechanical_tests)
    json_dump(args.analysis / "phenotypes" / "association_config.json", {
        "primary_outcome": "replicate-mean activity normalized by WT_star",
        "secondary_outcome": "below 50% of WT_star",
        "adjustment": ["binding strain", "Binding group", "High-Strain group"],
        "uncertainty": "weighted by reported activity SE with a robust SE floor",
        "permutations": args.permutations,
        "new_sequences_excluded": [n for n in names if n not in ann],
        "ordinary_prediction_control": "change in fraction of residues assigned Neq>1",
        "topology_beyond_Neq_test": "paper covariates plus predicted Neq change versus the same model plus one topology metric",
        "mechanics_scope": MECHANICAL_GROUPS,
        "mechanics_inference": "descriptive curve distances and exact label permutations",
    })


if __name__ == "__main__":
    main()
