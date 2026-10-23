"""Compare component-extremum coincidence at stable apices and reference positions.

Component extrema are raw scipy.signal.find_peaks calls, without smoothing or
an amplitude threshold. Coincidence means the exact same sequence position.
These descriptive counts are not a test of enrichment or causality.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks


SCHEMA = "esmfluc.uniform_attention_control.seed_average.v1"
EVIDENCE = "seed_averaged_intrinsic_signed_evidence"
ATTENTION = "seed_averaged_attention_column_mean"
INFLUENCE = "seed_averaged_observed_signed_influence"
KEY = ["condition", "split"]
COORDINATES = [
    "protein", "protein_length", "band_id", "sign", "apex_index_0based",
    "apex_residue_1based", "apex_amino_acid", "support_start_index_0based",
    "support_end_index_0based_inclusive", "support_start_residue_1based",
    "support_end_residue_1based_inclusive", "seed_support_count",
]
PER_APEX_COLUMNS = KEY + COORDINATES + [
    "class_support", "mean_observed_influence", "mean_signed_evidence",
    "mean_received_attention", "evidence_extremum", "attention_maximum",
    "component_classification",
]
CLASSES = ("evidence_only", "attention_only", "both", "neither")
MEASURES = ("evidence_extremum", "attention_maximum", *CLASSES)
REFERENCE_GROUPS = ("other_influence_extrema", "all_same_sign_residues")
REFERENCE_SITE_COLUMNS = KEY + [
    "protein", "protein_length", "sign", "class_support", "index_0based",
    "residue_1based", "amino_acid", "mean_observed_influence", "mean_signed_evidence",
    "mean_received_attention", "evidence_extremum", "attention_maximum",
    "component_classification",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value, name: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(f"{name}: expected an integer, found {value!r}")
    return int(numeric)


def profile_components(profile: dict):
    """Validate the separate seed means and locate their raw component extrema."""
    sequence = profile["sequence"]
    length = integer(profile["length"], "profile length")
    arrays = {name: np.asarray(profile[name], dtype=np.float64)
              for name in (EVIDENCE, ATTENTION, INFLUENCE)}
    if length != len(sequence) or length < 1:
        raise ValueError(f"{profile['name']}: sequence and length disagree")
    for name, values in arrays.items():
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"{profile['name']}: invalid {name}")
    s, b, influence = (arrays[name] for name in (EVIDENCE, ATTENTION, INFLUENCE))
    if (b < 0).any() or not np.isclose(b.sum(), 1, atol=5e-6, rtol=1e-6):
        raise ValueError(f"{profile['name']}: mean attention must be nonnegative and sum to one")
    evidence_peaks = {sign: set(find_peaks(sign * s)[0]) for sign in (-1, 1)}
    influence_peaks = {sign: set(find_peaks(sign * influence)[0]) for sign in (-1, 1)}
    attention_peaks = set(find_peaks(b)[0])
    return s, b, influence, evidence_peaks, influence_peaks, attention_peaks


def component_class(evidence: bool, attention: bool) -> str:
    return ("both" if evidence and attention else "evidence_only" if evidence
            else "attention_only" if attention else "neither")


def classify_apices(bands: pd.DataFrame, profile: dict) -> list[dict]:
    """Classify existing apices; do not redetect bands or multiply mean s and B."""
    s, b, influence, evidence_peaks, influence_peaks, attention_peaks = profile_components(profile)
    sequence, length = profile["sequence"], len(s)
    rows = []
    for band in bands.to_dict("records"):
        sign = integer(band["sign"], "sign")
        if sign not in (-1, 1):
            raise ValueError("Contribution sign must be -1 or 1")
        j = integer(band["apex_index_0based"], "apex index")
        left = integer(band["support_start_index_0based"], "band start")
        right = integer(band["support_end_index_0based_inclusive"], "band end")
        if not (0 <= left <= j <= right < length):
            raise ValueError(f"{band['band_id']}: band/apex coordinates out of bounds")
        if (integer(band["apex_residue_1based"], "apex residue") != j + 1
                or integer(band["support_start_residue_1based"], "band start residue") != left + 1
                or integer(band["support_end_residue_1based_inclusive"], "band end residue") != right + 1):
            raise ValueError(f"{band['band_id']}: zero-/one-based coordinates disagree")
        if (band["protein"] != profile["name"]
                or integer(band["protein_length"], "protein length") != length
                or band["apex_amino_acid"] != sequence[j]):
            raise ValueError(f"{band['band_id']}: profile and band sequence disagree")
        if j not in influence_peaks[sign] or sign * influence[j] <= 0:
            raise ValueError(f"{band['band_id']}: saved apex is not a signed raw influence extremum")
        saved = float(band["apex_signed_column_influence"])
        if not np.isclose(influence[j], saved, atol=1e-10, rtol=1e-8):
            raise ValueError(f"{band['band_id']}: saved apex influence differs from profile")
        evidence = j in evidence_peaks[sign] and sign * s[j] > 0
        attention = j in attention_peaks
        row = {name: band[name] for name in KEY + COORDINATES}
        row.update(
            class_support="multi-PB" if sign == 1 else "single-PB",
            mean_observed_influence=float(influence[j]),
            mean_signed_evidence=float(s[j]), mean_received_attention=float(b[j]),
            evidence_extremum=bool(evidence), attention_maximum=bool(attention),
            component_classification=component_class(evidence, attention),
        )
        rows.append(row)
    return rows


def reference_profile_rows(bands: pd.DataFrame, profile: dict,
                           *, condition: str, split: str) -> tuple[list[dict], list[dict]]:
    """Count other raw influence extrema and all same-sign eligible residues.

    Other extrema exclude only retained stable apex positions, not entire band
    intervals. They may include subthreshold, unstable or merged secondary peaks.
    """
    s, b, influence, evidence_peaks, influence_peaks, attention_peaks = profile_components(profile)
    length = len(s)
    eligible = set(range(1, length - 1))
    site_rows, count_rows = [], []
    for sign in (-1, 1):
        stable = set(bands.loc[bands.sign == sign, "apex_index_0based"].map(int))
        raw = {j for j in influence_peaks[sign] if sign * influence[j] > 0} & eligible
        if not stable <= raw:
            raise ValueError(f"{profile['name']}: stable apices must be eligible influence extrema")
        evidence = {j for j in evidence_peaks[sign] if sign * s[j] > 0}
        same_sign = {j for j in eligible if sign * influence[j] > 0}
        other = raw - stable
        context = dict(condition=condition, split=split, protein=profile["name"],
                       sign=sign, class_support="multi-PB" if sign == 1 else "single-PB")
        for j in sorted(other):
            e, a = j in evidence, j in attention_peaks
            site_rows.append({
                **context, "protein_length": length, "index_0based": j,
                "residue_1based": j + 1, "amino_acid": profile["sequence"][j],
                "mean_observed_influence": float(influence[j]),
                "mean_signed_evidence": float(s[j]), "mean_received_attention": float(b[j]),
                "evidence_extremum": e, "attention_maximum": a,
                "component_classification": component_class(e, a),
            })
        for group, positions in zip(REFERENCE_GROUPS, (other, same_sign)):
            e, a = positions & evidence, positions & attention_peaks
            count_rows.append({
                **context, "reference_group": group, "n_positions": len(positions),
                "n_evidence_extremum": len(e), "n_attention_maximum": len(a),
                "n_evidence_only": len(e - a), "n_attention_only": len(a - e),
                "n_both": len(e & a), "n_neither": len(positions - e - a),
            })
    return site_rows, count_rows


def summarize_references(counts: pd.DataFrame, cohorts: list[dict], *, by_sign: bool) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        selected = counts[(counts.condition == cohort["condition"]) & (counts.split == cohort["split"])]
        for sign in (-1, 1) if by_sign else (None,):
            for group in REFERENCE_GROUPS:
                subset = selected[selected.reference_group == group]
                if sign is not None:
                    subset = subset[subset.sign == sign]
                n = int(subset.n_positions.sum())
                row = {**cohort, "reference_group": group, "n_positions": n,
                       "n_proteins_with_positions": subset.loc[subset.n_positions > 0, "protein"].nunique()}
                if sign is not None:
                    row.update(sign=sign, class_support="multi-PB" if sign == 1 else "single-PB")
                for measure in MEASURES:
                    count = int(subset[f"n_{measure}"].sum())
                    row[f"n_{measure}"] = count
                    row[f"percent_{measure}"] = 100 * count / n if n else np.nan
                if sum(row[f"n_{category}"] for category in CLASSES) != n:
                    raise AssertionError("Reference categories do not partition the positions")
                rows.append(row)
    return pd.DataFrame(rows)


def add_reference_columns(summary: pd.DataFrame, references: pd.DataFrame, *, by_sign: bool) -> pd.DataFrame:
    keys = KEY + (["sign"] if by_sign else [])
    result = summary.copy()
    for group in REFERENCE_GROUPS:
        subset = references[references.reference_group == group]
        fields = ["n_positions", "n_evidence_extremum", "percent_evidence_extremum"]
        subset = subset[keys + fields].rename(columns={name: f"{group}_{name}" for name in fields})
        result = result.merge(subset, on=keys, how="left", validate="one_to_one")
    return result


def summarize(apices: pd.DataFrame, cohorts: list[dict], *, by_sign: bool) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        selected = apices[(apices.condition == cohort["condition"])
                          & (apices.split == cohort["split"])]
        for sign in (-1, 1) if by_sign else (None,):
            subset = selected if sign is None else selected[selected.sign == sign]
            n = len(subset)
            row = {**cohort, "n_apices": n,
                   "n_proteins_with_apices": subset.protein.nunique()}
            if sign is not None:
                row.update(sign=sign, class_support="multi-PB" if sign == 1 else "single-PB")
            for measure in MEASURES:
                count = (int(subset[measure].sum()) if measure in MEASURES[:2]
                         else int((subset.component_classification == measure).sum()))
                row[f"n_{measure}"] = count
                row[f"percent_{measure}"] = 100 * count / n if n else np.nan
            if sum(row[f"n_{category}"] for category in CLASSES) != n:
                raise AssertionError("Component categories do not partition the apices")
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(bands_csv: Path, profiles_manifest_tsv: Path, output_dir: Path,
            *, splits=("test",), conditions=None) -> dict[str, pd.DataFrame]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    manifest = pd.read_csv(profiles_manifest_tsv, sep="\t", keep_default_na=False)
    required = {*KEY, "profile_json_gz"}
    if not required <= set(manifest):
        raise ValueError(f"Profile manifest lacks {sorted(required - set(manifest))}")
    manifest = manifest[manifest.split.isin(splits)].copy()
    if conditions:
        manifest = manifest[manifest.condition.isin(conditions)].copy()
    if manifest.empty or manifest.duplicated(KEY).any():
        raise ValueError("Profile selection must contain one file per condition and split")
    if set(splits) - set(manifest.split) or (conditions and set(conditions) - set(manifest.condition)):
        raise ValueError("Requested conditions or splits are missing from the profile manifest")
    bands = pd.read_csv(bands_csv, keep_default_na=False)
    required = {*KEY, *COORDINATES, "apex_signed_column_influence", "primary_stable_band"}
    if not required <= set(bands):
        raise ValueError(f"Stable-band table lacks {sorted(required - set(bands))}")
    bands = bands[bands.split.isin(splits)].copy()
    if conditions:
        bands = bands[bands.condition.isin(conditions)].copy()
    selected_keys = set(map(tuple, manifest[KEY].to_numpy()))
    if set(map(tuple, bands[KEY].to_numpy())) - selected_keys:
        raise ValueError("Some selected bands have no matching profile file")
    if (bands.duplicated(KEY + ["band_id"]).any()
            or bands.duplicated(KEY + ["protein", "apex_index_0based"]).any()):
        raise ValueError("Duplicate stable band or apex")
    if not bands.primary_stable_band.astype(str).str.lower().eq("true").all():
        raise ValueError("Input contains bands not marked primary_stable_band")
    if any(integer(n, "seed support") not in (2, 3) for n in bands.seed_support_count):
        raise ValueError("Seed-stable bands must have support from two or three seeds")
    records, reference_sites, reference_counts, cohorts = [], [], [], []
    sources = [bands_csv.resolve(), profiles_manifest_tsv.resolve()]
    for record in manifest.sort_values(KEY).to_dict("records"):
        path = Path(record["profile_json_gz"])
        if not path.is_absolute():
            path = profiles_manifest_tsv.resolve().parent / path
        path = path.resolve()
        sources.append(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        if (data.get("schema_version") != SCHEMA or data.get("source_seeds") != [1, 2, 3]
                or any(data.get(k) != record[k] for k in KEY)):
            raise ValueError(f"{path}: expected matching three-seed mean component profiles")
        profiles = data["proteins"]
        names = [p["name"] for p in profiles]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: duplicate protein profiles")
        count = len(profiles)
        residues = sum(integer(p["length"], "length") for p in profiles)
        for name, value in (("protein_count", count), ("residue_count", residues)):
            if integer(data[name], name) != value or (name in record and integer(record[name], name) != value):
                raise ValueError(f"{path}: {name} differs from saved metadata")
        selected = bands[(bands.condition == record["condition"]) & (bands.split == record["split"])]
        if set(selected.protein) - set(names):
            raise ValueError(f"{path}: missing proteins required by the band catalog")
        grouped = {name: group for name, group in selected.groupby("protein", sort=False)}
        for profile in profiles:
            protein_bands = grouped.get(profile["name"], selected.iloc[:0])
            records.extend(classify_apices(protein_bands, profile))
            sites, counts = reference_profile_rows(
                protein_bands, profile, condition=record["condition"], split=record["split"])
            reference_sites.extend(sites)
            reference_counts.extend(counts)
        cohorts.append({**{k: record[k] for k in KEY}, "n_profile_proteins": count})
    apices = pd.DataFrame(records, columns=PER_APEX_COLUMNS).sort_values(
        KEY + ["protein", "apex_index_0based"], ignore_index=True)
    if len(apices) != len(bands):
        raise AssertionError("Not every selected stable apex was classified")
    per_protein = pd.DataFrame(reference_counts, columns=KEY + [
        "protein", "sign", "class_support", "reference_group", "n_positions",
        *[f"n_{measure}" for measure in MEASURES],
    ]).sort_values(KEY + ["protein", "sign", "reference_group"], ignore_index=True)
    reference_sign = summarize_references(per_protein, cohorts, by_sign=True)
    reference_condition = summarize_references(per_protein, cohorts, by_sign=False)
    outputs = {
        "per_apex_component_localization.csv": apices,
        "component_localization_by_condition_sign.csv": add_reference_columns(
            summarize(apices, cohorts, by_sign=True), reference_sign, by_sign=True),
        "component_localization_by_condition.csv": add_reference_columns(
            summarize(apices, cohorts, by_sign=False), reference_condition, by_sign=False),
        "per_reference_influence_extremum.csv": pd.DataFrame(
            reference_sites, columns=REFERENCE_SITE_COLUMNS).sort_values(
                KEY + ["protein", "sign", "index_0based"], ignore_index=True),
        "component_reference_counts_by_protein.csv": per_protein,
        "component_reference_rates_by_condition_sign.csv": reference_sign,
        "component_reference_rates_by_condition.csv": reference_condition,
    }
    parameters = {
        "analysis": "seed-stable influence apex / raw component extremum coincidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "splits": list(splits), "conditions": sorted(manifest.condition.unique()),
        "catalog_scope": "primary seed-stable bands, at least two of three seeds",
        "component_profiles": {"evidence": EVIDENCE, "attention": ATTENTION, "influence": INFLUENCE},
        "averaging": "Separate arithmetic means of each saved profile over seeds 1, 2, 3. Mean influence is NOT reconstructed as mean(s) * mean(B).",
        "extrema": "scipy.signal.find_peaks on the complete raw profile. Positive evidence maxima and negative evidence minima must have the source-band sign; attention uses maxima only. No smoothing, amplitude/prominence threshold, or coordinate tolerance.",
        "plateaus_and_endpoints": "SciPy default: midpoint of a flat peak, rounded down; sequence endpoints are not extrema.",
        "categories": list(CLASSES),
        "reference_groups": {
            "other_influence_extrema": "Same-sign raw local extrema of the three-seed mean influence profile, excluding retained primary seed-stable apex positions. No amplitude threshold or seed-support criterion is imposed on reference extrema. Entire band intervals are NOT excluded; references can include subthreshold, unstable and merged secondary extrema.",
            "all_same_sign_residues": "All non-endpoint residues with strictly positive or strictly negative three-seed mean influence, as appropriate. Includes stable apices and other extrema; exact-zero influence residues are excluded.",
        },
        "denominators": "Stable-apex percent_* columns use n_apices; reference-rate tables use n_positions. Both are pooled within condition/split/sign, not equally weighted by protein. Appended reference columns in the stable summaries carry each group's n_positions, n_evidence_extremum and percent_evidence_extremum. Empty groups have zero counts and missing percentages. Evidence/attention counts each include both.",
        "coordinate_convention": "Input-sequence indices, not PDB numbering; interval ends are inclusive.",
        "interpretation": "Descriptive comparisons of stable apices with other influence extrema and same-sign residues. Matching on being an influence extremum addresses one selection difference, but does not remove all algebraic or selection dependencies; this is not an enrichment test or causal intervention.",
        "command": [sys.executable, "-m", "esmfluc.analysis.bands", "apex-component-localization",
                    "--bands-csv", str(bands_csv.resolve()),
                    "--profiles-manifest-tsv", str(profiles_manifest_tsv.resolve()),
                    "--output-dir", str(output_dir.resolve()), "--splits", *splits,
                    *(["--conditions", *conditions] if conditions else [])],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__},
        "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sources],
        "source_sha256": sha256_file(Path(__file__)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, float_format="%.12g")
    parameters["outputs"] = {name: {"rows": len(frame), "sha256": sha256_file(output_dir / name)}
                             for name, frame in outputs.items()}
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands-csv", required=True, type=Path)
    parser.add_argument("--profiles-manifest-tsv", required=True, type=Path,
                        help="Seed-averaged control-profile manifest with profile_json_gz paths")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    outputs = analyze(args.bands_csv, args.profiles_manifest_tsv, args.output_dir,
                      splits=args.splits, conditions=args.conditions)
    print(f"Classified {len(outputs['per_apex_component_localization.csv'])} seed-stable apices.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
