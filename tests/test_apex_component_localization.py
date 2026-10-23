import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from esmfluc.analysis.bands.analyze_apex_component_localization import (
    ATTENTION, EVIDENCE, INFLUENCE, PER_APEX_COLUMNS, SCHEMA,
    analyze, classify_apices, summarize, reference_profile_rows,
    summarize_references, add_reference_columns,
)


def fixture():
    length = 17
    s = np.zeros(length)
    s[[0, 1, 2]] = [1, 3, 1]
    s[[4, 5, 6]] = [1, 2, 3]
    s[[8, 9, 10]] = [-1, -4, -1]
    s[[12, 13, 14]] = [-1, -2, -3]
    b = np.ones(length)
    b[[5, 9]] = 3
    b /= b.sum()
    influence = np.zeros(length)
    influence[[1, 5, 9, 13]] = [2, 3, -3, -2]
    profile = {"name": "protein", "sequence": "A" * length, "length": length,
               EVIDENCE: s.tolist(), ATTENTION: b.tolist(), INFLUENCE: influence.tolist()}
    bands = []
    for j in (1, 5, 9, 13):
        bands.append(dict(
            condition="model", split="test", protein="protein", protein_length=length,
            band_id=f"band_{j}", sign=int(np.sign(influence[j])),
            apex_index_0based=j, apex_residue_1based=j + 1, apex_amino_acid="A",
            support_start_index_0based=j, support_end_index_0based_inclusive=j,
            support_start_residue_1based=j + 1, support_end_residue_1based_inclusive=j + 1,
            seed_support_count=2, primary_stable_band=True,
            apex_signed_column_influence=influence[j],
        ))
    return pd.DataFrame(bands), profile


def write_inputs(directory: Path):
    bands, profile = fixture()
    zero = {**profile, "name": "no_apices", INFLUENCE: [0.] * profile["length"]}
    payload = dict(schema_version=SCHEMA, condition="model", split="test",
                   source_seeds=[1, 2, 3], protein_count=2,
                   residue_count=2 * profile["length"], proteins=[profile, zero])
    path = directory / "profiles.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(payload, handle)
    bands_csv = directory / "bands.csv"
    manifest_tsv = directory / "manifest.tsv"
    bands.to_csv(bands_csv, index=False)
    pd.DataFrame([dict(condition="model", split="test", profile_json_gz=path.name,
                       protein_count=2, residue_count=2 * profile["length"])]).to_csv(
        manifest_tsv, index=False, sep="\t")
    return bands_csv, manifest_tsv, path, payload


def reference_fixture():
    bands, profile = fixture()
    profile[INFLUENCE][0] = -100
    profile[INFLUENCE][2:4] = [1.2, 1.5]
    profile[INFLUENCE][15:17] = [-0.0001, 100]
    profile[EVIDENCE][3] = 4
    profile[EVIDENCE][15:17] = [1, 100]
    return bands, profile


class ApexComponentLocalizationTests(unittest.TestCase):
    def test_four_categories_and_signed_evidence(self):
        bands, profile = fixture()
        rows = classify_apices(bands, profile)
        self.assertEqual([r["component_classification"] for r in rows],
                         ["evidence_only", "attention_only", "both", "neither"])
        self.assertEqual([r["class_support"] for r in rows],
                         ["multi-PB", "multi-PB", "single-PB", "single-PB"])
        self.assertEqual(rows[2]["apex_residue_1based"], 10)
        self.assertEqual(rows[2]["mean_signed_evidence"], -4)

    def test_evidence_extremum_requires_band_sign(self):
        bands, profile = fixture()
        profile[EVIDENCE][0:3] = [-3, -1, -3]
        profile[EVIDENCE][8:11] = [3, 1, 3]
        rows = classify_apices(bands.iloc[[0, 2]], profile)
        self.assertFalse(any(r["evidence_extremum"] for r in rows))

    def test_plateau_midpoint_and_exact_coordinate_matching(self):
        bands, profile = fixture()
        profile[EVIDENCE][2] = 3
        self.assertTrue(classify_apices(bands.iloc[[0]], profile)[0]["evidence_extremum"])
        bands.loc[0, ["apex_index_0based", "support_start_index_0based",
                      "support_end_index_0based_inclusive"]] = 2
        bands.loc[0, ["apex_residue_1based", "support_start_residue_1based",
                      "support_end_residue_1based_inclusive"]] = 3
        profile[INFLUENCE][1:3] = [0, 2]
        self.assertFalse(classify_apices(bands.iloc[[0]], profile)[0]["evidence_extremum"])

    def test_saved_mean_influence_not_product_of_component_means(self):
        bands, profile = fixture()
        self.assertFalse(np.allclose(np.asarray(profile[EVIDENCE]) * profile[ATTENTION],
                                     profile[INFLUENCE]))
        self.assertEqual(len(classify_apices(bands, profile)), 4)

    def test_rejects_invalid_arrays_coordinates_sequences_and_influence(self):
        for kind in ("length", "nonfinite", "negative_attention", "attention_sum",
                     "coordinate", "amino_acid", "influence", "endpoint"):
            bands, profile = fixture()
            if kind == "length": profile[EVIDENCE].pop()
            elif kind == "nonfinite": profile[EVIDENCE][0] = float("nan")
            elif kind == "negative_attention": profile[ATTENTION][0] = -1
            elif kind == "attention_sum": profile[ATTENTION] = [1.] * profile["length"]
            elif kind == "coordinate": bands.loc[0, "apex_residue_1based"] = 10
            elif kind == "amino_acid": bands.loc[0, "apex_amino_acid"] = "G"
            elif kind == "influence": bands.loc[0, "apex_signed_column_influence"] = 99
            else:
                bands.loc[0, ["apex_index_0based", "support_start_index_0based",
                              "support_end_index_0based_inclusive"]] = 0
                bands.loc[0, ["apex_residue_1based", "support_start_residue_1based",
                              "support_end_residue_1based_inclusive"]] = 1
                profile[INFLUENCE][0] = 3
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                classify_apices(bands, profile)

    def test_summary_denominators_and_empty_groups(self):
        bands, profile = fixture()
        rows = pd.DataFrame(classify_apices(bands, profile), columns=PER_APEX_COLUMNS)
        cohorts = [dict(condition="model", split="test", n_profile_proteins=2),
                   dict(condition="empty_model", split="test", n_profile_proteins=2)]
        summary = summarize(rows, cohorts, by_sign=True)
        positive = summary[(summary.condition == "model") & (summary.sign == 1)].iloc[0]
        self.assertEqual(positive.n_apices, 2)
        self.assertEqual(positive.n_profile_proteins, 2)
        self.assertEqual(positive.n_proteins_with_apices, 1)
        self.assertEqual(positive.percent_evidence_extremum, 50)
        self.assertEqual(positive.percent_both, 0)
        empty = summary[summary.condition == "empty_model"]
        self.assertTrue(empty.n_apices.eq(0).all())
        self.assertTrue(empty.percent_evidence_extremum.isna().all())
        pooled = summarize(rows, cohorts[:1], by_sign=False).iloc[0]
        self.assertEqual(pooled.n_apices, 4)
        self.assertEqual(pooled.percent_both, 25)

    def test_reference_peaks_exclude_stable_apices_not_bands_or_small_peaks(self):
        bands, profile = reference_fixture()
        sites, counts = reference_profile_rows(bands, profile, condition="model", split="test")
        self.assertEqual({(r["sign"], r["index_0based"]) for r in sites}, {(1, 3), (-1, 15)})
        self.assertEqual({r["index_0based"] for r in sites} & set(bands.apex_index_0based), set())
        self.assertTrue(next(r for r in sites if r["sign"] == 1)["evidence_extremum"])
        self.assertFalse(next(r for r in sites if r["sign"] == -1)["evidence_extremum"])
        bands.loc[0, "support_end_index_0based_inclusive"] = 3
        bands.loc[0, "support_end_residue_1based_inclusive"] = 4
        merged_sites, merged_counts = reference_profile_rows(bands, profile, condition="model", split="test")
        self.assertEqual(sites, merged_sites)
        self.assertEqual(counts, merged_counts)

    def test_same_sign_reference_excludes_endpoints_and_exact_zero_influence(self):
        bands, profile = reference_fixture()
        _, counts = reference_profile_rows(bands, profile, condition="model", split="test")
        rows = {(r["reference_group"], r["sign"]): r for r in counts}
        positive = rows[("all_same_sign_residues", 1)]
        negative = rows[("all_same_sign_residues", -1)]
        self.assertEqual((positive["n_positions"], positive["n_evidence_extremum"]), (4, 2))
        self.assertEqual((negative["n_positions"], negative["n_evidence_extremum"]), (3, 1))
        for sign in (-1, 1):
            self.assertEqual(rows[("other_influence_extrema", sign)]["n_positions"], 1)

    def test_reference_rates_have_separate_denominators_and_merge_without_row_loss(self):
        bands, profile = reference_fixture()
        _, counts = reference_profile_rows(bands, profile, condition="model", split="test")
        cohort = [dict(condition="model", split="test", n_profile_proteins=1)]
        reference = summarize_references(pd.DataFrame(counts), cohort, by_sign=True)
        stable = summarize(pd.DataFrame(classify_apices(bands, profile)), cohort, by_sign=True)
        merged = add_reference_columns(stable, reference, by_sign=True)
        self.assertEqual(len(merged), 2)
        positive = merged[merged.sign == 1].iloc[0]
        self.assertEqual(positive.n_apices, 2)
        self.assertEqual(positive.percent_evidence_extremum, 50)
        self.assertEqual(positive.other_influence_extrema_n_positions, 1)
        self.assertEqual(positive.other_influence_extrema_percent_evidence_extremum, 100)
        self.assertEqual(positive.all_same_sign_residues_n_positions, 4)
        self.assertEqual(positive.all_same_sign_residues_percent_evidence_extremum, 50)
        pooled = summarize_references(pd.DataFrame(counts), cohort, by_sign=False)
        other = pooled[pooled.reference_group == "other_influence_extrema"].iloc[0]
        self.assertEqual(other.n_positions, 2)
        self.assertEqual(other.percent_evidence_extremum, 50)

    def test_reference_empty_groups_have_missing_percentages(self):
        bands, profile = fixture()
        sites, counts = reference_profile_rows(bands, profile, condition="model", split="test")
        self.assertEqual(sites, [])
        ref = summarize_references(pd.DataFrame(counts),
            [dict(condition="model", split="test", n_profile_proteins=1)], by_sign=True)
        other = ref[ref.reference_group == "other_influence_extrema"]
        self.assertTrue(other.n_positions.eq(0).all())
        self.assertTrue(other.percent_evidence_extremum.isna().all())

    def test_proteins_without_stable_apices_are_retained_in_reference_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bands, manifest, _, _ = write_inputs(root)
            frame = pd.read_csv(bands).iloc[:0]
            frame.to_csv(bands, index=False)
            outputs = analyze(bands, manifest, root / "results")
            self.assertEqual(len(outputs["per_apex_component_localization.csv"]), 0)
            sites = outputs["per_reference_influence_extremum.csv"]
            self.assertEqual(len(sites), 4)
            counts = outputs["component_reference_counts_by_protein.csv"]
            self.assertEqual(counts.protein.nunique(), 2)
            self.assertTrue(counts[counts.protein == "no_apices"].n_positions.eq(0).all())

    def test_exports_are_deterministic_and_provenance_is_verifiable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bands, manifest, _, _ = write_inputs(root)
            first = analyze(bands, manifest, root / "first")
            second = analyze(bands, manifest, root / "second")
            for name in first:
                self.assertEqual((root / "first" / name).read_bytes(),
                                 (root / "second" / name).read_bytes())
                pd.testing.assert_frame_equal(first[name], second[name])
            params = json.loads((root / "first/parameters.json").read_text())
            for source in params["inputs"]:
                self.assertEqual(source["sha256"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())
            for name, output in params["outputs"].items():
                self.assertEqual(output["sha256"], hashlib.sha256((root / "first" / name).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                analyze(bands, manifest, root / "first")

    def test_rejects_unstable_duplicate_and_mismatched_input(self):
        for kind in ("unstable", "support", "duplicate", "metadata", "missing_protein", "seeds"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bands_csv, manifest, profiles, payload = write_inputs(root)
                bands = pd.read_csv(bands_csv)
                if kind == "unstable": bands.loc[0, "primary_stable_band"] = False
                elif kind == "support": bands.loc[0, "seed_support_count"] = 1
                elif kind == "duplicate": bands = pd.concat([bands, bands.iloc[:1]])
                elif kind == "metadata": payload["condition"] = "other"
                elif kind == "missing_protein": bands.loc[0, "protein"] = "absent"
                else: payload["source_seeds"] = [1]
                bands.to_csv(bands_csv, index=False)
                with gzip.open(profiles, "wt") as handle:
                    json.dump(payload, handle)
                with self.assertRaises(ValueError):
                    analyze(bands_csv, manifest, root / "output")
                self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
