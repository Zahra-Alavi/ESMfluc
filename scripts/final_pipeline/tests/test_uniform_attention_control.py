import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from signed_band_analysis.build_uniform_attention_control_profiles import (
    PROFILE_FIELDS,
    SEED_AVERAGED_FIELDS,
    StreamingControlWriter,
    audit_seed_average_profile,
    build_control_profile,
    deterministic_shift_offset,
    generate_per_seed_profiles,
    generate_seed_averaged_profiles,
    iter_control_profiles,
    load_manifest,
    profile_similarity_rows,
    write_parameters,
)
from signed_band_analysis.extract_signed_contribution_bands import (
    CONTROL_AVERAGED_UNIFORM_FIELD,
    CONTROL_AVERAGED_SHIFTED_FIELD,
    CONTROL_SHIFTED_FIELD,
    CONTROL_UNIFORM_FIELD,
    iter_profiles as iter_detector_profiles,
    load_inputs as load_detector_inputs,
)
from signed_band_analysis.analyze_signed_band_seed_reproducibility import (
    validated_parameter_set,
)


def write_contribution(path: Path, seed: int, evidence, attention):
    evidence = np.asarray(evidence, dtype=float)
    attention = np.asarray(attention, dtype=float)
    observed = evidence * attention
    payload = {
        "schema_version": "esmfluc.signed_contributions.v1",
        "condition": "condition_a",
        "seed": seed,
        "split": "train",
        "protein_count": 1,
        "proteins": [{
            "name": "protein_a",
            "sequence": "ACDE",
            "length": 4,
            "intrinsic_signed_evidence": evidence.tolist(),
            "signed_column_influence": observed.tolist(),
            "attention_column_mean": attention.tolist(),
            # The streaming reader must skip these LxL fields.
            "attention_matrix": np.eye(4).tolist(),
            "contribution_matrix": np.eye(4).tolist(),
        }],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def make_control(name="protein_a", sequence="ACDE", evidence=None, attention=None):
    evidence = np.asarray(
        [1.0, -2.0, 3.0, -4.0] if evidence is None else evidence,
        dtype=float,
    )
    attention = np.asarray(
        [0.1, 0.2, 0.3, 0.4] if attention is None else attention,
        dtype=float,
    )
    length = len(sequence)
    shifted_attention = np.roll(attention, 1) if length > 1 else attention.copy()
    return {
        "name": name,
        "sequence": sequence,
        "length": length,
        "intrinsic_signed_evidence": evidence,
        "observed_signed_influence": evidence * attention,
        "uniform_signed_influence": evidence / length,
        "attention_column_mean": attention,
        "attention_amplification": length * attention,
        "shifted_attention_column_mean": shifted_attention,
        "shifted_attention_signed_influence": evidence * shifted_attention,
    }


def write_compact(path: Path, seed: int, profiles):
    profiles = list(profiles)
    with StreamingControlWriter(
        path,
        schema_version="esmfluc.uniform_attention_control.v1",
        metadata={
            "condition": "condition_a",
            "seed": seed,
            "split": "train",
            "protein_count": len(profiles),
            "residue_count": sum(profile["length"] for profile in profiles),
        },
        fields=PROFILE_FIELDS,
    ) as writer:
        for profile in profiles:
            writer.write(profile)


class UniformAttentionControlTests(unittest.TestCase):
    def test_control_identity_and_sign_audit(self):
        evidence = np.asarray([2.0, -4.0, 0.0, 8.0])
        attention = np.asarray([0.1, 0.2, 0.3, 0.4])
        control, audit = build_control_profile(
            {
                "name": "p",
                "sequence": "ACDE",
                "length": 4,
                "intrinsic_signed_evidence": evidence,
                "signed_column_influence": evidence * attention,
                "attention_column_mean": attention,
            },
            identity_atol=1e-12,
            identity_rtol=1e-12,
            normalization_atol=1e-12,
            normalization_rtol=1e-12,
        )
        np.testing.assert_allclose(
            control["uniform_signed_influence"], evidence / 4
        )
        np.testing.assert_allclose(
            control["attention_amplification"], attention * 4
        )
        np.testing.assert_allclose(
            control["observed_signed_influence"],
            control["uniform_signed_influence"]
            * control["attention_amplification"],
        )
        self.assertEqual(audit["exact_zero_evidence_residues"], 1)
        self.assertEqual(audit["unexpected_sign_disagreements"], 0)

    def test_shifted_attention_preserves_distribution_and_breaks_alignment(self):
        evidence = np.asarray([1.0, 2.0, 3.0, 4.0])
        attention = np.asarray([0.1, 0.2, 0.3, 0.4])
        control, audit = build_control_profile(
            {
                "name": "p", "sequence": "ACDE", "length": 4,
                "intrinsic_signed_evidence": evidence,
                "signed_column_influence": evidence * attention,
                "attention_column_mean": attention,
            },
            shift_offset=2,
            identity_atol=1e-12, identity_rtol=1e-12,
            normalization_atol=1e-12, normalization_rtol=1e-12,
        )
        np.testing.assert_allclose(
            control["shifted_attention_column_mean"], np.roll(attention, 2)
        )
        np.testing.assert_allclose(
            control["shifted_attention_signed_influence"],
            evidence * np.roll(attention, 2),
        )
        self.assertEqual(audit["shift_offset"], 2)
        first = deterministic_shift_offset(
            condition="c", seed=1, split="test", protein="p", length=4,
            random_seed=7,
        )
        second = deterministic_shift_offset(
            condition="c", seed=2, split="test", protein="p", length=4,
            random_seed=7,
        )
        # The same protein/condition rotation is used across model seeds. This
        # preserves seed-to-seed stability rather than destroying it by design.
        self.assertEqual(first, second)
        self.assertIn(first, (1, 2, 3))

    def test_nonzero_evidence_with_zero_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sign disagreements"):
            build_control_profile(
                {
                    "name": "p",
                    "sequence": "AC",
                    "length": 2,
                    "intrinsic_signed_evidence": np.asarray([1.0, -1.0]),
                    "signed_column_influence": np.asarray([0.0, -1.0]),
                    "attention_column_mean": np.asarray([0.0, 1.0]),
                },
                identity_atol=1e-12,
                identity_rtol=1e-12,
                normalization_atol=1e-12,
                normalization_rtol=1e-12,
            )

    def test_similarity_detects_attention_reranking(self):
        evidence = np.asarray([4.0, 3.0, 2.0, 1.0])
        attention = np.asarray([0.01, 0.01, 0.01, 0.97])
        control, _ = build_control_profile(
            {
                "name": "p",
                "sequence": "ACDE",
                "length": 4,
                "intrinsic_signed_evidence": evidence,
                "signed_column_influence": evidence * attention,
                "attention_column_mean": attention,
            },
            identity_atol=1e-12,
            identity_rtol=1e-12,
            normalization_atol=1e-12,
            normalization_rtol=1e-12,
        )
        positive = profile_similarity_rows(
            control, condition="c", seed=1, split="train"
        )[0]
        self.assertLess(positive["spearman_observed_uniform"], 0)
        self.assertEqual(positive["top_10pct_jaccard"], 0.0)
        self.assertGreater(
            positive["mean_high_influence_rank_displacement"], 0
        )

    def test_complete_absolute_correlation_and_g_isclose_are_defined(self):
        control = make_control(
            evidence=[4.0, 2.0, -3.0, -1.0],
            attention=[0.2500001, 0.4999999, 0.20, 0.05],
        )
        rows = profile_similarity_rows(
            control,
            condition="c",
            seed=1,
            split="train",
            g_uniform_atol=1e-6,
            g_uniform_rtol=0.0,
        )
        positive = rows[0]
        self.assertIn("pearson_complete_absolute_magnitude", positive)
        self.assertNotEqual(
            positive["pearson_complete_absolute_magnitude"],
            positive["pearson_observed_uniform"],
        )
        self.assertEqual(positive["g_fraction_isclose_1"], 0.5)
        self.assertEqual(positive["g_fraction_gt_1"], 0.5)
        self.assertEqual(positive["g_fraction_lt_1"], 0.0)
        self.assertAlmostEqual(
            positive["g_fraction_isclose_1"]
            + positive["g_fraction_gt_1"]
            + positive["g_fraction_lt_1"],
            1.0,
        )

    def test_invalid_profile_values_are_rejected(self):
        cases = [
            (
                "sum\\(B\\)",
                [1.0, -1.0],
                [0.4, 0.4],
            ),
            (
                "negative",
                [1.0, -1.0],
                [-0.1, 1.1],
            ),
            (
                "nonfinite",
                [np.nan, -1.0],
                [0.5, 0.5],
            ),
        ]
        for expected, evidence, attention in cases:
            evidence = np.asarray(evidence)
            attention = np.asarray(attention)
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    build_control_profile(
                        {
                            "name": "p",
                            "sequence": "AC",
                            "length": 2,
                            "intrinsic_signed_evidence": evidence,
                            "signed_column_influence": evidence * attention,
                            "attention_column_mean": attention,
                        },
                        identity_atol=1e-12,
                        identity_rtol=1e-12,
                        normalization_atol=1e-12,
                        normalization_rtol=1e-12,
                    )

    def test_malformed_source_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_columns = root / "missing.tsv"
            pd.DataFrame({"condition": ["c"]}).to_csv(
                missing_columns, sep="\t", index=False
            )
            with self.assertRaisesRegex(ValueError, "required columns"):
                load_manifest(missing_columns)

            source = root / "source.json"
            source.write_text("{}")
            duplicate = root / "duplicate.tsv"
            pd.DataFrame([
                {
                    "condition": "c", "seed": seed, "split": "train",
                    "json_gz": str(source), "protein_count": 1,
                    "residue_count": 2,
                }
                for seed in (1, 1, 2, 3)
            ]).to_csv(duplicate, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate manifest"):
                load_manifest(duplicate)

            incomplete = root / "incomplete.tsv"
            pd.DataFrame([
                {
                    "condition": "c", "seed": seed, "split": "train",
                    "json_gz": str(source), "protein_count": 1,
                    "residue_count": 2,
                }
                for seed in (1, 2)
            ]).to_csv(incomplete, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "expected seeds"):
                load_manifest(incomplete)

            missing_path = root / "missing_path.tsv"
            pd.DataFrame([
                {
                    "condition": "c", "seed": seed, "split": "train",
                    "json_gz": str(root / f"absent_{seed}.json"),
                    "protein_count": 1, "residue_count": 2,
                }
                for seed in (1, 2, 3)
            ]).to_csv(missing_path, sep="\t", index=False)
            with self.assertRaisesRegex(FileNotFoundError, "Missing contribution"):
                load_manifest(missing_path)

    def test_seed_order_and_sequence_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for case in ("order", "sequence"):
                rows = []
                for seed in (1, 2, 3):
                    first = make_control(name="protein_a")
                    second = make_control(name="protein_b")
                    profiles = [first, second]
                    if case == "order" and seed == 2:
                        profiles = [second, first]
                    if case == "sequence" and seed == 2:
                        profiles = [
                            make_control(name="protein_a", sequence="ACDF"),
                            second,
                        ]
                    path = root / f"{case}_seed_{seed}.json.gz"
                    write_compact(path, seed, profiles)
                    rows.append({
                        "condition": "condition_a",
                        "seed": seed,
                        "split": "train",
                        "profile_json_gz": str(path),
                        "protein_count": 2,
                        "residue_count": 8,
                    })
                manifests.append((case, pd.DataFrame(rows)))
            for case, manifest in manifests:
                with self.subTest(case=case):
                    with self.assertRaisesRegex(
                        ValueError,
                        "seed profiles differ in (name|sequence)",
                    ):
                        generate_seed_averaged_profiles(
                            manifest, root / f"output_{case}"
                        )

    def test_detector_accepts_control_manifests_and_uniform_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "control.json.gz"
            write_compact(compact, 1, [make_control()])
            manifest = root / "profile_manifest.tsv"
            pd.DataFrame([{
                "condition": "condition_a",
                "seed": 1,
                "split": "train",
                "profile_json_gz": str(compact),
            }]).to_csv(manifest, sep="\t", index=False)
            args = types.SimpleNamespace(
                manifest_tsv=str(manifest),
                input_json=None,
                conditions=None,
                splits=None,
            )
            specs = load_detector_inputs(args)
            self.assertEqual(specs[0].path, compact.resolve())
            metadata, profiles = iter_detector_profiles(
                compact, CONTROL_UNIFORM_FIELD
            )
            profile = list(profiles)[0]
            self.assertEqual(
                metadata["schema_version"],
                "esmfluc.uniform_attention_control.v1",
            )
            np.testing.assert_allclose(
                profile[CONTROL_UNIFORM_FIELD],
                profile["intrinsic_signed_evidence"] / profile["length"],
            )

    def test_stability_validator_accepts_only_matched_control_field_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_seed = root / "per_seed"
            mean = root / "mean"
            per_seed.mkdir()
            mean.mkdir()
            common = {
                "apex_method": "raw_mad_amplitude",
                "support_method": "half_intensity_merge",
                "terminal_exclusion": 0,
                "terminal_exclusion_fraction": 0.0,
                "amplitude_mad": 2.0,
            }
            (per_seed / "signed_band_parameters.json").write_text(json.dumps({
                **common,
                "influence_field": "uniform_signed_influence",
            }))
            mean_parameters = {
                **common,
                "influence_field": (
                    "seed_averaged_uniform_signed_influence"
                ),
            }
            mean_path = mean / "signed_band_parameters.json"
            mean_path.write_text(json.dumps(mean_parameters))
            args = types.SimpleNamespace(
                bands_csv=str(per_seed / "signed_bands.csv"),
                mean_bands_csv=str(mean / "signed_bands.csv"),
            )
            identifier, active, _paths = validated_parameter_set(args)
            self.assertIn("raw_amp_R2", identifier)
            self.assertEqual(active["amplitude_mad"], 2.0)

            mean_parameters["influence_field"] = (
                "seed_averaged_observed_signed_influence"
            )
            mean_path.write_text(json.dumps(mean_parameters))
            with self.assertRaisesRegex(ValueError, "not a matched profile pair"):
                validated_parameter_set(args)

    def test_stream_generation_and_three_seed_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            evidences = []
            observed = []
            attentions = [
                np.asarray([0.1, 0.2, 0.3, 0.4]),
                np.asarray([0.4, 0.3, 0.2, 0.1]),
                np.asarray([0.25, 0.25, 0.25, 0.25]),
            ]
            for seed in (1, 2, 3):
                evidence = np.asarray([seed, -2 * seed, 3 + seed, -4.0])
                attention = attentions[seed - 1]
                source = root / f"seed_{seed}.json"
                write_contribution(source, seed, evidence, attention)
                evidences.append(evidence)
                observed.append(evidence * attention)
                rows.append({
                    "condition": "condition_a",
                    "seed": seed,
                    "split": "train",
                    "json_gz": str(source),
                    "protein_count": 1,
                    "residue_count": 4,
                })
            output = root / "output"
            profile_manifest, audit = generate_per_seed_profiles(
                pd.DataFrame(rows),
                output,
                identity_atol=1e-12,
                identity_rtol=1e-12,
                normalization_atol=1e-12,
                normalization_rtol=1e-12,
                max_proteins_per_file=1,
            )
            averaged_manifest, averaging_audit = generate_seed_averaged_profiles(
                profile_manifest, output
            )

            self.assertEqual(audit["protein_profiles"], 3)
            self.assertTrue(audit["selection"]["debug_subset"])
            self.assertEqual(
                set(profile_manifest["run_scope"]), {"debug_subset"}
            )
            self.assertEqual(averaging_audit["protein_profiles"], 1)
            self.assertTrue(
                (output / "profiles" / "profile_manifest.tsv").is_file()
            )
            self.assertTrue(
                (
                    output / "profile_comparison"
                    / "per_protein_profile_similarity.csv.gz"
                ).is_file()
            )
            self.assertTrue(
                (
                    output / "profile_comparison"
                    / "per_protein_seed_averaged_profile_similarity.csv.gz"
                ).is_file()
            )
            per_seed_metadata, per_seed_iterator = iter_control_profiles(
                Path(profile_manifest.iloc[0]["profile_json_gz"]),
                fields=PROFILE_FIELDS,
            )
            self.assertEqual(per_seed_metadata["protein_count"], 1)
            self.assertEqual(len(list(per_seed_iterator)), 1)

            averaged_metadata, averaged_iterator = iter_control_profiles(
                Path(averaged_manifest.iloc[0]["profile_json_gz"]),
                fields=SEED_AVERAGED_FIELDS,
                expected_schema=(
                    "esmfluc.uniform_attention_control.seed_average.v1"
                ),
            )
            averaged = list(averaged_iterator)[0]
            self.assertEqual(averaged_metadata["source_seeds"], [1, 2, 3])
            self.assertEqual(averaged_manifest.iloc[0]["seed"], "mean_1_2_3")
            np.testing.assert_allclose(
                averaged["seed_averaged_intrinsic_signed_evidence"],
                np.mean(evidences, axis=0),
            )
            np.testing.assert_allclose(
                averaged["seed_averaged_observed_signed_influence"],
                np.mean(observed, axis=0),
            )
            self.assertFalse(np.allclose(
                averaged["seed_averaged_observed_signed_influence"],
                np.mean(evidences, axis=0) * np.mean(attentions, axis=0),
            ))
            np.testing.assert_allclose(
                averaged["seed_averaged_uniform_signed_influence"],
                np.mean(evidences, axis=0) / 4,
            )
            self.assertLess(
                averaging_audit["max_mean_attention_sum_abs_error"], 1e-12
            )
            self.assertLess(
                averaging_audit[
                    "max_mean_amplification_average_abs_error"
                ],
                1e-12,
            )
            self.assertLess(
                averaging_audit[
                    "max_mean_amplification_identity_abs_error"
                ],
                1e-12,
            )

            mean_metadata, mean_profiles = iter_detector_profiles(
                Path(averaged_manifest.iloc[0]["profile_json_gz"]),
                CONTROL_AVERAGED_UNIFORM_FIELD,
            )
            self.assertEqual(
                mean_metadata["schema_version"],
                "esmfluc.uniform_attention_control.seed_average.v1",
            )
            self.assertEqual(len(list(mean_profiles)), 1)
            _, shifted_mean_profiles = iter_detector_profiles(
                Path(averaged_manifest.iloc[0]["profile_json_gz"]),
                CONTROL_AVERAGED_SHIFTED_FIELD,
            )
            shifted_mean = list(shifted_mean_profiles)[0]
            self.assertEqual(
                shifted_mean[CONTROL_AVERAGED_SHIFTED_FIELD].shape, (4,)
            )
            _, shifted_seed_profiles = iter_detector_profiles(
                Path(profile_manifest.iloc[0]["profile_json_gz"]),
                CONTROL_SHIFTED_FIELD,
            )
            self.assertEqual(
                list(shifted_seed_profiles)[0][CONTROL_SHIFTED_FIELD].shape,
                (4,),
            )

            parameter_args = types.SimpleNamespace(
                conditions=["condition_a"],
                splits=["train"],
                max_proteins_per_file=1,
                identity_atol=1e-12,
                identity_rtol=1e-12,
                normalization_atol=1e-12,
                normalization_rtol=1e-12,
                g_uniform_atol=1e-6,
                g_uniform_rtol=1e-6,
                shift_random_seed=20260807,
            )
            write_parameters(
                output,
                manifest_path=root / "source_manifest.tsv",
                args=parameter_args,
                profile_manifest=profile_manifest,
                averaged_manifest=averaged_manifest,
                profile_audit=audit,
                averaging_audit=averaging_audit,
            )
            parameters = json.loads(
                (output / "uniform_attention_parameters.json").read_text()
            )
            self.assertEqual(
                parameters["status"],
                "detector-independent debug subset complete",
            )
            self.assertEqual(
                parameters["selection"]["max_proteins_per_file"], 1
            )
            self.assertEqual(
                parameters["selection"]["selected_conditions"],
                ["condition_a"],
            )


if __name__ == "__main__":
    unittest.main()
