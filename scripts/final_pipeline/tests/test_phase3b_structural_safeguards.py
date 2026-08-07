import types
import unittest

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_phase3b_structural_safeguards import (
    BASE_FEATURES,
    INCREMENTS,
    LOCAL_ANNOTATION_FEATURES,
    PROVENANCE_KEY,
    PROVENANCE_VALUE_COLUMNS,
    STAGES,
    audit_prediction_table,
    bootstrap_predictions,
    multiply_weights_by_protein,
    performance_from_predictions,
    protein_multiplicity_matrix,
    safe_weighted_auroc,
    signed_cohort_data,
    validate_candidate_provenance,
)
from signed_band_analysis.analyze_signed_band_external_structure import (
    reconstruct_matched_strata,
)


def cohort_args():
    return types.SimpleNamespace(
        reference_protein_coverage=0.8,
        reference_segment_resolution=0.8,
        strict_graph_protein_coverage=0.95,
    )


def candidate_row(
    *,
    protein,
    segment,
    selected_positive=False,
    clean_control=False,
    fully_resolved=True,
):
    return {
        "condition": "condition_1",
        "split": "test",
        "protein": protein,
        "q8": "C",
        "segment_id": f"{protein}_{segment}",
        "selected_positive": selected_positive,
        "selected_negative": False,
        "clean_control": clean_control,
        "accepted_structure_mapping": True,
        "protein_input_coverage": 1.0,
        "segment_resolved_fraction": 1.0 if fully_resolved else 0.8,
        "segment_ca_fully_resolved": fully_resolved,
        "experimental_geometry_complete": fully_resolved,
        "network_coordinate_complete": fully_resolved,
    }


def synthetic_predictions(conditions=("condition_1",), n_proteins=20):
    rows = []
    for condition in conditions:
        for protein_index in range(n_proteins):
            protein = f"p{protein_index:02d}"
            for selected in (0, 1):
                segment = f"{protein}_{selected}"
                for stage in STAGES:
                    if stage == "03_add_experimental_pdb_geometry":
                        score = 0.9 if selected else 0.1
                    elif stage in {
                        "04_add_contact_network", "05_add_ecod_domain",
                    }:
                        score = 0.92 if selected else 0.08
                    elif stage == "02_add_local_sequence_or_annotation_geometry":
                        score = (
                            0.60 + 0.01 * protein_index
                            if selected else 0.40 + 0.01 * protein_index
                        )
                    elif stage == "01_base_biophysics":
                        score = (
                            0.55 + 0.01 * protein_index
                            if selected else 0.45 + 0.01 * protein_index
                        )
                    else:
                        score = 0.5
                    rows.append({
                        "condition": condition,
                        "sign": 1,
                        "evaluation_split": "test",
                        "protein": protein,
                        "q8": "C",
                        "segment_id": segment,
                        "selected": selected,
                        "sample_weight": 0.5,
                        "structural_cohort": "reference_80",
                        "model_training_cohort": "reference_80",
                        "stage": stage,
                        "predicted_probability": score,
                        "fitted_model_identifier": f"{condition}_{stage}",
                        "source_candidate_table": "/tmp/candidates.csv.gz",
                        "source_external_features": "/tmp/external.csv.gz",
                    })
    return pd.DataFrame(rows)


class ProvenanceAuditTests(unittest.TestCase):
    @staticmethod
    def provenance_frame():
        row = {
            "condition": "condition_1",
            "split": "test",
            "protein": "p1",
            "segment_id": "segment_1",
            "protein_length": 20,
            "q8": "C",
            "start_index_0based": 3,
            "end_index_0based_exclusive": 7,
            "selected_positive": True,
            "selected_negative": False,
            "clean_control": False,
        }
        for feature in set(BASE_FEATURES + LOCAL_ANNOTATION_FEATURES):
            row[feature] = 0.25
        return pd.DataFrame([row])[PROVENANCE_KEY + PROVENANCE_VALUE_COLUMNS]

    def test_exact_candidate_derivative_passes(self):
        candidates = self.provenance_frame()
        report = validate_candidate_provenance(candidates.copy(), candidates)
        self.assertTrue(report["passed"])

    def test_changed_selection_is_rejected(self):
        candidates = self.provenance_frame()
        external = candidates.copy()
        external.loc[0, "selected_positive"] = False
        with self.assertRaises(ValueError):
            validate_candidate_provenance(external, candidates)


class ResolutionCohortTests(unittest.TestCase):
    def test_single_class_stratum_reconstructs_to_empty(self):
        frame = pd.DataFrame([{
            "condition": "c", "split": "test", "protein": "p",
            "q8": "C", "sign": 1, "selected": 1,
        }])
        self.assertTrue(reconstruct_matched_strata(frame).empty)

    def test_cases_and_controls_are_filtered_before_strata_reconstruction(self):
        frame = pd.DataFrame([
            candidate_row(protein="p1", segment="case", selected_positive=True),
            candidate_row(protein="p1", segment="control", clean_control=True),
            candidate_row(protein="p2", segment="case", selected_positive=True),
            candidate_row(
                protein="p2", segment="partial_control", clean_control=True,
                fully_resolved=False,
            ),
        ])
        result = signed_cohort_data(frame, "fully_resolved_local", cohort_args())
        self.assertEqual(set(result.protein), {"p1"})
        self.assertEqual(set(result.selected), {0, 1})

    def test_no_partially_resolved_control_remains(self):
        frame = pd.DataFrame([
            candidate_row(protein="p1", segment="case", selected_positive=True),
            candidate_row(protein="p1", segment="control", clean_control=True),
            candidate_row(
                protein="p1", segment="partial", clean_control=True,
                fully_resolved=False,
            ),
        ])
        result = signed_cohort_data(frame, "fully_resolved_local", cohort_args())
        controls = result[result.selected == 0]
        self.assertTrue(controls.segment_ca_fully_resolved.all())
        self.assertEqual(list(controls.segment_id), ["p1_control"])


class ProteinBootstrapTests(unittest.TestCase):
    def test_sampling_one_protein_includes_all_segments(self):
        frame = pd.DataFrame({
            "protein": ["p1", "p1", "p2"],
            "sample_weight": [0.2, 0.3, 0.4],
        })
        weights = multiply_weights_by_protein(frame, {"p1": 1, "p2": 0})
        np.testing.assert_allclose(weights, [0.2, 0.3, 0.0])

    def test_sampling_protein_twice_doubles_all_segment_weights(self):
        frame = pd.DataFrame({
            "protein": ["p1", "p1"],
            "sample_weight": [0.2, 0.3],
        })
        weights = multiply_weights_by_protein(frame, {"p1": 2})
        np.testing.assert_allclose(weights, [0.4, 0.6])

    def test_one_class_replicate_is_invalid(self):
        value, valid, reason = safe_weighted_auroc(
            [0, 1], [0.1, 0.9], [1.0, 0.0]
        )
        self.assertTrue(np.isnan(value))
        self.assertFalse(valid)
        self.assertEqual(reason, "one_outcome_class")

    def test_same_seed_reproduces_identical_multiplicities(self):
        proteins = ["p1", "p2", "p3"]
        first = protein_multiplicity_matrix(proteins, 10, 123)
        second = protein_multiplicity_matrix(proteins, 10, 123)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_smaller_and_expanded_models_use_same_draw(self):
        predictions = synthetic_predictions(n_proteins=8)
        replicates, _ = bootstrap_predictions(
            predictions,
            n_replicates=5,
            random_seed=123,
            include_validation=False,
            include_refit=False,
        )
        for _, group in replicates.groupby("replicate"):
            self.assertEqual(group.draw_checksum.nunique(), 1)

    def test_synchronized_conditions_use_identical_draws(self):
        predictions = synthetic_predictions(
            conditions=("condition_1", "condition_2"), n_proteins=8
        )
        replicates, synchronized = bootstrap_predictions(
            predictions,
            n_replicates=5,
            random_seed=123,
            include_validation=False,
            include_refit=False,
        )
        for _, group in replicates.groupby("replicate"):
            self.assertEqual(group.draw_checksum.nunique(), 1)
        self.assertTrue((synchronized.n_valid_conditions == 2).all())

    def test_reference_and_full_cohorts_share_draws(self):
        reference = synthetic_predictions(n_proteins=8)
        fully_resolved = reference[
            reference.protein.isin([f"p{i:02d}" for i in range(6)])
        ].copy()
        fully_resolved["structural_cohort"] = "fully_resolved_local"
        predictions = pd.concat([reference, fully_resolved], ignore_index=True)
        replicates, _ = bootstrap_predictions(
            predictions,
            n_replicates=5,
            random_seed=123,
            include_validation=False,
            include_refit=False,
        )
        for _, group in replicates.groupby("replicate"):
            self.assertEqual(group.draw_checksum.nunique(), 1)
        full_rows = replicates[
            replicates.structural_cohort == "fully_resolved_local"
        ]
        self.assertTrue((full_rows.n_proteins_in_universe == 8).all())
        self.assertTrue((full_rows.n_eligible_proteins_in_cohort == 6).all())

    def test_geometry_cohort_is_in_primary_fixed_model_bootstrap(self):
        reference = synthetic_predictions(n_proteins=8)
        geometry = reference.copy()
        geometry["structural_cohort"] = "fully_resolved_geometry"
        predictions = pd.concat([reference, geometry], ignore_index=True)
        replicates, _ = bootstrap_predictions(
            predictions,
            n_replicates=3,
            random_seed=123,
            include_validation=False,
            include_refit=False,
        )
        self.assertIn(
            "fully_resolved_geometry",
            set(replicates.structural_cohort),
        )

    def test_expected_condition_mismatch_is_rejected(self):
        predictions = synthetic_predictions(n_proteins=8)
        with self.assertRaises(ValueError):
            bootstrap_predictions(
                predictions,
                n_replicates=3,
                random_seed=123,
                include_validation=False,
                include_refit=False,
                expected_conditions=("condition_1", "condition_2"),
            )

    def test_known_positive_pdb_increment_has_positive_interval(self):
        predictions = synthetic_predictions(n_proteins=30)
        replicates, _ = bootstrap_predictions(
            predictions,
            n_replicates=200,
            random_seed=123,
            include_validation=False,
            include_refit=False,
        )
        values = replicates[
            (replicates.increment == "delta_experimental_pdb_geometry")
            & replicates.valid
        ].delta_auroc
        self.assertGreater(values.quantile(0.025), 0)

    def test_predictions_reconstruct_weighted_point_estimate(self):
        predictions = synthetic_predictions(n_proteins=10)
        reconstructed = performance_from_predictions(predictions)
        stage = reconstructed[
            reconstructed.stage == "03_add_experimental_pdb_geometry"
        ]
        self.assertEqual(len(stage), 1)
        self.assertAlmostEqual(stage.weighted_auroc.iloc[0], 1.0)

    def test_experimental_increment_uses_corrected_adjacent_stages(self):
        self.assertEqual(
            INCREMENTS["delta_experimental_pdb_geometry"],
            (
                "02_add_local_sequence_or_annotation_geometry",
                "03_add_experimental_pdb_geometry",
            ),
        )

    def test_bootstrap_universe_is_proteins_not_segments(self):
        predictions = synthetic_predictions(n_proteins=7)
        replicates, _ = bootstrap_predictions(
            predictions,
            n_replicates=3,
            random_seed=123,
            include_validation=False,
            include_refit=False,
        )
        self.assertTrue((replicates.n_proteins_in_universe == 7).all())

    def test_duplicate_saved_prediction_stage_fails_audit(self):
        predictions = synthetic_predictions(n_proteins=4)
        duplicated = pd.concat(
            [predictions, predictions.iloc[[0]]], ignore_index=True
        )
        report = audit_prediction_table(duplicated)
        self.assertFalse(report["passed"])
        self.assertEqual(report["duplicate_prediction_stage_rows"], 1)


if __name__ == "__main__":
    unittest.main()
