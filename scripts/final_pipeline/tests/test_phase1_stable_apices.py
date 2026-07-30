from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from signed_band_analysis.analyze_signed_band_seed_reproducibility import (
    band_interval_iou,
    band_support_segments,
    band_support_width,
    circular_shift_bands,
    finite_sample_std,
    fixed_apex_matches,
    interval_iou_matches,
    match_band_lists,
    pair_metrics,
    validated_parameter_set,
    z_score,
)
from signed_band_analysis.extract_signed_contribution_bands import (
    APEX_OUTPUT_COLUMNS,
    BAND_OUTPUT_COLUMNS,
    ApexCall,
    ScalePeak,
    active_detector_parameters,
    assign_merged_half_intensity_support,
    assign_sign_constrained_support,
    audit_apex_support_catalog,
    build_apex_rows,
    build_band_rows,
    consolidate_multiscale_apices,
    detect_multiscale_apices,
    detect_raw_mad_amplitude_apices,
)


def call(sign: int, apex: int, left: int | None = None, right: int | None = None) -> ApexCall:
    peak = ScalePeak(
        sign=sign,
        scale=1,
        apex=apex,
        signed_smoothed_score=float(sign),
        prominence=1.0,
        robust_prominence=3.0,
        left_half_prominence=apex if left is None else left,
        right_half_prominence=apex if right is None else right,
    )
    return ApexCall(sign=sign, apex=apex, representative=peak, scale_peaks=(peak,))


def context(length: int) -> dict:
    return {
        "condition": "condition",
        "seed": "average",
        "split": "train",
        "json_path": "synthetic.json",
        "influence_field": "seed_averaged_signed_column_influence",
        "protein": "synthetic",
        "protein_length": length,
        "eligible_start": 0,
        "eligible_end": length,
        "scale_tolerance": 2,
    }


def rows(profile: np.ndarray, calls: list[ApexCall]) -> list[dict]:
    result, _audit = build_apex_rows(
        profile,
        np.full(len(profile), np.nan),
        np.full(len(profile), np.nan),
        "A" * len(profile),
        calls,
        context(len(profile)),
    )
    return result


def test_isolated_positive_and_negative_sign_runs() -> None:
    positive = np.array([0.0, 1.0, 4.0, 1.0, 0.0])
    negative = -positive
    positive_rows = rows(positive, [call(1, 2)])
    negative_rows = rows(negative, [call(-1, 2)])
    assert (positive_rows[0]["support_start_index_0based"],
            positive_rows[0]["support_end_index_0based_inclusive"]) == (1, 3)
    assert (negative_rows[0]["support_start_index_0based"],
            negative_rows[0]["support_end_index_0based_inclusive"]) == (1, 3)


def test_adjacent_opposite_sign_runs_and_zeros_never_overlap() -> None:
    profile = np.array([1.0, 4.0, 1.0, 0.0, -1.0, -5.0, -1.0])
    result = rows(profile, [call(1, 1), call(-1, 5)])
    assert result[0]["support_end_index_0based_inclusive"] == 2
    assert result[1]["support_start_index_0based"] == 4
    assert all(not (row["support_start_index_0based"] <= 3 <=
                    row["support_end_index_0based_inclusive"]) for row in result)


def test_two_positive_apices_split_at_unique_valley() -> None:
    profile = np.array([0.0, 5.0, 3.0, 1.0, 2.0, 6.0, 0.0])
    result = rows(profile, [call(1, 1), call(1, 5)])
    assert result[0]["support_end_index_0based_inclusive"] == 2
    assert result[1]["support_start_index_0based"] == 4
    assert not (result[0]["support_start_index_0based"] <= 3 <=
                result[0]["support_end_index_0based_inclusive"])


def test_tied_valley_plateau_is_left_unassigned() -> None:
    profile = np.array([0.0, 5.0, 1.0, 1.0, 4.0, 0.0])
    result = rows(profile, [call(1, 1), call(1, 4)])
    assert result[0]["support_end_index_0based_inclusive"] == 1
    assert result[1]["support_start_index_0based"] == 4


def test_sign_run_without_apex_is_unassigned() -> None:
    profile = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    assert assign_sign_constrained_support(profile, [], 0, len(profile)) == {}


def test_short_profile_has_no_detected_apex() -> None:
    profile = np.array([1.0, 2.0])
    assert detect_multiscale_apices(profile, [1, 3, 5], 0, 2, 2.5, 3) == []


def test_raw_amplitude_apices_use_amplitude_not_prominence() -> None:
    # The second local maximum is only weakly isolated from its shoulders, but
    # it is retained because its raw amplitude exceeds the MAD cutoff.
    profile = np.array([
        0.0, 0.1, 5.0, 0.1, 3.2, 3.0, 3.1, 0.1,
        0.0, 0.0, 0.0, 0.0, -5.0, 0.0,
    ])
    detected = detect_raw_mad_amplitude_apices(
        profile, 0, len(profile), 1, len(profile) - 1, amplitude_mad=2.0
    )
    assert (1, 4) in [(item.sign, item.apex) for item in detected]
    assert all(
        abs(profile[item.apex]) / (
            1.4826 * np.median(np.abs(profile - np.median(profile)))
        ) >= 2.0
        for item in detected
    )


def test_raw_detector_includes_first_and_last_eligible_residues() -> None:
    profile = np.array([0.0, 5.0, 0.1, -0.1, 0.1, -0.1, -5.0, 0.0])
    detected = detect_raw_mad_amplitude_apices(
        profile,
        search_start=0,
        search_end=len(profile),
        eligible_start=1,
        eligible_end=len(profile) - 1,
        amplitude_mad=2.0,
    )
    assert [(item.sign, item.apex) for item in detected] == [(1, 1), (-1, 6)]


def test_half_intensity_intervals_merge_same_sign_overlaps() -> None:
    profile = np.array([0.0, 5.0, 4.0, 4.5, 0.0])
    merged, assignments, members = assign_merged_half_intensity_support(
        profile, [call(1, 1), call(1, 3)], 0, len(profile)
    )
    assert [(item.sign, item.apex) for item in merged] == [(1, 1)]
    assert (assignments[0].start, assignments[0].end) == (1, 3)
    assert members[0] == [1, 3]


def test_band_magnitude_fraction_rank_and_percentile_are_stored() -> None:
    profile = np.array([0.0, 5.0, 3.0, 0.0, -2.0, 0.0, 4.0, 0.0])
    result, _audit = build_apex_rows(
        profile,
        np.full(len(profile), np.nan),
        np.full(len(profile), np.nan),
        "A" * len(profile),
        [call(1, 1), call(-1, 4), call(1, 6)],
        context(len(profile)),
        support_method="half_intensity_merge",
    )
    positive = [row for row in result if row["sign"] == 1]
    assert positive[0]["apex_absolute_influence"] == 5.0
    assert positive[0]["band_integrated_magnitude"] == 8.0
    assert (
        positive[0]["band_fraction_of_eligible_absolute_influence"]
        == 8.0 / 14.0
    )
    assert positive[0]["absolute_apex_rank_within_protein_sign"] == 1
    assert positive[0]["absolute_apex_percentile_within_protein_sign"] == 100.0
    assert positive[1]["absolute_apex_rank_within_protein_sign"] == 2
    assert positive[1]["absolute_apex_percentile_within_protein_sign"] == 50.0
    assert [row["absolute_apex_rank_within_protein"] for row in result] == [1, 3, 2]
    assert [
        row["integrated_magnitude_rank_within_protein"] for row in result
    ] == [1, 3, 2]
    assert positive[0]["band_id"] == positive[0]["apex_id"]
    assert (
        positive[0]["start_index_0based"]
        == positive[0]["support_start_index_0based"]
    )
    assert (
        positive[0]["end_index_0based_inclusive"]
        == positive[0]["support_end_index_0based_inclusive"]
    )
    assert positive[0]["band_width"] == positive[0]["support_width"]


def test_primary_band_schema_contains_all_legacy_band_columns() -> None:
    assert set(BAND_OUTPUT_COLUMNS).issubset(APEX_OUTPUT_COLUMNS)


def test_band_fraction_uses_the_eligible_region_denominator() -> None:
    profile = np.array([100.0, 0.0, 5.0, 3.0, 0.0, 100.0])
    local_context = context(len(profile))
    local_context["eligible_start"] = 1
    local_context["eligible_end"] = len(profile) - 1
    result, _audit = build_apex_rows(
        profile,
        np.full(len(profile), np.nan),
        np.full(len(profile), np.nan),
        "A" * len(profile),
        [call(1, 2)],
        local_context,
        support_method="half_intensity_merge",
    )
    assert result[0]["band_integrated_magnitude"] == 8.0
    assert result[0]["band_fraction_of_eligible_absolute_influence"] == 1.0


def test_apex_near_eligible_boundary_stays_inside() -> None:
    profile = np.array([9.0, 5.0, 1.0, 0.0])
    assignment = assign_sign_constrained_support(
        profile, [call(1, 0)], 0, len(profile)
    )[0]
    assert assignment.start == 0
    assert assignment.end == 2


def test_cross_scale_apex_movement_consolidates_without_support() -> None:
    profile = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 7.0, 2.0, 0.0])
    peaks = [
        ScalePeak(1, 1, 4, 8.0, 4.0, 5.0, 3, 5),
        ScalePeak(1, 3, 5, 7.0, 3.0, 4.0, 4, 6),
    ]
    calls = consolidate_multiscale_apices(
        profile, peaks, min_scales=2, tolerance=2,
        eligible_start=0, eligible_end=len(profile),
    )
    assert [(item.sign, item.apex) for item in calls] == [(1, 4)]
    before = [(item.sign, item.apex) for item in calls]
    assign_sign_constrained_support(profile, calls, 0, len(profile))
    assert [(item.sign, item.apex) for item in calls] == before


def test_detector_is_symmetric_under_sign_inversion() -> None:
    profile = np.array([
        0.0, 0.2, 1.0, 5.0, 1.0, 0.2, 0.0,
        -0.2, -1.0, -5.0, -1.0, -0.2, 0.0,
    ])
    positive = detect_multiscale_apices(
        profile, [1, 3, 5], 0, len(profile), 0.5, 1
    )
    negative = detect_multiscale_apices(
        -profile, [1, 3, 5], 0, len(profile), 0.5, 1
    )
    calls_a = consolidate_multiscale_apices(
        profile, positive, 2, 2, 0, len(profile)
    )
    calls_b = consolidate_multiscale_apices(
        -profile, negative, 2, 2, 0, len(profile)
    )
    assert [(item.apex, item.sign) for item in calls_a] == [
        (item.apex, -item.sign) for item in calls_b
    ]


def test_support_audit_rejects_overlap_and_sign_crossing() -> None:
    profile = np.array([1.0, 4.0, 1.0, 0.0, -1.0])
    good = rows(profile, [call(1, 1), call(-1, 4)])
    audit = audit_apex_support_catalog(
        good, profile, 0, len(profile), [(1, 1), (-1, 4)]
    )
    assert audit["n_multiply_assigned_residues"] == 0
    bad = [dict(good[0])]
    bad[0]["support_end_index_0based_inclusive"] = 4
    bad[0]["support_width"] = 5
    try:
        audit_apex_support_catalog(bad, profile, 0, len(profile))
    except ValueError as error:
        assert "sign/zero" in str(error)
    else:
        raise AssertionError("Sign-crossing support should fail its audit")


def test_legacy_half_prominence_behavior_is_preserved() -> None:
    profile = np.array([0.0, 1.0, 5.0, 1.0, -1.0])
    legacy_call = call(1, 2, left=1, right=4)
    result = build_band_rows(
        profile,
        np.ones(len(profile)),
        np.ones(len(profile)),
        "AAAAA",
        clusters=None,
        context=context(len(profile)),
        apex_calls=[legacy_call],
    )
    assert result[0]["start_index_0based"] == 1
    assert result[0]["end_index_0based_inclusive"] == 4
    assert result[0]["band_width"] == 4


def test_fixed_seed_matching_is_one_to_one_and_width_independent() -> None:
    a = [
        {
            "apex_index_0based": 10,
            "start_index_0based": 0,
            "end_index_0based_inclusive": 20,
            "band_width": 21,
            "apex_signed_column_influence": 3.0,
            "representative_robust_prominence": 3.0,
        },
        {
            "apex_index_0based": 12,
            "start_index_0based": 12,
            "end_index_0based_inclusive": 12,
            "band_width": 1,
            "apex_signed_column_influence": 2.0,
            "representative_robust_prominence": 2.0,
        },
    ]
    b = [{
        "apex_index_0based": 11,
        "start_index_0based": 11,
        "end_index_0based_inclusive": 11,
        "band_width": 1,
        "apex_signed_column_influence": 1.0,
        "representative_robust_prominence": 1.0,
    }]
    matches = match_band_lists(a, b, 2, 10, matching_method="fixed_apex")
    assert len(matches) == 1
    far = [dict(b[0], apex_index_0based=16, band_width=100)]
    assert match_band_lists(a, far, 2, 10, matching_method="fixed_apex") == []


def test_interval_matching_retains_a_band_when_merged_apices_shift() -> None:
    mean = pd.DataFrame({
        "apex_index_0based": [19],
        "apex_signed_column_influence": [-0.015],
        "support_start_index_0based": [16],
        "support_end_index_0based_inclusive": [28],
    })
    seed = pd.DataFrame({
        "apex_index_0based": [25],
        "start_index_0based": [16],
        "end_index_0based_inclusive": [29],
        "apex_signed_column_influence": [-0.016],
    })
    matches = interval_iou_matches(mean, seed, min_interval_iou=0.5)
    seed_index, interval_iou = matches[0]
    assert seed_index == 0
    assert np.isclose(interval_iou, 13 / 14)
    assert fixed_apex_matches(mean, seed, tolerance=2) == {}


def circular_band(
    apex: int,
    start: int,
    end: int,
    width: int,
    band_id: str = "band",
    sign: int = 1,
    magnitude: float = 2.0,
) -> dict:
    return {
        "band_id": band_id,
        "seed": 1,
        "sign": sign,
        "apex_index_0based": apex,
        "apex_residue_1based": apex + 1,
        "start_index_0based": start,
        "end_index_0based_inclusive": end,
        "band_width": width,
        "apex_signed_column_influence": sign * magnitude,
        "band_integrated_magnitude": magnitude * width,
        "representative_robust_prominence": 3.0,
    }


def test_circular_shift_nonwrapping_support_is_exact() -> None:
    shifted = circular_shift_bands(
        [circular_band(4, 3, 5, 3)],
        eligible_start=1,
        eligible_end=11,
        offset=2,
        shift_intervals=True,
    )[0]
    assert shifted["apex_index_0based"] == 6
    assert shifted["start_index_0based"] == 5
    assert shifted["end_index_0based_inclusive"] == 7
    assert band_support_segments(shifted) == ((5, 7),)
    assert band_support_width(shifted) == shifted["band_width"] == 3


def test_circular_shift_right_boundary_wraps_without_clipping() -> None:
    shifted = circular_shift_bands(
        [circular_band(8, 7, 9, 3)],
        eligible_start=0,
        eligible_end=10,
        offset=1,
        shift_intervals=True,
    )[0]
    assert shifted["apex_index_0based"] == 9
    assert shifted["start_index_0based"] == 8
    assert shifted["end_index_0based_inclusive"] == 0
    assert band_support_segments(shifted) == ((0, 0), (8, 9))
    assert band_support_width(shifted) == shifted["band_width"] == 3


def test_every_circular_offset_preserves_the_complete_pattern() -> None:
    bands = [
        circular_band(2, 1, 3, 3, "a", 1, 2.0),
        circular_band(6, 6, 7, 2, "b", 1, 5.0),
    ]
    original_spacing = (
        bands[1]["apex_index_0based"] - bands[0]["apex_index_0based"]
    ) % 10
    for offset in range(10):
        shifted = circular_shift_bands(
            bands, 0, 10, offset, shift_intervals=True
        )
        assert len(shifted) == len(bands)
        shifted_spacing = (
            shifted[1]["apex_index_0based"]
            - shifted[0]["apex_index_0based"]
        ) % 10
        assert shifted_spacing == original_spacing
        for original, moved in zip(bands, shifted):
            assert band_support_width(moved) == original["band_width"]
            assert moved["band_width"] == original["band_width"]
            assert moved["band_id"] == original["band_id"]
            assert moved["sign"] == original["sign"]
            assert (
                moved["apex_signed_column_influence"]
                == original["apex_signed_column_influence"]
            )
            assert (
                moved["band_integrated_magnitude"]
                == original["band_integrated_magnitude"]
            )
            apex = moved["apex_index_0based"]
            assert any(
                start <= apex <= end
                for start, end in band_support_segments(moved)
            )
            assert all(
                0 <= start <= end < 10
                for start, end in band_support_segments(moved)
            )


def test_interval_iou_handles_linear_and_wrapped_supports() -> None:
    ordinary_a = circular_band(3, 2, 5, 4, "ordinary_a")
    ordinary_b = circular_band(5, 4, 7, 4, "ordinary_b")
    no_overlap = circular_band(4, 3, 4, 2, "none")
    wrapped_a = circular_shift_bands(
        [circular_band(8, 7, 9, 3, "wrapped_a")],
        0, 10, 1, True,
    )[0]
    # Support is {0, 8, 9}.
    wrapped_b = circular_shift_bands(
        [circular_band(9, 8, 9, 2, "wrapped_b")],
        0, 10, 1, True,
    )[0]
    # Support is {0, 9}.
    ordinary_edge = circular_band(1, 0, 2, 3, "ordinary_edge")
    assert np.isclose(band_interval_iou(ordinary_a, ordinary_b), 2 / 6)
    assert np.isclose(band_interval_iou(wrapped_a, ordinary_edge), 1 / 5)
    assert np.isclose(band_interval_iou(wrapped_a, wrapped_b), 2 / 3)
    assert band_interval_iou(wrapped_a, no_overlap) == 0.0
    assert band_interval_iou(wrapped_a, wrapped_a) == 1.0


def test_seeded_circular_null_is_reproducible() -> None:
    observed = [circular_band(2, 1, 3, 3, "observed")]
    shifted_source = [circular_band(7, 6, 8, 3, "shifted")]

    def null_values(random_seed: int) -> list[float]:
        rng = np.random.default_rng(random_seed)
        values = []
        for _ in range(20):
            shifted = circular_shift_bands(
                shifted_source,
                0,
                10,
                int(rng.integers(0, 10)),
                shift_intervals=True,
            )
            values.append(pair_metrics(
                observed, shifted, 2, 10, "interval_iou", 0.5
            )["jaccard"])
        return values

    assert null_values(123) == null_values(123)


def test_single_block_shift_statistics_are_na_without_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert np.isnan(finite_sample_std(np.asarray([0.5])))
        assert np.isnan(z_score(0.8, np.asarray([0.5])))
    assert caught == []


def test_mean_to_seed_matching_is_one_to_one() -> None:
    mean = pd.DataFrame({
        "apex_index_0based": [10, 12],
        "apex_signed_column_influence": [2.0, 2.0],
        "representative_robust_prominence": [2.0, 2.0],
    })
    seed = pd.DataFrame({
        "apex_index_0based": [11],
        "apex_signed_column_influence": [3.0],
        "representative_robust_prominence": [3.0],
    })
    assert len(fixed_apex_matches(mean, seed, 2)) == 1


def test_matching_ties_use_amplitude_not_prominence() -> None:
    mean = pd.DataFrame({
        "apex_index_0based": [10],
        "apex_signed_column_influence": [5.0],
    })
    seed = pd.DataFrame({
        "apex_index_0based": [9, 11],
        "apex_signed_column_influence": [1.0, 2.0],
        "representative_robust_prominence": [100.0, 0.0],
    })
    matches = fixed_apex_matches(mean, seed, 2)
    assert int(seed.loc[matches[0], "apex_index_0based"]) == 11

    seed["apex_signed_column_influence"] = [2.0, 2.0]
    matches = fixed_apex_matches(mean, seed, 2)
    assert int(seed.loc[matches[0], "apex_index_0based"]) == 9


def test_parameter_set_id_is_derived_and_mismatches_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
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
            "selected_proteins": ["protein"],
        }
        (per_seed / "signed_band_parameters.json").write_text(json.dumps({
            **common,
            "influence_field": "signed_column_influence",
        }))
        mean_parameters = {
            **common,
            "influence_field": "seed_averaged_signed_column_influence",
        }
        mean_path = mean / "signed_band_parameters.json"
        mean_path.write_text(json.dumps(mean_parameters))
        args = SimpleNamespace(
            bands_csv=str(per_seed / "signed_bands.csv"),
            mean_bands_csv=str(mean / "signed_bands.csv"),
        )
        identifier, active, _paths = validated_parameter_set(args)
        assert "R2" in identifier
        assert active["amplitude_mad"] == 2.0

        mean_parameters["amplitude_mad"] = 1.5
        mean_path.write_text(json.dumps(mean_parameters))
        try:
            validated_parameter_set(args)
        except ValueError as error:
            assert "do not match" in str(error)
        else:
            raise AssertionError("Mismatched extractor parameters must fail")

        per_seed_parameters = {
            **common,
            "amplitude_mad": 1.5,
            "influence_field": "signed_column_influence",
        }
        (per_seed / "signed_band_parameters.json").write_text(
            json.dumps(per_seed_parameters)
        )
        identifier_15, active_15, _paths = validated_parameter_set(args)
        assert "R1.5" in identifier_15
        assert identifier_15 != identifier
        assert active_15["amplitude_mad"] == 1.5


def test_raw_metadata_omits_inactive_multiscale_parameters() -> None:
    args = SimpleNamespace(
        apex_method="raw_mad_amplitude",
        support_method="half_intensity_merge",
        terminal_exclusion=0,
        terminal_exclusion_fraction=0.0,
        amplitude_mad=2.0,
        smooth_windows=[1, 3, 5],
        min_scales=2,
        scale_tolerance=2,
        prominence_mad=2.5,
        min_peak_distance=3,
    )
    active = active_detector_parameters(args)
    assert active["amplitude_mad"] == 2.0
    assert "prominence_mad" not in active
    assert "smooth_windows" not in active
