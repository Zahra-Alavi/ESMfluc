#!/usr/bin/env python3
"""Call positive and negative signed-influence apices and intervals.

The publication contribution files are very large because they contain two LxL
matrices.  This script scans their compact JSON representation and materializes
only the three O(L) key profiles needed here:

  intrinsic_signed_evidence s_j
  signed_column_influence   I_j
  attention_column_mean     B_j

The scanner deliberately skips contribution_matrix and attention_matrix without
parsing their numeric elements.  The legacy multiscale/half-prominence method
is preserved for exact comparison.  The simplified Phase 1 method selects raw
local extrema by absolute amplitude relative to a whole-profile MAD scale, then
uses half-apex-intensity intervals and merges overlapping same-sign intervals.
Prominence is reported as descriptive metadata but never determines inclusion.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths


DEFAULT_INFLUENCE_FIELD = "signed_column_influence"
AVERAGED_INFLUENCE_FIELD = "seed_averaged_signed_column_influence"
CONTROL_OBSERVED_FIELD = "observed_signed_influence"
CONTROL_UNIFORM_FIELD = "uniform_signed_influence"
CONTROL_AVERAGED_OBSERVED_FIELD = "seed_averaged_observed_signed_influence"
CONTROL_AVERAGED_UNIFORM_FIELD = "seed_averaged_uniform_signed_influence"
INFLUENCE_FIELDS = (
    DEFAULT_INFLUENCE_FIELD,
    AVERAGED_INFLUENCE_FIELD,
    CONTROL_OBSERVED_FIELD,
    CONTROL_UNIFORM_FIELD,
    CONTROL_AVERAGED_OBSERVED_FIELD,
    CONTROL_AVERAGED_UNIFORM_FIELD,
)
SIGNED_CONTRIBUTION_SCHEMA = "esmfluc.signed_contributions.v1"
CONTROL_SCHEMA = "esmfluc.uniform_attention_control.v1"
CONTROL_SEED_AVERAGE_SCHEMA = (
    "esmfluc.uniform_attention_control.seed_average.v1"
)

BAND_OUTPUT_COLUMNS = [
    "condition", "seed", "split", "json_path", "influence_field",
    "protein", "protein_length",
    "sign", "label", "apex_index_0based", "apex_residue_1based",
    "apex_amino_acid", "start_index_0based", "end_index_0based_inclusive",
    "start_residue_1based", "end_residue_1based_inclusive", "band_width",
    "band_sequence", "apex_signed_column_influence",
    "apex_intrinsic_signed_evidence", "apex_attention_column_mean",
    "band_signed_influence_sum", "band_signed_influence_mean",
    "band_absolute_influence_sum", "representative_smoothing_window",
    "representative_prominence", "representative_robust_prominence",
    "n_persistent_scales", "persistent_scales", "scale_apex_indices_0based",
    "distance_to_nearest_terminus", "near_terminal_10_or_2pct",
    "eligible_start_index_0based", "eligible_end_index_0based_exclusive",
    "band_number_within_protein", "band_id",
]

APEX_OUTPUT_COLUMNS = [
    "condition", "seed", "split", "json_path", "influence_field",
    "protein", "protein_length", "sign", "label", "object_type", "band_id",
    "band_number_within_protein", "apex_id", "apex_number_within_protein",
    "apex_index_0based", "apex_residue_1based",
    "apex_amino_acid", "apex_signed_column_influence",
    "apex_intrinsic_signed_evidence", "apex_attention_column_mean",
    "representative_smoothing_window", "representative_prominence",
    "representative_robust_prominence", "n_persistent_scales",
    "persistent_scales", "scale_apex_indices_0based",
    "support_start_index_0based", "support_end_index_0based_inclusive",
    "support_start_residue_1based", "support_end_residue_1based_inclusive",
    "support_width", "support_sequence", "support_signed_influence_sum",
    "support_signed_influence_mean", "support_absolute_influence_sum",
    "start_index_0based", "end_index_0based_inclusive",
    "start_residue_1based", "end_residue_1based_inclusive", "band_width",
    "band_sequence", "band_signed_influence_sum", "band_signed_influence_mean",
    "band_absolute_influence_sum",
    "sign_run_start_index_0based", "sign_run_end_index_0based_inclusive",
    "distance_to_nearest_terminus", "near_terminal_10_or_2pct",
    "eligible_start_index_0based", "eligible_end_index_0based_exclusive",
    "support_method", "detector_version", "apex_detection_method",
    "apex_absolute_influence", "apex_standardized_magnitude_R_p",
    "band_integrated_magnitude",
    "band_fraction_of_eligible_absolute_influence",
    "absolute_apex_rank_within_protein_sign",
    "absolute_apex_percentile_within_protein_sign",
    "absolute_apex_rank_within_protein",
    "integrated_magnitude_rank_within_protein_sign",
    "integrated_magnitude_percentile_within_protein_sign",
    "integrated_magnitude_rank_within_protein",
    "apex_abs_over_robust_scale", "n_merged_candidate_apices",
    "merged_candidate_apex_indices_0based",
]

DETECTOR_VERSION = "phase1_stable_apices_v1"
RAW_AMPLITUDE_DETECTOR_VERSION = "phase1_raw_amplitude_half_intensity_v1"


@dataclass(frozen=True)
class InputSpec:
    path: Path
    condition: str | None = None
    seed: int | str | None = None
    split: str | None = None


@dataclass(frozen=True)
class ScalePeak:
    sign: int
    scale: int
    apex: int
    signed_smoothed_score: float
    prominence: float
    robust_prominence: float
    left_half_prominence: int
    right_half_prominence: int


@dataclass(frozen=True)
class ApexCall:
    """One consolidated multiscale apex, before any support is assigned."""

    sign: int
    apex: int
    representative: ScalePeak
    scale_peaks: tuple[ScalePeak, ...]


@dataclass(frozen=True)
class SupportAssignment:
    """A contiguous sign-constrained territory assigned to one apex."""

    start: int
    end: int
    sign_run_start: int
    sign_run_end: int


class MarkerScanner:
    """Buffered marker scanner for compact UTF-8 JSON produced by the extractor."""

    def __init__(self, path: Path, chunk_size: int = 4 * 1024 * 1024):
        self.path = path
        self.chunk_size = chunk_size
        self.handle = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
        self.buffer = b""
        self.eof = False

    def close(self) -> None:
        self.handle.close()

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def take_until(self, marker: bytes, capture: bool = False) -> bytes | None:
        pieces = []
        keep = max(0, len(marker) - 1)
        while True:
            index = self.buffer.find(marker)
            if index >= 0:
                if capture:
                    pieces.append(self.buffer[:index])
                self.buffer = self.buffer[index + len(marker):]
                return b"".join(pieces) if capture else b""
            if self.eof:
                if capture:
                    pieces.append(self.buffer)
                self.buffer = b""
                return None
            if len(self.buffer) > keep:
                cut = len(self.buffer) - keep
                if capture:
                    pieces.append(self.buffer[:cut])
                self.buffer = self.buffer[cut:]
            self._fill()

    def _ensure(self, n: int = 1) -> bool:
        while len(self.buffer) < n and self._fill():
            pass
        return len(self.buffer) >= n

    def _skip_space(self) -> None:
        while True:
            match = re.match(rb"\s+", self.buffer)
            if match:
                self.buffer = self.buffer[match.end():]
                continue
            if self.buffer or not self._fill():
                return

    def read_json_string(self) -> str:
        self._skip_space()
        if not self._ensure() or self.buffer[:1] != b'"':
            raise ValueError(f"{self.path}: expected JSON string")
        pieces = [b'"']
        self.buffer = self.buffer[1:]
        escaped = False
        while True:
            if not self._ensure():
                raise ValueError(f"{self.path}: unterminated JSON string")
            byte = self.buffer[:1]
            self.buffer = self.buffer[1:]
            pieces.append(byte)
            if escaped:
                escaped = False
            elif byte == b"\\":
                escaped = True
            elif byte == b'"':
                return json.loads(b"".join(pieces).decode("utf-8"))

    def read_integer(self) -> int:
        self._skip_space()
        while True:
            match = re.match(rb"[-+]?\d+", self.buffer)
            if match:
                value = int(match.group())
                self.buffer = self.buffer[match.end():]
                return value
            if not self._fill():
                raise ValueError(f"{self.path}: expected integer")

    def read_flat_float_array(self) -> np.ndarray:
        if self.take_until(b"[") is None:
            raise ValueError(f"{self.path}: expected numeric array")
        payload = self.take_until(b"]", capture=True)
        if payload is None:
            raise ValueError(f"{self.path}: unterminated numeric array")
        if not payload.strip():
            return np.empty(0, dtype=float)
        return np.fromstring(payload.decode("ascii"), sep=",", dtype=float)


def header_scalar(prefix: bytes, key: str):
    marker = json.dumps(key, separators=(",", ":")).encode() + b":"
    index = prefix.find(marker)
    if index < 0:
        return None
    text = prefix[index + len(marker):].decode("utf-8")
    return json.JSONDecoder().raw_decode(text)[0]


def iter_profiles(
    path: Path,
    influence_field: str = DEFAULT_INFLUENCE_FIELD,
) -> tuple[dict, Iterator[dict]]:
    """Stream one supported source schema into the detector's profile interface.

    Signed-contribution files and compact uniform-control files intentionally
    retain distinct schemas.  This adapter validates each schema explicitly and
    exposes the selected influence field without treating a compact control as
    a signed-contribution file.
    """
    scanner = MarkerScanner(path)
    prefix = scanner.take_until(b'"proteins":[', capture=True)
    if prefix is None:
        scanner.close()
        raise ValueError(f"{path}: missing proteins array")
    metadata = {
        key: header_scalar(prefix, key)
        for key in (
            "schema_version", "condition", "seed", "split", "protein_count",
            "residue_count", "source_seeds",
        )
    }
    schema = metadata["schema_version"]
    allowed_fields = {
        SIGNED_CONTRIBUTION_SCHEMA: {
            DEFAULT_INFLUENCE_FIELD,
            AVERAGED_INFLUENCE_FIELD,
        },
        CONTROL_SCHEMA: {
            CONTROL_OBSERVED_FIELD,
            CONTROL_UNIFORM_FIELD,
        },
        CONTROL_SEED_AVERAGE_SCHEMA: {
            CONTROL_AVERAGED_OBSERVED_FIELD,
            CONTROL_AVERAGED_UNIFORM_FIELD,
        },
    }
    if schema not in allowed_fields:
        scanner.close()
        raise ValueError(
            f"{path}: unsupported schema_version={schema!r}"
        )
    if influence_field not in allowed_fields[schema]:
        scanner.close()
        raise ValueError(
            f"{path}: influence field {influence_field!r} is not available for "
            f"schema {schema!r}; choose one of "
            f"{sorted(allowed_fields[schema])}"
        )

    def generator() -> Iterator[dict]:
        count = 0
        residue_count = 0
        try:
            while scanner.take_until(b'"name":') is not None:
                name = scanner.read_json_string()
                if scanner.take_until(b'"sequence":') is None:
                    raise ValueError(f"{path}: missing sequence for {name}")
                sequence = scanner.read_json_string()
                if scanner.take_until(b'"length":') is None:
                    raise ValueError(f"{path}: missing length for {name}")
                length = scanner.read_integer()
                if schema == SIGNED_CONTRIBUTION_SCHEMA:
                    profile_fields = [
                        "intrinsic_signed_evidence",
                        DEFAULT_INFLUENCE_FIELD,
                    ]
                    if influence_field != DEFAULT_INFLUENCE_FIELD:
                        profile_fields.append(influence_field)
                    profile_fields.append("attention_column_mean")
                    evidence_field = "intrinsic_signed_evidence"
                    attention_field = "attention_column_mean"
                elif schema == CONTROL_SCHEMA:
                    profile_fields = [
                        "intrinsic_signed_evidence",
                        CONTROL_OBSERVED_FIELD,
                        CONTROL_UNIFORM_FIELD,
                        "attention_column_mean",
                        "attention_amplification",
                    ]
                    evidence_field = "intrinsic_signed_evidence"
                    attention_field = "attention_column_mean"
                else:
                    profile_fields = [
                        "seed_averaged_intrinsic_signed_evidence",
                        CONTROL_AVERAGED_OBSERVED_FIELD,
                        CONTROL_AVERAGED_UNIFORM_FIELD,
                        "seed_averaged_attention_column_mean",
                        "seed_averaged_attention_amplification",
                    ]
                    evidence_field = "seed_averaged_intrinsic_signed_evidence"
                    attention_field = "seed_averaged_attention_column_mean"
                profiles = {}
                for field in profile_fields:
                    marker = json.dumps(field).encode() + b":"
                    if scanner.take_until(marker) is None:
                        raise ValueError(f"{path}: missing {field} for {name}")
                    profiles[field] = scanner.read_flat_float_array()
                if length != len(sequence):
                    raise ValueError(
                        f"{path}: {name} length={length}, sequence length={len(sequence)}"
                    )
                for field, values in profiles.items():
                    if values.shape != (length,):
                        raise ValueError(
                            f"{path}: {name} {field} shape={values.shape}, expected {(length,)}"
                        )
                    if not np.all(np.isfinite(values)):
                        raise ValueError(
                            f"{path}: {name} {field} contains nonfinite values"
                        )
                evidence = profiles[evidence_field]
                attention = profiles[attention_field]
                if schema != SIGNED_CONTRIBUTION_SCHEMA and np.any(attention < 0):
                    raise ValueError(
                        f"{path}: {name} compact-control attention is negative"
                    )
                errors = []
                if schema == SIGNED_CONTRIBUTION_SCHEMA:
                    error = np.max(np.abs(
                        profiles[DEFAULT_INFLUENCE_FIELD]
                        - evidence * attention
                    )) if length else 0.0
                    errors.append(float(error))
                    if error > 5e-6:
                        raise ValueError(
                            f"{path}: {name} violates I_j=s_j*B_j; "
                            f"max error={error}"
                        )
                else:
                    if length <= 0:
                        raise ValueError(f"{path}: {name} has nonpositive length")
                    uniform_field = (
                        CONTROL_UNIFORM_FIELD
                        if schema == CONTROL_SCHEMA
                        else CONTROL_AVERAGED_UNIFORM_FIELD
                    )
                    amplification_field = (
                        "attention_amplification"
                        if schema == CONTROL_SCHEMA
                        else "seed_averaged_attention_amplification"
                    )
                    uniform_error = float(np.max(np.abs(
                        profiles[uniform_field] - evidence / length
                    )))
                    amplification_error = float(np.max(np.abs(
                        profiles[amplification_field] - length * attention
                    )))
                    errors.extend([uniform_error, amplification_error])
                    if uniform_error > 5e-6 or amplification_error > 5e-6:
                        raise ValueError(
                            f"{path}: {name} violates compact-control identities; "
                            f"uniform error={uniform_error}, "
                            f"amplification error={amplification_error}"
                        )
                    if not np.isclose(
                        np.sum(attention), 1.0, atol=5e-6, rtol=1e-6
                    ):
                        raise ValueError(
                            f"{path}: {name} compact-control attention does not "
                            "sum to one"
                        )
                    if schema == CONTROL_SCHEMA:
                        observed_error = float(np.max(np.abs(
                            profiles[CONTROL_OBSERVED_FIELD]
                            - evidence * attention
                        )))
                        decomposition_error = float(np.max(np.abs(
                            profiles[CONTROL_OBSERVED_FIELD]
                            - profiles[CONTROL_UNIFORM_FIELD]
                            * profiles[amplification_field]
                        )))
                        errors.extend([observed_error, decomposition_error])
                        if observed_error > 5e-6 or decomposition_error > 5e-6:
                            raise ValueError(
                                f"{path}: {name} violates per-seed control "
                                "decomposition"
                            )
                count += 1
                residue_count += length
                yield {
                    "name": name,
                    "sequence": sequence,
                    "length": length,
                    "source_schema_version": schema,
                    "profile_identity_max_abs_error": max(errors, default=0.0),
                    "intrinsic_signed_evidence": evidence,
                    "attention_column_mean": attention,
                    **profiles,
                }
            expected = metadata.get("protein_count")
            if expected is not None and count != int(expected):
                raise ValueError(f"{path}: streamed {count} proteins, expected {expected}")
            expected_residues = metadata.get("residue_count")
            if expected_residues is not None and residue_count != int(expected_residues):
                raise ValueError(
                    f"{path}: streamed {residue_count} residues, "
                    f"expected {expected_residues}"
                )
        finally:
            scanner.close()

    return metadata, generator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_json",
        action="append",
        help=(
            "One or more signed-contribution or compact uniform-control "
            ".json[.gz] files."
        ),
    )
    source.add_argument(
        "--manifest_tsv",
        help=(
            "TSV containing condition, seed, split, and either json_gz "
            "(signed contributions) or profile_json_gz (compact controls)."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--influence_field",
        choices=INFLUENCE_FIELDS,
        default=DEFAULT_INFLUENCE_FIELD,
        help=(
            "Residue-level signed influence profile used to call bands. The "
            "field must be valid for the selected input schema."
        ),
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help=(
            "Optional exact protein names to process. Intended for targeted "
            "audits; every requested protein must occur in every selected input file."
        ),
    )
    parser.add_argument("--smooth_windows", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--min_scales", type=int, default=2)
    parser.add_argument("--scale_tolerance", type=int, default=2)
    parser.add_argument("--prominence_mad", type=float, default=2.5)
    parser.add_argument("--min_peak_distance", type=int, default=3)
    parser.add_argument(
        "--apex_method",
        choices=["multiscale_prominence", "raw_mad_amplitude"],
        default="raw_mad_amplitude",
        help=(
            "multiscale_prominence preserves the original apex detector. "
            "raw_mad_amplitude selects signed raw local extrema using "
            "|I_p| / robust_scale >= --amplitude_mad; prominence is metadata only."
        ),
    )
    parser.add_argument(
        "--amplitude_mad",
        type=float,
        default=2.0,
        help="Raw-apex amplitude cutoff used by --apex_method raw_mad_amplitude.",
    )
    parser.add_argument(
        "--support_method",
        choices=["half_prominence", "sign_watershed", "half_intensity_merge"],
        default="half_intensity_merge",
        help=(
            "half_prominence preserves the legacy signed-band outputs; "
            "sign_watershed writes apex-first outputs with nonoverlapping, "
            "same-sign descriptive supports; half_intensity_merge creates "
            "contiguous >= half-apex-intensity intervals and merges overlapping "
            "same-sign intervals."
        ),
    )
    parser.add_argument("--terminal_exclusion", type=int, default=0)
    parser.add_argument("--terminal_exclusion_fraction", type=float, default=0.0)
    parser.add_argument("--max_proteins_per_file", type=int, default=0,
                        help="Debugging only; zero means all proteins.")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[InputSpec]:
    if args.manifest_tsv:
        manifest_path = Path(args.manifest_tsv).expanduser().resolve()
        frame = pd.read_csv(manifest_path, sep="\t")
        required = {"condition", "seed", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{manifest_path} lacks columns: {sorted(missing)}")
        path_columns = [
            column
            for column in ("json_gz", "profile_json_gz")
            if column in frame.columns
        ]
        if len(path_columns) != 1:
            raise ValueError(
                f"{manifest_path} must contain exactly one of json_gz or "
                "profile_json_gz"
            )
        path_column = path_columns[0]
        if args.conditions:
            frame = frame[frame["condition"].isin(args.conditions)]
        if args.splits:
            frame = frame[frame["split"].isin(args.splits)]
        specs = [
            InputSpec(
                path=Path(getattr(row, path_column)).expanduser().resolve(),
                condition=str(row.condition),
                seed=(
                    int(row.seed)
                    if str(row.seed).strip().lstrip("+-").isdigit()
                    else str(row.seed)
                ),
                split=str(row.split),
            )
            for row in frame.itertuples(index=False)
        ]
    else:
        specs = [InputSpec(Path(value).expanduser().resolve()) for value in args.input_json]
    if not specs:
        raise ValueError("No input JSON files selected")
    missing_paths = [str(spec.path) for spec in specs if not spec.path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing input JSON files: " + ", ".join(missing_paths[:10]))
    return specs


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(values))
    return float(scale) if np.isfinite(scale) else 0.0


def smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def call_scale_peaks(
    influence: np.ndarray,
    window: int,
    eligible_start: int,
    eligible_end: int,
    prominence_mad: float,
    min_peak_distance: int,
) -> list[ScalePeak]:
    smoothed = smooth_profile(influence, window)
    work = smoothed[eligible_start:eligible_end]
    if len(work) < 3:
        return []
    scale = robust_scale(work)
    if scale <= 0:
        return []
    threshold = prominence_mad * scale
    candidates = []
    for sign in (1, -1):
        directed = sign * work
        peaks, properties = find_peaks(
            directed,
            prominence=threshold,
            distance=max(1, int(min_peak_distance)),
        )
        if len(peaks) == 0:
            continue
        _width, _height, left_ips, right_ips = peak_widths(
            directed, peaks, rel_height=0.5,
            prominence_data=(
                properties["prominences"],
                properties["left_bases"],
                properties["right_bases"],
            ),
        )
        for index, local_apex in enumerate(peaks):
            apex = int(local_apex + eligible_start)
            if sign * influence[apex] <= 0:
                continue
            prominence = float(properties["prominences"][index])
            candidates.append(ScalePeak(
                sign=sign,
                scale=window,
                apex=apex,
                signed_smoothed_score=float(smoothed[apex]),
                prominence=prominence,
                robust_prominence=prominence / scale,
                left_half_prominence=max(
                    eligible_start, int(math.floor(left_ips[index])) + eligible_start
                ),
                right_half_prominence=min(
                    eligible_end - 1, int(math.ceil(right_ips[index])) + eligible_start
                ),
            ))
    return candidates


def detect_multiscale_apices(
    influence: np.ndarray,
    smooth_windows: list[int],
    eligible_start: int,
    eligible_end: int,
    prominence_mad: float,
    min_peak_distance: int,
) -> list[ScalePeak]:
    """Detect scale-specific extrema without assigning or consulting support."""
    peaks: list[ScalePeak] = []
    if eligible_end - eligible_start < 3:
        return peaks
    for window in smooth_windows:
        peaks.extend(call_scale_peaks(
            influence,
            window,
            eligible_start,
            eligible_end,
            prominence_mad,
            min_peak_distance,
        ))
    return peaks


def detect_raw_mad_amplitude_apices(
    influence: np.ndarray,
    search_start: int,
    search_end: int,
    eligible_start: int,
    eligible_end: int,
    amplitude_mad: float,
) -> list[ApexCall]:
    """Select signed raw local extrema by amplitude; retain prominence as metadata."""
    work = np.asarray(influence[search_start:search_end], dtype=float)
    scale = robust_scale(influence[eligible_start:eligible_end])
    if len(work) < 3 or scale <= 0:
        return []
    calls: list[ApexCall] = []
    for sign in (1, -1):
        directed = sign * work
        local_peaks, _properties = find_peaks(directed)
        if len(local_peaks) == 0:
            continue
        prominences = peak_prominences(directed, local_peaks)[0]
        for local_apex, prominence in zip(local_peaks, prominences):
            apex = int(local_apex + search_start)
            if sign * influence[apex] <= 0:
                continue
            if abs(float(influence[apex])) / scale < amplitude_mad:
                continue
            peak = ScalePeak(
                sign=sign,
                scale=1,
                apex=apex,
                signed_smoothed_score=float(influence[apex]),
                prominence=float(prominence),
                robust_prominence=float(prominence / scale),
                left_half_prominence=apex,
                right_half_prominence=apex,
            )
            calls.append(ApexCall(
                sign=sign,
                apex=apex,
                representative=peak,
                scale_peaks=(peak,),
            ))
    calls.sort(key=lambda call: (call.apex, -call.sign))
    return calls


def cluster_multiscale_peaks(
    peaks: list[ScalePeak], min_scales: int, tolerance: int
) -> list[list[ScalePeak]]:
    clusters: list[list[ScalePeak]] = []
    for sign in (1, -1):
        sign_peaks = sorted((peak for peak in peaks if peak.sign == sign), key=lambda x: x.apex)
        for peak in sign_peaks:
            choices = []
            for cluster_index, cluster in enumerate(clusters):
                if cluster[0].sign != sign:
                    continue
                center = float(np.median([item.apex for item in cluster]))
                distance = abs(peak.apex - center)
                if distance <= tolerance:
                    choices.append((distance, cluster_index))
            if choices:
                clusters[min(choices)[1]].append(peak)
            else:
                clusters.append([peak])

    persistent = []
    for cluster in clusters:
        by_scale = {}
        for peak in cluster:
            previous = by_scale.get(peak.scale)
            if previous is None or peak.robust_prominence > previous.robust_prominence:
                by_scale[peak.scale] = peak
        retained = list(by_scale.values())
        if len(retained) >= min_scales:
            persistent.append(retained)
    return persistent


def consolidate_multiscale_apices(
    influence: np.ndarray,
    peaks: list[ScalePeak],
    min_scales: int,
    tolerance: int,
    eligible_start: int,
    eligible_end: int,
) -> list[ApexCall]:
    """Consolidate persistent scale calls into apex positions.

    This is the legacy apex-position logic factored out of boundary assignment:
    no interval width participates in detection or consolidation.
    """
    calls: list[ApexCall] = []
    for cluster in cluster_multiscale_peaks(peaks, min_scales, tolerance):
        sign = cluster[0].sign
        candidate_min = max(
            eligible_start, min(item.apex for item in cluster) - tolerance
        )
        candidate_max = min(
            eligible_end - 1, max(item.apex for item in cluster) + tolerance
        )
        region = np.arange(candidate_min, candidate_max + 1)
        valid = region[sign * influence[region] > 0]
        if len(valid) == 0:
            continue
        apex = int(valid[np.argmax(sign * influence[valid])])
        calls.append(ApexCall(
            sign=sign,
            apex=apex,
            representative=max(cluster, key=lambda item: item.robust_prominence),
            scale_peaks=tuple(cluster),
        ))
    calls.sort(key=lambda call: (call.apex, -call.sign))
    return calls


def _sign_runs(
    influence: np.ndarray, eligible_start: int, eligible_end: int
) -> list[tuple[int, int, int]]:
    """Return maximal nonzero finite sign runs as (start, end, sign)."""
    runs: list[tuple[int, int, int]] = []
    index = eligible_start
    while index < eligible_end:
        value = influence[index]
        sign = 1 if np.isfinite(value) and value > 0 else -1 if np.isfinite(value) and value < 0 else 0
        if sign == 0:
            index += 1
            continue
        start = index
        index += 1
        while index < eligible_end:
            value = influence[index]
            current = 1 if np.isfinite(value) and value > 0 else -1 if np.isfinite(value) and value < 0 else 0
            if current != sign:
                break
            index += 1
        runs.append((start, index - 1, sign))
    return runs


def _valley_plateau(
    influence: np.ndarray, left_apex: int, right_apex: int
) -> tuple[int, int] | None:
    """Choose one deterministic minimum-|I| plateau between adjacent apices.

    If equally low disjoint plateaus exist, choose the plateau whose center is
    closest to the inter-apex midpoint, then the leftmost.  The chosen plateau
    is intentionally left unassigned.
    """
    start = left_apex + 1
    end = right_apex
    if start >= end:
        return None
    values = np.abs(influence[start:end])
    minimum = float(np.min(values))
    tied = np.flatnonzero(np.isclose(values, minimum, rtol=1e-12, atol=0.0))
    plateaus: list[tuple[int, int]] = []
    plateau_start = int(tied[0])
    previous = plateau_start
    for offset in tied[1:]:
        offset = int(offset)
        if offset != previous + 1:
            plateaus.append((start + plateau_start, start + previous))
            plateau_start = offset
        previous = offset
    plateaus.append((start + plateau_start, start + previous))
    midpoint = (left_apex + right_apex) / 2.0
    return min(
        plateaus,
        key=lambda bounds: (
            abs(((bounds[0] + bounds[1]) / 2.0) - midpoint),
            bounds[0],
        ),
    )


def assign_sign_constrained_support(
    influence: np.ndarray,
    apices: list[ApexCall],
    eligible_start: int,
    eligible_end: int,
) -> dict[int, SupportAssignment]:
    """Partition sign runs around all detected apices without overlap.

    Runs without an apex remain unassigned.  Tied valley plateaus between
    same-sign apices remain unassigned, which makes every returned support a
    single contiguous interval and prevents arbitrary assignment of ties.
    """
    assignments: dict[int, SupportAssignment] = {}
    for run_start, run_end, sign in _sign_runs(
        influence, eligible_start, eligible_end
    ):
        members = [
            (index, call)
            for index, call in enumerate(apices)
            if call.sign == sign and run_start <= call.apex <= run_end
        ]
        members.sort(key=lambda item: item[1].apex)
        if not members:
            continue
        positions = [call.apex for _, call in members]
        if len(set(positions)) != len(positions):
            raise ValueError(
                "Duplicate same-sign apex positions cannot receive "
                "nonoverlapping support"
            )
        starts = [run_start] * len(members)
        ends = [run_end] * len(members)
        for member_index in range(len(members) - 1):
            left = members[member_index][1].apex
            right = members[member_index + 1][1].apex
            plateau = _valley_plateau(influence, left, right)
            if plateau is None:
                ends[member_index] = left
                starts[member_index + 1] = right
            else:
                valley_start, valley_end = plateau
                ends[member_index] = valley_start - 1
                starts[member_index + 1] = valley_end + 1
        for member_index, (call_index, call) in enumerate(members):
            if not starts[member_index] <= call.apex <= ends[member_index]:
                raise ValueError("Valley partition excluded its own apex")
            assignments[call_index] = SupportAssignment(
                start=starts[member_index],
                end=ends[member_index],
                sign_run_start=run_start,
                sign_run_end=run_end,
            )
    if len(assignments) != len(apices):
        missing = sorted(set(range(len(apices))) - set(assignments))
        raise ValueError(f"Could not assign sign support to apex indices {missing}")
    return assignments


def assign_merged_half_intensity_support(
    influence: np.ndarray,
    apices: list[ApexCall],
    eligible_start: int,
    eligible_end: int,
) -> tuple[list[ApexCall], dict[int, SupportAssignment], dict[int, list[int]]]:
    """Build half-apex-intensity intervals and merge same-sign overlaps."""
    initial: list[tuple[ApexCall, int, int]] = []
    for call in apices:
        cutoff = 0.5 * abs(float(influence[call.apex]))
        start = end = call.apex
        while (
            start > eligible_start
            and call.sign * influence[start - 1] >= cutoff
        ):
            start -= 1
        while (
            end + 1 < eligible_end
            and call.sign * influence[end + 1] >= cutoff
        ):
            end += 1
        initial.append((call, start, end))

    merged_groups: list[list[tuple[ApexCall, int, int]]] = []
    for sign in (1, -1):
        sign_intervals = sorted(
            (item for item in initial if item[0].sign == sign),
            key=lambda item: (item[1], item[2], item[0].apex),
        )
        for item in sign_intervals:
            if (
                merged_groups
                and merged_groups[-1][0][0].sign == sign
                and item[1] <= max(member[2] for member in merged_groups[-1])
            ):
                merged_groups[-1].append(item)
            else:
                merged_groups.append([item])

    merged_calls: list[ApexCall] = []
    temporary: list[tuple[int, int, list[int]]] = []
    for group in merged_groups:
        sign = group[0][0].sign
        representative = max(
            (item[0] for item in group),
            key=lambda call: sign * influence[call.apex],
        )
        merged_calls.append(representative)
        temporary.append((
            min(item[1] for item in group),
            max(item[2] for item in group),
            sorted(item[0].apex for item in group),
        ))

    order = sorted(
        range(len(merged_calls)),
        key=lambda index: (merged_calls[index].apex, -merged_calls[index].sign),
    )
    ordered_calls = [merged_calls[index] for index in order]
    assignments: dict[int, SupportAssignment] = {}
    merged_members: dict[int, list[int]] = {}
    for new_index, old_index in enumerate(order):
        start, end, members = temporary[old_index]
        assignments[new_index] = SupportAssignment(
            start=start,
            end=end,
            sign_run_start=start,
            sign_run_end=end,
        )
        merged_members[new_index] = members
    return ordered_calls, assignments, merged_members


def audit_apex_support_catalog(
    rows: list[dict],
    influence: np.ndarray,
    eligible_start: int,
    eligible_end: int,
    expected_apices: list[tuple[int, int]] | None = None,
) -> dict:
    """Enforce the Phase 1 apex/support invariants for one protein."""
    assigned = np.zeros(len(influence), dtype=np.int16)
    observed_apices: list[tuple[int, int]] = []
    for row in rows:
        sign = int(row["sign"])
        apex = int(row["apex_index_0based"])
        start = int(row["support_start_index_0based"])
        end = int(row["support_end_index_0based_inclusive"])
        width = int(row["support_width"])
        if not eligible_start <= start <= apex <= end < eligible_end:
            raise ValueError(
                f"Band does not contain its apex or exceeds eligibility: "
                f"{row['band_id']}"
            )
        values = influence[start:end + 1]
        if not np.all(sign * values > 0):
            raise ValueError(
                f"Band crosses a sign/zero boundary: {row['band_id']}"
            )
        if width != end - start + 1:
            raise ValueError(f"Band width mismatch: {row['band_id']}")
        assigned[start:end + 1] += 1
        observed_apices.append((sign, apex))
    if np.any(assigned > 1):
        raise ValueError("Two or more band intervals overlap")
    if expected_apices is not None and sorted(observed_apices) != sorted(expected_apices):
        raise ValueError("Band assignment changed primary-apex number or positions")
    return {
        "n_bands": len(rows),
        "n_assigned_residues": int(np.sum(assigned == 1)),
        "n_multiply_assigned_residues": int(np.sum(assigned > 1)),
        "all_bands_contain_primary_apex": True,
        "all_bands_sign_constrained": True,
        "all_bands_inside_eligible_interval": True,
        "all_band_widths_exact": True,
        "apex_positions_unchanged": True,
    }


def build_band_rows(
    influence: np.ndarray,
    evidence: np.ndarray,
    breadth: np.ndarray,
    sequence: str,
    clusters: list[list[ScalePeak]] | None,
    context: dict,
    apex_calls: list[ApexCall] | None = None,
) -> list[dict]:
    if apex_calls is None:
        apex_calls = consolidate_multiscale_apices(
            influence,
            [peak for cluster in (clusters or []) for peak in cluster],
            min_scales=1,
            tolerance=context["scale_tolerance"],
            eligible_start=context["eligible_start"],
            eligible_end=context["eligible_end"],
        )
    rows = []
    for call in apex_calls:
        cluster = list(call.scale_peaks)
        sign = call.sign
        representative = call.representative
        apex = call.apex
        start = int(round(np.median([item.left_half_prominence for item in cluster])))
        end = int(round(np.median([item.right_half_prominence for item in cluster])))
        start = max(context["eligible_start"], min(start, apex))
        end = min(context["eligible_end"] - 1, max(end, apex))
        values = influence[start:end + 1]
        terminal_flag_width = max(10, int(math.ceil(0.02 * len(sequence))))
        label = "flexibility_supporting" if sign > 0 else "rigidity_supporting"
        rows.append({
            **{key: context[key] for key in (
                "condition", "seed", "split", "json_path", "influence_field",
                "protein", "protein_length"
            )},
            "sign": sign,
            "label": label,
            "apex_index_0based": apex,
            "apex_residue_1based": apex + 1,
            "apex_amino_acid": sequence[apex],
            "start_index_0based": start,
            "end_index_0based_inclusive": end,
            "start_residue_1based": start + 1,
            "end_residue_1based_inclusive": end + 1,
            "band_width": end - start + 1,
            "band_sequence": sequence[start:end + 1],
            "apex_signed_column_influence": float(influence[apex]),
            "apex_intrinsic_signed_evidence": float(evidence[apex]),
            "apex_attention_column_mean": float(breadth[apex]),
            "band_signed_influence_sum": float(np.sum(values)),
            "band_signed_influence_mean": float(np.mean(values)),
            "band_absolute_influence_sum": float(np.sum(np.abs(values))),
            "representative_smoothing_window": representative.scale,
            "representative_prominence": representative.prominence,
            "representative_robust_prominence": representative.robust_prominence,
            "n_persistent_scales": len(cluster),
            "persistent_scales": ",".join(str(value) for value in sorted(item.scale for item in cluster)),
            "scale_apex_indices_0based": ",".join(str(item.apex) for item in sorted(cluster, key=lambda x: x.scale)),
            "distance_to_nearest_terminus": min(apex, len(sequence) - 1 - apex),
            "near_terminal_10_or_2pct": bool(
                apex < terminal_flag_width or apex >= len(sequence) - terminal_flag_width
            ),
            "eligible_start_index_0based": context["eligible_start"],
            "eligible_end_index_0based_exclusive": context["eligible_end"],
        })
    rows.sort(key=lambda row: (row["apex_index_0based"], -row["sign"]))
    for number, row in enumerate(rows, start=1):
        row["band_number_within_protein"] = number
        safe_protein = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["protein"])
        row["band_id"] = (
            f"{row['condition']}__seed{row['seed']}__{row['split']}__{safe_protein}__"
            f"{row['label']}__{number:03d}"
        )
    return rows


def build_apex_rows(
    influence: np.ndarray,
    evidence: np.ndarray,
    breadth: np.ndarray,
    sequence: str,
    apex_calls: list[ApexCall],
    context: dict,
    support_method: str = "sign_watershed",
) -> tuple[list[dict], dict]:
    """Materialize bands, each represented by one primary signed apex."""
    if support_method == "sign_watershed":
        working_calls = apex_calls
        assignments = assign_sign_constrained_support(
            influence,
            working_calls,
            context["eligible_start"],
            context["eligible_end"],
        )
        merged_members = {
            index: [call.apex] for index, call in enumerate(working_calls)
        }
        detector_version = DETECTOR_VERSION
        apex_detection_method = "multiscale_prominence"
    elif support_method == "half_intensity_merge":
        working_calls, assignments, merged_members = (
            assign_merged_half_intensity_support(
                influence,
                apex_calls,
                context["eligible_start"],
                context["eligible_end"],
            )
        )
        detector_version = RAW_AMPLITUDE_DETECTOR_VERSION
        apex_detection_method = "raw_mad_amplitude"
    else:
        raise ValueError(f"Unsupported apex support method: {support_method}")

    rows: list[dict] = []
    terminal_flag_width = max(10, int(math.ceil(0.02 * len(sequence))))
    amplitude_scale = robust_scale(
        influence[context["eligible_start"]:context["eligible_end"]]
    )
    eligible_values = influence[
        context["eligible_start"]:context["eligible_end"]
    ]
    eligible_total_absolute_influence = float(
        np.sum(np.abs(eligible_values[np.isfinite(eligible_values)]))
    )
    for number, call in enumerate(working_calls, start=1):
        assignment = assignments[number - 1]
        start, end = assignment.start, assignment.end
        values = influence[start:end + 1]
        label = "flexibility_supporting" if call.sign > 0 else "rigidity_supporting"
        safe_protein = re.sub(r"[^A-Za-z0-9_.-]+", "_", context["protein"])
        band_id = (
            f"{context['condition']}__seed{context['seed']}__{context['split']}__"
            f"{safe_protein}__{label}__band_{number:03d}"
        )
        cluster = list(call.scale_peaks)
        row = {
            **{key: context[key] for key in (
                "condition", "seed", "split", "json_path", "influence_field",
                "protein", "protein_length"
            )},
            "sign": call.sign,
            "label": label,
            "object_type": (
                "merged_half_intensity_band_with_primary_apex"
                if support_method == "half_intensity_merge"
                else "signed_support_band_with_primary_apex"
            ),
            "band_id": band_id,
            "band_number_within_protein": number,
            "apex_id": band_id,
            "apex_number_within_protein": number,
            "apex_index_0based": call.apex,
            "apex_residue_1based": call.apex + 1,
            "apex_amino_acid": sequence[call.apex],
            "apex_signed_column_influence": float(influence[call.apex]),
            "apex_intrinsic_signed_evidence": float(evidence[call.apex]),
            "apex_attention_column_mean": float(breadth[call.apex]),
            "representative_smoothing_window": call.representative.scale,
            "representative_prominence": call.representative.prominence,
            "representative_robust_prominence": call.representative.robust_prominence,
            "n_persistent_scales": len(cluster),
            "persistent_scales": ",".join(
                str(value) for value in sorted(item.scale for item in cluster)
            ),
            "scale_apex_indices_0based": ",".join(
                str(item.apex) for item in sorted(cluster, key=lambda item: item.scale)
            ),
            "support_start_index_0based": start,
            "support_end_index_0based_inclusive": end,
            "support_start_residue_1based": start + 1,
            "support_end_residue_1based_inclusive": end + 1,
            "support_width": end - start + 1,
            "support_sequence": sequence[start:end + 1],
            "support_signed_influence_sum": float(np.sum(values)),
            "support_signed_influence_mean": float(np.mean(values)),
            "support_absolute_influence_sum": float(np.sum(np.abs(values))),
            "start_index_0based": start,
            "end_index_0based_inclusive": end,
            "start_residue_1based": start + 1,
            "end_residue_1based_inclusive": end + 1,
            "band_width": end - start + 1,
            "band_sequence": sequence[start:end + 1],
            "band_signed_influence_sum": float(np.sum(values)),
            "band_signed_influence_mean": float(np.mean(values)),
            "band_absolute_influence_sum": float(np.sum(np.abs(values))),
            "sign_run_start_index_0based": assignment.sign_run_start,
            "sign_run_end_index_0based_inclusive": assignment.sign_run_end,
            "distance_to_nearest_terminus": min(
                call.apex, len(sequence) - 1 - call.apex
            ),
            "near_terminal_10_or_2pct": bool(
                call.apex < terminal_flag_width
                or call.apex >= len(sequence) - terminal_flag_width
            ),
            "eligible_start_index_0based": context["eligible_start"],
            "eligible_end_index_0based_exclusive": context["eligible_end"],
            "support_method": support_method,
            "detector_version": detector_version,
            "apex_detection_method": apex_detection_method,
            "apex_absolute_influence": abs(float(influence[call.apex])),
            "apex_standardized_magnitude_R_p": (
                abs(float(influence[call.apex])) / amplitude_scale
                if amplitude_scale > 0 else np.nan
            ),
            "band_integrated_magnitude": float(np.sum(np.abs(values))),
            "band_fraction_of_eligible_absolute_influence": (
                float(np.sum(np.abs(values))) / eligible_total_absolute_influence
                if eligible_total_absolute_influence > 0 else np.nan
            ),
            "apex_abs_over_robust_scale": (
                abs(float(influence[call.apex])) / amplitude_scale
                if amplitude_scale > 0 else np.nan
            ),
            "n_merged_candidate_apices": len(merged_members[number - 1]),
            "merged_candidate_apex_indices_0based": ",".join(
                str(apex) for apex in merged_members[number - 1]
            ),
        }
        rows.append(row)

    for sign in (1, -1):
        sign_rows = [row for row in rows if int(row["sign"]) == sign]
        if not sign_rows:
            continue
        magnitudes = pd.Series([
            row["apex_absolute_influence"] for row in sign_rows
        ])
        ranks = magnitudes.rank(method="min", ascending=False)
        percentiles = (
            100.0
            * magnitudes.rank(method="average", pct=True, ascending=True)
        )
        for row, rank, percentile in zip(sign_rows, ranks, percentiles):
            row["absolute_apex_rank_within_protein_sign"] = int(rank)
            row["absolute_apex_percentile_within_protein_sign"] = float(percentile)
        integrated = pd.Series([
            row["band_integrated_magnitude"] for row in sign_rows
        ])
        integrated_ranks = integrated.rank(method="min", ascending=False)
        integrated_percentiles = (
            100.0
            * integrated.rank(method="average", pct=True, ascending=True)
        )
        for row, rank, percentile in zip(
            sign_rows, integrated_ranks, integrated_percentiles
        ):
            row["integrated_magnitude_rank_within_protein_sign"] = int(rank)
            row["integrated_magnitude_percentile_within_protein_sign"] = float(
                percentile
            )

    all_magnitudes = pd.Series([
        row["apex_absolute_influence"] for row in rows
    ])
    all_ranks = all_magnitudes.rank(method="min", ascending=False)
    for row, rank in zip(rows, all_ranks):
        row["absolute_apex_rank_within_protein"] = int(rank)
    all_integrated = pd.Series([
        row["band_integrated_magnitude"] for row in rows
    ])
    integrated_ranks = all_integrated.rank(method="min", ascending=False)
    for row, rank in zip(rows, integrated_ranks):
        row["integrated_magnitude_rank_within_protein"] = int(rank)

    audit = audit_apex_support_catalog(
        rows,
        influence,
        context["eligible_start"],
        context["eligible_end"],
        expected_apices=[(call.sign, call.apex) for call in working_calls],
    )
    audit["n_candidate_apices_before_interval_merge"] = len(apex_calls)
    audit["n_bands_after_interval_merge"] = len(working_calls)
    return rows, audit


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.smooth_windows)) != len(args.smooth_windows):
        raise ValueError("--smooth_windows must be unique")
    if any(window < 1 or window % 2 == 0 for window in args.smooth_windows):
        raise ValueError("--smooth_windows must contain positive odd integers")
    if not 1 <= args.min_scales <= len(args.smooth_windows):
        raise ValueError("--min_scales must be between 1 and the number of smoothing windows")
    if args.prominence_mad <= 0 or args.scale_tolerance < 0:
        raise ValueError("Prominence must be positive and tolerance nonnegative")
    if args.amplitude_mad <= 0:
        raise ValueError("--amplitude_mad must be positive")
    if args.terminal_exclusion < 0 or not 0 <= args.terminal_exclusion_fraction < 0.5:
        raise ValueError("Invalid terminal exclusion")
    expected_support = {
        "multiscale_prominence": {"half_prominence", "sign_watershed"},
        "raw_mad_amplitude": {"half_intensity_merge"},
    }
    if args.support_method not in expected_support[args.apex_method]:
        raise ValueError(
            f"--apex_method {args.apex_method} cannot be combined with "
            f"--support_method {args.support_method}"
        )


def active_detector_parameters(args: argparse.Namespace) -> dict:
    """Return only parameters that affect the selected detector and interval."""
    parameters = {
        "apex_method": args.apex_method,
        "support_method": args.support_method,
        "terminal_exclusion": args.terminal_exclusion,
        "terminal_exclusion_fraction": args.terminal_exclusion_fraction,
    }
    if args.apex_method == "raw_mad_amplitude":
        parameters["amplitude_mad"] = args.amplitude_mad
    else:
        parameters.update({
            "smooth_windows": args.smooth_windows,
            "min_scales": args.min_scales,
            "scale_tolerance": args.scale_tolerance,
            "prominence_mad": args.prominence_mad,
            "min_peak_distance": args.min_peak_distance,
        })
    return parameters


def main() -> None:
    args = parse_args()
    validate_args(args)
    specs = load_inputs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    band_rows = []
    summary_rows = []
    support_audits = []
    input_schema_versions: set[str] = set()
    object_name = "bands"
    for file_index, spec in enumerate(specs, start=1):
        metadata, profiles = iter_profiles(spec.path, args.influence_field)
        input_schema_versions.add(str(metadata["schema_version"]))
        condition = spec.condition or str(metadata["condition"])
        seed = spec.seed if spec.seed is not None else metadata["seed"]
        split = spec.split or str(metadata["split"])
        requested_proteins = set(args.proteins or [])
        found_proteins: set[str] = set()
        for protein_index, profile in enumerate(profiles, start=1):
            if requested_proteins and profile["name"] not in requested_proteins:
                continue
            if args.max_proteins_per_file and protein_index > args.max_proteins_per_file:
                break
            found_proteins.add(profile["name"])
            length = profile["length"]
            exclusion = max(
                args.terminal_exclusion,
                int(math.ceil(args.terminal_exclusion_fraction * length)),
            )
            search_start = exclusion
            search_end = length - exclusion
            # scipy.signal.find_peaks cannot select the two endpoints of its
            # search interval. Record that actual callable apex interval so the
            # circular-shift null uses exactly the same positional support.
            eligible_start = min(search_end, search_start + 1)
            eligible_end = max(eligible_start, search_end - 1)
            influence = profile[args.influence_field]
            if args.apex_method == "raw_mad_amplitude":
                apex_calls = detect_raw_mad_amplitude_apices(
                    influence,
                    search_start,
                    search_end,
                    eligible_start,
                    eligible_end,
                    args.amplitude_mad,
                )
            else:
                all_scale_peaks = detect_multiscale_apices(
                    influence,
                    args.smooth_windows,
                    search_start,
                    search_end,
                    args.prominence_mad,
                    args.min_peak_distance,
                )
                apex_calls = consolidate_multiscale_apices(
                    influence,
                    all_scale_peaks,
                    args.min_scales,
                    args.scale_tolerance,
                    eligible_start,
                    eligible_end,
                )
            context = {
                "condition": condition,
                "seed": seed,
                "split": split,
                "json_path": str(spec.path),
                "influence_field": args.influence_field,
                "protein": profile["name"],
                "protein_length": length,
                "eligible_start": eligible_start,
                "eligible_end": eligible_end,
                "scale_tolerance": args.scale_tolerance,
            }
            # An average of products is not generally the product of the
            # averaged factors.  Do not attach seed-specific s_j or B_j values
            # to an ensemble-mean I_j band as though they were its components.
            components_are_valid = (
                args.influence_field == DEFAULT_INFLUENCE_FIELD
                or profile["source_schema_version"]
                in {CONTROL_SCHEMA, CONTROL_SEED_AVERAGE_SCHEMA}
            )
            if components_are_valid:
                evidence = profile["intrinsic_signed_evidence"]
                breadth = profile["attention_column_mean"]
            else:
                evidence = np.full(length, np.nan, dtype=float)
                breadth = np.full(length, np.nan, dtype=float)
            if args.support_method == "half_prominence":
                rows = build_band_rows(
                    influence,
                    evidence,
                    breadth,
                    profile["sequence"],
                    clusters=None,
                    context=context,
                    apex_calls=apex_calls,
                )
            else:
                rows, protein_audit = build_apex_rows(
                    influence,
                    evidence,
                    breadth,
                    profile["sequence"],
                    apex_calls,
                    context,
                    support_method=args.support_method,
                )
                support_audits.append({
                    "condition": condition,
                    "seed": seed,
                    "split": split,
                    "protein": profile["name"],
                    **protein_audit,
                })
            band_rows.extend(rows)
            summary_rows.append({
                "condition": condition,
                "seed": seed,
                "split": split,
                "json_path": str(spec.path),
                "influence_field": args.influence_field,
                "protein": profile["name"],
                "protein_length": length,
                "eligible_start_index_0based": eligible_start,
                "eligible_end_index_0based_exclusive": eligible_end,
                "terminal_exclusion_used": exclusion,
                f"n_flexibility_supporting_{object_name}": sum(
                    row["sign"] > 0 for row in rows
                ),
                f"n_rigidity_supporting_{object_name}": sum(
                    row["sign"] < 0 for row in rows
                ),
                f"n_total_{object_name}": len(rows),
                "profile_identity_max_abs_error": profile["profile_identity_max_abs_error"],
            })
            print(
                f"[{file_index}/{len(specs)} {condition} seed={seed} {split}] "
                f"{protein_index} {profile['name']} L={length} "
                f"{object_name}={len(rows)}",
                flush=True,
            )
            if requested_proteins and found_proteins == requested_proteins:
                break
        if requested_proteins:
            missing_proteins = sorted(requested_proteins - found_proteins)
            if missing_proteins:
                raise ValueError(
                    f"{condition} seed={seed} split={split} is missing requested "
                    f"proteins: {missing_proteins}"
                )

    legacy_mode = args.support_method == "half_prominence"
    bands_path = output_dir / "signed_bands.csv"
    summary_path = output_dir / "signed_band_protein_summary.csv"
    output_columns = BAND_OUTPUT_COLUMNS if legacy_mode else APEX_OUTPUT_COLUMNS
    pd.DataFrame.from_records(band_rows, columns=output_columns).to_csv(
        bands_path, index=False
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    parameters = {
        "score": {
            DEFAULT_INFLUENCE_FIELD: (
                "I_j = (1/L) sum_i C_ij = s_j * (1/L) sum_i A_ij"
            ),
            AVERAGED_INFLUENCE_FIELD: (
                "mean_seed_I_j = mean_seed(s_j * B_j)"
            ),
            CONTROL_OBSERVED_FIELD: "I_j_observed = s_j * B_j",
            CONTROL_UNIFORM_FIELD: "I_j_uniform = s_j / L",
            CONTROL_AVERAGED_OBSERVED_FIELD: (
                "mean_seed_I_j_observed = mean_seed(s_j * B_j)"
            ),
            CONTROL_AVERAGED_UNIFORM_FIELD: (
                "mean_seed_I_j_uniform = mean_seed(s_j / L)"
            ),
        }[args.influence_field],
        "influence_field": args.influence_field,
        "component_profile_note": (
            "Apex intrinsic evidence and attention breadth are blank for the "
            "seed-averaged profile because mean(s_j * B_j) is not generally "
            "mean(s_j) * mean(B_j)."
            if args.influence_field == AVERAGED_INFLUENCE_FIELD
            else (
                "Apex fields contain the three-seed means of intrinsic evidence "
                "and attention breadth. They are descriptive components and are "
                "not factorized to reconstruct mean observed influence."
                if args.influence_field in {
                    CONTROL_AVERAGED_OBSERVED_FIELD,
                    CONTROL_AVERAGED_UNIFORM_FIELD,
                }
                else "Apex intrinsic evidence and attention breadth are from "
                "the same seed."
            )
        ),
        "input_schema_versions": sorted(input_schema_versions),
        "sign_labels": {
            "+1": "flexibility_supporting",
            "-1": "rigidity_supporting",
        },
        "coordinate_convention": {
            "*_index_0based": "Python indexing",
            "*_residue_1based": "biological residue numbering relative to the input sequence",
            "end_index_0based_inclusive": "inclusive band end",
        },
        "input_files": [str(spec.path) for spec in specs],
        "selected_conditions": sorted({
            str(row["condition"]) for row in summary_rows
        }),
        "selected_splits": sorted({
            str(row["split"]) for row in summary_rows
        }),
        "max_proteins_per_file": args.max_proteins_per_file,
        "debug_subset": bool(args.max_proteins_per_file or args.proteins),
        **active_detector_parameters(args),
        "eligible_apex_interval": (
            "[terminal_exclusion + 1, L - terminal_exclusion - 1) in 0-based "
            "half-open coordinates. With default exclusion=0, residues 1 "
            "through L-2 are eligible. find_peaks runs on the enclosing search "
            "interval, so eligibility is applied exactly once."
        ),
        "band_fraction_denominator": (
            "sum of |I_j| over the same eligible interval recorded by "
            "eligible_start/end; terminal-excluded residues are omitted"
        ),
        "band_boundary": {
            "half_prominence": (
                "median half-prominence limits across persistent scales"
            ),
            "sign_watershed": (
                "maximal same-sign runs partitioned at minimum-|I_j| valley plateaus"
            ),
            "half_intensity_merge": (
                "contiguous same-sign residues satisfying "
                "|I_j| >= 0.5 * |I_apex|; overlapping same-sign intervals merged"
            ),
        }[args.support_method],
        "n_protein_profiles": len(summary_rows),
        "n_bands": len(band_rows),
        "outputs": {
            "bands": str(bands_path),
            "protein_summary": str(summary_path),
        },
    }
    if args.proteins:
        parameters["selected_proteins"] = sorted(set(args.proteins))
    if not legacy_mode:
        raw_amplitude_mode = args.apex_method == "raw_mad_amplitude"
        parameters.update({
            "robust_scale_scope": (
                "sigma_MAD = 1.4826 * median(|I_j - median(I)|), computed once "
                "over the complete eligible raw protein profile"
                if raw_amplitude_mode
                else "one MAD-derived scale over the complete eligible protein "
                "profile at each smoothing window; this is not a sliding local scale"
            ),
            "apex_inclusion_rule": (
                "raw signed local maximum/minimum p with the appropriate sign "
                "and |I_p| / sigma_MAD >= amplitude_mad"
                if raw_amplitude_mode
                else "persistent multiscale local extremum passing the "
                "robust-prominence cutoff"
            ),
            "prominence_role": (
                "descriptive metadata only; it does not affect apex inclusion, "
                "interval merging, ranking, or later seed matching"
                if raw_amplitude_mode
                else "part of the multiscale apex inclusion rule"
            ),
            "support_method": args.support_method,
            "support_interpretation": (
                "Half-intensity bands are compact intensity cores around detected "
                "apices, not physical domains or inferred pathways. After "
                "overlapping same-sign intervals merge, one row represents one "
                "band and the strongest-amplitude candidate is its primary apex."
                if args.support_method == "half_intensity_merge"
                else "Sign-constrained supports are descriptive local influence "
                "territories, not physical domains or inferred pathways."
            ),
            "detector_version": (
                RAW_AMPLITUDE_DETECTOR_VERSION
                if raw_amplitude_mode else DETECTOR_VERSION
            ),
        })
        if args.support_method == "sign_watershed":
            parameters["tied_valley_rule"] = (
                "leave the selected minimum-|I_j| plateau unassigned; for "
                "disjoint equal plateaus choose closest to the apex midpoint, "
                "then leftmost"
            )
    parameter_path = output_dir / "signed_band_parameters.json"
    parameter_path.write_text(
        json.dumps(parameters, indent=2) + "\n"
    )
    if not legacy_mode:
        detector_version = (
            RAW_AMPLITUDE_DETECTOR_VERSION
            if args.apex_method == "raw_mad_amplitude"
            else DETECTOR_VERSION
        )
        audit = {
            "detector_version": detector_version,
            "support_method": args.support_method,
            "n_protein_profiles": len(summary_rows),
            "n_bands": len(band_rows),
            "n_assigned_residues": int(sum(
                row["n_assigned_residues"] for row in support_audits
            )),
            "n_multiply_assigned_residues": int(sum(
                row["n_multiply_assigned_residues"] for row in support_audits
            )),
            "opposite_sign_support_overlap_fraction": 0.0,
            "all_support_pair_overlap_fraction": 0.0,
            "all_invariants_pass": True,
            "invariants": [
                "every band interval contains its primary apex",
                "every band interval contains only influence having its apex sign",
                "no band-interval pair overlaps",
                "all band intervals stay inside the eligible interval",
                "band width equals end - start + 1",
                "no residue is assigned to more than one band",
                (
                    "overlapping same-sign half-intensity intervals are merged "
                    "and retain the strongest-amplitude apex"
                    if args.support_method == "half_intensity_merge"
                    else "apex number and positions are unchanged by interval assignment"
                ),
            ],
            "per_protein_audits": support_audits,
        }
        (output_dir / "band_interval_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
    print(json.dumps({
        "protein_profiles": len(summary_rows),
        object_name: len(band_rows),
        f"{object_name}_csv": str(bands_path),
        "protein_summary_csv": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
