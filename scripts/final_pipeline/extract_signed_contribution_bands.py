#!/usr/bin/env python3
"""Call positive and negative multiscale bands from signed column influence I_j.

The publication contribution files are very large because they contain two LxL
matrices.  This script scans their compact JSON representation and materializes
only the three O(L) key profiles needed here:

  intrinsic_signed_evidence s_j
  signed_column_influence   I_j
  attention_column_mean     B_j

The scanner deliberately skips contribution_matrix and attention_matrix without
parsing their numeric elements.  Positive I_j bands are labelled
``flexibility_supporting`` and negative bands ``rigidity_supporting``.
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
from scipy.signal import find_peaks, peak_widths


DEFAULT_INFLUENCE_FIELD = "signed_column_influence"
AVERAGED_INFLUENCE_FIELD = "seed_averaged_signed_column_influence"

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
    scanner = MarkerScanner(path)
    prefix = scanner.take_until(b'"proteins":[', capture=True)
    if prefix is None:
        scanner.close()
        raise ValueError(f"{path}: missing proteins array")
    metadata = {
        key: header_scalar(prefix, key)
        for key in ("schema_version", "condition", "seed", "split", "protein_count")
    }
    if metadata["schema_version"] != "esmfluc.signed_contributions.v1":
        scanner.close()
        raise ValueError(
            f"{path}: unsupported schema_version={metadata['schema_version']!r}"
        )

    def generator() -> Iterator[dict]:
        count = 0
        try:
            while scanner.take_until(b'"name":') is not None:
                name = scanner.read_json_string()
                if scanner.take_until(b'"sequence":') is None:
                    raise ValueError(f"{path}: missing sequence for {name}")
                sequence = scanner.read_json_string()
                if scanner.take_until(b'"length":') is None:
                    raise ValueError(f"{path}: missing length for {name}")
                length = scanner.read_integer()
                profiles = {}
                profile_fields = [
                    "intrinsic_signed_evidence",
                    DEFAULT_INFLUENCE_FIELD,
                ]
                if influence_field != DEFAULT_INFLUENCE_FIELD:
                    profile_fields.append(influence_field)
                profile_fields.append("attention_column_mean")
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
                error = np.max(np.abs(
                    profiles["signed_column_influence"]
                    - profiles["intrinsic_signed_evidence"]
                    * profiles["attention_column_mean"]
                )) if length else 0.0
                if error > 5e-6:
                    raise ValueError(
                        f"{path}: {name} violates I_j=s_j*B_j; max error={error}"
                    )
                count += 1
                yield {
                    "name": name,
                    "sequence": sequence,
                    "length": length,
                    "profile_identity_max_abs_error": float(error),
                    **profiles,
                }
            expected = metadata.get("protein_count")
            if expected is not None and count != int(expected):
                raise ValueError(f"{path}: streamed {count} proteins, expected {expected}")
        finally:
            scanner.close()

    return metadata, generator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_json", action="append", help="One or more signed-contribution .json[.gz] files.")
    source.add_argument(
        "--manifest_tsv",
        help="TSV containing json_gz, condition, seed, and split columns.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--influence_field",
        choices=[DEFAULT_INFLUENCE_FIELD, AVERAGED_INFLUENCE_FIELD],
        default=DEFAULT_INFLUENCE_FIELD,
        help=(
            "Residue-level signed influence profile used to call bands. "
            "Use seed_averaged_signed_column_influence for the three-seed mean."
        ),
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--smooth_windows", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--min_scales", type=int, default=2)
    parser.add_argument("--scale_tolerance", type=int, default=2)
    parser.add_argument("--prominence_mad", type=float, default=2.5)
    parser.add_argument("--min_peak_distance", type=int, default=3)
    parser.add_argument("--terminal_exclusion", type=int, default=0)
    parser.add_argument("--terminal_exclusion_fraction", type=float, default=0.0)
    parser.add_argument("--max_proteins_per_file", type=int, default=0,
                        help="Debugging only; zero means all proteins.")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[InputSpec]:
    if args.manifest_tsv:
        manifest_path = Path(args.manifest_tsv).expanduser().resolve()
        frame = pd.read_csv(manifest_path, sep="\t")
        required = {"json_gz", "condition", "seed", "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{manifest_path} lacks columns: {sorted(missing)}")
        if args.conditions:
            frame = frame[frame["condition"].isin(args.conditions)]
        if args.splits:
            frame = frame[frame["split"].isin(args.splits)]
        specs = [
            InputSpec(
                path=Path(row.json_gz).expanduser().resolve(),
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


def build_band_rows(
    influence: np.ndarray,
    evidence: np.ndarray,
    breadth: np.ndarray,
    sequence: str,
    clusters: list[list[ScalePeak]],
    context: dict,
) -> list[dict]:
    rows = []
    for cluster in clusters:
        sign = cluster[0].sign
        representative = max(cluster, key=lambda item: item.robust_prominence)
        candidate_min = max(context["eligible_start"], min(item.apex for item in cluster) - context["scale_tolerance"])
        candidate_max = min(context["eligible_end"] - 1, max(item.apex for item in cluster) + context["scale_tolerance"])
        region = np.arange(candidate_min, candidate_max + 1)
        valid = region[sign * influence[region] > 0]
        if len(valid) == 0:
            continue
        apex = int(valid[np.argmax(sign * influence[valid])])
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


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.smooth_windows)) != len(args.smooth_windows):
        raise ValueError("--smooth_windows must be unique")
    if any(window < 1 or window % 2 == 0 for window in args.smooth_windows):
        raise ValueError("--smooth_windows must contain positive odd integers")
    if not 1 <= args.min_scales <= len(args.smooth_windows):
        raise ValueError("--min_scales must be between 1 and the number of smoothing windows")
    if args.prominence_mad <= 0 or args.scale_tolerance < 0:
        raise ValueError("Prominence must be positive and tolerance nonnegative")
    if args.terminal_exclusion < 0 or not 0 <= args.terminal_exclusion_fraction < 0.5:
        raise ValueError("Invalid terminal exclusion")


def main() -> None:
    args = parse_args()
    validate_args(args)
    specs = load_inputs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    band_rows = []
    summary_rows = []
    for file_index, spec in enumerate(specs, start=1):
        metadata, profiles = iter_profiles(spec.path, args.influence_field)
        condition = spec.condition or str(metadata["condition"])
        seed = spec.seed if spec.seed is not None else metadata["seed"]
        split = spec.split or str(metadata["split"])
        for protein_index, profile in enumerate(profiles, start=1):
            if args.max_proteins_per_file and protein_index > args.max_proteins_per_file:
                break
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
            all_scale_peaks = []
            if search_end - search_start >= 3:
                for window in args.smooth_windows:
                    all_scale_peaks.extend(call_scale_peaks(
                        influence,
                        window,
                        search_start,
                        search_end,
                        args.prominence_mad,
                        args.min_peak_distance,
                    ))
            clusters = cluster_multiscale_peaks(
                all_scale_peaks, args.min_scales, args.scale_tolerance
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
            if args.influence_field == DEFAULT_INFLUENCE_FIELD:
                evidence = profile["intrinsic_signed_evidence"]
                breadth = profile["attention_column_mean"]
            else:
                evidence = np.full(length, np.nan, dtype=float)
                breadth = np.full(length, np.nan, dtype=float)
            rows = build_band_rows(
                influence,
                evidence,
                breadth,
                profile["sequence"],
                clusters,
                context,
            )
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
                "n_flexibility_supporting_bands": sum(row["sign"] > 0 for row in rows),
                "n_rigidity_supporting_bands": sum(row["sign"] < 0 for row in rows),
                "n_total_bands": len(rows),
                "profile_identity_max_abs_error": profile["profile_identity_max_abs_error"],
            })
            print(
                f"[{file_index}/{len(specs)} {condition} seed={seed} {split}] "
                f"{protein_index} {profile['name']} L={length} bands={len(rows)}",
                flush=True,
            )

    bands_path = output_dir / "signed_bands.csv"
    summary_path = output_dir / "signed_band_protein_summary.csv"
    pd.DataFrame.from_records(band_rows, columns=BAND_OUTPUT_COLUMNS).to_csv(
        bands_path, index=False
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    parameters = {
        "score": (
            "I_j = (1/L) sum_i C_ij = s_j * (1/L) sum_i A_ij"
            if args.influence_field == DEFAULT_INFLUENCE_FIELD
            else "mean_seed_I_j = (I_j_seed1 + I_j_seed2 + I_j_seed3) / 3"
        ),
        "influence_field": args.influence_field,
        "component_profile_note": (
            "Apex intrinsic evidence and attention breadth are blank for the "
            "seed-averaged profile because mean(s_j * B_j) is not generally "
            "mean(s_j) * mean(B_j)."
            if args.influence_field == AVERAGED_INFLUENCE_FIELD
            else "Apex intrinsic evidence and attention breadth are from the same seed."
        ),
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
        "smooth_windows": args.smooth_windows,
        "min_scales": args.min_scales,
        "scale_tolerance": args.scale_tolerance,
        "prominence_mad": args.prominence_mad,
        "min_peak_distance": args.min_peak_distance,
        "terminal_exclusion": args.terminal_exclusion,
        "terminal_exclusion_fraction": args.terminal_exclusion_fraction,
        "eligible_apex_interval": (
            "terminal-excluded search interval minus its two endpoints, which "
            "scipy.signal.find_peaks cannot select"
        ),
        "band_boundary": "median half-prominence limits across persistent scales",
        "n_protein_profiles": len(summary_rows),
        "n_bands": len(band_rows),
        "outputs": {
            "bands": str(bands_path),
            "protein_summary": str(summary_path),
        },
    }
    (output_dir / "signed_band_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n"
    )
    print(json.dumps({
        "protein_profiles": len(summary_rows),
        "bands": len(band_rows),
        "bands_csv": str(bands_path),
        "protein_summary_csv": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
