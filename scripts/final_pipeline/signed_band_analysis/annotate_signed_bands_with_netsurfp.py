#!/usr/bin/env python3
"""Join seed-averaged signed bands to Neq and NetSurfP annotations.

The script performs strict name/sequence/length/coordinate validation, builds a
single residue-level annotation table, and adds apex, local-window, and complete
band-interval annotations to every signed band.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


SPLITS = ("train", "validation", "test")
Q8_NONCOIL_ORDER = "GHIBEST"
Q8_ORDER = "GHIBESTC"
Q3_ORDER = "HEC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_csv_template", default="{split}_grouped_v1.csv")
    parser.add_argument(
        "--netsurfp_json_template",
        default="{split}_grouped_v1_netsurfp3.json",
    )
    parser.add_argument("--boundary_window", type=int, default=2)
    parser.add_argument("--local_window", type=int, default=5)
    parser.add_argument("--linker_min_length", type=int, default=2)
    parser.add_argument("--linker_max_length", type=int, default=20)
    parser.add_argument("--linker_flank_window", type=int, default=3)
    parser.add_argument("--neq_top_fraction", type=float, default=0.10)
    parser.add_argument("--neq_peak_smooth_window", type=int, default=3)
    parser.add_argument("--neq_peak_prominence_mad", type=float, default=1.0)
    parser.add_argument("--neq_peak_min_prominence", type=float, default=0.10)
    parser.add_argument("--neq_peak_min_distance", type=int, default=3)
    return parser.parse_args()


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(values))
    return scale if np.isfinite(scale) else 0.0


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float).copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(np.asarray(values, dtype=float), (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def transition_mask(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    if len(labels) > 1:
        transitions = np.flatnonzero(labels[:-1] != labels[1:])
        mask[transitions] = True
        mask[transitions + 1] = True
    return mask


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    hits = np.flatnonzero(mask)
    if len(hits) == 0:
        return np.full(n, np.nan, dtype=float)
    positions = np.arange(n)
    insertion = np.searchsorted(hits, positions)
    left_index = np.clip(insertion - 1, 0, len(hits) - 1)
    right_index = np.clip(insertion, 0, len(hits) - 1)
    return np.minimum(
        np.abs(positions - hits[left_index]),
        np.abs(positions - hits[right_index]),
    ).astype(float)


def structured_linker_mask(
    q3: np.ndarray,
    q8: np.ndarray,
    min_length: int,
    max_length: int,
    flank_window: int,
) -> np.ndarray:
    loop_like = np.isin(q8, list("CTS"))
    structured = np.isin(q3, list("HE"))
    mask = np.zeros(len(q3), dtype=bool)
    index = 0
    while index < len(q3):
        if not loop_like[index]:
            index += 1
            continue
        start = index
        while index < len(q3) and loop_like[index]:
            index += 1
        end = index - 1
        run_length = end - start + 1
        if not min_length <= run_length <= max_length:
            continue
        left = structured[max(0, start - flank_window):start]
        right = structured[end + 1:min(len(q3), end + 1 + flank_window)]
        if np.any(left) and np.any(right):
            mask[start:end + 1] = True
    return mask


def circular_difference_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + 180.0) % 360.0 - 180.0)


def build_protein_annotations(
    split: str,
    name: str,
    sequence: str,
    neq: np.ndarray,
    record: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    length = len(sequence)
    q3 = np.asarray(list(record["q3"]), dtype=object)
    q8 = np.asarray(list(record["q8"]), dtype=object)
    q3_prob = np.asarray(record["q3_prob"], dtype=float)
    q8_noncoil = np.asarray(record["q8_prob"], dtype=float)
    q8_coil = 1.0 - np.sum(q8_noncoil, axis=1)
    q8_prob = np.column_stack([q8_noncoil, q8_coil])
    if q3_prob.shape != (length, 3) or q8_prob.shape != (length, 8):
        raise ValueError(f"{split}/{name}: unexpected Q3/Q8 probability shape")
    if np.max(np.abs(np.sum(q3_prob, axis=1) - 1.0)) > 1e-4:
        raise ValueError(f"{split}/{name}: Q3 probabilities do not sum to one")
    if np.max(np.abs(np.sum(q8_prob, axis=1) - 1.0)) > 1e-4:
        raise ValueError(f"{split}/{name}: Q8 probabilities do not sum to one")
    predicted_q3 = np.asarray(list(Q3_ORDER))[np.argmax(q3_prob, axis=1)]
    predicted_q8 = np.asarray(list(Q8_ORDER))[np.argmax(q8_prob, axis=1)]
    if not np.array_equal(predicted_q3, q3) or not np.array_equal(predicted_q8, q8):
        raise ValueError(f"{split}/{name}: Q3/Q8 labels disagree with probabilities")

    q3_boundary = transition_mask(q3)
    q8_boundary = transition_mask(q8)
    q3_distance = distance_to_mask(q3_boundary)
    q8_distance = distance_to_mask(q8_boundary)
    linker = structured_linker_mask(
        q3,
        q8,
        args.linker_min_length,
        args.linker_max_length,
        args.linker_flank_window,
    )
    smoothed_neq = smooth(neq, args.neq_peak_smooth_window)
    prominence = max(
        args.neq_peak_min_prominence,
        args.neq_peak_prominence_mad * robust_scale(smoothed_neq),
    )
    neq_peak_indices, _ = find_peaks(
        smoothed_neq,
        prominence=prominence,
        distance=max(1, args.neq_peak_min_distance),
    )
    neq_peak = np.zeros(length, dtype=bool)
    neq_peak[neq_peak_indices] = True
    neq_peak_distance = distance_to_mask(neq_peak)
    # Retain all quantile ties and require actual flexibility.  Selecting an
    # exact count would arbitrarily mark positions among the many Neq == 1.0
    # residues when the quantile threshold lies at the rigid floor.
    neq_top_threshold = float(np.quantile(neq, 1.0 - args.neq_top_fraction))
    neq_top = (neq >= neq_top_threshold) & (neq > 1.0)
    neq_top_distance = distance_to_mask(neq_top)

    phi = np.asarray(record["phi"], dtype=float)
    psi = np.asarray(record["psi"], dtype=float)
    torsion_change = np.full(length, np.nan, dtype=float)
    if length > 1:
        torsion_change[1:] = np.hypot(
            circular_difference_degrees(phi[1:], phi[:-1]),
            circular_difference_degrees(psi[1:], psi[:-1]),
        )
    output = {
        "split": split,
        "protein": name,
        "protein_length": length,
        "residue_index_0based": np.arange(length),
        "residue_1based": np.arange(1, length + 1),
        "amino_acid": list(sequence),
        "neq": neq,
        "flexible_neq_gt1": neq > 1.0,
        "neq_top10_within_protein": neq_top,
        "neq_top10_threshold": np.full(length, neq_top_threshold),
        "neq_smoothed": smoothed_neq,
        "neq_peak": neq_peak,
        "distance_to_neq_peak": neq_peak_distance,
        "distance_to_neq_top10": neq_top_distance,
        "q3": q3,
        "q8": q8,
        "q3_boundary": q3_boundary,
        "q8_boundary": q8_boundary,
        "distance_to_q3_boundary": q3_distance,
        "distance_to_q8_boundary": q8_distance,
        "q3_boundary_within2": q3_distance <= args.boundary_window,
        "q8_boundary_within2": q8_distance <= args.boundary_window,
        "q8_turn_or_bend_TS": np.isin(q8, list("TS")),
        "q8_loop_turn_bend_CTS": np.isin(q8, list("CTS")),
        "structured_linker_loop": linker,
        "rsa": np.asarray(record["rsa"], dtype=float),
        "asa": np.asarray(record["asa"], dtype=float),
        "exposed_rsa_ge025": np.asarray(record["rsa"], dtype=float) >= 0.25,
        "disorder": np.asarray(record["disorder"], dtype=float),
        "disorder_ge05": np.asarray(record["disorder"], dtype=float) >= 0.5,
        "interface": np.asarray(record["interface"], dtype=int) > 0,
        "phi": phi,
        "psi": psi,
        "torsion_change_from_previous": torsion_change,
    }
    for index, label in enumerate(Q3_ORDER):
        output[f"q3_prob_{label}"] = q3_prob[:, index]
        output[f"q3_is_{label}"] = q3 == label
    for index, label in enumerate(Q8_ORDER):
        output[f"q8_prob_{label}"] = q8_prob[:, index]
        output[f"q8_is_{label}"] = q8 == label
    return pd.DataFrame(output)


def parse_neq(value) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def load_annotations(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    split_dir = Path(args.split_dir).expanduser().resolve()
    frames = []
    audit = {"splits": {}, "strain_available": False}
    for split in SPLITS:
        csv_path = split_dir / args.split_csv_template.format(split=split)
        json_path = split_dir / args.netsurfp_json_template.format(split=split)
        proteins = pd.read_csv(csv_path)
        records = json.loads(json_path.read_text())
        by_name = {str(record["desc"]): record for record in records}
        if len(by_name) != len(records):
            raise ValueError(f"{json_path}: duplicate desc identifiers")
        split_frames = []
        for row in proteins.itertuples(index=False):
            name = str(row.name)
            sequence = str(row.sequence)
            record = by_name.get(name)
            if record is None:
                raise ValueError(f"{split}/{name}: missing NetSurfP record")
            if str(record["seq"]) != sequence:
                raise ValueError(f"{split}/{name}: NetSurfP sequence mismatch")
            neq = parse_neq(row.neq)
            if len(neq) != len(sequence):
                raise ValueError(f"{split}/{name}: Neq length mismatch")
            for field in (
                "q3", "q8", "q3_prob", "q8_prob", "phi", "psi", "rsa",
                "asa", "disorder", "interface",
            ):
                if len(record[field]) != len(sequence):
                    raise ValueError(f"{split}/{name}: {field} length mismatch")
            split_frames.append(
                build_protein_annotations(split, name, sequence, neq, record, args)
            )
        split_frame = pd.concat(split_frames, ignore_index=True)
        frames.append(split_frame)
        audit["splits"][split] = {
            "protein_count": int(len(proteins)),
            "residue_count": int(len(split_frame)),
            "netsurfp_record_count": int(len(records)),
            "all_names_sequences_lengths_validated": True,
        }
    residue = pd.concat(frames, ignore_index=True)
    return residue, audit


def mean_boolean_or_numeric(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].astype(float).to_numpy()
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan


def annotate_bands(
    bands: pd.DataFrame,
    residue: pd.DataFrame,
    local_window: int,
) -> pd.DataFrame:
    residue_groups = {
        (split, protein): group.sort_values("residue_index_0based").reset_index(drop=True)
        for (split, protein), group in residue.groupby(["split", "protein"], sort=False)
    }
    apex_columns = [
        "neq", "flexible_neq_gt1", "neq_top10_within_protein", "neq_smoothed",
        "neq_peak", "distance_to_neq_peak", "distance_to_neq_top10", "q3", "q8",
        "q3_boundary", "q8_boundary", "distance_to_q3_boundary",
        "distance_to_q8_boundary", "q3_boundary_within2", "q8_boundary_within2",
        "q8_turn_or_bend_TS", "q8_loop_turn_bend_CTS", "structured_linker_loop",
        "rsa", "asa", "exposed_rsa_ge025", "disorder", "disorder_ge05",
        "interface", "phi", "psi", "torsion_change_from_previous",
    ] + [f"q3_prob_{label}" for label in Q3_ORDER] + [
        f"q8_prob_{label}" for label in Q8_ORDER
    ]
    interval_columns = [
        "neq", "flexible_neq_gt1", "neq_top10_within_protein", "neq_peak",
        "q3_boundary_within2", "q8_boundary_within2", "q8_turn_or_bend_TS",
        "q8_loop_turn_bend_CTS", "structured_linker_loop", "rsa",
        "exposed_rsa_ge025", "disorder", "disorder_ge05", "interface",
    ]
    rows = []
    for band in bands.itertuples(index=False):
        row = band._asdict()
        key = (str(band.split), str(band.protein))
        annotations = residue_groups.get(key)
        if annotations is None:
            raise ValueError(f"Band {band.band_id}: no residue annotations for {key}")
        length = len(annotations)
        if int(band.protein_length) != length:
            raise ValueError(f"Band {band.band_id}: protein length mismatch")
        apex = int(band.apex_index_0based)
        start = int(band.start_index_0based)
        end = int(band.end_index_0based_inclusive)
        if not (0 <= start <= apex <= end < length):
            raise ValueError(f"Band {band.band_id}: invalid coordinates")
        apex_row = annotations.iloc[apex]
        if str(apex_row.amino_acid) != str(band.apex_amino_acid):
            raise ValueError(f"Band {band.band_id}: apex amino-acid mismatch")
        for column in apex_columns:
            row[f"apex_{column}"] = apex_row[column]
        interval = annotations.iloc[start:end + 1]
        local = annotations.iloc[max(0, apex - local_window):min(length, apex + local_window + 1)]
        row["interval_annotation_residue_count"] = len(interval)
        row["interval_neq_max"] = float(interval["neq"].max())
        row["local_window_radius"] = local_window
        for column in interval_columns:
            row[f"interval_mean_{column}"] = mean_boolean_or_numeric(interval, column)
            row[f"local_mean_{column}"] = mean_boolean_or_numeric(local, column)
        for label in Q3_ORDER:
            row[f"interval_fraction_q3_{label}"] = float(np.mean(interval["q3"] == label))
            row[f"local_fraction_q3_{label}"] = float(np.mean(local["q3"] == label))
        for label in Q8_ORDER:
            row[f"interval_fraction_q8_{label}"] = float(np.mean(interval["q8"] == label))
            row[f"local_fraction_q8_{label}"] = float(np.mean(local["q8"] == label))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.boundary_window < 0 or args.local_window < 0:
        raise ValueError("Windows must be nonnegative")
    if not 0 < args.neq_top_fraction < 1:
        raise ValueError("--neq_top_fraction must be between zero and one")
    if args.neq_peak_smooth_window < 1 or args.neq_peak_smooth_window % 2 == 0:
        raise ValueError("--neq_peak_smooth_window must be a positive odd integer")
    bands_path = Path(args.bands_csv).expanduser().resolve()
    bands = pd.read_csv(bands_path)
    required = {
        "band_id", "condition", "split", "protein", "protein_length", "sign",
        "label", "apex_index_0based", "apex_amino_acid", "start_index_0based",
        "end_index_0based_inclusive", "band_width",
    }
    missing = required - set(bands.columns)
    if missing:
        raise ValueError(f"{bands_path} lacks columns: {sorted(missing)}")
    if bands["band_id"].duplicated().any():
        raise ValueError("band_id values must be unique")
    residue, audit = load_annotations(args)
    if residue.duplicated(["split", "protein", "residue_index_0based"]).any():
        raise ValueError("Duplicate residue annotation keys")
    annotated = annotate_bands(bands, residue, args.local_window)
    if not np.array_equal(
        annotated["interval_annotation_residue_count"].to_numpy(),
        annotated["band_width"].to_numpy(),
    ):
        raise ValueError("At least one band interval annotation count differs from band_width")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    residue_path = output_dir / "residue_biophysical_annotations.csv.gz"
    annotated_path = output_dir / "signed_bands_biophysical_annotations.csv.gz"
    residue.to_csv(residue_path, index=False, compression="gzip")
    annotated.to_csv(annotated_path, index=False, compression="gzip")
    audit.update({
        "bands_csv": str(bands_path),
        "split_dir": str(Path(args.split_dir).expanduser().resolve()),
        "residue_rows": int(len(residue)),
        "unique_proteins": int(residue[["split", "protein"]].drop_duplicates().shape[0]),
        "band_rows": int(len(annotated)),
        "conditions": sorted(annotated["condition"].unique().tolist()),
        "all_band_coordinates_and_apex_amino_acids_validated": True,
        "band_interval_counts_equal_band_width": True,
        "strain_note": (
            "No v2 per-residue strain_summary.csv files were found in data_splits; "
            "strain was not approximated or analyzed."
        ),
        "neq_top10_definition": (
            "Neq >= the within-protein (1 - neq_top_fraction) quantile and Neq > 1.0; "
            "all quantile ties are retained and proteins with no flexible residues have no hits."
        ),
        "parameters": vars(args),
        "outputs": {
            "residue_annotations": str(residue_path),
            "annotated_bands": str(annotated_path),
        },
    })
    audit_path = output_dir / "annotation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        "residue_rows": len(residue),
        "band_rows": len(annotated),
        "residue_annotations": str(residue_path),
        "annotated_bands": str(annotated_path),
        "audit": str(audit_path),
    }, indent=2))


if __name__ == "__main__":
    main()
