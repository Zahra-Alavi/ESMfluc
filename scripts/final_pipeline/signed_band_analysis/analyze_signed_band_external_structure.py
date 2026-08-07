#!/usr/bin/env python3
"""Phase 3B: analyze external structure and mechanics of Q8 selection.

Cases and controls are inherited from the Phase 3A Q8-segment candidate table:
cases contain a band apex of one sign, while controls are same-protein,
identical-Q8 segments outside every band interval. Experimental coordinates are
accepted only through the alignment-audited output of
Attention/build_contact_maps_from_pdb.py.

Structure-based sequential models are trained on train and evaluated without
refitting on validation/test. ATLAS strain is intentionally excluded from those
models because the current strain collection is test-only; it enters only
within-protein descriptive/confirmatory matched effects.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import zlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# NetworkX 2.3 still refers to NumPy aliases removed in NumPy 1.24.
for _alias, _type in (("int", int), ("float", float), ("bool", bool)):
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)
import networkx as nx  # noqa: E402


BASE_FEATURES = [
    "mean_neq", "max_neq", "mean_rsa", "max_rsa",
    "log_segment_length", "normalized_midpoint",
]
LOCAL_ANNOTATION_FEATURES = [
    "mean_torsion_change", "max_torsion_change",
    "q3_boundary_fraction", "mean_distance_to_q3_boundary",
    "structured_linker_fraction",
]
PDB_GEOMETRY_FEATURES = [
    "mean_ca_curvature_degrees",
    "max_ca_curvature_degrees", "mean_abs_ca_virtual_torsion_degrees",
    "max_abs_ca_virtual_torsion_degrees", "ca_end_to_end_ratio",
]
GEOMETRY_FEATURES = LOCAL_ANNOTATION_FEATURES + PDB_GEOMETRY_FEATURES
NETWORK_FEATURES = [
    "mean_contact_degree", "max_contact_degree",
    "mean_inverse_distance_weighted_degree",
    "max_inverse_distance_weighted_degree",
    "mean_betweenness", "max_betweenness", "mean_closeness",
    "mean_participation_coefficient", "community_boundary_fraction",
]
DOMAIN_FEATURES = [
    "mean_distance_to_domain_boundary", "near_domain_boundary_fraction",
    "mean_cross_domain_contacts", "mean_cross_domain_contact_fraction",
]
STRAIN_FEATURES = [
    "mean_strain", "max_strain", "std_strain",
    "mean_strain_spatial_gradient", "max_strain_spatial_gradient",
]
EXTERNAL_FEATURE_GROUPS = {
    "experimental_geometry": PDB_GEOMETRY_FEATURES,
    "contact_network": NETWORK_FEATURES,
    "inter_domain_geometry": DOMAIN_FEATURES,
    "mechanical_strain_test_only": STRAIN_FEATURES,
}
STAGES = {
    "00_q8_only": [],
    "01_base_biophysics": BASE_FEATURES,
    "02_add_local_sequence_or_annotation_geometry": (
        BASE_FEATURES + LOCAL_ANNOTATION_FEATURES
    ),
    "03_add_experimental_pdb_geometry": BASE_FEATURES + GEOMETRY_FEATURES,
    "04_add_contact_network": BASE_FEATURES + GEOMETRY_FEATURES + NETWORK_FEATURES,
    "05_add_ecod_domain": (
        BASE_FEATURES + GEOMETRY_FEATURES + NETWORK_FEATURES + DOMAIN_FEATURES
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_csv", required=True, help="Phase 3A q8_segment_candidates.csv.gz")
    parser.add_argument(
        "--contact_json", nargs="+", required=True,
        help="One or more alignment-audited contact .json/.json.gz files",
    )
    parser.add_argument("--ecod_csv", default=None, help="Optional ECOD annotation CSV with pdb_range")
    parser.add_argument("--strain_root", default=None, help="Optional test-only strain-result root")
    parser.add_argument("--mechanism_csv", default=None, help="Optional Phase 3C band mechanism table")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--terminal_exclusion", type=int, default=10)
    parser.add_argument("--min_mapping_identity", type=float, default=0.90)
    parser.add_argument("--min_input_coverage", type=float, default=0.80)
    parser.add_argument("--min_segment_resolved_fraction", type=float, default=0.80)
    parser.add_argument("--min_contact_sequence_separation", type=int, default=3)
    parser.add_argument("--domain_boundary_window", type=int, default=3)
    parser.add_argument("--betweenness_samples", type=int, default=64)
    parser.add_argument("--minimum_inference_proteins", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=123)
    args = parser.parse_args()
    for name in ("min_mapping_identity", "min_input_coverage", "min_segment_resolved_fraction"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must lie in [0, 1]")
    if args.minimum_inference_proteins < 2:
        parser.error("--minimum_inference_proteins must be at least 2")
    if args.terminal_exclusion < 0 or args.min_contact_sequence_separation < 1:
        parser.error("terminal exclusion must be nonnegative and sequence separation positive")
    return args


def read_json(path: str):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def safe_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def safe_max(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else np.nan


def bh(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return output
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index.to_numpy()[order]] = np.minimum(adjusted, 1.0)
    return output


def load_contacts(paths: Iterable[str]) -> dict[str, dict]:
    records = {}
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON list")
        for record in payload:
            name = str(record.get("name", ""))
            if not name:
                raise ValueError(f"{path}: contact record lacks name")
            if name in records:
                raise ValueError(f"Duplicate contact record for {name}")
            record["contact_source_json"] = str(Path(path).expanduser().resolve())
            records[name] = record
    return records


def audit_contacts(records: dict[str, dict], protein_split: dict[str, str], args) -> pd.DataFrame:
    rows = []
    for name in sorted(set(records) | set(protein_split)):
        record = records.get(name, {})
        alignment = record.get("alignment") or {}
        sequence_check = record.get("sequence_check") or {}
        high_quality = (
            record.get("status") == "ok"
            and float(alignment.get("identity", -1)) >= args.min_mapping_identity
            and float(alignment.get("input_coverage", -1)) >= args.min_input_coverage
            and record.get("contact_map_coordinate_system") == "input_sequence_0based"
        )
        rows.append({
            "protein": name, "split": record.get("split", protein_split.get(name)),
            "status": record.get("status", "missing_contact_record"), "error": record.get("error"),
            "mapping_status": record.get("mapping_status"), "mapping_accepted": high_quality,
            "pdb_id": record.get("pdb_id"), "chain": record.get("chain"),
            "experimental_method": record.get("experimental_method"),
            "resolution_angstrom": record.get("resolution_angstrom"),
            "n_models": record.get("n_models"),
            "n_residues_input": sequence_check.get("len_input", len(record.get("sequence", ""))),
            "n_residues_structure_ca": sequence_check.get("len_structure"),
            "alignment_identity": alignment.get("identity"),
            "input_coverage": alignment.get("input_coverage"),
            "structure_coverage": alignment.get("structure_coverage"),
            "n_input_unresolved": alignment.get("input_unresolved_count"),
            "n_structure_unmapped": alignment.get("structure_unmapped_count"),
            "n_contact_edges": record.get("n_contact_edges"),
            "contact_source_json": record.get("contact_source_json"),
        })
    return pd.DataFrame(rows)


def contact_edges(record: dict) -> list[tuple[int, int, float]]:
    if "contact_edges" in record:
        return [(int(i), int(j), float(distance)) for i, j, distance in record["contact_edges"]]
    if "contact_map" not in record:
        raise ValueError("record has neither contact_edges nor contact_map")
    matrix = np.asarray(record["contact_map"], dtype=bool)
    ii, jj = np.where(np.triu(matrix, k=1))
    coordinates = record.get("ca_coordinates_angstrom")
    edges = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        distance = np.nan
        if coordinates and coordinates[i] is not None and coordinates[j] is not None:
            distance = float(np.linalg.norm(np.asarray(coordinates[i]) - np.asarray(coordinates[j])))
        edges.append((i, j, distance))
    return edges


def ca_curvature(coordinates) -> np.ndarray:
    n = len(coordinates)
    output = np.full(n, np.nan)
    for index in range(1, n - 1):
        if any(coordinates[position] is None for position in (index - 1, index, index + 1)):
            continue
        left, center, right = (np.asarray(coordinates[position], float) for position in (index - 1, index, index + 1))
        first, second = center - left, right - center
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator > 0:
            output[index] = np.degrees(np.arccos(np.clip(np.dot(first, second) / denominator, -1, 1)))
    return output


def ca_virtual_torsion(coordinates) -> np.ndarray:
    """Absolute virtual C-alpha dihedral, assigned to the second central residue."""
    output = np.full(len(coordinates), np.nan)
    for index in range(1, len(coordinates) - 2):
        if any(coordinates[position] is None for position in range(index - 1, index + 3)):
            continue
        p0, p1, p2, p3 = (
            np.asarray(coordinates[position], float) for position in range(index - 1, index + 3)
        )
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        norm = np.linalg.norm(b1)
        if norm == 0:
            continue
        b1 = b1 / norm
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        if np.linalg.norm(v) == 0 or np.linalg.norm(w) == 0:
            continue
        output[index] = abs(float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))))
    return output


def graph_features(record: dict, args) -> tuple[dict[str, np.ndarray], list[tuple[int, int, float]]]:
    length = len(record["sequence"])
    resolved = np.asarray(record.get("resolved_mask", [True] * length), dtype=bool)
    edges = [
        (i, j, distance) for i, j, distance in contact_edges(record)
        if abs(i - j) >= args.min_contact_sequence_separation and resolved[i] and resolved[j]
    ]
    graph = nx.Graph()
    graph.add_nodes_from(np.flatnonzero(resolved).tolist())
    graph.add_edges_from((i, j, {"distance": distance}) for i, j, distance in edges)

    degree = np.full(length, np.nan)
    weighted_degree = np.full(length, np.nan)
    betweenness = np.full(length, np.nan)
    closeness = np.full(length, np.nan)
    participation = np.full(length, np.nan)
    community_boundary = np.full(length, np.nan)
    for node in graph.nodes:
        degree[node] = float(graph.degree[node])
        incident_distances = [data["distance"] for _, _, data in graph.edges(node, data=True)]
        if all(np.isfinite(distance) and distance > 0 for distance in incident_distances):
            weighted_degree[node] = sum(1.0 / distance for distance in incident_distances)

    if graph.number_of_nodes():
        seed = (args.random_seed + zlib.crc32(record["name"].encode())) % (2**32)
        k = min(args.betweenness_samples, graph.number_of_nodes())
        between = nx.betweenness_centrality(
            graph, k=None if k == graph.number_of_nodes() else k, normalized=True, seed=seed
        )
        close = nx.closeness_centrality(graph)
        for node, value in between.items():
            betweenness[node] = value
        for node, value in close.items():
            closeness[node] = value

        if graph.number_of_edges():
            communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        else:
            communities = [frozenset([node]) for node in graph.nodes]
        membership = {node: number for number, community in enumerate(communities) for node in community}
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                participation[node] = 0.0
                community_boundary[node] = 0.0
                continue
            counts = pd.Series([membership[neighbor] for neighbor in neighbors]).value_counts().to_numpy(float)
            participation[node] = 1.0 - float(np.sum((counts / len(neighbors)) ** 2))
            community_boundary[node] = float(any(membership[neighbor] != membership[node] for neighbor in neighbors))
    return {
        "resolved": resolved.astype(float), "contact_degree": degree,
        "inverse_distance_weighted_degree": weighted_degree,
        "betweenness": betweenness, "closeness": closeness,
        "participation_coefficient": participation,
        "community_boundary": community_boundary,
    }, edges


RANGE_PATTERN = re.compile(r"([^:,]+):(-?\d+)[A-Za-z]?\s*-\s*(-?\d+)[A-Za-z]?")


def domain_intervals(ecod: Optional[pd.DataFrame], protein: str) -> list[tuple[str, str, int, int]]:
    if ecod is None:
        return []
    rows = []
    for row in ecod[ecod.name.astype(str) == protein].itertuples(index=False):
        for chain, start, end in RANGE_PATTERN.findall(str(row.pdb_range)):
            rows.append((str(row.ecod_domain_id), chain.strip(), int(start), int(end)))
    return rows


def add_domain_features(
    record: dict, node: dict[str, np.ndarray], edges: list[tuple[int, int, float]],
    intervals: list[tuple[str, str, int, int]], args,
) -> dict[str, np.ndarray]:
    length = len(record["sequence"])
    labels = np.full(length, None, dtype=object)
    for mapping in record.get("residue_mapping", []):
        index = int(mapping["input_index_0based"])
        chain = str(mapping.get("pdb_chain_id") or mapping.get("pdb_segid") or "")
        resid = int(mapping["pdb_resid"])
        domains = sorted({domain for domain, expected_chain, start, end in intervals
                          if chain == expected_chain and min(start, end) <= resid <= max(start, end)})
        if domains:
            labels[index] = ";".join(domains)

    distance = np.full(length, np.nan)
    near = np.full(length, np.nan)
    labeled = np.flatnonzero(pd.notna(labels))
    if len(labeled):
        boundary_indices = []
        for label in sorted(set(labels[labeled])):
            positions = np.flatnonzero(labels == label)
            if len(positions):
                boundary_indices.extend([positions.min(), positions.max()])
        boundary_indices = np.asarray(sorted(set(boundary_indices)), dtype=int)
        for index in labeled:
            distance[index] = float(np.min(np.abs(boundary_indices - index)))
            near[index] = float(distance[index] <= args.domain_boundary_window)

    cross_count = np.full(length, np.nan)
    cross_fraction = np.full(length, np.nan)
    for index in labeled:
        cross_count[index] = 0.0
        cross_fraction[index] = 0.0
    total = np.zeros(length, dtype=float)
    cross = np.zeros(length, dtype=float)
    for i, j, _ in edges:
        if labels[i] is None or labels[j] is None:
            continue
        total[[i, j]] += 1
        if labels[i] != labels[j]:
            cross[[i, j]] += 1
    for index in labeled:
        cross_count[index] = cross[index]
        cross_fraction[index] = cross[index] / total[index] if total[index] else 0.0
    node.update({
        "domain_label": labels, "distance_to_domain_boundary": distance,
        "near_domain_boundary": near, "cross_domain_contacts": cross_count,
        "cross_domain_contact_fraction": cross_fraction,
    })
    return node


def load_strain(strain_root: Optional[str], protein: str, length: int, terminal: int):
    empty = {
        "strain": np.full(length, np.nan),
        "strain_spatial_gradient": np.full(length, np.nan),
        "strain_status": "not_requested" if not strain_root else "missing",
        "strain_source": None,
    }
    if not strain_root:
        return empty
    path = Path(strain_root).expanduser() / protein / "strain_summary.csv"
    empty["strain_source"] = str(path.resolve())
    if not path.exists():
        return empty
    data = pd.read_csv(path)
    required = {"residue", "ensemble_mean", "ensemble_std"}
    if required - set(data):
        empty["strain_status"] = "invalid_schema"
        return empty
    residues = pd.to_numeric(data.residue, errors="coerce").to_numpy()
    if len(data) != length or not np.array_equal(residues, np.arange(1, length + 1)):
        empty["strain_status"] = "index_or_length_mismatch"
        return empty
    values = pd.to_numeric(data.ensemble_mean, errors="coerce").to_numpy(float)
    if terminal:
        values[:terminal] = np.nan
        values[max(0, length - terminal):] = np.nan
    gradient = np.full(length, np.nan)
    gradient[1:] = np.abs(np.diff(values))
    empty.update({"strain": values, "strain_spatial_gradient": gradient, "strain_status": "ok"})
    return empty


def segment_path_ratio(coordinates, start: int, end: int) -> float:
    if end - start < 2:
        return np.nan
    points = coordinates[start:end]
    if any(point is None for point in points):
        return np.nan
    points = np.asarray(points, float)
    path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return float(np.linalg.norm(points[-1] - points[0]) / path) if path > 0 else np.nan


def coordinate_valid_mask(coordinates, length: int) -> np.ndarray:
    """Return which model-sequence positions have finite 3D C-alpha coordinates."""
    valid = np.zeros(length, dtype=bool)
    if coordinates is None or len(coordinates) != length:
        return valid
    for index, point in enumerate(coordinates):
        if point is None:
            continue
        values = np.asarray(point, dtype=float)
        valid[index] = values.shape == (3,) and bool(np.all(np.isfinite(values)))
    return valid


def segment_coordinate_completeness(
    *,
    coordinates,
    resolved_mask,
    curvature_values,
    torsion_values,
    contact_degree_values,
    start: int,
    end: int,
    mapping_accepted: bool,
) -> dict:
    """Audit coordinate and derived-feature completeness for one segment.

    Curvature is mathematically intended only at protein positions 1..L-2 and
    virtual torsion only at 1..L-3 (zero-based). A segment touching a true
    protein terminus is therefore not penalized for nonexistent outside-chain
    residues. Empty intended-position sets are complete by vacuous truth, while
    their finite fractions are reported as missing because no value is defined.
    End-to-end geometry additionally requires a segment of at least two residues.
    """
    length = len(resolved_mask)
    if not (0 <= start < end <= length):
        raise ValueError(f"Invalid segment [{start}, {end}) for protein length {length}")
    resolved = np.asarray(resolved_mask, dtype=bool)
    coordinate_valid = coordinate_valid_mask(coordinates, length)
    curvature = np.asarray(curvature_values, dtype=float)
    torsion = np.asarray(torsion_values, dtype=float)
    degree = np.asarray(contact_degree_values, dtype=float)
    if any(len(values) != length for values in (curvature, torsion, degree)):
        raise ValueError("Per-residue feature arrays must match protein length")

    segment_positions = np.arange(start, end, dtype=int)
    curvature_positions = np.arange(max(start, 1), min(end, length - 1), dtype=int)
    torsion_positions = np.arange(max(start, 1), min(end, length - 2), dtype=int)

    curvature_requirements = [
        bool(np.all(coordinate_valid[index - 1:index + 2]))
        for index in curvature_positions
    ]
    torsion_requirements = [
        bool(np.all(coordinate_valid[index - 1:index + 3]))
        for index in torsion_positions
    ]
    n_finite_curvature = int(np.isfinite(curvature[curvature_positions]).sum())
    n_finite_torsion = int(np.isfinite(torsion[torsion_positions]).sum())
    n_finite_degree = int(np.isfinite(degree[segment_positions]).sum())
    n_curvature_intended = len(curvature_positions)
    n_torsion_intended = len(torsion_positions)
    n_segment = len(segment_positions)

    segment_ca_complete = bool(
        mapping_accepted
        and np.all(resolved[segment_positions])
        and np.all(coordinate_valid[segment_positions])
    )
    curvature_complete = bool(mapping_accepted and all(curvature_requirements))
    torsion_complete = bool(mapping_accepted and all(torsion_requirements))
    end_to_end_complete = bool(
        mapping_accepted and n_segment >= 2 and np.all(coordinate_valid[segment_positions])
    )
    network_complete = bool(
        mapping_accepted
        and np.all(resolved[segment_positions])
        and n_finite_degree == n_segment
    )
    experimental_complete = bool(
        segment_ca_complete
        and curvature_complete
        and torsion_complete
        and end_to_end_complete
    )
    return {
        "n_segment_residues": n_segment,
        "n_resolved_segment_residues": int(resolved[segment_positions].sum()),
        "n_coordinate_resolved_segment_residues": int(
            coordinate_valid[segment_positions].sum()
        ),
        "n_intended_curvature_values": n_curvature_intended,
        "n_finite_curvature_values": n_finite_curvature,
        "curvature_finite_fraction": (
            n_finite_curvature / n_curvature_intended
            if n_curvature_intended else np.nan
        ),
        "n_intended_torsion_values": n_torsion_intended,
        "n_finite_torsion_values": n_finite_torsion,
        "torsion_finite_fraction": (
            n_finite_torsion / n_torsion_intended if n_torsion_intended else np.nan
        ),
        "n_finite_contact_degree_values": n_finite_degree,
        "contact_degree_finite_fraction": n_finite_degree / n_segment,
        "segment_ca_fully_resolved": segment_ca_complete,
        "curvature_coordinate_complete": curvature_complete,
        "torsion_coordinate_complete": torsion_complete,
        "end_to_end_coordinate_complete": end_to_end_complete,
        "network_coordinate_complete": network_complete,
        "experimental_geometry_complete": experimental_complete,
    }


def aggregate_segments(candidates: pd.DataFrame, records: dict, ecod, args) -> pd.DataFrame:
    protein_arrays = {}
    split_by_protein = dict(
        candidates[["protein", "split"]].drop_duplicates().astype(str).itertuples(index=False, name=None)
    )
    for protein in candidates.protein.astype(str).unique():
        record = records.get(protein)
        if not record or record.get("status") != "ok":
            continue
        alignment = record.get("alignment") or {}
        if (
            float(alignment.get("identity", -1)) < args.min_mapping_identity
            or float(alignment.get("input_coverage", -1)) < args.min_input_coverage
            or record.get("contact_map_coordinate_system") != "input_sequence_0based"
        ):
            continue
        node, edges = graph_features(record, args)
        coordinates = record.get("ca_coordinates_angstrom")
        if coordinates:
            node["ca_curvature_degrees"] = ca_curvature(coordinates)
            node["abs_ca_virtual_torsion_degrees"] = ca_virtual_torsion(coordinates)
        else:
            node["ca_curvature_degrees"] = np.full(len(record["sequence"]), np.nan)
            node["abs_ca_virtual_torsion_degrees"] = np.full(len(record["sequence"]), np.nan)
        node = add_domain_features(record, node, edges, domain_intervals(ecod, protein), args)
        if split_by_protein.get(protein) == "test":
            node.update(load_strain(args.strain_root, protein, len(record["sequence"]), args.terminal_exclusion))
        else:
            node.update(load_strain(None, protein, len(record["sequence"]), args.terminal_exclusion))
            node["strain_status"] = "unavailable_non_test_by_design"
        node["coordinates"] = coordinates
        node["record"] = record
        protein_arrays[protein] = node

    rows = []
    for candidate in candidates.to_dict("records"):
        row = dict(candidate)
        protein = str(row["protein"])
        start, end = int(row["start_index_0based"]), int(row["end_index_0based_exclusive"])
        arrays = protein_arrays.get(protein)
        row.update({
            "structure_available": arrays is not None,
            "mapping_status": None, "alignment_identity": np.nan,
            "protein_structure_coverage": np.nan, "segment_resolved_fraction": np.nan,
            "segment_structure_eligible": False, "strain_status": "structure_unavailable",
            "n_segment_residues": end - start,
            "n_resolved_segment_residues": 0,
            "n_coordinate_resolved_segment_residues": 0,
            "n_intended_curvature_values": max(
                0, min(end, int(row["protein_length"]) - 1) - max(start, 1)
            ),
            "n_finite_curvature_values": 0,
            "curvature_finite_fraction": np.nan,
            "n_intended_torsion_values": max(
                0, min(end, int(row["protein_length"]) - 2) - max(start, 1)
            ),
            "n_finite_torsion_values": 0,
            "torsion_finite_fraction": np.nan,
            "n_finite_contact_degree_values": 0,
            "contact_degree_finite_fraction": 0.0,
            "segment_ca_fully_resolved": False,
            "curvature_coordinate_complete": False,
            "torsion_coordinate_complete": False,
            "end_to_end_coordinate_complete": False,
            "network_coordinate_complete": False,
            "experimental_geometry_complete": False,
        })
        for feature in sum(EXTERNAL_FEATURE_GROUPS.values(), []):
            row[feature] = np.nan
        row["ca_end_to_end_ratio"] = np.nan
        if arrays is not None:
            record = arrays["record"]
            alignment = record["alignment"]
            resolved_fraction = safe_mean(arrays["resolved"][start:end])
            completeness = segment_coordinate_completeness(
                coordinates=arrays["coordinates"],
                resolved_mask=np.asarray(arrays["resolved"], dtype=bool),
                curvature_values=arrays["ca_curvature_degrees"],
                torsion_values=arrays["abs_ca_virtual_torsion_degrees"],
                contact_degree_values=arrays["contact_degree"],
                start=start,
                end=end,
                mapping_accepted=True,
            )
            row.update({
                "mapping_status": record.get("mapping_status"),
                "alignment_identity": alignment.get("identity"),
                "protein_structure_coverage": alignment.get("input_coverage"),
                "segment_resolved_fraction": resolved_fraction,
                "segment_structure_eligible": bool(
                    np.isfinite(resolved_fraction)
                    and resolved_fraction >= args.min_segment_resolved_fraction
                ),
                "strain_status": arrays["strain_status"],
                "mean_ca_curvature_degrees": safe_mean(arrays["ca_curvature_degrees"][start:end]),
                "max_ca_curvature_degrees": safe_max(arrays["ca_curvature_degrees"][start:end]),
                "mean_abs_ca_virtual_torsion_degrees": safe_mean(arrays["abs_ca_virtual_torsion_degrees"][start:end]),
                "max_abs_ca_virtual_torsion_degrees": safe_max(arrays["abs_ca_virtual_torsion_degrees"][start:end]),
                "ca_end_to_end_ratio": (
                    segment_path_ratio(arrays["coordinates"], start, end)
                    if arrays["coordinates"] else np.nan
                ),
                "mean_contact_degree": safe_mean(arrays["contact_degree"][start:end]),
                "max_contact_degree": safe_max(arrays["contact_degree"][start:end]),
                "mean_inverse_distance_weighted_degree": safe_mean(arrays["inverse_distance_weighted_degree"][start:end]),
                "max_inverse_distance_weighted_degree": safe_max(arrays["inverse_distance_weighted_degree"][start:end]),
                "mean_betweenness": safe_mean(arrays["betweenness"][start:end]),
                "max_betweenness": safe_max(arrays["betweenness"][start:end]),
                "mean_closeness": safe_mean(arrays["closeness"][start:end]),
                "mean_participation_coefficient": safe_mean(arrays["participation_coefficient"][start:end]),
                "community_boundary_fraction": safe_mean(arrays["community_boundary"][start:end]),
                "mean_distance_to_domain_boundary": safe_mean(arrays["distance_to_domain_boundary"][start:end]),
                "near_domain_boundary_fraction": safe_mean(arrays["near_domain_boundary"][start:end]),
                "mean_cross_domain_contacts": safe_mean(arrays["cross_domain_contacts"][start:end]),
                "mean_cross_domain_contact_fraction": safe_mean(arrays["cross_domain_contact_fraction"][start:end]),
                "mean_strain": safe_mean(arrays["strain"][start:end]),
                "max_strain": safe_max(arrays["strain"][start:end]),
                "std_strain": float(np.nanstd(arrays["strain"][start:end], ddof=1))
                    if np.isfinite(arrays["strain"][start:end]).sum() > 1 else np.nan,
                "mean_strain_spatial_gradient": safe_mean(arrays["strain_spatial_gradient"][start:end]),
                "max_strain_spatial_gradient": safe_max(arrays["strain_spatial_gradient"][start:end]),
                **completeness,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_rows(candidates: pd.DataFrame, sign: int) -> pd.DataFrame:
    selected = "selected_positive" if sign == 1 else "selected_negative"
    opposite = "selected_negative" if sign == 1 else "selected_positive"
    cases = candidates[candidates[selected] & ~candidates[opposite]].copy()
    controls = candidates[candidates.clean_control].copy()
    cases["selected"], controls["selected"] = 1, 0
    data = pd.concat([cases, controls], ignore_index=True)
    data["sign"] = sign
    return reconstruct_matched_strata(data)


def reconstruct_matched_strata(data: pd.DataFrame) -> pd.DataFrame:
    """Retain only protein/Q8/sign strata containing both cases and controls."""
    if data.empty:
        return data.copy()
    keys = ["condition", "split", "protein", "q8", "sign"]
    counts = data.groupby(keys + ["selected"]).size().unstack(fill_value=0)
    control_counts = (
        counts[0] if 0 in counts.columns
        else pd.Series(0, index=counts.index, dtype=int)
    )
    selected_counts = (
        counts[1] if 1 in counts.columns
        else pd.Series(0, index=counts.index, dtype=int)
    )
    valid = counts[(control_counts > 0) & (selected_counts > 0)].reset_index()[keys]
    if valid.empty:
        return data.iloc[0:0].copy()
    return data.merge(valid, on=keys, how="inner")


def matched_effects(data: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = sum(EXTERNAL_FEATURE_GROUPS.values(), [])
    keys = ["condition", "split", "protein", "sign", "q8"]
    rows = []
    for key, group in data.groupby(keys, sort=False):
        cases, controls = group[group.selected == 1], group[group.selected == 0]
        for feature in features:
            case = pd.to_numeric(cases[feature], errors="coerce")
            control = pd.to_numeric(controls[feature], errors="coerce")
            if case.notna().any() and control.notna().any():
                rows.append({
                    **dict(zip(keys, key)), "feature": feature,
                    "feature_group": next(name for name, values in EXTERNAL_FEATURE_GROUPS.items() if feature in values),
                    "n_selected_segments": int(case.notna().sum()),
                    "n_control_segments": int(control.notna().sum()),
                    "selected_mean": float(case.mean()), "control_mean": float(control.mean()),
                    "selected_minus_control": float(case.mean() - control.mean()),
                })
    per_protein = pd.DataFrame(rows)
    summaries = summarize_effect_rows(per_protein, args.minimum_inference_proteins)
    return per_protein, summaries


def summarize_effect_rows(per_protein: pd.DataFrame, minimum: int, extra_keys=None) -> pd.DataFrame:
    if per_protein.empty:
        return pd.DataFrame()
    extra_keys = extra_keys or []
    keys = ["condition", "split", "sign"] + extra_keys + ["q8", "feature", "feature_group"]
    rows = []
    for key, group in per_protein.groupby(keys, sort=False, dropna=False):
        values = group.selected_minus_control.dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        eligible = n >= minimum
        half = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if eligible else np.nan
        if eligible:
            try:
                pvalue = float(wilcoxon(values, zero_method="zsplit", method="approx").pvalue)
            except ValueError:
                pvalue = 1.0
        else:
            pvalue = np.nan
        rows.append({
            **dict(zip(keys, key)), "n_proteins": n,
            "n_selected_segments": int(group.n_selected_segments.sum()),
            "n_control_segments": int(group.n_control_segments.sum()),
            "selected_macro_protein_mean": float(group.selected_mean.mean()),
            "control_macro_protein_mean": float(group.control_mean.mean()),
            "selected_minus_control_macro_mean": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "inference_eligible": eligible, "wilcoxon_p_two_sided": pvalue,
        })
    result = pd.DataFrame(rows)
    result["wilcoxon_q_bh"] = bh(result.wilcoxon_p_two_sided)
    return result


def add_weights(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "split", "protein", "q8", "sign", "selected"]
    sizes = data.groupby(keys).size().rename("class_size").reset_index()
    data = data.merge(sizes, on=keys, how="left", validate="many_to_one")
    data["sample_weight"] = 0.5 / data.class_size
    return data


def within_stratum_concordance(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, int]:
    work = frame[["protein", "q8", "selected"]].copy()
    work["score"] = scores
    values = [roc_auc_score(group.selected, group.score)
              for _, group in work.groupby(["protein", "q8"])
              if group.selected.nunique() == 2]
    return (float(np.mean(values)) if values else np.nan, len(values))


def sequential_models(data: pd.DataFrame, args) -> pd.DataFrame:
    data = reconstruct_matched_strata(
        data[data.segment_structure_eligible].copy()
    )
    data = add_weights(data)
    rows = []
    for condition in sorted(data.condition.unique()):
        for sign in (-1, 1):
            train = data[(data.condition == condition) & (data.sign == sign) & (data.split == "train")]
            for stage, features in STAGES.items():
                if train.selected.nunique() < 2:
                    for split in ("validation", "test"):
                        rows.append({
                            "condition": condition, "sign": sign, "stage": stage,
                            "evaluation_split": split, "model_status": "not_fit_no_structural_train_contrast",
                        })
                    continue
                transformer = ColumnTransformer([
                    ("q8", OneHotEncoder(handle_unknown="ignore"), ["q8"]),
                    ("numeric", Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), features),
                ])
                model = Pipeline([
                    ("transform", transformer),
                    ("logistic", LogisticRegression(
                        C=1.0, max_iter=args.max_iter, solver="lbfgs",
                        random_state=args.random_seed,
                    )),
                ])
                try:
                    model.fit(train[["q8"] + features], train.selected,
                              logistic__sample_weight=train.sample_weight)
                except ValueError as exc:
                    for split in ("validation", "test"):
                        rows.append({
                            "condition": condition, "sign": sign, "stage": stage,
                            "evaluation_split": split, "model_status": f"not_fit:{exc}",
                        })
                    continue
                for split in ("validation", "test"):
                    evaluate = data[(data.condition == condition) & (data.sign == sign) & (data.split == split)]
                    if evaluate.selected.nunique() < 2:
                        rows.append({
                            "condition": condition, "sign": sign, "stage": stage,
                            "evaluation_split": split, "model_status": "not_evaluated_no_contrast",
                        })
                        continue
                    score = model.predict_proba(evaluate[["q8"] + features])[:, 1]
                    concordance, n_strata = within_stratum_concordance(evaluate, score)
                    rows.append({
                        "condition": condition, "sign": sign, "stage": stage,
                        "evaluation_split": split, "model_status": "ok",
                        "n_train_segments": len(train), "n_segments": len(evaluate),
                        "n_selected_segments": int(evaluate.selected.sum()),
                        "n_proteins": int(evaluate.protein.nunique()),
                        "weighted_auroc": float(roc_auc_score(
                            evaluate.selected, score, sample_weight=evaluate.sample_weight)),
                        "weighted_average_precision": float(average_precision_score(
                            evaluate.selected, score, sample_weight=evaluate.sample_weight)),
                        "macro_within_protein_q8_concordance": concordance,
                        "n_matched_protein_q8_strata": n_strata,
                    })
    result = pd.DataFrame(rows)
    if not result.empty and "weighted_auroc" in result and "macro_within_protein_q8_concordance" in result:
        ok = result.model_status == "ok"
        result.loc[ok, "delta_auroc_from_previous_stage"] = result[ok].groupby(
            ["condition", "sign", "evaluation_split"], sort=False
        ).weighted_auroc.diff()
        result.loc[ok, "delta_concordance_from_previous_stage"] = result[ok].groupby(
            ["condition", "sign", "evaluation_split"], sort=False
        ).macro_within_protein_q8_concordance.diff()
    return result


def mechanism_associations(external: pd.DataFrame, mechanism_path: Optional[str], args):
    if not mechanism_path:
        return pd.DataFrame(), pd.DataFrame()
    mechanism = pd.read_csv(mechanism_path)
    feature_names = sum(EXTERNAL_FEATURE_GROUPS.values(), [])
    candidate_groups = {
        key: group for key, group in external.groupby(["condition", "split", "protein"], sort=False)
    }
    band_rows, effect_rows = [], []
    for band in mechanism.itertuples(index=False):
        key = (str(band.condition), str(band.split), str(band.protein))
        group = candidate_groups.get(key)
        if group is None:
            continue
        apex = int(band.apex_index_0based)
        segment = group[(group.start_index_0based <= apex) & (group.end_index_0based_exclusive > apex)]
        if len(segment) != 1:
            continue
        segment = segment.iloc[0]
        controls = group[(group.q8 == segment.q8) & group.clean_control]
        output = {
            "band_id": band.band_id, "condition": band.condition, "split": band.split,
            "protein": band.protein, "sign": int(band.sign), "q8": segment.q8,
            "mechanism_class": band.mechanism_class, "apex_index_0based": apex,
            "segment_id": segment.segment_id, "n_same_protein_q8_control_segments": len(controls),
        }
        for feature in feature_names:
            output[feature] = segment[feature]
        band_rows.append(output)
        if controls.empty:
            continue
        for feature in feature_names:
            control_values = pd.to_numeric(controls[feature], errors="coerce")
            selected_value = pd.to_numeric(pd.Series([segment[feature]]), errors="coerce").iloc[0]
            if np.isfinite(selected_value) and control_values.notna().any():
                effect_rows.append({
                    "condition": band.condition, "split": band.split, "protein": band.protein,
                    "sign": int(band.sign), "mechanism_class": band.mechanism_class,
                    "q8": segment.q8, "feature": feature,
                    "feature_group": next(name for name, values in EXTERNAL_FEATURE_GROUPS.items() if feature in values),
                    "n_selected_segments": 1, "n_control_segments": int(control_values.notna().sum()),
                    "selected_mean": float(selected_value), "control_mean": float(control_values.mean()),
                    "selected_minus_control": float(selected_value - control_values.mean()),
                })
    effects = pd.DataFrame(effect_rows)
    if not effects.empty:
        effects = effects.groupby(
            ["condition", "split", "protein", "sign", "mechanism_class", "q8", "feature", "feature_group"],
            as_index=False,
        ).agg({
            "n_selected_segments": "sum", "n_control_segments": "sum",
            "selected_mean": "mean", "control_mean": "mean", "selected_minus_control": "mean",
        })
        summary = summarize_effect_rows(effects, args.minimum_inference_proteins, ["mechanism_class"])
    else:
        summary = pd.DataFrame()
    return pd.DataFrame(band_rows), summary


def coverage_summary(external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sign, selected_column, opposite in (
        (1, "selected_positive", "selected_negative"),
        (-1, "selected_negative", "selected_positive"),
    ):
        subset = external[(external[selected_column]) & ~(external[opposite])].copy()
        subset["sign"] = sign
        for key, group in subset.groupby(["condition", "split", "sign", "q8"], dropna=False):
            rows.append({
                **dict(zip(["condition", "split", "sign", "q8"], key)),
                "n_selected_segments": len(group), "n_proteins": group.protein.nunique(),
                "structure_available_fraction": float(group.structure_available.mean()),
                "structure_eligible_fraction": float(group.segment_structure_eligible.mean()),
                "fully_resolved_fraction": float((group.segment_resolved_fraction == 1).mean()),
                "strain_available_fraction": float((group.strain_status == "ok").mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates_csv)
    if args.conditions:
        candidates = candidates[candidates.condition.isin(args.conditions)].copy()
    required = {
        "condition", "split", "protein", "q8", "segment_id",
        "start_index_0based", "end_index_0based_exclusive",
        "selected_positive", "selected_negative", "clean_control", *BASE_FEATURES,
        *GEOMETRY_FEATURES[:5],
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"Candidate table lacks {sorted(missing)}")
    protein_splits = candidates[["protein", "split"]].drop_duplicates()
    if protein_splits.protein.duplicated().any():
        raise ValueError("A protein occurs in multiple splits")
    protein_split = dict(zip(protein_splits.protein.astype(str), protein_splits.split.astype(str)))
    contacts = load_contacts(args.contact_json)
    contact_audit = audit_contacts(contacts, protein_split, args)
    ecod = pd.read_csv(args.ecod_csv) if args.ecod_csv else None
    external = aggregate_segments(candidates, contacts, ecod, args)
    signed = pd.concat([analysis_rows(external, -1), analysis_rows(external, 1)], ignore_index=True)
    per_protein, effect_summary = matched_effects(signed, args)
    performance = sequential_models(signed, args)
    mechanism_bands, mechanism_summary = mechanism_associations(external, args.mechanism_csv, args)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    contact_audit.to_csv(output / "structure_mapping_audit.csv", index=False)
    external.to_csv(output / "external_features_by_q8_segment.csv.gz", index=False, compression="gzip")
    per_protein.to_csv(output / "matched_external_effects_by_protein.csv.gz", index=False, compression="gzip")
    effect_summary.to_csv(output / "matched_external_effect_summary.csv", index=False)
    performance.to_csv(output / "sequential_external_model_performance.csv", index=False)
    mechanism_summary.to_csv(output / "mechanism_class_external_associations.csv", index=False)
    mechanism_bands.to_csv(output / "mechanism_band_external_features.csv.gz", index=False, compression="gzip")
    coverage_summary(external).to_csv(output / "external_feature_coverage_summary.csv", index=False)
    parameters = {
        "phase": "3B",
        "candidate_object": "complete contiguous Q8 segment inherited from Phase 3A",
        "case": "segment contains one sign class of band apex",
        "control": "same-protein identical-Q8 clean control inherited from Phase 3A",
        "contact_json": [str(Path(path).expanduser().resolve()) for path in args.contact_json],
        "ecod_csv": str(Path(args.ecod_csv).expanduser().resolve()) if args.ecod_csv else None,
        "strain_root": str(Path(args.strain_root).expanduser().resolve()) if args.strain_root else None,
        "strain_policy": "test-only matched descriptive/confirmatory effects; excluded from trained models",
        "mechanism_csv": str(Path(args.mechanism_csv).expanduser().resolve()) if args.mechanism_csv else None,
        "feature_groups": EXTERNAL_FEATURE_GROUPS, "model_stages": STAGES,
        "model_training": "train only", "model_evaluation": ["validation", "test"],
        "structural_model_cohort": "alignment-accepted segments meeting resolved-fraction threshold",
        "terminal_exclusion": args.terminal_exclusion,
        "min_mapping_identity": args.min_mapping_identity,
        "min_input_coverage": args.min_input_coverage,
        "min_segment_resolved_fraction": args.min_segment_resolved_fraction,
        "min_contact_sequence_separation": args.min_contact_sequence_separation,
        "domain_boundary_window": args.domain_boundary_window,
        "betweenness_samples": args.betweenness_samples,
        "minimum_inference_proteins": args.minimum_inference_proteins,
        "random_seed": args.random_seed,
        "unavailable_feature_groups": [
            "hinge_axis proximity", "curated active sites/interfaces/known hinges", "ligand proximity",
        ],
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "candidate_rows": len(candidates), "contact_records": len(contacts),
        "accepted_contact_records": int(contact_audit.mapping_accepted.sum()),
        "external_rows": len(external), "matched_effect_rows": len(per_protein),
        "model_performance_rows": len(performance),
        "mechanism_band_rows": len(mechanism_bands), "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
