"""Experimental-coordinate audits, geometry and contact graph features."""
from __future__ import annotations
import gzip
import json
import zlib
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import networkx as nx

def read_json(path: str):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)

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
