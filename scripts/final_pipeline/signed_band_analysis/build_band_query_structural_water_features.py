#!/usr/bin/env python3
"""Build Phase 4 band-query structure and crystallographic-water features.

The output is a partitioned NPZ feature store keyed by
condition/split/protein.  Every numeric feature has shape
``(n_source_bands, protein_length)`` and therefore aligns directly with the
compact receiver profiles produced by
``signed_band_analysis.analyze_signed_band_query_receivers``.

Hydrogen positions and protonation states are usually absent from PDB files.
Accordingly, N/O/S proximity features emitted here are deliberately named
``polar_contact`` or ``water_compatible`` rather than hydrogen bonds.
Water paths contain only retained crystallographic water oxygens between the
query and source-band protein residues; they never traverse an intervening
protein residue.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# NetworkX 2.3 still refers to aliases removed in NumPy 1.24. Keep the same
# compatibility shim used by the established Phase 3B structural analysis.
for _alias, _type in (("int", int), ("float", float), ("bool", bool)):
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)
import networkx as nx  # noqa: E402


WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "DOD"})
POLAR_ELEMENTS = frozenset({"N", "O", "S"})
RANGE_PATTERN = re.compile(r"([^:,]+):(-?\d+)[A-Za-z]?\s*-\s*(-?\d+)[A-Za-z]?")

PAIR_FEATURES = (
    "query_resolved",
    "pair_structure_eligible",
    "minimum_ca_distance_angstrom",
    "direct_ca_contact",
    "query_to_band_contact_count",
    "query_to_band_contact_fraction",
    "ordinary_contact_path_exists",
    "shortest_ordinary_contact_path_edges",
    "nonlocal_contact_path_exists",
    "shortest_nonlocal_contact_path_edges",
    "same_community_as_band_apex",
    "same_community_as_band_majority",
    "query_distance_to_community_boundary_edges",
    "same_ecod_domain_as_band_apex",
    "shares_any_ecod_domain_with_band",
    "query_distance_to_ecod_boundary_sequence",
    "query_distance_to_band_domain_boundary_sequence",
    "relative_backbone_orientation_cosine",
    "absolute_backbone_orientation_cosine",
    "minimum_heavy_atom_distance_angstrom",
    "minimum_polar_atom_distance_angstrom",
    "direct_putative_polar_contact",
    "direct_putative_polar_contact_count",
    "water_structure_eligible",
    "query_water_contact_count",
    "band_water_contact_count",
    "shared_water_count",
    "shared_water_fraction_of_query_contacts",
    "shared_water_fraction_of_band_contacts",
    "one_water_bridge",
    "water_path_exists",
    "shortest_water_path_water_count",
    "water_path_at_most_1",
    "water_path_at_most_2",
    "water_path_at_most_3",
    "water_path_without_direct_ca_contact",
    "water_path_without_direct_polar_contact",
    "reachable_query_water_entry_count",
    "mean_query_contact_water_degree",
)

BAND_FEATURES = (
    "band_resolved_fraction",
    "band_resolved_residue_count",
    "band_apex_resolved",
    "band_community_majority_defined",
    "band_ecod_annotation_fraction",
)


@dataclass(frozen=True)
class Atom:
    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resid: int
    icode: str
    coordinate: tuple[float, float, float]
    occupancy: float
    bfactor: float
    element: str

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return self.chain, self.resid, self.icode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands_csv", required=True)
    parser.add_argument(
        "--contact_json",
        nargs="+",
        required=True,
        help="Alignment-audited Phase 3B contact JSON files for all splits",
    )
    parser.add_argument("--ecod_csv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--min_mapping_identity", type=float, default=0.90)
    parser.add_argument("--min_input_coverage", type=float, default=0.80)
    parser.add_argument("--min_band_resolved_fraction", type=float, default=0.80)
    parser.add_argument("--ca_contact_cutoff", type=float, default=8.0)
    parser.add_argument("--nonlocal_min_sequence_separation", type=int, default=3)
    parser.add_argument("--polar_contact_cutoff", type=float, default=3.5)
    parser.add_argument("--protein_water_cutoff", type=float, default=3.4)
    parser.add_argument("--water_water_cutoff", type=float, default=3.2)
    parser.add_argument("--minimum_water_occupancy", type=float, default=0.50)
    parser.add_argument("--maximum_water_bfactor_robust_z", type=float, default=3.0)
    parser.add_argument("--maximum_water_resolution", type=float, default=2.5)
    parser.add_argument(
        "--water_chain_policy",
        choices=("target_only_unambiguous", "target_contact"),
        default="target_only_unambiguous",
        help=(
            "The primary policy excludes waters also contacting a different "
            "crystallographic chain. Water chain identifiers are never used "
            "to assign waters to the target chain."
        ),
    )
    parser.add_argument(
        "--required_water_method_regex",
        default=r"X-RAY DIFFRACTION",
        help="Case-insensitive regex required for primary water eligibility",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_proteins", type=int, default=0)
    args = parser.parse_args()
    for name in (
        "min_mapping_identity",
        "min_input_coverage",
        "min_band_resolved_fraction",
        "minimum_water_occupancy",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must lie in [0, 1]")
    for name in (
        "ca_contact_cutoff",
        "polar_contact_cutoff",
        "protein_water_cutoff",
        "water_water_cutoff",
        "maximum_water_resolution",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.nonlocal_min_sequence_separation < 1:
        parser.error("--nonlocal_min_sequence_separation must be positive")
    return args


def read_json(path: str) -> object:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_contact_records(paths: Iterable[str]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for source in paths:
        payload = read_json(source)
        if not isinstance(payload, list):
            raise ValueError(f"{source}: expected a JSON list")
        for record in payload:
            protein = str(record.get("name", ""))
            if not protein:
                raise ValueError(f"{source}: contact record lacks name")
            if protein in records:
                raise ValueError(f"Duplicate contact record for {protein}")
            record["contact_source_json"] = str(Path(source).expanduser().resolve())
            records[protein] = record
    return records


def mapping_accepted(record: dict, args: argparse.Namespace) -> bool:
    alignment = record.get("alignment") or {}
    return bool(
        record.get("status") == "ok"
        and record.get("contact_map_coordinate_system") == "input_sequence_0based"
        and float(alignment.get("identity", -1)) >= args.min_mapping_identity
        and float(alignment.get("input_coverage", -1)) >= args.min_input_coverage
        and len(record.get("ca_coordinates_angstrom") or []) == len(record.get("sequence", ""))
    )


def _float_field(text: str) -> float:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return math.nan


def _element(line: str, atom_name: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    stripped = atom_name.strip()
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    return stripped[:1].upper()


def parse_pdb_first_model(path: Path) -> list[Atom]:
    """Parse one coordinate model and choose one occupancy-ranked altloc."""
    choices: dict[tuple, tuple[tuple[float, float, int], Atom]] = {}
    saw_model = False
    in_first_model = True
    order = 0
    with path.open("rt", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                continue
            if record == "ENDMDL" and saw_model:
                break
            if not in_first_model or record not in {"ATOM", "HETATM"}:
                continue
            try:
                atom = Atom(
                    record=record,
                    serial=int(line[6:11]),
                    name=line[12:16].strip().upper(),
                    altloc=line[16:17].strip().upper(),
                    resname=line[17:20].strip().upper(),
                    chain=line[21:22].strip(),
                    resid=int(line[22:26]),
                    icode=line[26:27].strip().upper(),
                    coordinate=(
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                    occupancy=_float_field(line[54:60]),
                    bfactor=_float_field(line[60:66]),
                    element=_element(line, line[12:16]),
                )
            except (TypeError, ValueError):
                continue
            key = (
                atom.record,
                atom.chain,
                atom.resid,
                atom.icode,
                atom.resname,
                atom.name,
            )
            preferred = 1.0 if atom.altloc in {"", "A", "1"} else 0.0
            occupancy = atom.occupancy if np.isfinite(atom.occupancy) else -1.0
            rank = (preferred, occupancy, -order)
            if key not in choices or rank > choices[key][0]:
                choices[key] = (rank, atom)
            order += 1
    return [value[1] for value in sorted(choices.values(), key=lambda item: item[1].serial)]


def contact_edges(record: dict) -> list[tuple[int, int, float]]:
    if "contact_edges" in record:
        return [
            (int(i), int(j), float(distance))
            for i, j, distance in record["contact_edges"]
        ]
    if "contact_map" not in record:
        raise ValueError("Contact record contains neither edges nor a dense map")
    matrix = np.asarray(record["contact_map"], dtype=bool)
    coordinates = record.get("ca_coordinates_angstrom")
    ii, jj = np.where(np.triu(matrix, k=1))
    edges = []
    for i, j in zip(ii.tolist(), jj.tolist()):
        distance = math.nan
        if coordinates and coordinates[i] is not None and coordinates[j] is not None:
            distance = float(
                np.linalg.norm(np.asarray(coordinates[i]) - np.asarray(coordinates[j]))
            )
        edges.append((i, j, distance))
    return edges


def adjacency_lists(
    length: int, edges: Iterable[tuple[int, int, float]]
) -> list[set[int]]:
    adjacency = [set() for _ in range(length)]
    for i, j, _ in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def multi_source_bfs(adjacency: list[set[int]], sources: Iterable[int]) -> np.ndarray:
    distance = np.full(len(adjacency), np.nan, dtype=float)
    queue: deque[int] = deque()
    for source in sorted(set(int(value) for value in sources)):
        distance[source] = 0.0
        queue.append(source)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if not np.isfinite(distance[neighbor]):
                distance[neighbor] = distance[node] + 1.0
                queue.append(neighbor)
    return distance


def community_membership(
    length: int, resolved: np.ndarray, nonlocal_edges: list[tuple[int, int, float]]
) -> tuple[np.ndarray, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(np.flatnonzero(resolved).tolist())
    graph.add_edges_from((i, j) for i, j, _ in nonlocal_edges)
    if graph.number_of_edges():
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    else:
        communities = [frozenset([node]) for node in graph.nodes]
    membership = np.full(length, -1, dtype=int)
    for number, community in enumerate(communities):
        membership[np.asarray(sorted(community), dtype=int)] = number
    boundary = set()
    for i, j in graph.edges:
        if membership[i] != membership[j]:
            boundary.update((i, j))
    boundary_distance = multi_source_bfs(
        adjacency_lists(length, nonlocal_edges), boundary
    ) if boundary else np.full(length, np.nan)
    return membership, boundary_distance


def local_backbone_tangents(coordinates: np.ndarray) -> np.ndarray:
    tangent = np.full_like(coordinates, np.nan, dtype=float)
    for index in range(1, len(coordinates) - 1):
        if np.isfinite(coordinates[[index - 1, index + 1]]).all():
            vector = coordinates[index + 1] - coordinates[index - 1]
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                tangent[index] = vector / norm
    return tangent


def robust_bfactor_limit(
    protein_bfactors: np.ndarray, maximum_robust_z: float
) -> tuple[float, float, float]:
    finite = protein_bfactors[np.isfinite(protein_bfactors)]
    if not len(finite):
        return math.nan, math.nan, math.nan
    median = float(np.median(finite))
    scaled_mad = float(1.4826 * np.median(np.abs(finite - median)))
    limit = median + maximum_robust_z * scaled_mad if scaled_mad > 0 else math.inf
    return median, scaled_mad, limit


def residue_atom_tables(
    atoms: list[Atom], record: dict
) -> tuple[list[np.ndarray], list[np.ndarray], list[Atom], list[Atom], set[str]]:
    length = len(record["sequence"])
    mapping_by_key = {}
    target_chains = set()
    for mapping in record.get("residue_mapping", []):
        chain = str(mapping.get("pdb_chain_id") or mapping.get("pdb_segid") or "")
        key = (chain, int(mapping["pdb_resid"]), str(mapping.get("pdb_insertion_code") or ""))
        mapping_by_key[key] = int(mapping["input_index_0based"])
        target_chains.add(chain)

    heavy_lists: list[list[tuple[float, float, float]]] = [[] for _ in range(length)]
    polar_lists: list[list[tuple[float, float, float]]] = [[] for _ in range(length)]
    target_atoms: list[Atom] = []
    foreign_polar: list[Atom] = []
    for atom in atoms:
        if atom.resname in WATER_RESNAMES:
            continue
        model_index = mapping_by_key.get(atom.residue_key)
        is_hydrogen = atom.element in {"H", "D"}
        if model_index is not None and not is_hydrogen:
            heavy_lists[model_index].append(atom.coordinate)
            target_atoms.append(atom)
            if atom.element in POLAR_ELEMENTS:
                polar_lists[model_index].append(atom.coordinate)
        elif (
            atom.record == "ATOM"
            and atom.chain not in target_chains
            and atom.element in POLAR_ELEMENTS
        ):
            foreign_polar.append(atom)

    heavy = [
        np.asarray(values, dtype=float).reshape((-1, 3))
        if values else np.empty((0, 3), dtype=float)
        for values in heavy_lists
    ]
    polar = [
        np.asarray(values, dtype=float).reshape((-1, 3))
        if values else np.empty((0, 3), dtype=float)
        for values in polar_lists
    ]
    return heavy, polar, target_atoms, foreign_polar, target_chains


def retained_water_network(
    atoms: list[Atom],
    target_polar_by_residue: list[np.ndarray],
    target_atoms: list[Atom],
    foreign_polar: list[Atom],
    args: argparse.Namespace,
) -> dict:
    target_polar_points = []
    target_polar_residue = []
    for index, points in enumerate(target_polar_by_residue):
        target_polar_points.extend(points.tolist())
        target_polar_residue.extend([index] * len(points))
    target_points = np.asarray(target_polar_points, dtype=float).reshape((-1, 3))
    target_residue = np.asarray(target_polar_residue, dtype=int)
    target_tree = cKDTree(target_points) if len(target_points) else None

    foreign_points = np.asarray(
        [atom.coordinate for atom in foreign_polar], dtype=float
    ).reshape((-1, 3))
    foreign_tree = cKDTree(foreign_points) if len(foreign_points) else None

    protein_b = np.asarray(
        [atom.bfactor for atom in target_atoms if atom.element not in {"H", "D"}],
        dtype=float,
    )
    protein_b_median, protein_b_mad, water_b_limit = robust_bfactor_limit(
        protein_b, args.maximum_water_bfactor_robust_z
    )

    all_waters = [
        atom for atom in atoms
        if atom.resname in WATER_RESNAMES and atom.element == "O"
    ]
    occupancy_pass = [
        water for water in all_waters
        if np.isfinite(water.occupancy)
        and water.occupancy >= args.minimum_water_occupancy
    ]
    bfactor_pass = [
        water for water in occupancy_pass
        if np.isfinite(water.bfactor)
        and (not np.isfinite(water_b_limit) or water.bfactor <= water_b_limit)
    ]
    retained = []
    excluded_foreign = 0
    for water in bfactor_pass:
        point = np.asarray(water.coordinate)
        target_contact = (
            target_tree is not None
            and bool(target_tree.query_ball_point(point, args.protein_water_cutoff))
        )
        if not target_contact:
            continue
        foreign_contact = (
            foreign_tree is not None
            and bool(foreign_tree.query_ball_point(point, args.protein_water_cutoff))
        )
        if args.water_chain_policy == "target_only_unambiguous" and foreign_contact:
            excluded_foreign += 1
            continue
        retained.append(water)

    water_coordinates = np.asarray(
        [water.coordinate for water in retained], dtype=float
    ).reshape((-1, 3))
    water_tree = cKDTree(water_coordinates) if len(water_coordinates) else None
    residue_to_waters = [set() for _ in target_polar_by_residue]
    if water_tree is not None:
        for point, residue_index in zip(target_points, target_residue):
            residue_to_waters[residue_index].update(
                water_tree.query_ball_point(point, args.protein_water_cutoff)
            )
    water_adjacency = [set() for _ in retained]
    if water_tree is not None:
        for i, j in water_tree.query_pairs(args.water_water_cutoff):
            water_adjacency[i].add(j)
            water_adjacency[j].add(i)

    return {
        "residue_to_waters": residue_to_waters,
        "water_adjacency": water_adjacency,
        "all_water_count": len(all_waters),
        "occupancy_pass_count": len(occupancy_pass),
        "bfactor_pass_count": len(bfactor_pass),
        "retained_water_count": len(retained),
        "excluded_foreign_chain_contact_count": excluded_foreign,
        "protein_bfactor_median": protein_b_median,
        "protein_bfactor_scaled_mad": protein_b_mad,
        "water_bfactor_limit": water_b_limit,
    }


def ecod_labels(
    ecod: Optional[pd.DataFrame], protein: str, record: dict
) -> tuple[list[set[str]], dict[str, tuple[int, int]]]:
    length = len(record["sequence"])
    labels = [set() for _ in range(length)]
    if ecod is None:
        return labels, {}
    subset = ecod[ecod.name.astype(str).eq(protein)]
    intervals = []
    for row in subset.itertuples(index=False):
        for chain, start, end in RANGE_PATTERN.findall(str(row.pdb_range)):
            intervals.append(
                (str(row.ecod_domain_id), chain.strip(), min(int(start), int(end)),
                 max(int(start), int(end)))
            )
    for mapping in record.get("residue_mapping", []):
        index = int(mapping["input_index_0based"])
        chain = str(mapping.get("pdb_chain_id") or mapping.get("pdb_segid") or "")
        resid = int(mapping["pdb_resid"])
        labels[index].update(
            domain for domain, expected_chain, start, end in intervals
            if chain == expected_chain and start <= resid <= end
        )
    boundaries = {}
    all_domains = sorted(set().union(*labels)) if labels else []
    for domain in all_domains:
        positions = [index for index, value in enumerate(labels) if domain in value]
        if positions:
            boundaries[domain] = (min(positions), max(positions))
    return labels, boundaries


def domain_boundary_distance(
    index: int, domains: set[str], boundaries: dict[str, tuple[int, int]]
) -> float:
    values = [
        min(abs(index - start), abs(index - end))
        for domain in domains
        for start, end in [boundaries[domain]]
        if domain in boundaries
    ]
    return float(min(values)) if values else math.nan


def unique_majority(values: Iterable[int]) -> Optional[int]:
    counts = Counter(int(value) for value in values if int(value) >= 0)
    if not counts:
        return None
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return int(ordered[0][0])


def min_distance_to_points(points: np.ndarray, target_points: np.ndarray) -> float:
    if not len(points) or not len(target_points):
        return math.nan
    tree = cKDTree(target_points)
    distance, _ = tree.query(points, k=1)
    return float(np.min(distance))


def direct_polar_contact_count(
    query_points: np.ndarray,
    band_polar_by_residue: list[tuple[int, np.ndarray]],
    query_index: int,
    cutoff: float,
) -> int:
    count = 0
    for band_index, points in band_polar_by_residue:
        if band_index == query_index or not len(query_points) or not len(points):
            continue
        tree = cKDTree(points)
        distance, _ = tree.query(query_points, k=1)
        if np.any(distance <= cutoff):
            count += 1
    return count


def water_distances(
    adjacency: list[set[int]], band_waters: set[int]
) -> np.ndarray:
    return multi_source_bfs(adjacency, band_waters)


def structure_arrays_for_protein(
    protein: str,
    record: dict,
    atoms: list[Atom],
    ecod: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    length = len(record["sequence"])
    resolved = np.asarray(record["resolved_mask"], dtype=bool)
    coordinates = np.full((length, 3), np.nan, dtype=float)
    for index, point in enumerate(record["ca_coordinates_angstrom"]):
        if point is not None:
            coordinates[index] = np.asarray(point, dtype=float)
    edges = [
        edge for edge in contact_edges(record)
        if resolved[edge[0]] and resolved[edge[1]]
        and edge[2] <= args.ca_contact_cutoff + 1e-6
    ]
    ordinary_adjacency = adjacency_lists(length, edges)
    nonlocal_edges = [
        edge for edge in edges
        if abs(edge[0] - edge[1]) >= args.nonlocal_min_sequence_separation
    ]
    nonlocal_adjacency = adjacency_lists(length, nonlocal_edges)
    membership, community_boundary_distance = community_membership(
        length, resolved, nonlocal_edges
    )
    tangents = local_backbone_tangents(coordinates)
    heavy, polar, target_atoms, foreign_polar, target_chains = residue_atom_tables(
        atoms, record
    )
    water = retained_water_network(
        atoms, polar, target_atoms, foreign_polar, args
    )
    labels, domain_boundaries = ecod_labels(ecod, protein, record)
    query_domain_boundary = np.asarray(
        [
            domain_boundary_distance(index, labels[index], domain_boundaries)
            for index in range(length)
        ],
        dtype=float,
    )
    return {
        "resolved": resolved,
        "coordinates": coordinates,
        "edges": edges,
        "ordinary_adjacency": ordinary_adjacency,
        "nonlocal_adjacency": nonlocal_adjacency,
        "community_membership": membership,
        "community_boundary_distance": community_boundary_distance,
        "tangents": tangents,
        "heavy": heavy,
        "polar": polar,
        "water": water,
        "ecod_labels": labels,
        "domain_boundaries": domain_boundaries,
        "query_domain_boundary": query_domain_boundary,
    }, {
        "target_chain_ids": ";".join(sorted(target_chains)),
        "n_target_heavy_atoms": len(target_atoms),
        "n_target_polar_atoms": int(sum(len(value) for value in polar)),
        "n_foreign_chain_polar_atoms": len(foreign_polar),
        "n_ordinary_contact_edges": len(edges),
        "n_nonlocal_contact_edges": len(nonlocal_edges),
        "n_contact_communities": int(len(set(membership[membership >= 0]))),
        "n_ecod_labeled_residues": int(sum(bool(value) for value in labels)),
        "n_ecod_domains": len(domain_boundaries),
        **{
            key: value for key, value in water.items()
            if key not in {"residue_to_waters", "water_adjacency"}
        },
    }


def water_method_eligible(record: dict, args: argparse.Namespace) -> tuple[bool, str]:
    method = str(record.get("experimental_method") or "")
    resolution = record.get("resolution_angstrom")
    if not re.search(args.required_water_method_regex, method, flags=re.IGNORECASE):
        return False, "experimental_method_not_eligible"
    if resolution is None or not np.isfinite(float(resolution)):
        return False, "resolution_missing"
    if float(resolution) > args.maximum_water_resolution:
        return False, "resolution_above_threshold"
    return True, "eligible"


def build_pair_features(
    bands: pd.DataFrame,
    record: dict,
    arrays: dict,
    water_primary_eligible: bool,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    length = len(record["sequence"])
    n_bands = len(bands)
    output = {
        feature: np.full((n_bands, length), np.nan, dtype=np.float32)
        for feature in PAIR_FEATURES
    }
    band_output = {
        feature: np.full(n_bands, np.nan, dtype=np.float32)
        for feature in BAND_FEATURES
    }
    indices = np.arange(length)
    resolved = arrays["resolved"]
    membership = arrays["community_membership"]
    labels = arrays["ecod_labels"]
    water = arrays["water"]
    water_degrees = np.asarray(
        [len(neighbors) for neighbors in water["water_adjacency"]], dtype=float
    )

    for band_number, band in enumerate(bands.itertuples(index=False)):
        start = int(band.start_index_0based)
        end = int(band.end_index_0based_inclusive) + 1
        apex = int(band.apex_index_0based)
        band_indices = np.arange(start, end, dtype=int)
        resolved_band = band_indices[resolved[band_indices]]
        resolved_fraction = len(resolved_band) / max(1, len(band_indices))
        eligible_band = resolved_fraction >= args.min_band_resolved_fraction
        pair_eligible = resolved & eligible_band
        band_output["band_resolved_fraction"][band_number] = resolved_fraction
        band_output["band_resolved_residue_count"][band_number] = len(resolved_band)
        band_output["band_apex_resolved"][band_number] = float(resolved[apex])
        output["query_resolved"][band_number] = resolved.astype(np.float32)
        output["pair_structure_eligible"][band_number] = pair_eligible.astype(np.float32)

        if not eligible_band or not len(resolved_band):
            continue

        band_ca = arrays["coordinates"][resolved_band]
        ca_tree = cKDTree(band_ca)
        ca_distance, _ = ca_tree.query(arrays["coordinates"][resolved], k=1)
        output["minimum_ca_distance_angstrom"][band_number, resolved] = ca_distance

        band_set = set(resolved_band.tolist())
        contact_count = np.asarray(
            [len(arrays["ordinary_adjacency"][query] & band_set) for query in indices],
            dtype=float,
        )
        output["query_to_band_contact_count"][band_number, pair_eligible] = (
            contact_count[pair_eligible]
        )
        output["query_to_band_contact_fraction"][band_number, pair_eligible] = (
            contact_count[pair_eligible] / len(resolved_band)
        )
        output["direct_ca_contact"][band_number, pair_eligible] = (
            contact_count[pair_eligible] > 0
        )

        ordinary_distance = multi_source_bfs(
            arrays["ordinary_adjacency"], resolved_band
        )
        nonlocal_distance = multi_source_bfs(
            arrays["nonlocal_adjacency"], resolved_band
        )
        for prefix, distance in (
            ("ordinary", ordinary_distance),
            ("nonlocal", nonlocal_distance),
        ):
            exists = np.isfinite(distance) & pair_eligible
            output[f"{prefix}_contact_path_exists"][band_number, pair_eligible] = (
                exists[pair_eligible]
            )
            output[f"shortest_{prefix}_contact_path_edges"][
                band_number, exists
            ] = distance[exists]

        apex_community = membership[apex] if resolved[apex] else -1
        majority = unique_majority(membership[resolved_band])
        band_output["band_community_majority_defined"][band_number] = float(
            majority is not None
        )
        if apex_community >= 0:
            output["same_community_as_band_apex"][band_number, pair_eligible] = (
                membership[pair_eligible] == apex_community
            )
        if majority is not None:
            output["same_community_as_band_majority"][
                band_number, pair_eligible
            ] = membership[pair_eligible] == majority
        output["query_distance_to_community_boundary_edges"][
            band_number, pair_eligible
        ] = arrays["community_boundary_distance"][pair_eligible]

        band_domains = set().union(*(labels[index] for index in resolved_band))
        apex_domains = labels[apex] if resolved[apex] else set()
        annotated_band = [index for index in resolved_band if labels[index]]
        band_output["band_ecod_annotation_fraction"][band_number] = (
            len(annotated_band) / len(resolved_band)
        )
        band_boundary_positions = [
            boundary
            for domain in band_domains
            for boundary in arrays["domain_boundaries"].get(domain, ())
        ]
        for query in np.flatnonzero(pair_eligible):
            query_domains = labels[query]
            if query_domains and apex_domains:
                output["same_ecod_domain_as_band_apex"][band_number, query] = float(
                    bool(query_domains & apex_domains)
                )
            if query_domains and band_domains:
                output["shares_any_ecod_domain_with_band"][band_number, query] = float(
                    bool(query_domains & band_domains)
                )
            output["query_distance_to_ecod_boundary_sequence"][
                band_number, query
            ] = arrays["query_domain_boundary"][query]
            if query_domains and band_boundary_positions:
                output["query_distance_to_band_domain_boundary_sequence"][
                    band_number, query
                ] = min(abs(query - value) for value in band_boundary_positions)

        if np.isfinite(arrays["tangents"][apex]).all():
            cosine = arrays["tangents"] @ arrays["tangents"][apex]
            orientation_valid = (
                pair_eligible & np.isfinite(arrays["tangents"]).all(axis=1)
            )
            output["relative_backbone_orientation_cosine"][
                band_number, orientation_valid
            ] = cosine[orientation_valid]
            output["absolute_backbone_orientation_cosine"][
                band_number, orientation_valid
            ] = np.abs(cosine[orientation_valid])

        band_heavy = np.concatenate(
            [arrays["heavy"][index] for index in resolved_band if len(arrays["heavy"][index])],
            axis=0,
        ) if any(len(arrays["heavy"][index]) for index in resolved_band) else np.empty((0, 3))
        band_polar_by_residue = [
            (index, arrays["polar"][index])
            for index in resolved_band if len(arrays["polar"][index])
        ]
        band_polar = np.concatenate(
            [value for _, value in band_polar_by_residue], axis=0
        ) if band_polar_by_residue else np.empty((0, 3))
        for query in np.flatnonzero(pair_eligible):
            output["minimum_heavy_atom_distance_angstrom"][
                band_number, query
            ] = min_distance_to_points(arrays["heavy"][query], band_heavy)
            output["minimum_polar_atom_distance_angstrom"][
                band_number, query
            ] = min_distance_to_points(arrays["polar"][query], band_polar)
            polar_count = direct_polar_contact_count(
                arrays["polar"][query],
                band_polar_by_residue,
                query,
                args.polar_contact_cutoff,
            )
            output["direct_putative_polar_contact_count"][
                band_number, query
            ] = polar_count
            output["direct_putative_polar_contact"][
                band_number, query
            ] = float(polar_count > 0)

        water_pair_eligible = pair_eligible & water_primary_eligible
        output["water_structure_eligible"][band_number, pair_eligible] = float(
            water_primary_eligible
        )
        if not water_primary_eligible:
            continue
        band_waters = set().union(
            *(water["residue_to_waters"][index] for index in resolved_band)
        )
        water_distance = water_distances(
            water["water_adjacency"], band_waters
        ) if band_waters else np.full(len(water["water_adjacency"]), np.nan)
        for query in np.flatnonzero(water_pair_eligible):
            query_waters = water["residue_to_waters"][query]
            # A water touching the query itself is not a bridge back to that
            # same residue when the query lies inside the source band.
            effective_band_waters = band_waters
            effective_water_distance = water_distance
            if query in band_set:
                effective_band_waters = set().union(
                    *(
                        water["residue_to_waters"][index]
                        for index in resolved_band
                        if index != query
                    )
                ) if len(resolved_band) > 1 else set()
                effective_water_distance = (
                    water_distances(
                        water["water_adjacency"], effective_band_waters
                    )
                    if effective_band_waters
                    else np.full(len(water["water_adjacency"]), np.nan)
                )
            shared = query_waters & effective_band_waters
            output["query_water_contact_count"][band_number, query] = len(query_waters)
            output["band_water_contact_count"][
                band_number, query
            ] = len(effective_band_waters)
            output["shared_water_count"][band_number, query] = len(shared)
            output["shared_water_fraction_of_query_contacts"][
                band_number, query
            ] = len(shared) / max(1, len(query_waters))
            output["shared_water_fraction_of_band_contacts"][
                band_number, query
            ] = len(shared) / max(1, len(effective_band_waters))
            output["one_water_bridge"][band_number, query] = float(bool(shared))
            reachable_entries = [
                water_index for water_index in query_waters
                if np.isfinite(effective_water_distance[water_index])
            ]
            output["reachable_query_water_entry_count"][
                band_number, query
            ] = len(reachable_entries)
            if query_waters:
                output["mean_query_contact_water_degree"][
                    band_number, query
                ] = float(np.mean(water_degrees[list(query_waters)]))
            if reachable_entries:
                water_count = int(
                    min(
                        effective_water_distance[index]
                        for index in reachable_entries
                    )
                    + 1
                )
                output["water_path_exists"][band_number, query] = 1.0
                output["shortest_water_path_water_count"][
                    band_number, query
                ] = water_count
            else:
                water_count = None
                output["water_path_exists"][band_number, query] = 0.0
            for maximum in (1, 2, 3):
                output[f"water_path_at_most_{maximum}"][
                    band_number, query
                ] = float(water_count is not None and water_count <= maximum)
            direct_ca = output["direct_ca_contact"][band_number, query] == 1
            direct_polar = (
                output["direct_putative_polar_contact"][band_number, query] == 1
            )
            output["water_path_without_direct_ca_contact"][
                band_number, query
            ] = float(water_count is not None and not direct_ca)
            output["water_path_without_direct_polar_contact"][
                band_number, query
            ] = float(water_count is not None and not direct_polar)
    return {**output, **band_output}


def output_path(root: Path, condition: str, split: str, protein: str) -> Path:
    return root / "pair_features" / condition / split / f"{protein}.npz"


def save_feature_store(
    path: Path,
    condition: str,
    split: str,
    protein: str,
    bands: pd.DataFrame,
    length: int,
    features: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(["esmfluc.phase4_structural_water.v1"]),
        condition=np.asarray([condition]),
        split=np.asarray([split]),
        protein=np.asarray([protein]),
        protein_length=np.asarray([length], dtype=np.int32),
        band_id=np.asarray(bands.band_id.astype(str).tolist(), dtype=str),
        sign=bands.sign.to_numpy(np.int8),
        apex_index_0based=bands.apex_index_0based.to_numpy(np.int32),
        start_index_0based=bands.start_index_0based.to_numpy(np.int32),
        end_index_0based_inclusive=bands.end_index_0based_inclusive.to_numpy(np.int32),
        pair_feature_names=np.asarray(PAIR_FEATURES, dtype=str),
        band_feature_names=np.asarray(BAND_FEATURES, dtype=str),
        **features,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bands = pd.read_csv(args.bands_csv)
    required_bands = {
        "condition", "split", "protein", "band_id", "sign",
        "apex_index_0based", "start_index_0based", "end_index_0based_inclusive",
    }
    missing = required_bands - set(bands)
    if missing:
        raise ValueError(f"Band table lacks {sorted(missing)}")
    if args.conditions:
        bands = bands[bands.condition.isin(args.conditions)]
    if args.splits:
        bands = bands[bands.split.isin(args.splits)]
    bands = bands.sort_values(["condition", "split", "protein", "band_id"])
    contacts = load_contact_records(args.contact_json)
    ecod = pd.read_csv(args.ecod_csv) if args.ecod_csv else None
    if ecod is not None and {"name", "ecod_domain_id", "pdb_range"} - set(ecod):
        raise ValueError("ECOD table requires name, ecod_domain_id, and pdb_range")

    grouped = {
        key: group.reset_index(drop=True)
        for key, group in bands.groupby(["condition", "split", "protein"], sort=False)
    }
    protein_contexts: dict[str, list[tuple[str, str, pd.DataFrame]]] = {}
    for (condition, split, protein), group in grouped.items():
        protein_contexts.setdefault(str(protein), []).append(
            (str(condition), str(split), group)
        )

    audit_rows = []
    manifest_rows = []
    processed = 0
    for protein in sorted(protein_contexts):
        if args.max_proteins and processed >= args.max_proteins:
            break
        processed += 1
        record = contacts.get(protein)
        base_audit = {
            "protein": protein,
            "contact_record_present": record is not None,
            "mapping_accepted": False,
            "status": "missing_contact_record",
            "error": None,
        }
        if record is None:
            audit_rows.append(base_audit)
            continue
        base_audit.update({
            "status": record.get("status"),
            "mapping_status": record.get("mapping_status"),
            "alignment_identity": (record.get("alignment") or {}).get("identity"),
            "input_coverage": (record.get("alignment") or {}).get("input_coverage"),
            "experimental_method": record.get("experimental_method"),
            "resolution_angstrom": record.get("resolution_angstrom"),
            "pdb_file": record.get("pdb_file"),
            "contact_source_json": record.get("contact_source_json"),
        })
        accepted = mapping_accepted(record, args)
        base_audit["mapping_accepted"] = accepted
        if not accepted:
            base_audit["status"] = "mapping_rejected"
            audit_rows.append(base_audit)
            continue
        try:
            pdb_path = Path(str(record["pdb_file"])).expanduser().resolve()
            atoms = parse_pdb_first_model(pdb_path)
            arrays, structure_audit = structure_arrays_for_protein(
                protein, record, atoms, ecod, args
            )
            water_eligible, water_reason = water_method_eligible(record, args)
            base_audit.update({
                "status": "ok",
                "pdb_atom_count_first_model": len(atoms),
                "water_primary_eligible": water_eligible,
                "water_eligibility_reason": water_reason,
                **structure_audit,
            })
            for condition, split, protein_bands in protein_contexts[protein]:
                destination = output_path(output, condition, split, protein)
                if destination.exists() and not args.overwrite:
                    status = "reused"
                else:
                    features = build_pair_features(
                        protein_bands, record, arrays, water_eligible, args
                    )
                    save_feature_store(
                        destination, condition, split, protein, protein_bands,
                        len(record["sequence"]), features
                    )
                    status = "written"
                manifest_rows.append({
                    "condition": condition,
                    "split": split,
                    "protein": protein,
                    "feature_npz": str(destination),
                    "status": status,
                    "n_bands": len(protein_bands),
                    "protein_length": len(record["sequence"]),
                    "n_band_query_pairs": len(protein_bands) * len(record["sequence"]),
                    "mapping_accepted": True,
                    "water_primary_eligible": water_eligible,
                })
        except Exception as exc:
            base_audit["status"] = "feature_build_error"
            base_audit["error"] = f"{type(exc).__name__}: {exc}"
        audit_rows.append(base_audit)

    pd.DataFrame(audit_rows).to_csv(output / "structure_water_input_audit.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output / "pair_feature_manifest.csv", index=False)
    parameters = {
        "schema_version": "esmfluc.phase4_structural_water.v1",
        "coordinate_system": "model/input sequence, 0-based",
        "contact_graph": {
            "ca_cutoff_angstrom": args.ca_contact_cutoff,
            "ordinary_path_includes_backbone_neighbors": True,
            "nonlocal_min_sequence_separation": args.nonlocal_min_sequence_separation,
            "community_graph": "nonlocal C-alpha contact graph; greedy modularity",
        },
        "quality": {
            "min_mapping_identity": args.min_mapping_identity,
            "min_input_coverage": args.min_input_coverage,
            "min_band_resolved_fraction": args.min_band_resolved_fraction,
            "unresolved_encoding": "NaN, never zero/no-contact",
        },
        "polar_geometry": {
            "atom_elements": sorted(POLAR_ELEMENTS),
            "cutoff_angstrom": args.polar_contact_cutoff,
            "interpretation": "putative polar heavy-atom contact, not a hydrogen bond",
        },
        "water_primary_definition": {
            "water_resnames": sorted(WATER_RESNAMES),
            "protein_water_cutoff_angstrom": args.protein_water_cutoff,
            "water_water_cutoff_angstrom": args.water_water_cutoff,
            "minimum_occupancy": args.minimum_water_occupancy,
            "maximum_bfactor_robust_z": args.maximum_water_bfactor_robust_z,
            "required_method_regex": args.required_water_method_regex,
            "maximum_resolution_angstrom": args.maximum_water_resolution,
            "chain_policy": args.water_chain_policy,
            "path_definition": (
                "query residue--water(s)--source-band residue; no intervening "
                "protein nodes"
            ),
            "interpretation": (
                "water-compatible geometric paths in one static coordinate "
                "model, not solution hydrogen-bond persistence"
            ),
        },
        "pair_features": list(PAIR_FEATURES),
        "band_features": list(BAND_FEATURES),
        "inputs": {
            "bands_csv": str(Path(args.bands_csv).expanduser().resolve()),
            "contact_json": [
                str(Path(value).expanduser().resolve()) for value in args.contact_json
            ],
            "ecod_csv": (
                str(Path(args.ecod_csv).expanduser().resolve())
                if args.ecod_csv else None
            ),
        },
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps({
        "proteins_considered": processed,
        "feature_partitions": len(manifest_rows),
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
