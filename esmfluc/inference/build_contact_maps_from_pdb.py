#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build sequence-indexed contact networks from experimental PDB structures.

The input CSV must contain ``name`` and ``sequence`` columns. ``name`` may be a
PDB-chain identifier such as ``2pbk_B`` or a CATH-style identifier such as
``1a39A00``.

Observed C-alpha residues are aligned explicitly to the input/model sequence.
Consequently, every emitted input-sequence index has an auditable PDB residue
mapping (or is marked unresolved); PDB residue numbers are never treated as
sequence indices. This matters for structures with unresolved residues,
insertion codes, engineered substitutions, or construct/domain boundaries.

The default dense ``contact_map`` is indexed by the input sequence and uses 0
for both noncontacts and unresolved pairs. Always use ``resolved_mask`` when
analyzing it. For network analysis, ``--map_representation edges`` is smaller
and retains C-alpha distances as ``[i, j, distance_angstrom]`` rows.
"""

import argparse
import gzip
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "MDAnalysis is required. Install with: pip install MDAnalysis"
    ) from exc

try:
    from Bio.Align import PairwiseAligner
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Biopython is required for sequence-to-structure mapping. "
        "Install with: pip install biopython"
    ) from exc


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common coordinate-file modifications with an unambiguous parent residue.
    "MSE": "M", "SEC": "U", "PYL": "O", "HYP": "P", "MLY": "K",
    "M3L": "K", "KCX": "K", "SEP": "S", "TPO": "T", "PTR": "Y",
    "CSO": "C", "CSD": "C", "CME": "C", "FME": "M",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build alignment-audited contact maps/networks from PDB files"
    )
    p.add_argument("--input_csv", required=True, help="CSV with columns name, sequence")
    p.add_argument("--output_json", required=True, help="Output .json or .json.gz path")
    p.add_argument(
        "--audit_csv",
        default=None,
        help="Mapping audit CSV (default: <output>.mapping_audit.csv)",
    )
    p.add_argument(
        "--pdb_dir", default="./data/pdb_cache",
        help="Directory containing/caching <pdb_id>.pdb files",
    )
    p.add_argument("--cutoff", type=float, default=8.0, help="C-alpha contact cutoff in Angstrom")
    p.add_argument(
        "--map_representation", choices=["dense", "edges", "both"], default="dense",
        help="Dense input-indexed map, sparse distance-bearing edges, or both",
    )
    p.add_argument(
        "--include_ca_coordinates", action="store_true",
        help="Include input-indexed C-alpha coordinates (null for unresolved residues)",
    )
    p.add_argument(
        "--min_mapping_identity", type=float, default=0.90,
        help="Identity threshold used to label a mapping high quality (default 0.90)",
    )
    p.add_argument(
        "--min_input_coverage", type=float, default=0.80,
        help="Resolved input-sequence coverage threshold (default 0.80)",
    )
    p.add_argument("--max_proteins", type=int, default=0, help="Optional cap for a smoke test")
    p.add_argument("--overwrite_pdb", action="store_true", help="Redownload cached PDB files")
    args = p.parse_args()
    if args.cutoff <= 0:
        p.error("--cutoff must be positive")
    for name in ("min_mapping_identity", "min_input_coverage"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            p.error(f"--{name} must lie in [0, 1]")
    return args


def default_audit_path(output_json: str) -> str:
    path = Path(output_json)
    name = path.name
    if name.endswith(".json.gz"):
        name = name[:-8]
    elif name.endswith(".json"):
        name = name[:-5]
    return str(path.with_name(f"{name}.mapping_audit.csv"))


def make_session() -> requests.Session:
    session = requests.Session()
    retry_kwargs = {
        "total": 4,
        "backoff_factor": 0.6,
        "status_forcelist": [429, 500, 502, 503, 504],
        "raise_on_status": False,
    }
    try:
        retry = Retry(allowed_methods=["GET"], **retry_kwargs)
    except TypeError:  # urllib3 < 1.26
        retry = Retry(method_whitelist=["GET"], **retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "ESMfluc-contactmap-builder/2.0"})
    return session


def parse_seq_id(seq_id: str) -> Tuple[str, str]:
    if "_" in seq_id:
        pdb_id, chain = seq_id.split("_", 1)
    elif len(seq_id) >= 5:
        # CATH IDs commonly look like 1a39A00: PDB 1a39, chain A, domain 00.
        pdb_id, chain = seq_id[:4], seq_id[4]
    else:
        raise ValueError(
            f"Invalid sequence ID {seq_id!r}; expected PDBID_CHAIN or PDBIDCHAIN..."
        )
    if not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id) or not chain:
        raise ValueError(f"Could not parse a four-character PDB ID and chain from {seq_id!r}")
    return pdb_id.lower(), chain


def download_pdb(
    session: requests.Session, pdb_id: str, out_dir: str, overwrite: bool = False
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path) and not overwrite:
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = session.get(url, timeout=40)
    response.raise_for_status()
    if not response.content.strip():
        raise RuntimeError(f"RCSB returned an empty file for {pdb_id}")
    tmp_path = f"{out_path}.part"
    with open(tmp_path, "wb") as handle:
        handle.write(response.content)
    os.replace(tmp_path, out_path)
    return out_path


def parse_pdb_metadata(pdb_file: str) -> Dict[str, object]:
    """Read the small set of experimental metadata needed for mapping audits."""
    method_parts: List[str] = []
    title_parts: List[str] = []
    resolution: Optional[float] = None
    with open(pdb_file, "rt", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "EXPDTA":
                method_parts.append(line[10:].strip())
            elif record == "TITLE":
                title_parts.append(line[10:].strip())
            elif line.startswith("REMARK   2 RESOLUTION."):
                match = re.search(r"RESOLUTION\.\s+([0-9]+(?:\.[0-9]+)?)\s+ANGSTROMS", line)
                if match:
                    resolution = float(match.group(1))
            elif record in {"ATOM", "HETATM"}:
                # Header records precede coordinates in standard PDB files.
                break
    return {
        "experimental_method": " ".join(method_parts) or None,
        "resolution_angstrom": resolution,
        "structure_title": " ".join(title_parts) or None,
    }


def select_chain_ca(universe: "mda.Universe", chain: str):
    selectors = [
        f"protein and name CA and chainID {chain}",
        f"protein and name CA and segid {chain}",
    ]
    for variant in sorted({chain, chain.upper(), chain.lower()}):
        selectors.extend([
            f"protein and name CA and chainID {variant}",
            f"protein and name CA and segid {variant}",
        ])
    for selector in selectors:
        selected = universe.select_atoms(selector)
        if selected.n_atoms > 0:
            return deduplicate_ca_atoms(selected)
    return universe.atoms[:0]


def _safe_atom_attr(atom, name: str, default):
    try:
        value = getattr(atom, name)
    except Exception:
        return default
    if value is None:
        return default
    return value


def deduplicate_ca_atoms(ca_atoms):
    """Choose one C-alpha per PDB residue, resolving alternate locations."""
    choices: Dict[Tuple[str, str, int, str], Tuple[Tuple[float, float, int], int]] = {}
    for position, atom in enumerate(ca_atoms):
        key = (
            str(_safe_atom_attr(atom, "chainID", "")),
            str(_safe_atom_attr(atom, "segid", "")),
            int(_safe_atom_attr(atom, "resid", position + 1)),
            str(_safe_atom_attr(atom, "icode", "")).strip(),
        )
        altloc = str(_safe_atom_attr(atom, "altLoc", "")).strip()
        occupancy = _safe_atom_attr(atom, "occupancy", np.nan)
        occupancy = float(occupancy) if np.isfinite(occupancy) else -1.0
        preferred_altloc = 1.0 if altloc in {"", "A", "1"} else 0.0
        rank = (preferred_altloc, occupancy, -position)
        if key not in choices or rank > choices[key][0]:
            choices[key] = (rank, position)
    indices = sorted(value[1] for value in choices.values())
    return ca_atoms[np.asarray(indices, dtype=int)]


def atomgroup_to_sequence(ca_atoms) -> str:
    return "".join(AA3_TO_1.get(str(name).upper(), "X") for name in ca_atoms.resnames)


def legacy_sequence_check(input_seq: str, structure_seq: str) -> Dict[str, object]:
    """Retain the original exact-length fields for older downstream readers."""
    same_length = len(input_seq) == len(structure_seq)
    if not input_seq or not structure_seq or not same_length:
        mismatch_count = None
        identity = None
    else:
        mismatch_count = sum(a != b for a, b in zip(input_seq, structure_seq))
        identity = 1.0 - mismatch_count / len(input_seq)
    return {
        "len_input": len(input_seq),
        "len_structure": len(structure_seq),
        "same_length": same_length,
        "mismatch_count": mismatch_count,
        "identity": identity,
    }


def align_structure_to_input(input_seq: str, structure_seq: str) -> Dict[str, object]:
    """Return a deterministic global alignment and bidirectional index maps."""
    input_seq = input_seq.upper()
    structure_seq = structure_seq.upper()
    if not input_seq or not structure_seq:
        raise ValueError("Cannot align an empty input or structure sequence")

    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    # Keep terminal and internal gaps under the same penalty. Free terminal gaps
    # can incorrectly move a true internal unresolved stretch to a terminus in
    # repetitive sequences. A global alignment still maps domain subsequences;
    # its score simply records the chain flanks that are absent from the input.

    alignment = aligner.align(input_seq, structure_seq)[0]
    input_to_structure: List[Optional[int]] = [None] * len(input_seq)
    structure_to_input: List[Optional[int]] = [None] * len(structure_seq)
    for input_block, structure_block in zip(alignment.aligned[0], alignment.aligned[1]):
        input_start, input_end = (int(x) for x in input_block)
        structure_start, structure_end = (int(x) for x in structure_block)
        if input_end - input_start != structure_end - structure_start:
            raise RuntimeError("Alignment returned unequal ungapped block lengths")
        for input_index, structure_index in zip(
            range(input_start, input_end), range(structure_start, structure_end)
        ):
            input_to_structure[input_index] = structure_index
            structure_to_input[structure_index] = input_index

    aligned_pairs = [
        (i, j) for i, j in enumerate(input_to_structure) if j is not None
    ]
    comparable = [(i, j) for i, j in aligned_pairs if structure_seq[j] != "X"]
    identical = sum(input_seq[i] == structure_seq[j] for i, j in comparable)
    identity = identical / len(comparable) if comparable else 0.0
    return {
        "method": "biopython_pairwise_global",
        "score": float(alignment.score),
        "aligned_pairs": len(aligned_pairs),
        "identical_pairs": int(identical),
        "comparable_pairs": len(comparable),
        "identity": float(identity),
        "input_coverage": len(aligned_pairs) / len(input_seq),
        "structure_coverage": len(aligned_pairs) / len(structure_seq),
        "input_unresolved_count": len(input_seq) - len(aligned_pairs),
        "structure_unmapped_count": len(structure_seq) - len(aligned_pairs),
        "input_to_structure_index": input_to_structure,
        "structure_to_input_index": structure_to_input,
    }


def pdb_residue_record(atom, structure_index: int, input_index: int, input_aa: str) -> dict:
    chain_id = str(_safe_atom_attr(atom, "chainID", ""))
    segid = str(_safe_atom_attr(atom, "segid", ""))
    resid = int(_safe_atom_attr(atom, "resid", structure_index + 1))
    icode = str(_safe_atom_attr(atom, "icode", "")).strip()
    structure_aa = AA3_TO_1.get(str(atom.resname).upper(), "X")
    residue_label = f"{chain_id or segid}:{resid}{icode}"
    return {
        "input_index_0based": int(input_index),
        "input_position_1based": int(input_index + 1),
        "input_aa": input_aa,
        "structure_index_0based": int(structure_index),
        "structure_aa": structure_aa,
        "pdb_chain_id": chain_id,
        "pdb_segid": segid,
        "pdb_resid": resid,
        "pdb_insertion_code": icode,
        "pdb_residue_label": residue_label,
        "sequence_match": bool(input_aa == structure_aa),
    }


def build_contact_outputs(
    ca_atoms,
    input_seq: str,
    alignment: dict,
    cutoff: float,
    representation: str,
    include_ca_coordinates: bool,
) -> dict:
    coords = np.asarray(ca_atoms.positions, dtype=float)
    distances = distance_array(coords, coords)
    structure_contacts = distances < cutoff
    np.fill_diagonal(structure_contacts, False)

    input_to_structure = alignment["input_to_structure_index"]
    structure_to_input = alignment["structure_to_input_index"]
    resolved_mask = [index is not None for index in input_to_structure]
    residue_mapping = []
    for input_index, structure_index in enumerate(input_to_structure):
        if structure_index is not None:
            residue_mapping.append(
                pdb_residue_record(
                    ca_atoms[structure_index], structure_index, input_index, input_seq[input_index]
                )
            )

    output = {
        "contact_map_coordinate_system": "input_sequence_0based",
        "unresolved_pair_encoding": 0,
        "resolved_mask": resolved_mask,
        "residue_mapping": residue_mapping,
    }

    if representation in {"dense", "both"}:
        dense = np.zeros((len(input_seq), len(input_seq)), dtype=np.uint8)
        mapped_structure = np.asarray(
            [i for i, mapped in enumerate(structure_to_input) if mapped is not None], dtype=int
        )
        mapped_input = np.asarray(
            [structure_to_input[i] for i in mapped_structure], dtype=int
        )
        dense[np.ix_(mapped_input, mapped_input)] = structure_contacts[
            np.ix_(mapped_structure, mapped_structure)
        ].astype(np.uint8)
        output["contact_map"] = dense.tolist()

    upper_i, upper_j = np.where(np.triu(structure_contacts, k=1))
    edges = []
    for structure_i, structure_j in zip(upper_i.tolist(), upper_j.tolist()):
        input_i = structure_to_input[structure_i]
        input_j = structure_to_input[structure_j]
        if input_i is None or input_j is None:
            continue
        if input_i > input_j:
            input_i, input_j = input_j, input_i
        edges.append([int(input_i), int(input_j), round(float(distances[structure_i, structure_j]), 4)])
    edges.sort(key=lambda row: (row[0], row[1]))
    output["n_contact_edges"] = len(edges)
    if representation in {"edges", "both"}:
        output["contact_edges"] = edges

    if include_ca_coordinates:
        input_coords: List[Optional[List[float]]] = [None] * len(input_seq)
        for input_index, structure_index in enumerate(input_to_structure):
            if structure_index is not None:
                input_coords[input_index] = [
                    round(float(value), 4) for value in coords[structure_index]
                ]
        output["ca_coordinates_angstrom"] = input_coords
    return output


def mapping_quality(alignment: dict, min_identity: float, min_coverage: float) -> str:
    if (
        alignment["input_coverage"] == 1.0
        and alignment["structure_coverage"] == 1.0
        and alignment["identity"] == 1.0
    ):
        return "exact"
    if alignment["identity"] >= min_identity and alignment["input_coverage"] >= min_coverage:
        return "aligned"
    return "low_quality"


def audit_row(record: dict) -> dict:
    sequence_check = record.get("sequence_check") or {}
    alignment = record.get("alignment") or {}
    return {
        "name": record.get("name"),
        "split": record.get("split"),
        "status": record.get("status"),
        "error": record.get("error"),
        "pdb_id": record.get("pdb_id"),
        "chain": record.get("chain"),
        "pdb_file": record.get("pdb_file"),
        "experimental_method": record.get("experimental_method"),
        "resolution_angstrom": record.get("resolution_angstrom"),
        "n_models": record.get("n_models"),
        "n_residues_input": sequence_check.get("len_input"),
        "n_residues_structure_ca": sequence_check.get("len_structure"),
        "legacy_same_length": sequence_check.get("same_length"),
        "legacy_identity": sequence_check.get("identity"),
        "mapping_status": record.get("mapping_status"),
        "alignment_identity": alignment.get("identity"),
        "input_coverage": alignment.get("input_coverage"),
        "structure_coverage": alignment.get("structure_coverage"),
        "n_input_unresolved": alignment.get("input_unresolved_count"),
        "n_structure_unmapped": alignment.get("structure_unmapped_count"),
        "n_contact_edges": record.get("n_contact_edges"),
        "cutoff_angstrom": record.get("cutoff_angstrom"),
        "map_representation": record.get("map_representation"),
    }


def write_json(records: Sequence[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(records, handle, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    session = make_session()
    audit_csv = args.audit_csv or default_audit_path(args.output_json)

    frame = pd.read_csv(args.input_csv)
    required = {"name", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    if frame["name"].astype(str).duplicated().any():
        duplicates = sorted(frame.loc[frame["name"].astype(str).duplicated(), "name"].astype(str).unique())
        raise ValueError(f"Input CSV contains duplicate names, including: {duplicates[:5]}")
    if args.max_proteins > 0:
        frame = frame.head(args.max_proteins).copy()

    records: List[dict] = []
    for _, row in frame.iterrows():
        seq_id = str(row["name"]).strip()
        input_seq = re.sub(r"\s+", "", str(row["sequence"])).upper()
        record = {
            "name": seq_id,
            "sequence": input_seq,
            "cutoff_angstrom": float(args.cutoff),
            "map_representation": args.map_representation,
            "status": "ok",
        }
        if "split" in frame.columns and pd.notna(row.get("split")):
            record["split"] = str(row["split"])

        try:
            pdb_id, chain = parse_seq_id(seq_id)
            record.update({"pdb_id": pdb_id, "chain": chain})
            pdb_file = download_pdb(
                session, pdb_id, args.pdb_dir, overwrite=args.overwrite_pdb
            )
            record["pdb_file"] = os.path.abspath(pdb_file)
            record.update(parse_pdb_metadata(pdb_file))

            universe = mda.Universe(pdb_file)
            record["n_models"] = int(len(universe.trajectory))
            ca_atoms = select_chain_ca(universe, chain)
            if ca_atoms.n_atoms == 0:
                raise RuntimeError(f"No C-alpha atoms found for chain {chain}")

            structure_seq = atomgroup_to_sequence(ca_atoms)
            sequence_check = legacy_sequence_check(input_seq, structure_seq)
            alignment = align_structure_to_input(input_seq, structure_seq)
            record.update({
                "structure_sequence": structure_seq,
                "n_residues_structure": int(ca_atoms.n_atoms),
                "sequence_check": sequence_check,
                "alignment": alignment,
                "mapping_status": mapping_quality(
                    alignment, args.min_mapping_identity, args.min_input_coverage
                ),
            })
            record.update(build_contact_outputs(
                ca_atoms=ca_atoms,
                input_seq=input_seq,
                alignment=alignment,
                cutoff=args.cutoff,
                representation=args.map_representation,
                include_ca_coordinates=args.include_ca_coordinates,
            ))
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    write_json(records, args.output_json)
    os.makedirs(os.path.dirname(os.path.abspath(audit_csv)), exist_ok=True)
    pd.DataFrame([audit_row(record) for record in records]).to_csv(audit_csv, index=False)

    n_ok = sum(record.get("status") == "ok" for record in records)
    quality_counts = pd.Series(
        [record.get("mapping_status", "error") for record in records]
    ).value_counts()
    print(f"Done. OK={n_ok}, errors={len(records) - n_ok}, total={len(records)}")
    print("Mapping status: " + ", ".join(f"{key}={value}" for key, value in quality_counts.items()))
    print(f"Output: {args.output_json}")
    print(f"Audit:  {audit_csv}")


if __name__ == "__main__":
    main()
