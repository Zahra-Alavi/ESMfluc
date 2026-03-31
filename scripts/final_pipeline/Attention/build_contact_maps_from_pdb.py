#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build per-protein contact maps (NxN) from PDB structures for a test set.

Expected input CSV columns:
  - name      (format: PDBID_CHAIN, e.g. 2pbk_B)
  - sequence  (one-letter amino-acid sequence)

For each row:
  1) Download PDB from RCSB if not present locally
  2) Load structure with MDAnalysis
  3) Select CA atoms for the requested chain
  4) Compute pairwise distance matrix
  5) Threshold into binary contact map (default cutoff 8.0 Å)
  6) Store contact map and metadata in one JSON record

Output JSON is a list of records, similar style to attention JSON.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

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


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build contact maps from PDB for test proteins")
    p.add_argument("--input_csv", required=True, help="CSV with columns name, sequence")
    p.add_argument("--output_json", required=True, help="Output JSON path")
    p.add_argument("--pdb_dir", default="./data/pdb_cache", help="Directory to cache downloaded PDB files")
    p.add_argument("--cutoff", type=float, default=8.0, help="Contact cutoff in Å (default 8.0)")
    p.add_argument("--max_proteins", type=int, default=0, help="Optional cap for quick test (0=all)")
    p.add_argument("--overwrite_pdb", action="store_true", help="Redownload PDB even if cached")
    return p.parse_args()


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "ESMfluc-contactmap-builder/1.0"})
    return s


def parse_seq_id(seq_id: str) -> Tuple[str, str]:
    if "_" not in seq_id:
        raise ValueError(f"Invalid seq id '{seq_id}' (expected PDBID_CHAIN)")
    pdb_id, chain = seq_id.split("_", 1)
    return pdb_id.lower(), chain


def download_pdb(session: requests.Session, pdb_id: str, out_dir: str, overwrite: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path) and not overwrite:
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = session.get(url, timeout=40)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def select_chain_ca(u: "mda.Universe", chain: str):
    # Try most common chain selector first
    sel = u.select_atoms(f"protein and name CA and chainID {chain}")
    if sel.n_atoms > 0:
        return sel

    # Fallback for structures where chain appears as segid
    sel = u.select_atoms(f"protein and name CA and segid {chain}")
    if sel.n_atoms > 0:
        return sel

    # Last-resort: try lowercase/uppercase variants
    variants = {chain, chain.upper(), chain.lower()}
    for c in variants:
        sel = u.select_atoms(f"protein and name CA and chainID {c}")
        if sel.n_atoms > 0:
            return sel
        sel = u.select_atoms(f"protein and name CA and segid {c}")
        if sel.n_atoms > 0:
            return sel

    return sel  # empty AtomGroup


def atomgroup_to_sequence(ca_atoms) -> str:
    return "".join(AA3_TO_1.get(str(rn).upper(), "X") for rn in ca_atoms.resnames)


def contact_map_from_ca(ca_atoms, cutoff: float) -> np.ndarray:
    coords = ca_atoms.positions
    d = distance_array(coords, coords)
    cm = (d < cutoff).astype(np.uint8)
    np.fill_diagonal(cm, 0)
    return cm


def compare_sequences(input_seq: str, struct_seq: str) -> Dict[str, float]:
    if len(input_seq) == 0 or len(struct_seq) == 0:
        return {
            "len_input": len(input_seq),
            "len_structure": len(struct_seq),
            "same_length": False,
            "mismatch_count": None,
            "identity": None,
        }

    if len(input_seq) != len(struct_seq):
        return {
            "len_input": len(input_seq),
            "len_structure": len(struct_seq),
            "same_length": False,
            "mismatch_count": None,
            "identity": None,
        }

    mism = sum(1 for a, b in zip(input_seq, struct_seq) if a != b)
    ident = 1.0 - (mism / len(input_seq))
    return {
        "len_input": len(input_seq),
        "len_structure": len(struct_seq),
        "same_length": True,
        "mismatch_count": int(mism),
        "identity": float(ident),
    }


def main() -> None:
    args = parse_args()
    session = make_session()

    df = pd.read_csv(args.input_csv)
    required = {"name", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    if args.max_proteins and args.max_proteins > 0:
        df = df.head(args.max_proteins).copy()

    out_records: List[dict] = []

    for _, row in df.iterrows():
        seq_id = str(row["name"]).strip()
        input_seq = str(row["sequence"]).strip()

        rec = {
            "name": seq_id,
            "sequence": input_seq,
            "cutoff_angstrom": float(args.cutoff),
            "status": "ok",
        }

        try:
            pdb_id, chain = parse_seq_id(seq_id)
            rec["pdb_id"] = pdb_id
            rec["chain"] = chain

            pdb_file = download_pdb(session, pdb_id, args.pdb_dir, overwrite=args.overwrite_pdb)
            u = mda.Universe(pdb_file)
            ca = select_chain_ca(u, chain)

            if ca.n_atoms == 0:
                raise RuntimeError(f"No CA atoms found for chain {chain}")

            struct_seq = atomgroup_to_sequence(ca)
            seq_cmp = compare_sequences(input_seq, struct_seq)

            cm = contact_map_from_ca(ca, args.cutoff)
            rec.update(
                {
                    "structure_sequence": struct_seq,
                    "n_residues_structure": int(ca.n_atoms),
                    "sequence_check": seq_cmp,
                    "contact_map": cm.tolist(),
                }
            )

        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)

        out_records.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out_records, f)

    n_ok = sum(1 for r in out_records if r.get("status") == "ok")
    print(f"Done. OK={n_ok}, errors={len(out_records) - n_ok}, total={len(out_records)}")
    print(f"Output: {args.output_json}")


if __name__ == "__main__":
    main()
