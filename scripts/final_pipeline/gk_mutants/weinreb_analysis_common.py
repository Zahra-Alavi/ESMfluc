#!/usr/bin/env python3
"""Shared, dependency-light utilities for the Weinreb mutant analysis.

Coordinates used by the contribution matrices are zero based.  FASTA and
``seq_pos`` coordinates are one based.  Paper numbering follows PDB residue
numbering and is explicitly read through the supplied position map.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


GK_MUTANTS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = GK_MUTANTS_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
WEINREB_RESULTS_ROOT = PIPELINE_ROOT / "results" / "weinreb2025_mutants_no_AV"
WEINREB_ANALYSIS_ROOT = WEINREB_RESULTS_ROOT / "mutation_network_analysis"
WEINREB_FASTA = PROJECT_ROOT / "data" / "weinreb2025_mutants_no_AV.fasta"


PRIMARY_CONDITION = "esm3_top28_bilstm_attn"
PRIMARY_SEEDS = (1, 2, 3)
NEW_SEQUENCES = ("A175G", "A175P", "A175G_A176G")


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    name = None
    chunks: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name = line[1:].split()[0]
            if not name or name in records:
                raise ValueError(f"Invalid or duplicate FASTA name: {name!r}")
            chunks = []
        elif name is None:
            raise ValueError("Sequence found before first FASTA header")
        else:
            chunks.append(line.upper())
    if name is not None:
        records[name] = "".join(chunks)
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records


def read_manifest(path: Path, condition: str = PRIMARY_CONDITION,
                  seeds: Sequence[int] = PRIMARY_SEEDS) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [r for r in rows if r.get("condition") == condition and
                int(r.get("seed", -1)) in set(seeds)]
    selected.sort(key=lambda r: int(r["seed"]))
    got = [int(r["seed"]) for r in selected]
    if got != sorted(seeds):
        raise ValueError(f"Expected seeds {list(seeds)} for {condition}; got {got}")
    return selected


@dataclass
class ProteinPayload:
    name: str
    sequence: str
    contribution: np.ndarray
    attention: np.ndarray
    margin: np.ndarray
    bias: float
    flexible_scores: np.ndarray
    neq_preds: np.ndarray


def load_seed(row: Mapping[str, str]) -> Dict[str, ProteinPayload]:
    attention_path = Path(row["attention_json"])
    contribution_path = Path(row["logit_contributions_npz"])
    records = json.loads(attention_path.read_text(encoding="utf-8"))
    payloads: Dict[str, ProteinPayload] = {}
    with np.load(contribution_path, allow_pickle=False) as archive:
        for rec in records:
            name = str(rec["name"])
            key = str(rec["flex_minus_rigid_logit_contribution_key"])
            c = np.asarray(archive[key], dtype=float)
            a = np.asarray(rec["attention_weights"], dtype=float)
            payloads[name] = ProteinPayload(
                name=name,
                sequence=str(rec["sequence"]),
                contribution=c,
                attention=a,
                margin=np.asarray(rec["flex_minus_rigid_logit_margin"], dtype=float),
                bias=float(rec["flex_minus_rigid_logit_margin_bias"]),
                flexible_scores=np.asarray(rec["flexible_scores"], dtype=float),
                neq_preds=np.asarray(rec["neq_preds"], dtype=float),
            )
    return payloads


def parse_position_map(path: Path, protein: str = "WT_star") -> Dict[int, Tuple[str, int]]:
    """Return fasta_pos -> (PDB chain, paper/PDB residue)."""
    result: Dict[int, Tuple[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["protein"] == protein:
                result[int(row["seq_pos"])] = (row["pdb_chain"], int(row["pdb_resi"]))
    if not result:
        raise ValueError(f"No {protein} entries in {path}")
    return result


def parse_ca(path: Path, chain: str = "A") -> Dict[int, np.ndarray]:
    coords: Dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[21].strip() != chain:
            continue
        try:
            resi = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        coords.setdefault(resi, xyz)
    return coords


def kabsch_displacements(reference: Mapping[int, np.ndarray], mobile: Mapping[int, np.ndarray]) -> Dict[int, float]:
    """Globally align mobile onto reference and return per-residue displacement."""
    common = sorted(set(reference) & set(mobile))
    if len(common) < 3:
        return {}
    x = np.vstack([reference[k] for k in common])
    y = np.vstack([mobile[k] for k in common])
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((y - yc).T @ (x - xc))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    aligned = (y - yc) @ rot + xc
    return {k: float(np.linalg.norm(aligned[i] - x[i])) for i, k in enumerate(common)}


def mutation_sites(sequence: str, wt_star: str) -> List[int]:
    if len(sequence) != len(wt_star):
        raise ValueError("Length-changing variants are not supported")
    return [i + 1 for i, (aa, wt) in enumerate(zip(sequence, wt_star)) if aa != wt]


def named_paper_sites(name: str) -> List[int]:
    if name in {"WT", "WT_star", "WT*"}:
        return []
    return [int(x) for x in re.findall(r"[A-Z](\d+)[A-Z]", name)]


def effect_classification(baseline: float, delta: float, eps: float = 1e-12) -> str:
    final = baseline + delta
    if baseline < -eps and final > eps or baseline > eps and final < -eps:
        return "polarity switch"
    if baseline >= 0:
        return "stronger flexibility support" if delta > 0 else "weaker flexibility support"
    return "weaker rigidity support" if delta > 0 else "stronger rigidity support"


def domain_annotation(paper_pos: int | None) -> Tuple[str, str]:
    """Coarse guanylate-kinase regions; deliberately allows hinge overlap in function.

    Boundaries are analysis annotations rather than claims of atomic precision.  They
    are recorded in the output configuration so sensitivity analyses can change them.
    """
    if paper_pos is None:
        return "unmapped", "missing structural coordinates"
    if 26 <= paper_pos <= 33:
        return "CORE", "P-loop"
    if 42 <= paper_pos <= 116:
        return "GMP-binding", "nucleotide-binding domain"
    if 126 <= paper_pos <= 165:
        return "LID", "LID motion"
    if 37 <= paper_pos <= 41 or 117 <= paper_pos <= 125 or 166 <= paper_pos <= 180:
        return "CORE/hinge", "hinge-adjacent"
    return "CORE", "core/other"


def reconstruct_signed_evidence(c: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float]:
    """Recover key evidence s from C=A*s using a stable least-squares estimate."""
    denom = np.sum(a * a, axis=0)
    s = np.divide(np.sum(a * c, axis=0), denom, out=np.zeros(c.shape[1]), where=denom > 0)
    err = float(np.max(np.abs(c - a * s[None, :])))
    return s, err


def mean_sd_se(values: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    n = values.shape[axis]
    sd = values.std(axis=axis, ddof=1) if n > 1 else np.zeros_like(mean)
    return mean, sd, sd / math.sqrt(n)


def sign_agreement(values: np.ndarray, axis: int = 0, tol: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    pos = np.sum(values > tol, axis=axis)
    neg = np.sum(values < -tol, axis=axis)
    return np.maximum(pos, neg)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def write_csv(path: Path, rows: Iterable[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_or_none(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
