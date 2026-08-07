"""Shared input and numerical helpers for the fixed-WT* GK v2 analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from weinreb_analysis_common import ProteinPayload, load_seed


CONDITIONS = (
    "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn",
    "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn",
    "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
)
PRIMARY_CONDITION = "esm3_top28_bilstm_attn"
SEEDS = (1, 2, 3)
PAPER_MUTANTS = (
    "S30G", "E88N", "A31V", "S53N", "S30Q", "R60K", "G33S",
    "T101A", "V32L", "G178S", "D179S", "A175T", "L174F", "Q177D",
    "E173N", "A176C", "L174G", "P29A", "E173H", "P29V", "A58V",
    "I92V", "S193A", "P59A", "S51G", "A127T", "G62S", "V25A",
    "R42H", "P46A", "V120A", "G94S", "P61A", "I118F",
)


def manifest_rows(path: Path) -> dict[tuple[str, int], dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {}
    for row in rows:
        key = (str(row["condition"]), int(row["seed"]))
        if key in result:
            raise ValueError(f"Duplicate manifest row: {key}")
        result[key] = row
    return result


def load_combined_runs(
    base_manifest: Path,
    addendum_manifest: Path,
    conditions: Sequence[str] = CONDITIONS,
    seeds: Sequence[int] = SEEDS,
) -> dict[tuple[str, int], dict[str, ProteinPayload]]:
    base = manifest_rows(base_manifest)
    addendum = manifest_rows(addendum_manifest)
    loaded = {}
    for condition in conditions:
        for seed in seeds:
            key = (condition, int(seed))
            if key not in base or key not in addendum:
                raise ValueError(f"Missing base or A176G addendum run: {key}")
            primary = load_seed(base[key])
            extra = load_seed(addendum[key])
            overlap = set(primary) & set(extra)
            if overlap:
                raise ValueError(f"Duplicate proteins in addendum {key}: {sorted(overlap)}")
            primary.update(extra)
            loaded[key] = primary
    return loaded


def reconstruct_evidence(contribution: np.ndarray, attention: np.ndarray):
    denominator = np.sum(attention * attention, axis=0)
    evidence = np.divide(
        np.sum(attention * contribution, axis=0),
        denominator,
        out=np.zeros(contribution.shape[1], dtype=float),
        where=denominator > 0,
    )
    error = float(np.max(np.abs(contribution - attention * evidence[None, :])))
    return evidence, error


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 1.0
    for reverse_index in range(len(p) - 1, -1, -1):
        original = order[reverse_index]
        running = min(running, p[original] * len(p) / (reverse_index + 1))
        adjusted[original] = running
    return np.minimum(adjusted, 1.0)


def write_csv(path: Path, rows: Iterable[Mapping]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
