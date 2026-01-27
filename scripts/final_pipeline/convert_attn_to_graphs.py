#!/usr/bin/env python3
"""
Convert attention matrices (JSON) to directional graphs.

Input JSON expected as a list of records or a dict with a top-level list under
`sequences`/`data`. Each record should include an identifier (e.g. `id`,
`seq_id`) and an attention matrix under a key like `attn` or `attention`.

Supported attention shapes:
- (L, L) -> single matrix
- (H, L, L) -> heads, optionally aggregate or export per-head
- (Layers, H, L, L) -> pick layer, aggregate heads or export per-head

Output: per-sequence JSON edge lists (and optional GraphML) in `--output-dir`.
"""
import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover - informative error when missing
    nx = None


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def find_matrices(obj: Any) -> List[Tuple[str, np.ndarray]]:
    """Extract (name, attention_weights) from JSON list of records.
    
    Expected format: list of dicts with keys 'name', 'sequence', 'attention_weights', etc.
    """
    if not isinstance(obj, list):
        raise ValueError("Input JSON must be a list of records")
    
    out = []
    for rec in obj:
        if not isinstance(rec, dict):
            continue
        
        # Extract name and attention_weights
        rec_id = rec.get("name")
        mat = rec.get("attention_weights")
        
        if rec_id is None or mat is None:
            continue
        
        try:
            arr = np.array(mat, dtype=float)
            out.append((str(rec_id), arr))
        except Exception as e:
            print(f"Warning: failed to parse record {rec_id}: {e}")
            continue
    
    return out


# Note: attention in this pipeline is a single-head LxL matrix and already
# softmax-normalized per-row (see models.SelfAttentionLayer). The converter
# therefore expects a 2D (L, L) attention matrix and does not perform
# aggregation across heads/layers or apply softmax.


def build_edges(
    mat: np.ndarray,
    threshold: float = 0.0,
    topk: int = 0,
    normalize: bool = False,
) -> List[Dict[str, Any]]:
    L = mat.shape[-1]
    edges: List[Dict[str, Any]] = []
    work = mat.copy()
    if normalize:
        row_sums = work.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            work = np.divide(work, row_sums, where=row_sums != 0)
    for i in range(L):
        row = work[i]
        if topk and topk > 0:
            # pick topk indices
            k = min(topk, L)
            idx = np.argpartition(-row, k - 1)[:k]
            for j in idx:
                w = float(row[j])
                if math.isfinite(w) and (w >= threshold):
                    edges.append({"source": int(i), "target": int(j), "weight": w})
        else:
            for j, wv in enumerate(row.tolist()):
                w = float(wv)
                if math.isfinite(w) and (w >= threshold):
                    edges.append({"source": int(i), "target": int(j), "weight": w})
    return edges


def write_json_edges(path: str, seq_id: str, L: int, edges: List[Dict[str, Any]]):
    out = {"id": seq_id, "length": L, "edges": edges}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def write_graphml(path: str, L: int, edges: List[Dict[str, Any]]):
    if nx is None:
        raise RuntimeError("networkx is required to write GraphML. Install via pip install networkx")
    G = nx.DiGraph()
    for i in range(L):
        G.add_node(i)
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    nx.write_graphml(G, path)


def process_one(
    rec_id: str,
    arr: np.ndarray,
    out_dir: str,
    threshold: float,
    topk: int,
    normalize: bool,
    out_format: str,
):
    """Process a single attention matrix assumed to be 2D (L, L).

    The attention matrices produced by `get_attn.py` are single-head and
    already softmax-normalized row-wise. This function enforces a 2D input
    and converts it to edge lists / GraphML.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D attention matrix (L,L); got shape {arr.shape}")

    mat = arr
    edges = build_edges(mat, threshold=threshold, topk=topk, normalize=normalize)
    L = mat.shape[-1]
    base = os.path.join(out_dir, f"{rec_id}")
    if out_format in ("json", "json_edges"):
        write_json_edges(base + ".json", rec_id, L, edges)
    if out_format in ("graphml", "all"):
        try:
            write_graphml(base + ".graphml", L, edges)
        except RuntimeError as e:
            print(f"Skipping GraphML for {rec_id}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input JSON path")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory")
    # Attention matrices are single-head LxL; aggregation/per-head options removed
    p.add_argument("--threshold", type=float, default=0.0, help="Minimum edge weight to keep")
    p.add_argument("--topk", type=int, default=0, help="Keep top-k targets per source (overrides threshold selection for sparsity)")
    p.add_argument("--normalize", action="store_true", help="Row-normalize attention before thresholding")
    # Note: do not apply softmax; matrices are already normalized by the model
    p.add_argument("--format", choices=("json", "graphml", "all", "json_edges"), default="json", help="Output format")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    obj = load_json(args.input)
    records = find_matrices(obj)
    if not records:
        print("No attention matrices found in input JSON. Check keys and structure.")
        return
    for rec_id, arr in records:
        try:
            process_one(
                rec_id=rec_id,
                arr=arr,
                out_dir=args.output_dir,
                threshold=args.threshold,
                topk=args.topk,
                normalize=args.normalize,
                out_format=args.format,
            )
            print(f"Wrote graphs for {rec_id}")
        except Exception as e:
            print(f"Failed to process {rec_id}: {e}")


if __name__ == "__main__":
    main()
