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


def aggregate_matrices(arr: np.ndarray, aggregate: str = "mean") -> np.ndarray:
    # arr may be shape (H,L,L) or (Layers,H,L,L) etc.
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if aggregate == "mean":
            return arr.mean(axis=0)
        if aggregate == "sum":
            return arr.sum(axis=0)
        if aggregate == "max":
            return arr.max(axis=0)
    if arr.ndim == 4:
        # default: mean over layers and heads
        if aggregate == "mean":
            return arr.mean(axis=(0, 1))
        if aggregate == "sum":
            return arr.sum(axis=(0, 1))
        if aggregate == "max":
            return arr.max(axis=(0, 1))
    raise ValueError(f"Unsupported attention array shape: {arr.shape}")


def row_softmax(mat: np.ndarray) -> np.ndarray:
    # apply softmax along last dimension (targets) for each source row
    m = mat - np.max(mat, axis=-1, keepdims=True)
    exp = np.exp(m)
    s = exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-12)
    return s


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
    aggregate: str,
    per_head: bool,
    layer: int,
    head: int,
    threshold: float,
    topk: int,
    normalize: bool,
    softmax: bool,
    out_format: str,
):
    # handle shapes
    if arr.ndim == 2:
        mats = [(None, arr)]
    elif arr.ndim == 3:
        # (H, L, L)
        if per_head:
            mats = [(h, arr[h]) for h in range(arr.shape[0])]
        else:
            mats = [(None, aggregate_matrices(arr, aggregate=aggregate))]
    elif arr.ndim == 4:
        # (Layers, H, L, L)
        if layer is not None:
            if layer < 0 or layer >= arr.shape[0]:
                raise ValueError("layer index out of bounds")
            layer_arr = arr[layer]
            if per_head:
                mats = [(h, layer_arr[h]) for h in range(layer_arr.shape[0])]
            else:
                mats = [(None, aggregate_matrices(layer_arr, aggregate=aggregate))]
        else:
            if per_head:
                # flatten layers*heads as separate items
                mats = []
                for Lidx in range(arr.shape[0]):
                    for h in range(arr.shape[1]):
                        mats.append((f"L{Lidx}_H{h}", arr[Lidx, h]))
            else:
                mats = [(None, aggregate_matrices(arr, aggregate=aggregate))]
    else:
        raise ValueError(f"Unsupported array ndim: {arr.ndim}")

    for tag, mat in mats:
        if softmax:
            mat = row_softmax(mat)
        edges = build_edges(mat, threshold=threshold, topk=topk, normalize=normalize)
        L = mat.shape[-1]
        # file naming
        safe_tag = f"_{tag}" if tag is not None else ""
        base = os.path.join(out_dir, f"{rec_id}{safe_tag}")
        if out_format in ("json", "json_edges"):
            write_json_edges(base + ".json", rec_id + (f"_{tag}" if tag else ""), L, edges)
        if out_format in ("graphml", "all"):
            try:
                write_graphml(base + ".graphml", L, edges)
            except RuntimeError as e:
                print(f"Skipping GraphML for {rec_id}{safe_tag}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input JSON path")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory")
    p.add_argument("--aggregate", choices=("mean", "sum", "max"), default="mean", help="How to aggregate heads/layers")
    p.add_argument("--per-head", action="store_true", help="Export per-head graphs instead of aggregated")
    p.add_argument("--layer", type=int, default=None, help="Pick a specific layer (0-based) from a 4D tensor")
    p.add_argument("--head", type=int, default=None, help="(Not used) reserved for future")
    p.add_argument("--threshold", type=float, default=0.0, help="Minimum edge weight to keep")
    p.add_argument("--topk", type=int, default=0, help="Keep top-k targets per source (overrides threshold selection for sparsity)")
    p.add_argument("--normalize", action="store_true", help="Row-normalize attention before thresholding")
    p.add_argument("--softmax", action="store_true", help="Apply softmax to rows before exporting")
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
                aggregate=args.aggregate,
                per_head=args.per_head,
                layer=args.layer,
                head=args.head,
                threshold=args.threshold,
                topk=args.topk,
                normalize=args.normalize,
                softmax=args.softmax,
                out_format=args.format,
            )
            print(f"Wrote graphs for {rec_id}")
        except Exception as e:
            print(f"Failed to process {rec_id}: {e}")


if __name__ == "__main__":
    main()
