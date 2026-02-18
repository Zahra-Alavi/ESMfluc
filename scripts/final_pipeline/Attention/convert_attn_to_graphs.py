#!/usr/bin/env python3
"""
Convert pre-computed attention matrices (JSON) to directional graphs.

Input JSON is expected as a list of records with the following structure:
  - 'name': sequence identifier
  - 'sequence': amino acid sequence string (for later coloring/analysis)
  - 'attention_weights': single-head LxL attention matrix (already softmax-normalized per row)
  - 'neq_preds': optional NEQ predictions
  - 'ss_pred': optional secondary structure predictions

The attention matrices are produced by get_attn.py and are single-head, 
already softmax-normalized row-wise.

Output: per-sequence JSON edge lists (with sequence and optional GraphML) in `--output-dir`.
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


def find_matrices(obj: Any) -> List[Tuple[str, str, np.ndarray]]:
    """Extract (name, sequence, attention_weights) from JSON list of records.
    
    Expected format: list of dicts with keys 'name', 'sequence', 'attention_weights', etc.
    Returns list of (seq_id, amino_acid_sequence, attention_matrix) tuples.
    """
    if not isinstance(obj, list):
        raise ValueError("Input JSON must be a list of records")
    
    out = []
    for rec in obj:
        if not isinstance(rec, dict):
            continue
        
        # Extract name, sequence, and attention_weights
        rec_id = rec.get("name")
        seq = rec.get("sequence")
        mat = rec.get("attention_weights")
        
        if rec_id is None or mat is None:
            print(f"Warning: record missing 'name' or 'attention_weights'")
            continue
        
        if seq is None:
            print(f"Warning: record {rec_id} missing 'sequence'; using empty string")
            seq = ""
        
        try:
            arr = np.array(mat, dtype=float)
            out.append((str(rec_id), str(seq), arr))
        except Exception as e:
            print(f"Warning: failed to parse record {rec_id}: {e}")
            continue
    
    return out


# Note: attention matrices from get_attn.py are single-head LxL and already
# softmax-normalized per-row. No aggregation or normalization is needed.


def build_edges(
    mat: np.ndarray,
    threshold: float = 0.0,
    topk: int = 0,
    percentile: int = 0,
    local_peaks: bool = False,
    normalize: bool = False,
) -> List[Dict[str, Any]]:
    """Build edge list from attention matrix with multiple sparsification options.
    
    Args:
        mat: Attention matrix (L, L)
        threshold: Absolute minimum edge weight
        topk: Keep only top-k connections per row (overrides threshold)
        percentile: Keep only top-X percentile weights per row (0-100)
        local_peaks: Keep only local maxima (peaks) in each row
        normalize: Row-normalize before filtering
    """
    L = mat.shape[-1]
    edges: List[Dict[str, Any]] = []
    work = mat.copy()
    
    if normalize:
        row_sums = work.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            work = np.divide(work, row_sums, where=row_sums != 0)
    
    for i in range(L):
        row = work[i]
        selected_indices = []
        
        if topk and topk > 0:
            # Keep top-k targets per query
            k = min(topk, L)
            selected_indices = list(np.argpartition(-row, k - 1)[:k])
        elif percentile > 0:
            # Keep only top-X percentile per row
            threshold_val = np.percentile(row, percentile)
            selected_indices = list(np.where(row >= threshold_val)[0])
        elif local_peaks:
            # Keep only local maxima (peaks) in the row
            for j in range(L):
                is_peak = False
                if j == 0:
                    is_peak = row[j] >= row[j + 1] if L > 1 else True
                elif j == L - 1:
                    is_peak = row[j] >= row[j - 1]
                else:
                    is_peak = row[j] >= row[j - 1] and row[j] >= row[j + 1]
                
                if is_peak and row[j] >= threshold:
                    selected_indices.append(j)
        else:
            # Standard thresholding
            selected_indices = list(np.where(row >= threshold)[0])
        
        # Add edges for selected indices
        for j in selected_indices:
            w = float(row[j])
            if math.isfinite(w) and (w >= threshold):
                edges.append({"source": int(i), "target": int(j), "weight": w})
    
    return edges


def write_json_edges(path: str, seq_id: str, sequence: str, L: int, edges: List[Dict[str, Any]]):
    out = {
        "id": seq_id,
        "sequence": sequence,
        "length": L,
        "edges": edges
    }
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
    
    # Write with explicit key definitions for Cytoscape compatibility
    nx.write_graphml(G, path)
    
    # Post-process to add explicit edge attribute key definition
    with open(path, 'r') as f:
        content = f.read()
    
    # Insert key definition if not present
    if '<key id="d0"' not in content:
        # Insert after <graph> tag
        content = content.replace(
            '<graph edgedefault="directed">',
            '<graph edgedefault="directed">\n'
            '  <key id="d0" for="edge" attr.name="weight" attr.type="double"/>'
        )
        with open(path, 'w') as f:
            f.write(content)


def process_one(
    rec_id: str,
    sequence: str,
    arr: np.ndarray,
    out_dir: str,
    threshold: float,
    topk: int,
    percentile: int,
    local_peaks: bool,
    normalize: bool,
    out_format: str,
):
    """Process a single attention matrix and sequence.

    The attention matrices produced by `get_attn.py` are single-head and
    already softmax-normalized row-wise. This function enforces a 2D input,
    stores the sequence for later coloring, and converts to edge lists / GraphML.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D attention matrix (L,L); got shape {arr.shape}")

    mat = arr
    edges = build_edges(
        mat, 
        threshold=threshold, 
        topk=topk, 
        percentile=percentile,
        local_peaks=local_peaks,
        normalize=normalize
    )
    L = mat.shape[-1]
    base = os.path.join(out_dir, f"{rec_id}")
    if out_format in ("json", "json_edges", "all"):
        write_json_edges(base + ".json", rec_id, sequence, L, edges)
    if out_format in ("graphml", "all"):
        try:
            write_graphml(base + ".graphml", L, edges)
        except RuntimeError as e:
            print(f"Skipping GraphML for {rec_id}: {e}")


def main():
    p = argparse.ArgumentParser(
        description="Convert pre-computed attention matrices to sparsified directional graphs with multiple filtering strategies."
    )
    p.add_argument("--input", "-i", required=True, help="Input JSON path with attention matrices")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory for graph files")
    
    # Sparsification options (use one or combine them)
    p.add_argument(
        "--threshold", 
        type=float, 
        default=0.0, 
        help="Minimum absolute edge weight to keep (default: 0.0, keep all)"
    )
    p.add_argument(
        "--topk", 
        type=int, 
        default=0, 
        help="Keep only top-k strongest targets per query (e.g., --topk 5). Overrides threshold."
    )
    p.add_argument(
        "--percentile", 
        type=int, 
        default=0,
        help="Keep only top-X percentile weights per row (0-100, e.g., --percentile 75 keeps top 25%%. Overrides threshold."
    )
    p.add_argument(
        "--local-peaks",
        action="store_true",
        help="Keep only local maxima (peaks) in each row. Useful for finding sharp attention patterns."
    )
    p.add_argument(
        "--normalize", 
        action="store_true", 
        help="Row-normalize attention before filtering (useful if combining with threshold)"
    )
    p.add_argument(
        "--format", 
        choices=("json", "graphml", "all", "json_edges"), 
        default="json", 
        help="Output format (default: json)"
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    obj = load_json(args.input)
    records = find_matrices(obj)
    if not records:
        print("No attention matrices found in input JSON. Check keys and structure.")
        return
    
    print(f"Sparsification settings:")
    print(f"  threshold={args.threshold}, topk={args.topk}, percentile={args.percentile}, local_peaks={args.local_peaks}")
    
    for rec_id, sequence, arr in records:
        try:
            process_one(
                rec_id=rec_id,
                sequence=sequence,
                arr=arr,
                out_dir=args.output_dir,
                threshold=args.threshold,
                topk=args.topk,
                percentile=args.percentile,
                local_peaks=args.local_peaks,
                normalize=args.normalize,
                out_format=args.format,
            )
            print(f"✓ Wrote graphs for {rec_id}")
        except Exception as e:
            print(f"✗ Failed to process {rec_id}: {e}")


if __name__ == "__main__":
    main()
