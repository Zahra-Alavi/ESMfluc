"""
cluster_attention_residues.py

For a given protein, cluster each residue into 3 groups based on its
attention profile, without assuming what the patterns mean:

  Group A:          first strong attention pattern
  Group B:          second strong attention pattern
  Background:       residues with weak/uniform attention (no strong pattern)

Method
------
For residue i, build a feature vector:
    feature[i] = [A[i, 0..N-1] || A[0..N-1, i]]   (row + column, length 2N)

These 2N-dimensional vectors capture both what residue i attends to (query
role) and what attends to it (key role).

Run K-means with K=3 in pure Python (no external dependencies beyond stdlib).
The "background" cluster is identified as the one with the lowest total
attention energy (mean of row_sum + col_sum across its members).
The remaining two clusters are labeled A and B by descending energy.

Optionally, pass --min_peak_pct to set a per-residue peak-attention threshold:
only residues whose column-maximum attention is in the top N% across the
protein are kept in Pattern A/B; the rest are demoted to Background.
This is percentile-based so it works consistently across proteins with
varying attention contrast.

Usage
-----
python cluster_attention_residues.py \
    --input  ../../data/attn_test_data_bilstm_frozen1-4.json \
    --protein 1jo0_A \
    [--n_init 10] \
    [--max_iter 100] \
    [--output results/clusters_1jo0_A.json]

Output (stdout + optional JSON):
    Background residues : [...]
    Pattern A residues  : [...]
    Pattern B residues  : [...]
"""

import argparse
import json
import math
import random
import sys


# ─────────────────────────────────────────────
#  Pure-Python K-means (no numpy / sklearn)
# ─────────────────────────────────────────────

def _sq_dist(a, b):
    """Squared Euclidean distance between two equal-length lists."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean_vec(vecs):
    """Element-wise mean of a list of equal-length lists."""
    n = len(vecs)
    d = len(vecs[0])
    return [sum(vecs[r][c] for r in range(n)) / n for c in range(d)]


def _normalize(vec):
    """L2-normalize a vector (in-place returns new list)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec[:]
    return [x / norm for x in vec]


def kmeans(features, k=3, n_init=10, max_iter=100, seed=42):
    """
    K-means clustering.

    Parameters
    ----------
    features : list of list[float]  (N x D)
    k        : number of clusters
    n_init   : number of independent restarts
    max_iter : max iterations per restart

    Returns
    -------
    labels   : list[int] of length N  (cluster index 0..k-1)
    inertia  : float  (within-cluster sum of squared distances)
    """
    rng = random.Random(seed)
    N = len(features)
    best_labels = None
    best_inertia = float('inf')

    for _ in range(n_init):
        # K-means++ initialisation
        centers = [features[rng.randrange(N)][:]]
        for _ in range(k - 1):
            dists = [min(_sq_dist(f, c) for c in centers) for f in features]
            total = sum(dists)
            if total == 0:
                centers.append(features[rng.randrange(N)][:])
                continue
            r = rng.random() * total
            cumul = 0.0
            chosen = 0
            for idx, d in enumerate(dists):
                cumul += d
                if cumul >= r:
                    chosen = idx
                    break
            centers.append(features[chosen][:])

        labels = [0] * N
        for iteration in range(max_iter):
            # Assign
            new_labels = [
                min(range(k), key=lambda c: _sq_dist(features[i], centers[c]))
                for i in range(N)
            ]
            if new_labels == labels and iteration > 0:
                break
            labels = new_labels
            # Update centroids
            for c in range(k):
                members = [features[i] for i in range(N) if labels[i] == c]
                if members:
                    centers[c] = _mean_vec(members)

        inertia = sum(
            _sq_dist(features[i], centers[labels[i]]) for i in range(N)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels[:]

    return best_labels, best_inertia


# ─────────────────────────────────────────────
#  Core clustering function
# ─────────────────────────────────────────────

def cluster_residues(attention_matrix, n_init=10, max_iter=100, seed=42,
                     min_peak_pct=None):
    """
    Cluster residues of a single protein into 3 groups.

    Parameters
    ----------
    attention_matrix : list[list[float]]
    min_peak_pct     : float or None
        If set (0–100), only residues whose column-maximum attention
        (peak weight received from any query) falls in the top
        min_peak_pct percent of all residues in this protein are kept
        in Pattern A or B. The rest are demoted to Background.
        E.g. min_peak_pct=25 → only the top 25% brightest residues
        can be in a pattern group.
        Default None = no threshold, pure K-means assignment.

    Returns
    -------
    dict with keys:
        'background' : list[int]  residue indices (0-based)
        'pattern_A'  : list[int]
        'pattern_B'  : list[int]
        'labels'     : list[int]  raw cluster label per residue (length N)
        'energies'   : list[float] mean attention energy per cluster
    """
    A = attention_matrix
    N = len(A)

    # Build feature vectors: row_i + col_i  (length 2N)
    raw_features = []
    for i in range(N):
        row = A[i][:]                      # query profile
        col = [A[j][i] for j in range(N)] # key profile
        raw_features.append(row + col)

    # L2-normalize so cluster assignment is driven by *shape*, not magnitude
    features = [_normalize(v) for v in raw_features]

    labels, _ = kmeans(features, k=3, n_init=n_init, max_iter=max_iter, seed=seed)

    # Column maximum: the single strongest attention weight received by residue i.
    # A residue in a bright blob has a high peak; diffuse background does not,
    # regardless of how many queries weakly attend to it.
    col_max = [max(A[j][i] for j in range(N)) for i in range(N)]

    # For cluster ranking use row+col sum energy (captures overall flow)
    col_sum = [sum(A[j][i] for j in range(N)) for i in range(N)]
    energy = [
        sum(A[i]) / N + col_sum[i] / N
        for i in range(N)
    ]

    # Identify background cluster = lowest mean energy
    cluster_energy = []
    for c in range(3):
        members = [i for i in range(N) if labels[i] == c]
        mean_e = sum(energy[m] for m in members) / len(members) if members else 0.0
        cluster_energy.append((mean_e, c))

    cluster_energy_sorted = sorted(cluster_energy)          # ascending energy
    bg_cluster  = cluster_energy_sorted[0][1]               # weakest = background
    pa_cluster  = cluster_energy_sorted[2][1]               # strongest = A
    pb_cluster  = cluster_energy_sorted[1][1]               # middle   = B

    background = sorted([i for i in range(N) if labels[i] == bg_cluster])
    pattern_a  = sorted([i for i in range(N) if labels[i] == pa_cluster])
    pattern_b  = sorted([i for i in range(N) if labels[i] == pb_cluster])

    # Optional percentile threshold: demote pattern residues whose peak
    # attention is not in the top min_peak_pct% of this protein to Background.
    # This is protein-agnostic — it adapts to each protein's attention contrast.
    if min_peak_pct is not None:
        sorted_peaks = sorted(col_max)
        cutoff_idx   = int(N * (1.0 - min_peak_pct / 100.0))
        cutoff_idx   = max(0, min(cutoff_idx, N - 1))
        cutoff       = sorted_peaks[cutoff_idx]
        promoted_bg  = [i for i in pattern_a + pattern_b if col_max[i] < cutoff]
        if promoted_bg:
            promoted_set = set(promoted_bg)
            pattern_a  = sorted([i for i in pattern_a  if i not in promoted_set])
            pattern_b  = sorted([i for i in pattern_b  if i not in promoted_set])
            background = sorted(background + promoted_bg)

    return {
        'background': background,
        'pattern_A':  pattern_a,
        'pattern_B':  pattern_b,
        'labels':     labels,
        'energies':   [ce[0] for ce in sorted(cluster_energy, key=lambda x: x[1])],
    }


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Cluster residues by attention pattern into 3 groups.')
    p.add_argument('--input',   required=True, help='Path to attention JSON file')
    p.add_argument('--protein', default=None,
                   help='Single protein name (e.g. 1jo0_A). Omit to process all proteins.')
    p.add_argument('--all',     action='store_true',
                   help='Process every protein in the file (same as omitting --protein).')
    p.add_argument('--n_init',  type=int, default=10,  help='K-means restarts (default 10)')
    p.add_argument('--max_iter',type=int, default=100, help='K-means max iterations (default 100)')
    p.add_argument('--seed',    type=int, default=42,  help='Random seed')
    p.add_argument('--min_peak_pct', type=float, default=None,
                   help='Only keep residues in Pattern A/B if their peak received '
                        'attention (max_j A[j,i]) is in the top N%% of the protein. '
                        'E.g. 25 = top 25%% (brightest quarter) only. '
                        'Protein-agnostic: adapts to each protein\'s contrast. '
                        'Default: no threshold (pure K-means assignment).')
    p.add_argument('--output',  default=None, help='Path to save JSON output')
    p.add_argument('--list_proteins', action='store_true',
                   help='List all available protein names in the file and exit')
    return p.parse_args()


def fmt_residues(indices, seq):
    if not indices:
        return '  (none)'
    return '  ' + '  '.join(
        f"{idx}({seq[idx] if idx < len(seq) else '?'})" for idx in indices
    )


def print_result(protein_name, seq, result):
    N = len(seq)
    print(f"\nProtein : {protein_name}  (length {N})")
    print(f"Sequence: {seq}")
    print()

    for name_str, indices in [
        ('Pattern A  ', result['pattern_A']),
        ('Pattern B  ', result['pattern_B']),
        ('Background ', result['background']),
    ]:
        print(f"{name_str} ({len(indices):3d} residues):")
        print(fmt_residues(indices, seq))
        print()

    cluster_name = {}
    if result['pattern_A']:
        cluster_name[result['labels'][result['pattern_A'][0]]] = 'Pattern A'
    if result['pattern_B']:
        cluster_name[result['labels'][result['pattern_B'][0]]] = 'Pattern B'
    if result['background']:
        cluster_name[result['labels'][result['background'][0]]] = 'Background'

    print("Cluster attention energies (row+col mean):")
    for c in range(3):
        lbl = cluster_name.get(c, f'Cluster {c}')
        print(f"  Cluster {c} ({lbl}): {result['energies'][c]:.6f}")
    print()


def process_record(record, n_init, max_iter, seed, min_peak_pct=None):
    """Cluster a single record and return the output dict."""
    A   = record['attention_weights']
    seq = record.get('sequence', '')
    res = cluster_residues(A, n_init=n_init, max_iter=max_iter, seed=seed,
                           min_peak_pct=min_peak_pct)
    return {
        'protein':    record['name'],
        'length':     len(A),
        'sequence':   seq,
        'pattern_A':  res['pattern_A'],
        'pattern_B':  res['pattern_B'],
        'background': res['background'],
        'labels':     res['labels'],
        'energies': {
            'pattern_A':  res['energies'][res['labels'][res['pattern_A'][0]]]
                          if res['pattern_A'] else None,
            'pattern_B':  res['energies'][res['labels'][res['pattern_B'][0]]]
                          if res['pattern_B'] else None,
            'background': res['energies'][res['labels'][res['background'][0]]]
                          if res['background'] else None,
        },
    }


def main():
    args = parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.list_proteins:
        print('\n'.join(r['name'] for r in data))
        return

    # ── Decide which records to process ──────────────────────────────────────
    run_all = args.all or (args.protein is None)

    if run_all:
        records = [r for r in data if len(r['attention_weights']) >= 20]
    else:
        record = next((r for r in data if r['name'] == args.protein), None)
        if record is None:
            print(f"ERROR: protein '{args.protein}' not found in {args.input}",
                  file=sys.stderr)
            print(f"Available: {[r['name'] for r in data][:10]} ...", file=sys.stderr)
            sys.exit(1)
        records = [record]

    # ── Process ───────────────────────────────────────────────────────────────
    all_results = []
    for i, rec in enumerate(records):
        if run_all:
            print(f"[{i+1}/{len(records)}] {rec['name']} ...", end='  ', flush=True)
        out = process_record(rec, args.n_init, args.max_iter, args.seed,
                              min_peak_pct=args.min_peak_pct)
        all_results.append(out)
        if run_all:
            print(f"A={len(out['pattern_A'])}  B={len(out['pattern_B'])}  "
                  f"bg={len(out['background'])}")
        else:
            # Single-protein: print full formatted output to stdout
            res = cluster_residues(rec['attention_weights'],
                                   n_init=args.n_init, max_iter=args.max_iter,
                                   seed=args.seed,
                                   min_peak_pct=args.min_peak_pct)
            print_result(rec['name'], rec.get('sequence', ''), res)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output:
        # Always save as a list (even for a single protein) for consistency
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} record(s) to {args.output}")
    elif run_all and not args.output:
        print("\n(Use --output <path> to save results to a JSON file.)")


if __name__ == '__main__':
    main()
