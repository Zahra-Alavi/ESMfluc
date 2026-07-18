#!/usr/bin/env python3
"""Create one fixed, leakage-audited ATLAS train/validation/test split.

The indivisible split unit is the connected component formed by the union of:
  * exact sequence identity;
  * MMseqs2 sequence-similarity edges; and
  * shared ECOD X-groups.

This script never overwrites the historical train_data.csv/test_data.csv files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_neq(value) -> list[float]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [float(item) for item in parsed]


def load_ecod_x_groups(path: Path, atlas_ids: set[str]):
    annotations = defaultdict(set)
    domain_rows = []
    normalized_ids = {atlas_id.lower(): atlas_id for atlas_id in atlas_ids}
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                raise ValueError(f"Malformed ECOD row with {len(fields)} columns")
            normalized_id = f"{fields[4]}_{fields[5]}".lower()
            if normalized_id not in normalized_ids:
                continue
            atlas_id = normalized_ids[normalized_id]
            # ECOD f_id is X.H.T.F; its first field is the stable X-group ID.
            x_id = f"ECOD_X_{fields[3].split('.')[0]}"
            annotations[atlas_id].add(x_id)
            domain_rows.append(
                {
                    "name": atlas_id,
                    "ecod_domain_id": fields[1],
                    "ecod_f_id": fields[3],
                    "ecod_x_id": x_id,
                    "ecod_x_name": fields[10].strip('"'),
                    "pdb_range": fields[6],
                }
            )
    missing = sorted(atlas_ids - annotations.keys())
    if missing:
        raise ValueError(f"No ECOD v285 annotation for {len(missing)} ATLAS entries: {missing[:10]}")
    return annotations, pd.DataFrame(domain_rows)


def load_duplicate_neq_tables(directory: Path | None):
    result = {}
    if directory is None:
        return result
    for path in directory.glob("*_Neq.tsv"):
        name = path.name[: -len("_Neq.tsv")].lower()
        table = pd.read_csv(path, sep="\t")
        required = {"Neq_R1", "Neq_R2", "Neq_R3"}
        if not required.issubset(table.columns):
            raise ValueError(f"{path} lacks {sorted(required - set(table.columns))}")
        result[name] = table[["Neq_R1", "Neq_R2", "Neq_R3"]].mean(axis=1).to_numpy()
    return result


def attach_names_and_labels(atlas_path: Path, neq_path: Path, duplicate_neq_dir: Path | None):
    atlas = pd.read_csv(atlas_path).rename(columns={"seqres": "sequence"})
    labels = pd.read_csv(neq_path)
    if not {"name", "sequence"}.issubset(atlas.columns):
        raise ValueError("ATLAS CSV must contain name and seqres/sequence")
    if not {"sequence", "neq"}.issubset(labels.columns):
        raise ValueError("Neq CSV must contain sequence and neq")

    atlas["sequence"] = atlas["sequence"].str.upper()
    labels["sequence"] = labels["sequence"].str.upper()
    labels["neq_values"] = labels["neq"].map(parse_neq)
    by_sequence = {sequence: group.copy() for sequence, group in labels.groupby("sequence", sort=False)}
    raw_duplicates = load_duplicate_neq_tables(duplicate_neq_dir)

    output = []
    for sequence, atlas_group in atlas.groupby("sequence", sort=False):
        if sequence not in by_sequence:
            raise ValueError(f"ATLAS sequence missing from Neq CSV: {atlas_group.name.tolist()}")
        label_group = by_sequence[sequence]
        if len(atlas_group) != len(label_group):
            raise ValueError(f"Multiplicity mismatch for sequence shared by {atlas_group.name.tolist()}")

        available = list(label_group.index)
        for _, atlas_row in atlas_group.sort_values("name").iterrows():
            mapping_method = "unique_sequence"
            if len(atlas_group) == 1:
                chosen = available[0]
            else:
                expected = raw_duplicates.get(atlas_row["name"].lower())
                if expected is None:
                    raise ValueError(
                        f"Ambiguous duplicate {atlas_row['name']}; supply its original *_Neq.tsv "
                        "through --duplicate-neq-dir"
                    )
                matches = []
                for candidate in available:
                    observed = np.asarray(label_group.loc[candidate, "neq_values"], dtype=float)
                    if len(observed) == len(expected) and np.allclose(observed, expected, atol=1e-12, rtol=0):
                        matches.append(candidate)
                if len(matches) != 1:
                    raise ValueError(f"Expected one label match for {atlas_row['name']}, found {len(matches)}")
                chosen = matches[0]
                mapping_method = "matched_original_atlas_neq"
            available.remove(chosen)
            neq_values = label_group.loc[chosen, "neq_values"]
            if len(neq_values) != len(sequence):
                raise ValueError(f"Length mismatch for {atlas_row['name']}: {len(sequence)} vs {len(neq_values)}")
            output.append(
                {
                    "name": atlas_row["name"],
                    "sequence": sequence,
                    "neq": str(neq_values),
                    "neq_values": neq_values,
                    "label_mapping_method": mapping_method,
                    "source_neq_row": int(chosen),
                }
            )
    return pd.DataFrame(output)


def run_mmseqs_all_vs_all(df, mmseqs_bin: str, work_dir: Path, min_seq_id: float, coverage: float):
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "atlas.fasta"
    hits = work_dir / "all_vs_all.tsv"
    with fasta.open("w") as handle:
        for row in df.itertuples():
            handle.write(f">{row.name}\n{row.sequence}\n")
    command = [
        mmseqs_bin,
        "easy-search",
        str(fasta),
        str(fasta),
        str(hits),
        str(work_dir / "tmp"),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--alignment-mode", "3",
        "-s", "7.5",
        "--max-seqs", "10000",
        "--threads", "1",
        "--format-output", "query,target,fident,qcov,tcov,evalue",
    ]
    subprocess.run(command, check=True)
    hit_df = pd.read_csv(
        hits,
        sep="\t",
        names=["query", "target", "fident", "qcov", "tcov", "evalue"],
    )
    return hit_df, command


def component_ids(values, union_find: UnionFind, prefix: str):
    members = defaultdict(list)
    for value in values:
        members[union_find.find(value)].append(value)
    ordered = sorted((sorted(group) for group in members.values()), key=lambda group: group[0])
    mapping = {}
    for index, group in enumerate(ordered, start=1):
        group_id = f"{prefix}_{index:04d}"
        for value in group:
            mapping[value] = group_id
    return mapping


def choose_splits(component_stats, fractions, seed: int, attempts: int = 2000):
    split_names = list(fractions)
    feature_columns = ["n_sequences", "n_residues", "class_0", "class_1", "class_2", "class_3"]
    matrix = component_stats[feature_columns].to_numpy(dtype=float)
    targets = np.outer(np.asarray([fractions[name] for name in split_names]), matrix.sum(axis=0))
    scale = np.maximum(targets, 1.0)
    rng = np.random.default_rng(seed)
    best_score, best_assignment = math.inf, None

    sizes = component_stats["n_sequences"].to_numpy()
    for attempt in range(attempts):
        if attempt == 0:
            order = np.argsort(-sizes, kind="stable")
        else:
            jitter = rng.uniform(0.85, 1.15, len(sizes))
            order = np.argsort(-(sizes * jitter), kind="stable")
        totals = np.zeros_like(targets)
        assignment = np.full(len(sizes), -1, dtype=int)
        for row_index in order:
            costs = []
            for split_index in range(len(split_names)):
                candidate = totals.copy()
                candidate[split_index] += matrix[row_index]
                relative_error = (candidate - targets) / scale
                costs.append(float(np.sum(relative_error**2)))
            minimum = min(costs)
            choices = [i for i, cost in enumerate(costs) if abs(cost - minimum) < 1e-12]
            selected = choices[0] if attempt == 0 else int(rng.choice(choices))
            assignment[row_index] = selected
            totals[selected] += matrix[row_index]
        final_score = float(np.sum(((totals - targets) / scale) ** 2))
        if final_score < best_score:
            best_score, best_assignment = final_score, assignment.copy()

    if best_assignment is None or set(best_assignment) != set(range(len(split_names))):
        raise RuntimeError("Unable to construct three non-empty component-level splits")
    return {
        component_stats.iloc[index]["union_group_id"]: split_names[split_index]
        for index, split_index in enumerate(best_assignment)
    }, best_score


def validate_no_leakage(manifest: pd.DataFrame, hits: pd.DataFrame):
    name_to_split = manifest.set_index("name")["split"].to_dict()
    failures = {
        "exact_sequence": int(manifest.groupby("exact_sequence_group_id")["split"].nunique().gt(1).sum()),
        "mmseqs_cluster": int(manifest.groupby("mmseqs_cluster_id")["split"].nunique().gt(1).sum()),
        "union_group": int(manifest.groupby("union_group_id")["split"].nunique().gt(1).sum()),
    }
    ecod_splits = defaultdict(set)
    for row in manifest.itertuples():
        for x_id in row.ecod_x_ids.split(";"):
            ecod_splits[x_id].add(row.split)
    failures["ecod_x_group"] = sum(len(splits) > 1 for splits in ecod_splits.values())
    cross_hits = hits[
        hits.apply(lambda row: name_to_split[row["query"]] != name_to_split[row["target"]], axis=1)
    ]
    failures["mmseqs_cross_split_edges"] = int(len(cross_hits))
    if any(failures.values()):
        raise RuntimeError(f"Leakage validation failed: {failures}")
    return failures


def write_readme(output_dir: Path, summary: dict):
    counts = summary["counts"]
    text = f"""# ATLAS grouped split v1

These files do not replace the legacy `../../data/train_data.csv` or test files.
They are a separately named, fixed 70/15/15 split for publication experiments.

## Membership rule

The indivisible unit is a connected component of the union of exact-sequence
groups, MMseqs2 hits at >=30% identity over >=80% of both sequences, and shared
ECOD develop285 X-groups. Components were assigned with split-construction seed
42. Model seeds 1/2/3 never participate in split construction.

ECOD X-groups are broad (possible homology), and transitive unioning produces a
largest component of {summary['largest_union_group']} proteins. It remains intact to prevent
ECOD X-groups from crossing split boundaries.

## Training files

- `train_grouped_v1.csv`: {counts['train']} proteins
- `validation_grouped_v1.csv`: {counts['validation']} proteins
- `test_grouped_v1.csv`: {counts['test']} proteins
- `test_grouped_v1.fasta`: named test sequences for attention extraction

`split_manifest_grouped_v1.csv` records every group ID. The JSON summary records
parameters, checksums, and leakage tests. `mmseqs/all_vs_all.tsv`, the ECOD subset,
and original Neq tables for ambiguous duplicate sequences provide the audit trail.
All reported cross-split leakage counts are zero.

## Reproduction

```bash
python3 -m pip install remotezip
python3 fetch_atlas_duplicate_neq.py --output-dir /tmp/atlas_duplicate_neq
python3 create_atlas_grouped_splits.py \\
  --ecod-domains /path/to/ecod.develop285.domains.txt \\
  --duplicate-neq-dir /tmp/atlas_duplicate_neq \\
  --mmseqs-bin /path/to/mmseqs \\
  --output-dir data_splits/atlas_grouped_v1
```

The generator refuses to overwrite a non-empty output directory.
"""
    (output_dir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-csv", type=Path, default=Path("../../data/atlas.csv"))
    parser.add_argument("--neq-csv", type=Path, default=Path("../../data/neq_original_data.csv"))
    parser.add_argument("--ecod-domains", type=Path, required=True)
    parser.add_argument("--duplicate-neq-dir", type=Path, required=True)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--output-dir", type=Path, default=Path("data_splits/atlas_grouped_v1"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    duplicate_source_dir = args.output_dir / "source_duplicate_neq"
    duplicate_source_dir.mkdir()
    duplicate_files = sorted(args.duplicate_neq_dir.glob("*_Neq.tsv"))
    for path in duplicate_files:
        shutil.copy2(path, duplicate_source_dir / path.name)

    all_data = attach_names_and_labels(args.atlas_csv, args.neq_csv, args.duplicate_neq_dir)
    excluded = all_data[all_data.sequence.str.len() > args.max_length].copy()
    data = all_data[all_data.sequence.str.len() <= args.max_length].copy().reset_index(drop=True)
    atlas_ids = set(data.name)

    ecod, ecod_rows = load_ecod_x_groups(args.ecod_domains, atlas_ids)
    hits, mmseqs_command = run_mmseqs_all_vs_all(
        data, args.mmseqs_bin, args.output_dir / "mmseqs", args.min_seq_id, args.coverage
    )

    exact_uf = UnionFind(atlas_ids)
    for _, group in data.groupby("sequence"):
        anchor = group.name.iloc[0]
        for member in group.name.iloc[1:]:
            exact_uf.union(anchor, member)
    exact_ids = component_ids(atlas_ids, exact_uf, "exact")

    mmseqs_uf = UnionFind(atlas_ids)
    for row in hits.itertuples():
        mmseqs_uf.union(row.query, row.target)
    mmseqs_ids = component_ids(atlas_ids, mmseqs_uf, "mmseqs")

    union_uf = UnionFind(atlas_ids)
    for group_id in set(exact_ids.values()):
        members = [name for name, value in exact_ids.items() if value == group_id]
        for member in members[1:]:
            union_uf.union(members[0], member)
    for group_id in set(mmseqs_ids.values()):
        members = [name for name, value in mmseqs_ids.items() if value == group_id]
        for member in members[1:]:
            union_uf.union(members[0], member)
    by_x = defaultdict(list)
    for name, x_ids in ecod.items():
        for x_id in x_ids:
            by_x[x_id].append(name)
    for members in by_x.values():
        for member in members[1:]:
            union_uf.union(members[0], member)
    union_ids = component_ids(atlas_ids, union_uf, "union")

    data["sequence_sha256"] = data.sequence.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    data["exact_sequence_group_id"] = data.name.map(exact_ids)
    data["mmseqs_cluster_id"] = data.name.map(mmseqs_ids)
    data["ecod_x_ids"] = data.name.map(lambda name: ";".join(sorted(ecod[name])))
    data["union_group_id"] = data.name.map(union_ids)
    data["length"] = data.sequence.str.len()
    classifiers = (
        lambda value: value <= 1,
        lambda value: 1 < value <= 2,
        lambda value: 2 < value <= 4,
        lambda value: value > 4,
    )
    for class_id, classifier in enumerate(classifiers):
        data[f"class_{class_id}"] = data.neq_values.map(
            lambda values, classify=classifier: sum(classify(value) for value in values)
        )

    component_stats = (
        data.groupby("union_group_id", as_index=False)
        .agg(
            n_sequences=("name", "size"),
            n_residues=("length", "sum"),
            class_0=("class_0", "sum"),
            class_1=("class_1", "sum"),
            class_2=("class_2", "sum"),
            class_3=("class_3", "sum"),
        )
    )
    split_by_group, balance_score = choose_splits(component_stats, fractions, args.split_seed)
    data["split"] = data.union_group_id.map(split_by_group)

    manifest_columns = [
        "name", "split", "sequence_sha256", "length", "source_neq_row",
        "label_mapping_method", "exact_sequence_group_id", "mmseqs_cluster_id",
        "ecod_x_ids", "union_group_id",
    ]
    manifest = data[manifest_columns].sort_values(["split", "union_group_id", "name"])
    leakage = validate_no_leakage(manifest, hits)

    base_columns = ["name", "sequence", "neq"]
    output_names = {
        "train": "train_grouped_v1.csv",
        "validation": "validation_grouped_v1.csv",
        "test": "test_grouped_v1.csv",
    }
    for split, filename in output_names.items():
        split_data = data[data.split == split].sort_values("name")
        split_data[base_columns].to_csv(args.output_dir / filename, index=False)
        with (args.output_dir / f"{split}_grouped_v1.fasta").open("w") as handle:
            for row in split_data.itertuples():
                handle.write(f">{row.name}\n{row.sequence}\n")

    manifest.to_csv(args.output_dir / "split_manifest_grouped_v1.csv", index=False)
    component_stats.assign(split=component_stats.union_group_id.map(split_by_group)).to_csv(
        args.output_dir / "group_manifest_grouped_v1.csv", index=False
    )
    ecod_rows.sort_values(["name", "ecod_domain_id"]).to_csv(
        args.output_dir / "ecod_v285_annotations.csv", index=False
    )
    excluded.assign(exclusion_reason=f"sequence_longer_than_{args.max_length}").drop(
        columns=["neq_values"]
    ).to_csv(args.output_dir / "excluded_entries.csv", index=False)

    version_output = subprocess.run([args.mmseqs_bin, "version"], check=True, capture_output=True, text=True)
    audited_outputs = [
        "train_grouped_v1.csv", "validation_grouped_v1.csv", "test_grouped_v1.csv",
        "split_manifest_grouped_v1.csv", "group_manifest_grouped_v1.csv",
        "ecod_v285_annotations.csv", "excluded_entries.csv", "mmseqs/all_vs_all.tsv",
    ]
    summary = {
        "dataset_version": "atlas_grouped_v1",
        "split_seed": args.split_seed,
        "split_seed_role": "split construction only; training seeds do not affect membership",
        "fractions_requested": fractions,
        "counts": data.split.value_counts().sort_index().to_dict(),
        "residue_counts": data.groupby("split").length.sum().sort_index().to_dict(),
        "n_union_groups": int(data.union_group_id.nunique()),
        "largest_union_group": int(data.groupby("union_group_id").size().max()),
        "n_ecod_x_groups": len(by_x),
        "n_mmseqs_qualifying_directed_hits": int(len(hits)),
        "ecod_coverage": f"{len(atlas_ids)}/{len(atlas_ids)}",
        "excluded_count": len(excluded),
        "balance_score": balance_score,
        "leakage_failures": leakage,
        "mmseqs": {
            "version": version_output.stdout.strip() or version_output.stderr.strip(),
            "command": [
                "mmseqs" if item == args.mmseqs_bin
                else item.replace(str(args.output_dir), "<output_dir>")
                for item in mmseqs_command
            ],
            "min_sequence_identity": args.min_seq_id,
            "minimum_coverage_of_both_sequences": args.coverage,
            "sensitivity": 7.5,
        },
        "ecod": {
            "version": "develop285",
            "source_url": "http://prodata.swmed.edu/ecod/distributions/ecod.develop285.domains.txt",
            "sha256": sha256_file(args.ecod_domains),
        },
        "inputs": {
            "atlas_csv": {"path": str(args.atlas_csv), "sha256": sha256_file(args.atlas_csv)},
            "neq_csv": {"path": str(args.neq_csv), "sha256": sha256_file(args.neq_csv)},
            "duplicate_neq_tables": {
                path.name: sha256_file(path) for path in duplicate_files
            },
        },
        "output_sha256": {
            relative_path: sha256_file(args.output_dir / relative_path)
            for relative_path in audited_outputs
        },
    }
    with (args.output_dir / "split_summary_grouped_v1.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_readme(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
