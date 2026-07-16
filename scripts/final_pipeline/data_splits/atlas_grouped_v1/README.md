# ATLAS grouped split v1

These files do not replace the legacy `../../data/train_data.csv` or test files.
They are a separately named, fixed 70/15/15 split for new publication runs.

## Membership rule

The indivisible unit is a connected component of the union of exact-sequence
groups, MMseqs2 hits at >=30% identity over >=80% of both sequences, and shared
ECOD develop285 X-groups. Components were assigned with split-construction seed
42. Model seeds 1/2/3 never participate in split construction.

ECOD X-groups are broad (possible homology), and transitive unioning produces a
largest component of 386 proteins. It is kept intact; this is the
cost of the requested no-shared-X-domain condition.

## Training files

- `train_grouped_v1.csv`: 967 proteins
- `validation_grouped_v1.csv`: 208 proteins
- `test_grouped_v1.csv`: 208 proteins
- `test_grouped_v1.fasta`: named test sequences for attention extraction

`split_manifest_grouped_v1.csv` records every group ID. The JSON summary records
parameters, checksums, and leakage tests. `mmseqs/all_vs_all.tsv`, the ECOD subset,
and original Neq tables for ambiguous duplicate sequences provide the audit trail.
All reported cross-split leakage counts are zero.

## Reproduction

```bash
python3 -m pip install remotezip
python3 fetch_atlas_duplicate_neq.py --output-dir /tmp/atlas_duplicate_neq
python3 create_atlas_grouped_splits.py \
  --ecod-domains /path/to/ecod.develop285.domains.txt \
  --duplicate-neq-dir /tmp/atlas_duplicate_neq \
  --mmseqs-bin /path/to/mmseqs \
  --output-dir data_splits/atlas_grouped_v1
```

The generator refuses to overwrite a non-empty output directory.
