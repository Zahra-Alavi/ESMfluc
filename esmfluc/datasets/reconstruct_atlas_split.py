"""Reconstruct fixed split inputs from ATLAS Neq tables, checking sequence hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def reconstruct(manifest: Path, table_dir: Path, output: Path, *, fetch=False, names=None):
    fixed = pd.read_csv(manifest)
    if fixed.name.duplicated().any():
        raise ValueError('Duplicate identifiers in split manifest')
    if names:
        missing = set(names) - set(fixed.name)
        if missing:
            raise ValueError(f'Unknown split identifiers: {sorted(missing)}')
        fixed = fixed[fixed.name.isin(names)]
    records, provenance = [], []
    for row in fixed.itertuples(index=False):
        path = table_dir / f'{row.name}_Neq.tsv'
        if not path.is_file():
            if not fetch:
                raise FileNotFoundError(f'{path}; retrieve the ATLAS analysis table or pass --fetch')
            from remotezip import RemoteZip
            url = f'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{row.name}/{row.name}_analysis.zip'
            with RemoteZip(url) as archive:
                raw = archive.read(path.name)
            table_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        table = pd.read_csv(path, sep='\t')
        required = ['seq', 'Neq_R1', 'Neq_R2', 'Neq_R3']
        if not set(required).issubset(table):
            raise ValueError(f'{path}: missing Neq-table columns')
        sequence = ''.join(table.seq.astype(str)).upper()
        if len(sequence) != row.length or hashlib.sha256(sequence.encode()).hexdigest() != row.sequence_sha256:
            raise ValueError(f'{row.name}: sequence differs from the frozen split')
        values = table[required[1:]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 1).any():
            raise ValueError(f'{row.name}: missing or invalid replicate targets')
        neq = table[required[1:]].mean(axis=1).tolist()
        records.append(dict(name=row.name, split=row.split, sequence=sequence, neq=str(neq)))
        provenance.append(dict(name=row.name, table=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    for split, group in pd.DataFrame(records).groupby('split'):
        group = group.sort_values('name')
        destination = output/f'{split}_grouped_v1.csv'
        group[['name','sequence','neq']].to_csv(destination, index=False)
        (output/f'{split}_grouped_v1.fasta').write_text(''.join(
            f'>{row.name}\n{row.sequence}\n' for row in group.itertuples(index=False)))
        checks[split] = dict(proteins=len(group), residues=int(group.sequence.str.len().sum()),
            csv_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    audit = dict(subset_only=bool(names), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        target='Arithmetic mean of three replicate-specific Neq values; no redefinition or threshold rounding.',
        split_counts=checks, inputs=provenance,
        note='A fresh upstream retrieval can differ from the original input; compare the frozen split summary hashes before claiming byte-identical reproduction.')
    (output/'retrieval_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'data/atlas_grouped_v1/split_manifest_grouped_v1.csv')
    p.add_argument('--table-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fetch', action='store_true', help='Retrieve missing tables from ATLAS analysis archives.')
    p.add_argument('--names', nargs='+', help='Explicit small-input check; not a complete fixed split.')
    args = p.parse_args()
    audit = reconstruct(args.manifest, args.table_dir, args.output_dir, fetch=args.fetch, names=args.names)
    print(json.dumps(audit['split_counts'], indent=2))


if __name__ == '__main__':
    main()
