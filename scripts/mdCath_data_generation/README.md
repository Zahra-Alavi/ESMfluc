# mdCATH Data Processing

Downloads MD trajectories from mdCATH and calculates per-residue Neq values.

## Files

- `main.py` - Downloads H5 files, converts to XTC, calculates Neq
- `convert_mdCath.py` - H5 conversion utilities from mdCATH repo
- `data_splitting.py` - Random train/test split by PDB ID
- `data_splitting_mmseqs2.py` - MMseqs2 clustering-based split (recommended)

## Setup

```bash
pip install pandas numpy requests pbxplore MDAnalysis
brew install mmseqs2  # for clustering-based split
```

## Generate Dataset

```bash
python main.py --domain_list_file all_domains.txt
```

Output: `mdcath_neq_dataset.csv`
- 5 temperatures: 320K, 348K, 379K, 413K, 450K
- 5 replicas per temperature (averaged together)
- ~5300 domains processed

## Split Data

### Option 1: MMseqs2 clustering (recommended)

```bash
python data_splitting_mmseqs2.py
```

Clusters sequences at 30% identity, then splits clusters into train/test. Prevents similar sequences from appearing in both sets.

Output: `train_split_mmseqs2.csv`, `test_split_mmseqs2.csv`

Options:
- `--seq_id_threshold 0.3` - clustering threshold (lower = stricter)
- `--test_size 0.20` - fraction for test set
- `--random_state 42` - random seed

### Option 2: Simple PDB grouping

```bash
python data_splitting.py
```

Groups by first 4 characters of domain ID (PDB code), then splits randomly.

Output: `train_split.csv`, `test_split.csv`

## Notes

- Dataset has 5,347 valid domains (20 failed during processing)
- Sequences are 50-500 residues long
- NaN domains are filtered out automatically during splitting
- MMseqs2 found ~5,300 clusters (most sequences are unique)
