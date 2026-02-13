# MDCATH Classification with Cluster-Aware Splitting

## Overview

This document describes the workflow for running classification experiments on the mdcath dataset (or any protein dataset) while preventing data leakage through cluster-aware data splitting.

## Problem

When working with protein sequences, similar proteins can lead to data leakage if they are split between training and test sets. This is because models can learn to recognize sequence similarity rather than the underlying biological properties.

## Solution

We use MMseqs2 to cluster similar proteins together, then ensure entire clusters are assigned to the same data split (train, validation, or test). This prevents data leakage while allowing comprehensive evaluation.

## Workflow

### Prerequisites

1. **MMseqs2 Installation**
   ```bash
   # Via conda
   conda install -c bioconda mmseqs2
   
   # Or via homebrew (macOS)
   brew install mmseqs2
   
   # Or download from: https://github.com/soedinglab/mmseqs2
   ```

2. **Python Dependencies**
   All required Python packages are already in `requirements.txt`.

### Step-by-Step Guide

#### Step 1: Cluster Sequences

Cluster protein sequences based on sequence similarity:

```bash
python cluster_sequences.py \
    --input ../../data/train_data.csv \
    --output clustered_data.csv \
    --min_seq_id 0.3 \
    --coverage 0.8
```

**Parameters:**
- `--min_seq_id`: Minimum sequence identity threshold (default: 0.3 = 30%)
- `--coverage`: Minimum alignment coverage (default: 0.8 = 80%)

**Output:**
A CSV file with all original columns plus:
- `cluster_id`: Representative sequence ID for the cluster
- `cluster_number`: Integer cluster ID (0, 1, 2, ...)

#### Step 2: Cluster-Aware Data Splitting

Split the clustered data while keeping clusters intact:

```bash
python cluster_aware_split.py \
    --input clustered_data.csv \
    --train_output train_data.csv \
    --test_output test_data.csv \
    --test_size 0.2 \
    --seed 42
```

**Parameters:**
- `--test_size`: Fraction of data for testing (default: 0.2)
- `--val_size`: Optional validation split (default: 0.0)
- `--val_output`: Output file for validation data (required if val_size > 0)
- `--seed`: Random seed for reproducibility

**Output:**
- Training data CSV
- Test data CSV
- Optional validation data CSV

The script verifies that no clusters appear in multiple splits.

#### Step 3: Run Temperature Experiments

Run classification experiments at different "temperatures" (neq_thresholds):

```bash
python run_temperature_experiments.py \
    --train_data train_data.csv \
    --test_data test_data.csv \
    --output_dir ./temperature_experiments \
    --esm_model esm2_t12_35M_UR50D \
    --epochs 20 \
    --batch_size 4 \
    --device cuda
```

**Default Temperature Configurations:**

1. **Binary (Low Temp)**: 2 classes, threshold at 1.0
2. **Binary (Mid Temp)**: 2 classes, threshold at 1.5
3. **Binary (High Temp)**: 2 classes, threshold at 2.0
4. **3-Class (Low)**: 3 classes, thresholds [1.0, 1.5]
5. **3-Class (Mid)**: 3 classes, thresholds [1.0, 2.0]
6. **4-Class (Standard)**: 4 classes, thresholds [1.0, 2.0, 4.0]
7. **4-Class (Fine)**: 4 classes, thresholds [1.0, 1.5, 2.5]
8. **5-Class**: 5 classes, thresholds [1.0, 1.5, 2.0, 3.0]

**Custom Configuration:**

Create a JSON file with your own configurations:

```json
[
  {
    "name": "custom_binary",
    "description": "Custom binary classification",
    "num_classes": 2,
    "neq_thresholds": [1.5],
    "architecture": "bilstm",
    "hidden_size": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "loss_function": "focal",
    "oversampling": true
  }
]
```

Then run:
```bash
python run_temperature_experiments.py \
    --train_data train_data.csv \
    --test_data test_data.csv \
    --config custom_config.json \
    --output_dir ./custom_experiments
```

### Complete Pipeline (One Command)

Run the entire pipeline with a single script:

```bash
bash run_mdcath_pipeline.sh \
    ../../data/train_data.csv \
    ./mdcath_experiments \
    0.3 \
    0.8 \
    0.2 \
    42
```

**Arguments:**
1. Input data CSV (default: ../../data/train_data.csv)
2. Output directory (default: ./mdcath_experiments)
3. Min sequence identity (default: 0.3)
4. Coverage (default: 0.8)
5. Test size (default: 0.2)
6. Random seed (default: 42)

## Output Structure

```
mdcath_experiments/
├── clustered_data.csv              # Sequences with cluster assignments
├── train_data.csv                  # Training split
├── test_data.csv                   # Test split
└── experiments/
    ├── experiment_summary.json     # Summary of all experiments
    ├── binary_low_temp/
    │   ├── experiment_config.json
    │   ├── training_log.txt
    │   └── [model checkpoints and results]
    ├── binary_mid_temp/
    ├── ...
    └── five_class/
```

## Understanding Clustering Parameters

### Minimum Sequence Identity (`--min_seq_id`)

- **0.3 (30%)**: Clusters proteins with ≥30% sequence identity
  - Good for preventing leakage from homologous proteins
  - Recommended starting point
  
- **0.5 (50%)**: Clusters proteins with ≥50% sequence identity
  - More conservative, keeps more distant homologs in different splits
  
- **0.7 (70%)**: Clusters very similar proteins
  - Very strict, may allow some homology-based leakage

### Coverage (`--coverage`)

- **0.8 (80%)**: The alignment must cover at least 80% of the sequence length
  - Prevents clustering based on small local similarities
  - Recommended for full-length proteins

## Best Practices

1. **Choose appropriate clustering parameters:**
   - For highly diverse datasets: `--min_seq_id 0.5`
   - For datasets with many homologs: `--min_seq_id 0.3`
   
2. **Verify cluster distribution:**
   - Check cluster size distribution after clustering
   - Ensure no single cluster dominates the dataset
   
3. **Document your splits:**
   - Save cluster assignments for reproducibility
   - Use the same random seed for consistent results
   
4. **Monitor for data imbalance:**
   - Cluster-aware splitting may create class imbalance
   - Use `--oversampling` flag if needed

## Troubleshooting

### MMseqs2 not found
```bash
# Install via conda
conda install -c bioconda mmseqs2
```

### Out of memory during clustering
```bash
# Reduce the number of sequences or use a machine with more RAM
# MMseqs2 requires significant memory for large datasets
```

### Cluster sizes too large
```bash
# Reduce min_seq_id threshold
python cluster_sequences.py --min_seq_id 0.2 ...
```

### Unbalanced splits
```bash
# Adjust test_size or use stratification (experimental)
python cluster_aware_split.py --stratify ...
```

## References

- MMseqs2: https://github.com/soedinglab/mmseqs2
- Steinegger M, Söding J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology, 2017.

## Contact

For questions or issues, please open an issue on GitHub.
