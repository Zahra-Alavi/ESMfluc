# Quick Reference: MDCATH Workflow

## One-Line Commands

### Complete Pipeline (Recommended)
```bash
bash run_mdcath_pipeline.sh ../../data/train_data.csv ./mdcath_experiments 0.3 0.8 0.2 42
```

### Individual Steps

**1. Cluster Sequences (requires MMseqs2)**
```bash
python cluster_sequences.py \
    --input ../../data/train_data.csv \
    --output clustered_data.csv \
    --min_seq_id 0.3 \
    --coverage 0.8
```

**2. Cluster-Aware Split**
```bash
python cluster_aware_split.py \
    --input clustered_data.csv \
    --train_output train_data.csv \
    --test_output test_data.csv \
    --test_size 0.2 \
    --seed 42
```

**3. Run Temperature Experiments**
```bash
# Using default configurations
python run_temperature_experiments.py \
    --train_data train_data.csv \
    --test_data test_data.csv \
    --output_dir ./temperature_experiments

# Using custom configurations
python run_temperature_experiments.py \
    --train_data train_data.csv \
    --test_data test_data.csv \
    --config example_temperature_config.json \
    --output_dir ./custom_experiments
```

## Common Use Cases

### Different Clustering Thresholds

**Conservative (50% identity)**
```bash
python cluster_sequences.py --input data.csv --output clustered.csv --min_seq_id 0.5
```

**Aggressive (30% identity)**
```bash
python cluster_sequences.py --input data.csv --output clustered.csv --min_seq_id 0.3
```

### Three-Way Split (Train/Val/Test)
```bash
python cluster_aware_split.py \
    --input clustered_data.csv \
    --train_output train.csv \
    --val_output val.csv \
    --test_output test.csv \
    --test_size 0.2 \
    --val_size 0.1
```

### Custom Temperature Ranges

Create `custom_temps.json`:
```json
[
  {
    "name": "low_temp",
    "num_classes": 2,
    "neq_thresholds": [1.0]
  },
  {
    "name": "mid_temp",
    "num_classes": 2,
    "neq_thresholds": [2.0]
  },
  {
    "name": "high_temp",
    "num_classes": 2,
    "neq_thresholds": [3.0]
  }
]
```

Then run:
```bash
python run_temperature_experiments.py \
    --train_data train.csv \
    --test_data test.csv \
    --config custom_temps.json
```

## Testing

**Test cluster-aware splitting (no MMseqs2 required)**
```bash
python test_cluster_split.py
```

## Troubleshooting

### MMseqs2 not installed
```bash
# Install via conda
conda install -c bioconda mmseqs2

# Or use alternative workflow (manual clustering)
# 1. Use external tool for clustering
# 2. Add 'cluster_number' column to your CSV
# 3. Skip step 1, start with step 2
```

### Out of memory
```bash
# Reduce batch size
python run_temperature_experiments.py --batch_size 2 ...

# Or use smaller ESM model
python run_temperature_experiments.py --esm_model esm2_t6_8M_UR50D ...
```

### Check experiment status
```bash
# View summary
cat temperature_experiments/experiment_summary.json

# View specific experiment log
cat temperature_experiments/binary_low_temp/training_log.txt
```

## File Structure

```
mdcath_experiments/
├── clustered_data.csv           # Sequences with cluster assignments
├── train_data.csv               # Training split (cluster-aware)
├── test_data.csv                # Test split (cluster-aware)
└── experiments/
    ├── experiment_summary.json  # All experiment results
    ├── binary_low_temp/
    │   ├── experiment_config.json
    │   ├── training_log.txt
    │   └── [model files]
    └── [other experiments]/
```

## Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `--min_seq_id` | 0.2-0.9 | Sequence identity threshold (lower = more clustering) |
| `--coverage` | 0.5-1.0 | Alignment coverage (higher = stricter) |
| `--test_size` | 0.1-0.3 | Fraction for test set |
| `--neq_thresholds` | [1.0], [1.5], [2.0] | Temperature boundaries |
| `--num_classes` | 2-5 | Classification granularity |

## Example Workflows

### Quick Test (Small Dataset)
```bash
# Use existing train_data.csv without clustering
python run_temperature_experiments.py \
    --train_data ../../data/train_data.csv \
    --test_data ../../data/test_data.csv \
    --epochs 5 \
    --batch_size 8
```

### Production Run (Full Pipeline)
```bash
# 1. Cluster at 30% identity
python cluster_sequences.py \
    --input full_dataset.csv \
    --output clustered.csv \
    --min_seq_id 0.3

# 2. Split 80/20
python cluster_aware_split.py \
    --input clustered.csv \
    --train_output train.csv \
    --test_output test.csv \
    --test_size 0.2

# 3. Run comprehensive experiments
python run_temperature_experiments.py \
    --train_data train.csv \
    --test_data test.csv \
    --esm_model esm2_t33_650M_UR50D \
    --epochs 50 \
    --batch_size 2
```
