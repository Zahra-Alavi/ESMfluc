# MDCATH Temperature Experiments

Orchestrated pipeline to train and evaluate models across 5 temperature splits (320K, 348K, 379K, 413K, 450K).

## Setup on Remote Machine

### 1. Copy required files to remote

From local machine:
```bash
# Copy the orchestration scripts
scp run_mdcath_experiments.py zahralab@sn4622116030:~/Desktop/ESMfluc_26/scripts/
scp run_mdcath_experiments.sh zahralab@sn4622116030:~/Desktop/ESMfluc_26/scripts/

# Copy train_unified.py if not already there
scp train_unified.py zahralab@sn4622116030:~/Desktop/ESMfluc_26/scripts/
```

### 2. Verify directory structure

On remote machine:
```
~/Desktop/ESMfluc_26/
├── data/
│   └── mdcath/
│       ├── test_split.fasta          ✓ (already created)
│       ├── train_320K.csv            ✓
│       ├── test_320K.csv             ✓
│       ├── train_348K.csv            ✓
│       ├── test_348K.csv             ✓
│       └── ... (other temperatures)  ✓
└── scripts/
    ├── train_unified.py              ← Copy this
    ├── get_attn.py                   ✓
    ├── models.py                     ✓
    ├── data_utils.py                 ✓
    ├── arguments.py                  ✓
    ├── run_mdcath_experiments.py     ← Copy this
    └── run_mdcath_experiments.sh     ← Copy this
```

## Running Experiments

### Option 1: Python Script (Recommended)

```bash
cd ~/Desktop/ESMfluc_26/scripts
python3 run_mdcath_experiments.py
```

**Features:**
- Better error handling
- Progress tracking
- Can resume with `--skip_training` flag

**Options:**
```bash
# Run specific temperatures only
python3 run_mdcath_experiments.py --temperatures 320K 348K

# Skip training and just collect results
python3 run_mdcath_experiments.py --skip_training
```

### Option 2: Bash Script

```bash
cd ~/Desktop/ESMfluc_26/scripts
bash run_mdcath_experiments.sh
```

## What Gets Executed

For each temperature (320K → 450K), the pipeline runs:

### 1. Classification Training
```
Task: Binary classification
Threshold: NEQ > 1.0
Model: BiLSTM + Self-Attention
Backbone: esm2_t33_650M_UR50D (fully unfrozen)
Hyperparams:
  - hidden_size: 512
  - num_layers: 3
  - batch_size: 4
  - patience: 5
  - LR scheduler: enabled
Output: results/mdcath/{temp}/classification/
```

### 2. Attention Extraction (Classification)
```
Extracts attention weights for all test sequences
Output: results/mdcath/{temp}/classification/*.json
```

### 3. Regression Training
```
Task: Regression (raw NEQ values)
Model: BiLSTM + Self-Attention (regression variant)
Same hyperparameters as classification
Output: results/mdcath/{temp}/regression/
```

### 4. Attention Extraction (Regression)
```
Extracts attention weights for all test sequences
Output: results/mdcath/{temp}/regression/*.json
```

## Output Structure

```
results/mdcath/
├── 320K/
│   ├── classification/
│   │   ├── best_model.pth
│   │   ├── run_summary.csv
│   │   ├── predictions.csv
│   │   ├── confusion_matrix.png
│   │   └── *.json (attention files)
│   └── regression/
│       ├── best_model.pth
│       ├── run_summary.csv
│       ├── predictions.csv
│       └── *.json (attention files)
├── 348K/
│   └── ... (same structure)
├── ... (other temperatures)
├── mdcath_full_summary.csv          ← All results combined
└── mdcath_summary_condensed.csv     ← Key metrics only
```

## Results Summary

After all 10 runs complete (5 temps × 2 tasks), the pipeline generates:

### `mdcath_full_summary.csv`
Complete results with all metrics and metadata

### `mdcath_summary_condensed.csv`
Key performance metrics:

| temperature | task_type      | test_accuracy | test_f1 | test_mse | test_mae | test_r2 |
|-------------|----------------|---------------|---------|----------|----------|---------|
| 320K        | classification | ...           | ...     | -        | -        | -       |
| 320K        | regression     | -             | -       | ...      | ...      | ...     |
| 348K        | classification | ...           | ...     | -        | -        | -       |
| ...         | ...            | ...           | ...     | ...      | ...      | ...     |

## Monitoring Progress

The script prints detailed progress:
- ✓ Training completion with timing
- ✓ Attention extraction status
- ✓ File creation confirmation
- ✓ Final summary table

Check logs:
```bash
# Run in background with logging
nohup python3 run_mdcath_experiments.py > mdcath_experiments.log 2>&1 &

# Monitor progress
tail -f mdcath_experiments.log
```

## Estimated Runtime

- **Per temperature**: ~2-4 hours (depends on convergence)
  - Classification: ~1-2 hours
  - Regression: ~1-2 hours
  - Attention extraction: ~10-20 minutes each
- **Total (5 temperatures)**: ~10-20 hours

## Troubleshooting

### "Data directory not found"
Ensure you're running from `~/Desktop/ESMfluc_26/scripts/`

### "train_unified.py not found"
Copy `train_unified.py` to the scripts directory

### CUDA out of memory
Reduce batch size in the script (currently set to 4)

### Resume interrupted run
The pipeline saves checkpoints. Just re-run and it will skip completed experiments.

## Configuration

To modify hyperparameters, edit the constants at the top of the script:

```python
# In run_mdcath_experiments.py
BATCH_SIZE = 4          # Reduce if OOM
NUM_LAYERS = 3
HIDDEN_SIZE = 512
PATIENCE = 5
FREEZE_LAYERS = "none"  # Fully unfrozen backbone
```
