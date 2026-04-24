# mdcath_pipeline

This folder contains the training pipeline for ESM-based protein flexibility prediction (Neq) using a PyTorch Lightning workflow. It includes dataset preparation, model and loss definitions, and a CLI entry point for training or validation-only runs.

## Usage

Run training with the default paths and settings:

```bash
python train.py
```

Run validation only with a specific checkpoint:

```bash
python train.py --test-only --checkpoint_path /path/to/checkpoint.ckpt
```

### Command Line Arguments

**General**

- `--test-only`: Run validation only without training.

**Data & Paths**

- `--train_path`: Path to training CSV (default: `../../data/mdcath/train_split_mmseqs2.csv`).
- `--val_path`: Path to validation CSV (default: `../../data/mdcath/test_split_mmseqs2.csv`).
- `--checkpoint_dir`: Output directory for checkpoints (default: `checkpoints/`).
- `--checkpoint_path`: Path to a specific `.ckpt` file for test-only runs.
- `--temperatures`: Comma-separated temperatures to include (default: `320,348,379,413,450`).

**Model Architecture**

- `--model_name`: Hugging Face ESM2 model name (default: `facebook/esm2_t6_8M_UR50D`).
- `--hidden_size`: Regressor head hidden dimension (default: `512`).
- `--max_len`: Maximum sequence length (default: `1024`).
- `--masked_value`: Mask value for padded labels (default: `-100`).
- `--num_unfreeze_layers`: Unfreeze last N ESM layers for fine-tuning (default: `0`).
- `--dropout_rate`: Dropout rate in regression head (default: `0.1`).

**Training Hyperparameters**

- `--batch_size`: Batch size (default: `16`).
- `--lr`: Base learning rate (auto-scaled by GPU count) (default: `1e-5`).
- `--weight_decay`: Weight decay for AdamW (default: `1e-2`).
- `--epochs`: Number of training epochs (default: `20`).
- `--early_stop_patience`: Early stopping patience (default: `3`).
- `--seed`: Random seed (default: `42`).

**Loss Function**

- `--loss_type`: Loss type: `weighted` or `standard` (default: `weighted`).
- `--weight_threshold`: Threshold for weighted loss (default: `3.0`).
- `--weight_factor`: Weight multiplier for high values (default: `5.0`).

**Computational Resources**

- `--num_workers`: DataLoader workers (default: `4`).
