# Summary: MDCATH Classification Implementation

## Overview
This implementation addresses the issue of re-running classification on mdcath at different temperatures while preventing data leakage through cluster-aware data splitting.

## Problem Statement
When working with protein datasets like mdcath:
1. Proteins with high sequence similarity can cause data leakage if split between train/test sets
2. Models might learn sequence similarity patterns rather than biological properties
3. Need to run experiments at different "temperatures" (neq_thresholds) to find optimal classification boundaries

## Solution Implemented

### 1. MMseqs2-Based Clustering (`cluster_sequences.py`)
- Clusters proteins by sequence similarity (default: 30% identity, 80% coverage)
- Uses MMseqs2's `easy-cluster` command for efficient clustering
- Outputs CSV with cluster assignments for each sequence
- Provides statistics on cluster sizes and distribution

**Key Features:**
- Automatic detection of sequence ID columns (name, id, or index)
- Configurable similarity thresholds
- Comprehensive error handling and logging

### 2. Cluster-Aware Data Splitting (`cluster_aware_split.py`)
- Ensures entire clusters are assigned to the same split (train/val/test)
- Prevents data leakage by keeping similar proteins together
- Supports both 2-way (train/test) and 3-way (train/val/test) splits
- Verifies no cluster overlap between splits

**Key Features:**
- Reproducible splits with random seed
- Configurable split ratios
- Automatic verification of no data leakage
- Detailed statistics for each split

### 3. Temperature Experiments Runner (`run_temperature_experiments.py`)
- Automates running classification at different neq_thresholds
- Supports both default and custom experiment configurations
- Runs each experiment independently with separate output directories
- Generates comprehensive summary reports

**Default Configurations:**
1. Binary (Low): 2 classes, threshold [1.0]
2. Binary (Mid): 2 classes, threshold [1.5]
3. Binary (High): 2 classes, threshold [2.0]
4. 3-Class (Low): 3 classes, thresholds [1.0, 1.5]
5. 3-Class (Mid): 3 classes, thresholds [1.0, 2.0]
6. 4-Class (Standard): 4 classes, thresholds [1.0, 2.0, 4.0]
7. 4-Class (Fine): 4 classes, thresholds [1.0, 1.5, 2.5]
8. 5-Class: 5 classes, thresholds [1.0, 1.5, 2.0, 3.0]

### 4. Complete Pipeline Script (`run_mdcath_pipeline.sh`)
- Orchestrates all three steps in sequence
- Single command execution for entire workflow
- Configurable via command-line arguments
- Detailed progress reporting

## Usage Examples

### Quick Start
```bash
bash run_mdcath_pipeline.sh ../../data/train_data.csv ./mdcath_experiments
```

### Step-by-Step
```bash
# 1. Cluster sequences
python cluster_sequences.py \
    --input ../../data/train_data.csv \
    --output clustered_data.csv \
    --min_seq_id 0.3 \
    --coverage 0.8

# 2. Cluster-aware split
python cluster_aware_split.py \
    --input clustered_data.csv \
    --train_output train_data.csv \
    --test_output test_data.csv \
    --test_size 0.2

# 3. Run temperature experiments
python run_temperature_experiments.py \
    --train_data train_data.csv \
    --test_data test_data.csv \
    --output_dir ./temperature_experiments
```

### Custom Configuration
```bash
# Create custom experiment config (JSON)
# Then run:
python run_temperature_experiments.py \
    --train_data train.csv \
    --test_data test.csv \
    --config custom_config.json
```

## Documentation Provided

1. **MDCATH_WORKFLOW.md**: Comprehensive guide with:
   - Detailed workflow explanation
   - Parameter descriptions
   - Troubleshooting guide
   - Best practices
   - Example configurations

2. **QUICK_REFERENCE.md**: Quick reference for:
   - Common commands
   - Use cases
   - Troubleshooting tips
   - File structure

3. **Updated README.md**: Links to new workflow documentation

4. **example_temperature_config.json**: Example custom configuration file

## Testing

**test_cluster_split.py** provides comprehensive tests:
- Creates synthetic clustered data
- Tests 2-way split (train/test)
- Tests 3-way split (train/val/test)
- Verifies no data leakage
- Validates split proportions
- All tests passing ✓

## Security

- CodeQL analysis: 0 vulnerabilities found
- All scripts use secure subprocess calls
- Proper input validation
- No hardcoded credentials or sensitive data

## Integration with Existing Codebase

The new workflow integrates seamlessly with existing training pipeline:
- Output CSV files compatible with existing `main.py` and `train.py`
- Uses same data format (sequence, neq columns)
- Supports all existing model architectures and hyperparameters
- No breaking changes to existing functionality

## Advantages

1. **Prevents Data Leakage**: Cluster-aware splitting ensures similar proteins don't appear in both train and test sets
2. **Comprehensive Temperature Exploration**: Automated testing of multiple classification thresholds
3. **Reproducible**: All scripts support random seeds for reproducibility
4. **Flexible**: Easy to customize clustering parameters and experiment configurations
5. **Well-Documented**: Extensive documentation and examples
6. **Tested**: Comprehensive test suite validates functionality

## Requirements

**New Dependencies:**
- MMseqs2 (for clustering, can be installed via conda: `conda install -c bioconda mmseqs2`)

**Python Dependencies (already in requirements.txt):**
- pandas
- scikit-learn
- numpy

## Output Structure

```
mdcath_experiments/
├── clustered_data.csv           # Sequences with cluster assignments
├── train_data.csv               # Training split
├── test_data.csv                # Test split
└── experiments/
    ├── experiment_summary.json  # Summary of all experiments
    ├── binary_low_temp/
    │   ├── experiment_config.json
    │   ├── training_log.txt
    │   └── [model checkpoints]
    ├── binary_mid_temp/
    └── [other experiments]/
```

## Future Enhancements

Potential improvements for future work:
1. Support for additional clustering tools (CD-HIT, BLAST)
2. Stratified sampling by cluster sizes
3. Cross-validation with cluster-aware folds
4. Parallel execution of experiments
5. Automatic result comparison and visualization

## Conclusion

This implementation provides a robust, well-tested, and well-documented solution for running classification experiments on mdcath (or any protein dataset) while preventing data leakage. The workflow is flexible, reproducible, and integrates seamlessly with the existing codebase.
