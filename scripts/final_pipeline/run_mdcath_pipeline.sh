#!/bin/bash
# -*- coding: utf-8 -*-
# Full pipeline for running mdcath classification experiments with cluster-aware splitting
# This script demonstrates the complete workflow from clustering to classification

set -e  # Exit on error

echo "=========================================="
echo "MDCATH Classification Pipeline"
echo "=========================================="
echo ""

# Configuration
INPUT_DATA="${1:-../../data/train_data.csv}"
OUTPUT_DIR="${2:-./mdcath_experiments}"
MIN_SEQ_ID="${3:-0.3}"
COVERAGE="${4:-0.8}"
TEST_SIZE="${5:-0.2}"
SEED="${6:-42}"

echo "Configuration:"
echo "  Input data: $INPUT_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Min sequence identity: $MIN_SEQ_ID"
echo "  Coverage: $COVERAGE"
echo "  Test size: $TEST_SIZE"
echo "  Random seed: $SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Cluster sequences using MMseqs2
echo "=========================================="
echo "Step 1: Clustering sequences with MMseqs2"
echo "=========================================="
echo ""

CLUSTERED_DATA="$OUTPUT_DIR/clustered_data.csv"

python cluster_sequences.py \
    --input "$INPUT_DATA" \
    --output "$CLUSTERED_DATA" \
    --min_seq_id "$MIN_SEQ_ID" \
    --coverage "$COVERAGE"

echo ""
echo "[OK] Clustering completed. Output: $CLUSTERED_DATA"
echo ""

# Step 2: Split data by clusters
echo "=========================================="
echo "Step 2: Cluster-aware data splitting"
echo "=========================================="
echo ""

TRAIN_DATA="$OUTPUT_DIR/train_data.csv"
TEST_DATA="$OUTPUT_DIR/test_data.csv"

python cluster_aware_split.py \
    --input "$CLUSTERED_DATA" \
    --train_output "$TRAIN_DATA" \
    --test_output "$TEST_DATA" \
    --test_size "$TEST_SIZE" \
    --seed "$SEED"

echo ""
echo "[OK] Data splitting completed."
echo "  Training data: $TRAIN_DATA"
echo "  Test data: $TEST_DATA"
echo ""

# Step 3: Run temperature experiments
echo "=========================================="
echo "Step 3: Running classification experiments"
echo "=========================================="
echo ""

EXPERIMENTS_DIR="$OUTPUT_DIR/experiments"

python run_temperature_experiments.py \
    --train_data "$TRAIN_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "$EXPERIMENTS_DIR" \
    --esm_model esm2_t12_35M_UR50D \
    --epochs 20 \
    --batch_size 4 \
    --device cuda \
    --seed "$SEED"

echo ""
echo "[OK] All experiments completed."
echo "  Results directory: $EXPERIMENTS_DIR"
echo ""

# Summary
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  1. Clustered data: $CLUSTERED_DATA"
echo "  2. Training data: $TRAIN_DATA"
echo "  3. Test data: $TEST_DATA"
echo "  4. Experiment results: $EXPERIMENTS_DIR"
echo ""
echo "To view experiment summary:"
echo "  cat $EXPERIMENTS_DIR/experiment_summary.json"
echo ""
