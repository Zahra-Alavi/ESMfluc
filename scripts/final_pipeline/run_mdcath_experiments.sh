#!/bin/bash
################################################################################
# MDCATH Temperature Experiments
# Train classification + regression models for each temperature split
################################################################################

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${SCRIPT_DIR}/../../data/mdcath"
TEST_FASTA="${DATA_DIR}/test_split.fasta"
TEST_CSV="${DATA_DIR}/test_split_mmseq2.csv"
TEMPERATURES=("320K" "348K" "379K" "413K" "450K")

# Model hyperparameters
BATCH_SIZE=4
ESM_MODEL="esm2_t33_650M_UR50D"
NUM_LAYERS=3
HIDDEN_SIZE=512
PATIENCE=5
DEVICE="cuda"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

################################################################################
# Functions
################################################################################

create_fasta() {
    local csv_file=$1
    local fasta_file=$2
    
    echo -e "${BLUE}Creating FASTA from ${csv_file}...${NC}"
    
    python3 << EOF
import pandas as pd
df = pd.read_csv('${csv_file}')
with open('${fasta_file}', 'w') as f:
    for _, row in df.iterrows():
        f.write(f">{row['protein_id']}\n{row['sequence']}\n")
print(f"Created {len(df)} sequences in ${fasta_file}")
EOF
}

train_classification() {
    local temp=$1
    local train_csv=$2
    local test_csv=$3
    local folder_name="${temp}/classification"
    
    echo -e "\n${GREEN}${'='*80}${NC}"
    echo -e "${GREEN}CLASSIFICATION TRAINING: ${temp}${NC}"
    echo -e "${GREEN}${'='*80}${NC}\n"
    
    python3 train_unified.py \
        --train_data_file "${train_csv}" \
        --test_data_file "${test_csv}" \
        --task_type classification \
        --neq_thresholds 1.0 \
        --num_classes 2 \
        --architecture bilstm_attention \
        --batch_size ${BATCH_SIZE} \
        --esm_model ${ESM_MODEL} \
        --num_layers ${NUM_LAYERS} \
        --hidden_size ${HIDDEN_SIZE} \
        --device ${DEVICE} \
        --patience ${PATIENCE} \
        --result_foldername "${folder_name}"
}

train_regression() {
    local temp=$1
    local train_csv=$2
    local test_csv=$3
    local folder_name="${temp}/regression"
    
    echo -e "\n${GREEN}${'='*80}${NC}"
    echo -e "${GREEN}REGRESSION TRAINING: ${temp}${NC}"
    echo -e "${GREEN}${'='*80}${NC}\n"
    
    python3 train_unified.py \
        --train_data_file "${train_csv}" \
        --test_data_file "${test_csv}" \
        --task_type regression \
        --architecture bilstm_attention \
        --batch_size ${BATCH_SIZE} \
        --esm_model ${ESM_MODEL} \
        --num_layers ${NUM_LAYERS} \
        --hidden_size ${HIDDEN_SIZE} \
        --device ${DEVICE} \
        --patience ${PATIENCE} \
        --result_foldername "${folder_name}"
}

extract_attention() {
    local checkpoint=$1
    local fasta_file=$2
    local output_dir=$3
    local task_type=$4
    
    echo -e "\n${BLUE}Extracting attention for ${task_type}...${NC}"
    
    if [ "${task_type}" == "classification" ]; then
        python3 get_attn.py \
            --checkpoint "${checkpoint}" \
    mkdir -p "${output_dir}"
    local output_base="${output_dir}/attention_weights"
    
    if [ "${task_type}" == "classification" ]; then
        python3 get_attn.py \
            --checkpoint "${checkpoint}" \
            --fasta_file "${fasta_file}" \
            --output "${output_base}" \
            --esm_model ${ESM_MODEL} \
            --architecture bilstm_attention \
            --task_type classification \
            --num_classes 2 \
            --hidden_size ${HIDDEN_SIZE} \
            --num_layers ${NUM_LAYERS} \
            --device ${DEVICE}
    else
        python3 get_attn.py \
            --checkpoint "${checkpoint}" \
            --fasta_file "${fasta_file}" \
            --output "${output_base}" \
            --esm_model ${ESM_MODEL} \
            --architecture bilstm_attention \
            --task_type regression \
            --num_outputs 1 \
            --hidden_size ${HIDDEN_SIZE} \
            --num_layers ${NUM_LAYERS}

################################################################################
# Main Execution
################################################################################

echo -e "\n${YELLOW}${'='*80}${NC}"
echo -e "${YELLOW}MDCATH TEMPERATURE EXPERIMENTS${NC}"
echo -e "${YELLOW}${'='*80}${NC}"
echo "Temperatures: ${TEMPERATURES[@]}"
echo "Tasks: Classification (threshold=1.0), Regression (raw NEQ)"
echo "Model: BiLSTM + ${ESM_MODEL}"
echo "Params: hidden=${HIDDEN_SIZE}, layers=${NUM_LAYERS}, batch=${BATCH_SIZE}"
echo "Backbone: Fully unfrozen"
echo "Output: ${SCRIPT_DIR}/results/"
echo -e "${YELLOW}${'='*80}${NC}\n"

# Check data directory
if [ ! -d "${DATA_DIR}" ]; then
    echo -e "${RED}ERROR: Data directory not found: ${DATA_DIR}${NC}"
    echo "Please ensure you're running from the scripts directory."
    echo "Current directory: $(pwd)"
    echo "Script directory: ${SCRIPT_DIR}"
    exit 1
fi

echo "Data directory: ${DATA_DIR}"
echo "Script directory: ${SCRIPT_DIR}"

# Create results directory
mkdir -p "${SCRIPT_DIR}/results"

# Create test FASTA file if it doesn't exist
if [ ! -f "${TEST_FASTA}" ]; then
    echo -e "\n${BLUE}Creating test FASTA file...${NC}"
    python3 create_mdcath_fasta.py \
        --test_csv "${TEST_CSV}" \
        --output "${TEST_FASTA}"
else
    echo -e "\n${GREEN}✓ Using existing test FASTA: ${TEST_FASTA}${NC}"
fi

# Process each temperature
for temp in "${TEMPERATURES[@]}"; do
    echo -e "\n${YELLOW}################################################################################${NC}"
    echo -e "${YELLOW}# PROCESSING TEMPERATURE: ${temp}${NC}"
    echo -e "${YELLOW}################################################################################${NC}\n"
    
    # File paths
    TRAIN_CSV="${DATA_DIR}/train_${temp}.csv"
    TEST_CSV="${DATA_DIR}/test_${temp}.csv"
    
    # Check files exist
    if [ ! -f "${TRAIN_CSV}" ] || [ ! -f "${TEST_CSV}" ]; then
        echo -e "${RED}ERROR: Missing CSV files for ${temp}${NC}"
        continue
    fi
    
    # 1. CLASSIFICATION
    train_classification "${temp}" "${TRAIN_CSV}" "${TEST_CSV}"
    
    CLS_CHECKPOINT="${SCRIPT_DIR}/results/${temp}/classification/best_model.pth"
    if [ -f "${CLS_CHECKPOINT}" ]; then
        CLS_OUTPUT="${SCRIPT_DIR}/results/${temp}/classification"
        extract_attention "${CLS_CHECKPOINT}" "${TEST_FASTA}" "${CLS_OUTPUT}" "classification"
    else
        echo -e "${RED}WARNING: Classification checkpoint not found${NC}"
    fi
    
    # 2. REGRESSION
    train_regression "${temp}" "${TRAIN_CSV}" "${TEST_CSV}"
    
    REG_CHECKPOINT="${SCRIPT_DIR}/results/${temp}/regression/best_model.pth"
    if [ -f "${REG_CHECKPOINT}" ]; then
        REG_OUTPUT="${SCRIPT_DIR}/results/${temp}/regression"
        extract_attention "${REG_CHECKPOINT}" "${TEST_FASTA}" "${REG_OUTPUT}" "regression"
    else
        echo -e "${RED}WARNING: Regression checkpoint not found${NC}"
    fi
done

################################################################################
# Collect Results
################################################################################

echo -e "\n${YELLOW}${'='*80}${NC}"
echo -e "${YELLOW}COLLECTING RESULTS${NC}"
echo -e "${YELLOW}${'='*80}${NC}\n"

python3 << 'PYTHON_SCRIPT'
import pandas as pd
from pathlib import Path
import sys
import os

# Get script directory from environment or use cwd
SCRIPT_DIR = Path(os.environ.get('SCRIPT_DIR', '.')).resolve()
RESULTS_ROOT = SCRIPT_DIR / "results"
TEMPERATURES = ["320K", "348K", "379K", "413K", "450K"]

all_results = []

for temp in TEMPERATURES:
    for task_type in ["classification", "regression"]:
        summary_file = RESULTS_ROOT / temp / task_type / "run_summary.csv"
        
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df['temperature'] = temp
            df['task_type'] = task_type
            all_results.append(df)
            print(f"✓ Loaded {summary_file}")
        else:
            print(f"✗ Missing {summary_file}")

if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    metadata_cols = ['temperature', 'task_type']
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]
    
    # Save full summary
    summary_output = RESULTS_ROOT / "mdcath_full_summary.csv"
    combined_df.to_csv(summary_output, index=False)
    print(f"\nSaved full summary: {summary_output}")
    
    # Create condensed summary
    metric_cols = [col for col in combined_df.columns 
                   if any(m in col.lower() for m in ['accuracy', 'f1', 'mse', 'mae', 'r2', 'rmse', 'pearson', 'spearman'])]
    
    if metric_cols:
        condensed_df = combined_df[metadata_cols + metric_cols]
        condensed_output = RESULTS_ROOT / "mdcath_summary_condensed.csv"
        condensed_df.to_csv(condensed_output, index=False)
        print(f"Saved condensed summary: {condensed_output}")
        
        print("\n" + "="*80)
        print("MDCATH EXPERIMENT SUMMARY")
        print("="*80)
        print(condensed_df.to_string(index=False))
else:
    print("ERROR: No results found!")
    sys.exit(1)
PYTHON_SCRIPT

echo -e "\n${GREEN}${'='*80}${NC}"
echo -e "${GREEN}ALL EXPERIMENTS COMPLETED!${NC}"
echo -e "${GREEN}Results saved in: ${SCRIPT_DIR}/results/${NC}"
echo -e "${GREEN}${'='*80}${NC}\n"
