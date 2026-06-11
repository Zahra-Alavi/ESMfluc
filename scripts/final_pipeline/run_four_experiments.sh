#!/usr/bin/env bash
# =============================================================================
# run_four_experiments.sh
#
# Trains 4 ESMfluc binary-classification models (BiLSTM+Attention head):
#   1. ESM2  frozen      ─┐ GPU 0 (sequential)
#   2. ESM2  finetuned   ─┘
#   3. ESM3  frozen      ─┐ GPU 1 (sequential)
#   4. ESM3  finetuned   ─┘
#
# Both GPU groups run in parallel so neither GPU sits idle.
#
# For each run:
#   - Saves best_model.pth to results/<run_name>/
#   - Extracts BiLSTM self-attention  → results/<run_name>/bilstm_attention.json
#   - Extracts backbone attention     → results/<run_name>/backbone_attention.json
#
# Usage (from scripts/final_pipeline/):
#   bash run_four_experiments.sh
#
# Optional overrides:
#   TRAIN_CSV   path to training CSV   (default: ../../data/train_data.csv)
#   TEST_CSV    path to test CSV       (default: ../../data/test_data.csv)
#   FASTA       path to test FASTA     (default: ../../data/test_data_sequences.fasta)
#   EPOCHS      max training epochs    (default: 80)
#   BATCH       batch size             (default: 4)
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_CSV="${TRAIN_CSV:-../../data/train_data.csv}"
TEST_CSV="${TEST_CSV:-../../data/test_data.csv}"
FASTA="${FASTA:-../../data/test_data_sequences.fasta}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-4}"

# ── Shared hyperparameters ────────────────────────────────────────────────────
# Note: --device is always "cuda"; CUDA_VISIBLE_DEVICES restricts which physical
# GPU is visible, so "cuda" inside each subshell maps to the assigned GPU.
COMMON_ARGS=(
    --train_data_file "$TRAIN_CSV"
    --test_data_file  "$TEST_CSV"
    --architecture    bilstm_attention
    --hidden_size     512
    --num_layers      3
    --dropout         0.3
    --bidirectional   1
    --num_classes     2
    --neq_thresholds  1.0
    --loss_function   focal
    --lr_scheduler    reduce_on_plateau
    --epochs          "$EPOCHS"
    --patience        5
    --batch_size      "$BATCH"
    --device          cuda
    --task_type       classification
    --seed            42
)

# ── Helper: run one experiment ────────────────────────────────────────────────
# Arguments: <run_name> <esm_model> <freeze_flag (--freeze_all_backbone or "")>
#            <is_esm3 ("true"|"false")> <gpu_id (0|1)>
run_experiment() {
    local RUN_NAME="$1"
    local ESM_MODEL="$2"
    local FREEZE_FLAG="$3"        # either "--freeze_all_backbone" or ""
    local IS_ESM3="$4"            # "true" or "false"
    local GPU_ID="$5"             # physical GPU index

    local OUT_DIR="results/${RUN_NAME}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "========================================================"
    echo "  EXPERIMENT: ${RUN_NAME}  [GPU ${GPU_ID}]"
    echo "  backbone: ${ESM_MODEL}  |  freeze: ${FREEZE_FLAG:-none}"
    echo "========================================================"

    # ── 1. Train ──────────────────────────────────────────────────────────────
    echo "[${RUN_NAME}] [1/3] Training on GPU ${GPU_ID}..."
    TRAIN_ARGS=(
        "${COMMON_ARGS[@]}"
        --esm_model         "$ESM_MODEL"
        --result_foldername "$RUN_NAME"
    )
    if [[ -n "$FREEZE_FLAG" ]]; then
        TRAIN_ARGS+=("$FREEZE_FLAG")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${OUT_DIR}/train.log"

    # best_model.pth is saved by train.py inside results/<run_name>/
    local CKPT="${OUT_DIR}/best_model.pth"
    if [[ ! -f "$CKPT" ]]; then
        # Fallback: use last_model.pth if early stopping never fired
        CKPT="${OUT_DIR}/last_model.pth"
    fi
    if [[ ! -f "$CKPT" ]]; then
        echo "ERROR [${RUN_NAME}]: No checkpoint found in ${OUT_DIR}. Training may have failed."
        exit 1
    fi
    echo "[${RUN_NAME}]   Checkpoint: $CKPT"

    # ── 2. BiLSTM self-attention ──────────────────────────────────────────────
    echo "[${RUN_NAME}] [2/3] Extracting BiLSTM self-attention on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES="$GPU_ID" python Attention/get_attn.py \
        --checkpoint    "$CKPT" \
        --fasta_file    "$FASTA" \
        --architecture  bilstm_attention \
        --esm_model     "$ESM_MODEL" \
        --hidden_size   512 \
        --num_layers    3 \
        --dropout       0.0 \
        --bidirectional 1 \
        --num_classes   2 \
        --output        "${OUT_DIR}/bilstm_attention.json" \
        2>&1 | tee "${OUT_DIR}/bilstm_attn.log"

    # ── 3. Backbone attention (before BiLSTM) ─────────────────────────────────
    echo "[${RUN_NAME}] [3/3] Extracting backbone attention on GPU ${GPU_ID}..."
    if [[ "$IS_ESM3" == "true" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" python Attention/extract_backbone_attn.py \
            --checkpoint "$CKPT" \
            --fasta_file "$FASTA" \
            --is_esm3 \
            --output     "${OUT_DIR}/backbone_attention.json" \
            2>&1 | tee "${OUT_DIR}/backbone_attn.log"
    else
        CUDA_VISIBLE_DEVICES="$GPU_ID" python Attention/extract_backbone_attn.py \
            --checkpoint "$CKPT" \
            --fasta_file "$FASTA" \
            --esm_model  "$ESM_MODEL" \
            --output     "${OUT_DIR}/backbone_attention.json" \
            2>&1 | tee "${OUT_DIR}/backbone_attn.log"
    fi

    echo "[${RUN_NAME}]   Done. Output folder: ${OUT_DIR}/"
    echo "    best_model.pth          <- trained checkpoint"
    echo "    bilstm_attention.json   <- self-attention after BiLSTM"
    echo "    backbone_attention.json <- ESM backbone attention (before BiLSTM)"
}

# ── Run all 4 experiments across 2 GPUs in parallel ──────────────────────────
#
#   GPU 0: esm2_frozen  → esm2_finetuned  (sequential on GPU 0)
#   GPU 1: esm3_frozen  → esm3_finetuned  (sequential on GPU 1)
#
# Both groups start at the same time; the script waits until both finish.

echo "Launching GPU 0 (ESM2 runs) and GPU 1 (ESM3 runs) in parallel..."

(
    set -euo pipefail
    run_experiment "esm2_frozen"    "esm2_t33_650M_UR50D"  "--freeze_all_backbone"  "false"  "0"
    run_experiment "esm2_finetuned" "esm2_t33_650M_UR50D"  ""                       "false"  "0"
) &
GPU0_PID=$!

(
    set -euo pipefail
    run_experiment "esm3_frozen"    "esm3_sm_open_v1"      "--freeze_all_backbone"  "true"   "1"
    run_experiment "esm3_finetuned" "esm3_sm_open_v1"      ""                       "true"   "1"
) &
GPU1_PID=$!

# Wait for both groups and capture exit codes
GPU0_STATUS=0
GPU1_STATUS=0
wait "$GPU0_PID" || GPU0_STATUS=$?
wait "$GPU1_PID" || GPU1_STATUS=$?

echo ""
echo "========================================================"
if [[ "$GPU0_STATUS" -ne 0 ]]; then
    echo "  FAILED: GPU 0 (ESM2) experiments exited with code ${GPU0_STATUS}"
fi
if [[ "$GPU1_STATUS" -ne 0 ]]; then
    echo "  FAILED: GPU 1 (ESM3) experiments exited with code ${GPU1_STATUS}"
fi
if [[ "$GPU0_STATUS" -eq 0 && "$GPU1_STATUS" -eq 0 ]]; then
    echo "  ALL 4 EXPERIMENTS COMPLETE"
    echo "  Results in: scripts/final_pipeline/results/"
    echo "    esm2_frozen/"
    echo "    esm2_finetuned/"
    echo "    esm3_frozen/"
    echo "    esm3_finetuned/"
fi
echo "========================================================"

# Propagate failure if either group failed
[[ "$GPU0_STATUS" -eq 0 && "$GPU1_STATUS" -eq 0 ]]
