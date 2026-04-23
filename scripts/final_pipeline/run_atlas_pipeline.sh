#!/usr/bin/env bash
# =============================================================================
# run_atlas_pipeline.sh
#
# Master orchestration script for the ATLAS attention experiments.
# Run this script from within scripts/final_pipeline/ or supply SCRIPT_DIR.
#
# Experiments (6 total):
#   esm2_binary_frozen     – ESM2 frozen,   binary classifier
#   esm2_binary_unfrozen   – ESM2 unfrozen, binary classifier
#   esm3_binary_frozen     – ESM3 frozen,   binary classifier
#   esm3_binary_unfrozen   – ESM3 unfrozen, binary classifier
#   esm2_regression_frozen – ESM2 frozen,   regression
#   esm2_3class_frozen     – ESM2 frozen,   3-class classifier
#
# For each experiment we:
#   1. Train the model (saves best_model.pth + metrics to results/<name>/)
#   2. Extract BiLSTM self-attention  → results/<name>/bilstm_attn.json
#   3. Extract backbone last-layer attention → results/<name>/backbone_attn.json
# Then run the unified attention analysis pipeline on all experiments.
#
# Usage:
#   cd /path/to/scripts/final_pipeline
#   bash run_atlas_pipeline.sh [OPTIONS]
#
# Options (all optional):
#   --gpu <id>         CUDA device id(s), e.g. "0" or "0,1"  (default: 0)
#   --skip_train       Skip training, only extract attention + analyse
#   --skip_extract     Skip attention extraction
#   --skip_analysis    Skip analysis
#   --only <name>      Run only a single experiment by name
#   --amp_dtype <dt>   bf16 or fp16  (default: bf16; use fp16 on V100/older)
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../data"
ATTN_DIR="${SCRIPT_DIR}/Attention"
RESULTS_DIR="${SCRIPT_DIR}/results"

TRAIN_CSV="${DATA_DIR}/train_data.csv"
TEST_CSV="${DATA_DIR}/test_data_with_names.csv"
FASTA="${DATA_DIR}/test_data_sequences.fasta"
REF_JSON="${DATA_DIR}/attn_bilstm_f1-4_nsp3_neq.json"

# ── Default options ────────────────────────────────────────────────────────────
GPU="0"
SKIP_TRAIN=0
SKIP_EXTRACT=0
SKIP_ANALYSIS=0
ONLY=""
AMP_DTYPE="bf16"

# Parse optional CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2";       shift 2 ;;
        --skip_train)     SKIP_TRAIN=1;    shift ;;
        --skip_extract)   SKIP_EXTRACT=1;  shift ;;
        --skip_analysis)  SKIP_ANALYSIS=1; shift ;;
        --only)       ONLY="$2";      shift 2 ;;
        --amp_dtype)  AMP_DTYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
# Ensure models.py (in this directory) is importable by all sub-scripts
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ── Fixed hyperparameters (identical across all experiments) ───────────────────
HIDDEN=512
LAYERS=3
DROPOUT=0.3
BATCH=4
EPOCHS=30
PATIENCE=10
WD=0.01
SEED=42
ESM2_MODEL="esm2_t33_650M_UR50D"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
header() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${GREEN}  $1${NC}"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
info()   { echo -e "${YELLOW}  [info]${NC} $1"; }

# ── Training helper ─────────────────────────────────────────────────────────────
# run_training <exp_name> <is_esm3:0|1> <frozen:0|1> <lr> <task> <num_classes> <thresholds...>
run_training() {
    local name="$1"
    local is_esm3="$2"
    local frozen="$3"
    local lr="$4"
    local task="$5"
    local num_classes="$6"
    shift 6
    local thresholds=("$@")   # remaining args are threshold values

    mkdir -p "${RESULTS_DIR}/${name}"
    info "Training ${name} …"

    # Build argument array
    local ARGS=(
        --train_data_file "${TRAIN_CSV}"
        --test_data_file  "${TEST_CSV}"
        --architecture    bilstm_attention
        --hidden_size     ${HIDDEN}
        --num_layers      ${LAYERS}
        --dropout         ${DROPOUT}
        --batch_size      ${BATCH}
        --epochs          ${EPOCHS}
        --patience        ${PATIENCE}
        --weight_decay    ${WD}
        --seed            ${SEED}
        --bidirectional   1
        --lr              ${lr}
        --lr_scheduler    reduce_on_plateau
        --mixed_precision
        --amp_dtype       ${AMP_DTYPE}
        --result_foldername "${name}"
        --device          cuda
    )

    # ESM backbone
    if [[ "${is_esm3}" -eq 1 ]]; then
        ARGS+=(--esm_model esm3_sm_open_v1)
    else
        ARGS+=(--esm_model "${ESM2_MODEL}")
    fi

    # Frozen backbone
    [[ "${frozen}" -eq 1 ]] && ARGS+=(--freeze_all_backbone)

    # For unfrozen runs, manage memory carefully
    if [[ "${frozen}" -eq 0 ]]; then
        if [[ "${is_esm3}" -eq 1 ]]; then
            # ESM3 unfrozen: too large for DataParallel with small batch.
            # Use single GPU + batch_size=1 + gradient accumulation (eff. batch=4).
            for i in "${!ARGS[@]}"; do
                if [[ "${ARGS[$i]}" == "--batch_size" ]]; then
                    ARGS[$((i+1))]="1"
                fi
            done
            ARGS+=(--gradient_accumulation_steps 4)
            export CUDA_VISIBLE_DEVICES="${GPU}"
        else
            # ESM2 unfrozen: single GPU (650M fits fine with batch=4)
            export CUDA_VISIBLE_DEVICES="${GPU}"
        fi
    else
        export CUDA_VISIBLE_DEVICES="${GPU}"
    fi

    # Task-specific args
    ARGS+=(--task_type "${task}")
    if [[ "${task}" == "classification" ]]; then
        ARGS+=(--num_classes "${num_classes}")
        ARGS+=(--loss_function focal)
        if [[ ${#thresholds[@]} -gt 0 ]]; then
            ARGS+=(--neq_thresholds "${thresholds[@]}")
        fi
    else
        # Regression – no class args, use default MSE-based loss
        ARGS+=(--loss_function crossentropy)   # maps to MSE in regression branch
    fi

    cd "${SCRIPT_DIR}"
    python main.py "${ARGS[@]}" 2>&1 | tee "${RESULTS_DIR}/${name}/train.log"
    export CUDA_VISIBLE_DEVICES="${GPU}"   # restore after training

    # Save the exact args used for reproducibility
    echo "${ARGS[@]}" > "${RESULTS_DIR}/${name}/args_used.txt"
    info "Training done → ${RESULTS_DIR}/${name}/"
}

# ── Attention extraction helper ─────────────────────────────────────────────────
extract_attn() {
    local name="$1"
    local is_esm3="$2"
    local task="$3"
    local num_classes="$4"   # ignored for regression

    local CKPT="${RESULTS_DIR}/${name}/best_model.pth"
    if [[ ! -f "${CKPT}" ]]; then
        info "WARNING: checkpoint not found at ${CKPT}, skipping extraction for ${name}"
        return
    fi

    local BILSTM_OUT="${RESULTS_DIR}/${name}/bilstm_attn.json"
    local BB_OUT="${RESULTS_DIR}/${name}/backbone_attn.json"

    # ── BiLSTM attention ────────────────────────────────────────────────────
    info "Extracting BiLSTM attention for ${name} …"
    local GET_ARGS=(
        --checkpoint    "${CKPT}"
        --fasta_file    "${FASTA}"
        --architecture  bilstm_attention
        --task_type     "${task}"
        --output        "${BILSTM_OUT}"
    )
    if [[ "${is_esm3}" -eq 1 ]]; then
        GET_ARGS+=(--is_esm3)
    else
        GET_ARGS+=(--esm_model "${ESM2_MODEL}")
    fi
    if [[ "${task}" == "classification" ]]; then
        GET_ARGS+=(--num_classes "${num_classes}")
    else
        GET_ARGS+=(--num_outputs 1)
    fi

    cd "${ATTN_DIR}"
    python get_attn.py "${GET_ARGS[@]}" 2>&1 | tee "${RESULTS_DIR}/${name}/bilstm_attn_extract.log"

    # ── Backbone attention ──────────────────────────────────────────────────
    info "Extracting backbone attention for ${name} …"
    local BB_ARGS=(
        --checkpoint  "${CKPT}"
        --fasta_file  "${FASTA}"
        --output      "${BB_OUT}"
    )
    if [[ "${is_esm3}" -eq 1 ]]; then
        BB_ARGS+=(--is_esm3)
    else
        BB_ARGS+=(--esm_model "${ESM2_MODEL}")
    fi

    cd "${ATTN_DIR}"
    python extract_backbone_attn.py "${BB_ARGS[@]}" 2>&1 | tee "${RESULTS_DIR}/${name}/backbone_attn_extract.log"

    info "Extraction done for ${name}"
}

# ── Analysis helper ─────────────────────────────────────────────────────────────
run_analysis() {
    header "Running unified attention analysis pipeline"
    mkdir -p "${RESULTS_DIR}/analysis"
    cd "${ATTN_DIR}"
    python run_attn_analysis.py \
        --exp_root   "${RESULTS_DIR}" \
        --neq_csv    "${TEST_CSV}" \
        --nsp3_csv   "${DATA_DIR}/test_data_nsp3.csv" \
        --output_dir "${RESULTS_DIR}/analysis" \
        2>&1 | tee "${RESULTS_DIR}/analysis/analysis.log"
    info "Analysis done → ${RESULTS_DIR}/analysis/"
}

# ── Experiment registry ─────────────────────────────────────────────────────────
# Format: name | is_esm3 | frozen | lr | task | num_classes | thresholds (space-sep)
# Using arrays of arrays (bash-style)
declare -a EXP_NAMES=(
    "esm2_binary_frozen"
    "esm2_binary_unfrozen"
    "esm3_binary_frozen"
    "esm3_binary_unfrozen"
    "esm2_regression_frozen"
    "esm2_3class_frozen"
)

# Parallel indexed arrays (same order as EXP_NAMES)
declare -a EXP_IS_ESM3=(  0  0  1  1  0  0 )
declare -a EXP_FROZEN=(   1  0  1  0  1  1 )
declare -a EXP_LR=(       "1e-3" "1e-5" "1e-3" "1e-5" "1e-3" "1e-3" )
declare -a EXP_TASK=(     "classification" "classification" "classification" "classification" "regression" "classification" )
declare -a EXP_NCLASS=(   2  2  2  2  1  3 )
declare -a EXP_THRESH=(   "1.0" "1.0" "1.0" "1.0" "" "1.0 2.0" )

# ── Main ────────────────────────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"

header "ATLAS Attention Experiments Pipeline"
info "Results root : ${RESULTS_DIR}"
info "GPU(s)       : ${GPU}"
info "AMP dtype    : ${AMP_DTYPE}"
echo ""

for i in "${!EXP_NAMES[@]}"; do
    name="${EXP_NAMES[$i]}"

    # Skip if --only was set and this isn't it
    if [[ -n "${ONLY}" && "${name}" != "${ONLY}" ]]; then
        continue
    fi

    is_esm3="${EXP_IS_ESM3[$i]}"
    frozen="${EXP_FROZEN[$i]}"
    lr="${EXP_LR[$i]}"
    task="${EXP_TASK[$i]}"
    nclass="${EXP_NCLASS[$i]}"
    thresh_str="${EXP_THRESH[$i]}"

    # Convert threshold string to array
    thresh_arr=()
    if [[ -n "${thresh_str}" ]]; then
        read -ra thresh_arr <<< "${thresh_str}"
    fi

    header "Experiment ${i}/${#EXP_NAMES[@]}: ${name}"

    if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
        run_training "${name}" "${is_esm3}" "${frozen}" "${lr}" \
                     "${task}" "${nclass}" "${thresh_arr[@]+"${thresh_arr[@]}"}"
    else
        info "Skipping training (--skip_train)"
    fi

    if [[ "${SKIP_EXTRACT}" -eq 0 ]]; then
        extract_attn "${name}" "${is_esm3}" "${task}" "${nclass}"
    else
        info "Skipping extraction (--skip_extract)"
    fi
done

if [[ "${SKIP_ANALYSIS}" -eq 0 ]]; then
    run_analysis
else
    info "Skipping analysis (--skip_analysis)"
fi

header "ALL DONE"
echo "Results in: ${RESULTS_DIR}"
