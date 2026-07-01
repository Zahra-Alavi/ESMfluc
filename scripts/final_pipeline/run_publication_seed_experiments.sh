#!/usr/bin/env bash
# =============================================================================
# Run publication-oriented comparable ESMfluc experiments.
#
# Conditions, each across 3 seeds by default:
#   1. ESM2 frozen BiLSTM-attention
#   2. ESM2 top-4-layer finetuned BiLSTM-attention
#   3. ESM3 frozen BiLSTM-attention
#   4. ESM3 top-4-layer finetuned BiLSTM-attention
#   5. ESM2 frozen linear classifier baseline
#
# Outputs are kept under one distinct result set:
#   results/<RESULT_SET>/runs/<condition>/seed_<seed>/
#
# Usage from scripts/final_pipeline/:
#   bash run_publication_seed_experiments.sh
#
# Useful overrides:
#   RESULT_SET=publication_comparable_v1
#   SEEDS="42 123 2025"
#   BATCH=1
#   EPOCHS=80
#   GPU_ESM2=0
#   GPU_ESM3=1
#   MIXED_PRECISION=1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULT_SET="${RESULT_SET:-publication_comparable_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="results/${RESULT_SET}"
LOG_DIR="${RESULT_ROOT}/logs"
MANIFEST="${RESULT_ROOT}/manifest.tsv"

mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/train_and_extract.log") 2>&1

TRAIN_CSV="${TRAIN_CSV:-../../data/train_data.csv}"
TEST_CSV="${TEST_CSV:-../../data/test_data.csv}"
TEST_CSV_WITH_NAMES="${TEST_CSV_WITH_NAMES:-../../data/test_data_with_names.csv}"
FASTA="${FASTA:-../../data/test_data_sequences.fasta}"

SEEDS="${SEEDS:-42 123 2025}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-1}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
LOSS_FUNCTION="${LOSS_FUNCTION:-focal}"
GPU_ESM2="${GPU_ESM2:-0}"
GPU_ESM3="${GPU_ESM3:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ESM2_MODEL="${ESM2_MODEL:-esm2_t33_650M_UR50D}"
ESM3_MODEL="${ESM3_MODEL:-esm3_sm_open_v1}"
ESM2_TOP4_FREEZE="${ESM2_TOP4_FREEZE:-0-28}"
ESM3_TOP4_FREEZE="${ESM3_TOP4_FREEZE:-0-43}"

enabled() {
    case "$1" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

echo "Result set: ${RESULT_SET}"
echo "Result root: ${SCRIPT_DIR}/${RESULT_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Batch size used for every run: ${BATCH}"
echo "ESM2 top-4 freeze range: ${ESM2_TOP4_FREEZE}"
echo "ESM3 top-4 freeze range: ${ESM3_TOP4_FREEZE}"
echo ""

echo "condition	seed	architecture	esm_model	is_esm3	freeze_mode	freeze_layers	run_dir	attention_json	checkpoint" > "$MANIFEST"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available. These experiments should be run on the remote GPU machine.")
print(f"CUDA OK: {torch.cuda.device_count()} GPU(s) visible")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

python - <<'PY'
try:
    from esm.pretrained import ESM3_sm_open_v0  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ERROR: ESM3 library is not available: {type(exc).__name__}: {exc}")
print("ESM3 import OK")
PY

common_train_args=(
    --train_data_file "$TRAIN_CSV"
    --test_data_file "$TEST_CSV"
    --hidden_size 512
    --num_layers 3
    --dropout 0.3
    --bidirectional 1
    --num_classes 2
    --neq_thresholds 1.0
    --loss_function "$LOSS_FUNCTION"
    --lr_scheduler reduce_on_plateau
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --batch_size "$BATCH"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --device cuda
    --task_type classification
)

if enabled "$MIXED_PRECISION"; then
    common_train_args+=(--mixed_precision --amp_dtype "$AMP_DTYPE")
fi

run_one() {
    local condition="$1"
    local seed="$2"
    local architecture="$3"
    local esm_model="$4"
    local is_esm3="$5"
    local freeze_mode="$6"
    local freeze_layers="$7"
    local gpu_id="$8"

    local run_rel="${RESULT_SET}/runs/${condition}/seed_${seed}"
    local run_dir="${RESULT_ROOT}/runs/${condition}/seed_${seed}"
    local checkpoint="${run_dir}/best_model.pth"
    local attention_json="${run_dir}/attention.json"

    mkdir -p "$run_dir"

    echo ""
    echo "============================================================"
    echo "Condition: ${condition}"
    echo "Seed: ${seed}"
    echo "Architecture: ${architecture}"
    echo "Backbone: ${esm_model}"
    echo "Freeze mode: ${freeze_mode} ${freeze_layers}"
    echo "GPU: ${gpu_id}"
    echo "Run dir: ${run_dir}"
    echo "============================================================"

    if [[ ! -f "$checkpoint" || "$SKIP_EXISTING" == "0" ]]; then
        train_args=(
            "${common_train_args[@]}"
            --architecture "$architecture"
            --esm_model "$esm_model"
            --seed "$seed"
            --result_foldername "$run_rel"
        )

        case "$freeze_mode" in
            frozen)
                train_args+=(--freeze_all_backbone)
                ;;
            top4)
                train_args+=(--freeze_layers "$freeze_layers")
                ;;
            none)
                ;;
            *)
                echo "ERROR: unknown freeze mode ${freeze_mode}"
                exit 1
                ;;
        esac

        CUDA_VISIBLE_DEVICES="$gpu_id" python main.py "${train_args[@]}" \
            2>&1 | tee "${run_dir}/train.log"
    else
        echo "Checkpoint exists and SKIP_EXISTING=${SKIP_EXISTING}; skipping training."
    fi

    if [[ ! -f "$checkpoint" && -f "${run_dir}/last_model.pth" ]]; then
        checkpoint="${run_dir}/last_model.pth"
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: no checkpoint found for ${condition} seed ${seed}"
        exit 1
    fi

    if [[ ! -f "$attention_json" || "$SKIP_EXISTING" == "0" ]]; then
        attn_args=(
            --checkpoint "$checkpoint"
            --fasta_file "$FASTA"
            --architecture "$architecture"
            --esm_model "$esm_model"
            --hidden_size 512
            --num_layers 3
            --dropout 0.0
            --bidirectional 1
            --num_classes 2
            --output "$attention_json"
        )
        if [[ "$is_esm3" == "true" ]]; then
            attn_args+=(--is_esm3)
        fi

        CUDA_VISIBLE_DEVICES="$gpu_id" python Attention/get_attn.py "${attn_args[@]}" \
            2>&1 | tee "${run_dir}/attention.log"
    else
        echo "Attention JSON exists and SKIP_EXISTING=${SKIP_EXISTING}; skipping extraction."
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$condition" "$seed" "$architecture" "$esm_model" "$is_esm3" \
        "$freeze_mode" "$freeze_layers" "$run_dir" "$attention_json" "$checkpoint" \
        >> "$MANIFEST"
}

for seed in $SEEDS; do
    run_one "esm2_frozen_bilstm_attn" "$seed" "bilstm_attention" "$ESM2_MODEL" "false" "frozen" "" "$GPU_ESM2"
    run_one "esm2_top4_bilstm_attn" "$seed" "bilstm_attention" "$ESM2_MODEL" "false" "top4" "$ESM2_TOP4_FREEZE" "$GPU_ESM2"
    run_one "esm3_frozen_bilstm_attn" "$seed" "bilstm_attention" "$ESM3_MODEL" "true" "frozen" "" "$GPU_ESM3"
    run_one "esm3_top4_bilstm_attn" "$seed" "bilstm_attention" "$ESM3_MODEL" "true" "top4" "$ESM3_TOP4_FREEZE" "$GPU_ESM3"
    run_one "esm2_frozen_linear" "$seed" "esm_linear" "$ESM2_MODEL" "false" "frozen" "" "$GPU_ESM2"
done

cp "$TEST_CSV" "${RESULT_ROOT}/test_data.csv"
cp "$FASTA" "${RESULT_ROOT}/test_data_sequences.fasta"
if [[ -f "$TEST_CSV_WITH_NAMES" ]]; then
    cp "$TEST_CSV_WITH_NAMES" "${RESULT_ROOT}/test_data_with_names.csv"
fi

echo ""
echo "All training and attention extraction jobs completed."
echo "Manifest: ${MANIFEST}"
echo "Result root: ${RESULT_ROOT}"
