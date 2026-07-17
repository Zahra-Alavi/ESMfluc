#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULT_ROOT="${RESULT_ROOT:-results/publication_comparable_v2}"
MANIFEST="${MANIFEST:-${RESULT_ROOT}/manifest.tsv}"
PYTHON_BIN="${PYTHON_BIN:-/home/zahralab/miniconda/envs/esm_env/bin/python}"
GPU_ESM2="${GPU_ESM2:-0}"
GPU_ESM3="${GPU_ESM3:-1}"
RESUME="${RESUME:-1}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-4}"
EXTRACTOR="Attention/extract_all_split_signed_contributions.py"
LOG_ROOT="${RESULT_ROOT}/logs/all_split_signed_contributions"

TRAIN_CSV="${RESULT_ROOT}/train_data.csv"
VALIDATION_CSV="${RESULT_ROOT}/validation_data.csv"
TEST_CSV="${RESULT_ROOT}/test_data.csv"

for required in "$MANIFEST" "$TRAIN_CSV" "$VALIDATION_CSV" "$TEST_CSV" "$EXTRACTOR"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing required file: $required" >&2
        exit 1
    fi
done

mkdir -p "$LOG_ROOT"

mapfile -t ESM2_RUNS < <(awk -F '\t' 'NR > 1 && $3 == "bilstm_attention" && $5 == "false" {print $1 "\t" $2 "\t" $4 "\t" $12}' "$MANIFEST")
mapfile -t ESM3_RUNS < <(awk -F '\t' 'NR > 1 && $3 == "bilstm_attention" && $5 == "true"  {print $1 "\t" $2 "\t" $4 "\t" $12}' "$MANIFEST")

if [[ "${#ESM2_RUNS[@]}" -ne 9 || "${#ESM3_RUNS[@]}" -ne 9 ]]; then
    echo "ERROR: expected 9 ESM2 and 9 ESM3 BiLSTM runs; found ${#ESM2_RUNS[@]} and ${#ESM3_RUNS[@]}" >&2
    exit 1
fi

run_queue() {
    local gpu="$1"
    local is_esm3="$2"
    shift 2
    local run condition seed esm_model checkpoint run_dir output_dir log_path
    for run in "$@"; do
        IFS=$'\t' read -r condition seed esm_model checkpoint <<< "$run"
        run_dir="${RESULT_ROOT}/runs/${condition}/seed_${seed}"
        output_dir="${run_dir}/all_split_signed_contributions"
        log_path="${LOG_ROOT}/${condition}_seed_${seed}.log"
        mkdir -p "$output_dir"
        args=(
            --checkpoint "$checkpoint"
            --condition "$condition"
            --seed "$seed"
            --esm_model "$esm_model"
            --split train "$TRAIN_CSV"
            --split validation "$VALIDATION_CSV"
            --split test "$TEST_CSV"
            --output_dir "$output_dir"
            --compression_level "$COMPRESSION_LEVEL"
        )
        if [[ "$RESUME" == "1" ]]; then
            args+=(--resume)
        fi
        if [[ "$is_esm3" == "1" ]]; then
            args+=(--is_esm3)
        fi
        echo "START gpu=${gpu} condition=${condition} seed=${seed} $(date --iso-8601=seconds)" | tee -a "$log_path"
        if CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PYTHON_BIN" "$EXTRACTOR" "${args[@]}" >> "$log_path" 2>&1; then
            echo "DONE gpu=${gpu} condition=${condition} seed=${seed} $(date --iso-8601=seconds)" | tee -a "$log_path"
        else
            status=$?
            echo "FAILED status=${status} gpu=${gpu} condition=${condition} seed=${seed} $(date --iso-8601=seconds)" | tee -a "$log_path"
            return "$status"
        fi
    done
}

echo "Launching ${#ESM2_RUNS[@]} ESM2 runs on physical GPU ${GPU_ESM2} and ${#ESM3_RUNS[@]} ESM3 runs on physical GPU ${GPU_ESM3}."
run_queue "$GPU_ESM2" 0 "${ESM2_RUNS[@]}" &
PID_ESM2=$!
run_queue "$GPU_ESM3" 1 "${ESM3_RUNS[@]}" &
PID_ESM3=$!

status=0
wait "$PID_ESM2" || status=$?
wait "$PID_ESM3" || status=$?
if [[ "$status" -eq 0 ]]; then
    "$PYTHON_BIN" Attention/audit_all_split_signed_contributions.py \
        --result_root "$RESULT_ROOT"
fi
exit "$status"
