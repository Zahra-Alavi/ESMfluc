#!/usr/bin/env bash
# =============================================================================
# Run publication-oriented comparable ESMfluc experiments.
#
# Conditions, each across 3 seeds by default:
#   1. ESM2 frozen BiLSTM-attention
#   2. ESM2 top-4-layer finetuned BiLSTM-attention
#   3. ESM2 top-28-layer finetuned BiLSTM-attention
#   4. ESM3 frozen BiLSTM-attention
#   5. ESM3 top-4-layer finetuned BiLSTM-attention
#   6. ESM3 top-28-layer finetuned BiLSTM-attention
#   7. ESM2 frozen linear classifier baseline
#
# Two independent, runtime-balanced worker queues keep both GPUs occupied.
# A failed run is recorded and its worker immediately advances to the next run.
#
# Outputs are kept under one distinct result set:
#   results/<RESULT_SET>/runs/<condition>/seed_<seed>/
#
# Usage from scripts/final_pipeline/:
#   bash run_publication_seed_experiments.sh
#
# Useful overrides:
#   RESULT_SET=publication_comparable_v2
#   SEEDS="1 2 3"
#   BATCH=2
#   EPOCHS=80
#   GPU_ESM2=0
#   GPU_ESM3=1
#   MIXED_PRECISION=1
#   EXTRACT_ESM2_BACKBONE_ATTN=1
#   REEXTRACT_ATTENTION=0
#   PYTHON_BIN=python
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULT_SET="${RESULT_SET:-publication_comparable_v2}"
RESULT_ROOT="results/${RESULT_SET}"
LOG_DIR="${RESULT_ROOT}/logs"
MANIFEST="${RESULT_ROOT}/manifest.tsv"
MANIFEST_HEADER="condition	seed	architecture	esm_model	is_esm3	freeze_mode	freeze_layers	run_dir	attention_json	logit_contributions_npz	backbone_attention_json	checkpoint"

SPLIT_DIR="${SPLIT_DIR:-data_splits/atlas_grouped_v1}"
TRAIN_CSV="${TRAIN_CSV:-${SPLIT_DIR}/train_grouped_v1.csv}"
VAL_CSV="${VAL_CSV:-${SPLIT_DIR}/validation_grouped_v1.csv}"
TEST_CSV="${TEST_CSV:-${SPLIT_DIR}/test_grouped_v1.csv}"
TEST_CSV_WITH_NAMES="${TEST_CSV_WITH_NAMES:-${TEST_CSV}}"
FASTA="${FASTA:-${SPLIT_DIR}/test_grouped_v1.fasta}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${SPLIT_DIR}/split_manifest_grouped_v1.csv}"
GROUP_MANIFEST="${GROUP_MANIFEST:-${SPLIT_DIR}/group_manifest_grouped_v1.csv}"
SPLIT_SUMMARY="${SPLIT_SUMMARY:-${SPLIT_DIR}/split_summary_grouped_v1.json}"

SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-2}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
LOSS_FUNCTION="${LOSS_FUNCTION:-focal}"
GPU_ESM2="${GPU_ESM2:-0}"
GPU_ESM3="${GPU_ESM3:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REEXTRACT_ATTENTION="${REEXTRACT_ATTENTION:-0}"
EXTRACT_ESM2_BACKBONE_ATTN="${EXTRACT_ESM2_BACKBONE_ATTN:-1}"
BACKBONE_OUTPUT_NAME="${BACKBONE_OUTPUT_NAME:-backbone_attention.json}"
EXTRACT_LOGIT_CONTRIBUTIONS="${EXTRACT_LOGIT_CONTRIBUTIONS:-1}"
LOGIT_CONTRIBUTION_OUTPUT_NAME="${LOGIT_CONTRIBUTION_OUTPUT_NAME:-flex_rigid_logit_contributions.npz}"
RESUME="${RESUME:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

ESM2_MODEL="${ESM2_MODEL:-esm2_t33_650M_UR50D}"
ESM3_MODEL="${ESM3_MODEL:-esm3_sm_open_v1}"
ESM2_TOP4_FREEZE="${ESM2_TOP4_FREEZE:-0-28}"
ESM3_TOP4_FREEZE="${ESM3_TOP4_FREEZE:-0-43}"
ESM2_TOP28_FREEZE="${ESM2_TOP28_FREEZE:-0-4}"
ESM3_TOP28_FREEZE="${ESM3_TOP28_FREEZE:-0-19}"

enabled() {
    case "$1" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ -d "$RESULT_ROOT" ]] && [[ -n "$(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    && ! enabled "$RESUME"; then
    echo "ERROR: ${RESULT_ROOT} already exists and is non-empty." >&2
    echo "Use a new RESULT_SET, or set RESUME=1 explicitly to resume it." >&2
    exit 1
fi

echo "Result set: ${RESULT_SET}"
echo "Result root: ${SCRIPT_DIR}/${RESULT_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Fixed validation CSV: ${VAL_CSV}"
echo "Batch size used for every run: ${BATCH}"
echo "ESM2 top-4 freeze range: ${ESM2_TOP4_FREEZE}"
echo "ESM3 top-4 freeze range: ${ESM3_TOP4_FREEZE}"
echo "ESM2 top-28 freeze range: ${ESM2_TOP28_FREEZE}"
echo "ESM3 top-28 freeze range: ${ESM3_TOP28_FREEZE}"
echo "Extract ESM2 backbone attention: ${EXTRACT_ESM2_BACKBONE_ATTN}"
echo "Extract exact flex-minus-rigid logit contributions: ${EXTRACT_LOGIT_CONTRIBUTIONS}"
echo "Re-extract model predictions/attention: ${REEXTRACT_ATTENTION}"
echo "Resume existing result set: ${RESUME}"
echo "PyTorch CUDA allocator: ${PYTORCH_CUDA_ALLOC_CONF}"
echo ""

for required_file in "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$FASTA" \
                     "$SPLIT_MANIFEST" "$GROUP_MANIFEST" "$SPLIT_SUMMARY"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required fixed-split file not found: ${required_file}" >&2
        exit 1
    fi
done

"$PYTHON_BIN" - "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV" "$FASTA" \
           "$SPLIT_MANIFEST" "$SPLIT_SUMMARY" <<'PY'
import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

train_path, val_path, test_path, fasta_path, manifest_path, summary_path = map(Path, sys.argv[1:])
frames = {
    "train": pd.read_csv(train_path),
    "validation": pd.read_csv(val_path),
    "test": pd.read_csv(test_path),
}
manifest = pd.read_csv(manifest_path)
summary = json.loads(summary_path.read_text())

for split, frame in frames.items():
    required = {"name", "sequence", "neq"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"ERROR: {split} CSV lacks {sorted(required - set(frame.columns))}")
    if frame["name"].duplicated().any():
        raise SystemExit(f"ERROR: duplicate names in {split} CSV")
    for row in frame.itertuples(index=False):
        if len(row.sequence) != len(ast.literal_eval(row.neq)):
            raise SystemExit(f"ERROR: sequence/Neq length mismatch for {row.name}")
    expected_count = int(summary["counts"][split])
    if len(frame) != expected_count:
        raise SystemExit(
            f"ERROR: {split} count {len(frame)} does not match split summary {expected_count}"
        )

for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
    overlap = set(frames[left].sequence) & set(frames[right].sequence)
    if overlap:
        raise SystemExit(f"ERROR: {len(overlap)} exact sequences shared by {left} and {right}")

if any(summary.get("leakage_failures", {}).values()):
    raise SystemExit(f"ERROR: recorded grouped-split leakage: {summary['leakage_failures']}")

for relative_path, expected in summary.get("output_sha256", {}).items():
    path = summary_path.parent / relative_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"ERROR: checksum mismatch for {path}")

for split, frame in frames.items():
    expected_names = set(manifest.loc[manifest["split"] == split, "name"])
    if set(frame["name"]) != expected_names:
        raise SystemExit(f"ERROR: {split} CSV membership differs from split manifest")

fasta_records = {}
name, sequence = None, []
for line in fasta_path.read_text().splitlines():
    if line.startswith(">"):
        if name is not None:
            fasta_records[name] = "".join(sequence)
        name, sequence = line[1:], []
    else:
        sequence.append(line.strip())
if name is not None:
    fasta_records[name] = "".join(sequence)
expected_test = dict(zip(frames["test"].name, frames["test"].sequence))
if fasta_records != expected_test:
    raise SystemExit("ERROR: test FASTA does not exactly match the fixed test CSV")

print("Fixed grouped-split audit passed:", {key: len(value) for key, value in frames.items()})
PY

"$PYTHON_BIN" - "$GPU_ESM2" "$GPU_ESM3" <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available. These experiments should be run on the remote GPU machine.")
count = torch.cuda.device_count()
print(f"CUDA OK: {count} GPU(s) visible")
for i in range(count):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
for label, value in (("GPU_ESM2", sys.argv[1]), ("GPU_ESM3", sys.argv[2])):
    try:
        index = int(value)
    except ValueError:
        raise SystemExit(f"ERROR: {label} must be one integer GPU index; got {value!r}")
    if not 0 <= index < count:
        raise SystemExit(f"ERROR: {label}={index}, but only GPU indices 0-{count - 1} exist")
if int(sys.argv[1]) == int(sys.argv[2]):
    raise SystemExit(
        "ERROR: GPU_ESM2 and GPU_ESM3 must be different for parallel execution."
    )
PY

"$PYTHON_BIN" - <<'PY'
try:
    from esm.pretrained import ESM3_sm_open_v0  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ERROR: ESM3 library is not available: {type(exc).__name__}: {exc}")
print("ESM3 import OK")
PY

# Do not initialize the result set until every read-only preflight has passed.
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/train_and_extract.log") 2>&1

PROVENANCE_DIR="${RESULT_ROOT}/pipeline_provenance"
mkdir -p "$PROVENANCE_DIR"
cp main.py train.py arguments.py models.py data_utils.py \
   Attention/get_attn.py Attention/extract_backbone_attn.py \
   run_publication_seed_experiments.sh "$PROVENANCE_DIR/"
{
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    echo "result_set=${RESULT_SET}"
    echo "seeds=${SEEDS}"
    echo "batch_size=${BATCH}"
    echo "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
    echo "train_csv=${TRAIN_CSV}"
    echo "validation_csv=${VAL_CSV}"
    echo "test_csv=${TEST_CSV}"
    git status --short 2>/dev/null || true
} > "${PROVENANCE_DIR}/run_environment.txt"

if [[ ! -s "$MANIFEST" ]]; then
    printf '%b\n' "$MANIFEST_HEADER" > "$MANIFEST"
fi

common_train_args=(
    --train_data_file "$TRAIN_CSV"
    --validation_data_file "$VAL_CSV"
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
    local worker_manifest="$9"

    local run_rel="${RESULT_SET}/runs/${condition}/seed_${seed}"
    local run_dir="${RESULT_ROOT}/runs/${condition}/seed_${seed}"
    local checkpoint="${run_dir}/best_model.pth"
    local attention_json="${run_dir}/attention.json"
    local logit_contributions_npz="${run_dir}/${LOGIT_CONTRIBUTION_OUTPUT_NAME}"
    local manifest_logit_contributions_npz=""
    local backbone_attention_json="${run_dir}/${BACKBONE_OUTPUT_NAME}"
    local manifest_backbone_attention_json=""
    local stage_file="${run_dir}/pipeline_stage.txt"

    mkdir -p "$run_dir"
    echo "setup" > "$stage_file"

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

    # A best checkpoint can be written several epochs before training ends.
    # Require final evaluation artifacts before treating training as complete,
    # otherwise an OOM-interrupted run could be mistaken for a finished run.
    local training_complete=1
    for artifact in "$checkpoint" "${run_dir}/metrics.json" "${run_dir}/run_summary.csv"; do
        if [[ ! -s "$artifact" ]]; then
            training_complete=0
        fi
    done

    if [[ "$training_complete" == "0" || "$SKIP_EXISTING" == "0" ]]; then
        if [[ -f "$checkpoint" && "$training_complete" == "0" ]]; then
            echo "Checkpoint exists, but final training/evaluation artifacts are missing; retraining from seed ${seed}."
        fi
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
            top4|top28)
                train_args+=(--freeze_layers "$freeze_layers")
                ;;
            none)
                ;;
            *)
                echo "ERROR: unknown freeze mode ${freeze_mode}"
                exit 1
                ;;
        esac

        echo "training" > "$stage_file"
        PYTHONHASHSEED="$seed" CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" main.py "${train_args[@]}" \
            2>&1 | tee "${run_dir}/train.log"
    else
        echo "Training and evaluation artifacts exist; skipping completed training."
    fi

    if [[ ! -f "$checkpoint" && -f "${run_dir}/last_model.pth" ]]; then
        checkpoint="${run_dir}/last_model.pth"
    fi
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: no checkpoint found for ${condition} seed ${seed}"
        exit 1
    fi

    local attention_complete=1
    if [[ ! -s "$attention_json" || ! -s "${run_dir}/attention.log" ]] \
        || ! grep -Fq "Constructed final DF with 208 rows" "${run_dir}/attention.log" \
        || ! grep -Fq "Saved final JSON" "${run_dir}/attention.log"; then
        attention_complete=0
    fi
    if enabled "$EXTRACT_LOGIT_CONTRIBUTIONS" \
        && [[ "$architecture" == "bilstm_attention" ]] \
        && { [[ ! -s "$logit_contributions_npz" ]] \
             || ! grep -Fq "Saved 208 exact flex-minus-rigid contribution matrices" "${run_dir}/attention.log"; }; then
        attention_complete=0
    fi

    local need_attention_extraction=0
    if [[ "$attention_complete" == "0" || "$SKIP_EXISTING" == "0" ]] \
        || enabled "$REEXTRACT_ATTENTION"; then
        need_attention_extraction=1
    fi

    if [[ "$need_attention_extraction" == "1" ]]; then
        echo "attention_extraction" > "$stage_file"
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
        if enabled "$EXTRACT_LOGIT_CONTRIBUTIONS" \
            && [[ "$architecture" == "bilstm_attention" ]]; then
            attn_args+=(--logit_contributions_output "$logit_contributions_npz")
        fi

        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" Attention/get_attn.py "${attn_args[@]}" \
            2>&1 | tee "${run_dir}/attention.log"
    else
        echo "Attention JSON exists and SKIP_EXISTING=${SKIP_EXISTING}; skipping extraction."
    fi

    if enabled "$EXTRACT_LOGIT_CONTRIBUTIONS" \
        && [[ "$architecture" == "bilstm_attention" ]]; then
        if [[ ! -f "$logit_contributions_npz" ]]; then
            echo "ERROR: missing exact logit-contribution sidecar: ${logit_contributions_npz}" >&2
            exit 1
        fi
        manifest_logit_contributions_npz="$logit_contributions_npz"
    fi

    if enabled "$EXTRACT_ESM2_BACKBONE_ATTN" \
        && [[ "$architecture" == "bilstm_attention" ]] \
        && [[ "$is_esm3" == "false" ]]; then
        local backbone_complete=1
        if [[ ! -s "$backbone_attention_json" || ! -s "${run_dir}/backbone_attention.log" ]] \
            || ! grep -Fq "Saved 208 records" "${run_dir}/backbone_attention.log"; then
            backbone_complete=0
        fi
        local backbone_was_extracted=0
        if [[ "$backbone_complete" == "0" || "$SKIP_EXISTING" == "0" ]]; then
            echo "backbone_attention_extraction" > "$stage_file"
            CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" Attention/extract_backbone_attn.py \
                --checkpoint "$checkpoint" \
                --fasta_file "$FASTA" \
                --esm_model "$esm_model" \
                --output "$backbone_attention_json" \
                2>&1 | tee "${run_dir}/backbone_attention.log"
            backbone_was_extracted=1
        else
            echo "Backbone attention JSON exists and SKIP_EXISTING=${SKIP_EXISTING}; skipping extraction."
        fi

        if [[ "$need_attention_extraction" == "1" || "$backbone_was_extracted" == "1" \
              || "$SKIP_EXISTING" == "0" ]] \
            && [[ -f "$backbone_attention_json" && -f "$attention_json" ]]; then
            echo "backbone_prediction_merge" > "$stage_file"
            "$PYTHON_BIN" - "$backbone_attention_json" "$attention_json" <<'PY'
import json
import sys
from pathlib import Path

backbone_path = Path(sys.argv[1])
attention_path = Path(sys.argv[2])
backbone = json.loads(backbone_path.read_text())
attention = json.loads(attention_path.read_text())
by_name = {record["name"]: record for record in attention}
backbone_names = {record["name"] for record in backbone}
if backbone_names != set(by_name):
    missing = sorted(set(by_name) - backbone_names)
    extra = sorted(backbone_names - set(by_name))
    raise SystemExit(
        f"Backbone/BiLSTM attention protein mismatch: missing={missing[:10]}, extra={extra[:10]}"
    )
for record in backbone:
    src = by_name[record["name"]]
    for key in ("neq_preds", "flexible_scores", "class_probs", "ss_pred"):
        if key in src:
            record[key] = src[key]
backbone_path.write_text(json.dumps(backbone, indent=2) + "\n")
PY
        fi
        manifest_backbone_attention_json="$backbone_attention_json"
    fi

    echo "finalizing" > "$stage_file"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$condition" "$seed" "$architecture" "$esm_model" "$is_esm3" \
        "$freeze_mode" "$freeze_layers" "$run_dir" "$attention_json" \
        "$manifest_logit_contributions_npz" "$manifest_backbone_attention_json" "$checkpoint" \
        >> "$worker_manifest"
    echo "completed" > "$stage_file"
}

WORKER_STATE_DIR="${RESULT_ROOT}/parallel_workers"
GPU0_WORKER_MANIFEST="${WORKER_STATE_DIR}/gpu_${GPU_ESM2}_manifest_rows.tsv"
GPU1_WORKER_MANIFEST="${WORKER_STATE_DIR}/gpu_${GPU_ESM3}_manifest_rows.tsv"
GPU0_FAILURES="${WORKER_STATE_DIR}/gpu_${GPU_ESM2}_failures.tsv"
GPU1_FAILURES="${WORKER_STATE_DIR}/gpu_${GPU_ESM3}_failures.tsv"
GPU0_WORKER_LOG="${LOG_DIR}/gpu_${GPU_ESM2}_worker.log"
GPU1_WORKER_LOG="${LOG_DIR}/gpu_${GPU_ESM3}_worker.log"
FAILURE_MANIFEST="${RESULT_ROOT}/failures.tsv"

mkdir -p "$WORKER_STATE_DIR"
: > "$GPU0_WORKER_MANIFEST"
: > "$GPU1_WORKER_MANIFEST"
printf "timestamp\tworker\tgpu\tcondition\tseed\tstage\texit_code\trun_dir\tworker_log\n" > "$GPU0_FAILURES"
printf "timestamp\tworker\tgpu\tcondition\tseed\tstage\texit_code\trun_dir\tworker_log\n" > "$GPU1_FAILURES"

execute_one() {
    local worker_name="$1"
    local worker_log="$2"
    local worker_manifest="$3"
    local worker_failures="$4"
    shift 4

    local condition="$1"
    local seed="$2"
    local gpu_id="$8"
    local run_dir="${RESULT_ROOT}/runs/${condition}/seed_${seed}"
    local stage_file="${run_dir}/pipeline_stage.txt"
    local exit_code

    # Isolate errexit/pipefail to this one run. A failure returns to the worker,
    # which records it and immediately advances to its next queued condition.
    set +e
    (
        set -Eeuo pipefail
        run_one "$@" "$worker_manifest"
    ) > >(tee -a "$worker_log") 2>&1
    exit_code=$?
    set -e

    if [[ "$exit_code" -ne 0 ]]; then
        local stage="unknown"
        if [[ -s "$stage_file" ]]; then
            stage="$(tr -d '\r\n' < "$stage_file")"
        fi
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date --iso-8601=seconds)" "$worker_name" "$gpu_id" \
            "$condition" "$seed" "$stage" "$exit_code" "$run_dir" "$worker_log" \
            >> "$worker_failures"
        echo "[${worker_name}] FAILED: ${condition} seed ${seed}, stage=${stage}, exit=${exit_code}. Continuing."
        return 0
    fi

    echo "[${worker_name}] COMPLETED: ${condition} seed ${seed}."
}

run_gpu0_balanced_queue() {
    local worker="gpu_${GPU_ESM2}_balanced"
    echo "[${worker}] Starting balanced queue on physical GPU ${GPU_ESM2}."
    for seed in $SEEDS; do
        execute_one "$worker" "$GPU0_WORKER_LOG" "$GPU0_WORKER_MANIFEST" "$GPU0_FAILURES" \
            "esm2_frozen_bilstm_attn" "$seed" "bilstm_attention" "$ESM2_MODEL" "false" "frozen" "" "$GPU_ESM2"
        execute_one "$worker" "$GPU0_WORKER_LOG" "$GPU0_WORKER_MANIFEST" "$GPU0_FAILURES" \
            "esm2_top4_bilstm_attn" "$seed" "bilstm_attention" "$ESM2_MODEL" "false" "top4" "$ESM2_TOP4_FREEZE" "$GPU_ESM2"
        execute_one "$worker" "$GPU0_WORKER_LOG" "$GPU0_WORKER_MANIFEST" "$GPU0_FAILURES" \
            "esm2_top28_bilstm_attn" "$seed" "bilstm_attention" "$ESM2_MODEL" "false" "top28" "$ESM2_TOP28_FREEZE" "$GPU_ESM2"
        execute_one "$worker" "$GPU0_WORKER_LOG" "$GPU0_WORKER_MANIFEST" "$GPU0_FAILURES" \
            "esm2_frozen_linear" "$seed" "esm_linear" "$ESM2_MODEL" "false" "frozen" "" "$GPU_ESM2"
        # Moving the light ESM3-frozen condition here balances measured queue
        # time while keeping the memory-heavy ESM3 runs on the other GPU.
        execute_one "$worker" "$GPU0_WORKER_LOG" "$GPU0_WORKER_MANIFEST" "$GPU0_FAILURES" \
            "esm3_frozen_bilstm_attn" "$seed" "bilstm_attention" "$ESM3_MODEL" "true" "frozen" "" "$GPU_ESM2"
    done
    echo "[${worker}] Queue finished."
}

run_gpu1_esm3_queue() {
    local worker="gpu_${GPU_ESM3}_esm3"
    echo "[${worker}] Starting ESM3 queue on physical GPU ${GPU_ESM3}."
    for seed in $SEEDS; do
        execute_one "$worker" "$GPU1_WORKER_LOG" "$GPU1_WORKER_MANIFEST" "$GPU1_FAILURES" \
            "esm3_top4_bilstm_attn" "$seed" "bilstm_attention" "$ESM3_MODEL" "true" "top4" "$ESM3_TOP4_FREEZE" "$GPU_ESM3"
        execute_one "$worker" "$GPU1_WORKER_LOG" "$GPU1_WORKER_MANIFEST" "$GPU1_FAILURES" \
            "esm3_top28_bilstm_attn" "$seed" "bilstm_attention" "$ESM3_MODEL" "true" "top28" "$ESM3_TOP28_FREEZE" "$GPU_ESM3"
    done
    echo "[${worker}] Queue finished."
}

echo "Launching two independent GPU queues."
echo "  GPU ${GPU_ESM2}: ESM2 conditions plus ESM3 frozen (balanced queue)"
echo "  GPU ${GPU_ESM3}: ESM3 top-4 and top-28"
run_gpu0_balanced_queue &
gpu0_worker_pid=$!
run_gpu1_esm3_queue &
gpu1_worker_pid=$!

set +e
wait "$gpu0_worker_pid"
gpu0_worker_status=$?
wait "$gpu1_worker_pid"
gpu1_worker_status=$?
set -e
if [[ "$gpu0_worker_status" -ne 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date --iso-8601=seconds)" "gpu_${GPU_ESM2}_balanced" "$GPU_ESM2" \
        "WORKER_PROCESS" "NA" "worker" "$gpu0_worker_status" "NA" "$GPU0_WORKER_LOG" \
        >> "$GPU0_FAILURES"
fi
if [[ "$gpu1_worker_status" -ne 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date --iso-8601=seconds)" "gpu_${GPU_ESM3}_esm3" "$GPU_ESM3" \
        "WORKER_PROCESS" "NA" "worker" "$gpu1_worker_status" "NA" "$GPU1_WORKER_LOG" \
        >> "$GPU1_FAILURES"
fi

# Rebuild the canonical manifest deterministically after both workers finish.
"$PYTHON_BIN" - "$GPU0_WORKER_MANIFEST" "$GPU1_WORKER_MANIFEST" "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

worker_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
output_path = Path(sys.argv[3])
header = [
    "condition", "seed", "architecture", "esm_model", "is_esm3",
    "freeze_mode", "freeze_layers", "run_dir", "attention_json",
    "logit_contributions_npz", "backbone_attention_json", "checkpoint",
]
condition_order = {
    name: index for index, name in enumerate([
        "esm2_frozen_bilstm_attn", "esm2_top4_bilstm_attn",
        "esm2_top28_bilstm_attn", "esm3_frozen_bilstm_attn",
        "esm3_top4_bilstm_attn", "esm3_top28_bilstm_attn",
        "esm2_frozen_linear",
    ])
}
rows = []
seen = set()
for path in worker_paths:
    for raw in path.read_text().splitlines():
        if not raw:
            continue
        values = raw.split("\t")
        if len(values) != len(header):
            raise SystemExit(f"Malformed worker manifest row in {path}: {raw!r}")
        row = dict(zip(header, values))
        key = (row["condition"], row["seed"])
        if key in seen:
            raise SystemExit(f"Duplicate completed run in worker manifests: {key}")
        seen.add(key)
        rows.append(row)
rows.sort(key=lambda row: (int(row["seed"]), condition_order[row["condition"]]))
with output_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"Canonical manifest contains {len(rows)} completed runs: {output_path}")
PY

{
    head -n 1 "$GPU0_FAILURES"
    tail -n +2 "$GPU0_FAILURES"
    tail -n +2 "$GPU1_FAILURES"
} > "$FAILURE_MANIFEST"

cp "$TRAIN_CSV" "${RESULT_ROOT}/train_data.csv"
cp "$VAL_CSV" "${RESULT_ROOT}/validation_data.csv"
cp "$TEST_CSV" "${RESULT_ROOT}/test_data.csv"
cp "$FASTA" "${RESULT_ROOT}/test_data_sequences.fasta"
cp "$SPLIT_MANIFEST" "${RESULT_ROOT}/split_manifest.csv"
cp "$GROUP_MANIFEST" "${RESULT_ROOT}/group_manifest.csv"
cp "$SPLIT_SUMMARY" "${RESULT_ROOT}/split_summary.json"
if [[ -f "$TEST_CSV_WITH_NAMES" ]]; then
    cp "$TEST_CSV_WITH_NAMES" "${RESULT_ROOT}/test_data_with_names.csv"
fi

echo ""
failure_count=$(( $(wc -l < "$FAILURE_MANIFEST") - 1 ))
completed_count=$(( $(wc -l < "$MANIFEST") - 1 ))
seed_count=$(wc -w <<< "$SEEDS")
expected_count=$(( seed_count * 7 ))
echo "Parallel queues finished: ${completed_count} completed run(s), ${failure_count} failure(s)."
echo "Expected runs for seeds [${SEEDS}]: ${expected_count}."
echo "Manifest: ${MANIFEST}"
echo "Failure manifest: ${FAILURE_MANIFEST}"
echo "Result root: ${RESULT_ROOT}"
if [[ $(( completed_count + failure_count )) -ne "$expected_count" ]]; then
    echo "ERROR: completed plus failed run count does not match the planned run count." >&2
    exit 1
fi
if [[ "$failure_count" -gt 0 ]]; then
    exit 1
fi
