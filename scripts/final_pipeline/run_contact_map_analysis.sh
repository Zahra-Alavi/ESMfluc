#!/usr/bin/env bash
# =============================================================================
# Build C-alpha contact maps and analyze attention/contact agreement.
#
# Main publication set:
#   1. Build results/<RESULT_SET>/contact_maps_ca8.json from test_data_with_names.csv
#   2. Run analyze_attention_contacts.py over the result manifest.
#
# Optional:
#   BUILD_PER_TEMPERATURE=1 also builds contact maps for
#   ../../data/mdcath/per_temperature/test_*.csv. Those maps are cached under
#   results/<RESULT_SET>/contact_maps/per_temperature/.
#   EXTRACT_BACKBONE=1 extracts ESM2 transformer backbone attention for each
#   non-ESM3 manifest run before analysis. ESM3 is intentionally skipped because
#   the installed ESM3 API does not expose transformer attentions cleanly.
#
# Notes:
#   - Contact building downloads PDB files from RCSB unless already cached.
#   - CATH-style IDs like 1a39A00 are parsed as PDB 1a39, chain A. If the full
#     chain does not match the test sequence, the downstream analysis skips it
#     unless --allow_length_mismatch is passed to analyze_attention_contacts.py.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
RESULT_SET="${RESULT_SET:-publication_comparable_v1}"
RESULT_ROOT="results/${RESULT_SET}"

BUILD_CONTACTS="${BUILD_CONTACTS:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
BUILD_PER_TEMPERATURE="${BUILD_PER_TEMPERATURE:-1}"
EXTRACT_BACKBONE="${EXTRACT_BACKBONE:-0}"
BACKBONE_OUTPUT_NAME="${BACKBONE_OUTPUT_NAME:-backbone_attention.json}"
BACKBONE_ONLY_ESM2="${BACKBONE_ONLY_ESM2:-1}"
BACKBONE_ARCHITECTURES="${BACKBONE_ARCHITECTURES:-bilstm_attention esm_linear}"

CONTACT_CUTOFF="${CONTACT_CUTOFF:-8.0}"
CONTACT_INPUT_CSV="${CONTACT_INPUT_CSV:-${RESULT_ROOT}/test_data_with_names.csv}"
CONTACT_JSON="${CONTACT_JSON:-${RESULT_ROOT}/contact_maps_ca8.json}"
PDB_DIR="${PDB_DIR:-${RESULT_ROOT}/pdb_cache}"

MIN_SEQ_SEP="${MIN_SEQ_SEP:-6}"
N_PERMUTATIONS="${N_PERMUTATIONS:-200}"
ATTENTION_SOURCES="${ATTENTION_SOURCES:-bilstm}"

mkdir -p "${RESULT_ROOT}/logs"
exec > >(tee -a "${RESULT_ROOT}/logs/contact_map_analysis.log") 2>&1

enabled() {
    case "$1" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

contains_word() {
    local needle="$1"
    local word
    for word in $BACKBONE_ARCHITECTURES; do
        [[ "$word" == "$needle" ]] && return 0
    done
    return 1
}

build_contacts_for_csv() {
    local input_csv="$1"
    local output_json="$2"
    local pdb_dir="$3"

    if [[ ! -f "$input_csv" ]]; then
        echo "[skip] missing CSV: $input_csv"
        return 0
    fi
    if [[ -f "$output_json" && "${REBUILD_CONTACTS:-0}" != "1" ]]; then
        echo "[skip] contact JSON exists: $output_json"
        return 0
    fi

    mkdir -p "$(dirname "$output_json")" "$pdb_dir"
    "$PYTHON" Attention/build_contact_maps_from_pdb.py \
        --input_csv "$input_csv" \
        --output_json "$output_json" \
        --pdb_dir "$pdb_dir" \
        --cutoff "$CONTACT_CUTOFF"
}

extract_backbone_attention_from_manifest() {
    local manifest="${RESULT_ROOT}/manifest.tsv"
    if [[ ! -f "$manifest" ]]; then
        echo "[skip] missing manifest: $manifest" >&2
        return 0
    fi

    echo "" >&2
    echo "Extracting ESM2 backbone attention from manifest" >&2
    echo "  manifest: $manifest" >&2
    echo "  output filename: $BACKBONE_OUTPUT_NAME" >&2
    echo "  architectures: $BACKBONE_ARCHITECTURES" >&2

    "$PYTHON" - "$manifest" "$RESULT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
result_root = Path(sys.argv[2])
for row in csv.DictReader(manifest.open(), delimiter="\t"):
    if row.get("is_esm3", "").lower() == "true":
        continue
    print("\t".join([
        row["condition"],
        row["seed"],
        row["architecture"],
        row["esm_model"],
        row["run_dir"],
        row["checkpoint"],
    ]))
PY
}

if [[ "$EXTRACT_BACKBONE" == "1" ]]; then
    while IFS=$'\t' read -r condition seed architecture esm_model run_dir checkpoint; do
        if [[ -z "${condition:-}" ]]; then
            continue
        fi
        if enabled "$BACKBONE_ONLY_ESM2" && [[ "$esm_model" != esm2_* ]]; then
            echo "[skip] ${condition} seed=${seed}: non-ESM2 model ${esm_model}"
            continue
        fi
        if ! contains_word "$architecture"; then
            echo "[skip] ${condition} seed=${seed}: architecture ${architecture}"
            continue
        fi
        output_json="${run_dir}/${BACKBONE_OUTPUT_NAME}"
        if [[ -f "$output_json" && "${REBUILD_BACKBONE:-0}" != "1" ]]; then
            echo "[skip] backbone attention exists: $output_json"
            continue
        fi
        if [[ ! -f "$checkpoint" ]]; then
            echo "[skip] missing checkpoint for ${condition} seed=${seed}: $checkpoint"
            continue
        fi
        echo "[extract] ${condition} seed=${seed} -> ${output_json}"
        "$PYTHON" Attention/extract_backbone_attn.py \
            --checkpoint "$checkpoint" \
            --fasta_file "${FASTA:-${RESULT_ROOT}/test_data_sequences.fasta}" \
            --esm_model "$esm_model" \
            --output "$output_json"
    done < <(extract_backbone_attention_from_manifest)
fi

if [[ "$BUILD_CONTACTS" == "1" ]]; then
    echo "Building main contact map JSON"
    echo "  input:  $CONTACT_INPUT_CSV"
    echo "  output: $CONTACT_JSON"
    build_contacts_for_csv "$CONTACT_INPUT_CSV" "$CONTACT_JSON" "$PDB_DIR"

    if [[ "$BUILD_PER_TEMPERATURE" == "1" ]]; then
        echo ""
        echo "Building per-temperature contact map JSONs"
        shopt -s nullglob
        for csv in ../../data/mdcath/per_temperature/test_*.csv; do
            stem="$(basename "$csv" .csv)"
            build_contacts_for_csv \
                "$csv" \
                "${RESULT_ROOT}/contact_maps/per_temperature/${stem}_contacts_ca${CONTACT_CUTOFF}.json" \
                "${RESULT_ROOT}/pdb_cache"
        done
        shopt -u nullglob
    fi
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
    echo ""
    echo "Running attention/contact analysis"
    # shellcheck disable=SC2206
    sources=( $ATTENTION_SOURCES )
    "$PYTHON" analyze_attention_contacts.py \
        --result_root "$RESULT_ROOT" \
        --contact_json "$CONTACT_JSON" \
        --attention_sources "${sources[@]}" \
        --min_seq_sep "$MIN_SEQ_SEP" \
        --n_permutations "$N_PERMUTATIONS" \
        --require_exact_sequence_match
fi

echo ""
echo "Done."
echo "Contact JSON: ${CONTACT_JSON}"
echo "Analysis: ${RESULT_ROOT}/analysis_contacts"
