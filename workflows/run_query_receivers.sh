#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${package_dir}/.." && pwd)"
cd "$repository_root"
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-python3}"
receivers_only="${RUN_RECEIVERS_ONLY:-0}"
condition_jobs="${RECEIVER_CONDITION_JOBS:-1}"

result_root="${RESULT_ROOT:-results}"
bands_root="$result_root/bands_seed_reproducibility"
bands_csv="${RECEIVER_BANDS_CSV:-$bands_root/reproducibility_interval_iou05_null_fixed/stable_signed_bands.csv}"
protein_summary_csv="${RECEIVER_PROTEIN_SUMMARY_CSV:-$bands_root/mean/signed_band_protein_summary.csv}"
manifest_tsv="${RECEIVER_MANIFEST_TSV:-$result_root/all_split_signed_contributions_manifest.tsv}"
residue_csv="${RECEIVER_RESIDUE_CSV:-$result_root/local_property_annotations/annotations/residue_biophysical_annotations.csv.gz}"
mechanism_csv="${RECEIVER_MECHANISM_CSV:-$result_root/evidence_attention_components/mechanism_by_band_seed_averaged.csv.gz}"
contact_root="${RECEIVER_CONTACT_ROOT:-$result_root/structure_contacts/contact_networks}"
ecod_csv="${RECEIVER_ECOD_CSV:-data/atlas_grouped_v1/ecod_v285_annotations.csv}"
original_receiver_root="${RECEIVER_BASE_ROOT:-$result_root/query_receivers_base}"
upgrade_root="${RECEIVER_OUTPUT_ROOT:-$result_root/query_receivers}"

conditions=(
  esm2_frozen_bilstm_attn
  esm2_top4_bilstm_attn
  esm2_top28_bilstm_attn
  esm3_frozen_bilstm_attn
  esm3_top4_bilstm_attn
  esm3_top28_bilstm_attn
)

if ! [[ "$condition_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "RECEIVER_CONDITION_JOBS must be a positive integer" >&2
  exit 2
fi

for required in \
  "$bands_csv" "$protein_summary_csv" "$manifest_tsv" \
  "$residue_csv" "$mechanism_csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required query-receiver input: $required" >&2
    exit 1
  fi
done

run_condition_jobs() {
  local callback="$1"
  local failure=0
  local condition
  local -a pids=()
  local -a names=()
  for condition in "${conditions[@]}"; do
    "$callback" "$condition" &
    pids+=("$!")
    names+=("$condition")
    if (( ${#pids[@]} >= condition_jobs )); then
      if ! wait "${pids[0]}"; then
        echo "Query-receiver condition failed: ${names[0]}" >&2
        failure=1
      fi
      pids=("${pids[@]:1}")
      names=("${names[@]:1}")
    fi
  done
  while (( ${#pids[@]} )); do
    if ! wait "${pids[0]}"; then
      echo "Query-receiver condition failed: ${names[0]}" >&2
      failure=1
    fi
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  done
  return "$failure"
}

run_nonstructural_condition() {
  local condition="$1"
  "$python_bin" -m esmfluc.analysis.bands query-receivers \
    --manifest_tsv "$manifest_tsv" \
    --bands_csv "$bands_csv" \
    --protein_summary_csv "$protein_summary_csv" \
    --residue_annotations_csv "$residue_csv" \
    --mechanism_csv "$mechanism_csv" \
    --output_dir "$original_receiver_root/$condition" \
    --conditions "$condition" \
    --receiver_quantile 0.90 \
    --low_receiver_quantile 0.50 \
    --long_range_min_separation 21 \
    --minimum_inference_proteins 10 \
    --max_model_rows_per_class_per_protein 50 \
    --progress_every 25 \
    --random_seed 123
}

if [[ "${RUN_RECEIVER_NONSTRUCTURAL:-1}" == "1" ]]; then
  run_condition_jobs run_nonstructural_condition
fi

build_and_analyze() {
  local analysis_label="$1"
  shift
  local feature_root="$upgrade_root/feature_store_${analysis_label}"
  local receiver_root="$upgrade_root/${analysis_label}"
  local feature_audit="$upgrade_root/audit_${analysis_label}_features.json"
  local final_audit="$upgrade_root/audit_${analysis_label}_complete.json"

  if [[ "$receivers_only" == "1" ]]; then
    if [[ ! -d "$feature_root/pair_features" ]]; then
      echo "Missing completed feature partitions: $feature_root/pair_features" >&2
      exit 1
    fi
    if [[ ! -f "$feature_root/parameters.json" ]]; then
      echo "Missing feature parameters: $feature_root/parameters.json" >&2
      exit 1
    fi
    if [[ ! -f "$feature_audit" ]]; then
      echo "Missing feature audit: $feature_audit" >&2
      exit 1
    fi
    if ! grep -q '"passed": true' "$feature_audit"; then
      echo "Feature audit did not pass: $feature_audit" >&2
      exit 1
    fi
    echo "Reusing audited feature store: $feature_root"
  else
    "$python_bin" -m esmfluc.analysis.bands build-query-structure \
      --bands_csv "$bands_csv" \
      --contact_json \
        "$contact_root/train_contacts_ca8.json.gz" \
        "$contact_root/validation_contacts_ca8.json.gz" \
        "$contact_root/test_contacts_ca8.json.gz" \
      --ecod_csv "$ecod_csv" \
      --output_dir "$feature_root" \
      "$@"

    "$python_bin" -m esmfluc.analysis.bands audit-query-receivers \
      --bands_csv "$bands_csv" \
      --feature_dir "$feature_root" \
      --output_json "$feature_audit"
  fi

  run_upgraded_condition() {
    local condition="$1"
    local output_dir="$receiver_root/$condition"
    "$python_bin" -m esmfluc.analysis.bands query-receivers \
      --manifest_tsv "$manifest_tsv" \
      --bands_csv "$bands_csv" \
      --protein_summary_csv "$protein_summary_csv" \
      --residue_annotations_csv "$residue_csv" \
      --mechanism_csv "$mechanism_csv" \
      --pairwise_structure_dir "$feature_root" \
      --receiver_cache_source_dir "$original_receiver_root/$condition" \
      --output_dir "$output_dir" \
      --conditions "$condition" \
      --aggregate_only \
      --receiver_quantile 0.90 \
      --low_receiver_quantile 0.50 \
      --long_range_min_separation 21 \
      --minimum_inference_proteins 10 \
      --max_model_rows_per_class_per_protein 50 \
      --progress_every 25 \
      --random_seed 123
  }
  run_condition_jobs run_upgraded_condition

  local receiver_dirs=()
  local condition
  for condition in "${conditions[@]}"; do
    receiver_dirs+=("$receiver_root/$condition")
  done
  "$python_bin" -m esmfluc.analysis.bands audit-query-receivers \
    --bands_csv "$bands_csv" \
    --feature_dir "$feature_root" \
    --receiver_output_dir "${receiver_dirs[@]}" \
    --output_json "$final_audit"
}

# Primary, prespecified crystallographic-water definition.
if [[ "${RUN_QUERY_RECEIVERS:-1}" == "1" && "${RUN_RECEIVER_PRIMARY:-1}" == "1" ]]; then
  build_and_analyze primary \
    --protein_water_cutoff 3.4 \
    --water_water_cutoff 3.2 \
    --polar_contact_cutoff 3.5 \
    --minimum_water_occupancy 0.50 \
    --maximum_water_bfactor_robust_z 3.0 \
    --maximum_water_resolution 2.5 \
    --water_chain_policy target_only_unambiguous
fi

# Optional sensitivity suite. It is deliberately opt-in because each setting
# creates a complete held-out analysis rather than changing cutoffs after
# inspecting the primary result.
if [[ "${RUN_QUERY_RECEIVERS:-1}" == "1" && "${RUN_RECEIVER_WATER_SENSITIVITY:-0}" == "1" ]]; then
  build_and_analyze strict_3p2 \
    --protein_water_cutoff 3.2 \
    --water_water_cutoff 3.2 \
    --polar_contact_cutoff 3.2 \
    --minimum_water_occupancy 1.00 \
    --maximum_water_bfactor_robust_z 2.0 \
    --maximum_water_resolution 2.2 \
    --water_chain_policy target_only_unambiguous

  build_and_analyze relaxed_3p5 \
    --protein_water_cutoff 3.5 \
    --water_water_cutoff 3.5 \
    --polar_contact_cutoff 3.5 \
    --minimum_water_occupancy 0.00 \
    --maximum_water_bfactor_robust_z 4.0 \
    --maximum_water_resolution 3.0 \
    --water_chain_policy target_only_unambiguous
fi
