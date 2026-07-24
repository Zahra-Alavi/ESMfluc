#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${package_dir}/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-python3}"

result_root="results/publication_comparable_v2"
bands_csv="$result_root/analysis_seed_averaged_signed_bands/signed_bands.csv"
protein_summary_csv="$result_root/analysis_seed_averaged_signed_bands/signed_band_protein_summary.csv"
manifest_tsv="$result_root/all_split_signed_contributions_manifest.tsv"
residue_csv="$result_root/analysis_seed_averaged_band_biophysics/annotations/residue_biophysical_annotations.csv.gz"
mechanism_csv="$result_root/analysis_seed_averaged_band_phase3c/mechanism_by_band_seed_averaged.csv.gz"
contact_root="$result_root/analysis_seed_averaged_band_external_structure/contact_networks"
ecod_csv="data_splits/atlas_grouped_v1/ecod_v285_annotations.csv"
original_receiver_root="$result_root/analysis_seed_averaged_band_query_receivers"
upgrade_root="$result_root/analysis_seed_averaged_band_query_receivers_upgraded"

conditions=(
  esm2_frozen_bilstm_attn
  esm2_top4_bilstm_attn
  esm2_top28_bilstm_attn
  esm3_frozen_bilstm_attn
  esm3_top4_bilstm_attn
  esm3_top28_bilstm_attn
)

build_and_analyze() {
  local analysis_label="$1"
  shift
  local feature_root="$upgrade_root/feature_store_${analysis_label}"
  local receiver_root="$upgrade_root/${analysis_label}"
  local feature_audit="$upgrade_root/audit_${analysis_label}_features.json"
  local final_audit="$upgrade_root/audit_${analysis_label}_complete.json"

  "$python_bin" -m signed_band_analysis build-query-structure \
    --bands_csv "$bands_csv" \
    --contact_json \
      "$contact_root/train_contacts_ca8.json.gz" \
      "$contact_root/validation_contacts_ca8.json.gz" \
      "$contact_root/test_contacts_ca8.json.gz" \
    --ecod_csv "$ecod_csv" \
    --output_dir "$feature_root" \
    "$@"

  "$python_bin" -m signed_band_analysis audit-phase4-upgraded \
    --bands_csv "$bands_csv" \
    --feature_dir "$feature_root" \
    --output_json "$feature_audit"

  local receiver_dirs=()
  for condition in "${conditions[@]}"; do
    local output_dir="$receiver_root/$condition"
    receiver_dirs+=("$output_dir")
    "$python_bin" -m signed_band_analysis query-receivers \
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
      --random_seed 123
  done

  "$python_bin" -m signed_band_analysis audit-phase4-upgraded \
    --bands_csv "$bands_csv" \
    --feature_dir "$feature_root" \
    --receiver_output_dir "${receiver_dirs[@]}" \
    --output_json "$final_audit"
}

# Primary, prespecified crystallographic-water definition.
if [[ "${RUN_PHASE4_PRIMARY:-1}" == "1" ]]; then
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
if [[ "${RUN_PHASE4_WATER_SENSITIVITY:-0}" == "1" ]]; then
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
