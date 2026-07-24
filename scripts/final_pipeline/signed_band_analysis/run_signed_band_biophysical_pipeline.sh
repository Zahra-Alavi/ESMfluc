#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${package_dir}/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-python3}"

bands_csv="results/publication_comparable_v2/analysis_seed_averaged_signed_bands/signed_bands.csv"
split_dir="data_splits/atlas_grouped_v1"
annotation_dir="results/publication_comparable_v2/analysis_seed_averaged_band_biophysics/annotations"
enrichment_dir="results/publication_comparable_v2/analysis_seed_averaged_band_biophysics/enrichment"
audit_json="results/publication_comparable_v2/analysis_seed_averaged_band_biophysics/pipeline_audit.json"
strain_args=()
if [[ -n "${STRAIN_ROOT:-}" ]]; then
  strain_args+=(--strain_root "$STRAIN_ROOT")
  enrichment_dir="results/publication_comparable_v2/analysis_seed_averaged_band_biophysics/enrichment_with_test_strain"
  audit_json="results/publication_comparable_v2/analysis_seed_averaged_band_biophysics/pipeline_audit_with_test_strain.json"
fi

"$python_bin" -m signed_band_analysis annotate-biophysics \
  --bands_csv "$bands_csv" \
  --split_dir "$split_dir" \
  --output_dir "$annotation_dir"

"$python_bin" -m signed_band_analysis biophysical-enrichment \
  --annotated_bands_csv "$annotation_dir/signed_bands_biophysical_annotations.csv.gz" \
  --residue_annotations_csv "$annotation_dir/residue_biophysical_annotations.csv.gz" \
  --protein_summary_csv "results/publication_comparable_v2/analysis_seed_averaged_signed_bands/signed_band_protein_summary.csv" \
  --output_dir "$enrichment_dir" \
  "${strain_args[@]}" \
  --n_block_shifts 1000 \
  --n_sign_flips 10000 \
  --random_seed 123

"$python_bin" -m signed_band_analysis audit-biophysics \
  --original_bands_csv "$bands_csv" \
  --annotation_dir "$annotation_dir" \
  --enrichment_dir "$enrichment_dir" \
  --output_json "$audit_json"
