#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${package_dir}/.." && pwd)"
cd "$repository_root"
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-python3}"

: "${BANDS_CSV:?Set BANDS_CSV to the stable band catalog.}"
: "${PROTEIN_SUMMARY_CSV:?Set PROTEIN_SUMMARY_CSV to the matching band protein summary.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new local-property output directory.}"

bands_csv="$BANDS_CSV"
protein_summary_csv="$PROTEIN_SUMMARY_CSV"
split_dir="${SPLIT_DIR:-data/atlas_grouped_v1}"
annotation_dir="${OUTPUT_ROOT}/annotations"
enrichment_dir="${OUTPUT_ROOT}/enrichment"
audit_json="${OUTPUT_ROOT}/pipeline_audit.json"
analysis_splits=("${ANALYSIS_SPLIT:-test}")
strain_args=()
if [[ -n "${STRAIN_ROOT:-}" ]]; then
  strain_args+=(--strain_root "$STRAIN_ROOT")
  enrichment_dir="${OUTPUT_ROOT}/enrichment_with_test_strain"
  audit_json="${OUTPUT_ROOT}/pipeline_audit_with_test_strain.json"
fi

"$python_bin" -m esmfluc.analysis.bands annotate-biophysics \
  --bands_csv "$bands_csv" \
  --split_dir "$split_dir" \
  --output_dir "$annotation_dir"

"$python_bin" -m esmfluc.analysis.bands biophysical-enrichment \
  --annotated_bands_csv "$annotation_dir/signed_bands_biophysical_annotations.csv.gz" \
  --residue_annotations_csv "$annotation_dir/residue_biophysical_annotations.csv.gz" \
  --protein_summary_csv "$protein_summary_csv" \
  --output_dir "$enrichment_dir" \
  "${strain_args[@]}" \
  --splits "${analysis_splits[@]}" \
  --n_block_shifts 1000 \
  --n_sign_flips 10000 \
  --random_seed 123

"$python_bin" -m esmfluc.analysis.bands audit-biophysics \
  --original_bands_csv "$bands_csv" \
  --annotation_dir "$annotation_dir" \
  --enrichment_dir "$enrichment_dir" \
  --output_json "$audit_json" \
  --splits "${analysis_splits[@]}"
