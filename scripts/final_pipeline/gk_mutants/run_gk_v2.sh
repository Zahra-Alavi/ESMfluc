#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/.." && pwd)"
cd "$root"

base="results/weinreb2025_mutants_no_AV"
out="$base/mutation_network_analysis_v2"
phase1="$out/phase1_wt_star"
uniform="$out/uniform_control"
fasta="/home/zahralab/Desktop/ESMfluc/data/weinreb2025_mutants_no_AV.fasta"
esm_python="${ESM_PYTHON:-/home/zahralab/miniconda/envs/esm_env/bin/python3.11}"

"$esm_python" extract_external_fasta_contributions.py \
  --fasta_file gk_mutants/inputs/A176G.fasta \
  --source_manifest results/publication_comparable_v2/manifest.tsv \
  --output_root "$base/exact_contributions_a176g_v2" \
  --seeds 1 2 3 \
  --conditions \
    esm2_frozen_bilstm_attn esm2_top4_bilstm_attn esm2_top28_bilstm_attn \
    esm3_frozen_bilstm_attn esm3_top4_bilstm_attn esm3_top28_bilstm_attn \
  --gpus 0 1 --workers 2

python3 gk_mutants/prepare_gk_v2_wt_profiles.py \
  --manifest "$base/exact_contributions_v2/manifest.tsv" \
  --output-dir "$phase1"

python3 -m signed_band_analysis add-seed-average \
  --manifest_tsv "$phase1/profiles_manifest.tsv" \
  --averaged_manifest_tsv "$phase1/averaged_profiles_manifest.tsv" \
  --splits gk_wt_star --rewrite_workers 3 --pigz_threads 2 --compression_level 6

python3 -m signed_band_analysis extract-bands \
  --manifest_tsv "$phase1/profiles_manifest.tsv" \
  --output_dir "$phase1/per_seed" --splits gk_wt_star \
  --apex_method raw_mad_amplitude --amplitude_mad 2.0 \
  --support_method half_intensity_merge \
  --terminal_exclusion 0 --terminal_exclusion_fraction 0.0

python3 -m signed_band_analysis extract-bands \
  --manifest_tsv "$phase1/averaged_profiles_manifest.tsv" \
  --output_dir "$phase1/mean" --splits gk_wt_star \
  --influence_field seed_averaged_signed_column_influence \
  --apex_method raw_mad_amplitude --amplitude_mad 2.0 \
  --support_method half_intensity_merge \
  --terminal_exclusion 0 --terminal_exclusion_fraction 0.0

python3 -m signed_band_analysis seed-reproducibility \
  --bands_csv "$phase1/per_seed/signed_bands.csv" \
  --protein_summary_csv "$phase1/per_seed/signed_band_protein_summary.csv" \
  --mean_bands_csv "$phase1/mean/signed_bands.csv" \
  --output_dir "$phase1/reproducibility_interval_iou05" \
  --splits gk_wt_star --matching_method interval_iou \
  --min_interval_iou 0.5 --min_seeds 2 --n_block_shifts 1 --random_seed 123

python3 -m signed_band_analysis build-uniform-control \
  --manifest_tsv "$phase1/profiles_manifest.tsv" \
  --output_dir "$uniform" --splits gk_wt_star

python3 -m signed_band_analysis extract-bands \
  --manifest_tsv "$uniform/profiles/profile_manifest.tsv" \
  --output_dir "$uniform/bands/per_seed_uniform" \
  --influence_field uniform_signed_influence --splits gk_wt_star \
  --apex_method raw_mad_amplitude --amplitude_mad 2.0 \
  --support_method half_intensity_merge

python3 -m signed_band_analysis extract-bands \
  --manifest_tsv "$uniform/seed_averaged_profiles/seed_averaged_profile_manifest.tsv" \
  --output_dir "$uniform/bands/mean_uniform" \
  --influence_field seed_averaged_uniform_signed_influence --splits gk_wt_star \
  --apex_method raw_mad_amplitude --amplitude_mad 2.0 \
  --support_method half_intensity_merge

python3 -m signed_band_analysis seed-reproducibility \
  --bands_csv "$uniform/bands/per_seed_uniform/signed_bands.csv" \
  --protein_summary_csv "$uniform/bands/per_seed_uniform/signed_band_protein_summary.csv" \
  --mean_bands_csv "$uniform/bands/mean_uniform/signed_bands.csv" \
  --output_dir "$uniform/reproducibility_interval_iou05" \
  --splits gk_wt_star --matching_method interval_iou \
  --min_interval_iou 0.5 --min_seeds 2 --n_block_shifts 1 --random_seed 123

python3 gk_mutants/analyze_gk_v2.py \
  --base-manifest "$base/exact_contributions_v2/manifest.tsv" \
  --a176g-manifest "$base/exact_contributions_a176g_v2/manifest.tsv" \
  --fasta "$fasta" \
  --position-map "$base/position_map_1ZNX_WT.csv" \
  --structure "$base/pdb_cache/1ZNX_WT.pdb" \
  --stable-bands "$phase1/reproducibility_interval_iou05/stable_signed_bands.csv" \
  --mean-bands "$phase1/mean/signed_bands.csv" \
  --uniform-stable-bands "$uniform/reproducibility_interval_iou05/stable_signed_bands.csv" \
  --uniform-mean-bands "$uniform/bands/mean_uniform/signed_bands.csv" \
  --uniform-profile-similarity "$uniform/profile_comparison/per_protein_seed_averaged_profile_similarity.csv.gz" \
  --phenotypes "$base/mutation_network_analysis/phenotypes/paper_annotations_joined.csv" \
  --legacy-hotspots "$base/mutation_network_analysis/hotspots/common_hotspots.csv" \
  --output-dir "$out" --permutations 10000 --random-seed 123

python3 gk_mutants/audit_gk_v2.py \
  --analysis "$out" \
  --a176g-summary "$base/exact_contributions_a176g_v2/extraction_summary.json" \
  --fasta "$fasta" --position-map "$base/position_map_1ZNX_WT.csv"
