#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pipeline_root="$(cd -- "$script_dir/.." && pwd)"
analysis_dir="$pipeline_root/results/weinreb2025_mutants_no_AV/mutation_network_analysis"

python3 "$script_dir/build_weinreb_delta_profiles.py" --output "$analysis_dir"
python3 "$script_dir/detect_weinreb_change_bands.py" --analysis "$analysis_dir"
python3 "$script_dir/decompose_weinreb_contributions.py" --analysis "$analysis_dir"
python3 "$script_dir/map_weinreb_receivers.py" --analysis "$analysis_dir"
python3 "$script_dir/analyze_weinreb_associations.py" --analysis "$analysis_dir"
python3 "$script_dir/replicate_weinreb_conditions.py" --analysis "$analysis_dir"
python3 "$script_dir/audit_weinreb_analysis.py" --analysis "$analysis_dir"

