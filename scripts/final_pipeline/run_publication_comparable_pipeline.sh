#!/usr/bin/env bash
# Run the full comparable publication batch:
#   1. train/extract attention for 5 conditions x 3 seeds
#   2. analyze seed variance and cross-condition agreement

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULT_SET="${RESULT_SET:-publication_comparable_$(date +%Y%m%d_%H%M%S)}"
export RESULT_SET

echo "Starting comparable publication pipeline: ${RESULT_SET}"

bash run_publication_seed_experiments.sh

python analyze_publication_seed_variance.py \
    --result_root "results/${RESULT_SET}"

echo ""
echo "Done."
echo "Result root: scripts/final_pipeline/results/${RESULT_SET}"
echo "Analysis: scripts/final_pipeline/results/${RESULT_SET}/analysis"
