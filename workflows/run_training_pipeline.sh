#!/usr/bin/env bash
# Run the full training batch:
#   1. train/extract attention for all conditions x seeds
#   2. analyze seed variance and cross-condition agreement

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPOSITORY_ROOT"
export PYTHONPATH="${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RESULT_SET="${RESULT_SET:-esmfluc_$(date +%Y%m%d_%H%M%S)}"
export RESULT_SET

echo "Starting training pipeline: ${RESULT_SET}"

bash workflows/run_seed_experiments.sh

python -m esmfluc.evaluation.analyze_seed_variance \
    --result_root "results/${RESULT_SET}"

echo ""
echo "Done."
echo "Result root: results/${RESULT_SET}"
echo "Analysis: results/${RESULT_SET}/analysis"
