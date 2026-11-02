#!/usr/bin/env bash
# Reproduces every result, table and figure of the paper from the public NHIS and NHANES files.
# Run from the repository root after placing the raw data in data/raw/ (see data/README.md).
# Stages (each can be run alone with --stage): prepare, tests, lca, models, measurement, postprocess, paper.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out data/proc
${PYTHON:-python3} code/run_experiments.py --stage all --rscript "${RSCRIPT:-Rscript}"
echo "Done. All outputs are in out/ (logs: out/run_*.log)."
