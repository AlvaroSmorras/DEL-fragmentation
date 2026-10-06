#!/bin/bash
# Stage 7: which DEL-enriched fragments/combinations exist in the catalogue,
# and which catalogue compounds carry them.
#
#   sbatch --account naiss2025-3-21-cpu slurm/catalogue_hits.sbatch
#   RANK=ratio DEL_RESULTS=results_binder_ratio OUT_DIR=results_binder_ratio/catalogue \
#       sbatch --account naiss2025-3-21-cpu slurm/catalogue_hits.sbatch
#SBATCH --job-name=brics-cat-hits
#SBATCH --output=logs/cat_hits_%j.out
#SBATCH --error=logs/cat_hits_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=06:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

python scripts/07_catalogue_hits.py \
    --del-results "${DEL_RESULTS:-results_full}" \
    --catalogue-work "${WORK_DIR:-work_catalogue}" \
    --catalogue-input "${INPUT_DIR:-data/Enamine_ll/sharded}" \
    --out-dir "${OUT_DIR:-results_full/catalogue}" \
    --rank "${RANK:-mh}" \
    --alpha "${ALPHA:-0.05}" --min-lo95 "${MIN_LO95:-1}" \
    --min-ratio "${MIN_RATIO:-0.01}" --min-binder "${MIN_BINDER:-5}" \
    --max-per-fragment "${MAX_PER_FRAGMENT:-1000}" \
    --workers "$SLURM_CPUS_PER_TASK"
