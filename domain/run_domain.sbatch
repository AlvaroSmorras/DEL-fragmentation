#!/bin/bash
# Domain comparison of DEL and catalogue at compound, fragment and pair level.
# Separate from the scoring pipeline: reads its work dirs, writes only its own.
#
#   sbatch --account naiss2025-3-21-cpu domain/run_domain.sbatch
#   TSNE=1 sbatch --account naiss2025-3-21-cpu domain/run_domain.sbatch
#SBATCH --job-name=brics-domain
#SBATCH --output=logs/domain_%j.out
#SBATCH --error=logs/domain_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=12:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-fragviz}"
source slurm/env.sh
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python domain/compare_domains.py \
    --out-dir "${OUT_DIR:-results_domain}" \
    --del-results "${DEL_RESULTS:-results_binder_ratio}" \
    --n-background "${N_BACKGROUND:-500000}" \
    --max-binders "${MAX_BINDERS:-0}" \
    --hit-selection "${HIT_SELECTION:-leader}" --hit-threshold "${HIT_THRESHOLD:-0.7}" \
    --hit-selection-levels "${HIT_SELECTION_LEVELS:-compounds}" \
    --pca-space "${PCA_SPACE:-ecfp}" \
    --levels "${LEVELS:-compounds,fragments,combinations}" \
    ${TSNE:+--tsne} \
    --workers "$SLURM_CPUS_PER_TASK"
