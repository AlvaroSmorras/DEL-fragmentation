#!/bin/bash
# Stage 8: ECFP4 PCA/UMAP maps of DEL vs catalogue fragments and combinations.
# Needs stage 7's count tables (results_full/catalogue/) and the `fragviz` env
# (rdkit + umap-learn; kept apart from `chemprop` so numba cannot pin its numpy).
#
#   sbatch --account naiss2025-3-21-cpu slurm/embed_space.sbatch
#   N_BACKGROUND=100000 sbatch --account naiss2025-3-21-cpu slurm/embed_space.sbatch
#SBATCH --job-name=brics-embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=06:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-fragviz}"
source slurm/env.sh
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/08_embed_space.py \
    ${HITS_DIR:+--hits-dir "$HITS_DIR"} \
    --del-work "${DEL_WORK:-work_full}" \
    --catalogue-work "${WORK_DIR:-work_catalogue}" \
    --del-results "${DEL_RESULTS:-results_full}" \
    --out-dir "${OUT_DIR:-results_full/embedding}" \
    --n-background "${N_BACKGROUND:-50000}" \
    --workers "$SLURM_CPUS_PER_TASK"
