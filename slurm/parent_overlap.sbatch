#!/bin/bash
#SBATCH --job-name=brics-parents
#SBATCH --output=logs/parents_%j.out
#SBATCH --error=logs/parents_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-fragviz}"
source slurm/env.sh
python domain/parent_overlap.py \
    --n-fragments "${N_FRAGMENTS:-400}" --n-pairs "${N_PAIRS:-400}" \
    --per-target "${PER_TARGET:-150}" --workers "$SLURM_CPUS_PER_TASK"
