#!/bin/bash
# Stage 1 as a Slurm array job: each task fragments one shard of the input.
#
# Submitted by slurm/submit.sh, which sets --array, --account and the exported
# settings.  Do not sbatch this directly unless you set them yourself.
#
# Tasks need no coordination: fragment ids are content hashes, so shards produce
# ids that already agree and output that merges by concatenation.
#SBATCH --job-name=brics-fragment
#SBATCH --output=logs/fragment_%A_%a.out
#SBATCH --error=logs/fragment_%A_%a.err
set -euo pipefail

cd "${PROJECT_DIR:?set by submit.sh}"
source slurm/env.sh

python scripts/01_fragment.py \
    --input-dir "$INPUT_DIR" \
    --work-dir "$WORK_DIR" \
    --min-hac "$MIN_HAC" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
    ${RESUME_FLAG:-}
