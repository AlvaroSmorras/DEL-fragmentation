#!/bin/bash
# Stage 1 over an unlabelled catalogue, as a Slurm array job.
#
# Identical to fragment_array.sbatch except that it passes --no-smiles-cache:
# a catalogue has no repeated SMILES, so memoising them only costs memory.
#SBATCH --job-name=brics-cat-fragment
#SBATCH --output=logs/cat_fragment_%A_%a.out
#SBATCH --error=logs/cat_fragment_%A_%a.err
set -euo pipefail

cd "${PROJECT_DIR:?set by submit_catalogue.sh}"
source slurm/env.sh

python scripts/01_fragment.py \
    --input-dir "$INPUT_DIR" \
    --work-dir "$WORK_DIR" \
    --min-hac "$MIN_HAC" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --no-smiles-cache \
    --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
    ${RESUME_FLAG:-}
