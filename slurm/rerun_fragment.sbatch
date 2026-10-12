#!/bin/bash
# Stage 1 as a Slurm array task, for the re-fragmentation at --min-hac 5 with
# the balanced cut selector.
#
# Submitted by slurm/rerun_fragmentation.sh, which sets --array, the resources
# and every exported variable below.  Do not sbatch this directly.
#
# One sbatch serves both libraries: they differ only in EXTRA_FLAGS, which
# carries --no-smiles-cache for the catalogue (no SMILES ever repeats there, so
# memoising them is pure memory cost - measured 0.00% hit rate).
#
# Tasks need no coordination: fragment ids are content hashes, so shards produce
# ids that already agree and output that merges by concatenation.
#SBATCH --output=logs/rerun_fragment_%A_%a.out
#SBATCH --error=logs/rerun_fragment_%A_%a.err
set -euo pipefail

cd "${PROJECT_DIR:?set by rerun_fragmentation.sh}"
source slurm/env.sh

# --min-hac and --selector are passed explicitly rather than left to the
# defaults in lib/fragmentation.py.  The settings land in the job log and in
# summary_fragment.json, so a work directory can always be traced back to the
# run that produced it even if those defaults move again.
python scripts/01_fragment.py \
    --input-dir "$INPUT_DIR" \
    --work-dir "$WORK_DIR" \
    --min-hac "$MIN_HAC" \
    --selector "$SELECTOR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
    ${EXTRA_FLAGS:-} \
    ${RESUME_FLAG:-}
