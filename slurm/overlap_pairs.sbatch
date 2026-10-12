#!/bin/bash
# Unbiased catalogue pass for pair overlap: searches the catalogue for a uniform
# sample of ALL distinct DEL pairs, not just the ones a binder carries.
# Writes to work_catalogue_overlap/, which only symlinks work_catalogue's
# stage 1 output, so the pipeline's own stage 2 results are never touched.
#SBATCH --job-name=brics-overlap
#SBATCH --output=logs/overlap_%j.out
#SBATCH --error=logs/overlap_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=12:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

python scripts/02_combine.py \
    --work-dir work_catalogue_overlap \
    --sizes 2 \
    --target-combos "${TARGETS:-results_domain/del_pair_sample.parquet}" \
    --no-long-table \
    --workers "$SLURM_CPUS_PER_TASK"
