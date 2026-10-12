#!/bin/bash
#SBATCH --job-name=brics-shared
#SBATCH --output=logs/shared_%j.out
#SBATCH --error=logs/shared_%j.err
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-fragviz}"
source slurm/env.sh
python domain/shared_compounds.py --workers "$SLURM_CPUS_PER_TASK" ${KEEP_PARTS:+--keep-parts}
