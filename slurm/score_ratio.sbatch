#!/bin/bash
# Fold the array job's output, then run the remaining stages on one node.
#
# Runs under --dependency=afterok on the array job, so it starts only if every
# shard succeeded.  Stages 2-6 do not shard, so they get one big node.
#SBATCH --job-name=brics-score
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err
set -euo pipefail

cd "${PROJECT_DIR:?set by submit.sh}"
source slurm/env.sh
W="$SLURM_CPUS_PER_TASK"

# Build the dictionary and the global summary from what the shards left behind.
python scripts/01_fragment.py --work-dir "$WORK_DIR" --reduce-only

# The per-compound table is one row per compound per combination - roughly 7
# billion rows for this library.  Dropping it costs stage 5 its exact recount
# (it falls back to summing members, which over-counted 1.75% of multi-member
# clusters on the reference data) and costs stage 6 entirely.
LONG_FLAG=""
CLUSTER_FLAG=""
if [ "${NO_LONG_TABLE:-0}" = "1" ]; then
    LONG_FLAG="--no-long-table"
    CLUSTER_FLAG="--approximate"
fi
NEW_RESULTS_DIR="results_by_ratio"

python scripts/02_combine.py --work-dir "$WORK_DIR" --sizes "$SIZES" --workers "$W" $LONG_FLAG
python scripts/03_enrich.py --work-dir "$WORK_DIR" --results-dir "$NEW_RESULTS_DIR" \
    --min-binder "$MIN_BINDER" --min-count "$MIN_COUNT" --stratify "$STRATIFY" --top 1500 --rank ratio

for size in ${SIZES//,/ }; do
    python scripts/04_inspect.py --input-dir "$INPUT_DIR" --work-dir "$WORK_DIR" \
        --results-dir "$NEW_RESULTS_DIR" --size "$size" --top 1500
done

python scripts/05_cluster.py --work-dir "$WORK_DIR" --results-dir "$NEW_RESULTS_DIR" \
    --mode "$CLUSTER_MODE" --stratify "$STRATIFY" --workers "$W" --top 1500 $CLUSTER_FLAG

if [ "${NO_LONG_TABLE:-0}" != "1" ]; then
    python scripts/06_validate_clusters.py --work-dir "$WORK_DIR" \
        --results-dir "$NEW_RESULTS_DIR" --workers "$W" || true
fi

echo "results in $RESULTS_DIR"
