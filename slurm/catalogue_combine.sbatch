#!/bin/bash
# Fold stage 1, then find the target combinations across the catalogue.
#
# There is no stage 3 and beyond here: an unlabelled catalogue has nothing to
# enrich against.  The output is which catalogue compounds carry the
# combinations a DEL run flagged.
#SBATCH --job-name=brics-cat-combine
#SBATCH --output=logs/cat_combine_%j.out
#SBATCH --error=logs/cat_combine_%j.err
set -euo pipefail

cd "${PROJECT_DIR:?set by submit_catalogue.sh}"
source slurm/env.sh

python scripts/01_fragment.py --work-dir "$WORK_DIR" --reduce-only

TARGET_FLAG=""
if [ -n "${TARGET_COMBOS:-}" ]; then
    [ -f "$TARGET_COMBOS" ] || { echo "no target list at $TARGET_COMBOS" >&2; exit 1; }
    TARGET_FLAG="--target-combos $TARGET_COMBOS"
fi

python scripts/02_combine.py \
    --work-dir "$WORK_DIR" \
    --sizes "$SIZES" \
    --workers "$SLURM_CPUS_PER_TASK" \
    $TARGET_FLAG

echo "catalogue fragments : $WORK_DIR/compound_fragments/"
echo "matching compounds  : $WORK_DIR/combinations/size${SIZES%%,*}/"
echo "per-combination hits: $WORK_DIR/combination_counts/size${SIZES%%,*}/"
