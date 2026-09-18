#!/usr/bin/env bash
# Run the whole pipeline end to end.
#
#   ./run_pipeline.sh                    # defaults
#   MIN_HAC=8 SIZES=2,3 ./run_pipeline.sh
#   CLUSTER_MODE=murcko ./run_pipeline.sh
#
# Stage 1 is the expensive one; re-running only stages 2-5 after changing SIZES
# or the scoring settings is enough, and much faster.
#
# Set PYTHON if the interpreter carrying RDKit is not the one on PATH, e.g.
#   PYTHON=~/anaconda3/bin/python ./run_pipeline.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-data/HGODEL}"
WORK_DIR="${WORK_DIR:-work}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MIN_HAC="${MIN_HAC:-6}"
SIZES="${SIZES:-2}"
WORKERS="${WORKERS:-$(($(nproc) - 1))}"
MIN_COUNT="${MIN_COUNT:-5}"
MIN_BINDER="${MIN_BINDER:-1}"
STRATIFY="${STRATIFY:-file}"
CLUSTER_MODE="${CLUSTER_MODE:-substituent}"
TOP="${TOP:-500}"

"$PYTHON" scripts/01_fragment.py --input-dir "$INPUT_DIR" --work-dir "$WORK_DIR" \
    --min-hac "$MIN_HAC" --workers "$WORKERS"
"$PYTHON" scripts/02_combine.py --work-dir "$WORK_DIR" --sizes "$SIZES" --workers "$WORKERS"
"$PYTHON" scripts/03_enrich.py --work-dir "$WORK_DIR" --results-dir "$RESULTS_DIR" \
    --min-count "$MIN_COUNT" --min-binder "$MIN_BINDER" --stratify "$STRATIFY" --top "$TOP"
for size in ${SIZES//,/ }; do
    "$PYTHON" scripts/04_inspect.py --input-dir "$INPUT_DIR" --work-dir "$WORK_DIR" \
        --results-dir "$RESULTS_DIR" --size "$size" --top "$TOP"
done
"$PYTHON" scripts/05_cluster.py --work-dir "$WORK_DIR" --results-dir "$RESULTS_DIR" \
    --mode "$CLUSTER_MODE" --stratify "$STRATIFY" --workers "$WORKERS" --top "$TOP"
