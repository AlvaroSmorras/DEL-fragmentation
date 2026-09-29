#!/usr/bin/env bash
# Submit the whole pipeline to Slurm: stage 1 as an array job, then everything
# else once every shard has succeeded.
#
#   ./slurm/submit.sh                       # uses the settings below
#   N_SHARDS=128 INPUT_DIR=data/BIG ./slurm/submit.sh
#   RESUME=1 ./slurm/submit.sh               # continue after a failure
#   CLEAN=1 ./slurm/submit.sh                # discard previous output first
#
# Sizing, from measurements on this pipeline:
#
#   * One input file is one task, so speedup caps at the *file count*, not the
#     core count. Keep N_SHARDS well under the number of input files, and run
#     scripts/00_shard_input.py first if the input is a few huge files.
#   * A worker needs ~157 MB of RDKit baseline plus ~236 MB per 100k compounds
#     *in one file*. At ~90k rows per file that is ~400 MB, hence --mem-per-cpu=1G
#     with headroom. Bigger input files need proportionally more.
#   * Throughput is ~255 compounds/s/worker, so
#       hours = compounds / (255 * N_SHARDS * CPUS_PER_TASK * 3600)
#     300M compounds over 64 tasks of 32 cores is roughly 10 minutes per task.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-data/raw_HGODEL_labeled/sharded}"
WORK_DIR="${WORK_DIR:-work_full}"
RESULTS_DIR="${RESULTS_DIR:-results_full}"
N_SHARDS="${N_SHARDS:-64}"
MIN_HAC="${MIN_HAC:-6}"
SIZES="${SIZES:-2}"
MIN_BINDER="${MIN_BINDER:-1}"
MIN_COUNT="${MIN_COUNT:-5}"
STRATIFY="${STRATIFY:-file}"
CLUSTER_MODE="${CLUSTER_MODE:-substituent}"
NO_LONG_TABLE="${NO_LONG_TABLE:-0}"

cd "$PROJECT_DIR"
mkdir -p logs

n_files=$(find "$INPUT_DIR" -name '*.parquet' 2>/dev/null | wc -l)
if [ "$n_files" -eq 0 ]; then
    echo "no parquet files under $INPUT_DIR" >&2
    exit 1
fi
if [ "$N_SHARDS" -gt "$n_files" ]; then
    echo "note: $N_SHARDS shards but only $n_files files; trimming to $n_files" >&2
    N_SHARDS="$n_files"
fi

# A shard never deletes output, since it cannot tell a sibling's work from a
# stale file, so stale output has to be dealt with before anything is submitted.
# Deleting it by default is too dangerous when WORK_DIR has a default value:
# say what to do instead, and make the caller choose.
if [ -n "$(ls -A "$WORK_DIR/compound_fragments" 2>/dev/null)" ]; then
    if [ "${CLEAN:-0}" = "1" ]; then
        echo "CLEAN=1: removing previous stage 1 output in $WORK_DIR"
        rm -rf "$WORK_DIR/compound_fragments" "$WORK_DIR/fragment_counts" \
               "$WORK_DIR/file_stats" "$WORK_DIR/shards"
    elif [ "${RESUME:-0}" = "1" ]; then
        echo "RESUME=1: keeping finished files, fragmenting only what is missing"
    else
        echo "error: $WORK_DIR already holds stage 1 output." >&2
        echo "  RESUME=1 ./slurm/submit.sh   continue an interrupted run" >&2
        echo "  CLEAN=1  ./slurm/submit.sh   delete it and start over" >&2
        echo "  WORK_DIR=... ./slurm/submit.sh   write somewhere else" >&2
        exit 1
    fi
fi
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

export PROJECT_DIR PYTHON INPUT_DIR WORK_DIR RESULTS_DIR N_SHARDS MIN_HAC \
       SIZES MIN_BINDER MIN_COUNT STRATIFY CLUSTER_MODE NO_LONG_TABLE RESUME_FLAG

fragment_id=$(sbatch --parsable --array="0-$((N_SHARDS - 1))" \
    --export=ALL slurm/fragment_array.sbatch)
echo "stage 1 array job : $fragment_id  ($N_SHARDS shards over $n_files files)"

score_id=$(sbatch --parsable --dependency="afterok:${fragment_id}" \
    --export=ALL slurm/score.sbatch)
echo "stages 2-6 job    : $score_id  (starts when every shard succeeds)"
echo
echo "watch:   squeue -j ${fragment_id},${score_id}"
echo "logs:    tail -f logs/fragment_${fragment_id}_0.out"
echo "cancel:  scancel ${fragment_id} ${score_id}"
