#!/usr/bin/env bash
# Fragment an unlabelled catalogue and find a DEL's combinations in it.
#
# RUN THIS DIRECTLY ON THE LOGIN NODE - it calls sbatch, it is not a job.
#
#   TARGET_COMBOS=targets.parquet INPUT_DIR=data/enamine_sharded ./slurm/submit_catalogue.sh
#   RESUME=1 ./slurm/submit_catalogue.sh
#
# Only stages 1 and 2 run: an unlabelled library has nothing to enrich against,
# so stages 3-6 do not apply and stage 3 refuses such a work directory outright.
#
# Build the target list first, from a LABELLED run:
#   python scripts/build_target_combos.py --work-dir work_full --out targets.parquet
#
# Sizing, all measured on this pipeline:
#   * A file is one pool task, never split, so total workers above the file
#     count buys nothing and wall time is
#       ceil(n_files / total_workers) x (rows_per_file x 3.9 ms)
#   * Peak memory per worker, measured. With the SMILES cache it is
#       157 MB + 231 MB per 100k rows in one file
#     and the catalogue job passes --no-smiles-cache, which drops it to
#       157 MB + 131 MB per 100k rows        (43% less - the cache never hits
#                                             on a catalogue, 0.00% measured)
#     So a 2.5M-row shard needs ~3.4 GB per worker, and MEM_PER_CPU=5G leaves
#     margin. A 10M-row shard still needs ~13 GB even without the cache, which
#     is why re-sharding to ~2.5M is worth the one extra pass.
#   * CPUS_PER_TASK x MEM_PER_CPU must fit the node.
set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "error: this script submits jobs, it is not one." >&2
    echo "  run it directly on the login node:  ./slurm/submit_catalogue.sh" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCOUNT="${ACCOUNT:-naiss2026-3-421-cpu}"
PARTITION="${PARTITION:-}"
INPUT_DIR="${INPUT_DIR:-data/catalogue_sharded}"
WORK_DIR="${WORK_DIR:-work_catalogue}"
TARGET_COMBOS="${TARGET_COMBOS:-targets.parquet}"
MIN_HAC="${MIN_HAC:-6}"
SIZES="${SIZES:-2}"
N_SHARDS="${N_SHARDS:-68}"
CPUS_PER_TASK="${CPUS_PER_TASK:-48}"
MEM_PER_CPU="${MEM_PER_CPU:-5G}"
ROWS_PER_FILE="${ROWS_PER_FILE:-2500000}"
FRAGMENT_TIME="${FRAGMENT_TIME:-24:00:00}"
COMBINE_CPUS="${COMBINE_CPUS:-32}"
COMBINE_MEM="${COMBINE_MEM:-120G}"
COMBINE_TIME="${COMBINE_TIME:-08:00:00}"

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
if [ -n "$TARGET_COMBOS" ] && [ ! -f "$TARGET_COMBOS" ]; then
    echo "error: no target list at $TARGET_COMBOS" >&2
    echo "  build one:  python scripts/build_target_combos.py --work-dir work_full --out $TARGET_COMBOS" >&2
    echo "  or set TARGET_COMBOS= (empty) to keep every combination - far larger output" >&2
    exit 1
fi

if [ -n "$(ls -A "$WORK_DIR/compound_fragments" 2>/dev/null)" ]; then
    if [ "${CLEAN:-0}" = "1" ]; then
        echo "CLEAN=1: removing previous stage 1 output in $WORK_DIR"
        rm -rf "$WORK_DIR/compound_fragments" "$WORK_DIR/fragment_counts" \
               "$WORK_DIR/file_stats" "$WORK_DIR/shards"
    elif [ "${RESUME:-0}" = "1" ]; then
        echo "RESUME=1: keeping finished files, fragmenting only what is missing"
    else
        echo "error: $WORK_DIR already holds stage 1 output." >&2
        echo "  RESUME=1 ./slurm/submit_catalogue.sh   continue an interrupted run" >&2
        echo "  CLEAN=1  ./slurm/submit_catalogue.sh   delete it and start over" >&2
        exit 1
    fi
fi
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

export PROJECT_DIR INPUT_DIR WORK_DIR TARGET_COMBOS MIN_HAC SIZES N_SHARDS \
       RESUME_FLAG CONDA_MODULE CONDA_ENV

SB_COMMON=(--account "$ACCOUNT" --export=ALL)
[ -n "$PARTITION" ] && SB_COMMON+=(--partition "$PARTITION")

fragment_id=$(sbatch --parsable "${SB_COMMON[@]}" \
    --array="0-$((N_SHARDS - 1))" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --time="$FRAGMENT_TIME" \
    slurm/catalogue_fragment.sbatch)

total_workers=$((N_SHARDS * CPUS_PER_TASK))
rounds=$(( (n_files + total_workers - 1) / total_workers ))
est_h=$(( rounds * ROWS_PER_FILE * 39 / 10000 / 3600 ))
echo "stage 1 array job : $fragment_id"
echo "    $N_SHARDS tasks x $CPUS_PER_TASK workers = $total_workers workers over $n_files files"
echo "    $rounds round(s) x ${ROWS_PER_FILE} rows -> roughly ${est_h} h wall"
if [ "$est_h" -ge 24 ]; then
    echo "    NOTE: that exceeds a 24 h wall. Raise N_SHARDS, or let it die and" >&2
    echo "          resubmit with RESUME=1 - finished files are never redone." >&2
fi

combine_id=$(sbatch --parsable "${SB_COMMON[@]}" \
    --dependency="afterok:${fragment_id}" \
    --cpus-per-task="$COMBINE_CPUS" \
    --mem="$COMBINE_MEM" \
    --time="$COMBINE_TIME" \
    slurm/catalogue_combine.sbatch)
echo "stage 2 job       : $combine_id  (starts when every shard succeeds)"
echo
echo "watch:   squeue -j ${fragment_id},${combine_id}"
echo "cancel:  scancel ${fragment_id} ${combine_id}"
