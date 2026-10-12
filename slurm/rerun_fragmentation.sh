#!/usr/bin/env bash
# Re-fragment the DEL and/or the catalogue at --min-hac 5 with the balanced cut
# selector, from scratch, into fresh work directories.
#
# RUN THIS DIRECTLY ON THE LOGIN NODE - it calls sbatch, it is not a job.
# It PRINTS A PLAN AND SUBMITS NOTHING unless you pass SUBMIT=1.  That is
# deliberate: the catalogue run writes the better part of a terabyte onto a
# filesystem that is already near full, so the plan is worth reading first.
#
#   ./slurm/rerun_fragmentation.sh                    # plan only, both libraries
#   LIBRARY=del ./slurm/rerun_fragmentation.sh        # plan, DEL only
#   SUBMIT=1 LIBRARY=del ./slurm/rerun_fragmentation.sh
#   SUBMIT=1 ./slurm/rerun_fragmentation.sh           # both
#   RESUME=1 SUBMIT=1 ./slurm/rerun_fragmentation.sh  # continue after a failure
#
# Why re-fragment at all
# ----------------------
# A five-membered ring is exactly five heavy atoms, so at the old --min-hac 6 no
# thiophene, furan, thiazole, pyrazole or cyclopentane joined by a single BRICS
# bond could ever be detached; and the old cut selector could only merge, so
# where cut sites were dense it collapsed molecules back to one piece.  Together
# those left 9.3M Enamine compounds uncut - 40% of that library's fragment
# dictionary.  See fragmentation_test/README.md.
#
# This writes to NEW directories by default.  The existing work_full and
# work_catalogue are left exactly as they are: they are still the basis of every
# result produced so far, and fragment ids from the two settings are not
# comparable, so the old and new output must never be mixed.  Stage 7 refuses
# such a mix unless --allow-settings-mismatch is passed.
set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "error: this script submits jobs, it is not one." >&2
    echo "  run it directly on the login node:  ./slurm/rerun_fragmentation.sh" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCOUNT="${ACCOUNT:-naiss2025-3-21-cpu}"
PARTITION="${PARTITION:-}"
LIBRARY="${LIBRARY:-both}"
MIN_HAC="${MIN_HAC:-5}"
SELECTOR="${SELECTOR:-balanced}"
SUBMIT="${SUBMIT:-0}"

# Sharding and resources are inherited unchanged from the runs that produced the
# current work directories - slurm/submit.sh for the DEL, slurm/submit_catalogue.sh
# for the catalogue - so this re-run is the same job shape as the last one, with
# only --min-hac and --selector different.  Empty means "do not pass the flag",
# which is how submit.sh leaves the DEL array on the partition defaults.
DEL_INPUT="${DEL_INPUT:-data/raw_HGODEL_labeled/sharded}"
DEL_WORK="${DEL_WORK:-work_full_hac5}"
DEL_SHARDS="${DEL_SHARDS:-64}"          # submit.sh
DEL_CPUS="${DEL_CPUS:-}"                # submit.sh passes none
DEL_MEM_PER_CPU="${DEL_MEM_PER_CPU:-}"  # submit.sh passes none
DEL_TIME="${DEL_TIME:-}"                # submit.sh passes none

CAT_INPUT="${CAT_INPUT:-data/Enamine_ll/sharded}"
CAT_WORK="${CAT_WORK:-work_catalogue_hac5}"
CAT_SHARDS="${CAT_SHARDS:-68}"              # submit_catalogue.sh
CAT_CPUS="${CAT_CPUS:-48}"                  # submit_catalogue.sh
CAT_MEM_PER_CPU="${CAT_MEM_PER_CPU:-5G}"    # submit_catalogue.sh
CAT_TIME="${CAT_TIME:-24:00:00}"            # submit_catalogue.sh

REDUCE_CPUS="${REDUCE_CPUS:-32}"            # as catalogue_combine.sbatch is submitted
REDUCE_MEM="${REDUCE_MEM:-120G}"
REDUCE_TIME="${REDUCE_TIME:-08:00:00}"

# Stage 1 output measured on the existing --min-hac 6 work directories, used
# only for the disk preflight.
DEL_MEASURED_GB="${DEL_MEASURED_GB:-18}"
CAT_MEASURED_GB="${CAT_MEASURED_GB:-792}"

# Lowering the floor from 6 to 5 takes the catalogue from 3.030 to 3.428
# fragments per compound (measured on 20,000 compounds), and stage 1 output is
# dominated by per-compound fragment rows, so it grows by about the same factor.
GROWTH_PERCENT="${GROWTH_PERCENT:-13}"

cd "$PROJECT_DIR"
mkdir -p logs

case "$LIBRARY" in
    del)       LIBRARIES=(del) ;;
    catalogue) LIBRARIES=(catalogue) ;;
    both)      LIBRARIES=(del catalogue) ;;
    *) echo "error: LIBRARY must be del, catalogue or both (got '$LIBRARY')" >&2; exit 1 ;;
esac

# ---- preflight -----------------------------------------------------------
total_needed_gb=0
for lib in "${LIBRARIES[@]}"; do
    if [ "$lib" = "del" ]; then
        input="$DEL_INPUT"; work="$DEL_WORK"; shards="$DEL_SHARDS"; measured=$DEL_MEASURED_GB
    else
        input="$CAT_INPUT"; work="$CAT_WORK"; shards="$CAT_SHARDS"; measured=$CAT_MEASURED_GB
    fi

    n_files=$(find "$input" -name '*.parquet' 2>/dev/null | wc -l)
    if [ "$n_files" -eq 0 ]; then
        echo "error: no parquet files under $input" >&2
        exit 1
    fi
    if [ "$shards" -gt "$n_files" ]; then
        echo "note: $shards shards but only $n_files files in $input; trimming" >&2
        shards="$n_files"
    fi

    # A shard never deletes output - it cannot tell a sibling's work from a
    # stale file - so a dirty work directory has to be resolved before anything
    # is submitted.
    if [ -n "$(ls -A "$work/compound_fragments" 2>/dev/null)" ]; then
        if [ "${CLEAN:-0}" = "1" ]; then
            echo "CLEAN=1: $work/ will have its stage 1 output removed before submitting"
        elif [ "${RESUME:-0}" = "1" ]; then
            echo "RESUME=1: $work/ keeps finished files, only what is missing is fragmented"
        else
            echo "error: $work already holds stage 1 output." >&2
            echo "  RESUME=1 ...   continue an interrupted run" >&2
            echo "  CLEAN=1  ...   delete it and start over" >&2
            echo "  DEL_WORK=/CAT_WORK=...   write somewhere else" >&2
            exit 1
        fi
    fi

    needed=$(( measured * (100 + GROWTH_PERCENT) / 100 ))
    total_needed_gb=$(( total_needed_gb + needed ))
    eval "${lib}_files=$n_files; ${lib}_shards=$shards; ${lib}_needed=$needed"
done

avail_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo
echo "disk:  about ${total_needed_gb} GB needed, ${avail_gb} GB free on $(df --output=target . | tail -1)"
if [ "$avail_gb" -lt "$total_needed_gb" ] && [ "${FORCE:-0}" = "1" ]; then
    echo "WARNING: not enough free space (${total_needed_gb} GB needed), continuing on FORCE=1." >&2
elif [ "$avail_gb" -lt "$total_needed_gb" ]; then
    echo "error: not enough free space for this run." >&2
    echo "  the old work directories hold ~810 GB; CLEAN them, or re-run one library" >&2
    echo "  at a time with LIBRARY=del / LIBRARY=catalogue, or set FORCE=1 to override" >&2
    exit 1
elif [ $(( avail_gb - total_needed_gb )) -lt 200 ]; then
    echo "WARNING: that leaves under 200 GB of headroom on a shared filesystem." >&2
fi

# ---- plan ----------------------------------------------------------------
echo
echo "plan:  --min-hac $MIN_HAC  --selector $SELECTOR"
for lib in "${LIBRARIES[@]}"; do
    eval "files=\$${lib}_files; shards=\$${lib}_shards; needed=\$${lib}_needed"
    if [ "$lib" = "del" ]; then
        work="$DEL_WORK"; cpus="$DEL_CPUS"; mem="$DEL_MEM_PER_CPU"; wall="$DEL_TIME"
    else
        work="$CAT_WORK"; cpus="$CAT_CPUS"; mem="$CAT_MEM_PER_CPU"; wall="$CAT_TIME"
    fi
    resources=""
    [ -n "$cpus" ] && resources="$resources, $cpus cpus"
    [ -n "$mem" ]  && resources="$resources, $mem/cpu"
    [ -n "$wall" ] && resources="$resources, wall $wall"
    [ -z "$resources" ] && resources=", partition defaults (as slurm/submit.sh)"
    printf '  %-10s %s -> %s\n' "$lib" "$files files" "$work"
    printf '  %-10s %s tasks%s, about %s GB out\n' "" "$shards" "$resources" "$needed"
done

if [ "$SUBMIT" != "1" ]; then
    echo
    echo "nothing submitted. Re-run with SUBMIT=1 to queue it:"
    echo "  SUBMIT=1 LIBRARY=$LIBRARY ./slurm/rerun_fragmentation.sh"
    exit 0
fi

# ---- submit --------------------------------------------------------------
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

SB_COMMON=(--account "$ACCOUNT" --export=ALL)
[ -n "$PARTITION" ] && SB_COMMON+=(--partition "$PARTITION")

submitted=()
for lib in "${LIBRARIES[@]}"; do
    eval "shards=\$${lib}_shards"
    if [ "$lib" = "del" ]; then
        INPUT_DIR="$DEL_INPUT"; WORK_DIR="$DEL_WORK"; cpus="$DEL_CPUS"
        mem="$DEL_MEM_PER_CPU"; wall="$DEL_TIME"; EXTRA_FLAGS=""
        name="brics-rerun-del"
    else
        INPUT_DIR="$CAT_INPUT"; WORK_DIR="$CAT_WORK"; cpus="$CAT_CPUS"
        mem="$CAT_MEM_PER_CPU"; wall="$CAT_TIME"; EXTRA_FLAGS="--no-smiles-cache"
        name="brics-rerun-cat"
    fi

    if [ "${CLEAN:-0}" = "1" ]; then
        rm -rf "$WORK_DIR/compound_fragments" "$WORK_DIR/fragment_counts" \
               "$WORK_DIR/file_stats" "$WORK_DIR/shards"
    fi

    N_SHARDS="$shards"
    export PROJECT_DIR INPUT_DIR WORK_DIR MIN_HAC SELECTOR N_SHARDS EXTRA_FLAGS \
           RESUME_FLAG CONDA_MODULE CONDA_ENV

    SB_ARRAY=(--job-name="$name" --array="0-$((shards - 1))")
    [ -n "$cpus" ] && SB_ARRAY+=(--cpus-per-task="$cpus")
    [ -n "$mem" ]  && SB_ARRAY+=(--mem-per-cpu="$mem")
    [ -n "$wall" ] && SB_ARRAY+=(--time="$wall")

    array_id=$(sbatch --parsable "${SB_COMMON[@]}" "${SB_ARRAY[@]}" \
        slurm/rerun_fragment.sbatch)

    reduce_id=$(sbatch --parsable "${SB_COMMON[@]}" \
        --dependency="afterok:${array_id}" \
        --cpus-per-task="$REDUCE_CPUS" \
        --mem="$REDUCE_MEM" \
        --time="$REDUCE_TIME" \
        slurm/rerun_reduce.sbatch)

    echo "$lib: shards $array_id -> reduce $reduce_id  ($WORK_DIR)"
    submitted+=("$array_id" "$reduce_id")
done

joined=$(IFS=,; echo "${submitted[*]}")
cat <<EOF

watch:   squeue -j $joined
cancel:  scancel ${submitted[*]}

Stage 1 only. Fragment ids change, so nothing downstream carries over - when the
reduce jobs finish, re-run the rest in this order:

  1. DEL stage 2 and onwards
       WORK_DIR=$DEL_WORK RESULTS_DIR=results_full_hac5 sbatch \\
           --account $ACCOUNT slurm/score.sbatch
  2. rebuild the catalogue target list - the old targets.parquet holds ids from
     the --min-hac 6 run and would match nothing
       python scripts/build_target_combos.py --work-dir $DEL_WORK \\
           --out targets_hac5.parquet
  3. catalogue stage 2
       WORK_DIR=$CAT_WORK TARGET_COMBOS=targets_hac5.parquet sbatch \\
           --account $ACCOUNT slurm/catalogue_combine.sbatch
  4. stage 7 onwards, pointing at the new directories throughout
EOF
