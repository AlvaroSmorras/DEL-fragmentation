#!/bin/bash
# Fold the shards of a re-fragmentation run into the fragment dictionary.
#
# Runs under --dependency=afterok on the array job, so it starts only if every
# shard succeeded.  Builds fragments.parquet and summary_fragment.json, which is
# what makes the work directory usable by stage 2 onwards.
#SBATCH --job-name=brics-rerun-reduce
#SBATCH --output=logs/rerun_reduce_%j.out
#SBATCH --error=logs/rerun_reduce_%j.err
set -euo pipefail

cd "${PROJECT_DIR:?set by rerun_fragmentation.sh}"
source slurm/env.sh

python scripts/01_fragment.py --work-dir "$WORK_DIR" --reduce-only

python - "$WORK_DIR" <<'PY'
import json, sys
from pathlib import Path

summary = json.loads((Path(sys.argv[1]) / "summary_fragment.json").read_text())
print(f"\n{sys.argv[1]}: {summary['n_compounds']:,} compounds, "
      f"{summary['n_unique_fragments']:,} distinct fragments, "
      f"min_hac={summary['min_hac']}, selector={summary.get('selector', 'greedy')}")
PY
