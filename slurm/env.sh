# Cluster environment, sourced by both job scripts.
#
# Check the exact module name once with:  module spider Miniforge
# then set CONDA_MODULE below or export it before submitting.
CONDA_MODULE="${CONDA_MODULE:-Miniforge}"
CONDA_ENV="${CONDA_ENV:-chemprop}"

module purge 2>/dev/null || true
module load "$CONDA_MODULE"

# `conda activate` is a shell function, and a batch shell has not sourced the
# hook that defines it, so activating without this fails with
# "CommandNotFoundError: Your shell has not been properly configured".
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# RDKit, numpy and pyarrow each start their own thread pool.  With one process
# per core those pools multiply and fight for the same cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

python -c "import rdkit, pyarrow, pandas, scipy" || {
    echo "environment '$CONDA_ENV' is missing a dependency" >&2
    exit 1
}
