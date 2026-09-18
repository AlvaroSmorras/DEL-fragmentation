"""External group-and-sum for tables far larger than memory.

The pipeline's reduce steps (fragment counts, combination counts) group by keys
whose cardinality grows almost linearly with the library: a 300M-compound set
yields roughly 9M fragments and 90M combinations, which a single in-memory
``groupby`` cannot hold.

Rows are therefore hash-partitioned onto disk by key, so every row sharing a key
lands in the same bucket, and each bucket is then grouped on its own.  Peak
memory is set by the bucket size rather than by the number of groups, and the
filter that drops uninteresting groups runs *inside* each bucket so the
concatenated result stays small.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Rows per bucket we aim for; small enough that one bucket groups comfortably.
TARGET_BUCKET_ROWS = 20_000_000
_MIX = np.uint64(0x9E3779B97F4A7C15)


def bucket_of(frame: pd.DataFrame, key_cols: Sequence[str], n_buckets: int) -> np.ndarray:
    """Assign each row a bucket from its key, so equal keys always agree."""
    if n_buckets == 1:
        return np.zeros(len(frame), dtype=np.int32)
    digest = np.zeros(len(frame), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for col in key_cols:
            values = frame[col].to_numpy(dtype=np.int64).view(np.uint64)
            digest ^= values * _MIX
            digest ^= digest >> np.uint64(29)
            digest *= _MIX
    return (digest % np.uint64(n_buckets)).astype(np.int32)


def choose_buckets(total_rows: int, target: int = TARGET_BUCKET_ROWS) -> int:
    return max(1, int(np.ceil(total_rows / target)))


def group_sum(
    frames: Iterable[pd.DataFrame],
    key_cols: Sequence[str],
    sum_cols: Sequence[str],
    tmp_dir: Path,
    first_cols: Sequence[str] = (),
    n_buckets: int = 1,
    post_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    """Sum ``sum_cols`` over ``key_cols`` across chunks too large to concatenate.

    ``first_cols`` are carried along by taking the first non-null value per
    group.  ``post_filter`` is applied per bucket, which is what keeps the
    result small when most groups are uninteresting.
    """
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    key_cols = list(key_cols)
    sum_cols = list(sum_cols)
    first_cols = list(first_cols)
    wanted = key_cols + sum_cols + first_cols

    # Phase 1 - scatter rows to buckets on disk.
    written = 0
    for seq, frame in enumerate(frames):
        if frame.empty:
            continue
        frame = frame[wanted]
        buckets = bucket_of(frame, key_cols, n_buckets)
        for bucket in np.unique(buckets):
            part = frame.loc[buckets == bucket]
            target = tmp_dir / f"b{int(bucket):04d}"
            target.mkdir(exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(part, preserve_index=False),
                target / f"{seq:06d}.parquet",
                compression="zstd",
            )
            written += len(part)

    # Phase 2 - group each bucket independently.
    how = {col: "sum" for col in sum_cols}
    how.update({col: "first" for col in first_cols})
    results = []
    bucket_dirs = sorted(tmp_dir.glob("b*"))
    for bucket_dir in bucket_dirs:
        parts = [pd.read_parquet(path) for path in sorted(bucket_dir.glob("*.parquet"))]
        if not parts:
            continue
        grouped = (
            pd.concat(parts, ignore_index=True)
            .groupby(key_cols, as_index=False, sort=False)
            .agg(how)
        )
        if post_filter is not None:
            grouped = post_filter(grouped)
        if not grouped.empty:
            results.append(grouped)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not results:
        return pd.DataFrame(columns=wanted)
    return pd.concat(results, ignore_index=True)


def read_frames(paths: Sequence[Path], columns: Sequence[str] | None = None):
    for path in paths:
        yield pd.read_parquet(path, columns=list(columns) if columns else None)


def total_rows(paths: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
