#!/usr/bin/env python
"""Stage 0 (optional) - re-split the input into evenly sized parquet files.

One input file is one task in stage 1, so the file count caps how many workers
can be busy at once, and each worker holds one whole file's working set in
memory.  A delivery of a few huge files therefore parallelises badly and needs
large nodes; many modest files parallelise well on small ones.

Splitting is safe.  Each file is a stratum, so splitting one sub-library into
many files turns one stratum into many - which Mantel-Haenszel handles without
moving the enrichments (measured: a 128-way split agrees to Spearman 0.9998).
**Never concatenate different sub-libraries into one file**, which is the
destructive direction: it pools strata that differ, and inflates enrichments
severalfold.

Files are streamed a row group at a time, so a source file far larger than
memory still splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.io_utils import add_project_root_to_path, input_files, progress  # noqa: E402

add_project_root_to_path()

DEFAULT_ROWS = 90_000


def _flush(batches: list[pa.RecordBatch], schema: pa.Schema, path: Path) -> None:
    pq.write_table(pa.Table.from_batches(batches, schema=schema), path, compression="zstd")


def split_file(path: Path, out_dir: Path, target_rows: int) -> int:
    """Split one file into even pieces of at most ``target_rows``, keeping its stem.

    Pieces are evened out rather than filled to the brim, so a file of 89k rows
    at a target of 20k becomes five of ~17.8k rather than four of 20k and a
    stub.  Tiny trailing files would waste a task slot each.
    """
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    n_rows = parquet.metadata.num_rows
    n_pieces = max(1, -(-n_rows // target_rows))
    target_rows = max(1, -(-n_rows // n_pieces))
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    written = 0

    def emit() -> None:
        nonlocal pending, pending_rows, written
        if pending_rows:
            _flush(pending, schema, out_dir / f"{path.stem}_{written:04d}.parquet")
            written += 1
            pending, pending_rows = [], 0

    for batch in parquet.iter_batches(batch_size=min(target_rows, 65_536)):
        offset = 0
        while offset < batch.num_rows:
            room = target_rows - pending_rows
            piece = batch.slice(offset, room)
            pending.append(piece)
            pending_rows += piece.num_rows
            offset += piece.num_rows
            if pending_rows >= target_rows:
                emit()
    emit()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                        help="rows per output file (default: %(default)s)")
    args = parser.parse_args()

    if args.out_dir.resolve() == args.input_dir.resolve():
        raise SystemExit("--out-dir must differ from --input-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = input_files(args.input_dir, args.pattern)
    total_in = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    written = 0
    for path in progress(files, len(files), "split"):
        written += split_file(path, args.out_dir, args.rows)

    total_out = sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(args.out_dir.glob("*.parquet")))
    if total_out != total_in:
        raise SystemExit(f"row count changed: {total_in:,} in, {total_out:,} out")
    print(f"{len(files):,} file(s), {total_in:,} rows -> {written:,} file(s) of at most {args.rows:,}")
    print(f"suggested stage 1 array size: {min(written, 256)} tasks (speedup caps at {written})")


if __name__ == "__main__":
    main()
