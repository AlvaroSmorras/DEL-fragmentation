#!/usr/bin/env python
"""Stage 2 - build the contiguous fragment combinations of every compound.

Reads stage 1's output and writes, under ``--work-dir``:

* ``combination_counts/size<k>/*.parquet``  per input file, how many binder and
                                            non-binder compounds carry each
                                            combination
* ``combinations/size<k>/*.parquet``        the long compound-to-combination
                                            table (``--no-long-table`` skips it)
* ``summary_combine.json``

A combination is contiguous when its fragments induce a connected subgraph of
the compound's fragment graph, so ``--sizes 2`` gives the fragment pairs that
are actually bonded to each other.

Combinations are stored as their member fragment ids in ``frag_0..frag_<k-1>``
int64 columns rather than as a joined string: grouping a string key through
pandas costs about 950 bytes per distinct combination, which at 90M
combinations would need far more memory than any single machine has.  Each
combination size gets its own directory so the schema stays fixed.

One input file is one stratum, and its counts are already the per-stratum counts
that stage 3 needs, so nothing has to be regrouped here.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.combinations import combinations_for_compound, combo_columns  # noqa: E402
from lib.io_utils import add_project_root_to_path, input_files, progress, read_summary, write_summary  # noqa: E402

add_project_root_to_path()

_OPTS: dict = {}


def _init_worker(opts: dict) -> None:
    _OPTS.update(opts)


def _write(rows: dict, columns: list[str], extra: dict, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = {col: pa.array(rows[col], pa.int64()) for col in columns}
    data.update(extra)
    pq.write_table(pa.table(data), directory / name, compression="zstd")


def _combine_file(path_str: str) -> dict:
    path = Path(path_str)
    sizes = _OPTS["sizes"]
    counts_root = Path(_OPTS["counts_root"])
    long_root = Path(_OPTS["long_root"]) if _OPTS["long_root"] else None

    table = pq.read_table(path)
    ids = table.column("CompoundIndex").to_pylist()
    activities = table.column("activity").to_pylist()
    frag_id_lists = table.column("frag_ids").to_pylist()
    src_lists = table.column("edge_src").to_pylist()
    dst_lists = table.column("edge_dst").to_pylist()

    # size -> combination tuple -> [binder compounds, non-binder compounds, example]
    counts: dict[int, dict[tuple, list]] = {size: {} for size in sizes}
    long_rows: dict[int, list] = {size: [] for size in sizes}
    n_compounds_with_combo = 0
    n_rows = 0

    for cid, act, frag_ids, src, dst in zip(ids, activities, frag_id_lists, src_lists, dst_lists):
        edges = list(zip(src, dst))
        combos = combinations_for_compound(frag_ids, edges, sizes)
        if not combos:
            continue
        n_compounds_with_combo += 1
        is_binder = int(act) == 1
        for combo in combos:
            bucket = counts[len(combo)]
            entry = bucket.get(combo)
            if entry is None:
                # An example compound is only useful for a combination a binder
                # actually carries, and carrying the string is the expensive part.
                bucket[combo] = [int(is_binder), int(not is_binder), cid if is_binder else None]
            else:
                entry[0] += is_binder
                entry[1] += not is_binder
                if is_binder and entry[2] is None:
                    entry[2] = cid
            if long_root is not None:
                long_rows[len(combo)].append((cid, int(act), combo))
            n_rows += 1

    n_unique = 0
    for size in sizes:
        columns = combo_columns(size)
        bucket = counts[size]
        keys = list(bucket)
        n_unique += len(keys)
        _write(
            {col: [key[i] for key in keys] for i, col in enumerate(columns)},
            columns,
            {
                "n_binder": pa.array([bucket[k][0] for k in keys], pa.int64()),
                "n_nonbinder": pa.array([bucket[k][1] for k in keys], pa.int64()),
                "example_compound": pa.array([bucket[k][2] for k in keys], pa.string()),
            },
            counts_root / f"size{size}",
            path.name,
        )
        if long_root is not None:
            rows = long_rows[size]
            _write(
                {col: [row[2][i] for row in rows] for i, col in enumerate(columns)},
                columns,
                {
                    "CompoundIndex": pa.array([row[0] for row in rows], pa.string()),
                    "activity": pa.array([row[1] for row in rows], pa.int8()),
                },
                long_root / f"size{size}",
                path.name,
            )

    return {
        "file": path.name,
        "n_compounds": len(ids),
        "n_compounds_with_combo": n_compounds_with_combo,
        "n_combination_rows": n_rows,
        "n_unique_in_file": n_unique,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--sizes", default="2",
                        help="comma-separated combination sizes, e.g. '2' or '2,3' (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--no-long-table", action="store_true",
                        help="only write the aggregated counts, not the per-compound table; "
                             "worth setting on very large libraries")
    args = parser.parse_args()

    work_dir = args.work_dir
    sizes = tuple(sorted({int(s) for s in args.sizes.split(",") if s.strip()}))
    if not sizes or min(sizes) < 2:
        raise SystemExit("--sizes must list integers >= 2")

    files = input_files(work_dir / "compound_fragments")
    counts_root = work_dir / "combination_counts"
    long_root = None if args.no_long_table else work_dir / "combinations"
    for directory in (counts_root, long_root):
        if directory is not None and directory.exists():
            shutil.rmtree(directory)

    started = time.time()
    opts = {
        "sizes": sizes,
        "counts_root": str(counts_root),
        "long_root": None if long_root is None else str(long_root),
    }
    totals: dict[str, int] = defaultdict(int)
    print(f"combining {len(files)} file(s) with {args.workers} worker(s), sizes={sizes}")
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(opts,)) as pool:
        stream = pool.imap_unordered(_combine_file, [str(f) for f in files])
        for result in progress(stream, len(files), "combine"):
            for key in ("n_compounds", "n_compounds_with_combo", "n_combination_rows", "n_unique_in_file"):
                totals[key] += result[key]

    stage1 = read_summary(work_dir / "summary_fragment.json")
    summary = {
        "sizes": list(sizes),
        "min_hac": stage1["min_hac"],
        "n_files": len(files),
        "long_table": long_root is not None,
        "elapsed_s": round(time.time() - started, 1),
        **dict(totals),
    }
    write_summary(work_dir / "summary_combine.json", summary)
    print(
        f"done in {summary['elapsed_s']}s: {totals['n_compounds_with_combo']:,} compounds "
        f"produced {totals['n_combination_rows']:,} combination rows"
    )


if __name__ == "__main__":
    main()
