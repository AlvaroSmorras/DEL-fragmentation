#!/usr/bin/env python
"""Stage 1 - break every compound into BRICS fragments and name them.

Writes, under ``--work-dir``:

* ``fragments.parquet``            the fragment dictionary
* ``compound_fragments/*.parquet`` per input file, each compound's fragments and
                                   the edges between them
* ``fragment_counts/*.parquet``    per input file, per-fragment compound counts
* ``summary_fragment.json``        totals, plus per-stratum totals for stage 3

Fragment ids are content hashes of the fragment SMILES, so a worker can write
its final output immediately: there is no global numbering pass over every
compound, and no dictionary to broadcast back to the workers.  The dictionary is
folded together afterwards from the small per-file count files.

That same property lets the work split across machines.  ``--shard i/N``
processes one slice of the input and stops before the fold, so N array tasks can
run with no coordination between them; ``--reduce-only`` then folds their output
into one dictionary.  Ids agree across shards because they depend only on the
fragment, never on what else a machine happened to see.

    sbatch --array=0-63 ... scripts/01_fragment.py --shard $SLURM_ARRAY_TASK_ID/64
    python scripts/01_fragment.py --reduce-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.aggregate import choose_buckets, group_sum, read_frames, total_rows  # noqa: E402
from lib.fragmentation import DEFAULT_MIN_HAC, fragment_id, fragment_smiles  # noqa: E402
from lib.io_utils import (  # noqa: E402
    add_project_root_to_path, input_files, progress, read_summary, write_summary,
)

add_project_root_to_path()

_OPTS: dict = {}


def _init_worker(opts: dict) -> None:
    _OPTS.update(opts)


def _fragment_file(path_str: str) -> dict:
    """Fragment one input parquet.  One input file is one stratum."""
    path = Path(path_str)
    min_hac = _OPTS["min_hac"]
    out_dir = Path(_OPTS["out_dir"])
    counts_dir = Path(_OPTS["counts_dir"])

    table = pq.read_table(path, columns=["CompoundIndex", "Smiles", "activity"])
    ids = table.column("CompoundIndex").to_pylist()
    smiles = table.column("Smiles").to_pylist()
    activities = table.column("activity").to_pylist()

    out_ids, out_act, out_frags, out_src, out_dst = [], [], [], [], []
    # fragment id -> [smiles, hac, attachments, compounds, binder compounds]
    counts: dict[int, list] = {}
    cache: dict[str, tuple | None] = {}
    failed: list[str] = []
    n_binders = n_nonbinders = 0

    for cid, smi, act in zip(ids, smiles, activities):
        cached = cache.get(smi, False)
        if cached is False:
            frag = fragment_smiles(smi, min_hac=min_hac)
            if frag is None:
                cached = None
            else:
                frag_ids = tuple(fragment_id(s) for s in frag.smiles)
                cached = (frag_ids, frag.smiles, frag.hac, frag.edges)
            cache[smi] = cached
        if cached is None:
            failed.append(cid)
            continue

        frag_ids, frag_smis, hacs, edges = cached
        is_binder = int(act) == 1
        n_binders += is_binder
        n_nonbinders += not is_binder
        for fid, frag_smi, hac in set(zip(frag_ids, frag_smis, hacs)):
            entry = counts.get(fid)
            if entry is None:
                counts[fid] = [frag_smi, hac, frag_smi.count("*"), 1, int(is_binder)]
            else:
                entry[3] += 1
                entry[4] += is_binder

        out_ids.append(cid)
        out_act.append(int(act))
        out_frags.append(list(frag_ids))
        out_src.append([edge[0] for edge in edges])
        out_dst.append([edge[1] for edge in edges])

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "CompoundIndex": pa.array(out_ids, pa.string()),
                "activity": pa.array(out_act, pa.int8()),
                "frag_ids": pa.array(out_frags, pa.list_(pa.int64())),
                "edge_src": pa.array(out_src, pa.list_(pa.int16())),
                "edge_dst": pa.array(out_dst, pa.list_(pa.int16())),
            }
        ),
        out_dir / path.name,
        compression="zstd",
    )

    keys = list(counts)
    counts_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "frag_id": pa.array(keys, pa.int64()),
                "frag_smiles": pa.array([counts[k][0] for k in keys], pa.string()),
                "hac": pa.array([counts[k][1] for k in keys], pa.int16()),
                "n_attachments": pa.array([counts[k][2] for k in keys], pa.int16()),
                "n_compounds": pa.array([counts[k][3] for k in keys], pa.int64()),
                "n_binder": pa.array([counts[k][4] for k in keys], pa.int64()),
            }
        ),
        counts_dir / path.name,
        compression="zstd",
    )
    # Written last, once both parquets are safely on disk: its presence is what
    # marks this file as finished, so --resume can trust it.
    stats_dir = Path(_OPTS["stats_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)
    if failed:
        (stats_dir / f"{path.stem}.failed.txt").write_text("\n".join(failed) + "\n")
    (stats_dir / f"{path.stem}.json").write_text(json.dumps({
        "stratum": path.stem,
        "n_compounds": len(out_ids),
        "n_binders": n_binders,
        "n_nonbinders": n_nonbinders,
        "n_unparsable_smiles": len(failed),
    }, sort_keys=True))
    return {"file": path.name, "n_compounds": len(out_ids)}


def _build_dictionary(counts_dir: Path, tmp_dir: Path) -> pd.DataFrame:
    """Fold per-file fragment counts into one dictionary, out of core."""
    files = sorted(counts_dir.glob("*.parquet"))
    n_buckets = choose_buckets(total_rows(files))
    print(f"folding fragment counts from {len(files)} file(s) into {n_buckets} bucket(s)")
    merged = group_sum(
        read_frames(files),
        key_cols=["frag_id"],
        sum_cols=["n_compounds", "n_binder"],
        first_cols=["frag_smiles", "hac", "n_attachments"],
        tmp_dir=tmp_dir,
        n_buckets=n_buckets,
    )
    merged["n_nonbinder"] = merged["n_compounds"] - merged["n_binder"]
    # Short names are cosmetic: most common first, so F000000 is the commonest.
    merged = merged.sort_values(["n_compounds", "frag_smiles"], ascending=[False, True])
    merged = merged.reset_index(drop=True)
    merged.insert(1, "frag_name", [f"F{i:06d}" for i in range(len(merged))])
    return merged


def _parse_shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise SystemExit(f"--shard wants 'i/N', e.g. '0/64', not {text!r}")
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"--shard index must be in 0..{count - 1}, got {index}")
    return index, count


def _shard_files(files: list[Path], index: int, count: int) -> list[Path]:
    """Split the input across shards, balancing by row count.

    Array tasks all wait for the slowest, so handing out the biggest files first
    and always to the least loaded shard keeps them finishing together.  The
    split depends only on the file set, so every task computes the same one.
    """
    weighted = sorted(
        ((pq.ParquetFile(path).metadata.num_rows, str(path), path) for path in files),
        key=lambda item: (-item[0], item[1]),
    )
    loads = [0] * count
    buckets: list[list[Path]] = [[] for _ in range(count)]
    for rows, _, path in weighted:
        target = min(range(count), key=lambda k: (loads[k], k))
        loads[target] += rows
        buckets[target].append(path)
    return sorted(buckets[index])


def _is_complete(work_dir: Path, stem: str) -> bool:
    """Has this input file already been fragmented, completely and readably?

    A job killed mid-write can leave a truncated parquet, so the stats file
    existing is not enough - both outputs have to open.
    """
    if not (work_dir / "file_stats" / f"{stem}.json").exists():
        return False
    for directory in ("compound_fragments", "fragment_counts"):
        path = work_dir / directory / f"{stem}.parquet"
        try:
            pq.ParquetFile(path).metadata
        except Exception:
            return False
    return True


def _collect_file_stats(work_dir: Path) -> tuple[dict, dict, int] | None:
    """Fold the per-file records into totals.

    These are per *input file*, not per shard or per run, so the totals come out
    the same however the work was divided up and however often it was resumed.
    """
    paths = sorted((work_dir / "file_stats").glob("*.json"))
    if not paths:
        return None
    totals = {"n_compounds": 0, "n_binders": 0, "n_nonbinders": 0}
    per_stratum: dict[str, dict] = {}
    n_failed = 0
    for path in paths:
        payload = json.loads(path.read_text())
        for key in totals:
            totals[key] += payload[key]
        per_stratum[payload["stratum"]] = {
            "n_binders": payload["n_binders"], "n_nonbinders": payload["n_nonbinders"],
        }
        n_failed += payload["n_unparsable_smiles"]
    print(f"folding {len(paths):,} per-file record(s)")
    return totals, per_stratum, n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data/HGODEL"))
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--min-hac", type=int, default=DEFAULT_MIN_HAC,
                        help="minimum heavy atoms per fragment (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=0, help="only process the first N files (for testing)")
    parser.add_argument("--shard", default=None, metavar="i/N",
                        help="fragment only shard i of N (0-based) and stop before the fold, "
                             "so array tasks can run on separate machines")
    parser.add_argument("--reduce-only", action="store_true",
                        help="skip fragmenting; fold what the shards already wrote")
    parser.add_argument("--resume", action="store_true",
                        help="skip input files that already have complete output, so a "
                             "failed run picks up where it stopped")
    args = parser.parse_args()

    shard = _parse_shard(args.shard) if args.shard else None
    if shard and args.reduce_only:
        raise SystemExit("--shard and --reduce-only are separate steps; run them one after the other")

    work_dir = args.work_dir
    out_dir = work_dir / "compound_fragments"
    counts_dir = work_dir / "fragment_counts"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    stats_dir = work_dir / "file_stats"
    n_processed = 0

    if not args.reduce_only:
        files = input_files(args.input_dir, args.pattern)
        if args.limit:
            files = files[: args.limit]
        if shard:
            files = _shard_files(files, *shard)
            if not files:
                raise SystemExit(f"shard {args.shard} is empty; use fewer shards than files")
        elif not args.resume:
            # Only a whole, fresh run may clear previous output - a shard would
            # be deleting its siblings' work, and a resume its own.
            for directory in (out_dir, counts_dir, stats_dir):
                if directory.exists():
                    shutil.rmtree(directory)

        if args.resume:
            wanted = len(files)
            files = [path for path in files if not _is_complete(work_dir, path.stem)]
            skipped = wanted - len(files)
            if skipped:
                print(f"resuming: {skipped:,} of {wanted:,} file(s) already done")
            if not files:
                print("nothing left to fragment")

        opts = {"min_hac": args.min_hac, "out_dir": str(out_dir),
                "counts_dir": str(counts_dir), "stats_dir": str(stats_dir)}
        label = f"shard {shard[0]}/{shard[1]}: " if shard else ""
        if files:
            print(f"{label}fragmenting {len(files)} file(s) with {args.workers} worker(s), "
                  f"min_hac={args.min_hac}")
            with mp.Pool(args.workers, initializer=_init_worker, initargs=(opts,)) as pool:
                stream = pool.imap_unordered(_fragment_file, [str(f) for f in files])
                for result in progress(stream, len(files), "fragment"):
                    n_processed += result["n_compounds"]

    if shard:
        shard_dir = work_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        write_summary(shard_dir / f"shard{shard[0]:04d}.json", {
            "shard": shard[0], "n_shards": shard[1], "min_hac": args.min_hac,
            "n_files": len(files), "n_compounds": n_processed,
            "elapsed_s": round(time.time() - started, 1),
        })
        print(f"shard {shard[0]}/{shard[1]} done in {time.time() - started:.1f}s: "
              f"{n_processed:,} compounds. Run --reduce-only once every shard has finished.")
        return

    collected = _collect_file_stats(work_dir)
    if collected is None:
        raise SystemExit(f"no per-file records under {stats_dir}; nothing to fold")
    totals, per_stratum, n_failed = collected
    n_input_files = len(per_stratum)
    failures = sorted(stats_dir.glob("*.failed.txt"))
    if failures:
        (work_dir / "failed_smiles.txt").write_text(
            "".join(path.read_text() for path in failures)
        )

    dictionary = _build_dictionary(counts_dir, work_dir / "_tmp_frag_merge")
    dictionary.to_parquet(work_dir / "fragments.parquet", index=False)

    summary = {
        "min_hac": args.min_hac,
        "input_dir": str(args.input_dir),
        "n_files": n_input_files,
        "n_unique_fragments": int(len(dictionary)),
        "n_unparsable_smiles": n_failed,
        "elapsed_s": round(time.time() - started, 1),
        "per_stratum": per_stratum,
        **totals,
    }
    write_summary(work_dir / "summary_fragment.json", summary)
    print(
        f"done in {summary['elapsed_s']}s: {totals['n_compounds']:,} compounds "
        f"({totals['n_binders']:,} binders / {totals['n_nonbinders']:,} non-binders), "
        f"{len(dictionary):,} unique fragments"
    )


if __name__ == "__main__":
    main()
