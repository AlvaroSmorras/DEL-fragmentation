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
"""

from __future__ import annotations

import argparse
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
from lib.io_utils import add_project_root_to_path, input_files, progress, write_summary  # noqa: E402

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
    return {
        "file": path.name,
        "stratum": path.stem,
        "n_compounds": len(out_ids),
        "n_binders": n_binders,
        "n_nonbinders": n_nonbinders,
        "failed": failed,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("data/HGODEL"))
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--min-hac", type=int, default=DEFAULT_MIN_HAC,
                        help="minimum heavy atoms per fragment (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=0, help="only process the first N files (for testing)")
    args = parser.parse_args()

    files = input_files(args.input_dir, args.pattern)
    if args.limit:
        files = files[: args.limit]
    work_dir = args.work_dir
    out_dir = work_dir / "compound_fragments"
    counts_dir = work_dir / "fragment_counts"
    for directory in (out_dir, counts_dir):
        if directory.exists():
            shutil.rmtree(directory)
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    opts = {"min_hac": args.min_hac, "out_dir": str(out_dir), "counts_dir": str(counts_dir)}
    totals = {"n_compounds": 0, "n_binders": 0, "n_nonbinders": 0}
    per_stratum: dict[str, dict] = {}
    failed: list[str] = []

    print(f"fragmenting {len(files)} file(s) with {args.workers} worker(s), min_hac={args.min_hac}")
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(opts,)) as pool:
        stream = pool.imap_unordered(_fragment_file, [str(f) for f in files])
        for result in progress(stream, len(files), "fragment"):
            for key in totals:
                totals[key] += result[key]
            failed.extend(result["failed"])
            per_stratum[result["stratum"]] = {
                "n_binders": result["n_binders"],
                "n_nonbinders": result["n_nonbinders"],
            }

    dictionary = _build_dictionary(counts_dir, work_dir / "_tmp_frag_merge")
    dictionary.to_parquet(work_dir / "fragments.parquet", index=False)
    if failed:
        (work_dir / "failed_smiles.txt").write_text("\n".join(failed) + "\n")

    summary = {
        "min_hac": args.min_hac,
        "input_dir": str(args.input_dir),
        "n_files": len(files),
        "n_unique_fragments": int(len(dictionary)),
        "n_unparsable_smiles": len(failed),
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
