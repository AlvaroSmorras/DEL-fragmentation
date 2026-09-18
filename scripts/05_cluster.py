#!/usr/bin/env python
"""Stage 5 - group near-identical combinations and pool their enrichment.

Combinations that differ only by a decoration - a fluorine here, a methyl there
- are usually one structural signal split across several rows, which wastes the
compounds supporting it.  This stage gives each fragment a scaffold key, pairs
those keys to cluster the combinations, and re-scores the clusters.

Writes, per combination size, under ``--results-dir``:

* ``cluster_enrichment_size<k>.parquet``  one row per cluster
* ``top_clusters_size<k>.csv``            the best-scoring clusters and members

Cluster counts are recounted from the per-compound table rather than summed
from the member combinations, because a compound carrying two members of one
cluster must still count once.  ``--approximate`` sums instead, for runs made
with ``--no-long-table``.

Pooling is only worth it when the members really are the same signal, so every
cluster reports its best single member alongside the pooled score: if pooling
dilutes rather than reinforces, the comparison shows it.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.aggregate import choose_buckets, group_sum, total_rows  # noqa: E402
from lib.combinations import combo_columns  # noqa: E402
from lib.enrichment import (  # noqa: E402
    DEFAULT_ALPHA, MH_TERMS, add_enrichment, add_mantel_haenszel,
    benjamini_hochberg, stratum_terms,
)
from lib.fragmentation import fragment_id  # noqa: E402
from lib.io_utils import add_project_root_to_path, progress, read_summary, write_summary  # noqa: E402
from lib.scaffolds import DEFAULT_MAX_STRIP, DEFAULT_MODE, MODES, scaffold_keys  # noqa: E402

add_project_root_to_path()

_OPTS: dict = {}


def _init_worker(opts: dict) -> None:
    _OPTS.update(opts)


def _keys_chunk(chunk: list[str]) -> list[str]:
    return scaffold_keys(chunk, _OPTS["mode"], _OPTS["max_strip"])


def _recount_file(path_str: str) -> str:
    """Per-stratum cluster counts, deduplicated per compound."""
    path = Path(path_str)
    mapping = pd.read_parquet(_OPTS["map_path"])
    keys = [col for col in mapping.columns if col.startswith("frag_")]
    out_dir = Path(_OPTS["counts_dir"])

    long_table = pd.read_parquet(path)
    merged = long_table.merge(mapping, on=keys, how="inner")
    # One compound carrying two members of a cluster still counts once.
    merged = merged.drop_duplicates(["CompoundIndex", "cluster_id"])

    grouped = merged.groupby("cluster_id", as_index=False).agg(
        n_binder=("activity", "sum"), n_total=("activity", "size")
    )
    grouped["n_nonbinder"] = grouped["n_total"] - grouped["n_binder"]
    binders = merged[merged["activity"] == 1]
    example = binders.groupby("cluster_id")["CompoundIndex"].first()
    grouped["example_compound"] = grouped["cluster_id"].map(example)

    out_dir.mkdir(parents=True, exist_ok=True)
    grouped.drop(columns=["n_total"]).to_parquet(out_dir / path.name, index=False)
    return path.stem


def _approximate_counts(counts_dir: Path, mapping: pd.DataFrame, out_dir: Path) -> None:
    """Sum member counts per stratum, double-counting shared compounds."""
    keys = [col for col in mapping.columns if col.startswith("frag_")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(counts_dir.glob("*.parquet")):
        frame = pd.read_parquet(path).merge(mapping, on=keys, how="inner")
        grouped = frame.groupby("cluster_id", as_index=False).agg(
            n_binder=("n_binder", "sum"),
            n_nonbinder=("n_nonbinder", "sum"),
            example_compound=("example_compound", "first"),
        )
        grouped.to_parquet(out_dir / path.name, index=False)


def _cluster_members(combos: pd.DataFrame, keys: dict[int, str], size: int) -> pd.DataFrame:
    """Attach each combination to its cluster, and describe the clusters."""
    cols = combo_columns(size)
    members = combos.copy()
    cluster_keys = [
        "|".join(sorted(keys.get(fid, str(fid)) for fid in row))
        for row in members[cols].to_numpy()
    ]
    members["cluster_key"] = cluster_keys
    members["cluster_id"] = [fragment_id(key) for key in cluster_keys]

    rank = "enrichment_mh_lo95" if "enrichment_mh_lo95" in members else "enrichment_factor_lo95"
    members = members.sort_values(rank, ascending=False)
    described = members.groupby("cluster_id", as_index=False).agg(
        cluster_key=("cluster_key", "first"),
        n_members=("combo_key", "size"),
        best_member=("combo_key", "first"),
        best_member_enrichment_lo95=(rank, "first"),
        best_member_n_binder=("n_binder", "first"),
        members=("combo_key", lambda names: ";".join(names)),
        member_smiles=("frag_smiles", "first"),
    )
    return members, described


def _score_size(size: int, args, stage1, keys: dict[int, str]) -> pd.DataFrame | None:
    path = args.results_dir / f"combination_enrichment_size{size}.parquet"
    if not path.exists():
        return None
    combos = pd.read_parquet(path)
    if combos.empty:
        return None
    cols = combo_columns(size)

    members, described = _cluster_members(combos, keys, size)
    print(f"  size {size}: {len(combos):,} combinations -> {len(described):,} clusters "
          f"({(described.n_members > 1).sum():,} with more than one member)")

    mapping = members[cols + ["cluster_id"]].drop_duplicates()
    map_path = args.work_dir / f"_cluster_map_size{size}.parquet"
    mapping.to_parquet(map_path, index=False)

    counts_dir = args.work_dir / "cluster_counts" / f"size{size}"
    if counts_dir.exists():
        shutil.rmtree(counts_dir)
    long_dir = args.work_dir / "combinations" / f"size{size}"

    if args.approximate or not long_dir.exists():
        if not args.approximate:
            print("    no per-compound table; falling back to summed member counts")
        _approximate_counts(args.work_dir / "combination_counts" / f"size{size}",
                            mapping, counts_dir)
    else:
        files = sorted(long_dir.glob("*.parquet"))
        opts = {"map_path": str(map_path), "counts_dir": str(counts_dir)}
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(opts,)) as pool:
            stream = pool.imap_unordered(_recount_file, [str(f) for f in files])
            for _ in progress(stream, len(files), "recount"):
                pass

    totals = {"n_binders": stage1["n_binders"], "n_nonbinders": stage1["n_nonbinders"]}
    per_stratum = stage1.get("per_stratum", {})
    stratify = args.stratify == "file"
    count_files = sorted(counts_dir.glob("*.parquet"))

    def frames():
        for path in count_files:
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            if stratify:
                stratum = per_stratum.get(path.stem, totals)
                terms = stratum_terms(frame["n_binder"], frame["n_nonbinder"],
                                      stratum["n_binders"], stratum["n_nonbinders"])
                for name, values in terms.items():
                    frame[name] = values
            frame["n_strata"] = np.int64(1)
            yield frame

    sums = ["n_binder", "n_nonbinder", "n_strata"] + (list(MH_TERMS) if stratify else [])
    merged = group_sum(
        frames(), key_cols=["cluster_id"], sum_cols=sums, first_cols=["example_compound"],
        tmp_dir=args.work_dir / f"_tmp_cluster_merge_size{size}",
        n_buckets=choose_buckets(total_rows(count_files)),
    )

    merged = add_enrichment(merged, totals["n_binders"], totals["n_nonbinders"],
                            alpha=args.alpha, with_test=False)
    if stratify:
        merged = add_mantel_haenszel(merged, alpha=args.alpha)
        merged["q_value_mh"] = benjamini_hochberg(merged["p_value_mh"].to_numpy())
        rank = "enrichment_mh_lo95"
        merged = merged.drop(columns=list(MH_TERMS))
    else:
        rank = "enrichment_factor_lo95"

    merged = merged.merge(described, on="cluster_id", how="left")
    merged["pooling_gain"] = merged[rank] / merged["best_member_enrichment_lo95"]
    merged["combo_size"] = np.int8(size)
    return merged.sort_values(rank, ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE,
                        help="how tightly fragments are normalised (default: %(default)s)")
    parser.add_argument("--max-strip", type=int, default=DEFAULT_MAX_STRIP,
                        help="atoms a fragment may lose before it keeps its exact "
                             "identity instead (default: %(default)s)")
    parser.add_argument("--stratify", choices=("file", "none"), default="file")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--approximate", action="store_true",
                        help="sum member counts instead of recounting compounds")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--top", type=int, default=500)
    args = parser.parse_args()

    stage1 = read_summary(args.work_dir / "summary_fragment.json")
    stage2 = read_summary(args.work_dir / "summary_combine.json")
    started = time.time()

    dictionary = pd.read_parquet(args.work_dir / "fragments.parquet")[["frag_id", "frag_smiles"]]
    smiles = dictionary["frag_smiles"].tolist()
    print(f"normalising {len(smiles):,} fragments (mode={args.mode}, max_strip={args.max_strip})")
    chunk_size = max(1, len(smiles) // (args.workers * 4) + 1)
    chunks = [smiles[i:i + chunk_size] for i in range(0, len(smiles), chunk_size)]
    opts = {"mode": args.mode, "max_strip": args.max_strip}
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(opts,)) as pool:
        parts = list(progress(pool.imap(_keys_chunk, chunks), len(chunks), "normalise"))
    keys = dict(zip(dictionary["frag_id"].tolist(), [k for part in parts for k in part]))
    print(f"  {len(set(keys.values())):,} distinct scaffold keys")

    summary = {"mode": args.mode, "max_strip": args.max_strip, "stratify": args.stratify,
               "approximate": args.approximate, "n_scaffold_keys": len(set(keys.values())),
               "sizes": {}}

    for size in stage2["sizes"]:
        scored = _score_size(size, args, stage1, keys)
        if scored is None or scored.empty:
            continue
        scored.to_parquet(args.results_dir / f"cluster_enrichment_size{size}.parquet", index=False)
        columns = [c for c in (
            "cluster_key", "n_members", "n_binder", "n_nonbinder", "n_strata",
            "enrichment_mh", "enrichment_mh_lo95", "q_value_mh",
            "enrichment_factor", "enrichment_factor_lo95",
            "best_member", "best_member_enrichment_lo95", "pooling_gain",
            "example_compound", "members",
        ) if c in scored.columns]
        scored.head(args.top)[columns].to_csv(
            args.results_dir / f"top_clusters_size{size}.csv", index=False)
        multi = scored[scored["n_members"] > 1]
        summary["sizes"][str(size)] = {
            "n_clusters": int(len(scored)),
            "n_multi_member": int(len(multi)),
            "median_pooling_gain_multi": float(multi["pooling_gain"].median()) if len(multi) else None,
        }
        print(f"  size {size}: wrote {len(scored):,} clusters")

    summary["elapsed_s"] = round(time.time() - started, 1)
    write_summary(args.results_dir / "summary_clusters.json", summary)
    print(f"done in {summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
