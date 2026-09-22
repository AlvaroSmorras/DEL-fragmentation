#!/usr/bin/env python
"""Stage 3 - score every fragment combination by how much binders prefer it.

Merges stage 2's per-stratum counts and writes, per combination size, under
``--results-dir``:

* ``combination_enrichment_size<k>.parquet``  one row per combination
* ``top_combinations_size<k>.csv``            the best-scoring ones, readable
* ``fragment_enrichment.parquet``             the same for single fragments
* ``summary_enrichment.json``

Two things make this work on libraries far larger than memory.

The merge is hash-partitioned onto disk rather than held in RAM, so its cost is
set by the size of one bucket instead of by the number of distinct
combinations.  And ``--min-binder`` is applied *inside* each bucket, before the
buckets are concatenated: a combination no binder carries cannot be enriched,
and on a large library with a low hit rate that is the overwhelming majority of
them, so dropping them during the merge is what keeps the result small.

By default each input file is treated as its own stratum and combinations are
scored with a Mantel-Haenszel estimate, which compares compounds only against
others from the same sub-library.  Sub-libraries differ wildly in how many
binders they contain, and a pooled score mostly rediscovers that difference.
Pass ``--stratify none`` for a library that ships as arbitrary shards of one
homogeneous set.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.aggregate import choose_buckets, group_sum, total_rows  # noqa: E402
from lib.combinations import combo_columns, combo_name  # noqa: E402
from lib.enrichment import (  # noqa: E402
    DEFAULT_ALPHA, DEFAULT_RANKING, MH_TERMS, RANKINGS, add_enrichment,
    add_mantel_haenszel, benjamini_hochberg, rank_columns, stratum_terms,
)
from lib.io_utils import add_project_root_to_path, progress, read_summary, write_summary  # noqa: E402

add_project_root_to_path()


def _stratified_frames(files, per_stratum, totals, stratify):
    """Per-file counts, carrying the Mantel-Haenszel terms for their stratum."""
    for path in files:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        if stratify:
            stratum = per_stratum.get(path.stem, totals)
            n_b, n_n = stratum["n_binders"], stratum["n_nonbinders"]
            terms = stratum_terms(frame["n_binder"], frame["n_nonbinder"], n_b, n_n)
            for name, values in terms.items():
                frame[name] = values
        frame["n_strata"] = np.int64(1)
        yield frame


def _load_fragment_labels(path: Path, needed: set[int]) -> dict[int, tuple[str, str, int]]:
    """Look up only the fragments that survived, a row group at a time.

    The dictionary has one row per fragment in the whole library; materialising
    all of it to label a few thousand results would undo the point of the merge.
    """
    labels: dict[int, tuple[str, str, int]] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["frag_id", "frag_name", "frag_smiles", "hac"]):
        frag_ids = batch.column("frag_id").to_pylist()
        names = batch.column("frag_name").to_pylist()
        smiles = batch.column("frag_smiles").to_pylist()
        hac = batch.column("hac").to_pylist()
        for fid, name, smi, h in zip(frag_ids, names, smiles, hac):
            if fid in needed:
                labels[fid] = (name, smi, h)
    return labels


def _score(size: int, files, args, stage1, stratify: bool) -> pd.DataFrame:
    totals = {"n_binders": stage1["n_binders"], "n_nonbinders": stage1["n_nonbinders"]}
    per_stratum = stage1.get("per_stratum", {})
    keys = combo_columns(size)
    sums = ["n_binder", "n_nonbinder", "n_strata"] + (list(MH_TERMS) if stratify else [])

    n_buckets = choose_buckets(total_rows(files))
    print(f"  size {size}: merging {len(files)} stratum file(s) into {n_buckets} bucket(s)")

    min_binder, min_total = args.min_binder, args.min_count

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[
            (frame["n_binder"] >= min_binder)
            & ((frame["n_binder"] + frame["n_nonbinder"]) >= min_total)
        ]

    merged = group_sum(
        _stratified_frames(files, per_stratum, totals, stratify),
        key_cols=keys,
        sum_cols=sums,
        first_cols=["example_compound"],
        tmp_dir=args.work_dir / f"_tmp_combo_merge_size{size}",
        n_buckets=n_buckets,
        post_filter=keep,
    )
    if merged.empty:
        return merged

    merged = add_enrichment(
        merged, totals["n_binders"], totals["n_nonbinders"], alpha=args.alpha, with_test=False
    )
    if stratify:
        merged = add_mantel_haenszel(merged, alpha=args.alpha)
        merged["q_value_mh"] = benjamini_hochberg(merged["p_value_mh"].to_numpy())
        merged = merged.drop(columns=list(MH_TERMS))
    else:
        from scipy.stats import hypergeom

        total = float(totals["n_binders"] + totals["n_nonbinders"])
        merged["p_value"] = hypergeom.sf(
            merged["n_binder"] - 1, total, float(totals["n_binders"]),
            merged["n_binder"] + merged["n_nonbinder"],
        )
        merged["q_value"] = benjamini_hochberg(merged["p_value"].to_numpy())

    rank_on = rank_columns(args.rank, merged.columns)
    merged = merged.sort_values(rank_on, ascending=False).reset_index(drop=True)
    # Downstream stages take the first N rows, so record what that order means.
    merged["rank_metric"] = args.rank

    needed = set()
    for col in keys:
        needed.update(merged[col].tolist())
    labels = _load_fragment_labels(args.work_dir / "fragments.parquet", needed)
    members = merged[keys].to_numpy()
    merged.insert(0, "combo_key", [
        combo_name(row, {fid: labels[fid][0] for fid in row if fid in labels}) for row in members
    ])
    merged["frag_smiles"] = [
        ".".join(labels[fid][1] for fid in row if fid in labels) for row in members
    ]
    merged["hac_total"] = [sum(labels[fid][2] for fid in row if fid in labels) for row in members]
    merged["combo_size"] = np.int8(size)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="pseudocount keeping ratios finite when a count is zero (default: %(default)s)")
    parser.add_argument("--min-binder", type=int, default=1,
                        help="binder compounds a combination needs to be kept; applied during the "
                             "merge, and the main control on output size (default: %(default)s)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="total compounds a combination needs to be kept (default: %(default)s)")
    parser.add_argument("--stratify", choices=("file", "none"), default="file",
                        help="'file' scores within each input file and pools with Mantel-Haenszel; "
                             "'none' scores against the whole library (default: %(default)s)")
    parser.add_argument("--rank", choices=sorted(RANKINGS), default=DEFAULT_RANKING,
                        help="how the result tables are ordered; stages 4 and 5 take the "
                             "first N rows, so this sets what 'top' means (default: %(default)s)")
    parser.add_argument("--top", type=int, default=1500, help="rows written to the top CSV per size")
    args = parser.parse_args()

    stage1 = read_summary(args.work_dir / "summary_fragment.json")
    stage2 = read_summary(args.work_dir / "summary_combine.json")
    stratify = args.stratify == "file"
    started = time.time()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rank": args.rank,
        "min_hac": stage1["min_hac"],
        "n_binders": stage1["n_binders"],
        "n_nonbinders": stage1["n_nonbinders"],
        "stratify": args.stratify,
        "n_strata": len(stage1.get("per_stratum", {})),
        "min_binder": args.min_binder,
        "min_count": args.min_count,
        "alpha": args.alpha,
        "sizes": {},
    }

    for size in stage2["sizes"]:
        files = sorted((args.work_dir / "combination_counts" / f"size{size}").glob("*.parquet"))
        if not files:
            continue
        scored = _score(size, files, args, stage1, stratify)
        if scored.empty:
            print(f"  size {size}: nothing passed the filters")
            continue
        scored.to_parquet(args.results_dir / f"combination_enrichment_size{size}.parquet", index=False)
        scored.head(args.top).to_csv(args.results_dir / f"top_combinations_size{size}.csv", index=False)
        q_col = "q_value_mh" if stratify else "q_value"
        summary["sizes"][str(size)] = {
            "n_combinations_kept": int(len(scored)),
            "n_significant_q05": int((scored[q_col] < 0.05).sum()),
        }
        print(f"  size {size}: kept {len(scored):,} combinations, "
              f"{summary['sizes'][str(size)]['n_significant_q05']:,} at q<0.05")

    frags = pd.read_parquet(args.work_dir / "fragments.parquet")
    frag_tested = (frags["n_binder"] + frags["n_nonbinder"]).to_numpy() >= args.min_count
    frags = add_enrichment(frags, stage1["n_binders"], stage1["n_nonbinders"],
                           alpha=args.alpha, tested=frag_tested)
    frags = frags.sort_values(
        rank_columns(args.rank, frags.columns), ascending=False
    ).reset_index(drop=True)
    frags["rank_metric"] = args.rank
    frags.to_parquet(args.results_dir / "fragment_enrichment.parquet", index=False)

    summary["elapsed_s"] = round(time.time() - started, 1)
    write_summary(args.results_dir / "summary_enrichment.json", summary)
    print(f"done in {summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
