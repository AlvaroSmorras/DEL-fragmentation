#!/usr/bin/env python
"""How much of the DEL's fragment and pair vocabulary the catalogue already has.

Separate from both the scoring pipeline and ``compare_domains.py``: this one is
about plain overlap, with no reference to enrichment.  Two counts are reported
for everything, because they answer different questions:

* **by type** - what share of *distinct* fragments (or pairs) is shared.  This
  is the vocabulary question, and it is dominated by the rare tail.
* **by instance** - what share of *compound-fragment occurrences* a shared
  fragment accounts for.  A handful of common fragments sit in most compounds,
  so this is the number that says how much of the library is reachable.

Fragments are exact and complete: both runs keep a full dictionary keyed by the
same content hash, so the join needs no sampling.

Pairs cannot be complete from what is on disk.  The catalogue's stage 2 ran
against a target list - every DEL pair a binder carries - so pairs outside that
list were never written, and the honest statement is a figure for that subset
plus the reason the rest is unknown.  ``--pair-sample`` writes a uniform sample
of *all* distinct DEL pairs as a target list for a fresh catalogue pass, which
is what makes the unbiased number obtainable; see ``domain/README.md``.

    python domain/overlap_stats.py --out-dir results_domain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from lib.combinations import combo_columns  # noqa: E402


def fragment_overlap(del_work: Path, cat_work: Path) -> dict:
    """Exact fragment overlap, by type and weighted by how often each is used."""
    dele = pq.read_table(del_work / "fragments.parquet",
                         columns=["frag_id", "n_compounds", "n_binder", "hac"]).to_pandas()
    cat = pq.read_table(cat_work / "fragments.parquet",
                        columns=["frag_id", "n_compounds"]).to_pandas()
    shared = np.isin(dele["frag_id"].to_numpy(), cat["frag_id"].to_numpy())
    dele["shared"] = shared

    instances = dele["n_compounds"].sum()
    binder_instances = dele["n_binder"].sum()
    out = {
        "del_fragment_types": int(len(dele)),
        "catalogue_fragment_types": int(len(cat)),
        "shared_types": int(shared.sum()),
        "shared_types_pct_of_del": round(100 * shared.mean(), 2),
        "shared_types_pct_of_catalogue": round(100 * shared.sum() / len(cat), 3),
        "del_fragment_instances": int(instances),
        "shared_instances_pct": round(100 * dele.loc[shared, "n_compounds"].sum() / instances, 2),
        "del_binder_instances": int(binder_instances),
        "shared_binder_instances_pct": round(
            100 * dele.loc[shared, "n_binder"].sum() / max(binder_instances, 1), 2),
    }
    # Shared fragments are the common ones; splitting by how many DEL compounds
    # carry a fragment shows how strongly that holds.
    bands = [(1, 10), (10, 100), (100, 1_000), (1_000, 10_000), (10_000, 10**12)]
    out["by_del_frequency"] = {
        f"{lo}-{hi if hi < 10**12 else 'inf'}": {
            "types": int(sel.sum()),
            "shared_pct": round(100 * dele.loc[sel, "shared"].mean(), 2) if sel.sum() else None,
        }
        for lo, hi in bands
        for sel in [(dele["n_compounds"] >= lo) & (dele["n_compounds"] < hi)]
    }
    out["by_fragment_size"] = {
        f"{lo}-{hi} heavy atoms": {
            "types": int(sel.sum()),
            "shared_pct": round(100 * dele.loc[sel, "shared"].mean(), 2) if sel.sum() else None,
        }
        for lo, hi in [(0, 8), (8, 11), (11, 14), (14, 18), (18, 999)]
        for sel in [(dele["hac"] >= lo) & (dele["hac"] < hi)]
    }
    return out


def pair_overlap(del_work: Path, cat_work: Path, size: int, targets: Path | None) -> dict:
    """Pair overlap over the subset the catalogue run actually searched for."""
    keys = combo_columns(size)
    found = ds.dataset(cat_work / "combination_counts" / f"size{size}").to_table(
        columns=keys + ["n_compounds"]).to_pandas()
    found = found.groupby(keys, sort=False)["n_compounds"].sum().reset_index()

    out = {"catalogue_pairs_found": int(len(found)),
           "catalogue_compound_pair_rows": int(found["n_compounds"].sum())}
    if targets is not None and targets.exists():
        searched = pq.read_table(targets, columns=keys).to_pandas()
        out["pairs_searched_for"] = int(len(searched))
        out["shared_pct_of_searched"] = round(100 * len(found) / max(len(searched), 1), 2)
        out["basis"] = (f"{len(searched):,} DEL pairs carried by at least one binder - the target "
                        f"list the catalogue was searched with. Pairs outside it were never "
                        f"written, so this is not the overlap over all DEL pairs.")

    # What the whole DEL pair vocabulary looks like, for scale.
    counts = ds.dataset(del_work / "combination_counts" / f"size{size}")
    out["del_pair_rows_before_dedup"] = int(counts.count_rows())
    return out


def sample_del_pairs(del_work: Path, size: int, fraction: int, out_path: Path) -> dict:
    """A uniform sample of distinct DEL pairs, as a target list for a new pass.

    Sampling by a hash of the pair rather than by row keeps it uniform over
    *types* and needs no global de-duplication first: the same pair hashes the
    same way in every file, so it is either in the sample everywhere or nowhere.
    """
    keys = combo_columns(size)
    files = sorted((del_work / "combination_counts" / f"size{size}").glob("*.parquet"))
    kept = []
    for path in files:
        table = pq.read_table(path, columns=keys)
        a = table[keys[0]].to_numpy().astype(np.uint64)
        b = table[keys[1]].to_numpy().astype(np.uint64)
        # Cheap 64-bit mix, so the low bits used for the modulo are not the
        # fragment hash's own low bits.
        mixed = (a * np.uint64(0x9E3779B97F4A7C15)) ^ (b + np.uint64(0x165667B19E3779F9))
        mixed ^= mixed >> np.uint64(31)
        sel = (mixed % np.uint64(fraction)) == 0
        if sel.any():
            kept.append(pd.DataFrame({keys[0]: table[keys[0]].to_numpy()[sel],
                                      keys[1]: table[keys[1]].to_numpy()[sel]}))
    sample = pd.concat(kept, ignore_index=True).drop_duplicates() if kept else pd.DataFrame(columns=keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.astype("int64").to_parquet(out_path, index=False)
    return {"sampled_distinct_del_pairs": int(len(sample)), "one_in": fraction,
            "estimated_distinct_del_pairs": int(len(sample) * fraction), "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-work", type=Path, default=PROJECT / "work_full")
    parser.add_argument("--catalogue-work", type=Path, default=PROJECT / "work_catalogue")
    parser.add_argument("--targets", type=Path, default=PROJECT / "targets.parquet")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results_domain")
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--pair-sample", type=int, default=0,
                        help="write a 1-in-N uniform sample of distinct DEL pairs as a target "
                             "list, for an unbiased catalogue pass (0 skips it)")
    parser.add_argument("--pair-sample-out", type=Path,
                        default=PROJECT / "results_domain" / "del_pair_sample.parquet")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = {"fragments": fragment_overlap(args.del_work, args.catalogue_work)}
    print(json.dumps(out["fragments"], indent=2), flush=True)
    out["pairs"] = pair_overlap(args.del_work, args.catalogue_work, args.size, args.targets)
    print(json.dumps(out["pairs"], indent=2), flush=True)
    if args.pair_sample:
        out["pair_sample"] = sample_del_pairs(args.del_work, args.size, args.pair_sample,
                                              args.pair_sample_out)
        print(json.dumps(out["pair_sample"], indent=2), flush=True)

    path = args.out_dir / "overlap_stats.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
