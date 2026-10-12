#!/usr/bin/env python
"""Find the compounds the DEL and the catalogue actually have in common.

Looking for shared compounds by sampling both libraries and intersecting is
hopeless: two 500k samples out of 2.6 billion and 24.6 billion will not collide
even if millions of compounds are shared.  The search has to be aimed.

A compound present in both libraries fragments identically in both, so **every**
one of its fragment pairs is a shared pair.  That inverts the problem: the
candidate set is exactly the compounds carrying a known shared pair, and a
compound carrying none of them cannot be shared.  That set is small enough to
enumerate on both sides.

Two passes and an intersection:

1. over the DEL, keep every compound carrying a shared pair and key it by a hash
   of its sorted fragment-id multiset - identical molecules fragment to
   identical multisets, so identical compounds collide by construction;
2. over the catalogue, do the same and keep only compounds whose key appears in
   the DEL set;
3. resolve both sides' SMILES and confirm each match by canonical form, since a
   multiset hash can in principle be shared by two different arrangements of the
   same fragments.

    python domain/shared_compounds.py --workers 48
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from lib.combinations import combo_columns, connected_subsets  # noqa: E402
from lib.io_utils import progress  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_TASK: dict = {}


def fragset_key(frag_ids) -> int:
    """A 64-bit key for a compound's multiset of fragment ids.

    Deterministic across processes - Python's own hash() is salted per run - and
    order independent, so the same molecule keys the same way wherever it is
    met.
    """
    blob = b"".join(int(f).to_bytes(8, "little", signed=True) for f in sorted(frag_ids))
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "big", signed=True)


def _init(task: dict) -> None:
    _TASK.update(task)


def _scan(name: str) -> int:
    """Keep this file's compounds that carry a shared pair, keyed by fragment set."""
    t = _TASK
    out_path = t["parts"] / f"{name}.parquet"
    if out_path.exists():
        return -1
    path = t["frag_dir"] / f"{name}.parquet"
    if not path.exists():
        return 0
    table = pq.read_table(path, columns=["CompoundIndex", "frag_ids", "edge_src", "edge_dst"])
    index = table["CompoundIndex"].to_numpy(zero_copy_only=False)
    records = table.select(["frag_ids", "edge_src", "edge_dst"]).to_pylist()

    wanted = t["pair_set"]
    keys, compounds = [], []
    for row, rec in enumerate(records):
        ids = rec["frag_ids"]
        edges = list(zip(rec["edge_src"], rec["edge_dst"]))
        carries = False
        for i, j in connected_subsets(len(ids), edges, 2):
            a, b = (ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i])
            if (a, b) in wanted:
                carries = True
                break
        if not carries:
            continue
        keys.append(fragset_key(ids))
        compounds.append(index[row])

    allowed = t["filter_keys"]
    if allowed is not None and keys:
        # Only fragment sets the DEL already has can ever match, so the
        # catalogue side is filtered before anything is written.
        found = np.asarray(keys, dtype=np.int64)
        position = np.searchsorted(allowed, found)
        position[position >= len(allowed)] = 0
        hit = allowed[position] == found
        keys = found[hit].tolist()
        compounds = [c for c, k in zip(compounds, hit) if k]

    pd.DataFrame({"key": np.array(keys, dtype=np.int64), "CompoundIndex": compounds}).to_parquet(
        out_path, index=False)
    return len(keys)


def shared_pairs(catalogue_work: Path, extra_work: Path | None, size: int) -> set:
    """Pairs the catalogue was searched for and found - known present in both."""
    keys = combo_columns(size)
    found: set = set()
    for work in [catalogue_work, extra_work]:
        directory = work / "combination_counts" / f"size{size}" if work else None
        if directory is None or not directory.exists():
            continue
        table = ds.dataset(directory).to_table(columns=keys).to_pandas().drop_duplicates()
        found.update(zip(table[keys[0]].astype("int64"), table[keys[1]].astype("int64")))
    return found


def run_pass(label: str, frag_dir: Path, parts: Path, pair_set: set, workers: int,
             filter_keys: np.ndarray | None = None) -> int:
    parts.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in frag_dir.glob("*.parquet"))
    task = {"frag_dir": frag_dir, "parts": parts, "pair_set": pair_set, "filter_keys": filter_keys}
    total = 0
    with Pool(workers, initializer=_init, initargs=(task,)) as pool:
        for n in progress(pool.imap_unordered(_scan, names), len(names), label):
            total += max(n, 0)
    return total


def smiles_for(compounds: set, input_dir: Path, workers: int) -> dict:
    """Fetch specific compounds' SMILES, scanning the inputs once."""
    global _TASK
    names = sorted(p.stem for p in input_dir.glob("*.parquet"))
    task = {"input_dir": input_dir, "wanted": compounds}
    found: dict = {}
    with Pool(workers, initializer=_init, initargs=(task,)) as pool:
        for part in progress(pool.imap_unordered(_lookup, names), len(names), "smiles"):
            found.update(part)
    return found


def _lookup(name: str) -> dict:
    t = _TASK
    path = t["input_dir"] / f"{name}.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path, columns=["CompoundIndex", "Smiles"])
    keep = pc.is_in(table["CompoundIndex"],
                    value_set=pa.array(sorted(t["wanted"]),
                                       table.schema.field("CompoundIndex").type))
    filtered = table.filter(keep)
    return dict(zip(filtered["CompoundIndex"].to_pylist(), filtered["Smiles"].to_pylist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-work", type=Path, default=PROJECT / "work_full")
    parser.add_argument("--catalogue-work", type=Path, default=PROJECT / "work_catalogue")
    parser.add_argument("--extra-work", type=Path, default=PROJECT / "work_catalogue_overlap",
                        help="a second catalogue run whose found pairs widen the shared set")
    parser.add_argument("--del-input", type=Path, default=PROJECT / "data/raw_HGODEL_labeled/sharded")
    parser.add_argument("--catalogue-input", type=Path, default=PROJECT / "data/Enamine_ll/sharded")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results_domain")
    parser.add_argument("--scratch", type=Path, default=PROJECT / "results_domain" / "_shared_parts")
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    pairs = shared_pairs(args.catalogue_work, args.extra_work, args.size)
    print(f"shared pairs known to exist in both libraries: {len(pairs):,}", flush=True)

    del_parts = args.scratch / "del"
    n_del = run_pass("DEL scan", args.del_work / "compound_fragments", del_parts, pairs, args.workers)
    unique_del = np.unique(np.concatenate([
        pd.read_parquet(p, columns=["key"])["key"].to_numpy()
        for p in sorted(del_parts.glob("*.parquet"))]))
    print(f"DEL compounds carrying a shared pair: {n_del:,} "
          f"({len(unique_del):,} distinct fragment sets)", flush=True)

    cat_parts = args.scratch / "catalogue"
    n_cat = run_pass("catalogue scan", args.catalogue_work / "compound_fragments", cat_parts,
                     pairs, args.workers, filter_keys=unique_del)
    cat_rows = pd.concat(
        (pd.read_parquet(p) for p in sorted(cat_parts.glob("*.parquet"))), ignore_index=True)
    print(f"catalogue compounds matching a DEL fragment set: {len(cat_rows):,}", flush=True)

    out = {"shared_pairs_used": len(pairs),
           "del_compounds_carrying_a_shared_pair": int(n_del),
           "del_distinct_fragment_sets": int(len(unique_del)),
           "catalogue_candidates": int(len(cat_rows))}

    if len(cat_rows):
        # The DEL side can run to hundreds of millions of rows, so it is filtered
        # a file at a time against the (small) set of keys the catalogue matched,
        # rather than concatenated first.
        matched = set(cat_rows["key"].tolist())
        del_rows = pd.concat(
            (frame[frame["key"].isin(matched)]
             for frame in (pd.read_parquet(p) for p in sorted(del_parts.glob("*.parquet")))),
            ignore_index=True)
        print(f"resolving SMILES for {len(del_rows):,} DEL and {len(cat_rows):,} catalogue "
              f"candidates", flush=True)
        del_smiles = smiles_for(set(del_rows["CompoundIndex"]), args.del_input, args.workers)
        cat_smiles = smiles_for(set(cat_rows["CompoundIndex"]), args.catalogue_input, args.workers)

        def canon(smiles):
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            return Chem.MolToSmiles(mol) if mol else None

        del_rows["canonical"] = [canon(del_smiles.get(c)) for c in del_rows["CompoundIndex"]]
        cat_rows["canonical"] = [canon(cat_smiles.get(c)) for c in cat_rows["CompoundIndex"]]
        merged = del_rows.merge(cat_rows, on="canonical", suffixes=("_del", "_catalogue"))
        merged = merged[merged["canonical"].notna()].drop_duplicates(
            ["CompoundIndex_del", "CompoundIndex_catalogue"])
        merged = merged[["canonical", "CompoundIndex_del", "CompoundIndex_catalogue"]]
        merged.to_csv(args.out_dir / "shared_compounds.csv", index=False)
        out["confirmed_shared_compounds"] = int(merged["canonical"].nunique())
        out["confirmed_pairs_of_ids"] = int(len(merged))
        out["fragment_set_collisions_not_confirmed"] = int(len(cat_rows) - len(merged))
        print(f"confirmed shared compounds: {out['confirmed_shared_compounds']:,}", flush=True)

    out["elapsed_s"] = round(time.time() - start, 1)
    (args.out_dir / "shared_compounds.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not args.keep_parts:
        shutil.rmtree(args.scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
