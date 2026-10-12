#!/usr/bin/env python
"""When a fragment or pair is shared, are the compounds carrying it shared too?

Overlap of the fragment vocabulary is one thing; whether that vocabulary is
assembled the same way is another, and it is the one that decides whether a
catalogue hit is a real analogue of a DEL hit or merely a molecule that happens
to contain the same ring.

For a stratified sample of fragments (and pairs) present in **both** libraries,
this collects the compounds carrying them on each side and asks:

* are any of the compounds themselves identical, by canonical SMILES;
* how similar is a DEL parent to the closest Enamine parent of the same
  fragment, against the baseline of two compounds drawn at random from the two
  libraries - which ``compare_domains.py`` measured at a median Tanimoto of
  about 0.40.

If sharing a fragment told us nothing about the rest of the molecule, the two
numbers would coincide.  How far the conditioned number sits above the baseline
is how much a shared fragment actually buys.

    python domain/parent_overlap.py --n-fragments 200 --n-pairs 200
"""

from __future__ import annotations

import argparse
import json
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
from rdkit.Chem import rdFingerprintGenerator

import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from lib.combinations import combo_columns, connected_subsets  # noqa: E402
from lib.io_utils import progress  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_TASK: dict = {}
_GEN = None


# ------------------------------------------------------------------ targets --

def choose_targets(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared fragments, stratified by how common they are, and shared pairs."""
    dele = pq.read_table(args.del_work / "fragments.parquet",
                         columns=["frag_id", "frag_smiles", "n_compounds", "hac"]).to_pandas()
    cat = pq.read_table(args.catalogue_work / "fragments.parquet",
                        columns=["frag_id", "n_compounds"]).to_pandas()
    cat = cat.rename(columns={"n_compounds": "n_catalogue"})
    shared = dele.merge(cat, on="frag_id", how="inner")
    rng = np.random.default_rng(args.seed)
    bands = [(1, 100), (100, 1_000), (1_000, 10_000), (10_000, 10**12)]
    picked = []
    per_band = max(1, args.n_fragments // len(bands))
    for lo, hi in bands:
        sel = shared[(shared["n_compounds"] >= lo) & (shared["n_compounds"] < hi)]
        if len(sel):
            take = rng.choice(len(sel), size=min(per_band, len(sel)), replace=False)
            picked.append(sel.iloc[np.sort(take)])
    fragments = pd.concat(picked, ignore_index=True)

    keys = combo_columns(args.size)
    pairs = ds.dataset(args.catalogue_work / "combination_counts" / f"size{args.size}").to_table(
        columns=keys + ["n_compounds"]).to_pandas()
    pairs = pairs.groupby(keys, sort=False)["n_compounds"].sum().reset_index()
    take = rng.choice(len(pairs), size=min(args.n_pairs, len(pairs)), replace=False)
    pairs = pairs.iloc[np.sort(take)].reset_index(drop=True)
    return fragments, pairs


# -------------------------------------------------------------- parent pass --

def _init(task: dict) -> None:
    _TASK.update(task)


def _collect(name: str):
    """Compounds in one file carrying a target fragment or target pair.

    Pairs come from a prebuilt long table when the run has one - the catalogue's
    stage 2 wrote exactly these pairs - and are enumerated from the fragment
    graph only when it does not, which is the case for the DEL.
    """
    t = _TASK
    frag_path = t["frag_dir"] / f"{name}.parquet"
    if not frag_path.exists():
        return [], []
    columns = ["CompoundIndex", "frag_ids"]
    if t["pair_set"] and t["pair_table_dir"] is None:
        columns += ["edge_src", "edge_dst"]
    table = pq.read_table(frag_path, columns=columns)
    index = table["CompoundIndex"].to_numpy(zero_copy_only=False)

    flat = pc.list_flatten(table["frag_ids"])
    parent = pc.list_parent_indices(table["frag_ids"])
    hit = pc.is_in(flat, value_set=t["frag_set"])
    frag_rows = pd.DataFrame({"target": pc.filter(flat, hit).to_numpy(),
                              "row": pc.filter(parent, hit).to_numpy()}).drop_duplicates()
    frag_rows = frag_rows.groupby("target", sort=False).head(t["per_file"])
    frag_hits = [(int(r.target), index[r.row]) for r in frag_rows.itertuples()]

    pair_hits: list[tuple[int, int, str]] = []
    if t["pair_set"]:
        if t["pair_table_dir"] is not None:
            path = t["pair_table_dir"] / f"{name}.parquet"
            if path.exists():
                long = pq.read_table(path, columns=["frag_0", "frag_1", "CompoundIndex"]).to_pandas()
                keep = [(a, b) in t["pair_set"] for a, b in zip(long["frag_0"], long["frag_1"])]
                long = long[keep].groupby(["frag_0", "frag_1"], sort=False).head(t["per_file"])
                pair_hits = [(int(a), int(b), c) for a, b, c
                             in zip(long["frag_0"], long["frag_1"], long["CompoundIndex"])]
        else:
            seen: dict[tuple, int] = {}
            records = table.select(["frag_ids", "edge_src", "edge_dst"]).to_pylist()
            for row, rec in enumerate(records):
                ids = rec["frag_ids"]
                edges = list(zip(rec["edge_src"], rec["edge_dst"]))
                for i, j in connected_subsets(len(ids), edges, 2):
                    key = (ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i])
                    if key in t["pair_set"] and seen.get(key, 0) < t["per_file"]:
                        seen[key] = seen.get(key, 0) + 1
                        pair_hits.append((key[0], key[1], index[row]))

    wanted = {c for _, c in frag_hits} | {c for _, _, c in pair_hits}
    if not wanted:
        return [], []
    smiles_table = pq.read_table(t["input_dir"] / f"{name}.parquet",
                                 columns=["CompoundIndex", "Smiles"])
    keep = pc.is_in(smiles_table["CompoundIndex"],
                    value_set=pa.array(sorted(wanted), smiles_table.schema.field("CompoundIndex").type))
    filtered = smiles_table.filter(keep)
    lookup = dict(zip(filtered["CompoundIndex"].to_pylist(), filtered["Smiles"].to_pylist()))

    frag_out = [{"target": target, "smiles": lookup[c]} for target, c in frag_hits if c in lookup]
    pair_out = [{"frag_0": a, "frag_1": b, "smiles": lookup[c]}
                for a, b, c in pair_hits if c in lookup]
    return frag_out, pair_out


def gather_parents(label: str, frag_dir: Path, input_dir: Path, frag_ids, pair_set,
                   per_file: int, workers: int, per_target: int, pair_table_dir: Path | None = None):
    names = sorted(p.stem for p in frag_dir.glob("*.parquet"))
    task = {"frag_dir": frag_dir, "input_dir": input_dir,
            "frag_set": pa.array(np.asarray(frag_ids, dtype=np.int64), pa.int64()),
            "pair_set": pair_set, "per_file": per_file, "pair_table_dir": pair_table_dir}
    frags, pairs = [], []
    with Pool(workers, initializer=_init, initargs=(task,)) as pool:
        for f, p in progress(pool.imap_unordered(_collect, names), len(names), f"{label} parents"):
            frags.extend(f)
            pairs.extend(p)
    frag_df = pd.DataFrame(frags)
    pair_df = pd.DataFrame(pairs)
    if len(frag_df):
        frag_df = frag_df.groupby("target", sort=False).head(per_target).reset_index(drop=True)
    if len(pair_df):
        pair_df = pair_df.groupby(["frag_0", "frag_1"], sort=False).head(per_target).reset_index(drop=True)
    return frag_df, pair_df


# ------------------------------------------------------------------ compare --

def _fingerprint(smiles: str):
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _GEN.GetFingerprintAsNumPy(mol).astype(bool), Chem.MolToSmiles(mol)


def _features(frame: pd.DataFrame, workers: int):
    with Pool(workers) as pool:
        out = pool.map(_fingerprint, frame["smiles"].tolist(), chunksize=200)
    ok = [i for i, item in enumerate(out) if item is not None]
    fps = np.array([out[i][0] for i in ok]) if ok else np.zeros((0, 1024), dtype=bool)
    canon = [out[i][1] for i in ok]
    return frame.iloc[ok].reset_index(drop=True), fps, canon


def _max_tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Best Tanimoto from each row of a to anything in b."""
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    af, bf = a.astype(np.float32), b.astype(np.float32)
    inter = af @ bf.T
    union = af.sum(1)[:, None] + bf.sum(1)[None, :] - inter
    return (inter / np.maximum(union, 1.0)).max(axis=1)


def compare(groups_del, groups_cat, key_cols, label: str) -> dict:
    """Per shared unit: identical parents, and how close the parents get."""
    per_unit, best_all, identical_units = [], [], 0
    shared_compounds = 0
    for key, left in groups_del.items():
        right = groups_cat.get(key)
        if right is None or not len(left["fps"]) or not len(right["fps"]):
            continue
        common = set(left["canon"]) & set(right["canon"])
        best = _max_tanimoto(left["fps"], right["fps"])
        best_all.append(best)
        shared_compounds += len(common)
        identical_units += 1 if common else 0
        per_unit.append({"key": str(key), "n_del_parents": int(len(left["fps"])),
                         "n_catalogue_parents": int(len(right["fps"])),
                         "identical_parents": int(len(common)),
                         "median_best_tanimoto": round(float(np.median(best)), 4),
                         "max_best_tanimoto": round(float(best.max()), 4)})
    values = np.concatenate(best_all) if best_all else np.array([])
    return {
        "label": label,
        "units_compared": len(per_unit),
        "units_with_an_identical_parent": identical_units,
        "identical_parent_compounds": shared_compounds,
        "parent_similarity": {
            "median": round(float(np.median(values)), 4) if len(values) else None,
            "mean": round(float(values.mean()), 4) if len(values) else None,
            "frac_above_0.4": round(float((values > 0.4).mean()), 4) if len(values) else None,
            "frac_above_0.6": round(float((values > 0.6).mean()), 4) if len(values) else None,
            "frac_above_0.8": round(float((values > 0.8).mean()), 4) if len(values) else None,
            "n": int(len(values)),
            "values": [round(float(v), 4) for v in values[:20000]],
        },
        "per_unit": per_unit,
    }


def _group(frame: pd.DataFrame, fps: np.ndarray, canon: list, key_cols: list) -> dict:
    out: dict = {}
    keys = frame[key_cols[0]] if len(key_cols) == 1 else list(
        zip(*[frame[c] for c in key_cols]))
    for position, key in enumerate(keys):
        key = key if len(key_cols) > 1 else int(key)
        bucket = out.setdefault(key, {"rows": []})
        bucket["rows"].append(position)
    for key, bucket in out.items():
        rows = bucket.pop("rows")
        bucket["fps"] = fps[rows]
        bucket["canon"] = [canon[r] for r in rows]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-work", type=Path, default=PROJECT / "work_full")
    parser.add_argument("--catalogue-work", type=Path, default=PROJECT / "work_catalogue")
    parser.add_argument("--del-input", type=Path, default=PROJECT / "data/raw_HGODEL_labeled/sharded")
    parser.add_argument("--catalogue-input", type=Path, default=PROJECT / "data/Enamine_ll/sharded")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results_domain")
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--n-fragments", type=int, default=200)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--per-target", type=int, default=150,
                        help="parent compounds kept per shared unit per library")
    parser.add_argument("--per-file", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    fragments, pairs = choose_targets(args)
    pair_set = {(int(a), int(b)) for a, b in zip(pairs["frag_0"], pairs["frag_1"])}
    print(f"targets: {len(fragments):,} shared fragments, {len(pair_set):,} shared pairs", flush=True)

    out: dict = {"n_fragment_targets": len(fragments), "n_pair_targets": len(pair_set),
                 "per_target": args.per_target}
    grouped: dict = {}
    catalogue_pairs = args.catalogue_work / "combinations" / f"size{args.size}"
    for label, frag_dir, input_dir, pair_dir in [
        ("DEL", args.del_work / "compound_fragments", args.del_input, None),
        ("Enamine", args.catalogue_work / "compound_fragments", args.catalogue_input,
         catalogue_pairs if catalogue_pairs.exists() else None),
    ]:
        frag_df, pair_df = gather_parents(label, frag_dir, input_dir,
                                          fragments["frag_id"].to_numpy(), pair_set,
                                          args.per_file, args.workers, args.per_target,
                                          pair_table_dir=pair_dir)
        print(f"{label}: {len(frag_df):,} fragment parents, {len(pair_df):,} pair parents", flush=True)
        frag_df, frag_fps, frag_canon = _features(frag_df, args.workers) if len(frag_df) else (frag_df, None, None)
        pair_df, pair_fps, pair_canon = _features(pair_df, args.workers) if len(pair_df) else (pair_df, None, None)
        grouped[label] = {
            "fragments": _group(frag_df, frag_fps, frag_canon, ["target"]) if len(frag_df) else {},
            "pairs": _group(pair_df, pair_fps, pair_canon, ["frag_0", "frag_1"]) if len(pair_df) else {},
        }

    out["fragments"] = compare(grouped["DEL"]["fragments"], grouped["Enamine"]["fragments"],
                               ["target"], "compounds sharing one fragment")
    out["pairs"] = compare(grouped["DEL"]["pairs"], grouped["Enamine"]["pairs"],
                           ["frag_0", "frag_1"], "compounds sharing one fragment pair")
    out["elapsed_s"] = round(time.time() - start, 1)

    for key in ("fragments", "pairs"):
        summary = {k: v for k, v in out[key].items() if k not in ("per_unit",)}
        summary["parent_similarity"] = {k: v for k, v in summary["parent_similarity"].items()
                                        if k != "values"}
        print(json.dumps(summary, indent=2), flush=True)
    (args.out_dir / "parent_overlap.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out_dir / 'parent_overlap.json'}")


if __name__ == "__main__":
    main()
