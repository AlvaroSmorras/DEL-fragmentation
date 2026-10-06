#!/usr/bin/env python
"""Stage 7 - which DEL-enriched fragments and combinations exist in a catalogue.

Joins a labelled run's enrichment tables (``results_full/``) against an
unlabelled catalogue run (``work_catalogue/``).  Fragment ids are content
hashes of the canonical fragment SMILES, so a DEL fragment and a catalogue
fragment are the same thing exactly when their ids match - including the BRICS
attachment labels.  No re-fragmentation is needed; the join is on ids.

Outputs, under ``--out-dir``:

* ``fragment_in_catalogue.parquet`` / ``combination_in_catalogue.parquet`` -
  every scored DEL fragment / combination with ``n_catalogue`` (catalogue
  compounds carrying it) and an ``enriched`` flag.
* ``enriched_*_in_catalogue.csv`` - the enriched subset, sorted by the ranking.
* ``enamine_compounds_with_enriched_fragment.csv`` /
  ``enamine_compounds_with_enriched_combination.csv`` - one row per catalogue
  compound and unit: compound id, compound SMILES, the unit's name/key and its
  SMILES, plus the DEL statistics behind it.
* ``catalogue_compounds_enriched_combinations.parquet`` - **every** catalogue
  compound carrying an enriched combination, with its SMILES.
* ``catalogue_compounds_enriched_fragments.parquet`` - a capped sample
  (``--max-per-fragment``) of catalogue compounds carrying each enriched
  fragment.  Enriched single fragments are often small and common (one sits in
  >100M catalogue compounds), so listing them all is neither cheap nor useful;
  the exact counts are in ``fragment_in_catalogue.parquet``.

The compound pass reads one catalogue file at a time - its stage 1 fragments,
its stage 2 target combinations and its input SMILES - so it parallelises over
files and can be resumed (``parts/`` keeps finished files).  Skip it with
``--no-compounds`` to get the count tables in about a minute.

``--rank`` picks what counts as enriched:

* ``mh`` (the default) - the stratified Mantel-Haenszel test: ``q < --alpha``
  and ``enrichment_mh_lo95 > --min-lo95``.  Single fragments fall back to the
  pooled columns, which are confounded by sub-library.
* ``ratio`` - raw binders per non-binder: ``binder_per_nonbinder >=
  --min-ratio`` with at least ``--min-binder`` binders.  This is the one to use
  for prospective transfer to a catalogue, and it needs a run scored by
  ``scripts/03b_enrich_by_ratio.py`` (e.g. ``results_binder_ratio/``), which is
  where the ``binder_per_nonbinder`` column comes from.  The library's own base
  rate is about 0.0005, so the default 0.01 is a 20x enrichment.

Combinations are only present in the catalogue run for the target list it was
built with (``scripts/build_target_combos.py``, every DEL pair with >= 1
binder); every scored combination is in that list, so nothing is missed.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.combinations import combo_columns  # noqa: E402
from lib.io_utils import add_project_root_to_path, progress, read_summary  # noqa: E402

add_project_root_to_path()

_TASK: dict = {}


def _enriched(frame: pd.DataFrame, args, q_col: str, lo_col: str) -> pd.Series:
    """Which rows count as enriched, under the chosen ranking."""
    if args.rank == "ratio":
        if "binder_per_nonbinder" not in frame.columns:
            raise SystemExit(
                "--rank ratio needs a 'binder_per_nonbinder' column; score the run with "
                "scripts/03b_enrich_by_ratio.py and point --del-results at its output"
            )
        return (frame["n_binder"] >= args.min_binder) & (frame["binder_per_nonbinder"] >= args.min_ratio)
    return (frame[q_col] < args.alpha) & (frame[lo_col] > args.min_lo95)


def _sort_key(args, lo_col: str) -> str:
    return "binder_per_nonbinder" if args.rank == "ratio" else lo_col


def _catalogue_combo_counts(work_dir: Path, size: int) -> pd.DataFrame:
    keys = combo_columns(size)
    table = ds.dataset(work_dir / "combination_counts" / f"size{size}").to_table(
        columns=keys + ["n_compounds", "example_compound"]
    ).to_pandas()
    grouped = table.groupby(keys, sort=False)
    out = grouped["n_compounds"].sum().rename("n_catalogue").to_frame()
    out["catalogue_example"] = grouped["example_compound"].first()
    return out.reset_index()


def _init(task: dict) -> None:
    _TASK.update(task)


def _process_file(name: str) -> tuple[str, int, int]:
    """Collect one catalogue file's hits and attach their SMILES."""
    t = _TASK
    parts = t["parts_dir"]
    combo_out = parts / f"{name}.combo.parquet"
    frag_out = parts / f"{name}.frag.parquet"
    if combo_out.exists() and frag_out.exists():
        return name, -1, -1

    keys = t["keys"]
    # Combinations: the stage 2 long table already lists matching compounds.
    combos = pd.read_parquet(t["combo_dir"] / f"{name}.parquet", columns=keys + ["CompoundIndex"])
    combos = combos.merge(t["combo_targets"], on=keys, how="inner")

    # Fragments: explode the per-compound id lists inside Arrow, keep the
    # enriched ids, and cap each fragment's hits for this file.
    table = pq.read_table(t["frag_dir"] / f"{name}.parquet", columns=["CompoundIndex", "frag_ids"])
    flat = pc.list_flatten(table["frag_ids"])
    parent = pc.list_parent_indices(table["frag_ids"])
    mask = pc.is_in(flat, value_set=t["frag_set"])
    hit_ids = pc.filter(flat, mask).to_numpy()
    hit_rows = pc.filter(parent, mask).to_numpy()
    frags = pd.DataFrame({"frag_id": hit_ids, "row": hit_rows}).drop_duplicates()
    if t["per_file_cap"]:
        frags = frags.groupby("frag_id", sort=False).head(t["per_file_cap"])
    frags["CompoundIndex"] = table["CompoundIndex"].take(pa.array(frags["row"].to_numpy())).to_numpy(
        zero_copy_only=False
    )
    frags = frags.drop(columns="row")
    frags["source_file"] = name

    wanted = set(combos["CompoundIndex"]).union(frags["CompoundIndex"])
    if wanted:
        smi = pq.read_table(t["input_dir"] / f"{name}.parquet", columns=["CompoundIndex", "Smiles"])
        keep = pc.is_in(smi["CompoundIndex"], value_set=pa.array(sorted(wanted), smi.schema.field("CompoundIndex").type))
        smi = smi.filter(keep).to_pandas().drop_duplicates("CompoundIndex")
        combos = combos.merge(smi, on="CompoundIndex", how="left")
        frags = frags.merge(smi, on="CompoundIndex", how="left")
    else:
        combos["Smiles"] = pd.Series(dtype=str)
        frags["Smiles"] = pd.Series(dtype=str)
    combos["source_file"] = name

    # Write the fragment part last: its presence marks the file as done.
    combos.to_parquet(combo_out, index=False)
    frags.to_parquet(frag_out, index=False)
    return name, len(combos), len(frags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-results", type=Path, default=Path("results_full"))
    parser.add_argument("--catalogue-work", type=Path, default=Path("work_catalogue"))
    parser.add_argument("--catalogue-input", type=Path, default=Path("data/Enamine_ll/sharded"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_full/catalogue"))
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--rank", choices=("mh", "ratio"), default="mh",
                        help="what counts as enriched (default: %(default)s)")
    parser.add_argument("--alpha", type=float, default=0.05, help="--rank mh: q-value cut")
    parser.add_argument("--min-ratio", type=float, default=0.01,
                        help="--rank ratio: binders per non-binder (default: %(default)s, "
                             "about 20x the library base rate)")
    parser.add_argument("--min-binder", type=int, default=5,
                        help="--rank ratio: binder compounds required (default: %(default)s)")
    parser.add_argument("--min-lo95", type=float, default=1.0,
                        help="--rank mh: 95%% lower bound the enrichment must exceed")
    parser.add_argument("--max-per-fragment", type=int, default=1000,
                        help="catalogue compounds kept per enriched fragment; 0 keeps every one "
                             "(default: %(default)s)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-compounds", action="store_true", help="count tables only")
    parser.add_argument("--limit-files", type=int, default=0, help="only the first N files (testing)")
    args = parser.parse_args()
    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = combo_columns(args.size)

    # ---- fragments -------------------------------------------------------
    frag = pd.read_parquet(args.del_results / "fragment_enrichment.parquet")
    cat_frag = pq.read_table(args.catalogue_work / "fragments.parquet",
                             columns=["frag_id", "frag_name", "n_compounds"]).to_pandas()
    cat_frag = cat_frag.rename(columns={"frag_name": "catalogue_frag_name", "n_compounds": "n_catalogue"})
    frag = frag.merge(cat_frag, on="frag_id", how="left")
    frag["n_catalogue"] = frag["n_catalogue"].fillna(0).astype("int64")
    frag["in_catalogue"] = frag["n_catalogue"] > 0
    # Single fragments only carry pooled statistics (see README, section 3).
    frag_enriched = _enriched(frag, args, "q_value", "enrichment_factor_lo95")
    # The pooled test is only defined where stage 3 tested the fragment at all.
    frag["enriched"] = frag_enriched if args.rank == "ratio" else (frag["tested"] & frag_enriched)
    frag.to_parquet(args.out_dir / "fragment_in_catalogue.parquet", index=False)
    enr_frag = frag[frag["enriched"]].sort_values(_sort_key(args, "enrichment_factor_lo95"), ascending=False)
    frag_cols = [c for c in ("frag_name", "frag_smiles", "hac", "n_binder", "n_nonbinder",
                             "binder_per_nonbinder", "enrichment_factor", "enrichment_factor_lo95",
                             "q_value", "in_catalogue", "n_catalogue", "catalogue_frag_name")
                 if c in enr_frag.columns]
    enr_frag[frag_cols].to_csv(args.out_dir / "enriched_fragments_in_catalogue.csv", index=False)

    # ---- combinations ----------------------------------------------------
    combo = pd.read_parquet(args.del_results / f"combination_enrichment_size{args.size}.parquet")
    combo = combo.merge(_catalogue_combo_counts(args.catalogue_work, args.size), on=keys, how="left")
    combo["n_catalogue"] = combo["n_catalogue"].fillna(0).astype("int64")
    combo["in_catalogue"] = combo["n_catalogue"] > 0
    combo["enriched"] = _enriched(combo, args, "q_value_mh", "enrichment_mh_lo95")
    rendered = args.del_results / f"top_combinations_smiles_size{args.size}.csv"
    if rendered.exists():
        smiles = pd.read_csv(rendered, usecols=["combo_key", "combination_smiles"])
        combo = combo.merge(smiles, on="combo_key", how="left")
    combo.to_parquet(args.out_dir / "combination_in_catalogue.parquet", index=False)
    enr_combo = combo[combo["enriched"]].sort_values(_sort_key(args, "enrichment_mh_lo95"), ascending=False)
    cols = [c for c in ("combo_key", "combination_smiles", "frag_smiles", "n_binder", "n_nonbinder",
                        "n_strata", "binder_per_nonbinder", "enrichment_mh", "enrichment_mh_lo95",
                        "q_value_mh", "in_catalogue", "n_catalogue", "catalogue_example")
            if c in enr_combo.columns]
    enr_combo[cols].to_csv(args.out_dir / "enriched_combinations_in_catalogue.csv", index=False)

    summary = {
        "rank": args.rank, "alpha": args.alpha, "min_lo95": args.min_lo95,
        "min_ratio": args.min_ratio, "min_binder": args.min_binder,
        "del_results": str(args.del_results),
        "fragments_scored": len(frag),
        "fragments_in_catalogue": int(frag["in_catalogue"].sum()),
        "fragments_enriched": int(frag["enriched"].sum()),
        "fragments_enriched_in_catalogue": int(enr_frag["in_catalogue"].sum()),
        "combinations_scored": len(combo),
        "combinations_in_catalogue": int(combo["in_catalogue"].sum()),
        "combinations_enriched": int(combo["enriched"].sum()),
        "combinations_enriched_in_catalogue": int(enr_combo["in_catalogue"].sum()),
        "catalogue_compounds_with_enriched_combination_rows": int(enr_combo["n_catalogue"].sum()),
    }
    print(json.dumps(summary, indent=2))

    if not args.no_compounds:
        catalogue_compounds(args, keys, enr_frag, enr_combo, summary)

    summary["elapsed_s"] = round(time.time() - start, 1)
    (args.out_dir / "summary_catalogue_hits.json").write_text(json.dumps(summary, indent=2) + "\n")


def _write_compound_table(hits: pd.DataFrame, stem: Path, unit_id: str, unit_smiles: str,
                          unit_kind: str) -> None:
    """One row per (catalogue compound, enriched unit), with both SMILES.

    This is the table to hand to someone picking compounds to buy, so it leads
    with the compound and names the unit it carries; the DEL statistics that
    made the unit interesting follow.  A combination's SMILES is its two
    fragments separated by a dot, as stage 3 writes them.
    """
    out = hits.rename(columns={
        "CompoundIndex": "enamine_compound_id",
        "Smiles": "enamine_smiles",
        unit_id: f"{unit_kind}_id",
        unit_smiles: f"{unit_kind}_smiles",
        "n_binder": "del_n_binder",
        "n_nonbinder": "del_n_nonbinder",
        "binder_per_nonbinder": "del_binder_per_nonbinder",
        "n_catalogue": f"n_enamine_compounds_with_{unit_kind}",
    })
    lead = ["enamine_compound_id", "enamine_smiles", f"{unit_kind}_id", f"{unit_kind}_smiles"]
    stats = [c for c in ("del_n_binder", "del_n_nonbinder", "del_binder_per_nonbinder",
                         "enrichment_factor", "enrichment_factor_lo95", "enrichment_mh",
                         "enrichment_mh_lo95", "q_value", "q_value_mh",
                         f"n_enamine_compounds_with_{unit_kind}") if c in out.columns]
    out = out[lead + stats]
    out.to_parquet(stem.with_suffix(".parquet"), index=False)
    # A multi-GB CSV helps nobody; the parquet beside it always holds every row.
    if len(out) <= 5_000_000:
        out.to_csv(stem.with_suffix(".csv"), index=False)
        print(f"wrote {stem.with_suffix('.csv')} ({len(out):,} rows)")
    else:
        print(f"wrote {stem.with_suffix('.parquet')} ({len(out):,} rows; too large for CSV)")


def catalogue_compounds(args, keys, enr_frag, enr_combo, summary) -> None:
    parts_dir = args.out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    combo_dir = args.catalogue_work / "combinations" / f"size{args.size}"
    names = sorted(p.stem for p in combo_dir.glob("*.parquet"))
    if args.limit_files:
        names = names[: args.limit_files]
    stage1 = read_summary(args.catalogue_work / "summary_fragment.json")
    print(f"compound pass over {len(names):,} catalogue files "
          f"({stage1.get('n_compounds', 0):,} compound rows)")

    targets = enr_combo[enr_combo["in_catalogue"]][keys + ["combo_key"]].reset_index(drop=True)
    frag_ids = enr_frag.loc[enr_frag["in_catalogue"], "frag_id"].to_numpy(dtype="int64")
    task = {
        "parts_dir": parts_dir, "keys": keys, "combo_dir": combo_dir,
        "frag_dir": args.catalogue_work / "compound_fragments",
        "input_dir": args.catalogue_input,
        "combo_targets": targets, "frag_set": pa.array(frag_ids, pa.int64()),
        # Each file keeps a few hits per fragment, so the global cap is filled
        # from many files rather than the first ones scanned.
        "per_file_cap": 0 if args.max_per_fragment == 0
                        else max(1, -(-3 * args.max_per_fragment // len(names))),
    }
    with Pool(args.workers, initializer=_init, initargs=(task,)) as pool:
        for _ in progress(pool.imap_unordered(_process_file, names), len(names), "compounds"):
            pass

    combo_cols = ["combo_key", "CompoundIndex", "Smiles", "source_file"]
    combo_hits = pd.concat(
        (pd.read_parquet(p, columns=combo_cols) for p in sorted(parts_dir.glob("*.combo.parquet"))),
        ignore_index=True,
    )
    stat_cols = [c for c in ("combo_key", "frag_smiles", "n_binder", "n_nonbinder",
                             "binder_per_nonbinder", "enrichment_mh", "enrichment_mh_lo95",
                             "q_value_mh", "n_catalogue") if c in enr_combo.columns]
    stats = enr_combo[stat_cols]
    combo_hits = combo_hits.merge(stats, on="combo_key", how="left")
    combo_hits.to_parquet(args.out_dir / "catalogue_compounds_enriched_combinations.parquet", index=False)
    _write_compound_table(
        combo_hits, args.out_dir / "enamine_compounds_with_enriched_combination",
        unit_id="combo_key", unit_smiles="frag_smiles", unit_kind="combination",
    )

    frag_hits = pd.concat(
        (pd.read_parquet(p) for p in sorted(parts_dir.glob("*.frag.parquet"))), ignore_index=True
    )
    if args.max_per_fragment:
        frag_hits = frag_hits.sample(frac=1.0, random_state=0).groupby("frag_id", sort=False).head(
            args.max_per_fragment
        )
    frag_hits = frag_hits.merge(
        enr_frag[[c for c in ("frag_id", "frag_name", "frag_smiles", "n_binder", "n_nonbinder",
                              "binder_per_nonbinder", "enrichment_factor", "enrichment_factor_lo95",
                              "q_value", "n_catalogue") if c in enr_frag.columns]],
        on="frag_id", how="left",
    ).sort_values([_sort_key(args, "enrichment_factor_lo95"), "frag_id"], ascending=[False, True])
    frag_hits.to_parquet(args.out_dir / "catalogue_compounds_enriched_fragments.parquet", index=False)
    _write_compound_table(
        frag_hits, args.out_dir / "enamine_compounds_with_enriched_fragment",
        unit_id="frag_name", unit_smiles="frag_smiles", unit_kind="fragment",
    )

    summary["catalogue_compound_rows_enriched_combinations"] = len(combo_hits)
    summary["catalogue_distinct_compounds_enriched_combinations"] = int(combo_hits["CompoundIndex"].nunique())
    summary["catalogue_compound_rows_enriched_fragments_sampled"] = len(frag_hits)
    summary["missing_smiles"] = int(combo_hits["Smiles"].isna().sum() + frag_hits["Smiles"].isna().sum())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
