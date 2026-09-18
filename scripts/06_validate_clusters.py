#!/usr/bin/env python
"""Stage 6 (optional) - choose how tightly to cluster, by measuring it.

Clustering is meant to denoise: variants of one scaffold pool their compounds,
so the enrichment estimate should move around less.  Whether a given setting
actually does that is an empirical question, and eyeballing clusters answers it
badly.

This splits the compounds of each stratum in half, scores every unit on each
half independently, and asks how well the two halves agree.  A setting that
genuinely denoises makes an estimate from half A predict half B better.

Comparing against the best-scoring member of a cluster would be misleading -
that member is picked after seeing the data, so its score is inflated by
selection.  Split-half agreement has no such bias.

    python scripts/06_validate_clusters.py --modes exact,substituent,murcko

Reports, per mode, agreement over all units and over just the units whose
cluster gained members, which is where clustering can possibly have helped.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.combinations import combo_columns  # noqa: E402
from lib.enrichment import MH_TERMS, add_mantel_haenszel, stratum_terms  # noqa: E402
from lib.fragmentation import fragment_id  # noqa: E402
from lib.io_utils import add_project_root_to_path, progress, read_summary  # noqa: E402
from lib.scaffolds import MODES, scaffold_keys  # noqa: E402

add_project_root_to_path()

_OPTS: dict = {}


def _init_worker(opts: dict) -> None:
    _OPTS.update(opts)


def _keys_chunk(chunk: list[str]) -> list[str]:
    return scaffold_keys(chunk, _OPTS["mode"], _OPTS["max_strip"])


def _half_of(compound_ids: pd.Series) -> np.ndarray:
    """Deterministically split compounds, so the halves are the same each run."""
    return compound_ids.map(
        lambda cid: hashlib.blake2b(cid.encode(), digest_size=4).digest()[0] & 1
    ).to_numpy()


def _half_estimates(pieces: dict[int, list[pd.DataFrame]], keys: list[str],
                    min_binder: int) -> pd.DataFrame:
    """Per-unit log enrichment from each half, for units supported in both."""
    halves = {}
    for half in (0, 1):
        if not pieces[half]:
            return pd.DataFrame()
        summed = pd.concat(pieces[half], ignore_index=True).groupby(keys, as_index=False).sum()
        halves[half] = add_mantel_haenszel(summed)[keys + ["enrichment_mh", "n_binder"]]

    both = halves[0].merge(halves[1], on=keys, suffixes=("_a", "_b"))
    both = both[(both["n_binder_a"] >= min_binder) & (both["n_binder_b"] >= min_binder)]
    a = both["enrichment_mh_a"].to_numpy()
    b = both["enrichment_mh_b"].to_numpy()
    usable = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)

    both = both.loc[usable].copy()
    both["log_a"] = np.log(a[usable])
    both["log_b"] = np.log(b[usable])
    both["mid"] = (both["log_a"] + both["log_b"]) / 2.0
    # Each half is an independent estimate, so half their squared difference
    # estimates the variance of a single-half estimate.
    both["noise_var"] = (both["log_a"] - both["log_b"]) ** 2 / 2.0
    return both


def _homogeneity(members: pd.DataFrame) -> tuple[float, int]:
    """Do the members of a cluster differ by more than measurement noise?

    Merging is only justified when members are interchangeable.  Comparing the
    spread of member estimates within a cluster against the noise on a single
    estimate gives a ratio near 1 when they are, and well above 1 when the mode
    is merging combinations that genuinely differ.
    """
    grouped = members.groupby("cluster_id")
    sizes = grouped.size()
    keep = sizes[sizes >= 2].index
    usable = members[members["cluster_id"].isin(keep)]
    if usable.empty:
        return float("nan"), 0
    within = usable.groupby("cluster_id")["mid"].var(ddof=1).dropna()
    noise = usable["noise_var"].mean()
    if not len(within) or not noise:
        return float("nan"), 0
    return float(within.mean() / noise), int(len(within))


def _agreement(pieces: dict[int, list[pd.DataFrame]], keys: list[str], min_binder: int) -> tuple:
    halves = {}
    for half in (0, 1):
        if not pieces[half]:
            return 0, float("nan"), float("nan")
        summed = pd.concat(pieces[half], ignore_index=True).groupby(keys, as_index=False).sum()
        halves[half] = add_mantel_haenszel(summed)[keys + ["enrichment_mh", "n_binder"]]

    both = halves[0].merge(halves[1], on=keys, suffixes=("_a", "_b"))
    both = both[(both["n_binder_a"] >= min_binder) & (both["n_binder_b"] >= min_binder)]
    a, b = both["enrichment_mh_a"].to_numpy(), both["enrichment_mh_b"].to_numpy()
    usable = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = np.log(a[usable]), np.log(b[usable])
    if len(a) < 10:
        return len(a), float("nan"), float("nan")
    return len(a), spearmanr(a, b).statistic, float(np.median(np.abs(a - b)))


def _evaluate(mode: str, args, stage1, dictionary: pd.DataFrame, long_files: list[Path]) -> dict:
    keys_by_frag = {}
    if mode != "exact":
        smiles = dictionary["frag_smiles"].tolist()
        chunk = max(1, len(smiles) // (args.workers * 4) + 1)
        chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=({"mode": mode, "max_strip": args.max_strip},)) as pool:
            parts = list(pool.imap(_keys_chunk, chunks))
        keys_by_frag = dict(zip(dictionary["frag_id"].tolist(),
                                [k for part in parts for k in part]))

    cols = combo_columns(args.size)
    combos = pd.read_parquet(
        args.results_dir / f"combination_enrichment_size{args.size}.parquet", columns=cols
    ).drop_duplicates()
    if mode == "exact":
        combos["cluster_id"] = [fragment_id("|".join(map(str, sorted(row))))
                                for row in combos[cols].to_numpy()]
    else:
        combos["cluster_id"] = [
            fragment_id("|".join(sorted(keys_by_frag.get(f, str(f)) for f in row)))
            for row in combos[cols].to_numpy()
        ]
    members = combos.groupby("cluster_id")[cols[0]].size().rename("n_members")
    combos = combos.merge(members, on="cluster_id")
    grouped = combos[combos["n_members"] > 1]

    pieces = {"combo": {0: [], 1: []}, "cluster": {0: [], 1: []},
              "combo_grouped": {0: [], 1: []}, "cluster_grouped": {0: [], 1: []}}
    per_stratum = stage1["per_stratum"]

    for path in progress(long_files, len(long_files), f"split[{mode}]"):
        frame = pd.read_parquet(path)
        if frame.empty or path.stem not in per_stratum:
            continue
        frame["half"] = _half_of(frame["CompoundIndex"])
        totals = frame.drop_duplicates("CompoundIndex").groupby("half")["activity"].agg(["sum", "size"])
        tagged = frame.merge(combos[cols + ["cluster_id", "n_members"]], on=cols, how="inner")

        for half in (0, 1):
            part = tagged[tagged["half"] == half]
            if part.empty or half not in totals.index:
                continue
            n_b = int(totals.loc[half, "sum"])
            n_n = int(totals.loc[half, "size"] - totals.loc[half, "sum"])
            if n_b == 0:
                continue
            for label, unit_keys, subset in (
                ("combo", cols, part),
                ("cluster", ["cluster_id"], part),
                ("combo_grouped", cols, part[part["n_members"] > 1]),
                ("cluster_grouped", ["cluster_id"], part[part["n_members"] > 1]),
            ):
                if subset.empty:
                    continue
                # A compound carrying two members of a cluster counts once.
                rows = subset.drop_duplicates(["CompoundIndex"] + unit_keys)
                counted = rows.groupby(unit_keys, as_index=False).agg(
                    n_binder=("activity", "sum"), n=("activity", "size")
                )
                counted["n_nonbinder"] = counted["n"] - counted["n_binder"]
                for name, values in stratum_terms(
                    counted["n_binder"], counted["n_nonbinder"], n_b, n_n
                ).items():
                    counted[name] = values
                pieces[label][half].append(
                    counted[unit_keys + ["n_binder", "n_nonbinder"] + list(MH_TERMS)]
                )

    result = {"mode": mode, "n_clusters": int(combos["cluster_id"].nunique()),
              "n_grouped_combinations": int(len(grouped))}
    for label, unit_keys in (("combo", cols), ("cluster", ["cluster_id"]),
                             ("combo_grouped", cols), ("cluster_grouped", ["cluster_id"])):
        n, rho, err = _agreement(pieces[label], unit_keys, args.min_binder)
        result[label] = {"n": n, "spearman": rho, "median_log_error": err}

    per_member = _half_estimates(pieces["combo"], cols, args.min_binder)
    if not per_member.empty:
        per_member = per_member.merge(combos[cols + ["cluster_id"]], on=cols, how="inner")
        ratio, n_clusters_checked = _homogeneity(per_member)
    else:
        ratio, n_clusters_checked = float("nan"), 0
    result["homogeneity"] = {"ratio": ratio, "n_clusters": n_clusters_checked}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--modes", default="exact,substituent,murcko")
    parser.add_argument("--max-strip", type=int, default=2)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--min-binder", type=int, default=3,
                        help="binders a unit needs in BOTH halves to be compared (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    stage1 = read_summary(args.work_dir / "summary_fragment.json")
    long_files = sorted((args.work_dir / "combinations" / f"size{args.size}").glob("*.parquet"))
    if not long_files:
        raise SystemExit("needs the per-compound table; re-run stage 2 without --no-long-table")
    dictionary = pd.read_parquet(args.work_dir / "fragments.parquet")[["frag_id", "frag_smiles"]]

    started = time.time()
    rows = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r}; pick from {MODES}")
        rows.append(_evaluate(mode, args, stage1, dictionary, long_files))

    print(f"\nsplit-half agreement (size {args.size}, >={args.min_binder} binders per half)")
    print(f"{'mode':<13}{'clusters':>10}{'grouped':>9}   "
          f"{'rho':>7}{'error':>8}   {'within/noise':>12}")
    baseline = None
    for row in rows:
        cell = row["cluster_grouped"]
        if baseline is None:
            baseline = row["combo_grouped"]
        print(f"{row['mode']:<13}{row['n_clusters']:>10,}{row['n_grouped_combinations']:>9,}   "
              f"{cell['spearman']:>7.4f}{cell['median_log_error']:>8.3f}"
              f"   {row['homogeneity']['ratio']:>12.2f}")
    if baseline and np.isfinite(baseline["spearman"]):
        print(f"{'(unclustered)':<13}{'':>10}{'':>9}   "
              f"{baseline['spearman']:>7.4f}{baseline['median_log_error']:>8.3f}"
              f"   {'-':>12}")
    print("\nrho and error cover only combinations whose cluster gained members, against the"
          "\nunclustered baseline: that comparison shows whether a mode denoises at all."
          "\n"
          "\nIt cannot pick the tightness - agreement always improves as clusters get coarser,"
          "\nup to the useless limit of one cluster. 'within/noise' is the check that does:"
          "\nhow far members of a cluster sit apart, over the noise on a single estimate."
          "\nNear 1 means members are interchangeable and pooling them is justified; well"
          "\nabove 1 means the mode is merging combinations that genuinely differ.")
    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
