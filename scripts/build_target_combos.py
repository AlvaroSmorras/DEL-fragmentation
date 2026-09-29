#!/usr/bin/env python
"""Build the set of combinations to look for in an unlabelled catalogue.

Stage 2 can restrict itself to a list of combinations (``--target-combos``).
This builds that list from a *labelled* run, which is what makes searching a
multi-billion-compound catalogue tractable: the catalogue is fragmented once,
and only the combinations worth finding are ever written out.

``--min-binder`` sets how inclusive the list is:

* ``1`` (the default) - every combination at least one binder carries. This is
  the one to use. It depends on the labels but **not on the scoring**, so it
  survives re-running stage 3 with a different alpha, support floor,
  stratification or ranking, and it stays valid if the enrichment cut moves.
* ``0`` - every combination in the DEL, including those no binder ever touched.
  Far larger and no more useful for this DEL: a combination with no binders
  cannot become enriched under any scoring.

For a different DEL, build a new list from its own work directory and re-run
stage 2 over the same catalogue fragments - stage 1 never has to be repeated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.combinations import combo_columns  # noqa: E402
from lib.io_utils import add_project_root_to_path, progress, read_summary  # noqa: E402

add_project_root_to_path()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="the LABELLED run's work directory")
    parser.add_argument("--out", type=Path, required=True, help="target combination parquet")
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--min-binder", type=int, default=1,
                        help="binder compounds a combination needs to be included "
                             "(default: %(default)s)")
    args = parser.parse_args()

    stage1 = read_summary(args.work_dir / "summary_fragment.json")
    if not stage1.get("labelled", True):
        raise SystemExit(f"{args.work_dir} is unlabelled; build the target list from a DEL run")

    keys = combo_columns(args.size)
    files = sorted((args.work_dir / "combination_counts" / f"size{args.size}").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no stage 2 counts under {args.work_dir}")

    found: set[tuple[int, int]] = set()
    for path in progress(files, len(files), "scan"):
        frame = pd.read_parquet(path, columns=keys + ["n_binder"])
        if args.min_binder > 0:
            frame = frame[frame["n_binder"] >= args.min_binder]
        found.update(map(tuple, frame[keys].to_numpy()))

    out = pd.DataFrame(sorted(found), columns=keys).astype("int64")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out):,} target combinations (min_binder={args.min_binder}) -> {args.out}")
    print(f"stage 2 holds these as a set of packed ids, about "
          f"{len(out) * 80 / 1e6:.0f} MB in the parent process")


if __name__ == "__main__":
    main()
