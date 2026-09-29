#!/usr/bin/env python
"""Stage 3, ordered by raw binders per non-binder instead of the default.

Same scoring as ``03_enrich.py`` - every column is computed identically - but the
tables come out sorted on ``binder_per_nonbinder`` (``n_binder / n_nonbinder``),
tie-broken by ``n_binder``.  Stages 4 and 5 take the first N rows of what stage 3
wrote, so this changes what they treat as the top of the list.

    python scripts/03b_enrich_by_ratio.py --work-dir work_full
    python scripts/04_inspect.py  --results-dir results_by_ratio --work-dir work_full
    python scripts/05_cluster.py  --results-dir results_by_ratio --work-dir work_full --rank ratio

It writes to ``results_by_ratio/`` by default so the default-ranked tables in
``results/`` survive; pass ``--results-dir`` to override.

**What this ordering does and does not do.** The raw ratio is the enrichment
factor without its denominator: dividing it by the library's own binder to
non-binder ratio gives ``enrichment_factor`` exactly.  What it lacks is any
discount for thin evidence.  A combination in one binder and no non-binders
scores infinity, and every such combination ties there - which is why binder
count breaks the tie.  It also compares across sub-libraries, so it carries the
confounding that the stratified ranking exists to remove.  Use it to read the
raw numbers; rank candidates on ``mh_lo95`` when the decision costs money.

This is a wrapper, not a copy: the scoring lives in ``03_enrich.py`` and changes
there apply here automatically.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RANKING = "ratio"
DEFAULT_RESULTS = "results_by_ratio"


def _load_stage3():
    path = Path(__file__).resolve().parent / "03_enrich.py"
    spec = importlib.util.spec_from_file_location("stage3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    argv = sys.argv[1:]
    if {"-h", "--help"} & set(argv):
        # Delegating means argparse would print stage 3's help and never this
        # script's, which is the part that says how the two differ.
        print(__doc__)
        print("Options are stage 3's, minus --rank:\n")
    if "--rank" in argv:
        raise SystemExit("this script is the 'ratio' ranking; use 03_enrich.py --rank to choose another")
    if not any(arg == "--results-dir" or arg.startswith("--results-dir=") for arg in argv):
        argv += ["--results-dir", DEFAULT_RESULTS]
    sys.argv = [sys.argv[0], *argv, "--rank", RANKING]
    _load_stage3().main()


if __name__ == "__main__":
    main()
