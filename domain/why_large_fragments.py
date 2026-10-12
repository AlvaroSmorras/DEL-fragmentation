#!/usr/bin/env python
"""Why are so many fragments large - the BRICS bond set, or --min-hac?

Two candidate causes, and they are separable rather than a matter of opinion:

* **BRICS** only cuts bonds matching its retrosynthetic rules.  A region with no
  such bond inside it cannot be split however small the size floor is.
* **``--min-hac``** undoes cuts until every fragment reaches the floor, merging
  small pieces into neighbours.  That inflates sizes by construction.

The test: fragment the same molecule twice, at ``--min-hac 1`` (pure BRICS, the
pipeline's documented equivalence) and at the real setting.  Every fragment of
the second run is a union of fragments of the first, so counting how many
pure-BRICS pieces each large fragment swallowed says which cause produced it.
One piece means BRICS never had a cut to make there; several means the size
floor merged them.

    python domain/why_large_fragments.py --n 5000 --min-hac 6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from lib.fragmentation import _brics_bonds, fragment_mol  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_MIN_HAC = 6


def _init(min_hac: int) -> None:
    global _MIN_HAC
    _MIN_HAC = min_hac


def _analyse(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    bonds = _brics_bonds(mol)
    base = fragment_mol(mol, min_hac=1)     # pure BRICS
    real = fragment_mol(mol, min_hac=_MIN_HAC)

    # Which pure-BRICS piece each atom belongs to, so a merged fragment can be
    # resolved into the pieces it swallowed.
    piece_of = {}
    for index, atoms in enumerate(base.atoms):
        for atom in atoms:
            piece_of[atom] = index

    rows = []
    for atoms, hac in zip(real.atoms, real.hac):
        pieces = {piece_of[a] for a in atoms if a in piece_of}
        sizes = [len(base.atoms[p]) for p in pieces]
        rows.append({
            "hac": int(hac),
            "n_brics_pieces": len(pieces),
            "largest_piece": max(sizes) if sizes else 0,
        })
    return {
        "compound_hac": n_atoms,
        "n_brics_bonds": len(bonds),
        "n_fragments_pure": len(base),
        "n_fragments_real": len(real),
        "fragments": rows,
    }


def load(path: Path, column: str, n: int, seed: int) -> list[str]:
    files = sorted(path.glob("*.parquet"))
    rng = np.random.default_rng(seed)
    chosen = [files[i] for i in rng.choice(len(files), size=min(8, len(files)), replace=False)]
    per_file = int(np.ceil(n / len(chosen)))
    out: list[str] = []
    for file in chosen:
        table = pq.read_table(file, columns=[column])
        take = rng.choice(table.num_rows, size=min(per_file, table.num_rows), replace=False)
        out.extend(table[column].take(np.sort(take)).to_pylist())
    return out[:n]


def summarise(results: list[dict], min_hac: int, big: int) -> dict:
    frames = pd.DataFrame([f for r in results for f in r["fragments"]])
    per_mol = pd.DataFrame([{k: v for k, v in r.items() if k != "fragments"} for r in results])
    large = frames[frames["hac"] >= big]

    # A fragment made of one pure-BRICS piece was never merged: its size is
    # whatever BRICS left, and no size floor could have reduced it.
    unmerged = large["n_brics_pieces"] == 1
    out = {
        "n_compounds": int(len(per_mol)),
        "compound_hac_median": float(per_mol["compound_hac"].median()),
        "brics_bonds_per_compound_median": float(per_mol["n_brics_bonds"].median()),
        "compounds_with_no_brics_bond_pct": round(100 * float((per_mol["n_brics_bonds"] == 0).mean()), 2),
        "fragments_pure_brics": int(per_mol["n_fragments_pure"].sum()),
        f"fragments_at_min_hac_{min_hac}": int(per_mol["n_fragments_real"].sum()),
        "fragment_hac_median": float(frames["hac"].median()),
        f"fragments_at_or_above_{big}_hac_pct": round(100 * float((frames["hac"] >= big).mean()), 2),
        "large_fragments": {
            "n": int(len(large)),
            "single_piece_pct": round(100 * float(unmerged.mean()), 2),
            "merged_pct": round(100 * float((~unmerged).mean()), 2),
            "median_pieces_when_merged": float(large.loc[~unmerged, "n_brics_pieces"].median())
            if (~unmerged).any() else None,
            "median_largest_piece_when_merged": float(
                large.loc[~unmerged, "largest_piece"].median()) if (~unmerged).any() else None,
            "pieces_histogram": {str(k): int(v) for k, v in
                                 sorted(Counter(large["n_brics_pieces"]).items())[:8]},
        },
        "all_fragments": {
            "single_piece_pct": round(100 * float((frames["n_brics_pieces"] == 1).mean()), 2),
        },
    }
    # Where the size floor could not help anyway: a single piece already over
    # the floor is BRICS's own doing.
    out["large_single_piece_hac_median"] = float(large.loc[unmerged, "hac"].median()) if unmerged.any() else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-input", type=Path, default=PROJECT / "data/raw_HGODEL_labeled/sharded")
    parser.add_argument("--catalogue-input", type=Path, default=PROJECT / "data/Enamine_ll/sharded")
    parser.add_argument("--out", type=Path, default=PROJECT / "results_domain" / "why_large_fragments.json")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--min-hac", type=int, default=6)
    parser.add_argument("--big", type=int, default=18, help="heavy atoms that count as a large fragment")
    parser.add_argument("--sweep", default="1,3,6,8", help="min-hac values to compare sizes over")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    out = {"n_requested": args.n, "big": args.big}
    for name, path in [("DEL", args.del_input), ("Enamine", args.catalogue_input)]:
        smiles = load(path, "Smiles", args.n, seed=0)
        with Pool(args.workers, initializer=_init, initargs=(args.min_hac,)) as pool:
            results = [r for r in pool.map(_analyse, smiles, chunksize=50) if r]
        out[name] = summarise(results, args.min_hac, args.big)
        print(f"== {name}\n{json.dumps(out[name], indent=2)}", flush=True)

        # What the size floor itself costs, across settings.
        sweep = {}
        for min_hac in [int(v) for v in args.sweep.split(",")]:
            with Pool(args.workers, initializer=_init, initargs=(min_hac,)) as pool:
                res = [r for r in pool.map(_analyse, smiles[:min(args.n, 2000)], chunksize=50) if r]
            sizes = np.array([f["hac"] for r in res for f in r["fragments"]])
            sweep[str(min_hac)] = {
                "fragments_per_compound": round(float(np.mean([r["n_fragments_real"] for r in res])), 2),
                "fragment_hac_median": float(np.median(sizes)),
                "fragment_hac_p90": float(np.percentile(sizes, 90)),
                f"pct_at_or_above_{args.big}": round(100 * float((sizes >= args.big).mean()), 2),
            }
        out[name]["min_hac_sweep"] = sweep
        print(f"   sweep {json.dumps(sweep)}", flush=True)

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
