#!/usr/bin/env python
"""Stage 4 (optional) - render the top combinations as real, connected SMILES.

``combination_enrichment.parquet`` describes a combination by its member
fragments, which is enough to rank but awkward to look at: the members are
written as separate SMILES with open attachment points.  This stage rebuilds
each top combination inside one of the compounds that actually contains it, so
the bond joining the fragments is restored.

Writes ``top_combinations_smiles.csv`` next to the enrichment table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.combinations import combo_columns, connected_subsets  # noqa: E402
from lib.fragmentation import fragment_id, fragment_mol, merge_fragments  # noqa: E402
from lib.io_utils import add_project_root_to_path, input_files, progress, read_summary  # noqa: E402

add_project_root_to_path()


def _lookup_smiles(input_dir: Path, pattern: str, wanted: set[str]) -> dict[str, str]:
    """Fetch the SMILES of specific compounds, scanning the inputs once."""
    found: dict[str, str] = {}
    files = input_files(input_dir, pattern)
    for path in progress(files, len(files), "lookup"):
        table = pd.read_parquet(path, columns=["CompoundIndex", "Smiles"])
        hits = table[table["CompoundIndex"].isin(wanted)]
        found.update(zip(hits["CompoundIndex"], hits["Smiles"]))
        if len(found) == len(wanted):
            break
    return found


def _combination_smiles(smiles: str, members: list[int], min_hac: int) -> str | None:
    """Rebuild one combination inside a compound known to contain it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frag = fragment_mol(mol, min_hac=min_hac)
    position_ids = [fragment_id(s) for s in frag.smiles]
    target = sorted(members)
    for subset in connected_subsets(len(frag), frag.edges, len(members)):
        if sorted(position_ids[pos] for pos in subset) == target:
            return merge_fragments(mol, frag, subset)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/HGODEL"))
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--size", type=int, default=2, help="combination size to inspect")
    args = parser.parse_args()

    stage1 = read_summary(args.work_dir / "summary_fragment.json")
    min_hac = stage1["min_hac"]

    path = args.results_dir / f"combination_enrichment_size{args.size}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run stage 3 first")
    combos = pd.read_parquet(path).head(args.top).copy()
    combos = combos[combos["example_compound"].notna()]
    if combos.empty:
        raise SystemExit("no combinations with an example compound to inspect")

    keys = combo_columns(args.size)
    wanted = set(combos["example_compound"])
    print(f"looking up {len(wanted):,} example compounds")
    smiles_of = _lookup_smiles(args.input_dir, args.pattern, wanted)

    rendered = []
    for compound, members in zip(combos["example_compound"], combos[keys].to_numpy()):
        smiles = smiles_of.get(compound)
        rendered.append(
            None if smiles is None else _combination_smiles(smiles, list(members), min_hac)
        )
    combos["combination_smiles"] = rendered

    columns = [
        col for col in (
            "combo_key", "combination_smiles", "n_binder", "n_nonbinder", "n_strata",
            "enrichment_factor", "enrichment_mh", "enrichment_mh_lo95",
            "enrichment_factor_lo95", "q_value_mh", "q_value",
            "example_compound", "frag_smiles",
        ) if col in combos.columns
    ]
    out_path = args.results_dir / f"top_combinations_smiles_size{args.size}.csv"
    combos[columns].to_csv(out_path, index=False)
    missing = combos["combination_smiles"].isna().sum()
    print(f"wrote {out_path} ({len(combos):,} rows, {missing} unrenderable)")


if __name__ == "__main__":
    main()
