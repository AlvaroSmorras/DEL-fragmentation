#!/usr/bin/env python
"""Stage 9 - a standalone HTML report of what the DEL and the catalogue share.

Reads stage 7's tables (and stage 8's maps, if they were made) and draws the
structures: the enriched fragments and fragment pairs that the catalogue
actually contains, each with the DEL evidence behind it and how many catalogue
compounds carry it.  Everything is inlined as SVG and base64 PNG, so the single
``report.html`` can be copied or emailed on its own.

    python scripts/09_report.py --results-dir results_binder_ratio

Structures are drawn from the fragment SMILES, attachment points included: a
``[16*]`` dummy marks where the fragment was bonded, which is part of what the
match means.  A pair is drawn as one molecule, re-bonded through a
BRICS-compatible attachment pair (the same joining stage 8 fingerprints).

``--top`` caps how many structures are drawn per panel; the full tables are
always the CSVs beside the report.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.io_utils import add_project_root_to_path  # noqa: E402

add_project_root_to_path()
RDLogger.DisableLog("rdApp.*")

# Categorical slots 1-3 of the reference palette, plus its ink and surfaces.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def _load_joiner():
    """Reuse stage 8's fragment joining rather than restating the BRICS rules."""
    import importlib.util
    path = Path(__file__).resolve().parent / "08_embed_space.py"
    spec = importlib.util.spec_from_file_location("stage8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._prepare


def _svg(mol: Chem.Mol, width: int = 260, height: int = 190) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int,)) or float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _card(svg: str, title: str, rows: list[tuple[str, str]]) -> str:
    stats = "".join(
        f'<div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>' for k, v in rows
    )
    return (f'<figure class="card"><div class="struct">{svg}</div>'
            f'<figcaption><div class="name">{html.escape(title)}</div>'
            f'<div class="stats">{stats}</div></figcaption></figure>')


def fragment_cards(frame: pd.DataFrame, top: int, ratio: bool) -> str:
    cards = []
    for _, row in frame.head(top).iterrows():
        mol = Chem.MolFromSmiles(row["frag_smiles"])
        if mol is None:
            continue
        stats = [("DEL binders", _fmt(row["n_binder"])),
                 ("DEL non-binders", _fmt(row["n_nonbinder"]))]
        if ratio and "binder_per_nonbinder" in row:
            stats.append(("binder/non-binder", _fmt(row["binder_per_nonbinder"], 3)))
        else:
            stats.append(("enrichment lo95", _fmt(row["enrichment_factor_lo95"])))
        stats.append(("Enamine compounds", _fmt(row["n_catalogue"])))
        cards.append(_card(_svg(mol), str(row["frag_name"]), stats))
    return "".join(cards)


def combination_cards(frame: pd.DataFrame, top: int, ratio: bool, prepare) -> str:
    cards = []
    for _, row in frame.head(top).iterrows():
        mol = prepare(row["frag_smiles"])
        if mol is None:
            continue
        stats = [("DEL binders", _fmt(row["n_binder"])),
                 ("DEL non-binders", _fmt(row["n_nonbinder"]))]
        if ratio and "binder_per_nonbinder" in row:
            stats.append(("binder/non-binder", _fmt(row["binder_per_nonbinder"], 3)))
        else:
            stats.append(("MH enrichment lo95", _fmt(row["enrichment_mh_lo95"])))
        stats.append(("Enamine compounds", _fmt(row["n_catalogue"])))
        cards.append(_card(_svg(mol, 300, 210), str(row["combo_key"]), stats))
    return "".join(cards)


def _png(path: Path) -> str:
    if not path.exists():
        return ('<p class="note">No embedding found - run <code>scripts/08_embed_space.py</code> '
                'to add the chemical-space maps.</p>')
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<a href="data:image/png;base64,{data}" target="_blank">'
            f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}"></a>')


def _stat_tiles(summary: dict, ratio: bool) -> str:
    criterion = (f"binders/non-binders &ge; {summary.get('min_ratio')} and &ge; "
                 f"{summary.get('min_binder')} binders" if ratio else
                 f"q &lt; {summary.get('alpha')} and lower bound &gt; {summary.get('min_lo95')}")
    tiles = [
        ("Enriched fragments", summary["fragments_enriched"],
         f"{summary['fragments_enriched_in_catalogue']:,} also in Enamine"),
        ("Enriched combinations", summary["combinations_enriched"],
         f"{summary['combinations_enriched_in_catalogue']:,} also in Enamine"),
        ("Enamine compounds carrying an enriched combination",
         summary.get("catalogue_distinct_compounds_enriched_combinations",
                     summary["catalogue_compounds_with_enriched_combination_rows"]), "distinct compounds"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-n">{value:,}</div>'
        f'<div class="tile-l">{html.escape(label)}</div>'
        f'<div class="tile-s">{html.escape(sub)}</div></div>'
        for label, value, sub in tiles
    )
    return f'<div class="tiles">{cells}</div><p class="note">Enriched means {criterion}.</p>'


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; --surface: {surface}; --ink: {ink}; --ink2: {ink2};
           --muted: {muted}; --grid: {grid}; --accent: {green}; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 32px 16px 64px; background: var(--surface); color: var(--ink);
          font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  main {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  h2 {{ font-size: 19px; margin: 44px 0 6px; padding-bottom: 6px; border-bottom: 1px solid var(--grid); }}
  p.sub {{ color: var(--ink2); margin: 0 0 8px; }}
  p.note {{ color: var(--ink2); font-size: 13.5px; }}
  code {{ background: #f2f1ec; padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 20px 0 8px; }}
  .tile {{ border: 1px solid var(--grid); border-radius: 10px; padding: 16px 18px; }}
  .tile-n {{ font-size: 30px; font-weight: 650; letter-spacing: -0.5px; }}
  .tile-l {{ font-size: 13.5px; margin-top: 2px; }}
  .tile-s {{ font-size: 12.5px; color: var(--ink2); margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; margin-top: 18px; }}
  .card {{ margin: 0; border: 1px solid var(--grid); border-radius: 10px; overflow: hidden; background: #fff; }}
  .struct {{ display: flex; justify-content: center; align-items: center; padding: 6px 4px 0; min-height: 190px; }}
  .struct svg {{ max-width: 100%; height: auto; }}
  figcaption {{ padding: 10px 14px 14px; border-top: 1px solid var(--grid); }}
  .name {{ font-weight: 600; font-size: 13.5px; margin-bottom: 6px; }}
  .stats {{ display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 12.5px; }}
  .stats .k {{ color: var(--ink2); }}
  .stats .v {{ text-align: right; font-variant-numeric: tabular-nums; }}
  img {{ width: 100%; height: auto; border: 1px solid var(--grid); border-radius: 10px; margin-top: 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 12px; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); }}
  th {{ color: var(--ink2); font-weight: 600; }}
  td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  footer {{ margin-top: 48px; color: var(--muted); font-size: 12.5px; }}
</style></head>
<body><main>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{tiles}

<h2>Enriched fragments present in Enamine</h2>
<p class="note">{n_frag_shown} of {n_frag_total:,}, strongest first. A <code>[16*]</code> dummy marks
an attachment point: the match is the same fragment attached through the same chemistry.</p>
<div class="grid">{fragment_cards}</div>

<h2>Enriched fragment pairs present in Enamine</h2>
<p class="note">{n_combo_shown} of {n_combo_total:,}, strongest first. Each pair is drawn re-bonded
into one molecule; the remaining dummies are where the pair attached to the rest of the compound.</p>
<div class="grid">{combination_cards}</div>

<h2>Chemical space</h2>
<p class="note">ECFP4 (radius 2, 2048 bits), PCA fitted on the background only, UMAP on Jaccard
distance. Click a map to open it full size.</p>
{fragment_map}
{combination_map}

<h2>Files</h2>
{files}

<footer>{footer}</footer>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results_binder_ratio"),
                        help="the DEL results directory stage 7 wrote under (default: %(default)s)")
    parser.add_argument("--catalogue-dir", type=Path, default=None,
                        help="stage 7 output (default: <results-dir>/catalogue)")
    parser.add_argument("--embedding-dir", type=Path, default=None,
                        help="stage 8 output (default: <results-dir>/embedding)")
    parser.add_argument("--out", type=Path, default=None, help="default: <catalogue-dir>/report.html")
    parser.add_argument("--top", type=int, default=60, help="structures drawn per panel")
    parser.add_argument("--catalogue-name", default="Enamine REAL lead-like")
    args = parser.parse_args()

    cat_dir = args.catalogue_dir or args.results_dir / "catalogue"
    emb_dir = args.embedding_dir or args.results_dir / "embedding"
    out = args.out or cat_dir / "report.html"
    summary = json.loads((cat_dir / "summary_catalogue_hits.json").read_text())
    ratio = summary.get("rank") == "ratio"

    frag = pd.read_csv(cat_dir / "enriched_fragments_in_catalogue.csv")
    combo = pd.read_csv(cat_dir / "enriched_combinations_in_catalogue.csv")
    frag_in = frag[frag["in_catalogue"]]
    combo_in = combo[combo["in_catalogue"]]

    files = [p for p in sorted(cat_dir.iterdir()) if p.is_file() and p.suffix in (".csv", ".parquet")]
    rows = "".join(
        f'<tr><td><code>{html.escape(str(p.relative_to(cat_dir.parent)))}</code></td>'
        f'<td class="n">{p.stat().st_size / 1e6:,.1f} MB</td></tr>' for p in files
    )
    table = f'<table><thead><tr><th>File</th><th class="n">Size</th></tr></thead><tbody>{rows}</tbody></table>'

    scoring = ("raw binders per non-binder (<code>03b_enrich_by_ratio.py</code>), the ranking meant for "
               "prospective transfer" if ratio else
               "stratified Mantel-Haenszel enrichment (<code>03_enrich.py</code>)")
    page = TEMPLATE.format(
        title=f"DEL fragments found in {args.catalogue_name}",
        subtitle=(f"BRICS fragments and contiguous fragment pairs enriched in DEL binders, matched by "
                  f"content-hashed fragment id against {args.catalogue_name}. Scored on {scoring}."),
        tiles=_stat_tiles(summary, ratio),
        n_frag_shown=min(args.top, len(frag_in)), n_frag_total=len(frag_in),
        n_combo_shown=min(args.top, len(combo_in)), n_combo_total=len(combo_in),
        fragment_cards=fragment_cards(frag_in, args.top, ratio),
        combination_cards=combination_cards(combo_in, args.top, ratio, _load_joiner()),
        fragment_map=_png(emb_dir / "fragments_embedding.png"),
        combination_map=_png(emb_dir / "combinations_embedding.png"),
        files=table,
        footer=f"Generated by scripts/09_report.py from {cat_dir}.",
        surface=SURFACE, ink=INK, ink2=INK_2, muted=MUTED, grid=GRID, green=GREEN,
    )
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{min(args.top, len(frag_in))} fragments + {min(args.top, len(combo_in))} pairs drawn)")


if __name__ == "__main__":
    main()
