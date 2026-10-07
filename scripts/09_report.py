#!/usr/bin/env python
"""Stage 9 - a report of what the DEL and the catalogue share, as HTML and PDF.

Reads stage 7's tables (and stage 8's maps, if they were made) and draws the
structures: the enriched fragments and fragment pairs that the catalogue
actually contains, each with the DEL evidence behind it and how many catalogue
compounds carry it.  The HTML inlines everything as SVG and base64 PNG, so a
single ``report.html`` can be copied or emailed on its own; ``--pdf`` writes the
same content paged for print.

    python scripts/09_report.py --results-dir results_binder_ratio
    python scripts/09_report.py --results-dir results_binder_ratio --min-enrichment 1
    python scripts/09_report.py --results-dir results_full --no-pdf

``--min-enrichment`` keeps only units scoring at least that much on whichever
ranking stage 7 used - binders per non-binder for a ``--rank ratio`` run, the
enrichment lower bound otherwise - so ``--min-enrichment 1`` on a ratio run
shows only units whose binders outnumber their non-binders.  It filters what
the report draws, not the tables, which always hold every enriched unit.

Each card shows raw ``binders / non-binders`` rather than their quotient: a
ratio alone hides whether it came from 400 binders or from 5.

Structures are drawn from the fragment SMILES, attachment points included: a
``[16*]`` dummy marks where the fragment was bonded, which is part of what the
match means.  A pair is drawn as one molecule, re-bonded through a
BRICS-compatible attachment pair (the same joining stage 8 fingerprints).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.io_utils import add_project_root_to_path  # noqa: E402

add_project_root_to_path()
RDLogger.DisableLog("rdApp.*")

# Categorical slots 1-3 of the reference palette, plus its ink and surfaces.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

COLUMNS = 5          # structures per row, both media
PDF_ROWS = 4         # rows of structures per PDF page


@dataclass
class Card:
    """One structure and the numbers under it, independent of the medium."""

    title: str
    mol: Chem.Mol
    stats: list[tuple[str, str]]


def _load_joiner():
    """Reuse stage 8's fragment joining rather than restating the BRICS rules."""
    import importlib.util
    path = Path(__file__).resolve().parent / "08_embed_space.py"
    spec = importlib.util.spec_from_file_location("stage8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._prepare


def _fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def rank_column(frame: pd.DataFrame, default: str) -> str:
    return "binder_per_nonbinder" if "binder_per_nonbinder" in frame.columns else default


def rank_label(column: str) -> str:
    return "binders per non-binder" if column == "binder_per_nonbinder" else "enrichment lower bound"


def build_cards(frame: pd.DataFrame, rank_col: str, top: int, prepare=None) -> list[Card]:
    """Rows -> drawable cards, strongest first, skipping anything RDKit refuses."""
    cards: list[Card] = []
    for _, row in frame.iterrows():
        if len(cards) >= top:
            break
        smiles = row["frag_smiles"]
        mol = prepare(smiles) if prepare is not None else Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        stats = [("binders / non-binders", f"{_fmt(row['n_binder'])} / {_fmt(row['n_nonbinder'])}")]
        # The counts are the ratio, spelled out - printing both says it twice.
        if rank_col != "binder_per_nonbinder" and rank_col in row:
            stats.append((rank_label(rank_col), _fmt(row[rank_col], 3)))
        stats.append(("Enamine compounds", _fmt(row["n_catalogue"])))
        title = str(row["combo_key"] if "combo_key" in row else row["frag_name"])
        cards.append(Card(title, mol, stats))
    return cards


# ----------------------------------------------------------------------- HTML --

def _svg(mol: Chem.Mol, width: int = 230, height: int = 170) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _html_cards(cards: list[Card]) -> str:
    out = []
    for card in cards:
        stats = "".join(
            f'<div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
            for k, v in card.stats
        )
        out.append(f'<figure class="card"><div class="struct">{_svg(card.mol)}</div>'
                   f'<figcaption><div class="name">{html.escape(card.title)}</div>'
                   f'<div class="stats">{stats}</div></figcaption></figure>')
    return "".join(out)


def _png_tag(path: Path) -> str:
    if not path.exists():
        return ('<p class="note">No embedding found - run <code>scripts/08_embed_space.py</code> '
                'to add the chemical-space maps.</p>')
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<a href="data:image/png;base64,{data}" target="_blank">'
            f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}"></a>')


def _stat_tiles(summary: dict, ratio: bool, shown: str) -> str:
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
                     summary["catalogue_compounds_with_enriched_combination_rows"]),
         "distinct compounds"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-n">{value:,}</div>'
        f'<div class="tile-l">{html.escape(label)}</div>'
        f'<div class="tile-s">{html.escape(sub)}</div></div>'
        for label, value, sub in tiles
    )
    return (f'<div class="tiles">{cells}</div>'
            f'<p class="note">Enriched means {criterion}.{shown}</p>')


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
  main {{ max-width: 1480px; margin: 0 auto; }}
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
  .grid {{ display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
  @media (max-width: 1180px) {{ .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
  @media (max-width: 720px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .card {{ margin: 0; border: 1px solid var(--grid); border-radius: 10px; overflow: hidden; background: #fff; }}
  .struct {{ display: flex; justify-content: center; align-items: center; padding: 6px 4px 0; min-height: 170px; }}
  .struct svg {{ max-width: 100%; height: auto; }}
  figcaption {{ padding: 9px 11px 12px; border-top: 1px solid var(--grid); }}
  .name {{ font-weight: 600; font-size: 12.5px; margin-bottom: 5px; }}
  .stats {{ display: grid; grid-template-columns: auto 1fr; gap: 2px 8px; font-size: 11.5px; }}
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
<p class="note">{frag_caption}</p>
<div class="grid">{fragment_cards}</div>

<h2>Enriched fragment pairs present in Enamine</h2>
<p class="note">{combo_caption}</p>
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


# ------------------------------------------------------------------------ PDF --

def _card_png(mol: Chem.Mol, width: int = 520, height: int = 260) -> bytes:
    """Cairo rather than SVG: matplotlib cannot place an SVG inside a figure."""
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().bondLineWidth = 3
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _pdf_section(pdf, cards: list[Card], heading: str, caption: str) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    per_page = COLUMNS * PDF_ROWS
    pages = max(1, -(-len(cards) // per_page))
    for page in range(pages):
        chunk = cards[page * per_page:(page + 1) * per_page]
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")  # A4 landscape
        fig.text(0.04, 0.955, heading + (f" ({page + 1}/{pages})" if pages > 1 else ""),
                 fontsize=14, color=INK, weight="semibold")
        fig.text(0.04, 0.925, caption, fontsize=8.5, color=INK_2)
        top, bottom, left, right = 0.895, 0.03, 0.04, 0.985
        cell_w = (right - left) / COLUMNS
        cell_h = (top - bottom) / PDF_ROWS
        # Draw the structure at the panel's own aspect ratio, so RDKit fills the
        # space instead of letting imshow letterbox a mismatched canvas.
        panel_w, panel_h = (cell_w - 0.008) * 11.69, cell_h * 0.62 * 8.27
        px_h = int(round(520 * panel_h / panel_w))
        for index, card in enumerate(chunk):
            row, col = divmod(index, COLUMNS)
            x0 = left + col * cell_w
            y0 = top - (row + 1) * cell_h
            # Structure on top, its numbers in the lower third of the cell.
            ax = fig.add_axes((x0 + 0.004, y0 + cell_h * 0.34, cell_w - 0.008, cell_h * 0.62))
            ax.imshow(mpimg.imread(io.BytesIO(_card_png(card.mol, 520, px_h)), format="png"))
            ax.axis("off")
            fig.text(x0 + 0.008, y0 + cell_h * 0.30, card.title, fontsize=7.5,
                     color=INK, weight="semibold")
            for line, (key, value) in enumerate(card.stats):
                y = y0 + cell_h * 0.30 - 0.022 * (line + 1)
                fig.text(x0 + 0.008, y, key, fontsize=6.5, color=INK_2)
                fig.text(x0 + cell_w - 0.012, y, value, fontsize=6.5, color=INK, ha="right")
        pdf.savefig(fig)
        plt.close(fig)


def _pdf_cover(pdf, title: str, subtitle: str, summary: dict, ratio: bool, lines: list[str]) -> None:
    import matplotlib.pyplot as plt
    from textwrap import fill

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(0.06, 0.88, title, fontsize=22, color=INK, weight="semibold")
    fig.text(0.06, 0.825, fill(subtitle, 110), fontsize=9.5, color=INK_2, va="top")
    tiles = [
        ("Enriched fragments", summary["fragments_enriched"],
         f"{summary['fragments_enriched_in_catalogue']:,} also in Enamine"),
        ("Enriched combinations", summary["combinations_enriched"],
         f"{summary['combinations_enriched_in_catalogue']:,} also in Enamine"),
        ("Enamine compounds with\nan enriched combination",
         summary.get("catalogue_distinct_compounds_enriched_combinations",
                     summary["catalogue_compounds_with_enriched_combination_rows"]),
         "distinct compounds"),
    ]
    for index, (label, value, sub) in enumerate(tiles):
        x = 0.06 + index * 0.30
        fig.text(x, 0.63, f"{value:,}", fontsize=26, color=INK, weight="semibold")
        fig.text(x, 0.585, label, fontsize=9.5, color=INK, va="top")
        # A two-line label pushes its own sub-line down, or they collide.
        fig.text(x, 0.545 - 0.026 * label.count("\n"), sub, fontsize=8.5, color=INK_2, va="top")
    criterion = (f"binders/non-binders >= {summary.get('min_ratio')} on at least "
                 f"{summary.get('min_binder')} binders" if ratio else
                 f"q < {summary.get('alpha')} and lower bound > {summary.get('min_lo95')}")
    body = [f"Enriched means {criterion}.", *lines]
    for index, line in enumerate(body):
        fig.text(0.06, 0.44 - index * 0.042, fill(line, 115), fontsize=9, color=INK_2, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_image(pdf, path: Path, heading: str) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    if not path.exists():
        return
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(0.04, 0.955, heading, fontsize=14, color=INK, weight="semibold")
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.90))
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def write_pdf(out: Path, title: str, subtitle: str, summary: dict, ratio: bool,
              frag_cards: list[Card], combo_cards: list[Card], captions: tuple[str, str],
              emb_dir: Path, notes: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out) as pdf:
        _pdf_cover(pdf, title, subtitle, summary, ratio, notes)
        _pdf_section(pdf, frag_cards, "Enriched fragments present in Enamine", captions[0])
        _pdf_section(pdf, combo_cards, "Enriched fragment pairs present in Enamine", captions[1])
        _pdf_image(pdf, emb_dir / "fragments_embedding.png", "Chemical space - fragments")
        _pdf_image(pdf, emb_dir / "combinations_embedding.png", "Chemical space - fragment pairs")


# ----------------------------------------------------------------------- main --

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
    parser.add_argument("--min-enrichment", type=float, default=None,
                        help="only draw units scoring at least this on the ranking stage 7 used "
                             "(binders per non-binder, or the enrichment lower bound)")
    parser.add_argument("--no-pdf", action="store_true", help="write only the HTML")
    parser.add_argument("--catalogue-name", default="Enamine REAL lead-like")
    args = parser.parse_args()

    cat_dir = args.catalogue_dir or args.results_dir / "catalogue"
    emb_dir = args.embedding_dir or args.results_dir / "embedding"
    out = args.out or cat_dir / "report.html"
    summary = json.loads((cat_dir / "summary_catalogue_hits.json").read_text())
    ratio = summary.get("rank") == "ratio"

    frag = pd.read_csv(cat_dir / "enriched_fragments_in_catalogue.csv")
    combo = pd.read_csv(cat_dir / "enriched_combinations_in_catalogue.csv")
    frag_rank = rank_column(frag, "enrichment_factor_lo95")
    combo_rank = rank_column(combo, "enrichment_mh_lo95")
    frag_in = frag[frag["in_catalogue"]]
    combo_in = combo[combo["in_catalogue"]]
    n_frag_all, n_combo_all = len(frag_in), len(combo_in)

    shown = ""
    if args.min_enrichment is not None:
        frag_in = frag_in[frag_in[frag_rank] >= args.min_enrichment]
        combo_in = combo_in[combo_in[combo_rank] >= args.min_enrichment]
        shown = (f" Showing only units with {rank_label(combo_rank)} &ge; "
                 f"{args.min_enrichment:g}: {len(frag_in):,} of {n_frag_all:,} fragments and "
                 f"{len(combo_in):,} of {n_combo_all:,} pairs.")

    frag_cards = build_cards(frag_in, frag_rank, args.top)
    combo_cards = build_cards(combo_in, combo_rank, args.top, prepare=_load_joiner())

    frag_caption = (f"{len(frag_cards)} of {len(frag_in):,}, strongest first. A [16*] dummy marks an "
                    f"attachment point: the match is the same fragment attached through the same chemistry.")
    combo_caption = (f"{len(combo_cards)} of {len(combo_in):,}, strongest first. Each pair is drawn "
                     f"re-bonded into one molecule; the remaining dummies are where the pair attached "
                     f"to the rest of the compound.")

    files = [p for p in sorted(cat_dir.iterdir()) if p.is_file() and p.suffix in (".csv", ".parquet")]
    rows = "".join(
        f'<tr><td><code>{html.escape(str(p.relative_to(cat_dir.parent)))}</code></td>'
        f'<td class="n">{p.stat().st_size / 1e6:,.1f} MB</td></tr>' for p in files
    )
    table = f'<table><thead><tr><th>File</th><th class="n">Size</th></tr></thead><tbody>{rows}</tbody></table>'

    scoring_html = ("raw binders per non-binder (<code>03b_enrich_by_ratio.py</code>), the ranking meant "
                    "for prospective transfer" if ratio else
                    "stratified Mantel-Haenszel enrichment (<code>03_enrich.py</code>)")
    title = f"DEL fragments found in {args.catalogue_name}"
    subtitle_html = (f"BRICS fragments and contiguous fragment pairs enriched in DEL binders, matched by "
                     f"content-hashed fragment id against {args.catalogue_name}. Scored on {scoring_html}.")
    page = TEMPLATE.format(
        title=title, subtitle=subtitle_html, columns=COLUMNS,
        tiles=_stat_tiles(summary, ratio, shown),
        frag_caption=html.escape(frag_caption).replace("[16*]", "<code>[16*]</code>"),
        combo_caption=html.escape(combo_caption),
        fragment_cards=_html_cards(frag_cards),
        combination_cards=_html_cards(combo_cards),
        fragment_map=_png_tag(emb_dir / "fragments_embedding.png"),
        combination_map=_png_tag(emb_dir / "combinations_embedding.png"),
        files=table,
        footer=f"Generated by scripts/09_report.py from {cat_dir}.",
        surface=SURFACE, ink=INK, ink2=INK_2, muted=MUTED, grid=GRID, green=GREEN,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frag_cards)} fragments + {len(combo_cards)} pairs drawn)")

    if not args.no_pdf:
        scoring = ("raw binders per non-binder, the ranking meant for prospective transfer"
                   if ratio else "stratified Mantel-Haenszel enrichment")
        notes = [f"Scored on {scoring}.",
                 "Cards show raw binder and non-binder counts: a ratio alone hides whether it came "
                 "from 400 binders or from 5.",
                 f"Full tables are the CSV and parquet files in {cat_dir}."]
        if shown:
            notes.insert(1, shown.strip().replace("&ge;", ">="))
        pdf_path = out.with_suffix(".pdf")
        write_pdf(pdf_path, title,
                  f"BRICS fragments and contiguous fragment pairs enriched in DEL binders, matched by "
                  f"content-hashed fragment id against {args.catalogue_name}.",
                  summary, ratio, frag_cards, combo_cards, (frag_caption, combo_caption),
                  emb_dir, notes)
        print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
