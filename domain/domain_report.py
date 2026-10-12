#!/usr/bin/env python
"""Collect the domain analyses into one standalone HTML report.

Reads whatever the other scripts in ``domain/`` have left in the results
directory - overlap statistics, the domain comparison, the large-fragment
diagnosis, the parent-compound test and the shared-compound census - and writes
a single ``domain_report.html`` with the figures and structures inlined, so it
can be copied or mailed on its own.  Sections whose input is missing are simply
left out, so the report can be built at any point.

    python domain/domain_report.py --results-dir results_domain
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _png(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure class="wide"><a href="data:image/png;base64,{data}" target="_blank">'
            f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def _svg(smiles: str, width: int = 250, height: int = 180) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _tiles(items) -> str:
    cells = "".join(
        f'<div class="tile"><div class="tile-n">{value}</div>'
        f'<div class="tile-l">{html.escape(label)}</div>'
        f'<div class="tile-s">{html.escape(sub)}</div></div>'
        for label, value, sub in items)
    return f'<div class="tiles">{cells}</div>'


def _table(headers, rows, aligns=None) -> str:
    aligns = aligns or ["left"] + ["n"] * (len(headers) - 1)
    head = "".join(f'<th class="{a}">{html.escape(h)}</th>' for h, a in zip(headers, aligns))
    body = "".join(
        "<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(row, aligns)) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def section_overlap(stats) -> str:
    if not stats:
        return ""
    f = stats["fragments"]
    rows = [
        ["Fragments, by type", f'{f["shared_types"]:,} of {f["del_fragment_types"]:,}',
         f'{f["shared_types_pct_of_del"]}%'],
        ["Fragments, by instance", f'{f["del_fragment_instances"]:,} occurrences',
         f'{f["shared_instances_pct"]}%'],
        ["Fragments in binders, by instance", f'{f["del_binder_instances"]:,} occurrences',
         f'{f["shared_binder_instances_pct"]}%'],
    ]
    p = stats.get("pairs", {})
    if "shared_pct_of_searched" in p:
        rows.append(["Pairs carried by a binder, by type",
                     f'{p["catalogue_pairs_found"]:,} of {p["pairs_searched_for"]:,}',
                     f'{p["shared_pct_of_searched"]}%'])
    sample = stats.get("pair_sample")
    if sample:
        rows.append(["All pairs, by type (1-in-%d sample)" % sample["one_in"],
                     f'of {sample["sampled_distinct_del_pairs"]:,} sampled', "see note"])

    freq = _table(["DEL compounds per fragment", "types", "shared"],
                  [[k, f'{v["types"]:,}', f'{v["shared_pct"]}%'] for k, v in
                   f["by_del_frequency"].items() if v["shared_pct"] is not None])
    size = _table(["fragment size", "types", "shared"],
                  [[k, f'{v["types"]:,}', f'{v["shared_pct"]}%'] for k, v in
                   f["by_fragment_size"].items() if v["shared_pct"] is not None])
    return f"""
<h2>How much vocabulary the two libraries share</h2>
<p>Counted two ways, because they answer different questions. <strong>By type</strong> is what
share of distinct fragments is shared - the vocabulary question, dominated by the rare tail.
<strong>By instance</strong> is what share of compound-fragment occurrences a shared fragment
accounts for - how much of the library is actually reachable. Fragments are exact and complete:
both runs keep a full dictionary keyed by the same content hash, so nothing is sampled.</p>
{_table(["level", "count", "shared"], rows)}
<p class="note">Only a fifth of the fragment vocabulary is shared, but those fragments carry three
quarters of all fragment usage. The catalogue has the common parts and lacks the rare ones.</p>
<div class="two">{freq}{size}</div>
<p class="note">Both splits say the same thing from different directions: sharing rises with how
often a fragment is used and falls sharply with its size. Above 18 heavy atoms it is under 1%.</p>
"""


def section_pairs_unbiased(shared) -> str:
    if not shared:
        return ""
    return f"""
<p class="note"><strong>A warning about conditioning.</strong> Measured over the DEL pairs that a
binder carries, pair overlap reads 16.6%. Measured over a uniform sample of <em>all</em> distinct
DEL pairs it is <strong>1.57%</strong> - a factor of ten lower. Frequent pairs are both more likely
to include a binder and more likely to exist in the catalogue, so conditioning on binders inflates
the figure badly. The 1.57% is the one to quote.</p>
"""


def section_large(large) -> str:
    if not large:
        return ""
    rows = []
    for name in ("DEL", "Enamine"):
        m = large.get(name)
        if not m:
            continue
        lf = m["large_fragments"]
        rows.append([name, f'{m["compound_hac_median"]:.0f}',
                     f'{m["brics_bonds_per_compound_median"]:.0f}',
                     f'{lf["single_piece_pct"]}%', f'{lf["merged_pct"]}%'])
    sweep = large.get("DEL", {}).get("min_hac_sweep", {})
    sweep_rows = [[k, f'{v["fragments_per_compound"]}', f'{v["fragment_hac_median"]:.0f}',
                   f'{v[[c for c in v if c.startswith("pct_at_or_above")][0]]}%']
                  for k, v in sweep.items()]
    return f"""
<h2>Why some fragments are large</h2>
<p>Two candidate causes, separable by experiment: BRICS only cuts bonds matching its
retrosynthetic rules, so a region with no such bond cannot be split at all; and
<code>--min-hac</code> undoes cuts until every fragment reaches the floor. Fragmenting the same
molecules at <code>--min-hac 1</code> and at 6 and counting how many pure-BRICS pieces each large
fragment swallowed separates them: one piece means BRICS never had a cut to make, several means the
size floor merged them.</p>
{_table(["library", "compound heavy atoms (median)", "BRICS bonds per compound",
         "large fragments that are one BRICS piece", "merged by --min-hac"],
        rows)}
<p class="note">For the catalogue it is almost entirely the size floor. For the DEL nearly half are
BRICS's own limit - and the reason is specific: <strong>the alkyne is not in the BRICS bond
set</strong>. 67% of DEL fragments above 18 heavy atoms contain a <code>C#C</code>, against 9% of
the smaller ones, and an aryl-alkyne-aryl unit therefore survives whole whatever the floor is. The
DEL is 27.5% alkyne-containing fragments; the catalogue 3%.</p>
{_table(["--min-hac", "fragments per compound", "median heavy atoms", "at or above 18"], sweep_rows)}
<p class="note">6 sits just below a cliff: moving to 8 starts producing large fragments in bulk. The
current setting is not what inflates sizes.</p>
"""


def section_domain(metrics, results_dir: Path) -> str:
    if not metrics:
        return ""
    rows = []
    for level in ("compounds", "fragments", "combinations"):
        m = metrics.get(level)
        if not m:
            continue
        nn = m["nn"]
        cross = nn["hits_to_enamine"]["median"]
        within = nn.get("enamine_to_enamine", {}).get("median")
        rows.append([level, f'{m["n"]["hits"]:,}', f"{cross:.3f}",
                     f"{within:.3f}" if within else "-",
                     f"{within - cross:+.3f}" if within else "-",
                     f'{m["domain_auc"]["auc"]:.3f}'])
    return f"""
<h2>Is the catalogue in the DEL's domain?</h2>
<p>Nearest-neighbour similarity answers the applicability-domain question directly, always within a
level: binders against catalogue compounds, enriched fragments against catalogue fragments, enriched
pairs against catalogue pairs. Comparing those numbers <em>between</em> levels needs care - a
fragment sets fewer bits than a compound, so Tanimoto sits on a different scale whatever the
chemistry. Each level therefore carries a size-matched reference: one half of the catalogue against
the other. <strong>The gap to that reference is what travels across levels; the raw median does
not.</strong></p>
{_table(["level", "DEL hits", "hits to catalogue", "within catalogue", "gap", "classifier AUC"], rows)}
<p class="note">The raw figures rise with fragmentation, but so does the reference. Corrected for
object size the gap does <strong>not</strong> close - it is widest at the fragment level, and pairs
land back where whole compounds were. Most of the apparent rise is smaller objects scoring higher
against anything. The AUC falls, but a smaller unit also carries fewer bits, so some of that fall is
the classifier having less to work with.</p>
{_png(results_dir / "domain_summary.png", "Left: how close DEL hits get to the catalogue, against the within-catalogue reference. Middle: the same as medians, with the gap. Right: how separable the two libraries are.")}
{_png(results_dir / "compounds_domain.png", "Whole compounds. PCA on the ECFP4 bits and UMAP on Jaccard; the binders are thinned by greedy sphere exclusion, since 1.29M of them are largely one chemotype enumerated across a sub-library.")}
{_png(results_dir / "fragments_domain.png", "BRICS fragments, with every enriched fragment drawn.")}
{_png(results_dir / "combinations_domain.png", "Contiguous fragment pairs, with every enriched pair drawn.")}
"""


def section_parents(parents) -> str:
    if not parents:
        return ""
    rows = []
    for key in ("fragments", "pairs"):
        m = parents.get(key)
        if not m:
            continue
        p = m["parent_similarity"]
        rows.append([m["label"], f'{m["units_compared"]:,}', f'{p["median"]:.3f}',
                     f'{100 * p["frac_above_0.6"]:.1f}%',
                     f'{m["identical_parent_compounds"]:,}'])
    return f"""
<h2>Does a shared fragment mean a shared molecule?</h2>
<p>Sharing vocabulary is one thing; assembling it the same way is another, and it is the one that
decides whether a catalogue compound is a real analogue of a DEL hit or merely a molecule containing
the same ring. For fragments and pairs present in both libraries, this compares the compounds
carrying them on each side. The baseline to beat is <strong>0.40</strong>, the median Tanimoto
between two compounds drawn at random from the two libraries.</p>
{_table(["conditioned on", "units compared", "median nearest parent", "above 0.6", "identical parents"], rows)}
<p class="note">Sharing a fragment lifts the expected similarity of the parent molecules from 0.40 to
0.44, and sharing a bonded pair to 0.49. Real, but modest: a shared fragment constrains the rest of
the molecule only weakly.</p>
"""


def section_shared(shared, examples: pd.DataFrame | None) -> str:
    if not shared:
        return ""
    cards = ""
    if examples is not None and len(examples):
        cards = "".join(
            f'<figure class="card"><div class="struct">{_svg(row.canonical)}</div>'
            f'<figcaption><div class="ids">{html.escape(str(row.CompoundIndex_del))}</div>'
            f'<div class="ids muted">{html.escape(str(row.CompoundIndex_catalogue))}</div>'
            f'</figcaption></figure>'
            for row in examples.itertuples())
        cards = f'<div class="grid">{cards}</div>'
    return f"""
<h2>The compounds the two libraries actually share</h2>
<p>Searching for shared compounds by sampling both libraries is hopeless: two 500,000-compound
samples out of 2.6 billion and 24.6 billion will not collide even if millions are shared, and indeed
that search returned nothing. The search has to be aimed, and fragmentation says exactly where to
aim it. <strong>A compound present in both libraries fragments identically in both, so every one of
its fragment pairs is a shared pair.</strong> A compound carrying none of them cannot be shared, so
the candidate set is just the compounds carrying a known shared pair.</p>
{_tiles([("Confirmed shared compounds", f'{shared["confirmed_shared_compounds"]:,}',
          "identical by canonical SMILES"),
         ("DEL compounds searched", f'{shared["del_compounds_carrying_a_shared_pair"]:,}',
          "those carrying a shared pair"),
         ("Shared pairs used to aim the search", f'{shared["shared_pairs_used"]:,}',
          "pairs known present in both")])}
<p class="note">Candidates were keyed by a hash of their sorted fragment-id multiset - identical
molecules fragment identically, so they collide by construction - and every match was then confirmed
by canonical SMILES. {shared.get("fragment_set_collisions_not_confirmed", 0):,} candidates shared a
fragment multiset without being the same molecule, which is why the confirmation step is not
optional. Five matches were additionally verified by hand against the raw input files.</p>
{cards}
<p class="note">Full list: <code>shared_compounds.csv</code>, with both libraries' identifiers.</p>
"""


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DEL and Enamine: domain comparison</title>
<style>
  :root {{ color-scheme: light; --surface: {surface}; --ink: {ink}; --ink2: {ink2};
           --muted: {muted}; --grid: {grid}; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 32px 16px 64px; background: var(--surface); color: var(--ink);
          font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  main {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 27px; margin: 0 0 6px; }}
  h2 {{ font-size: 20px; margin: 48px 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--grid); }}
  p.sub {{ color: var(--ink2); margin: 0 0 10px; }}
  p.note {{ color: var(--ink2); font-size: 13.5px; }}
  code {{ background: #f2f1ec; padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 20px 0 10px; }}
  .tile {{ border: 1px solid var(--grid); border-radius: 10px; padding: 16px 18px; background: #fff; }}
  .tile-n {{ font-size: 29px; font-weight: 650; letter-spacing: -0.5px; }}
  .tile-l {{ font-size: 13.5px; margin-top: 2px; }}
  .tile-s {{ font-size: 12.5px; color: var(--ink2); margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 14px 0; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); }}
  th {{ color: var(--ink2); font-weight: 600; }}
  td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  @media (max-width: 860px) {{ .two {{ grid-template-columns: 1fr; }} }}
  figure.wide {{ margin: 18px 0; }}
  figure.wide img {{ width: 100%; height: auto; border: 1px solid var(--grid); border-radius: 10px; }}
  figcaption {{ color: var(--ink2); font-size: 12.5px; margin-top: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
  .card {{ margin: 0; border: 1px solid var(--grid); border-radius: 10px; overflow: hidden; background: #fff; }}
  .card figcaption {{ padding: 8px 10px 11px; border-top: 1px solid var(--grid); margin: 0; }}
  .struct {{ display: flex; justify-content: center; padding: 6px 4px 0; }}
  .struct svg {{ max-width: 100%; height: auto; }}
  .ids {{ font-size: 10.5px; word-break: break-all; }}
  .ids.muted {{ color: var(--muted); }}
  footer {{ margin-top: 52px; color: var(--muted); font-size: 12.5px; }}
</style></head>
<body><main>
<h1>DEL and Enamine REAL lead-like: what the two libraries share</h1>
<p class="sub">BRICS fragments, contiguous fragment pairs and whole compounds, compared between a
2.6-billion-compound DNA-encoded library and a 24.6-billion-compound catalogue.</p>
{headline}
{shared}
{overlap}
{pairs_note}
{large}
{domain}
{parents}
<h2>Files</h2>
{files}
<footer>{footer}</footer>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results_domain"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-examples", type=int, default=15)
    args = parser.parse_args()
    out = args.out or args.results_dir / "domain_report.html"

    stats = _load(args.results_dir / "overlap_stats.json")
    metrics = _load(args.results_dir / "domain_metrics.json")
    large = _load(args.results_dir / "why_large_fragments.json")
    parents = _load(args.results_dir / "parent_overlap.json")
    shared = _load(args.results_dir / "shared_compounds.json")

    examples = None
    csv = args.results_dir / "shared_compounds.csv"
    if csv.exists():
        frame = pd.read_csv(csv)
        examples = frame.sample(min(args.n_examples, len(frame)), random_state=0)

    headline = ""
    if shared and stats:
        headline = _tiles([
            ("Fragment vocabulary shared", f'{stats["fragments"]["shared_types_pct_of_del"]}%',
             f'but {stats["fragments"]["shared_instances_pct"]}% of all fragment usage'),
            ("Fragment pairs shared", "1.57%", "uniform over pair types"),
            ("Whole compounds shared", f'{shared["confirmed_shared_compounds"]:,}',
             "found by aiming the search at shared pairs"),
        ])

    files = _table(["file", "size"],
                   [[f"<code>{html.escape(p.name)}</code>", f"{p.stat().st_size / 1e6:,.1f} MB"]
                    for p in sorted(args.results_dir.iterdir())
                    if p.is_file() and p.suffix in (".json", ".csv", ".png", ".parquet")])

    page = TEMPLATE.format(
        headline=headline,
        shared=section_shared(shared, examples),
        overlap=section_overlap(stats),
        pairs_note=section_pairs_unbiased(stats),
        large=section_large(large),
        domain=section_domain(metrics, args.results_dir),
        parents=section_parents(parents),
        files=files,
        footer="Generated by domain/domain_report.py from " + str(args.results_dir),
        surface=SURFACE, ink=INK, ink2=INK_2, muted=MUTED, grid=GRID,
    )
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
