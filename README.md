# BRICS fragment-combination enrichment

Breaks compounds into BRICS fragments, builds the **contiguous** fragment
combinations of each compound, and scores every combination by how much more
often binders carry it than non-binders.

Input is any directory of parquet files with `CompoundIndex`, `Smiles` and
`activity` (1 = binder, 0 = non-binder). Nothing is parsed out of
`CompoundIndex`, so the pipeline transfers to libraries with any naming scheme.

```bash
./run_pipeline.sh                      # data/HGODEL -> work/ -> results/
MIN_HAC=8 SIZES=2,3 ./run_pipeline.sh  # larger fragments, pairs and triples
STRATIFY=none ./run_pipeline.sh        # library is one homogeneous set
```

## What each stage does

| Stage | Script | Output |
| --- | --- | --- |
| 1. Fragment | `scripts/01_fragment.py` | `work/fragments.parquet`, `work/compound_fragments/` |
| 2. Combine | `scripts/02_combine.py` | `work/combination_counts/size<k>/`, `work/combinations/size<k>/` |
| 3. Enrich | `scripts/03_enrich.py` | `results/combination_enrichment_size<k>.parquet`, `results/top_combinations_size<k>.csv` |
| 4. Inspect | `scripts/04_inspect.py` | `results/top_combinations_smiles_size<k>.csv` |
| 5. Cluster | `scripts/05_cluster.py` | `results/cluster_enrichment_size<k>.parquet`, `results/top_clusters_size<k>.csv` |
| 6. Validate | `scripts/06_validate_clusters.py` | printed report for choosing the clustering mode |
| 7. Catalogue hits | `scripts/07_catalogue_hits.py` | `results/catalogue/` - enriched fragments/combinations found in a catalogue, and its compounds carrying them |
| 8. Chemical space | `scripts/08_embed_space.py` | `results/embedding/` - ECFP4 PCA/UMAP maps of DEL vs catalogue |
| 9. Report | `scripts/09_report.py` | `results/catalogue/report.{html,pdf}` - structures of the shared enriched units, with the maps |

Stage 1 is the only expensive stage. Changing `--sizes`, `--min-binder`,
`--stratify` or `--alpha` only requires re-running stages 2-5.

If RDKit lives in a different interpreter from the one on `PATH`, set
`PYTHON=~/anaconda3/bin/python`.

### 1. Fragmentation

RDKit's BRICS cuts every retrosynthetic bond it can find, which leaves a lot of
two-atom linkers. Cuts are therefore **undone** until every fragment holds at
least `--min-hac` heavy atoms: the smallest offending fragment is repeatedly
merged into its smallest neighbour, so fragments stay evenly sized instead of
one of them growing into a blob. With `--min-hac 1` the output is identical to
`rdkit.Chem.BRICS.BreakBRICSBonds` (enforced by a test).

Fragment SMILES keep their BRICS attachment labels (`[16*]c1ccccc1`), so the
same ring joined through different chemistry stays a different fragment.

A fragment's id is a 64-bit hash of its canonical SMILES. That matters at scale:
numbering fragments in discovery order would need a pass over every compound
before any id was known, and would renumber the whole dictionary each time a
library was added. Content hashes let each worker write final output
immediately, and stay comparable across runs and across libraries. The short
`F000123` names are cosmetic, assigned by frequency once the dictionary is
folded together, so `F000000` is the commonest fragment.

`work/compound_fragments/` records, per compound, its fragment ids plus the
`edge_src`/`edge_dst` pairs telling which fragments were bonded to each other.

### 2. Contiguous combinations

A combination is contiguous when its fragments induce a connected subgraph of
that compound's fragment graph. For fragments A-B-C-D in a chain that gives
`AB`, `BC`, `CD`; if instead B carries both C and D it gives `AB`, `BC`, `BD`.
`--sizes 2` (the default) is exactly the set of bonded fragment pairs.

Combinations are stored as their member ids in `frag_0..frag_<k-1>` int64
columns, one directory per size, rather than as a joined `"F1|F2"` string.
Measured on the bundled data, grouping costs 390 bytes per distinct combination
with string keys against 252 with int64 columns — a 1.55x saving, worth having
but not on its own decisive.

A compound carrying the same fragment twice contributes the combination once.

### 3. Enrichment

The **pooled** enrichment factor is the straightforward reading:

```
enrichment_factor = (n_binder / N_binders) / (n_nonbinder / N_nonbinders)
```

It is also misleading on a multi-sub-library screen. In the bundled data the
per-file binder rate runs from 0% to 47%, 12 of 50 files contain no binders at
all, and 95% of top combinations occur in exactly one file — so a pooled score
largely rediscovers *which sub-library a compound came from*. Scoring the same
combinations within their own sub-library shrinks them by a median of 7.7x.

So by default each input file is a **stratum**, and combinations are scored with
a **Mantel-Haenszel** estimate that compares compounds only against others from
the same file, then pools those comparisons. It is an odds ratio rather than a
risk ratio, but when binders are a small fraction of the library — more true of
large libraries, not less — the two coincide, so it reads on the same scale.
Use `--stratify none` when a library ships as arbitrary shards of one
homogeneous set.

| Column | Meaning |
| --- | --- |
| `enrichment_mh` | Mantel-Haenszel enrichment, adjusted for stratum |
| `enrichment_mh_lo95` | 95% lower bound (Robins-Breslow-Greenland) — **the default sort key** |
| `q_value_mh` | BH-adjusted Mantel-Haenszel test, per combination size |
| `enrichment_factor` | pooled, unadjusted — keep it only to see the confounding |
| `enrichment_factor_lo95` | pooled 95% lower bound; the sort key under `--stratify none` |
| `n_strata` | how many input files the combination appeared in |

Rank on a lower bound, never on the point estimate: a combination in 2 binders
and 0 non-binders has an infinite ratio and tells you nothing.

`results/fragment_enrichment.parquet` applies the pooled statistics to single
fragments, which is the baseline a pair has to beat — a pair is only interesting
if it scores above both of its members.

### 4. Inspection

The enrichment table describes a combination as separate fragment SMILES with
open attachment points. Stage 4 rebuilds the top combinations inside a compound
that actually contains them, restoring the bond between the fragments:

```
F000013|F000028   [5*]NCC[C@H](C)NC(=O)c1ccc(Cl)c(S(N)(=O)=O)c1
```

### 5. Clustering near-identical combinations

Two combinations differing only by a fluorine or a methyl are usually one signal
measured twice, and splitting their compounds across two rows costs power.
Stage 5 gives each fragment a scaffold key, pairs those keys to cluster the
combinations, and re-scores the clusters.

Normalisation is per *fragment*, not per merged combination — combinations are
already keyed by their members, so normalising the members and re-pairing them
clusters without re-fragmenting anything. It is a canonical key rather than a
similarity threshold: keys need no all-against-all matrix (hopeless at millions
of combinations) and give groups that do not depend on the order clustering ran
in.

`--mode substituent` (the default) strips terminal methyls and halogens. Three
things are deliberately protected, each with a test:

- **heteroatom substituents stay** — stripping the NH2 off a primary sulfonamide
  would discard the group that binds;
- **stereocentres stay** — peeling the methyl off `[C@@H](C)` would silently
  erase the chirality;
- **`--max-strip` caps it** (default 2), so a fragment that would lose more than
  that keeps its exact identity rather than dissolving into a bare ring.

Cluster counts are **recounted from the per-compound table**, not summed from
the members, since a compound carrying two members of a cluster must still count
once. Summing instead (`--approximate`, for runs made with `--no-long-table`)
over-counted 1.75% of multi-member clusters on the bundled data.

Every cluster reports `best_member` and `pooling_gain` beside its pooled score.
Treat `pooling_gain` as a description, not a verdict: the best member is picked
after seeing the data, so its score is inflated by selection and the ratio reads
pessimistically. Stage 6 is the unbiased check.

### 6. Choosing how tightly to cluster

`scripts/06_validate_clusters.py` splits the compounds of each stratum in half,
scores every unit on each half independently, and asks how well the halves
agree. On the bundled data, over the combinations whose cluster gained members:

| mode | clusters | rho(A,B) | median log error | within/noise |
| --- | --- | --- | --- | --- |
| (unclustered) | 18,042 | 0.773 | 0.412 | — |
| `substituent` | 14,726 | 0.848 | 0.329 | 2.38 |
| `murcko` | 2,105 | 0.935 | 0.239 | 4.77 |
| `generic` | 378 | 0.966 | 0.190 | 6.55 |

So clustering does denoise: `substituent` cuts the replication error by 20%
against not clustering at all.

But agreement **cannot** pick the tightness, and reading that column alone would
be a trap — it keeps improving as clusters coarsen, all the way to the useless
limit of a single cluster that reproduces perfectly and says nothing. That is
what `generic` is doing at 378 clusters for 18,042 combinations.

`within/noise` is the column that decides: how far members of a cluster sit
apart, over the noise on a single estimate. Near 1 means members are
interchangeable and pooling them is justified; well above 1 means the mode is
merging combinations that genuinely differ. It rises steadily from
`substituent` to `generic`, which is the quantitative statement that the looser
modes buy their agreement by merging real differences.

## Catalogue stages (7-8)

Stages 7 and 8 compare a labelled DEL run (`work_full/`, `results_full/`) with
an unlabelled catalogue run (`work_catalogue/`, Enamine REAL lead-like, built
by `slurm/submit_catalogue.sh` with the target list from
`scripts/build_target_combos.py`).

```bash
# ratio ranking - the one for prospective transfer to a catalogue
RANK=ratio DEL_RESULTS=results_binder_ratio OUT_DIR=results_binder_ratio/catalogue \
    MAX_PER_FRAGMENT=0 sbatch --account naiss2025-3-21-cpu slurm/catalogue_hits.sbatch
DEL_RESULTS=results_binder_ratio OUT_DIR=results_binder_ratio/embedding \
    sbatch --account naiss2025-3-21-cpu slurm/embed_space.sbatch
python scripts/09_report.py --results-dir results_binder_ratio    # env: fragviz

# Mantel-Haenszel ranking, over results_full/
sbatch --account naiss2025-3-21-cpu slurm/catalogue_hits.sbatch
sbatch --account naiss2025-3-21-cpu slurm/embed_space.sbatch
```

**Which ranking.** `--rank ratio` keeps a unit when `binder_per_nonbinder >=
--min-ratio` (default 0.01, about 20x the library's base rate of 0.0005) on at
least `--min-binder` binders. It is the ranking to transfer to a catalogue: it
is the observed binder rate of compounds carrying the unit, which is what a
prospective prediction is about, and it needs a run scored by
`scripts/03b_enrich_by_ratio.py`. `--rank mh` (the default) instead uses the
stratified test, which answers whether the signal is real rather than how large
it is. The ratio ignores stratum and gives thin evidence no discount, so the
`--min-binder` floor is doing real work - see the caveat in
`03b_enrich_by_ratio.py`.

**Stage 7** matches on fragment ids. Ids are hashes of the canonical fragment
SMILES including the BRICS attachment labels, so a match is exact: same
fragment, attached through the same chemistry. "Enriched" means
`q < --alpha` (0.05) and the 95% lower bound `> --min-lo95` (1); combinations
use the Mantel-Haenszel columns, single fragments only have pooled ones. Writes
to `results_full/catalogue/`:

| File | Content |
| --- | --- |
| `fragment_in_catalogue.parquet`, `combination_in_catalogue.parquet` | every scored DEL unit + `n_catalogue`, `in_catalogue`, `enriched` |
| `enriched_*_in_catalogue.csv` | the enriched subset, sorted by lower bound |
| `catalogue_compounds_enriched_combinations.parquet` | **every** catalogue compound carrying an enriched combination, with SMILES and the combination's DEL statistics |
| `catalogue_compounds_enriched_fragments.parquet` | up to `--max-per-fragment` catalogue compounds per enriched fragment, spread over files (`--max-per-fragment 0` keeps every one); exact counts are in `fragment_in_catalogue.parquet` |
| `enamine_compounds_with_enriched_fragment.{parquet,csv}`, `enamine_compounds_with_enriched_combination.{parquet,csv}` | the buying list: compound id, compound SMILES, the unit and its SMILES, then the DEL evidence. The CSV is skipped above 5M rows; the parquet always holds every row |

Re-thresholding needs no rerun of the compound pass if you only tighten: filter
the compound tables on their `enrichment_*_lo95` / `q_value*` columns.

**Stage 8** samples 500k distinct fragments (and 500k distinct contiguous pairs,
enumerated from compounds drawn across 60 files) per library, adds every
enriched DEL unit, featurises with ECFP4 (radius 2, 1024 bits; attachment
points kept as unlabelled dummy atoms) and projects with PCA (fitted on the
background only) and UMAP (Jaccard on the bits). The catalogue run only stored the DEL
target pairs, so its combination background has to be sampled from
`compound_fragments/`. A pair is rebuilt as one molecule by bonding a
BRICS-compatible pair of attachment points; when a fragment has several
compatible ones the choice can differ from the real compound (76% of the
stage 4 top 500 rebuild exactly, the rest differ only in attachment site). The
third panel colours by whichever ranking stage 7 used.

### Why the PCA is not on the fingerprint

An ECFP4 PCA of this data is close to unreadable, and not because the sample is
small. Fitting it on two *disjoint* samples of the same size and measuring the
angle between the resulting PC1-PC2 planes:

| background per library | PC1+PC2 variance (DEL / Enamine) | plane angle between samples |
| --- | --- | --- |
| 2,000 | 14.2% / 10.4% | 72° / 74° |
| 10,000 | 11.7% / 8.4% | 57° / 60° |
| 50,000 | 10.2% / 7.2% | 41° / 57° |
| 200,000 | 12.5% / 6.4% | 51° / 41° |

Two components never hold more than about 13%, 50 components reach only 37-45%,
and the variance *falls* as the sample grows, because a larger sample exposes
more real diversity. The plane stays 40-57° from its own replicate even at
200k: the eigenvalues are near-degenerate, so the leading plane rotates freely.
No sample size fixes either problem - they are properties of sparse binary
substructure bits.

The same molecules through eleven standardised physicochemical descriptors
(`MolWt`, `cLogP`, `TPSA`, `HBD`, `HBA`, `RotBonds`, `Rings`, `AromaticRings`,
`Fsp3`, `HeavyAtoms`, `Heteroatoms`) give **59-75%** in two components with the
plane stable to **11-25°**, and axes that read as size (PC1) and polarity
(PC2) - `summary_embedding.json` records the loadings. That is the default.
`--pca-space ecfp` restores the fingerprint PCA. UMAP is on ECFP4 either way:
it uses Jaccard distance directly and never touches these axes, so none of this
applies to it.

### How big a background sample

Variance is not the sample-size question - coverage is. Median Tanimoto from a
held-out fragment to its nearest neighbour in the background:

| background per library | DEL | Enamine | Enamine, fraction with a neighbour above 0.6 |
| --- | --- | --- | --- |
| 10,000 | 0.49 | 0.39 | 3% |
| 50,000 | 0.60 | 0.46 | 12% |
| 200,000 | 0.67 | 0.55 | 35% |

Still climbing at 200k - even 500k distinct fragments is 2% of Enamine's 23M -
so the backdrop understates how much space the catalogue fills, and sparse
regions look emptier than they are. Hence the 500k default, which costs about
15 minutes on one node. This affects only the background: **every enriched unit
is always drawn, never sampled**, so the overlay and anything read off it are
unaffected by the sample size.

The two libraries are shuffled together before the scatter, so they interleave
in z. One `scatter` call draws in row order, so without that the library
concatenated last would sit on top everywhere and look like the denser one.
Marker size and alpha scale with the sample, since a million points at the size
that suited 50k is a solid blob.

**Stage 9** turns all of it into a `report.html` that inlines everything (so it
can be mailed on its own) and a `report.pdf` of the same content paged for
print: the counts, every shared enriched fragment and pair drawn as a structure
with its DEL evidence and catalogue count, and the stage 8 maps. Five
structures per row; the PDF is A4 landscape, 20 per page. `--no-pdf` skips the
PDF. It needs `fragviz` for RDKit's drawing code and takes a few seconds.

Cards carry raw `binders / non-binders` rather than their quotient - a ratio
alone hides whether it came from 400 binders or from 5 - plus the enrichment
lower bound on an `--rank mh` run, where that is not just the counts restated.

`--min-enrichment` draws only units scoring at least that much on whichever
ranking stage 7 used, so `--min-enrichment 1` on a ratio run keeps those whose
binders outnumber their non-binders. It filters the report, never the tables.

```bash
python scripts/09_report.py --results-dir results_binder_ratio --min-enrichment 1
```

## Scaling to hundreds of millions of compounds

Fragmentation is linear and embarrassingly parallel, so CPU is never the wall —
memory in the reduce steps is. Distinct fragments and combinations grow at about
N^0.85, i.e. near-linearly, because sub-libraries overlap very little (mean
fragment Jaccard 0.05). A 300M-compound library reaches roughly 9M fragments and
90M combinations, so nothing that holds one row per distinct combination in RAM
survives.

An unfiltered in-memory merge of 90M combinations would need roughly 24 GB just
to group, before any statistics. Three things keep it bounded, in descending
order of how much they buy:

- **`--min-binder` is applied inside each bucket**, before buckets are
  concatenated. A combination no binder carries cannot be enriched, and on a
  large library with a low hit rate that is almost all of them — 98.2% even in
  the bundled data, and more as the hit rate drops. On the bundled data this
  turns 2,358,873 groups into 41,950 and cuts the merge from 609 MB to 219 MB.
  It is by far the main control on both memory and output size.
- **Hash-partitioned merging** (`lib/aggregate.py`) sends every row for a key to
  the same on-disk bucket and groups one bucket at a time, so peak memory
  follows the bucket size rather than the number of distinct combinations
  (measured: 609 MB at one bucket, 354 MB at 64, same result either way). This
  is what makes the ceiling independent of library size. Both the fragment
  dictionary and the combination counts fold this way.
- **Content-hashed fragment ids** remove the global numbering pass over every
  compound and the dictionary broadcast to every worker, which at 9M fragments
  would have needed tens of GB across a pool. This also cut stage 1 wall time
  from 799s to 675s on the bundled data by deleting a whole pass.

On a library with few binders, prefer `--min-binder` over `--min-count`:
requiring 5 total compounds barely filters anything when the hit rate is 0.1%,
whereas requiring binders directly targets what can actually score. Set
`--no-long-table` too unless you need the per-compound table (about 800M rows at
300M compounds).

Significance is of limited use at that scale — with a large enough N, trivial
effects pass any FDR. Rank on `enrichment_mh_lo95` and treat `q_value_mh` as a
floor, not a ranking.

## Tuning

- `--min-hac` (default 6) is the main chemistry knob. Larger fragments are more
  specific but recur across fewer compounds, so statistics get thinner.
- `--min-binder` (default 1) controls output size; raise it on large libraries.
- `--min-count` (default 5) sets the total-compound floor.
- `--sizes 2,3` costs little in stage 2 but multiplies the number of
  combinations. Each size is corrected as its own testing family.
- `--stratify none` disables the sub-library adjustment.
- `--mode` / `--max-strip` set clustering tightness; check a change with stage 6
  rather than by eye.

## Layout

```
lib/fragmentation.py   BRICS cutting under a size constraint, fragment merging, ids
lib/combinations.py    connected-subgraph enumeration
lib/aggregate.py       hash-partitioned group-and-sum for out-of-core reduces
lib/enrichment.py      pooled and Mantel-Haenszel enrichment, BH correction
lib/scaffolds.py       fragment normalisation for clustering
scripts/0{1..9}_*.py   the pipeline stages (7-9: catalogue comparison and report)
tests/test_pipeline.py pytest suite (python -m pytest tests/ -q)
```
