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
scripts/0{1..6}_*.py   the pipeline stages
tests/test_pipeline.py pytest suite (python -m pytest tests/ -q)
```
