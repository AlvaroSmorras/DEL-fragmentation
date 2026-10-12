#!/usr/bin/env python
"""Does fragmenting bring a catalogue into the DEL's domain?

A standalone comparison, separate from the scoring pipeline: it reads that
pipeline's work directories but writes only to its own results directory and
changes nothing the pipeline depends on.

The question is whether DEL and catalogue chemistry that looks disjoint as whole
compounds becomes shared once the compounds are cut up.  So the same comparison
runs at three levels:

* ``compounds``     - whole molecules, as the libraries ship them.
* ``fragments``     - the BRICS fragments of stage 1.
* ``combinations``  - the contiguous fragment pairs of stage 2.

Each level has three groups: a **DEL** background, an **Enamine** background,
and the DEL units that actually matter - its **binders** at compound level, the
**enriched** fragments and pairs at the other two.  That group is never sampled
down into the background, because asking "is this reachable from the catalogue"
is only meaningful for the units worth reaching.

It is, however, **thinned by greedy Tanimoto sphere exclusion**
(``--hit-selection leader``, ``--hit-threshold``).  A DEL's binders are largely
one chemotype enumerated across a sub-library - 1.29M of them collapse to a
small fraction at 0.7 - so drawing them all paints a block that hides both
backgrounds and overstates the space the hits occupy, and weights every metric
by how often a series was enumerated rather than by how many distinct things
were found.  ``--hit-selection none`` keeps every one.

Three things get measured, and the projections are the least of them:

1. **Projections** - PCA on the ECFP4 bits and UMAP on ECFP4 Jaccard, both
   fitted with the backgrounds only steering the axes.  The linear view is
   structural by default: a descriptor PCA answers whether the libraries share
   physicochemical *properties*, which is already known to be no, and putting it
   beside a structural question invites reading a property gap as a structural
   one.  ``--pca-space descriptors`` or ``both`` restores it, labelled as
   property space.  ``--tsne`` adds t-SNE on a subsample; see ``--help``.
2. **Nearest-neighbour similarity** - for each DEL unit, the best Tanimoto to
   any unit in the Enamine sample, always *within* a level: binders against
   catalogue compounds, enriched fragments against catalogue fragments,
   enriched pairs against catalogue pairs.  Nothing is ever compared to a
   different kind of object.

   Comparing those numbers *between* levels is a different matter, and the
   cross-library value alone will not carry: a fragment sets fewer bits than a
   compound, so Tanimoto between fragments sits on a different scale whatever
   the chemistry.  Each level therefore also measures a **within-library
   reference** - one half of Enamine against the other, and the same for the
   DEL.  That is what this object size scores when both sides are the same
   library.  The quantity that travels across levels is the gap between the
   cross-library curve and that reference, not the raw median.
3. **Domain separability** - the AUC of a logistic regression trained to tell
   DEL from Enamine on ECFP4.  0.5 means the two are indistinguishable, 1.0
   means perfectly separable.  Comparing this across the three levels is the
   cleanest answer to the question in the title: if fragmentation helps, the AUC
   falls as the unit gets smaller.  Read it beside the nearest-neighbour gap
   rather than alone - a smaller unit carries fewer bits, so some of any fall is
   the classifier having less to work with.

Plus diversity per group (mean pairwise distance, distinct Murcko scaffolds and
their entropy) and exact-overlap counts.

    python domain/compare_domains.py --out-dir results_domain --workers 48
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Scaffolds import MurckoScaffold

import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from lib.combinations import connected_subsets  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds",
                    "Rings", "AromaticRings", "Fsp3", "HeavyAtoms", "Heteroatoms"]
LEVELS = ("compounds", "fragments", "combinations")

# Reference palette slots 1-3: DEL, Enamine, and the DEL units that matter.
COLORS = {"DEL": "#2a78d6", "Enamine": "#eb6834", "hits": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

_GEN = None
_BITS = 1024
_STAGE8 = None


def stage8():
    """Reuse the embedding stage's fragment handling rather than restating it."""
    global _STAGE8
    if _STAGE8 is None:
        path = PROJECT / "scripts" / "08_embed_space.py"
        spec = importlib.util.spec_from_file_location("stage8", path)
        _STAGE8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_STAGE8)
    return _STAGE8


# ------------------------------------------------------------- featurisation --

def _init_worker(bits: int) -> None:
    global _GEN, _BITS
    _BITS = bits
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)


def _mol(smiles: str, is_unit: bool):
    """A whole compound parses directly; a fragment or pair goes through stage 8."""
    if not is_unit:
        return Chem.MolFromSmiles(smiles)
    return stage8()._prepare(smiles)


def _featurise(arg):
    smiles, is_unit = arg
    mol = _mol(smiles, is_unit)
    if mol is None:
        return None
    try:
        desc = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMD.CalcTPSA(mol),
                rdMD.CalcNumHBD(mol), rdMD.CalcNumHBA(mol), rdMD.CalcNumRotatableBonds(mol),
                rdMD.CalcNumRings(mol), rdMD.CalcNumAromaticRings(mol),
                rdMD.CalcFractionCSP3(mol), float(mol.GetNumHeavyAtoms()),
                float(rdMD.CalcNumHeteroatoms(mol))]
    except Exception:
        return None
    if not np.isfinite(desc).all():
        return None
    # Canonical form, so the exact-overlap count compares like with like even
    # when the two libraries write the same molecule differently.
    return _GEN.GetFingerprintAsNumPy(mol).astype(bool), desc, Chem.MolToSmiles(mol)


def featurise(smiles: list[str], is_unit: bool, workers: int, bits: int):
    with Pool(workers, initializer=_init_worker, initargs=(bits,)) as pool:
        out = pool.map(_featurise, ((s, is_unit) for s in smiles), chunksize=1000)
    ok = np.array([item is not None for item in out])
    fps = np.zeros((len(out), bits), dtype=bool)
    desc = np.zeros((len(out), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    canonical = []
    for i, item in enumerate(out):
        if item is not None:
            fps[i], desc[i], smi = item
            canonical.append(smi)
    return fps[ok], desc[ok], ok, canonical


def _murcko(arg):
    smiles, is_unit = arg
    mol = _mol(smiles, is_unit)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


# ------------------------------------------------------------------ sampling --

def _sample_del_file(arg):
    """Every binder in one DEL file, plus a random slice of its non-binders."""
    path, per_file, seed = arg
    table = pq.read_table(path, columns=["Smiles", "activity"]).to_pandas()
    binders = table.loc[table["activity"] == 1, "Smiles"].tolist()
    others = table.loc[table["activity"] != 1, "Smiles"]
    rng = np.random.default_rng(seed)
    take = rng.choice(len(others), size=min(per_file, len(others)), replace=False)
    return binders, others.iloc[np.sort(take)].tolist()


def _sample_catalogue_file(arg):
    path, per_file, seed = arg
    table = pq.read_table(path, columns=["Smiles"])
    rng = np.random.default_rng(seed)
    take = rng.choice(table.num_rows, size=min(per_file, table.num_rows), replace=False)
    return table["Smiles"].take(np.sort(take)).to_pylist()


def sample_compounds(args, rng):
    del_files = sorted((args.del_input).glob("*.parquet"))
    cat_files = sorted((args.catalogue_input).glob("*.parquet"))
    cat_files = [cat_files[i] for i in rng.choice(len(cat_files),
                                                  size=min(args.n_files, len(cat_files)),
                                                  replace=False)]
    # The DEL pass reads every file because binders are rare and all are wanted.
    per_del = int(np.ceil(args.n_background / len(del_files) * 1.1))
    per_cat = int(np.ceil(args.n_background / len(cat_files) * 1.1))
    print(f"compounds: scanning {len(del_files)} DEL files for binders, "
          f"{len(cat_files)} catalogue files", flush=True)
    with Pool(args.workers) as pool:
        del_out = pool.map(_sample_del_file,
                           [(p, per_del, i) for i, p in enumerate(del_files)], chunksize=4)
        cat_out = pool.map(_sample_catalogue_file,
                           [(p, per_cat, 10_000 + i) for i, p in enumerate(cat_files)], chunksize=4)
    binders = [s for b, _ in del_out for s in b]
    del_bg = [s for _, o in del_out for s in o]
    cat_bg = [s for part in cat_out for s in part]
    return (_take(del_bg, args.n_background, rng), _take(binders, args.max_binders, rng),
            _take(cat_bg, args.n_background, rng))


def _take(items, n, rng):
    if n <= 0 or len(items) <= n:
        return list(items)
    idx = rng.choice(len(items), size=n, replace=False)
    return [items[i] for i in np.sort(idx)]


def sample_units(args, level, rng):
    """Fragments or pairs: backgrounds from each library, plus the enriched DEL set."""
    s8 = stage8()
    hits_dir = args.del_results / "catalogue"
    if level == "fragments":
        del_bg = s8.sample_fragments(args.del_work, args.n_background, rng)["frag_smiles"].tolist()
        cat_bg = s8.sample_fragments(args.catalogue_work, args.n_background, rng)["frag_smiles"].tolist()
        table = pd.read_parquet(hits_dir / "fragment_in_catalogue.parquet",
                                columns=["frag_smiles", "enriched"])
    else:
        del_bg = s8.sample_combinations(args.del_work, args.n_background, args.n_files, rng)["smiles"].tolist()
        cat_bg = s8.sample_combinations(args.catalogue_work, args.n_background, args.n_files, rng)["smiles"].tolist()
        table = pd.read_parquet(hits_dir / "combination_in_catalogue.parquet",
                                columns=["frag_smiles", "enriched"])
    hits = table.loc[table["enriched"], "frag_smiles"].tolist()
    return del_bg, hits, cat_bg


# ------------------------------------------------------------------- metrics --

def _packed(fps: np.ndarray) -> np.ndarray:
    """Bits packed into uint64 lanes, for popcount-based Tanimoto."""
    padded = np.packbits(fps, axis=1)
    if padded.shape[1] % 8:
        padded = np.pad(padded, ((0, 0), (0, 8 - padded.shape[1] % 8)))
    return padded.view(np.uint64)


def nearest_tanimoto(queries: np.ndarray, reference: np.ndarray, n_queries: int,
                     rng) -> np.ndarray:
    """Best Tanimoto from each of a sample of queries to anything in reference.

    Exact, not approximate: each query is compared against the whole reference
    by popcount over packed bits, which is fast enough that an index would only
    add a way to be subtly wrong.
    """
    if len(queries) == 0 or len(reference) == 0:
        return np.array([])
    idx = rng.choice(len(queries), size=min(n_queries, len(queries)), replace=False)
    q = _packed(queries[idx])
    ref = _packed(reference)
    q_pop = np.bitwise_count(q).sum(axis=1).astype(np.int32)
    ref_pop = np.bitwise_count(ref).sum(axis=1).astype(np.int32)
    best = np.zeros(len(q))
    for i in range(len(q)):
        inter = np.bitwise_count(np.bitwise_and(q[i], ref)).sum(axis=1).astype(np.int32)
        union = q_pop[i] + ref_pop - inter
        best[i] = float((inter / np.maximum(union, 1)).max())
    return best


def internal_diversity(fps: np.ndarray, n: int, rng) -> float:
    """Mean pairwise Tanimoto *distance* inside a group, on a subsample."""
    if len(fps) < 2:
        return float("nan")
    idx = rng.choice(len(fps), size=min(n, len(fps)), replace=False)
    packed = _packed(fps[idx])
    pop = np.bitwise_count(packed).sum(axis=1)
    total, count = 0.0, 0
    for i in range(len(packed) - 1):
        inter = np.bitwise_count(np.bitwise_and(packed[i], packed[i + 1:])).sum(axis=1)
        union = pop[i] + pop[i + 1:] - inter
        total += float((1.0 - inter / np.maximum(union, 1)).sum())
        count += len(inter)
    return total / max(count, 1)


def diversity_select(fps: np.ndarray, threshold: float, max_n: int, rng,
                     chunk: int = 1024) -> np.ndarray:
    """Greedy sphere exclusion: keep a point only if nothing kept is that similar.

    A DEL's binders are mostly near-duplicates of each other - one chemotype
    enumerated across a sub-library - so plotting all of them paints a solid
    block that hides both backgrounds and overstates how much space the hits
    occupy. Sphere exclusion reduces them to their distinct chemistry, which is
    what a domain comparison is about. Candidates are visited in random order,
    so the survivors are not biased toward whatever the file order happened to
    be.

    Shared-bit counts come from a float32 matrix product rather than popcounts:
    the arithmetic is the same but BLAS threads it, which is the difference
    between minutes and hours at a million binders. Each block is first screened
    against everything already kept, and only the survivors - normally a small
    fraction - are then walked one at a time, since they can still exclude each
    other.
    """
    n = len(fps)
    if n == 0:
        return np.arange(0)
    cap = max_n if max_n else n
    bits = fps.astype(np.float32)
    pop = bits.sum(axis=1)
    order = rng.permutation(n)
    selected = np.empty((cap, fps.shape[1]), dtype=np.float32)
    selected_pop = np.empty(cap, dtype=np.float32)
    keep: list[int] = []

    def similar_to_selected(rows: np.ndarray) -> np.ndarray:
        """Max Tanimoto of each row to anything kept so far."""
        k = len(keep)
        if k == 0:
            return np.zeros(len(rows))
        inter = bits[rows] @ selected[:k].T
        union = pop[rows][:, None] + selected_pop[:k][None, :] - inter
        return (inter / np.maximum(union, 1.0)).max(axis=1)

    for start in range(0, n, chunk):
        block = order[start:start + chunk]
        block = block[similar_to_selected(block) < threshold]
        for i in block:
            if len(keep) and similar_to_selected(np.array([i]))[0] >= threshold:
                continue
            selected[len(keep)] = bits[i]
            selected_pop[len(keep)] = pop[i]
            keep.append(int(i))
            if len(keep) >= cap:
                return np.sort(np.array(keep, dtype=np.int64))
    return np.sort(np.array(keep, dtype=np.int64))


def domain_auc(fps_a: np.ndarray, fps_b: np.ndarray, n: int, rng) -> dict:
    """How separable the two libraries are, as a classifier's held-out AUC.

    0.5 means a model cannot tell them apart - the domains coincide.  Near 1.0
    means every unit carries a tell.  The number is only comparable across
    levels because the same model and the same sample size are used at each.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    n = min(n, len(fps_a), len(fps_b))
    if n < 100:
        return {"auc": float("nan"), "n_per_class": int(n)}
    a = fps_a[rng.choice(len(fps_a), size=n, replace=False)]
    b = fps_b[rng.choice(len(fps_b), size=n, replace=False)]
    X = np.vstack([a, b]).astype(np.float32)
    y = np.hstack([np.zeros(n), np.ones(n)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    model.fit(X_tr, y_tr)
    return {"auc": float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])),
            "n_per_class": int(n)}


def scaffold_stats(smiles: list[str], is_unit: bool, n: int, workers: int, rng) -> dict:
    sample = _take(smiles, n, rng)
    with Pool(workers) as pool:
        keys = pool.map(_murcko, ((s, is_unit) for s in sample), chunksize=500)
    keys = [k for k in keys if k]
    if not keys:
        return {"n_scaffolds": 0, "scaffold_entropy": float("nan"), "n_sampled": 0}
    counts = pd.Series(keys).value_counts()
    p = counts / counts.sum()
    return {"n_scaffolds": int(len(counts)),
            "scaffolds_per_1k": round(len(counts) / len(keys) * 1000, 1),
            "scaffold_entropy": float(-(p * np.log(p)).sum()),
            "n_sampled": int(len(keys))}


# --------------------------------------------------------------- projections --

def _tanimoto_distance(a_bits, a_pop, b_bits, b_pop) -> np.ndarray:
    """1 - Tanimoto, as a dense block, via BLAS rather than popcounts."""
    inter = a_bits @ b_bits.T
    union = a_pop[:, None] + b_pop[None, :] - inter
    return 1.0 - inter / np.maximum(union, 1.0)


def landmark_mds(fps: np.ndarray, groups: np.ndarray, n_landmarks: int, rng,
                 n_components: int = 2, chunk: int = 50_000):
    """Classical MDS on Tanimoto distance, through landmarks.

    PCA on the bits is structural but badly conditioned - its leading plane is
    nearly degenerate, so it rotates between samples.  Classical MDS instead
    embeds the distance that the rest of this comparison is phrased in, which is
    Tanimoto, and its axes are ordered by eigenvalue rather than by variance of
    an arbitrary binary coordinate system.

    A full N x N distance matrix is out of the question at a million units, so
    this is the landmark form (de Silva & Tenenbaum): embed a few thousand
    landmarks exactly, then place every other unit from its distances to those
    landmarks alone.  Landmarks are split evenly between the two libraries, so
    neither one defines the axes on its own.
    """
    bits = fps.astype(np.float32)
    pop = bits.sum(axis=1)
    picks = []
    for name in ("DEL", "Enamine"):
        pos = np.flatnonzero(groups == name)
        if len(pos):
            picks.append(rng.choice(pos, size=min(n_landmarks // 2, len(pos)), replace=False))
    land = np.sort(np.concatenate(picks))
    n_land = len(land)

    squared = _tanimoto_distance(bits[land], pop[land], bits[land], pop[land]) ** 2
    centring = np.eye(n_land, dtype=np.float64) - 1.0 / n_land
    gram = -0.5 * centring @ squared.astype(np.float64) @ centring
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = values[values > 0].sum()
    top = np.maximum(values[:n_components], 1e-12)

    # Out-of-sample placement: x = -1/2 . L^-1/2 V' (d^2 - mean column of D^2).
    column_mean = squared.mean(axis=0)
    projector = (vectors[:, :n_components] / np.sqrt(top)).astype(np.float32)
    coords = np.empty((len(bits), n_components), dtype=np.float32)
    for start in range(0, len(bits), chunk):
        stop = start + chunk
        d2 = _tanimoto_distance(bits[start:stop], pop[start:stop], bits[land], pop[land]) ** 2
        coords[start:stop] = -0.5 * (d2 - column_mean) @ projector
    variance = (values[:n_components] / positive).tolist() if positive > 0 else [np.nan] * n_components
    return coords, variance, int(n_land)


def project(fps: np.ndarray, desc: np.ndarray, groups: np.ndarray, args) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    background = np.isin(groups, ["DEL", "Enamine"])
    out: dict = {}
    # The question here is structural overlap, so the linear view is on the bits
    # by default. A descriptor PCA answers a different question - whether the
    # libraries share physicochemical properties - which is already known to be
    # no, and would make the figure look like a verdict on structure that it is
    # not. It stays available, and `both` shows the contrast deliberately.
    if args.pca_space in ("ecfp", "both"):
        pca = PCA(n_components=2, random_state=args.seed).fit(
            fps[background].astype(np.float32))
        out["pca"] = pca.transform(fps.astype(np.float32))
        out["pca_variance"] = pca.explained_variance_ratio_.tolist()
        out["pca_label"] = "ECFP4 PCA"
    if args.pca_space in ("descriptors", "both"):
        scaler = StandardScaler().fit(desc[background])
        matrix = scaler.transform(desc)
        dpca = PCA(n_components=2, random_state=args.seed).fit(matrix[background])
        key = "dpca" if args.pca_space == "both" else "pca"
        out[key] = dpca.transform(matrix)
        out[f"{key}_variance"] = dpca.explained_variance_ratio_.tolist()
        out["descriptor_loadings"] = {
            f"PC{i + 1}": dict(sorted(zip(DESCRIPTOR_NAMES, comp), key=lambda kv: -abs(kv[1]))[:4])
            for i, comp in enumerate(dpca.components_)}
        if args.pca_space == "descriptors":
            out["pca_label"] = "descriptor PCA"
    if args.mds:
        coords, variance, n_land = landmark_mds(fps, groups, args.mds_landmarks, 
                                                np.random.default_rng(args.seed))
        out["mds"] = coords
        out["mds_variance"] = variance
        out["mds_landmarks"] = n_land

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="jaccard",
                        n_jobs=args.workers, low_memory=True, random_state=None, verbose=True)
    out["umap"] = reducer.fit_transform(fps)

    if args.tsne:
        # t-SNE has no Barnes-Hut implementation for Jaccard and does not scale
        # to millions, so it runs on a PCA of the bits over a subsample.  Its
        # between-cluster distances are not meaningful; it is here to confirm
        # the local picture, not to measure the gap.
        from sklearn.manifold import TSNE
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(fps), size=min(args.tsne_n, len(fps)), replace=False)
        reduced = PCA(n_components=50, random_state=args.seed).fit_transform(
            fps[idx].astype(np.float32))
        out["tsne"] = TSNE(n_components=2, init="pca", perplexity=50,
                           random_state=args.seed).fit_transform(reduced)
        out["tsne_index"] = idx
    return out


def plot_level(frame: pd.DataFrame, level: str, meta: dict, out: Path, hit_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in ("pca", "dpca", "mds", "umap", "tsne") if f"{m}_1" in frame.columns]
    var = meta.get("pca_variance", [np.nan, np.nan])
    # One row gets unreadably wide past three panels, so the grid wraps.
    n_panels = len(methods) + 1
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.6 * n_rows),
                             facecolor=SURFACE, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    for ax, method in zip(axes, methods):
        sub = frame.dropna(subset=[f"{method}_1"])
        bg = sub[sub["group"] != "hits"]
        hits = sub[sub["group"] == "hits"]
        size = float(np.clip(600_000 / max(len(bg), 1), 0.4, 2.0))
        alpha = float(np.clip(120 / np.sqrt(max(len(bg), 1)), 0.10, 0.40))
        # A handful of hits has to be enlarged to be visible at all; a group as
        # big as the backgrounds is a third population and is drawn like one,
        # interleaved in z, or it would simply bury them.
        crowded = len(hits) > 0.2 * max(len(bg), 1)
        if crowded:
            everything = sub.sample(frac=1.0, random_state=0)
            ax.scatter(everything[f"{method}_1"], everything[f"{method}_2"], s=size,
                       c=everything["group"].map(COLORS), alpha=alpha, linewidths=0,
                       rasterized=True)
        else:
            shuffled = bg.sample(frac=1.0, random_state=0)
            ax.scatter(shuffled[f"{method}_1"], shuffled[f"{method}_2"], s=size,
                       c=shuffled["group"].map(COLORS), alpha=alpha, linewidths=0,
                       rasterized=True)
            ax.scatter(hits[f"{method}_1"], hits[f"{method}_2"], s=max(size * 6, 7.0),
                       color=COLORS["hits"], alpha=0.75, linewidths=0.25, edgecolors=SURFACE,
                       rasterized=True)
        if method in ("pca", "dpca"):
            which = meta.get("pca_label", "PCA") if method == "pca" else "descriptor PCA"
            ratios = var if method == "pca" else meta.get("dpca_variance", [np.nan, np.nan])
            ax.set_xlabel(f"{which} PC1 ({ratios[0]:.0%})", color=INK_2)
            ax.set_ylabel(f"PC2 ({ratios[1]:.0%})", color=INK_2)
            ax.set_title(which + (" - property space" if method == "dpca" else " - structure"),
                         loc="left", color=INK)
        elif method == "mds":
            ratios = meta.get("mds_variance", [np.nan, np.nan])
            ax.set_xlabel(f"PCo1 ({ratios[0]:.0%})", color=INK_2)
            ax.set_ylabel(f"PCo2 ({ratios[1]:.0%})", color=INK_2)
            ax.set_title(f"Tanimoto MDS - {meta.get('mds_landmarks', 0):,} landmarks",
                         loc="left", color=INK)
        else:
            ax.set_xlabel(f"{method.upper()} 1", color=INK_2)
            ax.set_ylabel(f"{method.upper()} 2", color=INK_2)
            ax.set_title(f"{method.upper()} on ECFP4", loc="left", color=INK)
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Nearest-neighbour similarity to Enamine: the domain question, as a curve.
    ax = axes[len(methods)]
    curves = [("DEL to Enamine", "del_to_enamine", COLORS["DEL"], "-"),
              (f"{hit_label} to Enamine", "hits_to_enamine", COLORS["hits"], "-"),
              ("Enamine to DEL", "enamine_to_del", COLORS["Enamine"], "-"),
              ("DEL to DEL (reference)", "del_to_del", COLORS["DEL"], ":"),
              ("Enamine to Enamine (ref.)", "enamine_to_enamine", COLORS["Enamine"], ":")]
    for name, key, color, style in curves:
        values = np.asarray(meta.get("nn", {}).get(key, {}).get("values", []))
        if len(values):
            ordered = np.sort(values)
            ax.plot(ordered, np.linspace(0, 1, len(ordered)), color=color, lw=2,
                    ls=style, label=name)
    ax.set_xlabel("best Tanimoto to the other library", color=INK_2)
    ax.set_ylabel("fraction at or below", color=INK_2)
    ax.set_title("nearest-neighbour similarity", loc="left", color=INK)
    ax.set_xlim(0, 1)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    leg.get_frame().set_edgecolor(GRID)

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[g],
                          label=f"{n} ({int((frame['group'] == g).sum()):,})")
               for g, n in [("DEL", "DEL"), ("Enamine", "Enamine"), ("hits", hit_label)]]
    axes[0].legend(handles=handles, loc="best", fontsize=8.5, frameon=True, markerscale=1.2)
    fig.suptitle(f"DEL vs Enamine REAL lead-like - {level}", x=0.01, ha="left",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if n_rows == 1 else 0.97))
    fig.savefig(out, dpi=190, facecolor=SURFACE)
    plt.close(fig)


def plot_summary(metrics: dict, out: Path) -> None:
    """The one figure that answers the question, across the three levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [lv for lv in LEVELS if lv in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor=SURFACE)
    shades = {"compounds": "#9ec5f0", "fragments": "#5b9ae0", "combinations": "#2a78d6"}

    ax = axes[0]
    for level in levels:
        values = np.asarray(metrics[level]["nn"]["hits_to_enamine"].get("values", []))
        if len(values):
            ax.plot(np.sort(values), np.linspace(0, 1, len(values)), lw=2.2,
                    color=shades[level], label=level)
        ref = np.asarray(metrics[level]["nn"].get("enamine_to_enamine", {}).get("values", []))
        if len(ref):
            ax.plot(np.sort(ref), np.linspace(0, 1, len(ref)), lw=1.4, ls=":",
                    color=shades[level])
    ax.set_title("DEL hits into Enamine (solid) vs within Enamine (dotted)",
                 loc="left", color=INK)
    ax.set_xlabel("nearest-neighbour Tanimoto", color=INK_2)
    ax.set_ylabel("fraction at or below", color=INK_2)
    ax.set_xlim(0, 1)

    # Raw medians are not comparable across levels - a pair is a smaller object
    # than a compound and scores differently against anything. The within-library
    # number is the same object size measured against its own chemistry, so the
    # distance between the two bars is what carries across levels.
    ax = axes[1]
    cross = [metrics[lv]["nn"]["hits_to_enamine"].get("median", np.nan) for lv in levels]
    within = [metrics[lv]["nn"].get("enamine_to_enamine", {}).get("median", np.nan)
              for lv in levels]
    x = np.arange(len(levels))
    ax.bar(x - 0.19, cross, width=0.36, color=[shades[lv] for lv in levels],
           label="DEL hits to Enamine")
    ax.bar(x + 0.19, within, width=0.36, color="#c9c8c2", label="within Enamine (same size)")
    for xi, (a, b) in enumerate(zip(cross, within)):
        ax.text(xi - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=9.5, color=INK)
        ax.text(xi + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=9.5, color=INK_2)
        ax.text(xi, max(a, b) + 0.075, f"gap {b - a:+.2f}", ha="center", fontsize=9.5,
                color=INK, weight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1)
    ax.set_title("median nearest neighbour, against a size-matched reference",
                 loc="left", color=INK)
    ax.set_ylabel("Tanimoto", color=INK_2)
    ax.legend(fontsize=8.5, frameon=True, loc="upper right").get_frame().set_edgecolor(GRID)

    ax = axes[2]
    aucs = [metrics[lv]["domain_auc"]["auc"] for lv in levels]
    bars = ax.bar(levels, aucs, color=[shades[lv] for lv in levels], width=0.6)
    for bar, value in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", fontsize=10, color=INK)
    ax.axhline(0.5, color=INK_2, lw=1.2, ls="--")
    ax.text(0.02, 0.52, "0.5 = indistinguishable", transform=ax.get_yaxis_transform(),
            fontsize=9, color=INK_2)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("DEL vs Enamine separability (classifier AUC)", loc="left", color=INK)
    ax.set_xlabel("smaller units carry fewer bits, so read this with the left panels",
                  color=INK_2, fontsize=8.5)
    ax.set_ylabel("held-out AUC", color=INK_2)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].legend(fontsize=9, frameon=True).get_frame().set_edgecolor(GRID)
    fig.suptitle("Does fragmenting bring Enamine into the DEL's domain?",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=190, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- main --

def exact_overlap(level: str, args, smiles, raw, is_del, is_cat, is_hit) -> dict:
    """How much of the DEL exists in the catalogue *identically*.

    Comparing the two samples would measure the sampling, not the libraries, so
    fragments are looked up in the catalogue's whole 23M dictionary by content
    hash.  Pairs cannot be: the catalogue run only stored the pairs a DEL
    binder carries, so the honest figure there is the one stage 7 already
    computed over that target list, and the background is simply unknown.
    """
    if level == "fragments":
        from lib.fragmentation import fragment_id
        catalogue = pq.read_table(args.catalogue_work / "fragments.parquet",
                                  columns=["frag_id"])["frag_id"].to_numpy()
        catalogue = np.sort(catalogue)
        out = {"basis": "full catalogue fragment dictionary"}
        for name, mask in [("del", is_del), ("hits", is_hit)]:
            # Hash the fragment as the pipeline writes it, attachment labels and
            # all: the canonical form used elsewhere here has those stripped, and
            # hashing it would silently match almost nothing.
            ids = np.array([fragment_id(raw[i]) for i in np.flatnonzero(mask)], dtype=np.int64)
            if not len(ids):
                continue
            found = int(np.isin(ids, catalogue, assume_unique=False).sum())
            out[f"{name}_in_enamine"] = found
            out[f"{name}_in_enamine_frac"] = round(found / len(ids), 4)
        return out

    if level == "combinations":
        table = pd.read_parquet(args.del_results / "catalogue" / "combination_in_catalogue.parquet",
                                columns=["enriched", "in_catalogue"])
        hits = table[table["enriched"]]
        return {"basis": "stage 7 target list (every DEL pair with a binder)",
                "hits_in_enamine": int(hits["in_catalogue"].sum()),
                "hits_in_enamine_frac": round(float(hits["in_catalogue"].mean()), 4),
                "del_in_enamine": None,
                "del_in_enamine_note": "catalogue pairs outside the target list were never counted"}

    # Whole compounds: only the two samples can be compared, and the answer is
    # expected to be nil - which is itself the point of the comparison.
    canon_del = {smiles[i] for i in np.flatnonzero(is_del)}
    canon_cat = {smiles[i] for i in np.flatnonzero(is_cat)}
    canon_hit = {smiles[i] for i in np.flatnonzero(is_hit)}
    return {"basis": f"the two {len(canon_cat):,}-compound samples, not the whole libraries",
            "del_in_enamine": len(canon_del & canon_cat),
            "del_in_enamine_frac": round(len(canon_del & canon_cat) / max(len(canon_del), 1), 6),
            "hits_in_enamine": len(canon_hit & canon_cat),
            "hits_in_enamine_frac": round(len(canon_hit & canon_cat) / max(len(canon_hit), 1), 6)}


def run_level(level: str, args, rng) -> dict:
    start = time.time()
    is_unit = level != "compounds"
    if level == "compounds":
        del_bg, hits, cat_bg = sample_compounds(args, rng)
        hit_label = "DEL binders"
    else:
        del_bg, hits, cat_bg = sample_units(args, level, rng)
        hit_label = "DEL enriched"
    print(f"[{level}] {len(del_bg):,} DEL + {len(cat_bg):,} Enamine + {len(hits):,} {hit_label}",
          flush=True)

    smiles = del_bg + hits + cat_bg
    groups = np.array(["DEL"] * len(del_bg) + ["hits"] * len(hits) + ["Enamine"] * len(cat_bg))
    fps, desc, ok, canonical = featurise(smiles, is_unit, args.workers, args.bits)
    raw = [s for s, keep in zip(smiles, ok) if keep]
    smiles = canonical
    groups = groups[ok]
    print(f"[{level}] featurised {len(fps):,} ({int((~ok).sum())} unparsable)", flush=True)

    select_here = (args.hit_selection == "leader"
                   and level in {lv.strip() for lv in args.hit_selection_levels.split(",")})
    if select_here and (groups == "hits").sum() > 1:
        hit_pos = np.flatnonzero(groups == "hits")
        picked = hit_pos[diversity_select(fps[hit_pos], args.hit_threshold,
                                          args.max_hits_selected, rng)]
        drop = np.setdiff1d(hit_pos, picked)
        mask = np.ones(len(groups), dtype=bool)
        mask[drop] = False
        fps, desc, groups = fps[mask], desc[mask], groups[mask]
        smiles = [s for s, keep in zip(smiles, mask) if keep]
        raw = [s for s, keep in zip(raw, mask) if keep]
        metrics_selection = {"mode": "leader", "threshold": args.hit_threshold,
                             "before": int(len(hit_pos)), "after": int(len(picked))}
        print(f"[{level}] diversity selection kept {len(picked):,} of {len(hit_pos):,} "
              f"{hit_label} at Tanimoto {args.hit_threshold}", flush=True)
    else:
        metrics_selection = {"mode": "none", "kept_all": int((groups == "hits").sum())}

    is_del, is_cat, is_hit = groups == "DEL", groups == "Enamine", groups == "hits"
    metrics = {
        "n": {"DEL": int(is_del.sum()), "Enamine": int(is_cat.sum()), "hits": int(is_hit.sum())},
        "hit_label": hit_label,
        "hit_selection": metrics_selection,
    }
    metrics["exact_overlap"] = exact_overlap(level, args, smiles, raw, is_del, is_cat, is_hit)

    print(f"[{level}] nearest-neighbour similarity", flush=True)
    metrics["nn"] = {}
    # Tanimoto between small objects is not on the same scale as Tanimoto
    # between whole compounds, so a cross-level comparison of the cross-library
    # numbers alone would mostly measure object size. Each level therefore also
    # gets a within-library reference: one half of a library queried against the
    # other. That is the similarity this object size produces when the two sides
    # are the same chemistry, and the gap to it is what actually travels across
    # levels.
    del_half = np.flatnonzero(is_del)
    cat_half = np.flatnonzero(is_cat)
    del_a, del_b = del_half[: len(del_half) // 2], del_half[len(del_half) // 2:]
    cat_a, cat_b = cat_half[: len(cat_half) // 2], cat_half[len(cat_half) // 2:]
    for key, q, r in [("del_to_enamine", fps[is_del], fps[is_cat]),
                      ("hits_to_enamine", fps[is_hit], fps[is_cat]),
                      ("enamine_to_del", fps[is_cat], fps[is_del]),
                      ("del_to_del", fps[del_a], fps[del_b]),
                      ("enamine_to_enamine", fps[cat_a], fps[cat_b]),
                      ("hits_to_del", fps[is_hit], fps[is_del])]:
        values = nearest_tanimoto(q, r, args.n_queries, rng)
        metrics["nn"][key] = {
            "median": round(float(np.median(values)), 4) if len(values) else None,
            "mean": round(float(values.mean()), 4) if len(values) else None,
            "frac_above_0.4": round(float((values > 0.4).mean()), 4) if len(values) else None,
            "frac_above_0.6": round(float((values > 0.6).mean()), 4) if len(values) else None,
            "frac_above_0.8": round(float((values > 0.8).mean()), 4) if len(values) else None,
            "n_queries": int(len(values)),
            "values": [round(float(v), 4) for v in values],
        }

    print(f"[{level}] diversity and separability", flush=True)
    metrics["diversity"] = {
        name: {"mean_pairwise_distance": round(internal_diversity(fps[mask], args.n_diversity, rng), 4),
               **scaffold_stats([smiles[i] for i in np.flatnonzero(mask)], is_unit,
                                args.n_scaffolds, args.workers, rng)}
        for name, mask in [("DEL", is_del), ("Enamine", is_cat), ("hits", is_hit)]
    }
    metrics["domain_auc"] = domain_auc(fps[is_del], fps[is_cat], args.n_classifier, rng)
    metrics["hits_vs_enamine_auc"] = domain_auc(fps[is_hit], fps[is_cat], args.n_classifier, rng)

    print(f"[{level}] projecting {len(fps):,} points", flush=True)
    projected = project(fps, desc, groups, args)
    frame = pd.DataFrame({"smiles": smiles, "group": groups})
    frame["pca_1"], frame["pca_2"] = projected["pca"][:, 0], projected["pca"][:, 1]
    if "dpca" in projected:
        frame["dpca_1"], frame["dpca_2"] = projected["dpca"][:, 0], projected["dpca"][:, 1]
        metrics["dpca_variance"] = projected["dpca_variance"]
    if "mds" in projected:
        frame["mds_1"], frame["mds_2"] = projected["mds"][:, 0], projected["mds"][:, 1]
        metrics["mds_variance"] = projected["mds_variance"]
        metrics["mds_landmarks"] = projected["mds_landmarks"]
    frame["umap_1"], frame["umap_2"] = projected["umap"][:, 0], projected["umap"][:, 1]
    if "tsne" in projected:
        frame["tsne_1"] = np.nan
        frame["tsne_2"] = np.nan
        frame.loc[projected["tsne_index"], "tsne_1"] = projected["tsne"][:, 0]
        frame.loc[projected["tsne_index"], "tsne_2"] = projected["tsne"][:, 1]
    metrics["pca_variance"] = projected["pca_variance"]
    metrics["pca_label"] = projected.get("pca_label", "PCA")
    if "descriptor_loadings" in projected:
        metrics["descriptor_loadings"] = {k: {n: round(float(w), 3) for n, w in v.items()}
                                          for k, v in projected["descriptor_loadings"].items()}

    frame.to_parquet(args.out_dir / f"{level}_domain.parquet", index=False)
    plot_level(frame, level, metrics, args.out_dir / f"{level}_domain.png", hit_label)
    metrics["elapsed_s"] = round(time.time() - start, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-work", type=Path, default=PROJECT / "work_full")
    parser.add_argument("--catalogue-work", type=Path, default=PROJECT / "work_catalogue")
    parser.add_argument("--del-results", type=Path, default=PROJECT / "results_binder_ratio")
    parser.add_argument("--del-input", type=Path, default=PROJECT / "data/raw_HGODEL_labeled/sharded")
    parser.add_argument("--catalogue-input", type=Path, default=PROJECT / "data/Enamine_ll/sharded")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results_domain")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--n-background", type=int, default=500_000,
                        help="units sampled per library per level (default: %(default)s)")
    parser.add_argument("--max-binders", type=int, default=0,
                        help="cap on DEL binders drawn; 0 keeps every one (default: %(default)s)")
    parser.add_argument("--n-files", type=int, default=120,
                        help="catalogue files a compound or pair sample is drawn from")
    parser.add_argument("--hit-selection", choices=("none", "leader"), default="leader",
                        help="thin the hits by greedy Tanimoto sphere exclusion before "
                             "measuring and plotting them (default: %(default)s)")
    parser.add_argument("--hit-selection-levels", default="compounds",
                        help="levels the thinning applies to. Only the binders need it: there "
                             "are 1.29M of them and they are largely one chemotype enumerated "
                             "across a sub-library. The enriched fragments and pairs are a few "
                             "thousand distinct units already, so thinning them would throw away "
                             "signal rather than redundancy (default: %(default)s)")
    parser.add_argument("--hit-threshold", type=float, default=0.7,
                        help="similarity above which two hits count as the same chemistry")
    parser.add_argument("--max-hits-selected", type=int, default=60_000,
                        help="stop selecting after this many; 0 for no cap")
    parser.add_argument("--n-queries", type=int, default=3000,
                        help="queries per nearest-neighbour distribution")
    parser.add_argument("--n-diversity", type=int, default=4000,
                        help="units per group for mean pairwise distance")
    parser.add_argument("--n-scaffolds", type=int, default=50_000,
                        help="units per group for the Murcko scaffold count")
    parser.add_argument("--n-classifier", type=int, default=60_000,
                        help="units per class for the separability AUC")
    parser.add_argument("--pca-space", choices=("ecfp", "descriptors", "both"), default="ecfp",
                        help="what the linear projection runs on; the question here is "
                             "structural, so the default is the bits (default: %(default)s)")
    parser.add_argument("--mds", action="store_true", default=False,
                        help="add classical MDS on Tanimoto distance via landmarks. Off by "
                             "default: measured on this data it lands on essentially the ECFP4 "
                             "PCA, up to a reflection, because Tanimoto between sparse bit "
                             "vectors of similar weight is close to a monotone function of "
                             "their Euclidean distance - and classical MDS on Euclidean "
                             "distance *is* PCA. It is kept because it is better conditioned "
                             "(Procrustes disparity 0.005 between independent landmark sets, "
                             "against a 47 degree plane rotation for the PCA), so it is the "
                             "one to reach for if the PCA ever has to be trusted in detail.")
    parser.add_argument("--mds-landmarks", type=int, default=2000,
                        help="landmarks embedded exactly before the rest are placed")
    parser.add_argument("--bits", type=int, default=1024)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--tsne", action="store_true",
                        help="add a t-SNE panel on a subsample; UMAP already answers the "
                             "question and t-SNE's between-cluster distances do not mean "
                             "anything, so this is a cross-check, not a measurement")
    parser.add_argument("--tsne-n", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from the saved coordinates and metrics, "
                             "recomputing nothing")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.replot:
        metrics = json.loads((args.out_dir / "domain_metrics.json").read_text())
        for level in args.levels.split(","):
            path = args.out_dir / f"{level}_domain.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if not args.mds:
                frame = frame.drop(columns=[c for c in ("mds_1", "mds_2") if c in frame.columns])
            if args.pca_space != "both":
                frame = frame.drop(columns=[c for c in ("dpca_1", "dpca_2") if c in frame.columns])
            plot_level(frame, level, metrics[level], args.out_dir / f"{level}_domain.png",
                       metrics[level].get("hit_label", "DEL hits"))
            print(f"replotted {level}")
        if all(level in metrics for level in LEVELS):
            plot_summary(metrics, args.out_dir / "domain_summary.png")
        return

    # Re-running one level must not delete the others' results.
    metrics_path = args.out_dir / "domain_metrics.json"
    out = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    out.update({"n_background": args.n_background, "bits": args.bits, "seed": args.seed,
                "max_binders": args.max_binders, "tsne": args.tsne})
    for level in args.levels.split(","):
        out[level] = run_level(level, args, rng)
        # Written after each level, so a long run is readable while it continues.
        metrics_path.write_text(json.dumps(out, indent=2))
        summary = {k: v for k, v in out[level].items() if k != "nn"}
        print(f"[{level}] {json.dumps(summary)[:600]}", flush=True)

    if all(level in out for level in LEVELS):
        plot_summary(out, args.out_dir / "domain_summary.png")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
