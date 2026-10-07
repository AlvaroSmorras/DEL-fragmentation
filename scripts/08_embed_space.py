#!/usr/bin/env python
"""Stage 8 - map DEL and catalogue fragments/combinations into one chemical space.

Featurises fragments and size-2 combinations with ECFP4 (Morgan radius 2,
1024 bits), projects them with UMAP (Jaccard on the bits) and PCA, and overlays
the DEL-enriched ones, marked by whether the catalogue carries them.

What goes into each map:

* **background** - a uniform random sample of *distinct* fragments (or
  combinations) from each library.  Fragments come straight from each run's
  dictionary.  Combinations come from a random sample of compounds (spread over
  many input files), whose contiguous pairs are enumerated and de-duplicated:
  the catalogue run only stored the DEL target pairs, so its full combination
  space has to be sampled from ``compound_fragments/``.
* **enriched** - every DEL fragment / combination flagged ``enriched`` by stage
  7 (``results_full/catalogue/*_in_catalogue.parquet``), so run stage 7 first
  (``--no-compounds`` is enough).

Attachment points stay in the molecule as plain dummy atoms: they mark where
the fragment was bonded, which is chemically meaningful, but their BRICS label
numbers would otherwise split identical rings into different bits.  A
combination is rebuilt as one molecule by re-bonding its two fragments through
a BRICS-compatible pair of attachment points.

PCA runs on physicochemical descriptors by default, not on the fingerprint.
Measured on this data, an ECFP4 PCA puts only 6-13% of the variance in its first
two components and its PC1-PC2 plane lands 41-57 degrees away when refitted on a
disjoint sample of the same size - at every size up to 200k, since the
eigenvalues are near-degenerate rather than the sample being small.  So those
axes are close to an arbitrary rotation and should not be read.  The same
molecules through eleven standardised descriptors give 59-75% in two components
and a plane stable to 11-25 degrees.  ``--pca-space ecfp`` restores the old
behaviour; UMAP uses ECFP4 either way, which is where fingerprint similarity
belongs.

Sample size is a separate question from variance.  The background sample is what
decides coverage: a held-out Enamine fragment's nearest neighbour in a 50k
background has median Tanimoto 0.46, rising to 0.55 at 200k and still climbing,
so the default background is 500k per library.  Enriched units are never
sampled - they are all drawn - so the overlay is unaffected either way.

PCA is fitted on the background only, so the enriched points do not steer the
axes.  UMAP is fitted on everything at once.

Needs ``umap-learn`` (the ``fragviz`` conda env has it).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors as rdMD

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.combinations import connected_subsets  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DEFAULT_BITS = 1024
_GEN = None

# BRICS label pair -> bond type of the bond they came from.
_BRICS_BOND: dict[frozenset[int], Chem.BondType] = {}
for _group in BRICS.reactionDefs:
    for _a, _b, _bt in _group:
        _key = frozenset((int(re.match(r"\d+", _a).group()), int(re.match(r"\d+", _b).group())))
        _BRICS_BOND[_key] = Chem.BondType.DOUBLE if _bt == "=" else Chem.BondType.SINGLE


# ---------------------------------------------------------------- chemistry --

def join_fragments(smiles_a: str, smiles_b: str) -> Chem.Mol | None:
    """Bond two BRICS fragments through a compatible pair of attachment points."""
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    n_a = mol_a.GetNumAtoms()
    mol = Chem.RWMol(Chem.CombineMols(mol_a, mol_b))
    dummies_a = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIdx() < n_a]
    dummies_b = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIdx() >= n_a]
    if not dummies_a or not dummies_b:
        return None
    pairs = [(da, db) for da in dummies_a for db in dummies_b]
    # Several dummies can be compatible; any choice gives the same pair of
    # fragments, so take the first deterministic one.
    chosen = next(
        ((da, db) for da, db in pairs if frozenset((da.GetIsotope(), db.GetIsotope())) in _BRICS_BOND),
        pairs[0],
    )
    da, db = chosen
    bond_type = _BRICS_BOND.get(frozenset((da.GetIsotope(), db.GetIsotope())), Chem.BondType.SINGLE)
    na, nb = da.GetNeighbors()[0].GetIdx(), db.GetNeighbors()[0].GetIdx()
    mol.AddBond(na, nb, bond_type)
    for idx in sorted((da.GetIdx(), db.GetIdx()), reverse=True):
        mol.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol.GetMol()


def _prepare(smiles: str) -> Chem.Mol | None:
    """Fragment or ``A.B`` combination SMILES -> one molecule, dummy labels cleared."""
    parts = smiles.split(".")
    mol = Chem.MolFromSmiles(smiles) if len(parts) == 1 else join_fragments(parts[0], parts[1])
    if mol is None:
        return None
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return mol.GetMol()


def _descriptors(mol: Chem.Mol) -> list[float] | None:
    """The physicochemical axes the default PCA runs on.

    Unlike the fingerprint these are few, continuous and correlated, which is
    exactly why PCA behaves on them: measured on this data two components hold
    59-75% of the variance and the plane reproduces to 11-25 degrees between
    disjoint samples, against 6-13% and 41-57 degrees for ECFP4.
    """
    try:
        return [
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMD.CalcTPSA(mol),
            rdMD.CalcNumHBD(mol), rdMD.CalcNumHBA(mol), rdMD.CalcNumRotatableBonds(mol),
            rdMD.CalcNumRings(mol), rdMD.CalcNumAromaticRings(mol),
            rdMD.CalcFractionCSP3(mol), float(mol.GetNumHeavyAtoms()),
            float(rdMD.CalcNumHeteroatoms(mol)),
        ]
    except Exception:
        return None


DESCRIPTOR_NAMES = ["MolWt", "cLogP", "TPSA", "HBD", "HBA", "RotBonds",
                    "Rings", "AromaticRings", "Fsp3", "HeavyAtoms", "Heteroatoms"]


def _init_worker(bits: int) -> None:
    global _GEN
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)


def _featurise(smiles: str):
    mol = _prepare(smiles)
    if mol is None:
        return None
    descriptors = _descriptors(mol)
    if descriptors is None or not np.isfinite(descriptors).all():
        return None
    return _GEN.GetFingerprintAsNumPy(mol).astype(bool), descriptors, Chem.MolToSmiles(mol)


def featurise(smiles: list[str], workers: int, bits: int):
    """Fingerprints, descriptors and canonical SMILES in one pass over the molecules."""
    with Pool(workers, initializer=_init_worker, initargs=(bits,)) as pool:
        out = pool.map(_featurise, smiles, chunksize=500)
    ok = np.array([item is not None for item in out])
    fps = np.zeros((len(out), bits), dtype=bool)
    desc = np.zeros((len(out), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    canonical: list[str | None] = []
    for i, item in enumerate(out):
        if item is None:
            canonical.append(None)
            continue
        fps[i], desc[i], smi = item
        canonical.append(smi)
    return fps, desc, ok, canonical


# ----------------------------------------------------------------- sampling --

def _dictionary(work_dir: Path, ids: np.ndarray | None = None) -> pd.DataFrame:
    table = pq.read_table(work_dir / "fragments.parquet", columns=["frag_id", "frag_smiles"])
    if ids is not None:
        table = table.filter(pc.is_in(table["frag_id"], value_set=pa.array(ids, pa.int64())))
    return table.to_pandas()


def sample_fragments(work_dir: Path, n: int, rng: np.random.Generator) -> pd.DataFrame:
    frags = _dictionary(work_dir)
    take = rng.choice(len(frags), size=min(n, len(frags)), replace=False)
    return frags.iloc[np.sort(take)].reset_index(drop=True)


def sample_combinations(work_dir: Path, n: int, n_files: int, rng: np.random.Generator) -> pd.DataFrame:
    """Distinct contiguous pairs from compounds drawn across many input files."""
    files = sorted((work_dir / "compound_fragments").glob("*.parquet"))
    files = [files[i] for i in rng.choice(len(files), size=min(n_files, len(files)), replace=False)]
    per_file = max(1, int(np.ceil(2 * n / len(files))))
    pairs: set[tuple[int, int]] = set()
    for path in files:
        table = pq.read_table(path, columns=["frag_ids", "edge_src", "edge_dst"])
        rows = rng.choice(table.num_rows, size=min(per_file, table.num_rows), replace=False)
        sub = table.take(pa.array(np.sort(rows))).to_pylist()
        for rec in sub:
            ids = rec["frag_ids"]
            edges = list(zip(rec["edge_src"], rec["edge_dst"]))
            for i, j in connected_subsets(len(ids), edges, 2):
                pairs.add(tuple(sorted((ids[i], ids[j]))))
    pairs_arr = np.array(sorted(pairs), dtype=np.int64)
    take = rng.choice(len(pairs_arr), size=min(n, len(pairs_arr)), replace=False)
    out = pd.DataFrame(pairs_arr[np.sort(take)], columns=["frag_0", "frag_1"])
    smiles = _dictionary(work_dir, np.unique(out[["frag_0", "frag_1"]].to_numpy()))
    smiles_of = dict(zip(smiles["frag_id"], smiles["frag_smiles"]))
    out["smiles"] = [f"{smiles_of[a]}.{smiles_of[b]}" for a, b in zip(out["frag_0"], out["frag_1"])]
    return out


# ------------------------------------------------------------------ embedding --

def embed(points: pd.DataFrame, fps: np.ndarray, desc: np.ndarray, args) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    background = points["group"].isin(["DEL", "Enamine"]).to_numpy()
    if args.pca_space == "descriptors":
        # Standardised, or MolWt's hundreds would swamp Fsp3's 0-1.
        scaler = StandardScaler().fit(desc[background])
        matrix = scaler.transform(desc)
    else:
        matrix = fps.astype(np.float32)
    pca = PCA(n_components=2, random_state=args.seed).fit(matrix[background])
    xy = pca.transform(matrix)
    points["pca_1"], points["pca_2"] = xy[:, 0], xy[:, 1]
    points.attrs["pca_variance"] = pca.explained_variance_ratio_.tolist()
    points.attrs["pca_space"] = args.pca_space
    if args.pca_space == "descriptors":
        # What the axes mean, which is half the reason to use descriptors.
        points.attrs["pca_loadings"] = {
            f"PC{i + 1}": {name: round(float(weight), 3)
                           for name, weight in sorted(zip(DESCRIPTOR_NAMES, component),
                                                      key=lambda kv: -abs(kv[1]))[:5]}
            for i, component in enumerate(pca.components_)
        }

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="jaccard",
                        n_jobs=args.workers, low_memory=True, verbose=True)
    uv = reducer.fit_transform(fps)
    points["umap_1"], points["umap_2"] = uv[:, 0], uv[:, 1]
    return points


# --------------------------------------------------------------------- plots --

# Reference categorical slots 1-3 (validated all-pairs for scatter, light mode).
COLORS = {"DEL": "#2a78d6", "Enamine": "#eb6834", "found": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def label_of(method: str, pca_name: str) -> str:
    return pca_name if method == "pca" else "UMAP"


def title_of(lo_col: str) -> str:
    return "binders per non-binder" if lo_col == "binder_per_nonbinder" else "enrichment lower bound"


def plot(points: pd.DataFrame, kind: str, out: Path, lo_col: str, seed: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    var = points.attrs.get("pca_variance", [np.nan, np.nan])
    space = points.attrs.get("pca_space", "ecfp")
    pca_name = "descriptor PCA" if space == "descriptors" else "ECFP4 PCA"
    # Interleave the two libraries in z. One scatter call draws in row order, so
    # without this the library concatenated last would sit on top everywhere and
    # whichever that is would look like the denser one.
    bg = points[points["group"].isin(["DEL", "Enamine"])].sample(frac=1.0, random_state=seed)
    # A million points at one size is a solid blob; shrink the mark as it grows.
    bg_size = float(np.clip(750_000 / max(len(bg), 1), 0.45, 1.8))
    bg_alpha = float(np.clip(120 / np.sqrt(max(len(bg), 1)), 0.12, 0.40))
    enr = points[points["group"] == "enriched"]
    found, absent = enr[enr["in_catalogue"]], enr[~enr["in_catalogue"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), facecolor=SURFACE)
    for row, (method, labels) in enumerate([
        ("pca", (f"{pca_name} PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")),
        ("umap", ("UMAP 1", "UMAP 2")),
    ]):
        x, y = f"{method}_1", f"{method}_2"
        ax_lib, ax_hit, ax_strength = axes[row]

        for name in ("DEL", "Enamine"):
            sel = bg[bg["group"] == name]
            ax_lib.scatter([], [], s=30, color=COLORS[name], label=f"{name} ({len(sel):,} sampled)")
        ax_lib.scatter(bg[x], bg[y], s=bg_size, c=bg["group"].map(COLORS), alpha=bg_alpha,
                       linewidths=0, rasterized=True)
        ax_lib.set_title(f"{label_of(method, pca_name)} - DEL vs Enamine {kind}", loc="left")

        ax_hit.scatter(bg[x], bg[y], s=bg_size, color="#c9c8c2", alpha=bg_alpha, linewidths=0,
                       rasterized=True, label="background (both libraries)")
        ax_hit.scatter(absent[x], absent[y], s=10, facecolors="none", edgecolors=INK_2, linewidths=0.4, alpha=0.6,
                       label=f"enriched, not in Enamine ({len(absent):,})", rasterized=True)
        ax_hit.scatter(found[x], found[y], s=12, color=COLORS["found"], edgecolors=SURFACE, linewidths=0.4,
                       label=f"enriched, in Enamine ({len(found):,})", rasterized=True)
        ax_hit.set_title(f"{label_of(method, pca_name)} - DEL-enriched {kind}", loc="left")

        ax_strength.scatter(bg[x], bg[y], s=bg_size, color="#c9c8c2", alpha=bg_alpha, linewidths=0,
                            rasterized=True)
        ordered = enr.sort_values(lo_col)
        # A pair with no non-binders has an infinite ratio; it colours as the
        # strongest finite one rather than breaking the log scale.
        finite = ordered[lo_col][np.isfinite(ordered[lo_col]) & (ordered[lo_col] > 0)]
        floor = finite.min() if len(finite) else 1.0
        ceiling = max(finite.max(), floor * 10) if len(finite) else floor * 10
        lo = ordered[lo_col].clip(lower=floor, upper=ceiling)
        sc = ax_strength.scatter(ordered[x], ordered[y], s=10, c=lo, cmap="Blues",
                                 norm=LogNorm(vmin=floor, vmax=ceiling),
                                 edgecolors=INK_2, linewidths=0.2, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax_strength, fraction=0.04, pad=0.01)
        cbar.set_label(f"{lo_col} (log)", color=INK_2)
        ax_strength.set_title(f"{label_of(method, pca_name)} - {title_of(lo_col)}", loc="left")

        for ax in (ax_lib, ax_hit, ax_strength):
            ax.set_xlabel(labels[0], color=INK_2)
            ax.set_ylabel(labels[1], color=INK_2)
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=INK_2, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.title.set_color(INK)
        for ax in (ax_lib, ax_hit):
            leg = ax.legend(loc="best", fontsize=8, frameon=True, markerscale=1.5)
            leg.get_frame().set_edgecolor(GRID)

    fig.suptitle(f"Chemical space of BRICS {kind}: DEL (HGODEL) and Enamine REAL lead-like "
                 f"({pca_name}, UMAP on ECFP4)",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- main --

def _rank_column(path: Path, default: str) -> str:
    """Colour by binders per non-binder when stage 7 ranked that way."""
    names = set(pq.ParquetFile(path).schema_arrow.names)
    return "binder_per_nonbinder" if "binder_per_nonbinder" in names else default


def run(kind: str, args, rng: np.random.Generator) -> dict:
    hits_dir = args.hits_dir or args.del_results / "catalogue"
    if kind == "fragments":
        lo_col = _rank_column(hits_dir / "fragment_in_catalogue.parquet", "enrichment_factor_lo95")
        table = pd.read_parquet(hits_dir / "fragment_in_catalogue.parquet",
                                columns=["frag_id", "frag_smiles", "enriched", "in_catalogue",
                                         lo_col, "n_catalogue"])
        enr = table[table["enriched"]].rename(columns={"frag_smiles": "smiles"})
        del_bg = sample_fragments(args.del_work, args.n_background, rng).rename(columns={"frag_smiles": "smiles"})
        cat_bg = sample_fragments(args.catalogue_work, args.n_background, rng).rename(
            columns={"frag_smiles": "smiles"})
        key = ["frag_id"]
    else:
        lo_col = _rank_column(hits_dir / "combination_in_catalogue.parquet", "enrichment_mh_lo95")
        table = pd.read_parquet(hits_dir / "combination_in_catalogue.parquet",
                                columns=["frag_0", "frag_1", "combo_key", "frag_smiles", "enriched",
                                         "in_catalogue", lo_col, "n_catalogue"])
        enr = table[table["enriched"]].rename(columns={"frag_smiles": "smiles"})
        del_bg = sample_combinations(args.del_work, args.n_background, args.n_files, rng)
        cat_bg = sample_combinations(args.catalogue_work, args.n_background, args.n_files, rng)
        key = ["frag_0", "frag_1"]

    points = pd.concat([
        del_bg.assign(group="DEL"),
        cat_bg.assign(group="Enamine"),
        enr.assign(group="enriched"),
    ], ignore_index=True)
    points["in_catalogue"] = points["in_catalogue"].astype("boolean")
    print(f"[{kind}] {len(del_bg):,} DEL + {len(cat_bg):,} Enamine background, {len(enr):,} enriched")

    fps, desc, ok, canonical = featurise(points["smiles"].tolist(), args.workers, args.fp_bits)
    points["mol_smiles"] = canonical
    points, fps, desc = points[ok].reset_index(drop=True), fps[ok], desc[ok]
    print(f"[{kind}] {int((~ok).sum())} could not be featurised; embedding {len(points):,}")

    points = embed(points, fps, desc, args)
    variance = points.attrs["pca_variance"]
    keep = key + [c for c in ("combo_key", "smiles", "mol_smiles", "group", "in_catalogue", lo_col,
                               "n_catalogue", "pca_1", "pca_2", "umap_1", "umap_2") if c in points.columns]
    points[keep].to_parquet(args.out_dir / f"{kind}_embedding.parquet", index=False)
    plot(points, kind, args.out_dir / f"{kind}_embedding.png", lo_col, seed=args.seed)
    return {"n_points": len(points), "n_failed": int((~ok).sum()), "pca_variance": variance,
            "pca_loadings": points.attrs.get("pca_loadings"),
            "n_enriched": int((points["group"] == "enriched").sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--del-work", type=Path, default=Path("work_full"))
    parser.add_argument("--catalogue-work", type=Path, default=Path("work_catalogue"))
    parser.add_argument("--del-results", type=Path, default=Path("results_full"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_full/embedding"))
    parser.add_argument("--hits-dir", type=Path, default=None,
                        help="stage 7 output (default: <del-results>/catalogue)")
    parser.add_argument("--kinds", default="fragments,combinations")
    parser.add_argument("--n-background", type=int, default=500_000,
                        help="distinct fragments/combinations sampled per library")
    parser.add_argument("--n-files", type=int, default=60,
                        help="input files the combination sample is drawn from, per library")
    parser.add_argument("--pca-space", choices=("descriptors", "ecfp"), default="descriptors",
                        help="what the PCA runs on; UMAP always uses ECFP4 (default: %(default)s)")
    parser.add_argument("--fp-bits", type=int, default=DEFAULT_BITS,
                        help="ECFP4 fingerprint length (default: %(default)s)")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = {"n_background": args.n_background, "n_bits": args.fp_bits, "radius": 2,
               "pca_space": args.pca_space, "descriptors": DESCRIPTOR_NAMES,
               "n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "seed": args.seed}
    for kind in args.kinds.split(","):
        start = time.time()
        summary[kind] = run(kind, args, rng)
        summary[kind]["elapsed_s"] = round(time.time() - start, 1)
        print(json.dumps(summary[kind]))
    (args.out_dir / "summary_embedding.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
