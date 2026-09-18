"""Normalising fragments so that near-identical ones share a key.

Two combinations that differ only by a fluorine or a methyl are almost always
the same structural signal measured twice, and splitting their compounds across
two rows costs statistical power.  Grouping them back together needs a rule for
"same scaffold".

The rule here is a canonical key rather than a distance threshold.  Keys are
O(1) to compare, need no all-against-all similarity matrix (hopeless once a
library yields millions of combinations), and give reproducible groups that do
not depend on the order the clustering ran in.

Normalisation happens per *fragment*, not per merged combination: combinations
are already keyed by their member fragments, so normalising the members and
re-pairing them clusters combinations without re-fragmenting anything.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

MODES = ("exact", "substituent", "murcko", "generic")
DEFAULT_MODE = "substituent"
# Removing more than this many atoms from one fragment stops being a decoration
# change, so such a fragment keeps its exact identity instead.
DEFAULT_MAX_STRIP = 2

_STRIPPABLE = {6, 9, 17, 35, 53}  # carbon (methyl) and the halogens


def _prepared(frag_smiles: str) -> Chem.Mol | None:
    """Parse a fragment and drop the BRICS isotope labels from its dummies.

    The labels record which bond type the fragment was cut at.  Two otherwise
    identical scaffolds cut at different bond types should still cluster
    together, so the labels go but the attachment points stay.
    """
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return editable.GetMol()


def _strip_decorations(mol: Chem.Mol) -> tuple[Chem.Mol, int]:
    """Remove terminal methyls and halogens, returning the mol and atoms removed.

    Only singly-bonded terminal carbons and halogens go.  Heteroatom terminals
    stay - dropping the NH2 of a primary sulfonamide, say, would throw away the
    very group that binds.  Atoms attached to a stereocentre stay too, since
    removing them would silently erase the stereochemistry.
    """
    editable = Chem.RWMol(mol)
    doomed = []
    for atom in editable.GetAtoms():
        if atom.GetDegree() != 1 or atom.GetAtomicNum() not in _STRIPPABLE:
            continue
        if atom.IsInRing():
            continue
        bond = atom.GetBonds()[0]
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        neighbour = bond.GetOtherAtom(atom)
        if neighbour.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        doomed.append(atom.GetIdx())

    for idx in sorted(doomed, reverse=True):
        editable.RemoveAtom(idx)
    return editable.GetMol(), len(doomed)


def scaffold_key(
    frag_smiles: str, mode: str = DEFAULT_MODE, max_strip: int = DEFAULT_MAX_STRIP
) -> str:
    """The key a fragment clusters under.  Falls back to its own SMILES."""
    if mode == "exact":
        return frag_smiles
    mol = _prepared(frag_smiles)
    if mol is None:
        return frag_smiles

    try:
        if mode == "substituent":
            stripped, removed = _strip_decorations(mol)
            if removed > max_strip:
                return frag_smiles
            return Chem.MolToSmiles(stripped)
        if mode == "murcko":
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        if mode == "generic":
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
    except Exception:
        return frag_smiles
    raise ValueError(f"unknown normalisation mode {mode!r}; pick one of {MODES}")


def scaffold_keys(
    frag_smiles: list[str], mode: str = DEFAULT_MODE, max_strip: int = DEFAULT_MAX_STRIP
) -> list[str]:
    return [scaffold_key(smi, mode, max_strip) for smi in frag_smiles]
