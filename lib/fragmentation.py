"""BRICS fragmentation of molecules under a minimum fragment-size constraint.

RDKit's BRICS implementation cuts every retrosynthetic bond it finds, which
leaves a lot of tiny linker fragments (``[5*]N[5*]`` and friends).  We want
fragments of at least ``min_hac`` heavy atoms, so cuts are *undone* until every
fragment is large enough.  The bonds that survive define both the fragments and
the fragment adjacency graph used downstream to build contiguous combinations.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

DEFAULT_MIN_HAC = 6

# Width of the content-derived fragment id.  A 64-bit digest collides with
# probability ~2e-6 across 9M fragments, which is the scale a 300M-compound
# library reaches.
_ID_BYTES = 8


def fragment_id(frag_smiles: str) -> int:
    """A stable id derived from the fragment itself, not from where it was seen.

    Numbering fragments in discovery order would mean a global pass over every
    compound before any id is known, and would renumber the whole dictionary
    whenever a library is added.  Hashing the canonical SMILES instead lets
    workers write final output immediately, and keeps ids comparable across runs
    and across libraries.
    """
    digest = hashlib.blake2b(frag_smiles.encode(), digest_size=_ID_BYTES).digest()
    return int.from_bytes(digest, "big", signed=True)


@dataclass(frozen=True)
class Fragmentation:
    """The BRICS decomposition of one molecule.

    ``edges`` holds pairs of indices into ``smiles``/``hac``; two fragments are
    joined by an edge when a broken BRICS bond connected them, i.e. when they
    are contiguous in the parent molecule.
    """

    smiles: tuple[str, ...]
    hac: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    edge_labels: tuple[tuple[int, int], ...]
    atoms: tuple[tuple[int, ...], ...] = ()
    cut_bonds: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.smiles)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _brics_bonds(mol: Chem.Mol) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    """BRICS bonds as ``(bond_idx, (atom_a, atom_b), (label_a, label_b))``.

    Each label describes the atom it sits next to, matching the convention of
    :func:`rdkit.Chem.BRICS.BreakBRICSBonds`.
    """
    bonds = []
    for (atom_a, atom_b), (label_a, label_b) in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(atom_a, atom_b)
        if bond is None:  # pragma: no cover - defensive
            continue
        bonds.append((bond.GetIdx(), (atom_a, atom_b), (int(label_a), int(label_b))))
    return bonds


def _components(n_atoms: int, bonds: Sequence[tuple[int, int, int]], cut: set[int]) -> list[int]:
    """Map every atom to the id of the fragment it lands in once ``cut`` is broken."""
    uf = _UnionFind(n_atoms)
    for bond_idx, atom_a, atom_b in bonds:
        if bond_idx not in cut:
            uf.union(atom_a, atom_b)
    return [uf.find(i) for i in range(n_atoms)]


def _select_cuts(mol: Chem.Mol, bonds, min_hac: int) -> set[int]:
    """Drop cuts until no fragment is smaller than ``min_hac`` heavy atoms.

    The smallest offending fragment is repeatedly merged into its smallest
    neighbour, which keeps the surviving fragments as evenly sized as possible
    instead of growing one blob.  Ties break on atom/bond index so the result is
    deterministic.
    """
    cut = {bond_idx for bond_idx, _, _ in bonds}
    bond_atoms = {bond_idx: atoms for bond_idx, atoms, _ in bonds}
    n_atoms = mol.GetNumAtoms()
    all_bonds = [
        (bond.GetIdx(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    ]

    while cut:
        root_of = _components(n_atoms, all_bonds, cut)
        size: dict[int, int] = defaultdict(int)
        for root in root_of:
            size[root] += 1

        # Cuts still separating two distinct fragments, grouped per fragment.
        touching: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for bond_idx in cut:
            atom_a, atom_b = bond_atoms[bond_idx]
            root_a, root_b = root_of[atom_a], root_of[atom_b]
            if root_a != root_b:
                touching[root_a].append((bond_idx, root_b))
                touching[root_b].append((bond_idx, root_a))

        offenders = [root for root, n in size.items() if n < min_hac and touching[root]]
        if not offenders:
            break

        target = min(offenders, key=lambda root: (size[root], root))
        bond_idx, _ = min(touching[target], key=lambda item: (size[item[1]], item[0]))
        cut.discard(bond_idx)

    # A cut whose ends fell back into one fragment would only open a ring.
    root_of = _components(n_atoms, all_bonds, cut)
    return {b for b in cut if root_of[bond_atoms[b][0]] != root_of[bond_atoms[b][1]]}


def fragment_mol(mol: Chem.Mol, min_hac: int = DEFAULT_MIN_HAC) -> Fragmentation:
    """Decompose ``mol`` into BRICS fragments of at least ``min_hac`` heavy atoms."""
    bonds = _brics_bonds(mol)
    cut = _select_cuts(mol, bonds, min_hac) if bonds else set()

    if not cut:
        smiles = Chem.MolToSmiles(mol)
        atoms = (tuple(range(mol.GetNumAtoms())),)
        return Fragmentation((smiles,), (mol.GetNumAtoms(),), (), (), atoms, ())

    kept = [entry for entry in bonds if entry[0] in cut]
    bond_indices, dummy_labels = [], []
    for bond_idx, (atom_a, atom_b), (label_a, label_b) in kept:
        bond = mol.GetBondWithIdx(bond_idx)
        # FragmentOnBonds labels the dummy *replacing* the begin atom first, and
        # that dummy hangs off the end atom - so the end atom's label leads.
        if bond.GetBeginAtomIdx() == atom_a:
            dummy_labels.append((label_b, label_a))
        else:
            dummy_labels.append((label_a, label_b))
        bond_indices.append(bond_idx)

    broken = Chem.FragmentOnBonds(mol, bond_indices, dummyLabels=dummy_labels)
    pieces = Chem.GetMolFrags(broken, asMols=False)

    n_atoms = mol.GetNumAtoms()
    frag_of_atom = [-1] * n_atoms
    hac, frag_atoms = [], []
    for frag_id, atoms in enumerate(pieces):
        # Dummies are appended after the original atoms, so anything below
        # n_atoms is a real atom of the parent molecule.
        own = tuple(atom for atom in atoms if atom < n_atoms)
        for atom in own:
            frag_of_atom[atom] = frag_id
        frag_atoms.append(own)
        hac.append(len(own))

    smiles = tuple(Chem.MolToSmiles(piece) for piece in Chem.GetMolFrags(broken, asMols=True))

    edges, edge_labels = [], []
    for _, (atom_a, atom_b), (label_a, label_b) in kept:
        frag_a, frag_b = frag_of_atom[atom_a], frag_of_atom[atom_b]
        if frag_a > frag_b:
            frag_a, frag_b = frag_b, frag_a
            label_a, label_b = label_b, label_a
        edges.append((frag_a, frag_b))
        edge_labels.append((label_a, label_b))

    return Fragmentation(
        smiles,
        tuple(hac),
        tuple(edges),
        tuple(edge_labels),
        tuple(frag_atoms),
        tuple(bond_indices),
    )


def fragment_smiles(smiles: str, min_hac: int = DEFAULT_MIN_HAC) -> Fragmentation | None:
    """Fragment a SMILES string, returning ``None`` when RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return fragment_mol(mol, min_hac=min_hac)


def merge_fragments(
    mol: Chem.Mol, fragmentation: Fragmentation, positions: Sequence[int]
) -> str:
    """SMILES of a contiguous group of fragments, as one connected molecule.

    Only the bonds leaving the group are cut, so the bonds *inside* it are
    restored and the result is what the combination actually looks like in the
    parent compound.
    """
    wanted = set(positions)
    atoms = {atom for pos in wanted for atom in fragmentation.atoms[pos]}
    if not atoms:
        raise ValueError("empty fragment selection")

    boundary, labels = [], []
    for bond_idx in fragmentation.cut_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (begin in atoms) == (end in atoms):
            continue
        boundary.append(bond_idx)
        labels.append(_brics_label_pair(mol, bond_idx))

    if not boundary:
        return Chem.MolToSmiles(mol)

    broken = Chem.FragmentOnBonds(mol, boundary, dummyLabels=labels)
    for piece_atoms, piece in zip(
        Chem.GetMolFrags(broken, asMols=False), Chem.GetMolFrags(broken, asMols=True)
    ):
        if atoms & set(piece_atoms):
            return Chem.MolToSmiles(piece)
    raise RuntimeError("fragment group not found after cutting")  # pragma: no cover


def _brics_label_pair(mol: Chem.Mol, bond_idx: int) -> tuple[int, int]:
    """BRICS dummy labels for one bond, ordered the way FragmentOnBonds wants."""
    bond = mol.GetBondWithIdx(bond_idx)
    for _, (atom_a, atom_b), (label_a, label_b) in _brics_bonds(mol):
        if {atom_a, atom_b} == {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}:
            if bond.GetBeginAtomIdx() == atom_a:
                return (label_b, label_a)
            return (label_a, label_b)
    return (0, 0)  # pragma: no cover - bond came from FindBRICSBonds
