"""Enumeration of contiguous fragment combinations.

A compound's fragments form a graph whose edges are the BRICS bonds that were
cut.  A combination is "contiguous" exactly when the fragments it names induce a
connected subgraph, so for the usual size of 2 the combinations are simply the
edges: fragments A-B-C-D in a chain give AB, BC and CD, while a B that carries
both C and D gives AB, BC and BD.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

DEFAULT_SIZES = (2,)


def _adjacency(n_nodes: int, edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for node_a, node_b in edges:
        if node_a != node_b:
            adj[node_a].add(node_b)
            adj[node_b].add(node_a)
    return adj


def connected_subsets(
    n_nodes: int, edges: Sequence[tuple[int, int]], size: int
) -> list[tuple[int, ...]]:
    """All sets of ``size`` nodes that induce a connected subgraph."""
    if size <= 0 or size > n_nodes:
        return []
    if size == 1:
        return [(node,) for node in range(n_nodes)]
    if size == 2:
        return sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})

    adj = _adjacency(n_nodes, edges)
    seen: set[frozenset[int]] = set()
    found: list[tuple[int, ...]] = []
    for start in range(n_nodes):
        # Only grow with nodes above `start`, so each subset is reached once
        # from its lowest-numbered member.
        stack = [(frozenset((start,)), {n for n in adj[start] if n > start})]
        while stack:
            nodes, frontier = stack.pop()
            if len(nodes) == size:
                if nodes not in seen:
                    seen.add(nodes)
                    found.append(tuple(sorted(nodes)))
                continue
            for node in frontier:
                grown = nodes | {node}
                extra = {n for n in adj[node] if n > start and n not in grown}
                stack.append((grown, (frontier | extra) - grown - {node}))
    found.sort()
    return found


def combinations_for_compound(
    frag_ids: Sequence[int],
    edges: Sequence[tuple[int, int]],
    sizes: Iterable[int] = DEFAULT_SIZES,
) -> list[tuple[int, ...]]:
    """Contiguous combinations of a compound, as sorted tuples of fragment ids.

    ``frag_ids`` maps a compound-local fragment position to its global id, so a
    compound carrying the same fragment twice yields the combination once.
    """
    out: set[tuple[int, ...]] = set()
    for size in sizes:
        for subset in connected_subsets(len(frag_ids), edges, size):
            out.add(tuple(sorted(frag_ids[pos] for pos in subset)))
    return sorted(out)


def combo_columns(size: int) -> list[str]:
    """Column names holding a combination's member fragment ids."""
    return [f"frag_{i}" for i in range(size)]


def combo_name(frag_ids: Sequence[int], name_of: dict[int, str]) -> str:
    """Human-readable key for a combination, e.g. ``F000012|F000345``.

    Fragment ids are 64-bit content hashes, so they are stable but unreadable;
    the short names come from the dictionary and are only worth resolving for
    the handful of combinations that reach a report.
    """
    return "|".join(name_of.get(fid, f"?{fid:016x}") for fid in frag_ids)
