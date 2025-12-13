"""Cotree data structure and operations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True, slots=True)
class Cotree:
    """
    Recursive cotree representation of a cograph.

    A cotree is either:
    - A single vertex (op="vertex", children=())
    - A disjoint union of cotrees (op="sum", children=(g1, g2, ...))
    - A complete join of cotrees (op="product", children=(g1, g2, ...))

    Properties n, edges, and profile are computed lazily and cached.
    """
    op: Literal["vertex", "sum", "product"]
    children: tuple[Cotree, ...]
    # Cached computed values
    _n: int | None = None
    _edges: int | None = None
    _profile: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.op == "vertex":
            assert len(self.children) == 0
        else:
            assert len(self.children) >= 2

    @property
    def n(self) -> int:
        """Number of vertices."""
        if self._n is not None:
            return self._n
        if self.op == "vertex":
            result = 1
        else:
            result = sum(c.n for c in self.children)
        object.__setattr__(self, "_n", result)
        return result

    @property
    def edges(self) -> int:
        """Number of edges."""
        if self._edges is not None:
            return self._edges
        if self.op == "vertex":
            result = 0
        elif self.op == "sum":
            result = sum(c.edges for c in self.children)
        else:  # product
            child_edges = sum(c.edges for c in self.children)
            # Add all edges between components
            cross_edges = 0
            child_ns = [c.n for c in self.children]
            for i in range(len(child_ns)):
                for j in range(i + 1, len(child_ns)):
                    cross_edges += child_ns[i] * child_ns[j]
            result = child_edges + cross_edges
        object.__setattr__(self, "_edges", result)
        return result

    @property
    def profile(self) -> tuple[int, ...]:
        """K_{i,j} profile: profile[i] = max j such that K_{i,j} is a subgraph."""
        if self._profile is not None:
            return self._profile
        from .profile import compute_profile
        result = compute_profile(self)
        object.__setattr__(self, "_profile", result)
        return result

    def structure_str(self, use_n: bool = True) -> str:
        """
        Human-readable structure string.

        Args:
            use_n: If True, show vertex count for leaves. If False, show "1".
        """
        if self.op == "vertex":
            return str(self.n) if use_n else "1"
        op_char = "S" if self.op == "sum" else "P"
        children_str = ",".join(c.structure_str(use_n) for c in self.children)
        return f"{op_char}({children_str})"

    def __repr__(self) -> str:
        return f"Cotree({self.structure_str()}, n={self.n}, edges={self.edges})"


def vertex() -> Cotree:
    """Create a single vertex cotree."""
    return Cotree(op="vertex", children=())


def sum_graphs(g1: Cotree, g2: Cotree) -> Cotree:
    """
    Create disjoint union (sum) of two cographs.

    The result has no edges between vertices of g1 and g2.
    """
    return Cotree(op="sum", children=(g1, g2))


def product_graphs(g1: Cotree, g2: Cotree) -> Cotree:
    """
    Create complete join (product) of two cographs.

    Every vertex of g1 is connected to every vertex of g2.
    """
    return Cotree(op="product", children=(g1, g2))


def to_adjacency(cotree: Cotree) -> np.ndarray:
    """
    Convert cotree to adjacency matrix.

    Returns:
        n x n symmetric 0-1 matrix where A[i,j] = 1 iff vertices i and j are adjacent.
    """
    n = cotree.n
    adj = np.zeros((n, n), dtype=np.int8)
    _fill_adjacency(cotree, adj, 0)
    return adj


def _fill_adjacency(cotree: Cotree, adj: np.ndarray, offset: int) -> int:
    """
    Recursively fill adjacency matrix.

    Args:
        cotree: The cotree to process
        adj: The adjacency matrix to fill
        offset: Starting vertex index for this subtree

    Returns:
        The next available vertex index after this subtree
    """
    if cotree.op == "vertex":
        return offset + 1

    # Process children and track their ranges
    child_ranges: list[tuple[int, int]] = []
    current = offset
    for child in cotree.children:
        start = current
        current = _fill_adjacency(child, adj, current)
        child_ranges.append((start, current))

    # For product, add edges between all pairs of children
    if cotree.op == "product":
        for i, (s1, e1) in enumerate(child_ranges):
            for j, (s2, e2) in enumerate(child_ranges):
                if i < j:
                    adj[s1:e1, s2:e2] = 1
                    adj[s2:e2, s1:e1] = 1

    return current


def to_graph6(cotree: Cotree) -> str:
    """
    Convert cotree to graph6 format string.

    graph6 is a compact ASCII representation of undirected graphs.
    See: http://users.cecs.anu.edu.au/~bdm/data/formats.txt
    """
    adj = to_adjacency(cotree)
    n = cotree.n

    # Encode n
    if n <= 62:
        n_bytes = bytes([n + 63])
    elif n <= 258047:
        n_bytes = bytes([126, (n >> 12) + 63, ((n >> 6) & 63) + 63, (n & 63) + 63])
    else:
        raise ValueError(f"Graph too large for graph6: {n} vertices")

    # Encode adjacency upper triangle
    bits: list[int] = []
    for j in range(1, n):
        for i in range(j):
            bits.append(int(adj[i, j]))

    # Pad to multiple of 6
    while len(bits) % 6 != 0:
        bits.append(0)

    # Convert to bytes
    adj_bytes: list[int] = []
    for k in range(0, len(bits), 6):
        val = 0
        for b in range(6):
            val = (val << 1) | bits[k + b]
        adj_bytes.append(val + 63)

    return n_bytes.decode("ascii") + bytes(adj_bytes).decode("ascii")


def from_graph6(s: str) -> np.ndarray:
    """
    Parse graph6 string to adjacency matrix.

    Args:
        s: graph6 format string

    Returns:
        Adjacency matrix
    """
    data = [ord(c) - 63 for c in s]
    idx = 0

    # Decode n
    if data[0] != 63:
        n = data[0]
        idx = 1
    elif data[1] != 63:
        n = (data[1] << 12) | (data[2] << 6) | data[3]
        idx = 4
    else:
        raise ValueError("Graph too large")

    # Decode adjacency
    bits: list[int] = []
    for byte in data[idx:]:
        for b in range(5, -1, -1):
            bits.append((byte >> b) & 1)

    adj = np.zeros((n, n), dtype=np.int8)
    bit_idx = 0
    for j in range(1, n):
        for i in range(j):
            if bit_idx < len(bits):
                adj[i, j] = bits[bit_idx]
                adj[j, i] = bits[bit_idx]
                bit_idx += 1

    return adj
