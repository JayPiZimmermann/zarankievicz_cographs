"""Memory-efficient storage for extremal cographs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Iterator
from collections import defaultdict
import numpy as np
import pickle
from pathlib import Path


@dataclass(slots=True)
class CompactGraph:
    """
    Memory-efficient graph representation.

    Instead of full Cotree objects, store minimal construction info.
    Structure strings are computed lazily only when needed.
    """
    edges: int
    op: Literal["v", "s", "p"]  # vertex, sum, product
    # For non-vertex: indices into parent's graph lists
    child1_n: int = 0
    child1_profile_hash: int = 0
    child1_idx: int = 0
    child2_n: int = 0
    child2_profile_hash: int = 0
    child2_idx: int = 0

    def is_vertex(self) -> bool:
        return self.op == "v"


class FastRegistry:
    """
    High-performance registry using numpy arrays and hash-based lookup.

    Structure:
        _profiles[n] = dict: profile_hash -> profile_array
        _graphs[n] = dict: profile_hash -> list of CompactGraph
        _max_edges[n] = dict: profile_hash -> max edge count

    All data stays in RAM. Checkpoints use pickle for speed.
    """

    def __init__(self):
        # profile_hash -> numpy array
        self._profiles: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        # n -> profile_hash -> list of CompactGraph
        self._graphs: dict[int, dict[int, list[CompactGraph]]] = defaultdict(lambda: defaultdict(list))
        # n -> profile_hash -> max edges
        self._max_edges: dict[int, dict[int, int]] = defaultdict(dict)

        # Initialize with single vertex
        v_profile = np.array([1, 0], dtype=np.int32)
        v_hash = self._hash_profile(v_profile)
        self._profiles[1][v_hash] = v_profile
        self._graphs[1][v_hash] = [CompactGraph(edges=0, op="v")]
        self._max_edges[1][v_hash] = 0

    def _hash_profile(self, profile: np.ndarray) -> int:
        """Compute hash of profile for dict key."""
        return hash(profile.tobytes())

    def get_profile(self, n: int, profile_hash: int) -> np.ndarray | None:
        """Get profile array by hash."""
        return self._profiles[n].get(profile_hash)

    def add(
        self,
        n: int,
        profile: np.ndarray,
        edges: int,
        op: Literal["s", "p"],
        child1_n: int,
        child1_hash: int,
        child1_idx: int,
        child2_n: int,
        child2_hash: int,
        child2_idx: int
    ) -> bool:
        """
        Add a graph to the registry if it's extremal.

        Returns True if added, False if not extremal.
        """
        profile_hash = self._hash_profile(profile)

        # Check if we have this profile
        if profile_hash in self._max_edges[n]:
            max_e = self._max_edges[n][profile_hash]
            if edges < max_e:
                return False  # Not extremal
            elif edges == max_e:
                # Same edge count - add if structure is new
                # For now, always add (dedup later)
                pass
            else:
                # New maximum - clear old and add
                self._graphs[n][profile_hash] = []
                self._max_edges[n][profile_hash] = edges
        else:
            # New profile
            self._profiles[n][profile_hash] = profile.copy()
            self._max_edges[n][profile_hash] = edges

        graph = CompactGraph(
            edges=edges,
            op=op,
            child1_n=child1_n,
            child1_profile_hash=child1_hash,
            child1_idx=child1_idx,
            child2_n=child2_n,
            child2_profile_hash=child2_hash,
            child2_idx=child2_idx
        )
        self._graphs[n][profile_hash].append(graph)
        return True

    def get_graphs(self, n: int, profile_hash: int) -> list[CompactGraph]:
        """Get all graphs for given n and profile."""
        return self._graphs[n].get(profile_hash, [])

    def get_all_profiles(self, n: int) -> list[tuple[int, np.ndarray]]:
        """Get all (hash, profile) pairs for vertex count n."""
        return [(h, p) for h, p in self._profiles[n].items()]

    def get_all_hashes(self, n: int) -> list[int]:
        """Get all profile hashes for vertex count n."""
        return list(self._profiles[n].keys())

    def profile_count(self, n: int) -> int:
        """Number of distinct profiles for n vertices."""
        return len(self._profiles[n])

    def graph_count(self, n: int) -> int:
        """Total graphs for n vertices."""
        return sum(len(gs) for gs in self._graphs[n].values())

    def total_graphs(self) -> int:
        """Total graphs across all n."""
        return sum(self.graph_count(n) for n in self._graphs)

    def max_n(self) -> int:
        """Maximum vertex count in registry."""
        return max(self._graphs.keys()) if self._graphs else 0

    def iter_graphs_for_n(self, n: int) -> Iterator[tuple[int, np.ndarray, int, CompactGraph]]:
        """
        Iterate over all graphs for n vertices.

        Yields: (profile_hash, profile_array, graph_idx, graph)
        """
        for profile_hash, profile in self._profiles[n].items():
            for idx, graph in enumerate(self._graphs[n][profile_hash]):
                yield profile_hash, profile, idx, graph

    def get_avoiding(
        self,
        n: int,
        s: int,
        t: int
    ) -> list[tuple[int, np.ndarray, int, CompactGraph]]:
        """
        Get extremal K_{s,t}-free graphs on n vertices.

        Returns list of (edges, profile, profile_hash, graph) tuples.
        """
        from .profile_ops import profile_avoids_kst

        max_edges = -1
        candidates = []

        for profile_hash, profile in self._profiles[n].items():
            if profile_avoids_kst(profile, s, t):
                graphs = self._graphs[n][profile_hash]
                if graphs:
                    edge_count = graphs[0].edges
                    if edge_count > max_edges:
                        max_edges = edge_count
                        candidates = [(edge_count, profile, profile_hash, g) for g in graphs]
                    elif edge_count == max_edges:
                        candidates.extend((edge_count, profile, profile_hash, g) for g in graphs)

        return candidates

    def reconstruct_structure(self, n: int, profile_hash: int, graph_idx: int) -> str:
        """
        Lazily reconstruct structure string for a graph.

        This traverses the construction tree to build the string.
        """
        graph = self._graphs[n][profile_hash][graph_idx]

        if graph.op == "v":
            return "1"

        # Recursively get child structures
        child1_str = self.reconstruct_structure(
            graph.child1_n,
            graph.child1_profile_hash,
            graph.child1_idx
        )
        child2_str = self.reconstruct_structure(
            graph.child2_n,
            graph.child2_profile_hash,
            graph.child2_idx
        )

        op_char = "S" if graph.op == "s" else "P"
        return f"{op_char}({child1_str},{child2_str})"

    def save_checkpoint(self, path: Path, completed_n: int):
        """Save registry to binary file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "completed_n": completed_n,
            "profiles": dict(self._profiles),
            "graphs": {n: dict(gs) for n, gs in self._graphs.items()},
            "max_edges": {n: dict(me) for n, me in self._max_edges.items()}
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_checkpoint(cls, path: Path) -> tuple[FastRegistry, int]:
        """Load registry from binary file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        registry = cls.__new__(cls)
        registry._profiles = defaultdict(dict, data["profiles"])
        registry._graphs = defaultdict(lambda: defaultdict(list))
        for n, gs in data["graphs"].items():
            for h, graphs in gs.items():
                registry._graphs[n][h] = graphs
        registry._max_edges = defaultdict(dict)
        for n, me in data["max_edges"].items():
            registry._max_edges[n] = me

        return registry, data["completed_n"]

    def statistics(self) -> dict:
        """Get registry statistics."""
        stats = {
            "max_n": self.max_n(),
            "total_graphs": self.total_graphs(),
            "by_n": {}
        }
        for n in sorted(self._graphs.keys()):
            stats["by_n"][n] = {
                "profiles": self.profile_count(n),
                "graphs": self.graph_count(n)
            }
        return stats
