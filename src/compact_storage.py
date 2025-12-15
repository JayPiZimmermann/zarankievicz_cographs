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
    depth: int = 0  # max number of operation switches from root to leaf
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

    def __init__(self, use_profile_domination: bool = False, use_depth_domination: bool = False):
        # profile_hash -> numpy array
        self._profiles: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        # n -> profile_hash -> list of CompactGraph
        self._graphs: dict[int, dict[int, list[CompactGraph]]] = defaultdict(lambda: defaultdict(list))
        # n -> profile_hash -> max edges
        self._max_edges: dict[int, dict[int, int]] = defaultdict(dict)
        # Optimization flags
        self._use_profile_domination = use_profile_domination
        self._use_depth_domination = use_depth_domination

        # Initialize with single vertex
        v_profile = np.array([1, 0], dtype=np.int32)
        v_hash = self._hash_profile(v_profile)
        self._profiles[1][v_hash] = v_profile
        self._graphs[1][v_hash] = [CompactGraph(edges=0, op="v", depth=0)]
        self._max_edges[1][v_hash] = 0

    def _hash_profile(self, profile: np.ndarray) -> int:
        """Compute hash of profile for dict key."""
        return hash(profile.tobytes())

    def _profile_dominates(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Check if profile p1 dominates profile p2.

        p1 dominates p2 if p1[i] >= p2[i] for all i.
        This means any K_{i,j} contained in graph with p2 is also in graph with p1.
        """
        return np.all(p1 >= p2)

    def get_profile(self, n: int, profile_hash: int) -> np.ndarray | None:
        """Get profile array by hash."""
        return self._profiles[n].get(profile_hash)

    def check_profile_domination(
        self,
        n: int,
        profile: np.ndarray,
        edges: int
    ) -> tuple[bool, list[int]]:
        """
        Check if a profile should be added based on domination rules.

        Returns:
            (should_add, profiles_to_remove)
            - should_add: True if profile should be added
            - profiles_to_remove: List of profile hashes to remove (dominated by new)
        """
        if not self._use_profile_domination:
            return True, []

        profiles_to_remove = []
        for existing_hash, existing_profile in self._profiles[n].items():
            existing_edges = self._max_edges[n][existing_hash]

            # Check if new dominates existing
            # If existing >= new (profile) AND new >= existing (edges): remove existing
            if self._profile_dominates(existing_profile, profile) and edges >= existing_edges:
                profiles_to_remove.append(existing_hash)

            # Check if existing dominates new
            # If new >= existing (profile) AND existing >= new (edges): don't add new
            if self._profile_dominates(profile, existing_profile) and existing_edges >= edges:
                return False, []  # New profile is dominated, don't add

        return True, profiles_to_remove

    def add(
        self,
        n: int,
        profile: np.ndarray,
        edges: int,
        op: Literal["s", "p"],
        depth: int,
        child1_n: int,
        child1_hash: int,
        child1_idx: int,
        child2_n: int,
        child2_hash: int,
        child2_idx: int
    ) -> bool:
        """
        Add a graph to the registry if it's extremal.

        With profile domination enabled:
        - Prunes profiles dominated by others with more/equal edges
        - Removes dominated profiles when adding a dominating one

        With depth domination enabled:
        - Only keeps graphs with minimal depth for same profile and edge count

        Returns True if added, False if not extremal or duplicate.
        """
        profile_hash = self._hash_profile(profile)

        # Profile domination check (if enabled)
        should_add, profiles_to_remove = self.check_profile_domination(n, profile, edges)
        if not should_add:
            return False

        # Remove dominated profiles
        for h in profiles_to_remove:
            del self._profiles[n][h]
            del self._graphs[n][h]
            del self._max_edges[n][h]

        # Check if we have this profile
        if profile_hash in self._max_edges[n]:
            max_e = self._max_edges[n][profile_hash]
            if edges < max_e:
                return False  # Not extremal
            elif edges == max_e:
                existing_graphs = self._graphs[n][profile_hash]

                # Depth domination check (if enabled)
                if self._use_depth_domination and existing_graphs:
                    min_depth = min(g.depth for g in existing_graphs)
                    if depth > min_depth:
                        return False  # New graph has greater depth, don't add
                    elif depth < min_depth:
                        # New graph has smaller depth, remove all existing
                        self._graphs[n][profile_hash] = []
                    # If depth == min_depth, continue to canonical check

                # Same edge count - add only if structure is new (canonical check)
                new_canonical = self._compute_canonical_for_new(
                    op, child1_n, child1_hash, child1_idx,
                    child2_n, child2_hash, child2_idx
                )
                # Check against existing graphs
                for idx in range(len(self._graphs[n][profile_hash])):
                    if self.reconstruct_canonical(n, profile_hash, idx) == new_canonical:
                        return False  # Duplicate (isomorphic)
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
            depth=depth,
            child1_n=child1_n,
            child1_profile_hash=child1_hash,
            child1_idx=child1_idx,
            child2_n=child2_n,
            child2_profile_hash=child2_hash,
            child2_idx=child2_idx
        )
        self._graphs[n][profile_hash].append(graph)
        return True

    def add_batch(
        self,
        n: int,
        candidates: list[tuple]
    ) -> int:
        """
        Add multiple graphs in batch, with optimized profile domination checks.

        Args:
            candidates: List of (profile_bytes, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx)

        Returns:
            Number of graphs actually added
        """
        if not candidates:
            return 0

        # Group candidates by (profile_hash, edges)
        from collections import defaultdict
        grouped = defaultdict(list)

        for candidate in candidates:
            profile_bytes, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx = candidate
            profile = np.frombuffer(profile_bytes, dtype=np.int32).copy()
            profile_hash = self._hash_profile(profile)
            key = (profile_hash, edges)
            grouped[key].append((profile, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx))

        # Process each unique (profile, edges) combination
        graphs_added = 0
        for (profile_hash, edges), graph_list in grouped.items():
            # Get profile from first item (all items in group have same profile)
            profile = graph_list[0][0]

            # Check domination ONCE for this (profile, edges) combination
            should_add, profiles_to_remove = self.check_profile_domination(n, profile, edges)
            if not should_add:
                continue  # Skip all graphs with this (profile, edges)

            # Remove dominated profiles
            for h in profiles_to_remove:
                if h in self._profiles[n]:  # Check existence since batch might remove same profile multiple times
                    del self._profiles[n][h]
                    del self._graphs[n][h]
                    del self._max_edges[n][h]

            # Now add all graphs with this (profile, edges)
            for graph_data in graph_list:
                _, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx = graph_data

                # Check if we have this profile
                if profile_hash in self._max_edges[n]:
                    max_e = self._max_edges[n][profile_hash]
                    if edges < max_e:
                        continue  # Not extremal
                    elif edges == max_e:
                        existing_graphs = self._graphs[n][profile_hash]

                        # Depth domination check (if enabled)
                        if self._use_depth_domination and existing_graphs:
                            min_depth = min(g.depth for g in existing_graphs)
                            if depth > min_depth:
                                continue  # New graph has greater depth, don't add
                            elif depth < min_depth:
                                # New graph has smaller depth, remove all existing
                                self._graphs[n][profile_hash] = []
                            # If depth == min_depth, continue to canonical check

                        # Same edge count - add only if structure is new (canonical check)
                        new_canonical = self._compute_canonical_for_new(
                            op, c1_n, c1_hash, c1_idx,
                            c2_n, c2_hash, c2_idx
                        )
                        # Check against existing graphs
                        is_duplicate = False
                        for idx in range(len(self._graphs[n][profile_hash])):
                            if self.reconstruct_canonical(n, profile_hash, idx) == new_canonical:
                                is_duplicate = True
                                break
                        if is_duplicate:
                            continue  # Duplicate (isomorphic)
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
                    depth=depth,
                    child1_n=c1_n,
                    child1_profile_hash=c1_hash,
                    child1_idx=c1_idx,
                    child2_n=c2_n,
                    child2_profile_hash=c2_hash,
                    child2_idx=c2_idx
                )
                self._graphs[n][profile_hash].append(graph)
                graphs_added += 1

        return graphs_added

    def get_graphs(self, n: int, profile_hash: int) -> list[CompactGraph]:
        """Get all graphs for given n and profile."""
        return self._graphs[n].get(profile_hash, [])

    def get_depth(self, n: int, profile_hash: int, idx: int) -> int:
        """Get the depth of a specific graph."""
        graphs = self._graphs[n].get(profile_hash, [])
        if idx < len(graphs):
            return graphs[idx].depth
        return 0

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

    def _collect_children_flat(
        self,
        n: int,
        profile_hash: int,
        graph_idx: int,
        target_op: str,
        use_canonical: bool = False
    ) -> list[str]:
        """
        Collect all child strings, flattening nested same-type operations.

        For example, P(a, P(b, c)) flattens to [a, b, c].
        """
        graph = self._graphs[n][profile_hash][graph_idx]

        if graph.op == "v":
            return ["1"]

        if graph.op != target_op:
            # Different operation type, return as single child
            if use_canonical:
                return [self.reconstruct_canonical(n, profile_hash, graph_idx)]
            else:
                return [self.reconstruct_structure(n, profile_hash, graph_idx)]

        # Same operation type, recursively collect children
        result = []
        result.extend(self._collect_children_flat(
            graph.child1_n, graph.child1_profile_hash, graph.child1_idx,
            target_op, use_canonical
        ))
        result.extend(self._collect_children_flat(
            graph.child2_n, graph.child2_profile_hash, graph.child2_idx,
            target_op, use_canonical
        ))
        return result

    def reconstruct_structure(self, n: int, profile_hash: int, graph_idx: int) -> str:
        """
        Lazily reconstruct structure string for a graph.

        Flattens nested same-type operations: P(a,P(b,c)) becomes P(a,b,c).
        """
        graph = self._graphs[n][profile_hash][graph_idx]

        if graph.op == "v":
            return "1"

        op_char = "S" if graph.op == "s" else "P"
        # Flatten nested same-type operations
        child_strs = self._collect_children_flat(
            n, profile_hash, graph_idx, graph.op, use_canonical=False
        )
        return f"{op_char}({','.join(child_strs)})"

    def reconstruct_canonical(self, n: int, profile_hash: int, graph_idx: int) -> str:
        """
        Reconstruct canonical structure string for a graph.

        Flattens nested same-type operations and sorts children lexicographically
        to ensure isomorphic graphs produce identical strings.
        """
        graph = self._graphs[n][profile_hash][graph_idx]

        if graph.op == "v":
            return "1"

        op_char = "S" if graph.op == "s" else "P"
        # Flatten nested same-type operations and sort
        child_strs = self._collect_children_flat(
            n, profile_hash, graph_idx, graph.op, use_canonical=True
        )
        child_strs.sort()
        return f"{op_char}({','.join(child_strs)})"

    def _compute_canonical_for_new(
        self,
        op: Literal["s", "p"],
        child1_n: int,
        child1_hash: int,
        child1_idx: int,
        child2_n: int,
        child2_hash: int,
        child2_idx: int
    ) -> str:
        """Compute canonical string for a graph being added (before it's in registry)."""
        target_op = op
        # Collect and flatten children
        child_strs = []
        child_strs.extend(self._collect_children_flat(
            child1_n, child1_hash, child1_idx, target_op, use_canonical=True
        ))
        child_strs.extend(self._collect_children_flat(
            child2_n, child2_hash, child2_idx, target_op, use_canonical=True
        ))
        child_strs.sort()

        op_char = "S" if op == "s" else "P"
        return f"{op_char}({','.join(child_strs)})"

    def save_checkpoint(self, path: Path, completed_n: int):
        """Save registry to binary file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "completed_n": completed_n,
            "profiles": dict(self._profiles),
            "graphs": {n: dict(gs) for n, gs in self._graphs.items()},
            "max_edges": {n: dict(me) for n, me in self._max_edges.items()},
            "use_profile_domination": self._use_profile_domination,
            "use_depth_domination": self._use_depth_domination
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
        registry._use_profile_domination = data.get("use_profile_domination", False)
        registry._use_depth_domination = data.get("use_depth_domination", False)

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
