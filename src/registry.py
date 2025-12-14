"""Registry for storing and looking up extremal cographs."""

from __future__ import annotations
from collections import defaultdict
from typing import Iterator

from .cotree import Cotree, vertex
from .profile import profile_avoids, contains_biclique


class Registry:
    """
    Storage for extremal cographs organized by vertex count and profile.

    Structure: n -> profile -> list of (edges, cotree) pairs

    For each (n, profile), we keep only graphs with maximum edge count,
    as these are the candidates for being extremal for some K_{s,t} avoidance.
    """

    def __init__(self):
        # n -> profile -> list of (edges, cotree)
        self._data: dict[int, dict[tuple[int, ...], list[tuple[int, Cotree]]]] = defaultdict(dict)
        # Initialize with single vertex
        v = vertex()
        self.add(v)

    def add(self, cotree: Cotree) -> bool:
        """
        Add a cotree to the registry if it's extremal for its profile.

        Args:
            cotree: The cotree to add

        Returns:
            True if the cotree was added (is extremal), False otherwise
        """
        n = cotree.n
        profile = cotree.profile
        edges = cotree.edges

        profile_dict = self._data[n]

        if profile not in profile_dict:
            # New profile, add it
            profile_dict[profile] = [(edges, cotree)]
            return True

        existing = profile_dict[profile]
        max_edges = existing[0][0]  # All entries have same edge count

        if edges > max_edges:
            # New maximum, replace all
            profile_dict[profile] = [(edges, cotree)]
            return True
        elif edges == max_edges:
            # Same edge count, add to list (different structure)
            # Check if structure already exists using canonical form
            canonical = cotree.canonical_str()
            for _, existing_ct in existing:
                if existing_ct.canonical_str() == canonical:
                    return False  # Already have this structure (isomorphic)
            existing.append((edges, cotree))
            return True
        else:
            # Fewer edges, don't add
            return False

    def get_by_n(self, n: int) -> dict[tuple[int, ...], list[tuple[int, Cotree]]]:
        """Get all profiles and their graphs for vertex count n."""
        return dict(self._data.get(n, {}))

    def get_by_profile(self, n: int, profile: tuple[int, ...]) -> list[tuple[int, Cotree]]:
        """Get graphs with specific vertex count and profile."""
        return self._data.get(n, {}).get(profile, [])

    def get_avoiding(self, n: int, s: int, t: int) -> list[tuple[int, Cotree]]:
        """
        Get extremal K_{s,t}-free graphs on n vertices.

        Returns graphs with maximum edges among those avoiding K_{s,t}.

        Args:
            n: Number of vertices
            s: Left side of forbidden biclique
            t: Right side of forbidden biclique

        Returns:
            List of (edges, cotree) pairs for extremal K_{s,t}-free graphs
        """
        profile_dict = self._data.get(n, {})
        if not profile_dict:
            return []

        # Find max edges among K_{s,t}-free graphs
        max_edges = -1
        candidates: list[tuple[int, Cotree]] = []

        for profile, graphs in profile_dict.items():
            if profile_avoids(profile, s, t):
                edge_count = graphs[0][0]  # All have same edge count
                if edge_count > max_edges:
                    max_edges = edge_count
                    candidates = list(graphs)
                elif edge_count == max_edges:
                    candidates.extend(graphs)

        return candidates

    def extremal_number(self, n: int, s: int, t: int) -> int:
        """
        Get the extremal number ex(n, K_{s,t}) for cographs.

        This is the maximum number of edges in a K_{s,t}-free cograph on n vertices.

        Args:
            n: Number of vertices
            s: Left side of forbidden biclique
            t: Right side of forbidden biclique

        Returns:
            Maximum edge count, or -1 if no such graph exists
        """
        graphs = self.get_avoiding(n, s, t)
        if not graphs:
            return -1
        return graphs[0][0]

    def iter_by_n(self, n: int) -> Iterator[tuple[tuple[int, ...], int, Cotree]]:
        """
        Iterate over all (profile, edges, cotree) for vertex count n.

        Yields:
            Tuples of (profile, edges, cotree)
        """
        for profile, graphs in self._data.get(n, {}).items():
            for edges, cotree in graphs:
                yield profile, edges, cotree

    def profile_count(self, n: int) -> int:
        """Count number of distinct profiles for vertex count n."""
        return len(self._data.get(n, {}))

    def graph_count(self, n: int) -> int:
        """Count total number of graphs stored for vertex count n."""
        return sum(len(graphs) for graphs in self._data.get(n, {}).values())

    def total_graphs(self) -> int:
        """Count total number of graphs in registry."""
        return sum(self.graph_count(n) for n in self._data)

    def max_n(self) -> int:
        """Get maximum vertex count in registry."""
        return max(self._data.keys()) if self._data else 0

    def prune_containing(self, T: int) -> int:
        """
        Remove all graphs containing K_{T,T}.

        Args:
            T: Threshold - remove graphs with K_{T,T}

        Returns:
            Number of profiles removed
        """
        removed = 0
        for n in list(self._data.keys()):
            profile_dict = self._data[n]
            profiles_to_remove = []
            for profile in profile_dict:
                if contains_biclique(profile, T, T):
                    profiles_to_remove.append(profile)
            for profile in profiles_to_remove:
                del profile_dict[profile]
                removed += 1
        return removed

    def to_dict(self) -> dict:
        """
        Convert registry to serializable dictionary.

        Returns:
            Dictionary suitable for JSON serialization
        """
        result = {}
        for n, profile_dict in self._data.items():
            result[str(n)] = {}
            for profile, graphs in profile_dict.items():
                profile_key = ",".join(str(p) for p in profile)
                result[str(n)][profile_key] = [
                    {"edges": edges, "structure": cotree.structure_str()}
                    for edges, cotree in graphs
                ]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Registry:
        """
        Reconstruct registry from serialized dictionary.

        Note: This reconstructs cotrees from structure strings, which requires
        rebuilding them. For large registries, use the cache module instead.
        """
        from .builder import parse_structure

        registry = cls.__new__(cls)
        registry._data = defaultdict(dict)

        for n_str, profile_dict in data.items():
            n = int(n_str)
            for profile_str, graphs in profile_dict.items():
                profile = tuple(int(p) for p in profile_str.split(","))
                registry._data[n][profile] = [
                    (g["edges"], parse_structure(g["structure"]))
                    for g in graphs
                ]

        return registry

    def statistics(self) -> dict:
        """
        Get statistics about the registry.

        Returns:
            Dictionary with various statistics
        """
        stats = {
            "max_n": self.max_n(),
            "total_graphs": self.total_graphs(),
            "by_n": {}
        }
        for n in sorted(self._data.keys()):
            stats["by_n"][n] = {
                "profiles": self.profile_count(n),
                "graphs": self.graph_count(n)
            }
        return stats
