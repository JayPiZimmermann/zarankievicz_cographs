"""Incremental file-based cache for registry with per-n saves."""

from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any

from .cotree import Cotree
from .registry import Registry
from .builder import parse_structure
from .export import export_extremal_for_biclique


DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"


def get_run_dir(run_name: str, cache_dir: Path | None = None) -> Path:
    """Get the directory for a specific run."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    return cache_dir / f"registries_{run_name}"


def profile_to_filename(profile: tuple[int, ...]) -> str:
    """Convert profile tuple to safe filename."""
    return "profile_" + "_".join(str(p) for p in profile) + ".json"


def filename_to_profile(filename: str) -> tuple[int, ...]:
    """Convert filename back to profile tuple."""
    # "profile_2_1_0.json" -> (2, 1, 0)
    parts = filename.replace("profile_", "").replace(".json", "").split("_")
    return tuple(int(p) for p in parts)


class IncrementalRegistry:
    """
    Registry that saves incrementally to disk after each vertex count.

    Structure:
        cache/registries_{run_name}/
        ├── metadata.json           # N, T, completed_n, timestamps
        ├── n1/
        │   └── profile_1_0.json    # graphs for profile (1,0)
        ├── n2/
        │   ├── profile_2_0_0.json
        │   └── profile_2_1_0.json
        ├── ...
        └── exports/
            ├── extremal_K22.json
            └── ...
    """

    def __init__(self, run_name: str, N: int, T: int | None = None,
                 cache_dir: Path | None = None, s_max: int = 7, t_max: int = 7):
        self.run_name = run_name
        self.N = N
        self.T = T
        self.s_max = s_max
        self.t_max = t_max
        self.run_dir = get_run_dir(run_name, cache_dir)
        self.exports_dir = self.run_dir / "exports"

        # In-memory registry for current computation
        self._data: dict[int, dict[tuple[int, ...], list[tuple[int, Cotree]]]] = defaultdict(dict)

        # Track completed n values
        self.completed_n: set[int] = set()
        self.start_time: str | None = None

    def exists(self) -> bool:
        """Check if this run already exists on disk."""
        return (self.run_dir / "metadata.json").exists()

    def is_complete(self) -> bool:
        """Check if this run is fully complete."""
        if not self.exists():
            return False
        meta = self._load_metadata()
        return meta.get("completed_n", 0) >= self.N

    def load_existing(self) -> int:
        """
        Load existing run state from disk.

        Returns:
            The next n to compute (1 if fresh start, or completed_n + 1)
        """
        if not self.exists():
            return 1

        meta = self._load_metadata()

        # Verify parameters match
        if meta.get("N") != self.N or meta.get("T") != self.T:
            raise ValueError(
                f"Run parameters mismatch: existing N={meta.get('N')}, T={meta.get('T')} "
                f"vs requested N={self.N}, T={self.T}"
            )

        max_completed = meta.get("completed_n", 0)
        self.start_time = meta.get("start_time")

        # Load all completed n values
        for n in range(1, max_completed + 1):
            self._load_n(n)
            self.completed_n.add(n)

        return max_completed + 1

    def _load_metadata(self) -> dict:
        """Load metadata from disk."""
        meta_path = self.run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to disk."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "N": self.N,
            "T": self.T,
            "s_max": self.s_max,
            "t_max": self.t_max,
            "completed_n": max(self.completed_n) if self.completed_n else 0,
            "start_time": self.start_time or datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_graphs": sum(
                sum(len(graphs) for graphs in pdict.values())
                for pdict in self._data.values()
            )
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _load_n(self, n: int):
        """Load all profiles for vertex count n from disk."""
        n_dir = self.run_dir / f"n{n}"
        if not n_dir.exists():
            return

        self._data[n] = {}
        for profile_file in n_dir.glob("profile_*.json"):
            profile = filename_to_profile(profile_file.name)
            with open(profile_file) as f:
                data = json.load(f)

            graphs = []
            for g in data["graphs"]:
                cotree = parse_structure(g["structure"])
                graphs.append((g["edges"], cotree))
            self._data[n][profile] = graphs

    def save_n(self, n: int):
        """Save all profiles for vertex count n to disk."""
        n_dir = self.run_dir / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)

        for profile, graphs in self._data.get(n, {}).items():
            filename = profile_to_filename(profile)
            data = {
                "n": n,
                "profile": list(profile),
                "graphs": [
                    {"edges": edges, "structure": cotree.structure_str()}
                    for edges, cotree in graphs
                ]
            }
            with open(n_dir / filename, "w") as f:
                json.dump(data, f, indent=2)

        self.completed_n.add(n)
        self._save_metadata()

    def add(self, cotree: Cotree) -> bool:
        """Add a cotree to the registry if it's extremal for its profile."""
        n = cotree.n
        profile = cotree.profile
        edges = cotree.edges

        profile_dict = self._data[n]

        if profile not in profile_dict:
            profile_dict[profile] = [(edges, cotree)]
            return True

        existing = profile_dict[profile]
        max_edges = existing[0][0]

        if edges > max_edges:
            profile_dict[profile] = [(edges, cotree)]
            return True
        elif edges == max_edges:
            for _, existing_ct in existing:
                if existing_ct.structure_str() == cotree.structure_str():
                    return False
            existing.append((edges, cotree))
            return True
        return False

    def get_avoiding(self, n: int, s: int, t: int) -> list[tuple[int, Cotree]]:
        """Get extremal K_{s,t}-free graphs on n vertices."""
        from .profile import profile_avoids

        profile_dict = self._data.get(n, {})
        if not profile_dict:
            return []

        max_edges = -1
        candidates: list[tuple[int, Cotree]] = []

        for profile, graphs in profile_dict.items():
            if profile_avoids(profile, s, t):
                edge_count = graphs[0][0]
                if edge_count > max_edges:
                    max_edges = edge_count
                    candidates = list(graphs)
                elif edge_count == max_edges:
                    candidates.extend(graphs)

        return candidates

    def profile_count(self, n: int) -> int:
        """Count distinct profiles for vertex count n."""
        return len(self._data.get(n, {}))

    def graph_count(self, n: int) -> int:
        """Count total graphs for vertex count n."""
        return sum(len(graphs) for graphs in self._data.get(n, {}).values())

    def max_n(self) -> int:
        """Get maximum completed vertex count."""
        return max(self.completed_n) if self.completed_n else 0

    def update_exports(self, up_to_n: int):
        """Update export files with current data up to n."""
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        # Export for each (s,t) pair
        for s in range(2, self.s_max + 1):
            for t in range(s, self.t_max + 1):
                self._export_biclique(s, t, up_to_n)

    def _export_biclique(self, s: int, t: int, up_to_n: int):
        """Export extremal data for K_{s,t}."""
        from .cotree import to_graph6
        from .export import analyze_extremal

        data = {
            "s": s,
            "t": t,
            "computed_up_to_n": up_to_n,
            "extremal_by_n": {}
        }

        for n in range(1, up_to_n + 1):
            graphs = self.get_avoiding(n, s, t)
            if graphs:
                entry = {
                    "ex": graphs[0][0],
                    "count": len(graphs),
                    "graphs": [g.structure_str() for _, g in graphs],
                    "graph6": [to_graph6(g) for _, g in graphs],
                    "analyses": [analyze_extremal(g) for _, g in graphs]
                }
                data["extremal_by_n"][str(n)] = entry

        path = self.exports_dir / f"extremal_K{s}{t}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_registry(self) -> Registry:
        """Convert to standard Registry object."""
        registry = Registry.__new__(Registry)
        registry._data = defaultdict(dict, self._data)
        return registry


def list_runs(cache_dir: Path | None = None) -> list[dict]:
    """List all available runs."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    runs = []
    for run_dir in cache_dir.glob("registries_*"):
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            runs.append({
                "name": run_dir.name.replace("registries_", ""),
                "path": run_dir,
                "N": meta.get("N"),
                "T": meta.get("T"),
                "completed_n": meta.get("completed_n", 0),
                "total_graphs": meta.get("total_graphs", 0),
                "last_update": meta.get("last_update")
            })

    return sorted(runs, key=lambda r: r["name"])
