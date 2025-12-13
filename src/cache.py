"""JSON cache for registry persistence."""

from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any

from .registry import Registry
from .builder import parse_structure


# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"


def get_cache_path(N: int, T: int | None = None, cache_dir: Path | None = None) -> Path:
    """
    Get the cache file path for given parameters.

    Args:
        N: Maximum vertex count
        T: K_{T,T} pruning threshold (None = no pruning)
        cache_dir: Cache directory (default: ./cache)

    Returns:
        Path to cache file
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if T is None:
        filename = f"registry_N{N}.json"
    else:
        filename = f"registry_N{N}_T{T}.json"

    return cache_dir / filename


def save_registry(
    registry: Registry,
    path: Path | str | None = None,
    N: int | None = None,
    T: int | None = None,
    metadata: dict[str, Any] | None = None
) -> Path:
    """
    Save registry to JSON file.

    Args:
        registry: The registry to save
        path: Explicit path (overrides N/T-based naming)
        N: Maximum vertex count (for automatic naming)
        T: Pruning threshold (for automatic naming)
        metadata: Additional metadata to include

    Returns:
        Path where registry was saved
    """
    if path is None:
        if N is None:
            N = registry.max_n()
        path = get_cache_path(N, T)
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Build save data
    data = {
        "metadata": {
            "max_n": registry.max_n(),
            "total_graphs": registry.total_graphs(),
            "pruning_threshold": T,
            "saved_at": datetime.now().isoformat(),
            **(metadata or {})
        },
        "registry": registry.to_dict()
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_registry(
    path: Path | str | None = None,
    N: int | None = None,
    T: int | None = None
) -> tuple[Registry, dict]:
    """
    Load registry from JSON file.

    Args:
        path: Explicit path (overrides N/T-based lookup)
        N: Maximum vertex count (for automatic lookup)
        T: Pruning threshold (for automatic lookup)

    Returns:
        Tuple of (registry, metadata)

    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    if path is None:
        if N is None:
            raise ValueError("Must provide either path or N")
        path = get_cache_path(N, T)
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    registry = _rebuild_registry(data["registry"])
    metadata = data.get("metadata", {})

    return registry, metadata


def _rebuild_registry(data: dict) -> Registry:
    """
    Rebuild registry from serialized data.

    This reconstructs Cotree objects from structure strings.
    """
    from collections import defaultdict
    from .cotree import Cotree

    registry = Registry.__new__(Registry)
    registry._data = defaultdict(dict)

    for n_str, profile_dict in data.items():
        n = int(n_str)
        for profile_str, graphs in profile_dict.items():
            profile = tuple(int(p) for p in profile_str.split(","))
            registry._data[n][profile] = []
            for g in graphs:
                cotree = parse_structure(g["structure"])
                registry._data[n][profile].append((g["edges"], cotree))

    return registry


def cache_exists(N: int, T: int | None = None, cache_dir: Path | None = None) -> bool:
    """Check if cache file exists for given parameters."""
    return get_cache_path(N, T, cache_dir).exists()


def list_caches(cache_dir: Path | None = None) -> list[dict]:
    """
    List all available cache files.

    Args:
        cache_dir: Cache directory to scan

    Returns:
        List of dicts with cache info (path, N, T, metadata)
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    if not cache_dir.exists():
        return []

    caches = []
    for path in cache_dir.glob("registry_*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            metadata = data.get("metadata", {})

            # Parse N and T from filename
            name = path.stem
            parts = name.split("_")
            N = int(parts[1][1:])  # "N20" -> 20
            T = int(parts[2][1:]) if len(parts) > 2 else None  # "T5" -> 5

            caches.append({
                "path": path,
                "N": N,
                "T": T,
                "metadata": metadata
            })
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            continue

    return sorted(caches, key=lambda c: (c["N"], c["T"] or 0))


def find_best_cache(
    target_N: int,
    T: int | None = None,
    cache_dir: Path | None = None
) -> Path | None:
    """
    Find the best available cache to start from.

    Returns the largest cache with N <= target_N and compatible T.

    Args:
        target_N: Target vertex count
        T: Required pruning threshold
        cache_dir: Cache directory

    Returns:
        Path to best cache, or None if no suitable cache exists
    """
    caches = list_caches(cache_dir)
    best = None
    best_n = 0

    for cache in caches:
        # Must not exceed target
        if cache["N"] > target_N:
            continue

        # T compatibility:
        # - If we need T=None, cache must have T=None
        # - If we need T=k, cache can have T=None or T>=k
        cache_T = cache["T"]
        if T is None:
            if cache_T is not None:
                continue
        else:
            if cache_T is not None and cache_T < T:
                continue

        if cache["N"] > best_n:
            best_n = cache["N"]
            best = cache["path"]

    return best


def clear_cache(cache_dir: Path | None = None) -> int:
    """
    Clear all cache files.

    Args:
        cache_dir: Cache directory

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    if not cache_dir.exists():
        return 0

    count = 0
    for path in cache_dir.glob("registry_*.json"):
        path.unlink()
        count += 1

    return count


def cache_info(path: Path | str) -> dict:
    """
    Get information about a cache file without fully loading it.

    Args:
        path: Path to cache file

    Returns:
        Dictionary with cache metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "metadata": data.get("metadata", {}),
        "vertex_counts": sorted(int(k) for k in data.get("registry", {}).keys())
    }
