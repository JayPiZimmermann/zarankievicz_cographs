"""High-performance parallel builder for extremal cographs."""

from __future__ import annotations
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Callable
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

from .compact_storage import FastRegistry, CompactGraph
from .profile_ops import (
    sum_profile_fast, product_profile_fast,
    sum_profile_check_ktt, product_profile_check_ktt,
    profile_avoids_kst
)


def _process_partition_pair(
    args: tuple
) -> list[tuple]:
    """
    Worker function: process all profile pairs for (n1, n2).

    Args:
        args: (n1, n2, profiles1_data, profiles2_data, T)
            profiles_data: list of (hash, profile_bytes, graphs_data)
            graphs_data: list of (edges, idx) for each graph

    Returns:
        List of results: (new_profile_bytes, edges, op, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx)
    """
    n1, n2, profiles1_data, profiles2_data, T = args

    results = []

    for hash1, pbytes1, graphs1 in profiles1_data:
        p1 = np.frombuffer(pbytes1, dtype=np.int32).copy()

        for hash2, pbytes2, graphs2 in profiles2_data:
            p2 = np.frombuffer(pbytes2, dtype=np.int32).copy()

            # Fast K_{T,T} checks at profile level
            skip_sum = T is not None and sum_profile_check_ktt(p1, p2, T)
            skip_product = T is not None and product_profile_check_ktt(p1, p2, T)

            if skip_sum and skip_product:
                continue

            # Compute profiles once per profile pair
            sum_profile = None if skip_sum else sum_profile_fast(p1, p2)
            prod_profile = None if skip_product else product_profile_fast(p1, p2)

            # Compute edge counts
            e1_base = graphs1[0][0] if graphs1 else 0  # All graphs in profile have same edge contribution
            e2_base = graphs2[0][0] if graphs2 else 0

            sum_edges = e1_base + e2_base
            prod_edges = e1_base + e2_base + n1 * n2

            # For each graph pair (only need one representative per profile for the profile)
            # But we track all graph indices for reconstruction
            for edges1, idx1 in graphs1:
                for edges2, idx2 in graphs2:
                    # Symmetry: skip if n1 == n2 and hash1 > hash2
                    if n1 == n2 and hash1 > hash2:
                        continue
                    # Also skip if same profile and idx1 > idx2
                    if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                        continue

                    if not skip_sum:
                        results.append((
                            sum_profile.tobytes(),
                            sum_edges,
                            "s",
                            n1, hash1, idx1,
                            n2, hash2, idx2
                        ))

                    if not skip_product:
                        results.append((
                            prod_profile.tobytes(),
                            prod_edges,
                            "p",
                            n1, hash1, idx1,
                            n2, hash2, idx2
                        ))

    return results


def build_fast(
    N: int,
    T: int | None = None,
    num_workers: int | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 5,
    progress_callback: Callable[[int, int, int, int, int, int, float], None] | None = None
) -> FastRegistry:
    """
    Build registry using parallel processing.

    Args:
        N: Maximum vertex count
        T: Pruning threshold for K_{T,T}
        num_workers: Number of parallel workers (default: cpu_count)
        checkpoint_dir: Directory for checkpoints (None = no checkpoints)
        checkpoint_interval: Save checkpoint every N vertex counts
        progress_callback: callback(n, N, added, profiles, total, cumulative, time_sec)

    Returns:
        FastRegistry with all computed data
    """
    if num_workers is None:
        num_workers = cpu_count()

    registry = FastRegistry()
    start_n = 2

    # Try to load checkpoint
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / f"checkpoint_N{N}_T{T}.pkl"
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            registry, completed_n = FastRegistry.load_checkpoint(checkpoint_path)
            start_n = completed_n + 1
            print(f"Resuming from n={start_n}")

    cumulative = sum(registry.graph_count(n) for n in range(1, start_n))

    for target_n in range(start_n, N + 1):
        start_time = time.time()
        graphs_added = 0

        # Prepare work items for parallel processing
        work_items = []

        for n1 in range(1, target_n // 2 + 1):
            n2 = target_n - n1

            # Get profile data for n1
            profiles1_data = []
            for h, p in registry.get_all_profiles(n1):
                graphs = registry.get_graphs(n1, h)
                graphs_data = [(g.edges, idx) for idx, g in enumerate(graphs)]
                profiles1_data.append((h, p.tobytes(), graphs_data))

            # Get profile data for n2
            profiles2_data = []
            for h, p in registry.get_all_profiles(n2):
                graphs = registry.get_graphs(n2, h)
                graphs_data = [(g.edges, idx) for idx, g in enumerate(graphs)]
                profiles2_data.append((h, p.tobytes(), graphs_data))

            if profiles1_data and profiles2_data:
                work_items.append((n1, n2, profiles1_data, profiles2_data, T))

        # Process in parallel
        if num_workers > 1 and len(work_items) > 1:
            with Pool(processes=min(num_workers, len(work_items))) as pool:
                all_results = pool.map(_process_partition_pair, work_items)
        else:
            all_results = [_process_partition_pair(item) for item in work_items]

        # Merge results into registry
        for results in all_results:
            for result in results:
                (profile_bytes, edges, op,
                 c1_n, c1_hash, c1_idx,
                 c2_n, c2_hash, c2_idx) = result

                profile = np.frombuffer(profile_bytes, dtype=np.int32).copy()

                if registry.add(
                    target_n, profile, edges, op,
                    c1_n, c1_hash, c1_idx,
                    c2_n, c2_hash, c2_idx
                ):
                    graphs_added += 1

        elapsed = time.time() - start_time
        cumulative += registry.graph_count(target_n)

        if progress_callback:
            progress_callback(
                target_n, N, graphs_added,
                registry.profile_count(target_n),
                registry.graph_count(target_n),
                cumulative,
                elapsed
            )

        # Checkpoint
        if checkpoint_dir and target_n % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_N{N}_T{T}.pkl"
            registry.save_checkpoint(checkpoint_path, target_n)

    # Final checkpoint
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / f"checkpoint_N{N}_T{T}.pkl"
        registry.save_checkpoint(checkpoint_path, N)

    return registry


def export_extremal_analysis(
    registry: FastRegistry,
    s: int,
    t: int,
    output_path: Path
) -> None:
    """Export analysis of extremal K_{s,t}-free graphs."""
    import json

    data = {
        "s": s,
        "t": t,
        "max_n": registry.max_n(),
        "extremal_by_n": {}
    }

    for n in range(1, registry.max_n() + 1):
        candidates = registry.get_avoiding(n, s, t)
        if not candidates:
            continue

        structures = []
        for edges, profile, graph in candidates:
            # Find this graph's index
            graphs_list = registry._graphs[n][hash(profile.tobytes())]
            idx = graphs_list.index(graph) if graph in graphs_list else 0

            struct_str = registry.reconstruct_structure(
                n, hash(profile.tobytes()), idx
            )

            structures.append({
                "structure": struct_str,
                "edges": edges,
                "last_op": "product" if graph.op == "p" else ("sum" if graph.op == "s" else "vertex"),
                "component_sizes": [graph.child1_n, graph.child2_n] if graph.op != "v" else []
            })

        data["extremal_by_n"][str(n)] = {
            "ex": candidates[0][0],
            "count": len(candidates),
            "structures": structures
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def check_conjecture_fast(
    registry: FastRegistry,
    s_min: int = 2,
    s_max: int = 7,
    t_max: int = 7
) -> dict:
    """
    Check if all extremal K_{s,t}-free graphs are connected.

    Returns detailed analysis.
    """
    results = {
        "all_connected": True,
        "exceptions": [],
        "by_st": {}
    }

    for s in range(s_min, s_max + 1):
        for t in range(s, t_max + 1):
            key = f"K{s}{t}"
            st_result = {
                "all_connected": True,
                "exceptions": [],
                "component_sizes": []
            }

            for n in range(1, registry.max_n() + 1):
                candidates = registry.get_avoiding(n, s, t)

                for edges, profile, graph in candidates:
                    if graph.op != "p" and graph.op != "v":
                        # Sum operation = disconnected
                        st_result["all_connected"] = False
                        results["all_connected"] = False

                        # Reconstruct structure for exception
                        graphs_list = registry._graphs[n][hash(profile.tobytes())]
                        idx = graphs_list.index(graph) if graph in graphs_list else 0
                        struct = registry.reconstruct_structure(n, hash(profile.tobytes()), idx)

                        exception = {
                            "n": n,
                            "s": s,
                            "t": t,
                            "structure": struct,
                            "last_op": "sum"
                        }
                        st_result["exceptions"].append(exception)
                        results["exceptions"].append(exception)

                    elif graph.op == "p":
                        # Track component sizes
                        min_size = min(graph.child1_n, graph.child2_n)
                        st_result["component_sizes"].append((n, min_size))

            results["by_st"][key] = st_result

    return results
