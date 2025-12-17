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


def _profile_dominates(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if profile p1 dominates profile p2 (p1[i] >= p2[i] for all i)."""
    return np.all(p1 >= p2)


def _filter_antichain(candidates: list[tuple]) -> list[tuple]:
    """
    Filter candidates to keep only non-dominated (antichain).

    Args:
        candidates: List of (p1_hash, p2_hash, result_profile, result_edges, ...)

    Returns:
        List of non-dominated candidates
    """
    if not candidates:
        return []

    # Sort by edges descending for efficiency
    candidates.sort(key=lambda x: x[3], reverse=True)

    antichain = []
    for candidate in candidates:
        profile = candidate[2]
        edges = candidate[3]

        # Check if dominated by any in antichain
        dominated = False
        for ac in antichain:
            ac_prof = ac[2]
            ac_edges = ac[3]
            # candidate is dominated if: candidate.profile >= ac.profile AND ac.edges >= candidate.edges
            if _profile_dominates(profile, ac_prof) and ac_edges >= edges:
                dominated = True
                break

        if not dominated:
            # Remove any antichain members dominated by this candidate
            # ac is dominated if: ac.profile >= candidate.profile AND candidate.edges >= ac.edges
            antichain = [ac for ac in antichain
                        if not (_profile_dominates(ac[2], profile) and edges >= ac[3])]
            antichain.append(candidate)

    return antichain


def _compute_viable_combinations(
    n1: int,
    n2: int,
    profiles1_data: list,
    profiles2_data: list,
    T: int | None
) -> dict:
    """
    Pre-compute which profile combinations are non-dominated.

    Args:
        n1, n2: Vertex counts
        profiles1_data: List of (hash, profile_bytes, max_edges, graphs_info)
        profiles2_data: List of (hash, profile_bytes, max_edges, graphs_info)
        T: K_{T,T} pruning threshold

    Returns:
        {
            'sum': [(p1_hash, p2_hash, result_profile_bytes, result_edges, graphs1_info, graphs2_info), ...],
            'product': [(p1_hash, p2_hash, result_profile_bytes, result_edges, graphs1_info, graphs2_info), ...]
        }
    """
    sum_candidates = []
    prod_candidates = []

    # Generate all profile combinations
    for hash1, pbytes1, max_edges1, graphs1_info in profiles1_data:
        p1 = np.frombuffer(pbytes1, dtype=np.int32).copy()

        for hash2, pbytes2, max_edges2, graphs2_info in profiles2_data:
            p2 = np.frombuffer(pbytes2, dtype=np.int32).copy()

            # Fast K_{T,T} checks at profile level
            skip_sum = T is not None and sum_profile_check_ktt(p1, p2, T)
            skip_product = T is not None and product_profile_check_ktt(p1, p2, T)

            if not skip_sum:
                # Sum operation
                sum_prof = sum_profile_fast(p1, p2)
                sum_edges = max_edges1 + max_edges2
                sum_candidates.append((hash1, hash2, sum_prof, sum_edges, graphs1_info, graphs2_info))

            if not skip_product:
                # Product operation
                prod_prof = product_profile_fast(p1, p2)
                prod_edges = max_edges1 + max_edges2 + n1 * n2
                prod_candidates.append((hash1, hash2, prod_prof, prod_edges, graphs1_info, graphs2_info))

    # Filter to antichain (non-dominated)
    sum_viable = _filter_antichain(sum_candidates)
    prod_viable = _filter_antichain(prod_candidates)

    return {'sum': sum_viable, 'product': prod_viable}


def _process_partition_pair_lattice(
    args: tuple
) -> list[tuple]:
    """
    Worker function with lattice-based pre-filtering.

    Only generates graphs for non-dominated profile combinations.

    Args:
        args: (n1, n2, profiles1_data, profiles2_data, T)

    Returns:
        List of results: (new_profile_bytes, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx)
    """
    n1, n2, profiles1_data, profiles2_data, T = args

    # Assert canonical ordering at partition level
    assert n1 <= n2, f"Expected n1 <= n2, got n1={n1}, n2={n2}"

    # Pre-compute viable (non-dominated) profile combinations
    viable = _compute_viable_combinations(n1, n2, profiles1_data, profiles2_data, T)

    results = []

    # Process sum viable combinations
    for hash1, hash2, result_prof, result_edges, graphs1_info, graphs2_info in viable['sum']:
        # Enforce canonical ordering: when n1 == n2, only process hash1 <= hash2
        if n1 == n2 and hash1 > hash2:
            continue

        # Generate graphs for this viable combination
        for idx1, op1, depth1 in graphs1_info:
            for idx2, op2, depth2 in graphs2_info:
                # When profiles are identical, enforce idx1 <= idx2
                if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                    continue

                # Calculate depth for sum operation
                contrib1 = depth1 if op1 == "s" or op1 == "v" else depth1 + 1
                contrib2 = depth2 if op2 == "s" or op2 == "v" else depth2 + 1
                sum_depth = max(contrib1, contrib2)

                results.append((
                    result_prof.tobytes(),
                    result_edges,
                    "s",
                    sum_depth,
                    n1, hash1, idx1,
                    n2, hash2, idx2
                ))

    # Process product viable combinations
    for hash1, hash2, result_prof, result_edges, graphs1_info, graphs2_info in viable['product']:
        # Enforce canonical ordering: when n1 == n2, only process hash1 <= hash2
        if n1 == n2 and hash1 > hash2:
            continue

        # Generate graphs for this viable combination
        for idx1, op1, depth1 in graphs1_info:
            for idx2, op2, depth2 in graphs2_info:
                # When profiles are identical, enforce idx1 <= idx2
                if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                    continue

                # Calculate depth for product operation
                contrib1 = depth1 if op1 == "p" or op1 == "v" else depth1 + 1
                contrib2 = depth2 if op2 == "p" or op2 == "v" else depth2 + 1
                prod_depth = max(contrib1, contrib2)

                results.append((
                    result_prof.tobytes(),
                    result_edges,
                    "p",
                    prod_depth,
                    n1, hash1, idx1,
                    n2, hash2, idx2
                ))

    return results


def _process_partition_pair(
    args: tuple
) -> list[tuple]:
    """
    Worker function: process all profile pairs for (n1, n2).

    Args:
        args: (n1, n2, profiles1_data, profiles2_data, T)
            profiles_data: list of (hash, profile_bytes, max_edges, graphs_info)
            graphs_info: list of (idx, op, depth) for each graph with this profile

    Returns:
        List of results: (new_profile_bytes, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx)
    """
    n1, n2, profiles1_data, profiles2_data, T = args

    # Assert canonical ordering at partition level
    assert n1 <= n2, f"Expected n1 <= n2, got n1={n1}, n2={n2}"

    results = []

    for hash1, pbytes1, max_edges1, graphs1_info in profiles1_data:
        p1 = np.frombuffer(pbytes1, dtype=np.int32).copy()

        for hash2, pbytes2, max_edges2, graphs2_info in profiles2_data:
            # Enforce canonical ordering: when n1 == n2, only process hash1 <= hash2
            if n1 == n2 and hash1 > hash2:
                continue

            p2 = np.frombuffer(pbytes2, dtype=np.int32).copy()

            # Fast K_{T,T} checks at profile level
            skip_sum = T is not None and sum_profile_check_ktt(p1, p2, T)
            skip_product = T is not None and product_profile_check_ktt(p1, p2, T)

            if skip_sum and skip_product:
                continue

            # Compute profiles once per profile pair
            sum_profile = None if skip_sum else sum_profile_fast(p1, p2)
            prod_profile = None if skip_product else product_profile_fast(p1, p2)

            # Calculate edge counts arithmetically from children's edge counts
            # edges(S(a,b)) = edges(a) + edges(b)
            # edges(P(a,b)) = edges(a) + edges(b) + n(a) * n(b)
            sum_edges = max_edges1 + max_edges2
            prod_edges = max_edges1 + max_edges2 + n1 * n2

            # For each graph pair
            # Track all graph indices for reconstruction
            for idx1, op1, depth1 in graphs1_info:
                for idx2, op2, depth2 in graphs2_info:
                    # When profiles are identical, enforce idx1 <= idx2
                    if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                        continue

                    # Calculate depth for sum and product operations
                    # depth = max number of operation switches from root to leaf
                    if not skip_sum:
                        # For S(a, b): if child has same op, no switch; if different, +1 switch
                        contrib1 = depth1 if op1 == "s" or op1 == "v" else depth1 + 1
                        contrib2 = depth2 if op2 == "s" or op2 == "v" else depth2 + 1
                        sum_depth = max(contrib1, contrib2)

                        results.append((
                            sum_profile.tobytes(),
                            sum_edges,
                            "s",
                            sum_depth,
                            n1, hash1, idx1,
                            n2, hash2, idx2
                        ))

                    if not skip_product:
                        # For P(a, b): if child has same op, no switch; if different, +1 switch
                        contrib1 = depth1 if op1 == "p" or op1 == "v" else depth1 + 1
                        contrib2 = depth2 if op2 == "p" or op2 == "v" else depth2 + 1
                        prod_depth = max(contrib1, contrib2)

                        results.append((
                            prod_profile.tobytes(),
                            prod_edges,
                            "p",
                            prod_depth,
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
    export_dir: Path | None = None,
    s_max: int = 7,
    t_max: int = 7,
    use_profile_domination: bool = False,
    use_profile_domination_lattice: bool = False,
    use_depth_domination: bool = False,
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
        export_dir: Directory for incremental exports (None = no exports)
        s_max: Maximum s value for exports
        t_max: Maximum t value for exports
        use_profile_domination: Enable profile domination pruning (batch mode)
        use_profile_domination_lattice: Enable lattice-based profile domination (pre-filter)
        use_depth_domination: Enable depth domination pruning
        progress_callback: callback(n, N, added, profiles, total, cumulative, time_sec)

    Returns:
        FastRegistry with all computed data
    """
    if num_workers is None:
        num_workers = cpu_count()

    # Lattice mode implies registry-level domination
    # (lattice pre-filters combinations, but registry still needs to maintain antichain)
    effective_profile_domination = use_profile_domination or use_profile_domination_lattice

    registry = FastRegistry(
        use_profile_domination=effective_profile_domination,
        use_depth_domination=use_depth_domination
    )
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
                max_edges = registry._max_edges[n1][h]
                graphs = registry.get_graphs(n1, h)
                graphs_info = [(idx, g.op, g.depth) for idx, g in enumerate(graphs)]
                profiles1_data.append((h, p.tobytes(), max_edges, graphs_info))

            # Get profile data for n2
            profiles2_data = []
            for h, p in registry.get_all_profiles(n2):
                max_edges = registry._max_edges[n2][h]
                graphs = registry.get_graphs(n2, h)
                graphs_info = [(idx, g.op, g.depth) for idx, g in enumerate(graphs)]
                profiles2_data.append((h, p.tobytes(), max_edges, graphs_info))

            if profiles1_data and profiles2_data:
                work_items.append((n1, n2, profiles1_data, profiles2_data, T))

        # Process in parallel
        # Use lattice worker if lattice mode is enabled
        worker_func = _process_partition_pair_lattice if use_profile_domination_lattice else _process_partition_pair

        if num_workers > 1 and len(work_items) > 1:
            with Pool(processes=min(num_workers, len(work_items))) as pool:
                all_results = pool.map(worker_func, work_items)
        else:
            all_results = [worker_func(item) for item in work_items]

        # Merge results into registry
        # Collect all candidates for batch processing
        all_candidates = []
        for results in all_results:
            all_candidates.extend(results)

        # Use batch addition for better performance with profile domination
        graphs_added = registry.add_batch(target_n, all_candidates)

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

        # Incremental export after each n
        if export_dir:
            for s in range(1, min(s_max, T or s_max) + 1):
                for t in range(s, min(t_max, T or t_max) + 1):
                    export_path = export_dir / f"extremal_K{s}{t}.json"
                    export_extremal_analysis(registry, s, t, export_path)

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
        for edges, profile, profile_hash, graph in candidates:
            # Find this graph's index
            graphs_list = registry._graphs[n][profile_hash]
            idx = graphs_list.index(graph) if graph in graphs_list else 0

            struct_str = registry.reconstruct_structure(
                n, profile_hash, idx
            )

            structures.append({
                "structure": struct_str,
                "edges": edges,
                "depth": graph.depth,
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

                for edges, profile, profile_hash, graph in candidates:
                    if graph.op != "p" and graph.op != "v":
                        # Sum operation = disconnected
                        st_result["all_connected"] = False
                        results["all_connected"] = False

                        # Reconstruct structure for exception
                        graphs_list = registry._graphs[n][profile_hash]
                        idx = graphs_list.index(graph) if graph in graphs_list else 0
                        struct = registry.reconstruct_structure(n, profile_hash, idx)

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
