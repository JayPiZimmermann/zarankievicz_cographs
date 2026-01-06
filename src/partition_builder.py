"""Partition-based builder: compute n from all partitions n'+n''=n.

This approach makes each step independent of N, computing only based on
previously computed values. This enables true incremental computation.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Callable, Literal
from multiprocessing import Pool, cpu_count
import numpy as np

from .compact_storage import FastRegistry, CompactGraph
from .profile_ops import (
    sum_profile_fast, product_profile_fast,
    sum_profile_check_ktt, product_profile_check_ktt,
    sum_profile_check_kst, product_profile_check_kst
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


def _compute_viable_combinations_lattice(
    n1: int,
    n2: int,
    profiles1_data: list,
    profiles2_data: list,
    T: int | None,
    S: int | None = None,
    S_max: int | None = None
) -> dict:
    """
    Pre-compute which profile combinations are non-dominated (lattice mode).

    This filters combinations at the profile pair level before expanding to graph pairs.

    Args:
        n1, n2: Vertex counts
        profiles1_data: List of (hash, profile_bytes, max_edges, graphs_info)
        profiles2_data: List of (hash, profile_bytes, max_edges, graphs_info)
        T: K_{T,T} pruning threshold (legacy, kept for backward compatibility)
        S: Profile truncation index (only store profile up to index S)
        S_max: Prune K_{S,S_max} instead of K_{T,T}

    Returns:
        {
            'sum': [(p1_hash, p2_hash, result_profile_bytes, result_edges, graphs1_info, graphs2_info), ...],
            'product': [(p1_hash, p2_hash, result_profile_bytes, result_edges, graphs1_info, graphs2_info), ...]
        }
    """
    sum_candidates = []
    prod_candidates = []

    # Determine pruning parameters
    prune_s = S if S_max is not None else T
    prune_t = S_max if S_max is not None else T

    # Generate all profile combinations
    for hash1, pbytes1, max_edges1, graphs1_info in profiles1_data:
        p1 = np.frombuffer(pbytes1, dtype=np.int32).copy()

        for hash2, pbytes2, max_edges2, graphs2_info in profiles2_data:
            # Enforce canonical ordering: when n1 == n2, only process hash1 <= hash2
            if n1 == n2 and hash1 > hash2:
                continue

            p2 = np.frombuffer(pbytes2, dtype=np.int32).copy()

            # Fast K_{s,t} checks at profile level
            skip_sum = False
            skip_product = False
            if prune_s is not None and prune_t is not None:
                skip_sum = sum_profile_check_kst(p1, p2, prune_s, prune_t)
                skip_product = product_profile_check_kst(p1, p2, prune_s, prune_t)

            if skip_sum and skip_product:
                continue

            # Compute profiles once per profile pair (with optional truncation)
            sum_profile = None if skip_sum else sum_profile_fast(p1, p2, S)
            prod_profile = None if skip_product else product_profile_fast(p1, p2, S)

            # Calculate edge counts
            sum_edges = max_edges1 + max_edges2
            prod_edges = max_edges1 + max_edges2 + n1 * n2

            if not skip_sum:
                sum_candidates.append((
                    hash1, hash2,
                    sum_profile,
                    sum_edges,
                    graphs1_info,
                    graphs2_info
                ))

            if not skip_product:
                prod_candidates.append((
                    hash1, hash2,
                    prod_profile,
                    prod_edges,
                    graphs1_info,
                    graphs2_info
                ))

    # Filter to antichain
    sum_viable = _filter_antichain(sum_candidates)
    prod_viable = _filter_antichain(prod_candidates)

    return {
        'sum': sum_viable,
        'product': prod_viable
    }


def _process_partition_pair(
    args: tuple
) -> list[tuple]:
    """
    Worker function: process all profile pairs for (n1, n2).

    Args:
        args: (n1, n2, profiles1_data, profiles2_data, T, use_lattice, S, S_max)
            profiles_data: list of (hash, profile_bytes, max_edges, graphs_info)
            graphs_info: list of (idx, op, depth) for each graph with this profile
            use_lattice: if True, use lattice-based pre-filtering
            S: Optional profile truncation index
            S_max: Optional K_{S,S_max} pruning threshold

    Returns:
        List of results: (new_profile_bytes, edges, op, depth, c1_n, c1_hash, c1_idx, c2_n, c2_hash, c2_idx)
    """
    n1, n2, profiles1_data, profiles2_data, T, use_lattice, S, S_max = args

    # Assert canonical ordering at partition level
    assert n1 <= n2, f"Expected n1 <= n2, got n1={n1}, n2={n2}"

    results = []

    if use_lattice:
        # Lattice mode: pre-filter profile combinations
        viable = _compute_viable_combinations_lattice(n1, n2, profiles1_data, profiles2_data, T, S, S_max)

        # Process sum operations
        for hash1, hash2, sum_profile, sum_edges, graphs1_info, graphs2_info in viable['sum']:
            for idx1, op1, depth1 in graphs1_info:
                for idx2, op2, depth2 in graphs2_info:
                    # When profiles are identical, enforce idx1 <= idx2
                    if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                        continue

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

        # Process product operations
        for hash1, hash2, prod_profile, prod_edges, graphs1_info, graphs2_info in viable['product']:
            for idx1, op1, depth1 in graphs1_info:
                for idx2, op2, depth2 in graphs2_info:
                    # When profiles are identical, enforce idx1 <= idx2
                    if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                        continue

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

    else:
        # Standard mode: process all profile pairs
        # Determine pruning parameters
        prune_s = S if S_max is not None else T
        prune_t = S_max if S_max is not None else T

        for hash1, pbytes1, max_edges1, graphs1_info in profiles1_data:
            p1 = np.frombuffer(pbytes1, dtype=np.int32).copy()

            for hash2, pbytes2, max_edges2, graphs2_info in profiles2_data:
                # Enforce canonical ordering: when n1 == n2, only process hash1 <= hash2
                if n1 == n2 and hash1 > hash2:
                    continue

                p2 = np.frombuffer(pbytes2, dtype=np.int32).copy()

                # Fast K_{s,t} checks at profile level
                skip_sum = False
                skip_product = False
                if prune_s is not None and prune_t is not None:
                    skip_sum = sum_profile_check_kst(p1, p2, prune_s, prune_t)
                    skip_product = product_profile_check_kst(p1, p2, prune_s, prune_t)

                if skip_sum and skip_product:
                    continue

                # Compute profiles once per profile pair (with optional truncation)
                sum_profile = None if skip_sum else sum_profile_fast(p1, p2, S)
                prod_profile = None if skip_product else product_profile_fast(p1, p2, S)

                # Calculate edge counts arithmetically from children's edge counts
                sum_edges = max_edges1 + max_edges2
                prod_edges = max_edges1 + max_edges2 + n1 * n2

                # For each graph pair
                for idx1, op1, depth1 in graphs1_info:
                    for idx2, op2, depth2 in graphs2_info:
                        # When profiles are identical, enforce idx1 <= idx2
                        if n1 == n2 and hash1 == hash2 and idx1 > idx2:
                            continue

                        # Calculate depth for sum and product operations
                        if not skip_sum:
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


def build_partition(
    target_n: int,
    registry: FastRegistry,
    T: int | None = None,
    num_workers: int | None = None,
    use_profile_domination_lattice: bool = False,
    progress_callback: Callable[[int, int, int, int, float], None] | None = None,
    S: int | None = None,
    S_max: int | None = None
) -> int:
    """
    Build all graphs of size target_n from partitions n1 + n2 = target_n.

    This function only depends on already-computed data (n < target_n),
    making it independent of any maximum N parameter.

    Args:
        target_n: The vertex count to compute
        registry: FastRegistry with data for n < target_n already computed
        T: Pruning threshold for K_{T,T} (legacy, prefer S/S_max)
        num_workers: Number of parallel workers (default: cpu_count)
        use_profile_domination_lattice: Enable lattice-based profile domination
        progress_callback: callback(n, added, profiles, total, time_sec)
        S: Profile truncation index (only store profile up to index S)
        S_max: Prune K_{S,S_max} instead of K_{T,T}

    Returns:
        Number of graphs added at target_n
    """
    if num_workers is None:
        num_workers = cpu_count()

    start_time = time.time()

    # Prepare work items for all partitions n1 + n2 = target_n
    work_items = []

    for n1 in range(1, target_n // 2 + 1):
        n2 = target_n - n1

        # Verify data exists for both n1 and n2
        if registry.profile_count(n1) == 0 or registry.profile_count(n2) == 0:
            continue

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
            work_items.append((n1, n2, profiles1_data, profiles2_data, T, use_profile_domination_lattice, S, S_max))

    # Process in parallel
    if num_workers > 1 and len(work_items) > 1:
        with Pool(processes=min(num_workers, len(work_items))) as pool:
            all_results = pool.map(_process_partition_pair, work_items)
    else:
        all_results = [_process_partition_pair(item) for item in work_items]

    # Merge results into registry
    all_candidates = []
    for results in all_results:
        all_candidates.extend(results)

    # Use batch addition for better performance
    graphs_added = registry.add_batch(target_n, all_candidates)

    elapsed = time.time() - start_time

    if progress_callback:
        progress_callback(
            target_n,
            graphs_added,
            registry.profile_count(target_n),
            registry.graph_count(target_n),
            elapsed
        )

    return graphs_added


def build_range(
    start_n: int,
    end_n: int,
    registry: FastRegistry | None = None,
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
    progress_callback: Callable[[int, int, int, int, int, int, float], None] | None = None,
    S: int | None = None,
    S_max: int | None = None
) -> FastRegistry:
    """
    Build registry for range [start_n, end_n] using partition-based approach.

    Each n is computed independently from partitions, without depending on end_n.
    This enables true incremental computation where you can extend to larger n
    without recomputing.

    Args:
        start_n: Starting vertex count (must have data for n < start_n in registry)
        end_n: Ending vertex count
        registry: Existing registry (or None to start fresh from n=1)
        T: Pruning threshold for K_{T,T} (legacy, prefer S/S_max)
        num_workers: Number of parallel workers (default: cpu_count)
        checkpoint_dir: Directory for checkpoints (None = no checkpoints)
        checkpoint_interval: Save checkpoint every N vertex counts
        export_dir: Directory for incremental exports (None = no exports)
        s_max: Maximum s value for exports (overridden by S if provided)
        t_max: Maximum t value for exports (overridden by S_max if provided)
        use_profile_domination: Enable profile domination pruning (batch mode)
        use_profile_domination_lattice: Enable lattice-based profile domination
        use_depth_domination: Enable depth domination pruning
        progress_callback: callback(n, end_n, added, profiles, total, cumulative, time_sec)
        S: Profile truncation index (only store profile up to index S)
        S_max: Prune K_{S,S_max} and export up to K_{i,S_max} for i <= S

    Returns:
        FastRegistry with computed data
    """
    if num_workers is None:
        num_workers = cpu_count()

    # Initialize or use existing registry
    if registry is None:
        effective_profile_domination = use_profile_domination or use_profile_domination_lattice
        registry = FastRegistry(
            use_profile_domination=effective_profile_domination,
            use_depth_domination=use_depth_domination
        )

    # Try to load checkpoint if starting fresh
    actual_start = start_n
    if checkpoint_dir and start_n == 2:
        checkpoint_path = checkpoint_dir / f"checkpoint_partition_T{T}.pkl"
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            registry, completed_n = FastRegistry.load_checkpoint(checkpoint_path)
            actual_start = completed_n + 1
            print(f"Resuming from n={actual_start}")

    cumulative = sum(registry.graph_count(n) for n in range(1, actual_start))

    # Build each n from partitions
    for target_n in range(actual_start, end_n + 1):
        step_start = time.time()

        graphs_added = build_partition(
            target_n=target_n,
            registry=registry,
            T=T,
            num_workers=num_workers,
            use_profile_domination_lattice=use_profile_domination_lattice,
            progress_callback=None,  # Handle progress at this level
            S=S,
            S_max=S_max
        )

        cumulative += registry.graph_count(target_n)
        elapsed = time.time() - step_start

        if progress_callback:
            progress_callback(
                target_n, end_n, graphs_added,
                registry.profile_count(target_n),
                registry.graph_count(target_n),
                cumulative,
                elapsed
            )

        # Incremental export after each n
        if export_dir:
            from .fast_builder import export_extremal_analysis
            # Determine export range based on S and S_max if provided
            effective_s_max = S if S is not None else s_max

            # When S is set but S_max is not: export K_{i,j} for i <= S and j <= current n
            # This makes the cache N-independent (no max limit on j)
            if S is not None and S_max is None:
                # Export up to current n, no upper limit
                for s in range(1, effective_s_max + 1):
                    for t in range(s, target_n + 1):
                        export_path = export_dir / f"extremal_K{s}{t}.json"
                        export_extremal_analysis(registry, s, t, export_path)
            else:
                # S_max is set OR neither S nor S_max is set: use appropriate limits
                effective_t_max = S_max if S_max is not None else t_max
                for s in range(1, effective_s_max + 1):
                    for t in range(s, min(effective_t_max, target_n) + 1):
                        export_path = export_dir / f"extremal_K{s}{t}.json"
                        export_extremal_analysis(registry, s, t, export_path)

        # Checkpoint
        if checkpoint_dir and target_n % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_partition_T{T}.pkl"
            registry.save_checkpoint(checkpoint_path, target_n)

    # Final checkpoint
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / f"checkpoint_partition_T{T}.pkl"
        registry.save_checkpoint(checkpoint_path, end_n)

    return registry


def check_conjecture_partition(
    registry: FastRegistry,
    s_min: int = 2,
    s_max: int = 7,
    t_max: int = 7
) -> dict:
    """
    Check if all extremal K_{s,t}-free graphs are connected.

    Args:
        registry: FastRegistry with computed data
        s_min: Minimum s to check
        s_max: Maximum s to check
        t_max: Maximum t to check

    Returns:
        Dictionary with results
    """
    from .fast_builder import check_conjecture_fast
    return check_conjecture_fast(registry, s_min, s_max, t_max)


def _collect_product_component_sizes(
    registry: FastRegistry,
    n: int,
    profile_hash: int,
    graph_idx: int
) -> list[int]:
    """
    Collect all leaf component sizes from a flattened product tree.

    For P(a, P(b, c)) this returns sizes of [a, b, c] (flattened).
    For P(a, S(...)) this returns [size(a), size(S(...))].
    """
    graph = registry._graphs[n][profile_hash][graph_idx]

    if graph.op == "v":
        return [1]

    if graph.op == "s":
        # Sum node - return total size as single component
        return [n]

    # Product node - recursively collect from children
    result = []

    # Child 1
    child1_graph = registry._graphs[graph.child1_n][graph.child1_profile_hash][graph.child1_idx]
    if child1_graph.op == "p":
        # Flatten nested products
        result.extend(_collect_product_component_sizes(
            registry, graph.child1_n, graph.child1_profile_hash, graph.child1_idx
        ))
    else:
        # Sum or vertex - don't flatten further
        result.append(graph.child1_n)

    # Child 2
    child2_graph = registry._graphs[graph.child2_n][graph.child2_profile_hash][graph.child2_idx]
    if child2_graph.op == "p":
        # Flatten nested products
        result.extend(_collect_product_component_sizes(
            registry, graph.child2_n, graph.child2_profile_hash, graph.child2_idx
        ))
    else:
        # Sum or vertex - don't flatten further
        result.append(graph.child2_n)

    return result


def _can_partition_to_sum(sizes: list[int], target: int) -> bool:
    """
    Check if a subset of sizes can sum to exactly target.

    Uses dynamic programming subset sum.
    """
    if target == 0:
        return True
    if not sizes or target < 0:
        return False

    total = sum(sizes)
    if target > total:
        return False

    # DP: dp[i] = True if sum i is achievable
    dp = [False] * (target + 1)
    dp[0] = True

    for size in sizes:
        # Iterate backwards to avoid using same element twice
        for i in range(target, size - 1, -1):
            if dp[i - size]:
                dp[i] = True

    return dp[target]


def _get_start_index(profile: np.ndarray) -> int:
    """
    Get the start index of a biclique-profile.

    Start index is the minimal i >= 1 such that profile[i] is "constrained",
    meaning it's less than what an unconstrained graph could have.

    For a complete graph K_n:
    - profile[i] = n - i (the max t such that K_{i,t} is contained)

    So start index is the first i where profile[i] < n - i.
    """
    n = int(profile[0])
    for i in range(1, len(profile)):
        # For complete graph on n vertices, K_{i, n-i} is contained
        # If profile[i] < n - i, this position is constrained
        expected_unconstrained = n - i if i < n else 0
        if i < len(profile) and profile[i] < expected_unconstrained:
            return i
    return len(profile)  # No constraint found (complete graph)


def check_theorem(
    registry: FastRegistry,
    s_max: int = 7,
    t_max: int = 7,
    small_n_threshold: int = 10
) -> dict:
    """
    Check the main theorem (Theorem 4):
    For biclique-profile P with start index s >= 2, extremal graphs are
    P(G_1, G_2) where the multiplication components can be partitioned
    such that one subset sums to s-1.

    This checks ACTUAL PROFILES from the registry, not just K_{s,t} pairs.

    Args:
        registry: FastRegistry with computed data
        s_max: Maximum s to check (for filtering profiles)
        t_max: Maximum t to check (unused, kept for compatibility)
        small_n_threshold: For n <= this, include full extremal graph structures

    Returns:
        Dictionary keyed by n, containing:
        - counterexamples: list of violations with info about other graphs
        - profiles: dict by profile_hash showing all extremal graphs
        - summary: statistics for this n
    """
    results = {}

    for n in range(1, registry.max_n() + 1):
        n_result = {
            "counterexamples": [],
            "profiles": {},
            "summary": {
                "total_profiles": 0,
                "profiles_with_start_index_ge_2": 0,
                "profiles_all_satisfy": 0,
                "profiles_some_satisfy": 0,
                "profiles_none_satisfy": 0
            }
        }

        # Iterate through ALL profiles for this n
        for profile_hash, profile in registry._profiles[n].items():
            n_result["summary"]["total_profiles"] += 1

            start_index = _get_start_index(profile)

            # Only check profiles with start index >= 2
            if start_index < 2 or start_index > s_max:
                continue

            n_result["summary"]["profiles_with_start_index_ge_2"] += 1

            graphs = registry._graphs[n].get(profile_hash, [])
            if not graphs:
                continue

            expected_sum = start_index - 1  # s - 1

            profile_info = {
                "profile": profile.tolist(),
                "profile_hash": profile_hash,
                "start_index": start_index,
                "n": n,
                "edge_count": graphs[0].edges,
                "graph_count": len(graphs),
                "graphs_satisfying": [],
                "graphs_not_satisfying": [],
                "has_satisfying_graph": False
            }

            for idx, graph in enumerate(graphs):
                struct_str = registry.reconstruct_structure(n, profile_hash, idx)

                # Get flattened product component sizes
                component_sizes = _collect_product_component_sizes(
                    registry, n, profile_hash, idx
                )

                graph_info = {
                    "structure": struct_str,
                    "edges": graph.edges,
                    "depth": graph.depth,
                    "last_op": "product" if graph.op == "p" else ("sum" if graph.op == "s" else "vertex"),
                    "immediate_components": [graph.child1_n, graph.child2_n] if graph.op != "v" else [],
                    "flattened_components": component_sizes,
                    "satisfies_theorem": False,
                    "reason": ""
                }

                # Check theorem conditions
                if graph.op == "v":
                    graph_info["satisfies_theorem"] = True
                    graph_info["reason"] = "single vertex"
                elif graph.op == "s":
                    graph_info["satisfies_theorem"] = False
                    graph_info["reason"] = "disconnected (sum at root)"
                elif graph.op == "p":
                    if _can_partition_to_sum(component_sizes, expected_sum):
                        graph_info["satisfies_theorem"] = True
                        graph_info["reason"] = f"components {component_sizes} partition with sum {expected_sum}"
                    else:
                        graph_info["satisfies_theorem"] = False
                        graph_info["reason"] = f"components {component_sizes} cannot partition with sum {expected_sum}"

                if graph_info["satisfies_theorem"]:
                    profile_info["graphs_satisfying"].append(graph_info)
                    profile_info["has_satisfying_graph"] = True
                else:
                    profile_info["graphs_not_satisfying"].append(graph_info)

            # Classify this profile
            if len(profile_info["graphs_not_satisfying"]) == 0:
                n_result["summary"]["profiles_all_satisfy"] += 1
            elif profile_info["has_satisfying_graph"]:
                n_result["summary"]["profiles_some_satisfy"] += 1
            else:
                n_result["summary"]["profiles_none_satisfy"] += 1

            # Add counterexamples with context
            for g in profile_info["graphs_not_satisfying"]:
                n_result["counterexamples"].append({
                    "profile": profile.tolist(),
                    "start_index": start_index,
                    "expected_partition_sum": expected_sum,
                    "structure": g["structure"],
                    "flattened_components": g["flattened_components"],
                    "reason": g["reason"],
                    "has_other_satisfying_graph": profile_info["has_satisfying_graph"],
                    "satisfying_alternatives": [
                        alt["structure"] for alt in profile_info["graphs_satisfying"]
                    ] if profile_info["has_satisfying_graph"] else []
                })

            # Store full profile info for all n
            # Add formatted profile with infinity
            profile_info["profile_display"] = ["∞"] + profile.tolist()[1:]
            n_result["profiles"][str(profile_hash)] = profile_info

        results[n] = n_result

    return results
