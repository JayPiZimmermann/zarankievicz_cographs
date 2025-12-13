"""Incremental builder with per-n saving and export updates."""

from __future__ import annotations
from typing import Callable
from datetime import datetime

from .cotree import Cotree, vertex, sum_graphs, product_graphs
from .builder import product_would_contain_ktt, sum_would_contain_ktt
from .incremental_cache import IncrementalRegistry


def build_incremental(
    run_name: str,
    N: int,
    T: int | None = None,
    s_max: int = 7,
    t_max: int = 7,
    progress_callback: Callable[[int, int, int, int, int, int, int], None] | None = None
) -> IncrementalRegistry:
    """
    Build registry incrementally with saves after each n.

    Args:
        run_name: Name for this run (used in folder name)
        N: Maximum vertex count
        T: Pruning threshold (skip graphs containing K_{T,T})
        s_max: Maximum s for exports
        t_max: Maximum t for exports
        progress_callback: callback(n, N, added, skipped_profiles, profiles, total, cumulative)

    Returns:
        IncrementalRegistry with all computed data
    """
    registry = IncrementalRegistry(run_name, N, T, s_max=s_max, t_max=t_max)

    # Check if run exists and is complete
    if registry.is_complete():
        print(f"Run '{run_name}' already complete (N={N}, T={T})")
        registry.load_existing()
        return registry

    # Load existing progress
    start_n = registry.load_existing()
    if start_n > 1:
        print(f"Resuming run '{run_name}' from n={start_n}")
    else:
        registry.start_time = datetime.now().isoformat()
        # Initialize with single vertex
        v = vertex()
        registry.add(v)
        registry.save_n(1)
        registry.update_exports(1)
        start_n = 2

    cumulative_graphs = sum(registry.graph_count(n) for n in range(1, start_n))

    for target_n in range(start_n, N + 1):
        graphs_added = 0
        skipped_profiles = 0

        # Generate all ways to split target_n = n1 + n2 with n1 <= n2
        for n1 in range(1, target_n // 2 + 1):
            n2 = target_n - n1

            # Get all profile combinations
            profiles1 = list(registry._data.get(n1, {}).items())
            profiles2 = list(registry._data.get(n2, {}).items())

            for profile1, graphs1 in profiles1:
                for profile2, graphs2 in profiles2:
                    # Fast profile-level pruning
                    if T is not None:
                        skip_sum = sum_would_contain_ktt(profile1, profile2, n1, n2, T)
                        skip_product = product_would_contain_ktt(profile1, profile2, n1, n2, T)
                        if skip_sum and skip_product:
                            skipped_profiles += 1
                            continue
                    else:
                        skip_sum = False
                        skip_product = False

                    # For each pair of graphs
                    for _, g1 in graphs1:
                        for _, g2 in graphs2:
                            # Symmetry: skip if n1 == n2 and g1 > g2
                            if n1 == n2:
                                s1, s2 = g1.structure_str(), g2.structure_str()
                                if s1 > s2:
                                    continue

                            # Try sum
                            if not skip_sum:
                                sum_g = sum_graphs(g1, g2)
                                if registry.add(sum_g):
                                    graphs_added += 1

                            # Try product
                            if not skip_product:
                                prod_g = product_graphs(g1, g2)
                                if registry.add(prod_g):
                                    graphs_added += 1

        # Save this n to disk
        registry.save_n(target_n)

        # Update exports incrementally
        registry.update_exports(target_n)

        cumulative_graphs += registry.graph_count(target_n)

        if progress_callback:
            progress_callback(
                target_n, N, graphs_added, skipped_profiles,
                registry.profile_count(target_n),
                registry.graph_count(target_n),
                cumulative_graphs
            )

    return registry


def analyze_extremal_structure(
    registry: IncrementalRegistry,
    s: int, t: int,
    n_max: int | None = None
) -> list[dict]:
    """
    Analyze the structure of extremal K_{s,t}-free graphs.

    Returns analysis of last operation and component sizes.
    """
    from .export import analyze_extremal

    if n_max is None:
        n_max = registry.max_n()

    results = []
    for n in range(1, n_max + 1):
        graphs = registry.get_avoiding(n, s, t)
        if not graphs:
            continue

        entry = {
            "n": n,
            "ex": graphs[0][0],
            "count": len(graphs),
            "all_connected": all(g.op == "product" for _, g in graphs),
            "structures": []
        }

        for edges, cotree in graphs:
            analysis = analyze_extremal(cotree)
            entry["structures"].append({
                "structure": analysis["structure_str"],
                "last_op": analysis["last_op"],
                "component_sizes": analysis["component_sizes"],
                "component_edges": analysis["component_edges"],
                "depth": analysis["depth"]
            })

        results.append(entry)

    return results


def check_conjecture(
    registry: IncrementalRegistry,
    s_min: int = 2,
    s_max: int = 7,
    t_max: int = 7
) -> dict:
    """
    Check if extremal K_{s,t}-free graphs (s,t >= 2) are all connected
    (last operation is product) and analyze component sizes.

    Returns summary of findings.
    """
    results = {
        "all_connected": True,
        "exceptions": [],
        "component_size_patterns": {},
        "by_st": {}
    }

    for s in range(s_min, s_max + 1):
        for t in range(s, t_max + 1):
            key = f"K{s}{t}"
            analysis = analyze_extremal_structure(registry, s, t)

            st_result = {
                "all_connected": True,
                "exceptions": [],
                "typical_first_component": [],
            }

            for entry in analysis:
                n = entry["n"]
                if not entry["all_connected"]:
                    st_result["all_connected"] = False
                    results["all_connected"] = False
                    for struct in entry["structures"]:
                        if struct["last_op"] != "product":
                            exception = {
                                "n": n,
                                "s": s, "t": t,
                                "structure": struct["structure"],
                                "last_op": struct["last_op"]
                            }
                            st_result["exceptions"].append(exception)
                            results["exceptions"].append(exception)

                # Track first component sizes for connected graphs
                for struct in entry["structures"]:
                    if struct["last_op"] == "product" and struct["component_sizes"]:
                        first_size = min(struct["component_sizes"])
                        st_result["typical_first_component"].append((n, first_size))

            results["by_st"][key] = st_result

    return results
