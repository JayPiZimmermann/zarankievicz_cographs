"""Dynamic programming builder for extremal cographs."""

from __future__ import annotations
import re
from typing import Callable

from .cotree import Cotree, vertex, sum_graphs, product_graphs
from .profile import contains_biclique
from .registry import Registry


def product_would_contain_ktt(
    p1: tuple[int, ...], p2: tuple[int, ...],
    n1: int, n2: int, T: int
) -> bool:
    """
    Fast check if product of two graphs would contain K_{T,T'} for some T' >= T.

    For product G × H, profile_prod[s] = max_{a+c=s} (p1[a] + p2[c]).
    We need to check if profile_prod[T] >= T.

    This is O(T) instead of computing the full O(n1+n2) profile.
    """
    # Check profile_prod[T] = max_{a+c=T} (p1[a] + p2[c])
    max_val = 0
    a_min = max(0, T - n2)
    a_max = min(T, n1)
    for a in range(a_min, a_max + 1):
        c = T - a
        v1 = p1[a] if a < len(p1) else 0
        v2 = p2[c] if c < len(p2) else 0
        max_val = max(max_val, v1 + v2)
        if max_val >= T:
            return True  # Early exit
    return max_val >= T


def sum_would_contain_ktt(
    p1: tuple[int, ...], p2: tuple[int, ...],
    n1: int, n2: int, T: int
) -> bool:
    """
    Fast check if sum of two graphs would contain K_{T,T'} for some T' >= T.

    For sum G + H, profile_sum[i] = max(p1[i], p2[i]).
    We need to check if profile_sum[T] >= T.
    """
    v1 = p1[T] if T < len(p1) else 0
    v2 = p2[T] if T < len(p2) else 0
    return max(v1, v2) >= T


def build_up_to(
    N: int,
    T: int | None = None,
    registry: Registry | None = None,
    progress_callback: Callable[[int, int, int, int, int], None] | None = None
) -> Registry:
    """
    Build registry of extremal cographs up to N vertices.

    Uses dynamic programming: for each target vertex count, combine all pairs
    of smaller graphs via sum and product operations.

    Args:
        N: Maximum number of vertices to build up to
        T: If provided, prune graphs containing K_{T,T} to save memory
        registry: Optional existing registry to extend (will be modified)
        progress_callback: Optional callback(current_n, total_n, graphs_added, profiles, total_for_n)

    Returns:
        Registry containing extremal cographs for n=1..N
    """
    if registry is None:
        registry = Registry()

    start_n = registry.max_n() + 1

    for target_n in range(start_n, N + 1):
        graphs_added = 0

        # Generate all ways to split target_n = n1 + n2 with n1 <= n2
        for n1 in range(1, target_n // 2 + 1):
            n2 = target_n - n1

            # Skip if we don't have graphs for these sizes
            if n1 not in registry._data or n2 not in registry._data:
                continue

            # Get all profile combinations
            profiles1 = list(registry._data[n1].items())
            profiles2 = list(registry._data[n2].items())

            for profile1, graphs1 in profiles1:
                for profile2, graphs2 in profiles2:
                    # Fast profile-level pruning checks
                    if T is not None:
                        skip_sum = sum_would_contain_ktt(profile1, profile2, n1, n2, T)
                        skip_product = product_would_contain_ktt(profile1, profile2, n1, n2, T)
                        if skip_sum and skip_product:
                            continue  # Skip this entire profile pair
                    else:
                        skip_sum = False
                        skip_product = False

                    # For each pair of profiles, try sum and product
                    for _, g1 in graphs1:
                        for _, g2 in graphs2:
                            # Skip if n1 == n2 and g1 > g2 (symmetry)
                            if n1 == n2:
                                s1, s2 = g1.structure_str(), g2.structure_str()
                                if s1 > s2:
                                    continue

                            # Try sum (only if not pre-filtered)
                            if not skip_sum:
                                sum_g = sum_graphs(g1, g2)
                                if registry.add(sum_g):
                                    graphs_added += 1

                            # Try product (only if not pre-filtered)
                            if not skip_product:
                                prod_g = product_graphs(g1, g2)
                                if registry.add(prod_g):
                                    graphs_added += 1

        if progress_callback:
            profiles = registry.profile_count(target_n)
            total_for_n = registry.graph_count(target_n)
            progress_callback(target_n, N, graphs_added, profiles, total_for_n)

    return registry


def parse_structure(s: str) -> Cotree:
    """
    Parse structure string back to Cotree.

    Structure strings are like "P(S(1,2),3)" or "S(1,P(2,3))".

    Args:
        s: Structure string

    Returns:
        Reconstructed Cotree
    """
    s = s.strip()

    # Base case: just a number (leaf = vertex or collection of vertices)
    if s.isdigit():
        n = int(s)
        if n == 1:
            return vertex()
        else:
            # n > 1: this represents n disconnected vertices = sum of n vertices
            result = vertex()
            for _ in range(n - 1):
                result = sum_graphs(result, vertex())
            return result

    # Recursive case: S(...) or P(...)
    if s.startswith("S(") and s.endswith(")"):
        op = "sum"
        inner = s[2:-1]
    elif s.startswith("P(") and s.endswith(")"):
        op = "product"
        inner = s[2:-1]
    else:
        raise ValueError(f"Invalid structure string: {s}")

    # Parse comma-separated children (handling nested parentheses)
    children = _split_args(inner)
    if len(children) < 2:
        raise ValueError(f"Need at least 2 children: {s}")

    child_trees = [parse_structure(c) for c in children]

    if op == "sum":
        result = child_trees[0]
        for child in child_trees[1:]:
            result = sum_graphs(result, child)
    else:
        result = child_trees[0]
        for child in child_trees[1:]:
            result = product_graphs(result, child)

    return result


def _split_args(s: str) -> list[str]:
    """Split comma-separated arguments, respecting parentheses nesting."""
    args = []
    current = []
    depth = 0

    for char in s:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        args.append(''.join(current).strip())

    return args


def build_incremental(
    registry: Registry,
    target_n: int,
    T: int | None = None
) -> int:
    """
    Build graphs for exactly one vertex count.

    This is useful for incremental building or parallel computation.

    Args:
        registry: Registry to extend (must have all n < target_n)
        target_n: The vertex count to build
        T: Optional K_{T,T} pruning threshold

    Returns:
        Number of graphs added
    """
    graphs_added = 0

    for n1 in range(1, target_n // 2 + 1):
        n2 = target_n - n1

        if n1 not in registry._data or n2 not in registry._data:
            continue

        profiles1 = list(registry._data[n1].items())
        profiles2 = list(registry._data[n2].items())

        for profile1, graphs1 in profiles1:
            for profile2, graphs2 in profiles2:
                # Fast profile-level pruning
                if T is not None:
                    skip_sum = sum_would_contain_ktt(profile1, profile2, n1, n2, T)
                    skip_product = product_would_contain_ktt(profile1, profile2, n1, n2, T)
                    if skip_sum and skip_product:
                        continue
                else:
                    skip_sum = False
                    skip_product = False

                for _, g1 in graphs1:
                    for _, g2 in graphs2:
                        if n1 == n2:
                            s1, s2 = g1.structure_str(), g2.structure_str()
                            if s1 > s2:
                                continue

                        if not skip_sum:
                            sum_g = sum_graphs(g1, g2)
                            if registry.add(sum_g):
                                graphs_added += 1

                        if not skip_product:
                            prod_g = product_graphs(g1, g2)
                            if registry.add(prod_g):
                                graphs_added += 1

    return graphs_added


def estimate_complexity(N: int) -> str:
    """
    Estimate the computational complexity of building up to N vertices.

    This is a rough estimate based on the combinatorial explosion.

    Args:
        N: Maximum vertex count

    Returns:
        Human-readable complexity estimate
    """
    # Very rough estimate: number of graphs grows roughly exponentially
    # The number of cographs on n vertices is sequence A000084 in OEIS
    estimated_cographs = [1, 1, 2, 4, 10, 24, 66, 180, 522, 1532, 4624, 14136]

    if N < len(estimated_cographs):
        return f"~{estimated_cographs[N]} cographs on {N} vertices"
    else:
        # Exponential extrapolation
        ratio = estimated_cographs[-1] / estimated_cographs[-2]
        estimate = int(estimated_cographs[-1] * (ratio ** (N - len(estimated_cographs) + 1)))
        return f"~{estimate:,} cographs on {N} vertices (rough estimate)"
