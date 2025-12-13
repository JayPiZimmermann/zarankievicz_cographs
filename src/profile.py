"""K_{i,j} profile computation for cographs.

The profile of a graph G on n vertices is a tuple (p[0], p[1], ..., p[n]) where
p[i] = max{j : K_{i,j} is a subgraph of G}.

Key property: K_{s,t} = K_{t,s}, so profile[s] >= t implies profile[t] >= s.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cotree import Cotree


def compute_profile(cotree: Cotree) -> tuple[int, ...]:
    """
    Compute the K_{i,j} profile of a cotree.

    For a graph G on n vertices, profile[i] = max j such that K_{i,j} is a subgraph.

    Returns:
        Tuple of length n+1 where result[i] = max j with K_{i,j} in G.
    """
    if cotree.op == "vertex":
        # Single vertex: K_{0,1} exists (trivially), K_{1,0} exists, no K_{1,j} for j>0
        return (1, 0)

    # For sum/product, compute from children profiles
    if len(cotree.children) == 2:
        p1 = cotree.children[0].profile
        p2 = cotree.children[1].profile
        n1 = cotree.children[0].n
        n2 = cotree.children[1].n

        if cotree.op == "sum":
            return sum_profile(p1, p2, n1, n2)
        else:  # product
            return product_profile(p1, p2, n1, n2)
    else:
        # Multiple children: fold pairwise
        result_profile = cotree.children[0].profile
        result_n = cotree.children[0].n

        for child in cotree.children[1:]:
            if cotree.op == "sum":
                result_profile = sum_profile(result_profile, child.profile, result_n, child.n)
            else:
                result_profile = product_profile(result_profile, child.profile, result_n, child.n)
            result_n += child.n

        return result_profile


def sum_profile(p1: tuple[int, ...], p2: tuple[int, ...], n1: int, n2: int) -> tuple[int, ...]:
    """
    Compute profile of disjoint union (sum) of two graphs.

    For sum G + H:
    - K_{s,t} in G+H iff K_{s,t} in G or K_{s,t} in H
    - profile_sum[0] = n1 + n2
    - profile_sum[i] = max(p1[i], p2[i]) for i > 0

    Args:
        p1: Profile of first graph (length n1+1)
        p2: Profile of second graph (length n2+1)
        n1: Vertex count of first graph
        n2: Vertex count of second graph

    Returns:
        Profile of sum graph (length n1+n2+1)
    """
    n = n1 + n2
    result = [0] * (n + 1)

    # profile[0] = total vertices (K_{0,n} always exists)
    result[0] = n

    # For i > 0, take max of both profiles (extended with 0 for out-of-range)
    for i in range(1, n + 1):
        v1 = p1[i] if i < len(p1) else 0
        v2 = p2[i] if i < len(p2) else 0
        result[i] = max(v1, v2)

    return tuple(result)


def product_profile(p1: tuple[int, ...], p2: tuple[int, ...], n1: int, n2: int) -> tuple[int, ...]:
    """
    Compute profile of complete join (product) of two graphs.

    For product G x H (all edges between G and H):
    - K_{s,t} in G x H iff there exist a,b,c,d with:
      - a + c = s (left side split between G and H)
      - b + d = t (right side split between G and H)
      - K_{a,b} in G (profile_G[a] >= b)
      - K_{c,d} in H (profile_H[c] >= d)
    - Formula: profile_prod[s] = max_{a+c=s} (p1[a] + p2[c])

    Args:
        p1: Profile of first graph (length n1+1)
        p2: Profile of second graph (length n2+1)
        n1: Vertex count of first graph
        n2: Vertex count of second graph

    Returns:
        Profile of product graph (length n1+n2+1)
    """
    n = n1 + n2
    result = [0] * (n + 1)

    # profile[0] = total vertices
    result[0] = n

    # Max-convolution: profile_prod[s] = max_{a+c=s} (p1[a] + p2[c])
    for s in range(n + 1):
        best = 0
        # a ranges from 0 to min(s, n1)
        # c = s - a ranges from s to max(0, s-n1)
        a_min = max(0, s - n2)
        a_max = min(s, n1)
        for a in range(a_min, a_max + 1):
            c = s - a
            val1 = p1[a] if a < len(p1) else 0
            val2 = p2[c] if c < len(p2) else 0
            best = max(best, val1 + val2)
        result[s] = best

    return tuple(result)


def contains_biclique(profile: tuple[int, ...], s: int, t: int) -> bool:
    """
    Check if a graph with given profile contains K_{s,t}.

    Args:
        profile: The K_{i,j} profile
        s: Left side of biclique
        t: Right side of biclique

    Returns:
        True if K_{s,t} is contained in the graph
    """
    # K_{s,t} = K_{t,s}, so check both ways
    if s < len(profile) and profile[s] >= t:
        return True
    if t < len(profile) and profile[t] >= s:
        return True
    return False


def profile_avoids(profile: tuple[int, ...], s: int, t: int) -> bool:
    """
    Check if a graph with given profile avoids K_{s,t} (is K_{s,t}-free).

    Args:
        profile: The K_{i,j} profile
        s: Left side of forbidden biclique
        t: Right side of forbidden biclique

    Returns:
        True if the graph does NOT contain K_{s,t}
    """
    return not contains_biclique(profile, s, t)


def truncate_profile(profile: tuple[int, ...], T: int) -> tuple[int, ...]:
    """
    Truncate profile to first T+1 entries.

    For tracking K_{s,t} with s,t <= T, we only need profile[0:T+1].

    Args:
        profile: Full profile
        T: Maximum index to keep

    Returns:
        Truncated profile of length min(T+1, len(profile))
    """
    return profile[:T + 1]


def profile_dominates(p1: tuple[int, ...], p2: tuple[int, ...]) -> bool:
    """
    Check if profile p1 dominates p2 (p1 >= p2 componentwise).

    If p1 dominates p2, then any K_{s,t} in p2 is also in p1.

    Args:
        p1: First profile
        p2: Second profile

    Returns:
        True if p1[i] >= p2[i] for all i
    """
    # Extend shorter profile with zeros
    max_len = max(len(p1), len(p2))
    for i in range(max_len):
        v1 = p1[i] if i < len(p1) else 0
        v2 = p2[i] if i < len(p2) else 0
        if v1 < v2:
            return False
    return True


def normalize_profile(profile: tuple[int, ...]) -> tuple[int, ...]:
    """
    Normalize profile to canonical form for hashing.

    Removes trailing zeros (except we always keep at least 2 elements).

    Args:
        profile: Input profile

    Returns:
        Normalized profile
    """
    # Find last non-zero entry
    last_nonzero = 0
    for i in range(len(profile)):
        if profile[i] != 0:
            last_nonzero = i

    # Keep at least 2 elements (profile[0] and profile[1])
    end = max(last_nonzero + 1, 2)
    return profile[:end]
