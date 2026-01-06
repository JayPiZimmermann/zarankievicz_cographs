#!/usr/bin/env python3
"""
Verify Theorem small_cases_classification for s <= t <= min(12, 2s-1).

The theorem states: For s,t in [12] with s <= t <= 2s - 1 and n >= 2t - 1,
any (s,t)-extremal cograph on n vertices has structure G_0 x G_1 where
|G_0| = s-1 and G_1 is (1,t)-extremal.

Proof approach:
For large n, any (s,t)-extremal cograph has structure:
    G = H_0 + G_0 x (H_1 + k * K_t)

We verify that for all (s,t) pairs in the range, the maximizing triples
(H_0, G_0, H_1) have H_0 empty and H_1 is (1,t)-extremal.

Bounds:
- H_0: at most binom(s-1, 2) + 1 vertices (use cached extremal numbers)
- G_0: exactly s-1 vertices (no lattice reduction: need actual graphs)
- H_1: at most (t-1) components, each at most 2(t-1) vertices with max_degree < t
"""

import sys
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Iterator
from math import comb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profile_ops import (
    sum_profile_fast, product_profile_fast,
    profile_avoids_kst
)


# =============================================================================
# Cached extremal numbers from exports
# =============================================================================

def load_cached_extremal_numbers(exports_dir: Path, s: int, t: int) -> dict[int, int]:
    """
    Load cached extremal numbers ex(n, K_{s,t}) from exports directory.

    Returns:
        Dict: n -> ex(n, K_{s,t})
    """
    filename = f"extremal_K{s}{t}.json"
    filepath = exports_dir / filename

    if not filepath.exists():
        return {}

    with open(filepath, 'r') as f:
        data = json.load(f)

    result = {}
    for n_str, info in data.get('extremal_by_n', {}).items():
        result[int(n_str)] = info['ex']

    return result


def get_h0_max_edges_from_cache(exports_dir: Path, s: int, t: int, n: int) -> int:
    """
    Get maximum edge count for H_0 on n vertices that is K_{s,t}-free.

    Uses cached ex(n, K_{s,t}) values.
    """
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Load cached ex(n, K_{s,t})
    cache = load_cached_extremal_numbers(exports_dir, s, t)
    if n in cache:
        return cache[n]

    # Fallback: use complete graph (only valid if n < s, which means K_{s,t} can't exist)
    if n < s:
        return n * (n - 1) // 2

    # If not in cache and n >= s, we have a problem - return 0 as safe default
    print(f"Warning: ex({n}, K_{{{s},{t}}}) not in cache, using 0")
    return 0


def get_1t_extremal_edges(exports_star_dir: Path, t: int, n: int) -> int:
    """
    Get ex(n, K_{1,t}) = max edges in n-vertex graph with max degree < t.

    Uses cached values from exports_star directory.
    """
    if n == 0:
        return 0

    cache = load_cached_extremal_numbers(exports_star_dir, 1, t)
    if n in cache:
        return cache[n]

    # Fallback: compute it
    # For max degree t-1, best is (t-1)-regular if possible
    # Edges = n * (t-1) / 2 if n*(t-1) is even
    if n * (t - 1) % 2 == 0:
        return n * (t - 1) // 2
    else:
        # Can't be exactly regular, one less edge
        return (n * (t - 1) - 1) // 2


# =============================================================================
# Core data structures
# =============================================================================

class SimpleGraph:
    """Simple cograph representation for enumeration."""
    __slots__ = ['n', 'edges', 'profile', 'op', 'children', 'max_degree']

    def __init__(self, n: int, edges: int, profile: tuple, op: str,
                 children: tuple = (), max_degree: int = 0):
        self.n = n
        self.edges = edges
        self.profile = profile  # tuple for hashing
        self.op = op  # 'v', 's', 'p'
        self.children = children
        self.max_degree = max_degree

    def __repr__(self):
        return f"G(n={self.n}, e={self.edges}, deg<={self.max_degree})"


def vertex() -> SimpleGraph:
    """Create single vertex."""
    return SimpleGraph(n=1, edges=0, profile=(1, 0), op='v', max_degree=0)


def sum_graphs(g1: SimpleGraph, g2: SimpleGraph) -> SimpleGraph:
    """Disjoint union of two graphs."""
    n = g1.n + g2.n
    edges = g1.edges + g2.edges

    # Profile: pointwise max
    p1 = np.array(g1.profile, dtype=np.int32)
    p2 = np.array(g2.profile, dtype=np.int32)
    profile_arr = sum_profile_fast(p1, p2)
    profile = tuple(profile_arr.tolist())

    max_degree = max(g1.max_degree, g2.max_degree)

    return SimpleGraph(n, edges, profile, 's', (g1, g2), max_degree)


def product_graphs(g1: SimpleGraph, g2: SimpleGraph) -> SimpleGraph:
    """Complete join of two graphs."""
    n = g1.n + g2.n
    edges = g1.edges + g2.edges + g1.n * g2.n

    # Profile: max-convolution
    p1 = np.array(g1.profile, dtype=np.int32)
    p2 = np.array(g2.profile, dtype=np.int32)
    profile_arr = product_profile_fast(p1, p2)
    profile = tuple(profile_arr.tolist())

    # Max degree in product: vertex in G1 sees all of G2 plus its neighbors in G1
    max_degree = max(g1.max_degree + g2.n, g2.max_degree + g1.n)

    return SimpleGraph(n, edges, profile, 'p', (g1, g2), max_degree)


def clique(k: int) -> SimpleGraph:
    """Create K_k clique."""
    if k == 1:
        return vertex()
    g = vertex()
    for _ in range(k - 1):
        g = product_graphs(g, vertex())
    return g


def empty_graph(k: int) -> SimpleGraph:
    """Create E_k (k independent vertices)."""
    if k == 1:
        return vertex()
    g = vertex()
    for _ in range(k - 1):
        g = sum_graphs(g, vertex())
    return g


# =============================================================================
# Enumeration with Pareto filtering
# =============================================================================

def build_pareto_registry(
    max_n: int,
    max_degree_constraint: int | None = None,
    use_lattice_reduction: bool = True
) -> dict[int, list[SimpleGraph]]:
    """
    Build registry of Pareto-optimal graphs up to max_n vertices.

    Args:
        max_n: Maximum vertex count
        max_degree_constraint: If set, prune graphs with max_degree >= this value
        use_lattice_reduction: If True, only keep one graph per (profile, edges) pair

    Returns:
        Dict: n -> list of Pareto-optimal graphs
    """
    # n -> profile -> (max_edges, list of graphs with that edge count)
    registry: dict[int, dict[tuple, tuple[int, list[SimpleGraph]]]] = defaultdict(dict)

    # Initialize with single vertex
    v = vertex()
    registry[1][v.profile] = (0, [v])

    for target_n in range(2, max_n + 1):
        candidates: dict[tuple, tuple[int, list[SimpleGraph]]] = {}

        # Generate from all partitions n1 + n2 = target_n
        for n1 in range(1, target_n // 2 + 1):
            n2 = target_n - n1

            if n1 not in registry or n2 not in registry:
                continue

            for profile1, (edges1, graphs1) in registry[n1].items():
                for profile2, (edges2, graphs2) in registry[n2].items():
                    # Try sum and product
                    for g1 in graphs1:
                        for g2 in graphs2:
                            # Symmetry: skip if n1 == n2 and g1 > g2 (by id)
                            if n1 == n2 and id(g1) > id(g2):
                                continue

                            # Try sum
                            sum_g = sum_graphs(g1, g2)
                            if max_degree_constraint is None or sum_g.max_degree < max_degree_constraint:
                                _add_candidate(candidates, sum_g, use_lattice_reduction)

                            # Try product
                            prod_g = product_graphs(g1, g2)
                            if max_degree_constraint is None or prod_g.max_degree < max_degree_constraint:
                                _add_candidate(candidates, prod_g, use_lattice_reduction)

        registry[target_n] = candidates

    # Convert to list format
    result = {}
    for n in range(1, max_n + 1):
        result[n] = []
        for profile, (edges, graphs) in registry.get(n, {}).items():
            result[n].extend(graphs)

    return result


def _add_candidate(
    candidates: dict[tuple, tuple[int, list[SimpleGraph]]],
    g: SimpleGraph,
    use_lattice_reduction: bool
):
    """Add graph to candidates with Pareto filtering."""
    profile = g.profile
    edges = g.edges

    if profile not in candidates:
        candidates[profile] = (edges, [g])
    else:
        max_edges, graphs = candidates[profile]
        if edges > max_edges:
            candidates[profile] = (edges, [g])
        elif edges == max_edges:
            if not use_lattice_reduction:
                # Check for duplicate structure (simple approximation)
                graphs.append(g)
            # With lattice reduction, we only keep one representative


def build_registry_no_lattice(
    max_n: int,
    max_degree_constraint: int | None = None
) -> dict[int, dict[tuple, list[SimpleGraph]]]:
    """
    Build registry without lattice reduction - keep all profile-extremal graphs.

    Args:
        max_n: Maximum vertex count
        max_degree_constraint: If set, prune graphs with max_degree >= this value

    Returns:
        Dict: n -> profile -> list of graphs with max edges for that profile
    """
    # n -> profile -> (max_edges, list of graphs)
    registry: dict[int, dict[tuple, tuple[int, list[SimpleGraph]]]] = defaultdict(dict)

    # Initialize with single vertex
    v = vertex()
    if max_degree_constraint is None or v.max_degree < max_degree_constraint:
        registry[1][v.profile] = (0, [v])

    for target_n in range(2, max_n + 1):
        candidates: dict[tuple, tuple[int, list[SimpleGraph]]] = {}

        for n1 in range(1, target_n // 2 + 1):
            n2 = target_n - n1

            if n1 not in registry or n2 not in registry:
                continue

            for profile1, (edges1, graphs1) in registry[n1].items():
                for profile2, (edges2, graphs2) in registry[n2].items():
                    for g1 in graphs1:
                        for g2 in graphs2:
                            if n1 == n2 and id(g1) > id(g2):
                                continue

                            # Try sum
                            sum_g = sum_graphs(g1, g2)
                            if max_degree_constraint is None or sum_g.max_degree < max_degree_constraint:
                                _add_candidate_no_lattice(candidates, sum_g)

                            # Try product
                            prod_g = product_graphs(g1, g2)
                            if max_degree_constraint is None or prod_g.max_degree < max_degree_constraint:
                                _add_candidate_no_lattice(candidates, prod_g)

        registry[target_n] = candidates

    # Convert to final format
    result = {}
    for n in range(1, max_n + 1):
        result[n] = {}
        for profile, (edges, graphs) in registry.get(n, {}).items():
            result[n][profile] = graphs

    return result


def _add_candidate_no_lattice(
    candidates: dict[tuple, tuple[int, list[SimpleGraph]]],
    g: SimpleGraph
):
    """Add graph without lattice reduction."""
    profile = g.profile
    edges = g.edges

    if profile not in candidates:
        candidates[profile] = (edges, [g])
    else:
        max_edges, graphs = candidates[profile]
        if edges > max_edges:
            candidates[profile] = (edges, [g])
        elif edges == max_edges:
            graphs.append(g)


# =============================================================================
# Theorem verification
# =============================================================================

def get_max_edge_count_for_n(
    registry: dict[int, list[SimpleGraph]],
    n: int
) -> int:
    """Get maximum edge count across all graphs on n vertices."""
    if n not in registry or not registry[n]:
        return 0
    return max(g.edges for g in registry[n])


def is_connected(g: SimpleGraph) -> bool:
    """Check if a graph is connected (root operation is product)."""
    return g.op == 'v' or g.op == 'p'


def filter_connected_components(
    registry: dict[int, dict[tuple, list[SimpleGraph]]]
) -> list[SimpleGraph]:
    """Extract all connected graphs from registry."""
    result = []
    for n in sorted(registry.keys()):
        for profile, graphs in registry[n].items():
            for g in graphs:
                if is_connected(g):
                    result.append(g)
    return result


def get_compatible_h1_components(
    g0: SimpleGraph,
    h1_components: list[SimpleGraph],
    s: int, t: int
) -> list[SimpleGraph]:
    """Filter H_1 components that are compatible with G_0 (product avoids K_{s,t})."""
    compatible = []
    for comp in h1_components:
        prod = product_graphs(g0, comp)
        if check_profile_avoids(prod.profile, s, t):
            compatible.append(comp)
    return compatible


def filter_to_max_edges_per_n(components: list[SimpleGraph]) -> dict[int, tuple[int, list[SimpleGraph]]]:
    """
    For each vertex count, keep only components with maximum edges.

    Returns:
        Dict: n -> (max_edges, list of components with that edge count)
    """
    by_n: dict[int, tuple[int, list[SimpleGraph]]] = {}

    for comp in components:
        n = comp.n
        e = comp.edges

        if n not in by_n:
            by_n[n] = (e, [comp])
        else:
            max_e, comps = by_n[n]
            if e > max_e:
                by_n[n] = (e, [comp])
            elif e == max_e:
                comps.append(comp)

    return by_n


def enumerate_h1_combinations(
    components_by_n: dict[int, tuple[int, list[SimpleGraph]]],
    max_components: int,
    max_total_vertices: int
) -> Iterator[tuple[int, int, list[tuple[int, int]]]]:
    """
    Enumerate all H_1 combinations (multisets of components).

    Args:
        components_by_n: n -> (max_edges, components) for compatible components
        max_components: Maximum number of components (t-1)
        max_total_vertices: Maximum total vertices

    Yields:
        (total_vertices, total_edges, list of (n, edges) for each component)
    """
    from itertools import combinations_with_replacement

    # Build list of (n, edges) pairs - one per vertex count
    component_options = [(n, edges) for n, (edges, _) in sorted(components_by_n.items())]

    # Empty H_1
    yield (0, 0, [])

    if not component_options:
        return

    # Enumerate combinations with replacement
    for num_comp in range(1, max_components + 1):
        for combo in combinations_with_replacement(range(len(component_options)), num_comp):
            selected = [component_options[i] for i in combo]
            total_v = sum(n for n, _ in selected)
            total_e = sum(e for _, e in selected)

            if total_v <= max_total_vertices:
                yield (total_v, total_e, selected)


def check_profile_avoids(profile: tuple, s: int, t: int) -> bool:
    """Check if profile avoids K_{s,t}."""
    profile_arr = np.array(profile, dtype=np.int32)
    return profile_avoids_kst(profile_arr, s, t)


def compute_structure_edges(
    h0_edges: int,
    g0: SimpleGraph,
    h1_vertices: int,
    h1_edges: int,
    k: int,  # number of K_t cliques to add
    t: int
) -> int:
    """
    Compute total edges for structure H_0 + G_0 x (H_1 + k * K_t).
    """
    s_minus_1 = g0.n

    # Edges in H_0
    total = h0_edges

    # Edges in G_0
    total += g0.edges

    # Edges in H_1 + k * K_t
    inner_vertices = h1_vertices + k * t
    inner_edges = h1_edges + k * comb(t, 2)  # K_t has t*(t-1)/2 edges
    total += inner_edges

    # Cross edges: G_0 x (H_1 + k * K_t)
    total += s_minus_1 * inner_vertices

    return total


def compute_edge_density(g0: SimpleGraph, t: int) -> float:
    """
    Compute edge density per pumped K_t vertex.

    Adding one K_t adds:
    - t*(t-1)/2 internal edges
    - (s-1) * t cross edges

    Per vertex: (t-1)/2 + (s-1)
    """
    s_minus_1 = g0.n
    return (t - 1) / 2 + s_minus_1


def verify_theorem_for_st(
    s: int, t: int,
    g0_registry: dict[int, dict[tuple, list[SimpleGraph]]],
    h1_all_components: list[SimpleGraph],
    exports_dir: Path,
    exports_star_dir: Path,
    verbose: bool = True
) -> dict:
    """
    Verify the theorem for a specific (s, t) pair.

    Args:
        s, t: Biclique parameters
        g0_registry: Pre-built registry for G_0 candidates
        h1_all_components: Pre-built list of connected H_1 component candidates
        exports_dir: Directory with cached K_{s,t} extremal numbers (for H_0)
        exports_star_dir: Directory with cached K_{1,t} extremal numbers (for H_1 verification)
        verbose: Print progress

    Returns:
        Dict with verification results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Checking (s={s}, t={t})")
        print(f"{'='*60}")

    # Bounds from the theorem proof
    max_h0_vertices = comb(s - 1, 2) + 1
    max_g0_vertices = s - 1
    max_h1_components = t - 1
    max_h1_component_size = 2 * (t - 1)
    max_h1_total_vertices = 2 * (t - 1) ** 2

    if verbose:
        print(f"Bounds: H_0 <= {max_h0_vertices} vertices, G_0 = {max_g0_vertices} vertices")
        print(f"        H_1: <= {max_h1_components} components, each <= {max_h1_component_size} vertices")

    # Load cached K_{s,t}-extremal numbers for H_0
    cached_st = load_cached_extremal_numbers(exports_dir, s, t)
    cache_st_max_n = max(cached_st.keys()) if cached_st else 0

    if verbose:
        if cached_st:
            print(f"Loaded cached ex(n, K_{{{s},{t}}}) for H_0, n up to {cache_st_max_n}")
        else:
            print(f"Warning: No cached data for K_{{{s},{t}}}")

    # Load cached (1,t)-extremal numbers for H_1 verification
    cached_1t = load_cached_extremal_numbers(exports_star_dir, 1, t)
    cache_1t_max_n = max(cached_1t.keys()) if cached_1t else 0

    if verbose:
        if cached_1t:
            print(f"Loaded cached ex(n, K_{{1,{t}}}) for H_1 verification, n up to {cache_1t_max_n}")
        else:
            print(f"Warning: No cached data for K_{{1,{t}}} in exports_star")

    # Get G_0 graphs (exactly s-1 vertices)
    g0_graphs = []
    if max_g0_vertices in g0_registry:
        for profile, graphs in g0_registry[max_g0_vertices].items():
            g0_graphs.extend(graphs)

    if verbose:
        print(f"Found {len(g0_graphs)} G_0 candidates on {max_g0_vertices} vertices")

    # Filter H_1 components to those with n <= max_h1_component_size and max_degree < t
    h1_components_for_t = [
        c for c in h1_all_components
        if c.n <= max_h1_component_size and c.max_degree < t
    ]

    if verbose:
        print(f"Found {len(h1_components_for_t)} H_1 component candidates (connected, max_degree < {t})")

    # For each residue class mod t, track best configurations
    # best_by_residue[q] = {'edges': ..., 'h0_v': ..., 'g0': ..., 'h1_config': ..., ...}
    best_by_residue: dict[int, dict] = {q: {'edges': -1} for q in range(t)}

    # Also track all optimal-form configurations (H_0 empty, all H_1 components (1,t)-extremal)
    optimal_by_residue: dict[int, dict] = {q: {'edges': -1} for q in range(t)}

    # Large N for comparison
    test_N = 1000 + t

    # Iterate over G_0 candidates
    for g0 in g0_graphs:
        # Check G_0 × K_t avoids K_{s,t}
        kt = clique(t)
        if not check_profile_avoids(product_graphs(g0, kt).profile, s, t):
            continue

        # Find compatible H_1 components for this G_0
        compatible_h1 = get_compatible_h1_components(g0, h1_components_for_t, s, t)

        if verbose:
            print(f"  G_0 (edges={g0.edges}): {len(compatible_h1)} compatible H_1 components")

        # Filter to max edges per vertex count
        h1_by_n = filter_to_max_edges_per_n(compatible_h1)

        # Build table: h1_total_v -> (max_edges_of_product, list of component configs)
        # where component config is list of (n, edges) tuples
        product_table: dict[int, tuple[int, list[list[tuple[int, int]]]]] = {}

        for h1_v, h1_e, h1_config in enumerate_h1_combinations(
            h1_by_n, max_h1_components, max_h1_total_vertices
        ):
            # Compute edges in G_0 × H_1
            # G_0 has g0.edges, H_1 has h1_e, cross edges = |G_0| * |H_1|
            product_edges = g0.edges + h1_e + g0.n * h1_v

            if h1_v not in product_table:
                product_table[h1_v] = (product_edges, [h1_config])
            else:
                max_e, configs = product_table[h1_v]
                if product_edges > max_e:
                    product_table[h1_v] = (product_edges, [h1_config])
                elif product_edges == max_e:
                    configs.append(h1_config)

        # Now iterate over H_0 sizes and compute residues
        for h0_v in range(0, max_h0_vertices + 1):
            h0_edges = get_h0_max_edges_from_cache(exports_dir, s, t, h0_v)

            for h1_v, (product_edges, h1_configs) in product_table.items():
                # Core vertex count
                core_v = h0_v + g0.n + h1_v
                residue = core_v % t

                # For test_N, compute pump count
                target_v = test_N - (test_N % t) + residue
                if target_v < test_N:
                    target_v += t
                k = (target_v - core_v) // t
                if k < 0:
                    continue

                # Total edges = H_0 edges + product edges + pump edges
                # Pump: k copies of K_t, each adds binom(t,2) internal + g0.n * t cross edges
                pump_edges = k * (comb(t, 2) + g0.n * t)
                total_edges = h0_edges + product_edges + pump_edges

                # Check each H_1 configuration
                for h1_config in h1_configs:
                    # Check if all components are (1,t)-extremal
                    all_1t_extremal = True
                    for comp_n, comp_e in h1_config:
                        expected_e = get_1t_extremal_edges(exports_star_dir, t, comp_n)
                        if comp_e != expected_e:
                            all_1t_extremal = False
                            break

                    is_optimal_form = (h0_v == 0) and all_1t_extremal

                    config_info = {
                        'edges': total_edges,
                        'h0_vertices': h0_v,
                        'h0_edges': h0_edges,
                        'g0_edges': g0.edges,
                        'g0_profile': g0.profile,
                        'h1_vertices': h1_v,
                        'h1_edges': sum(e for _, e in h1_config),
                        'h1_components': h1_config,
                        'h1_all_1t_extremal': all_1t_extremal,
                        'is_optimal_form': is_optimal_form,
                        'k': k,
                        'total_vertices': target_v
                    }

                    # Update best overall
                    if total_edges > best_by_residue[residue].get('edges', -1):
                        best_by_residue[residue] = config_info

                    # Update best optimal-form
                    if is_optimal_form and total_edges > optimal_by_residue[residue].get('edges', -1):
                        optimal_by_residue[residue] = config_info

    # Check results
    all_optimal = True
    results = {
        's': s,
        't': t,
        'by_residue': {},
        'all_optimal_form': True
    }

    if verbose:
        print(f"\nResults by residue class (mod {t}):")

    for residue in range(t):
        best = best_by_residue[residue]
        optimal = optimal_by_residue[residue]

        if best.get('edges', -1) < 0:
            if verbose:
                print(f"  {residue}: No valid configuration found")
            continue

        is_optimal = best.get('is_optimal_form', False)

        # Check if there's an optimal-form config with same edge count
        has_optimal_alternative = (
            optimal.get('edges', -1) == best['edges']
        )

        if not is_optimal:
            all_optimal = False
            results['all_optimal_form'] = False

        # Detailed status
        h0_ok = best.get('h0_vertices', -1) == 0
        h1_ok = best.get('h1_all_1t_extremal', False)

        if h0_ok and h1_ok:
            status = "OK"
        else:
            issues = []
            if not h0_ok:
                issues.append(f"H_0={best['h0_vertices']}v")
            if not h1_ok:
                issues.append(f"H_1 components not all (1,{t})-extremal")
            status = "ISSUE: " + ", ".join(issues)
            if has_optimal_alternative:
                status += " [but optimal alternative exists]"

        results['by_residue'][residue] = {
            'best': best,
            'optimal_alternative': optimal if has_optimal_alternative and not is_optimal else None,
            'is_optimal': is_optimal,
            'has_optimal_alternative': has_optimal_alternative
        }

        if verbose:
            h1_comp_str = str(best.get('h1_components', []))
            print(f"  {residue}: H_0={best.get('h0_vertices')}v/{best.get('h0_edges')}e, "
                  f"G_0={s-1}v/{best.get('g0_edges')}e, "
                  f"H_1={best.get('h1_vertices')}v/{best.get('h1_edges')}e "
                  f"comps={h1_comp_str} -> {status}")

    return results


def main(max_bound: int = 6, exports_dir: str | None = None, exports_star_dir: str | None = None):
    """Run verification for all (s,t) pairs in the theorem range."""
    from pathlib import Path as P

    if exports_dir:
        exports_path = P(exports_dir)
    else:
        exports_path = P(__file__).parent.parent / "exports_lattice_12_2"

    if exports_star_dir:
        exports_star_path = P(exports_star_dir)
    else:
        exports_star_path = P(__file__).parent.parent / "exports_star"

    print("="*70)
    print("Verifying Theorem small_cases_classification")
    print(f"For s <= t <= min({max_bound}, 2s-1), checking extremal cograph structure")
    print(f"Using K_{{s,t}} cache from: {exports_path}")
    print(f"Using K_{{1,t}} cache from: {exports_star_path}")
    print("="*70)

    # Pre-build registries for max bounds
    # G_0 needs up to max_bound - 1 vertices
    max_g0_vertices = max_bound - 1
    # H_1 components need up to 2*(max_bound - 1) vertices with max_degree < max_bound
    max_h1_component_size = 2 * (max_bound - 1)

    print(f"\nPre-building G_0 registry (up to {max_g0_vertices} vertices, no lattice reduction)...")
    g0_registry = build_registry_no_lattice(max_g0_vertices)
    g0_count = sum(
        len(graphs)
        for n_dict in g0_registry.values()
        for graphs in n_dict.values()
    )
    print(f"  Built {g0_count} G_0 candidates")

    print(f"Pre-building H_1 component registry (up to {max_h1_component_size} vertices, no lattice reduction)...")
    # Build without max_degree constraint first, then filter per (s,t)
    h1_full_registry = build_registry_no_lattice(max_h1_component_size)

    # Extract connected components
    h1_all_components = filter_connected_components(h1_full_registry)
    print(f"  Built {len(h1_all_components)} connected H_1 component candidates")

    all_results = []
    failures = []
    cache = {}  # Cache for extremal triples

    for s in range(2, max_bound + 1):
        t_max = min(max_bound, 2*s - 1)
        for t in range(s, t_max + 1):
            result = verify_theorem_for_st(
                s, t,
                g0_registry=g0_registry,
                h1_all_components=h1_all_components,
                exports_dir=exports_path,
                exports_star_dir=exports_star_path,
                verbose=True
            )
            all_results.append(result)

            # Cache the extremal triples for each residue class
            cache_key = f"s{s}_t{t}"
            cache[cache_key] = {
                's': s,
                't': t,
                'residue_classes': {}
            }
            for q in range(t):
                residue_info = result['by_residue'].get(q, {})
                best = residue_info.get('best', {})
                if best.get('edges', -1) >= 0:
                    cache[cache_key]['residue_classes'][q] = {
                        'h0_vertices': best.get('h0_vertices'),
                        'h0_edges': best.get('h0_edges'),
                        'g0_vertices': s - 1,
                        'g0_edges': best.get('g0_edges'),
                        'g0_profile': list(best.get('g0_profile', [])),
                        'h1_vertices': best.get('h1_vertices'),
                        'h1_edges': best.get('h1_edges'),
                        'h1_components': best.get('h1_components', []),
                        'h1_all_1t_extremal': best.get('h1_all_1t_extremal', False),
                        'is_optimal_form': best.get('is_optimal_form', False),
                        'has_optimal_alternative': residue_info.get('has_optimal_alternative', False)
                    }

            if not result['all_optimal_form']:
                failures.append((s, t))

    # Save cache to file
    cache_path = P(__file__).parent / "extremal_triples_cache.json"
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved to: {cache_path}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total (s,t) pairs checked: {len(all_results)}")

    if not failures:
        print("RESULT: Theorem verified for all (s,t) pairs!")
        print("All extremal structures have H_0 empty AND all H_1 components are (1,t)-extremal.")
    else:
        print(f"RESULT: Found {len(failures)} pairs with issues:")
        for s, t in failures:
            # Check if all residues have optimal alternatives
            result = next(r for r in all_results if r['s'] == s and r['t'] == t)
            all_have_alt = all(
                info.get('has_optimal_alternative', False)
                for info in result['by_residue'].values()
            )
            suffix = " (but all have optimal alternatives)" if all_have_alt else ""
            print(f"  (s={s}, t={t}){suffix}")

    return len(failures) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify small cases classification theorem")
    parser.add_argument("--max-bound", type=int, default=6,
                        help="Maximum value for s and t (default: 6)")
    parser.add_argument("--exports-dir", type=str, default=None,
                        help="Directory with cached K_{s,t} extremal numbers (default: exports_lattice_12_2)")
    parser.add_argument("--exports-star-dir", type=str, default=None,
                        help="Directory with cached K_{1,t} extremal numbers (default: exports_star)")
    args = parser.parse_args()

    success = main(max_bound=args.max_bound, exports_dir=args.exports_dir,
                   exports_star_dir=args.exports_star_dir)
    sys.exit(0 if success else 1)
