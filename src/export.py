"""Export functionality for extremal cograph data."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any

from .cotree import Cotree, to_graph6
from .registry import Registry


def analyze_extremal(cotree: Cotree) -> dict[str, Any]:
    """
    Analyze structure of an extremal cograph.

    Args:
        cotree: The cotree to analyze

    Returns:
        Dictionary with analysis:
        - n: vertex count
        - edges: edge count
        - last_op: "vertex" | "sum" | "product"
        - component_sizes: tuple of child vertex counts (for non-vertex)
        - component_edges: tuple of child edge counts
        - depth: cotree depth
        - structure_str: human-readable structure like "P(S(1,2),3)"
    """
    result = {
        "n": cotree.n,
        "edges": cotree.edges,
        "last_op": cotree.op,
        "structure_str": cotree.structure_str(),
        "depth": _cotree_depth(cotree),
    }

    if cotree.op == "vertex":
        result["component_sizes"] = ()
        result["component_edges"] = ()
    else:
        result["component_sizes"] = tuple(c.n for c in cotree.children)
        result["component_edges"] = tuple(c.edges for c in cotree.children)

    return result


def _cotree_depth(cotree: Cotree) -> int:
    """Compute depth of cotree (number of operations from root to deepest leaf)."""
    if cotree.op == "vertex":
        return 0
    return 1 + max(_cotree_depth(c) for c in cotree.children)


def export_extremal_table(
    registry: Registry,
    s_max: int,
    t_max: int,
    path: Path | str,
    n_max: int | None = None
) -> None:
    """
    Export table of extremal numbers ex(n, K_{s,t}) to CSV.

    Args:
        registry: Registry containing cographs
        s_max: Maximum s value (inclusive)
        t_max: Maximum t value (inclusive)
        path: Output CSV path
        n_max: Maximum n (default: registry.max_n())
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if n_max is None:
        n_max = registry.max_n()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "s", "t", "ex(n;Kst)", "count", "last_ops", "structures"])

        for n in range(1, n_max + 1):
            for s in range(1, s_max + 1):
                for t in range(s, t_max + 1):  # s <= t by symmetry
                    graphs = registry.get_avoiding(n, s, t)
                    if not graphs:
                        continue

                    ex = graphs[0][0]
                    count = len(graphs)
                    last_ops = ";".join(sorted(set(g.op for _, g in graphs)))
                    structures = ";".join(g.structure_str() for _, g in graphs)

                    writer.writerow([n, s, t, ex, count, last_ops, structures])


def export_extremal_for_biclique(
    registry: Registry,
    s: int,
    t: int,
    path: Path | str,
    n_max: int | None = None
) -> None:
    """
    Export detailed data for extremal K_{s,t}-free cographs to JSON.

    Args:
        registry: Registry containing cographs
        s: Left side of forbidden biclique
        t: Right side of forbidden biclique
        path: Output JSON path
        n_max: Maximum n (default: registry.max_n())
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if n_max is None:
        n_max = registry.max_n()

    data = {
        "s": s,
        "t": t,
        "extremal_by_n": {}
    }

    for n in range(1, n_max + 1):
        graphs = registry.get_avoiding(n, s, t)
        if not graphs:
            continue

        entry = {
            "ex": graphs[0][0],
            "count": len(graphs),
            "graphs": [g.structure_str() for _, g in graphs],
            "graph6": [to_graph6(g) for _, g in graphs],
            "analyses": [analyze_extremal(g) for _, g in graphs]
        }
        data["extremal_by_n"][str(n)] = entry

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_graphs_graph6(
    registry: Registry,
    n: int,
    s: int,
    t: int,
    path: Path | str
) -> int:
    """
    Export extremal K_{s,t}-free graphs on n vertices in graph6 format.

    Args:
        registry: Registry containing cographs
        n: Number of vertices
        s: Left side of forbidden biclique
        t: Right side of forbidden biclique
        path: Output file path (one graph per line)

    Returns:
        Number of graphs exported
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    graphs = registry.get_avoiding(n, s, t)

    with open(path, "w") as f:
        for _, cotree in graphs:
            f.write(to_graph6(cotree) + "\n")

    return len(graphs)


def print_extremal_table(
    registry: Registry,
    s_max: int,
    t_max: int,
    n_max: int | None = None
) -> None:
    """
    Print extremal numbers table to stdout.

    Args:
        registry: Registry containing cographs
        s_max: Maximum s value
        t_max: Maximum t value
        n_max: Maximum n (default: registry.max_n())
    """
    if n_max is None:
        n_max = registry.max_n()

    # Header
    print(f"\nExtremal numbers ex(n, K_{{s,t}}) for cographs")
    print("=" * 60)

    for s in range(1, s_max + 1):
        for t in range(s, t_max + 1):
            print(f"\nK_{{{s},{t}}}-free cographs:")
            print(f"{'n':>4} | {'ex':>6} | {'count':>5} | structures")
            print("-" * 50)

            for n in range(1, n_max + 1):
                graphs = registry.get_avoiding(n, s, t)
                if graphs:
                    ex = graphs[0][0]
                    count = len(graphs)
                    structs = ", ".join(g.structure_str() for _, g in graphs[:3])
                    if len(graphs) > 3:
                        structs += f", ... (+{len(graphs) - 3})"
                    print(f"{n:>4} | {ex:>6} | {count:>5} | {structs}")


def summarize_extremal(
    registry: Registry,
    s: int,
    t: int,
    n_max: int | None = None
) -> list[dict]:
    """
    Get summary of extremal K_{s,t}-free graphs.

    Args:
        registry: Registry containing cographs
        s: Left side of forbidden biclique
        t: Right side of forbidden biclique
        n_max: Maximum n

    Returns:
        List of dicts with n, ex, count, graphs info
    """
    if n_max is None:
        n_max = registry.max_n()

    result = []
    for n in range(1, n_max + 1):
        graphs = registry.get_avoiding(n, s, t)
        if graphs:
            result.append({
                "n": n,
                "ex": graphs[0][0],
                "count": len(graphs),
                "structures": [g.structure_str() for _, g in graphs],
                "last_ops": list(set(g.op for _, g in graphs))
            })

    return result


def compare_with_turan(
    registry: Registry,
    r: int,
    n_max: int | None = None
) -> list[dict]:
    """
    Compare K_{r,r}-free cograph extremal numbers with Turan numbers.

    The Turan number ex(n, K_r) (complete graph) is known, and for bipartite
    graphs we have Zarankiewicz numbers z(n; r, r).

    Args:
        registry: Registry containing cographs
        r: Size of forbidden K_{r,r}
        n_max: Maximum n

    Returns:
        List of comparison dicts
    """
    if n_max is None:
        n_max = registry.max_n()

    result = []
    for n in range(1, n_max + 1):
        graphs = registry.get_avoiding(n, r, r)
        if graphs:
            ex_cograph = graphs[0][0]
            # Zarankiewicz bound: z(n; r, r) <= (r-1)^{1/r} * n^{2-1/r} / 2 + (r-1)*n/2
            # This is just an upper bound
            result.append({
                "n": n,
                "ex_cograph": ex_cograph,
                "structures": [g.structure_str() for _, g in graphs[:3]]
            })

    return result


def export_all(
    registry: Registry,
    output_dir: Path | str,
    s_max: int = 4,
    t_max: int = 4
) -> dict[str, Path]:
    """
    Export all data in various formats.

    Args:
        registry: Registry to export
        output_dir: Output directory
        s_max: Maximum s for tables
        t_max: Maximum t for tables

    Returns:
        Dictionary mapping export type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # CSV table of all extremal numbers
    csv_path = output_dir / "extremal_table.csv"
    export_extremal_table(registry, s_max, t_max, csv_path)
    paths["table_csv"] = csv_path

    # JSON files for each (s,t) pair
    for s in range(2, s_max + 1):
        for t in range(s, t_max + 1):
            json_path = output_dir / f"extremal_K{s}{t}.json"
            export_extremal_for_biclique(registry, s, t, json_path)
            paths[f"K{s}{t}_json"] = json_path

    # Summary statistics
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(registry.statistics(), f, indent=2)
    paths["statistics"] = stats_path

    return paths
