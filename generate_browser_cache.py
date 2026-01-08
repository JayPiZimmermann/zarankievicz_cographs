#!/usr/bin/env python3
"""
Generate browser D3 cache with per-parameter JSON files.
Depth is calculated as the longest path from root to any leaf (number of edges).

Memory-efficient: processes one parameter set at a time and writes immediately.
"""

import json
import sys
import gc
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.builder import parse_structure
from src.cotree import Cotree


def calculate_cotree_depth(node: Cotree, parent_op: str = None) -> int:
    """
    Calculate depth as the max number of operation switches from root to leaf.

    Depth increases when traversing from a sum to product or vice versa.
    Vertices don't count. Same-type consecutive operations don't add depth.

    Examples:
    - "1" -> depth 0 (just a vertex)
    - "P(1,1)" -> depth 0 (no switches, just P to vertices)
    - "P(S(1,1),1)" -> depth 1 (one switch: P->S)
    - "P(S(P(1,1),1),1)" -> depth 2 (two switches: P->S->P)
    - "S(P(1,1),P(1,1))" -> depth 1 (one switch: S->P)
    """
    if node.op == "vertex":
        return 0

    if not node.children:
        return 0

    # Count switch if operation changes from parent (and parent is not None/vertex)
    switch = 0
    if parent_op is not None and parent_op != "vertex" and parent_op != node.op:
        switch = 1

    max_child_depth = max(
        calculate_cotree_depth(child, node.op) for child in node.children
    )
    return switch + max_child_depth


def get_vertex_ids(node: Cotree, vertex_map: dict) -> list:
    """Get all vertex IDs in a subtree."""
    if node.op == "vertex":
        return [vertex_map[id(node)]]

    vids = []
    for child in node.children:
        vids.extend(get_vertex_ids(child, vertex_map))
    return vids


def assign_vertex_ids(node: Cotree, vertex_map: dict, counter: list):
    """Assign IDs to vertices in tree and build mapping."""
    if node.op == "vertex":
        vid = counter[0]
        counter[0] += 1
        vertex_map[id(node)] = vid
        return

    for child in node.children:
        assign_vertex_ids(child, vertex_map, counter)


def build_edge_list(node: Cotree, vertex_map: dict) -> list:
    """Build edge list from cotree structure."""
    edges = []

    if node.op == "vertex":
        return edges

    child_vertices = []
    for child in node.children:
        vids = get_vertex_ids(child, vertex_map)
        child_vertices.append(vids)
        edges.extend(build_edge_list(child, vertex_map))

    if node.op == "product":
        for i in range(len(child_vertices)):
            for j in range(i + 1, len(child_vertices)):
                for v1 in child_vertices[i]:
                    for v2 in child_vertices[j]:
                        edges.append({'source': v1, 'target': v2})

    return edges


def count_vertices(node: Cotree) -> int:
    """Count vertices in a subtree."""
    return node.n  # Use the built-in property


def count_edges_from_structure(node: Cotree) -> int:
    """
    Count edges in the cograph without building the full edge list.
    More memory efficient for large graphs.
    """
    if node.op == "vertex":
        return 0

    total_edges = 0

    # Count edges from children recursively
    child_sizes = []
    for child in node.children:
        total_edges += count_edges_from_structure(child)
        child_sizes.append(child.n)

    # If product node, add edges between all pairs of children's vertices
    if node.op == "product":
        for i in range(len(child_sizes)):
            for j in range(i + 1, len(child_sizes)):
                total_edges += child_sizes[i] * child_sizes[j]

    return total_edges


def get_root_child_info(node: Cotree, vertex_map: dict) -> list:
    """Get info for each direct child of root (size and vertex IDs)."""
    if node.op == "vertex":
        return [{'size': 1, 'vertices': [vertex_map[id(node)]]}]
    return [
        {'size': child.n, 'vertices': get_vertex_ids(child, vertex_map)}
        for child in node.children
    ]


def process_structure(struct_str, s, t, n, struct_idx):
    """
    Process a structure string and return graph data.

    Cograph rendering (nodes/links) only available when s <= 4 AND t <= 20.
    Otherwise only cotree data is stored.
    """
    root = parse_structure(struct_str)

    # Calculate correct depth (longest path from root to leaf)
    depth = calculate_cotree_depth(root)

    # Get root operation
    root_op = root.op

    # Determine if we should include full cograph data
    # Only render cograph when s <= 4 AND t <= 20
    include_cograph = (s <= 4) and (t <= 20)

    if include_cograph:
        # Assign vertex IDs
        vertex_map = {}
        counter = [0]
        assign_vertex_ids(root, vertex_map, counter)

        # Build nodes (simple for now, positions set by D3)
        n_vertices = counter[0]
        nodes = [{'id': i} for i in range(n_vertices)]

        # Build edges
        edges = build_edge_list(root, vertex_map)

        # Get root child info for component detection
        root_children = get_root_child_info(root, vertex_map)

        return {
            'n': n,
            'struct_index': struct_idx,
            'edges_count': len(edges),
            'structure': struct_str,
            'depth': depth,
            'root_op': root_op,
            'root_children': root_children,  # List of {size, vertices}
            'nodes': nodes,
            'links': edges,
            'cotree_only': False
        }
    else:
        # Only cotree data - no cograph rendering
        # Calculate edge count from structure without building full edge list
        edges_count = count_edges_from_structure(root)

        return {
            'n': n,
            'struct_index': struct_idx,
            'edges_count': edges_count,
            'structure': struct_str,
            'depth': depth,
            'root_op': root_op,
            'cotree_only': True
        }


def collect_param_files(script_dir: Path) -> dict:
    """
    Scan export folders and collect file paths for each (s, t) parameter set.
    Returns a dict mapping (s, t) -> list of file paths containing that data.
    """
    param_files = defaultdict(list)

    for path in script_dir.iterdir():
        if path.is_dir() and path.name.startswith("exports"):
            for filepath in path.glob("extremal_K*.json"):
                # Quick parse to get s, t without loading full file
                try:
                    with open(filepath) as f:
                        # Read just enough to get s and t
                        content = f.read(500)
                        import re
                        s_match = re.search(r'"s"\s*:\s*(\d+)', content)
                        t_match = re.search(r'"t"\s*:\s*(\d+)', content)
                        if s_match and t_match:
                            s = int(s_match.group(1))
                            t = int(t_match.group(1))
                            param_files[(s, t)].append(filepath)
                except Exception:
                    pass

    return param_files


def normalize_structure(struct_str: str) -> str:
    """
    Normalize a structure string for deduplication.
    Removes whitespace and sorts children within S() and P() (commutative operations).
    """
    # Remove all whitespace
    s = struct_str.replace(" ", "").replace("\t", "").replace("\n", "")

    # Parse and sort children recursively
    def parse_and_sort(s, pos=0):
        if pos >= len(s):
            return "", pos

        if s[pos] == '1':
            return "1", pos + 1

        # Check for S( or P(
        if s[pos:pos+2] in ('S(', 'P('):
            op = s[pos]
            pos += 2  # skip "S(" or "P("
            children = []

            while pos < len(s) and s[pos] != ')':
                if s[pos] == ',':
                    pos += 1
                    continue
                child, pos = parse_and_sort(s, pos)
                if child:
                    children.append(child)

            pos += 1  # skip ")"

            # Sort children for canonical form (S and P are commutative)
            children.sort()
            return f"{op}({','.join(children)})", pos

        return "", pos + 1

    result, _ = parse_and_sort(s)
    return result if result else s


def process_param_set(s: int, t: int, filepaths: list, cache_dir: Path) -> tuple:
    """
    Process all files for a single (s, t) parameter set and write cache.
    Deduplicates graphs based on normalized cotree structure strings.
    Returns (n_values_count, total_graphs_count).
    """
    # Use dict to deduplicate: {n: {normalized_structure: graph_obj}}
    n_data = defaultdict(dict)

    for filepath in filepaths:
        try:
            with open(filepath) as f:
                data = json.load(f)

            for n_str, n_info in data.get("extremal_by_n", {}).items():
                n = int(n_str)

                # Skip large n to avoid memory issues with edge generation
                if n > 60:
                    continue

                # Get structures
                if "structures" in n_info:
                    structures_list = n_info["structures"]
                elif "graphs" in n_info:
                    graphs = n_info.get("graphs", [])
                    analyses = n_info.get("analyses", [])
                    structures_list = []
                    for graph_str, analysis in zip(graphs, analyses):
                        structures_list.append({
                            "structure": analysis.get("structure_str", graph_str),
                            "edges": analysis.get("edges", n_info.get("ex", 0))
                        })
                else:
                    continue

                for struct_idx, struct_data in enumerate(structures_list):
                    struct_str = struct_data.get("structure", struct_data.get("structure_str", ""))
                    if not struct_str:
                        continue

                    # Normalize structure for deduplication
                    normalized = normalize_structure(struct_str)

                    # Skip if we already have this structure for this n (deduplication)
                    if normalized in n_data[n]:
                        continue

                    try:
                        graph_obj = process_structure(struct_str, s, t, n, struct_idx)
                        graph_obj['s'] = s
                        graph_obj['t'] = t
                        # Use normalized structure as key for deduplication
                        n_data[n][normalized] = graph_obj
                    except Exception:
                        pass

        except Exception:
            pass

    # Convert dict to list and reassign struct_index
    output_data = {}
    for n in sorted(n_data.keys()):
        graphs = list(n_data[n].values())
        # Reassign struct_index after deduplication
        for idx, g in enumerate(graphs):
            g['struct_index'] = idx
        output_data[str(n)] = graphs

    # Write cache file for this parameter set
    if output_data:
        cache_file = cache_dir / f"K{s}_{t}.json"
        with open(cache_file, 'w') as f:
            json.dump(output_data, f)

    total_graphs = sum(len(g) for g in output_data.values())
    return len(output_data), total_graphs


def main():
    script_dir = Path(__file__).parent
    cache_dir = script_dir / "browser_d3_cache"

    # Create cache directory
    cache_dir.mkdir(exist_ok=True)

    print("Scanning export folders...")
    param_files = collect_param_files(script_dir)

    if not param_files:
        print("No export files found!")
        return

    total_params = len(param_files)
    print(f"Found {total_params} parameter sets to process\n")

    available_params = set()
    processed = 0

    # Process each parameter set one at a time (memory efficient)
    for (s, t), filepaths in sorted(param_files.items()):
        processed += 1
        print(f"[{processed}/{total_params}] Processing K{s},{t}...", end=" ", flush=True)

        n_count, graph_count = process_param_set(s, t, filepaths, cache_dir)

        if n_count > 0:
            available_params.add((s, t))
            print(f"{n_count} n values, {graph_count} graphs")
        else:
            print("no data")

        # Force garbage collection after each parameter set
        gc.collect()

    # Write index file with available parameters
    index = {
        'params': sorted(list(available_params)),
        'param_count': len(available_params)
    }

    with open(cache_dir / "index.json", 'w') as f:
        json.dump(index, f)

    print(f"\nCache written to {cache_dir}/")
    print(f"Total: {len(available_params)} parameter sets")


if __name__ == "__main__":
    main()
