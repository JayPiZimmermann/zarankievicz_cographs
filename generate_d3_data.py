#!/usr/bin/env python3
"""
Generate D3.js-ready data for interactive cograph visualization.
Computes structure-aware layouts respecting cotree operations.
"""

import json
import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from analyze_conjectures import parse_structure, StructureNode


class VertexLayouter:
    """
    Hierarchical circle layout for cotrees.
    Each sum/product distributes children equally on a circle.
    Circles nest within circles to reflect the tree structure.
    """

    def __init__(self):
        self.vertex_id = 0
        self.vertex_info = {}
        self.component_id = 0

    def _count_vertices(self, node):
        """Count vertices in a subtree."""
        if node.op == "vertex":
            return 1
        return sum(self._count_vertices(c) for c in node.children)

    def layout_cotree(self, node, cx=500, cy=500, radius=400, depth=0, component_path=None):
        """
        Recursively layout cotree using hierarchical circles.

        Args:
            node: The cotree node to layout
            cx, cy: Center position for this subtree
            radius: Available radius for this subtree
            depth: Current depth in the tree
            component_path: Path of components from root

        Returns: (list of vertex info dicts, list of vertex ids)
        """
        if component_path is None:
            component_path = []

        if node.op == "vertex":
            # Leaf vertex: place at center
            vid = self.vertex_id
            self.vertex_id += 1

            info = {
                'id': vid,
                'x': cx,
                'y': cy,
                'depth': depth,
                'type': 'vertex',
                'components': list(component_path)
            }
            self.vertex_info[vid] = info
            return [info], [vid]

        # Internal node (sum or product): distribute children on a circle
        n_children = len(node.children)
        if n_children == 0:
            return [], []

        all_infos = []
        all_vids = []

        # Create component ID for this node
        comp_id = self.component_id
        self.component_id += 1

        # Calculate child sizes for proportional spacing
        child_sizes = [self._count_vertices(c) for c in node.children]
        total_size = sum(child_sizes)

        # For single child, place at center with reduced radius
        if n_children == 1:
            child_path = component_path + [{'id': comp_id, 'op': node.op, 'child': 0, 'level': depth}]
            child_radius = radius * 0.8
            infos, vids = self.layout_cotree(
                node.children[0], cx, cy, child_radius, depth + 1, child_path
            )
            all_infos.extend(infos)
            all_vids.extend(vids)
            return all_infos, all_vids

        # Calculate radius for child placement circle
        # Children are placed on this circle, with their own sub-circles inside
        if n_children == 2:
            # For 2 children, place them opposite each other
            placement_radius = radius * 0.5
            child_radius = radius * 0.4
        else:
            # For more children, use geometry to fit circles
            # Each child gets an arc of 2*pi/n, child circles shouldn't overlap
            placement_radius = radius * 0.55
            # Child radius based on chord length between adjacent children
            chord = 2 * placement_radius * math.sin(math.pi / n_children)
            child_radius = min(chord * 0.45, radius * 0.4)

        # Distribute children on circle, proportional to their size
        current_angle = -math.pi / 2  # Start from top

        for i, (child, size) in enumerate(zip(node.children, child_sizes)):
            # Angular span proportional to vertex count
            angle_span = 2 * math.pi * size / total_size
            child_angle = current_angle + angle_span / 2

            # Position for this child's center
            child_cx = cx + placement_radius * math.cos(child_angle)
            child_cy = cy + placement_radius * math.sin(child_angle)

            # Scale child radius by relative size
            size_factor = math.sqrt(size / (total_size / n_children)) if total_size > 0 else 1
            scaled_child_radius = child_radius * min(size_factor, 1.5)

            child_path = component_path + [{'id': comp_id, 'op': node.op, 'child': i, 'level': depth}]

            infos, vids = self.layout_cotree(
                child, child_cx, child_cy, scaled_child_radius, depth + 1, child_path
            )

            # Tag vertices with their component info
            for vid in vids:
                self.vertex_info[vid][f'{node.op}_component'] = i

            all_infos.extend(infos)
            all_vids.extend(vids)

            current_angle += angle_span

        return all_infos, all_vids


def build_edge_list(node, vertex_map):
    """
    Build edge list from cotree structure.
    vertex_map: maps nodes to their vertex IDs
    """
    edges = []

    if node.op == "vertex":
        return edges

    # Get vertices for each child
    child_vertices = []
    for child in node.children:
        vids = get_vertex_ids(child, vertex_map)
        child_vertices.append(vids)
        # Recursively get edges from children
        edges.extend(build_edge_list(child, vertex_map))

    if node.op == "sum":
        # Sum: edges within each child, no edges between children
        pass

    elif node.op == "product":
        # Product: complete bipartite between all pairs of children
        for i in range(len(child_vertices)):
            for j in range(i + 1, len(child_vertices)):
                for v1 in child_vertices[i]:
                    for v2 in child_vertices[j]:
                        edges.append({'source': v1, 'target': v2})

    return edges


def get_vertex_ids(node, vertex_map):
    """Get all vertex IDs in a subtree."""
    if node.op == "vertex":
        return [vertex_map[id(node)]]

    vids = []
    for child in node.children:
        vids.extend(get_vertex_ids(child, vertex_map))
    return vids


def assign_vertex_ids(node, vertex_map, counter):
    """Assign IDs to vertices in tree and build mapping."""
    if node.op == "vertex":
        vid = counter[0]
        counter[0] += 1
        vertex_map[id(node)] = vid
        return

    for child in node.children:
        assign_vertex_ids(child, vertex_map, counter)


def main():
    parser = argparse.ArgumentParser(description="Generate D3.js visualization data from extremal graph exports")
    parser.add_argument("export_dir", nargs="?", default="exports",
                        help="Path to the exports directory (default: exports)")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    print(f"Generating D3.js visualization data from: {export_dir}")

    if not export_dir.exists():
        print(f"Error: exports directory not found: {export_dir}")
        sys.exit(1)

    # Load all files and sort by s,t values from JSON content
    all_files = list(export_dir.glob("extremal_K*.json"))
    file_data = []
    for filepath in all_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            file_data.append((data["s"], data["t"], filepath, data))
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)

    # Sort by (s, t) numerically
    file_data.sort(key=lambda x: (x[0], x[1]))
    print(f"Found {len(file_data)} K_{{s,t}} files")

    # Process graphs
    all_graphs = []
    graph_count = 0

    for s, t, filepath, data in file_data:
        # Process each n value
        for n_str in sorted(data.get("extremal_by_n", {}).keys(), key=int):
            n = int(n_str)
            n_data = data["extremal_by_n"][n_str]

            # Take only first few structures per (s,t,n)
            max_per_n = 3
            for struct_idx, struct_data in enumerate(n_data["structures"][:max_per_n]):
                struct_str = struct_data["structure"]
                edges_count = struct_data["edges"]

                try:
                    # Parse structure
                    root = parse_structure(struct_str)

                    # Assign vertex IDs
                    vertex_map = {}
                    counter = [0]
                    assign_vertex_ids(root, vertex_map, counter)

                    # Compute layout
                    layouter = VertexLayouter()
                    vertex_infos, vertex_ids = layouter.layout_cotree(root)

                    # Build edges
                    edges = build_edge_list(root, vertex_map)

                    # Add sum/product component properties for force simulation
                    for vinfo in vertex_infos:
                        # Find first sum and product components in path
                        sum_comp = None
                        product_comp = None
                        for comp in vinfo.get('components', []):
                            if comp['op'] == 'sum' and sum_comp is None:
                                sum_comp = f"{comp['id']}_{comp['child']}"
                            elif comp['op'] == 'product' and product_comp is None:
                                product_comp = f"{comp['id']}_{comp['child']}"
                        vinfo['sum_component'] = sum_comp
                        vinfo['product_component'] = product_comp

                    # Calculate depth: use from JSON if available, otherwise compute from vertex depths
                    graph_depth = struct_data.get("depth")
                    if graph_depth is None and vertex_infos:
                        graph_depth = max(v.get('depth', 0) for v in vertex_infos)

                    # Build graph object
                    graph_obj = {
                        'id': f"K{s}_{t}_n{n}_{struct_idx}",
                        'label': f"K_{{{s},{t}}}-free, n={n}",
                        's': s,
                        't': t,
                        'n': n,
                        'depth': graph_depth,
                        'struct_index': struct_idx,
                        'total_structures': len(n_data["structures"]),
                        'edges_count': edges_count,
                        'structure': struct_str,
                        'nodes': vertex_infos,
                        'links': edges
                    }

                    all_graphs.append(graph_obj)
                    graph_count += 1

                    if graph_count % 100 == 0:
                        print(f"  Processed {graph_count} graphs...")

                except Exception as e:
                    print(f"Error processing {filepath} n={n} struct {struct_idx}: {e}", file=sys.stderr)

    # Write output to the exports folder as cache
    output_file = export_dir / "visualization_cache.json"
    with open(output_file, 'w') as f:
        json.dump(all_graphs, f, indent=2)

    print(f"\nGenerated {len(all_graphs)} graphs")
    print(f"Output written to: {output_file}")

    # Size statistics
    sizes = defaultdict(int)
    for g in all_graphs:
        sizes[g['n']] += 1

    print(f"\nGraphs by size:")
    for n in sorted(sizes.keys()):
        print(f"  n={n}: {sizes[n]} graphs")


if __name__ == "__main__":
    main()
