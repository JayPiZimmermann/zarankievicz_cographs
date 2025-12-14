#!/usr/bin/env python3
"""
Generate D3.js-ready data for interactive cograph visualization.
Computes structure-aware layouts respecting cotree operations.
"""

import json
import sys
import math
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from analyze_conjectures import parse_structure, StructureNode


class VertexLayouter:
    """Manages vertex layout based on cotree structure."""

    def __init__(self):
        self.vertex_id = 0
        self.vertex_info = {}  # vid -> {pos, group, depth, type}

    def layout_cotree(self, node, cx=0, cy=0, width=1000, height=1000, depth=0, parent_id=None):
        """
        Recursively compute layout positions for cotree.
        Returns: list of vertex info dicts
        """
        if node.op == "vertex":
            vid = self.vertex_id
            self.vertex_id += 1

            info = {
                'id': vid,
                'x': cx + width / 2,
                'y': cy + height / 2,
                'depth': depth,
                'type': 'vertex',
                'group': parent_id if parent_id is not None else vid,
                'parent': parent_id
            }
            self.vertex_info[vid] = info
            return [info], [vid]

        if node.op == "sum":
            # Sum: clique - arrange children in circular pattern
            return self._layout_sum(node, cx, cy, width, height, depth, parent_id)

        elif node.op == "product":
            # Product: independent sets - arrange in separated groups
            return self._layout_product(node, cx, cy, width, height, depth, parent_id)

        return [], []

    def _layout_sum(self, node, cx, cy, width, height, depth, parent_id):
        """Layout sum node: arrange children in circular cluster."""
        group_id = self.vertex_id  # Use first vertex ID as group ID

        all_infos = []
        all_vids = []

        # First pass: get all children
        child_results = []
        for child in node.children:
            infos, vids = self.layout_cotree(
                child, cx, cy, width, height, depth + 1, group_id
            )
            child_results.append((infos, vids))
            all_infos.extend(infos)
            all_vids.extend(vids)

        # Arrange in circle
        n_vertices = len(all_vids)
        if n_vertices == 0:
            return all_infos, all_vids

        radius = min(width, height) * 0.35
        center_x = cx + width / 2
        center_y = cy + height / 2

        # Distribute vertices around circle
        for i, vid in enumerate(all_vids):
            angle = 2 * math.pi * i / n_vertices
            self.vertex_info[vid]['x'] = center_x + radius * math.cos(angle)
            self.vertex_info[vid]['y'] = center_y + radius * math.sin(angle)
            self.vertex_info[vid]['sum_group'] = group_id

        return all_infos, all_vids

    def _layout_product(self, node, cx, cy, width, height, depth, parent_id):
        """Layout product node: arrange children in separated horizontal groups."""
        all_infos = []
        all_vids = []

        n_children = len(node.children)
        if n_children == 0:
            return all_infos, all_vids

        # Calculate horizontal spacing for children
        child_width = width / n_children
        gap = child_width * 0.1  # 10% gap between groups

        for i, child in enumerate(node.children):
            child_cx = cx + i * child_width + gap / 2
            child_width_actual = child_width - gap

            infos, vids = self.layout_cotree(
                child, child_cx, cy, child_width_actual, height,
                depth + 1, parent_id
            )

            all_infos.extend(infos)
            all_vids.extend(vids)

            # Mark as product group
            for vid in vids:
                self.vertex_info[vid]['product_group'] = i
                self.vertex_info[vid]['product_parent'] = parent_id

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
    print("Generating D3.js visualization data...")

    export_dir = Path("exports")
    if not export_dir.exists():
        print("Error: exports directory not found")
        sys.exit(1)

    # Load exceptional cases
    exceptional_cases = set()
    if Path("partition_presence.csv").exists():
        with open("partition_presence.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["has_partition"] == "0":
                    key = (int(row["s"]), int(row["t"]), int(row["n"]))
                    exceptional_cases.add(key)

    print(f"Loaded {len(exceptional_cases)} exceptional cases")

    # Process graphs
    all_files = sorted(export_dir.glob("extremal_K*.json"))
    all_graphs = []

    graph_count = 0
    max_graphs = 500  # Limit for reasonable browser performance

    for filepath in all_files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            s = data["s"]
            t = data["t"]

            # Process each n value
            for n_str in sorted(data.get("extremal_by_n", {}).keys()):
                n = int(n_str)

                if n > 25:  # Skip large graphs
                    continue

                n_data = data["extremal_by_n"][n_str]

                # Take only first few structures per (s,t,n)
                max_per_n = 3
                for struct_idx, struct_data in enumerate(n_data["structures"][:max_per_n]):
                    if graph_count >= max_graphs:
                        break

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

                        # Check if exceptional
                        is_exceptional = (s, t, n) in exceptional_cases

                        # Build graph object
                        graph_obj = {
                            'id': f"K{s}_{t}_n{n}_{struct_idx}",
                            'label': f"K_{{{s},{t}}}-free, n={n}",
                            's': s,
                            't': t,
                            'n': n,
                            'struct_index': struct_idx,
                            'total_structures': len(n_data["structures"]),
                            'edges_count': edges_count,
                            'structure': struct_str,
                            'is_exceptional': is_exceptional,
                            'nodes': vertex_infos,
                            'links': edges
                        }

                        all_graphs.append(graph_obj)
                        graph_count += 1

                        if graph_count % 50 == 0:
                            print(f"  Processed {graph_count} graphs...")

                    except Exception as e:
                        print(f"Error processing {filepath} n={n} struct {struct_idx}: {e}", file=sys.stderr)

                if graph_count >= max_graphs:
                    break

        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)

        if graph_count >= max_graphs:
            print(f"Reached maximum of {max_graphs} graphs")
            break

    # Write output
    output_file = "extremal_graphs.json"
    with open(output_file, 'w') as f:
        json.dump(all_graphs, f, indent=2)

    print(f"\nGenerated {len(all_graphs)} graphs")
    print(f"Output written to: {output_file}")

    # Statistics
    exceptional_count = sum(1 for g in all_graphs if g['is_exceptional'])
    print(f"Exceptional graphs: {exceptional_count}")
    print(f"Regular graphs: {len(all_graphs) - exceptional_count}")

    # Size statistics
    sizes = defaultdict(int)
    for g in all_graphs:
        sizes[g['n']] += 1

    print(f"\nGraphs by size:")
    for n in sorted(sizes.keys()):
        print(f"  n={n}: {sizes[n]} graphs")


if __name__ == "__main__":
    main()
