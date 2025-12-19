#!/usr/bin/env python3
"""
Generate D3.js-ready data for interactive cograph visualization.
Computes structure-aware layouts respecting cotree operations.
"""

import json
import sys
import math
import csv
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from analyze_conjectures import parse_structure, StructureNode


class VertexLayouter:
    """Manages vertex layout based on cotree structure."""

    def __init__(self):
        self.vertex_id = 0
        self.vertex_info = {}  # vid -> {pos, group, depth, type}
        self.component_id = 0
        self.components = []  # list of {id, op, level, vertices}

    def _count_vertices(self, node):
        """Count vertices in a subtree."""
        if node.op == "vertex":
            return 1
        return sum(self._count_vertices(c) for c in node.children)

    def _is_single_vertex(self, node):
        """Check if node is a single vertex."""
        return node.op == "vertex"

    def _get_structure_type(self, node):
        """
        Analyze structure type for layout decisions.
        Returns: 'vertex', 'clique', 'independent', 'multipartite', 'hub_star', 'general'
        """
        if node.op == "vertex":
            return "vertex"

        if node.op == "sum":
            # Sum of vertices = independent set (no edges within)
            if all(c.op == "vertex" for c in node.children):
                return "independent"
            return "general_sum"

        if node.op == "product":
            # Product of vertices = clique
            if all(c.op == "vertex" for c in node.children):
                return "clique"

            # Check for hub pattern: one single vertex + other structures
            single_vertices = [c for c in node.children if self._is_single_vertex(c)]
            other_children = [c for c in node.children if not self._is_single_vertex(c)]

            if len(single_vertices) == 1 and len(other_children) >= 1:
                return "hub_star"

            # Product of sums = complete multipartite
            if all(c.op == "sum" or c.op == "vertex" for c in node.children):
                return "multipartite"

            return "general_product"

        return "general"

    def layout_cotree(self, node, cx=0, cy=0, width=1000, height=1000, depth=0, parent_id=None, component_path=None):
        """
        Recursively compute layout positions for cotree.
        Returns: list of vertex info dicts, list of vertex ids

        component_path: list of (component_id, op_type) tuples from root to current
        """
        if component_path is None:
            component_path = []

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
                'parent': parent_id,
                'components': list(component_path)  # Copy the path
            }
            self.vertex_info[vid] = info
            return [info], [vid]

        structure = self._get_structure_type(node)

        if structure == "hub_star":
            return self._layout_hub_star(node, cx, cy, width, height, depth, parent_id, component_path)
        elif structure == "multipartite":
            return self._layout_multipartite(node, cx, cy, width, height, depth, parent_id, component_path)
        elif node.op == "sum":
            return self._layout_sum(node, cx, cy, width, height, depth, parent_id, component_path)
        elif node.op == "product":
            return self._layout_product(node, cx, cy, width, height, depth, parent_id, component_path)

        return [], []

    def _layout_hub_star(self, node, cx, cy, width, height, depth, parent_id, component_path):
        """
        Layout hub-star pattern: central vertex connected to multiple structures.
        Places hub in center, arranges other children radially.
        """
        center_x = cx + width / 2
        center_y = cy + height / 2

        # Separate hub vertex from other children
        hub_child = None
        other_children = []
        for child in node.children:
            if self._is_single_vertex(child) and hub_child is None:
                hub_child = child
            else:
                other_children.append(child)

        all_infos = []
        all_vids = []

        # Create component for this product node
        parent_comp_id = self.component_id
        self.component_id += 1

        # Layout hub vertex at center (component 0)
        group_id = self.vertex_id
        hub_comp_path = component_path + [{'id': parent_comp_id, 'op': 'product', 'child': 0, 'level': depth}]
        hub_infos, hub_vids = self.layout_cotree(
            hub_child, cx, cy, width, height, depth + 1, group_id, hub_comp_path
        )
        # Force hub to exact center
        for vid in hub_vids:
            self.vertex_info[vid]['x'] = center_x
            self.vertex_info[vid]['y'] = center_y
            self.vertex_info[vid]['is_hub'] = True
        all_infos.extend(hub_infos)
        all_vids.extend(hub_vids)

        # Layout other children radially
        n_branches = len(other_children)
        if n_branches == 0:
            return all_infos, all_vids

        # Calculate sizes for proportional angular allocation
        branch_sizes = [self._count_vertices(c) for c in other_children]
        total_size = sum(branch_sizes)

        # Radial distance based on total graph size
        base_radius = min(width, height) * 0.35

        current_angle = -math.pi / 2  # Start from top

        for i, (child, size) in enumerate(zip(other_children, branch_sizes)):
            # Angular span proportional to size
            angle_span = 2 * math.pi * size / total_size
            branch_angle = current_angle + angle_span / 2

            # Distance from center - larger branches go further out
            branch_radius = base_radius * (0.8 + 0.4 * size / max(branch_sizes))

            # Calculate bounding box for this branch
            branch_cx = center_x + branch_radius * math.cos(branch_angle)
            branch_cy = center_y + branch_radius * math.sin(branch_angle)

            # Size of branch area
            branch_size = min(width, height) * 0.3 * math.sqrt(size / total_size)

            # Component path for this child (child index i+1, since hub is 0)
            child_comp_path = component_path + [{'id': parent_comp_id, 'op': 'product', 'child': i + 1, 'level': depth}]

            infos, vids = self.layout_cotree(
                child,
                branch_cx - branch_size / 2,
                branch_cy - branch_size / 2,
                branch_size, branch_size,
                depth + 1, group_id, child_comp_path
            )

            # Mark branch membership
            for vid in vids:
                self.vertex_info[vid]['branch'] = i
                self.vertex_info[vid]['branch_angle'] = branch_angle

            all_infos.extend(infos)
            all_vids.extend(vids)

            current_angle += angle_span

        return all_infos, all_vids

    def _layout_multipartite(self, node, cx, cy, width, height, depth, parent_id, component_path):
        """
        Layout complete multipartite graph: product of independent sets.
        Arranges parts radially with vertices of each part clustered.
        """
        center_x = cx + width / 2
        center_y = cy + height / 2

        all_infos = []
        all_vids = []

        n_parts = len(node.children)
        if n_parts == 0:
            return all_infos, all_vids

        group_id = self.vertex_id

        # For small number of parts, use radial layout
        # For 2 parts (bipartite), use left-right layout
        if n_parts == 2:
            return self._layout_bipartite(node, cx, cy, width, height, depth, parent_id, component_path)

        # Create component for this product node
        parent_comp_id = self.component_id
        self.component_id += 1

        base_radius = min(width, height) * 0.35

        for i, child in enumerate(node.children):
            angle = 2 * math.pi * i / n_parts - math.pi / 2

            part_cx = center_x + base_radius * math.cos(angle)
            part_cy = center_y + base_radius * math.sin(angle)

            # Size for this part
            part_size = min(width, height) * 0.25

            # Component path for this child
            child_comp_path = component_path + [{'id': parent_comp_id, 'op': 'product', 'child': i, 'level': depth}]

            infos, vids = self.layout_cotree(
                child,
                part_cx - part_size / 2,
                part_cy - part_size / 2,
                part_size, part_size,
                depth + 1, group_id, child_comp_path
            )

            # Mark part membership
            for vid in vids:
                self.vertex_info[vid]['multipartite_part'] = i

            all_infos.extend(infos)
            all_vids.extend(vids)

        return all_infos, all_vids

    def _layout_bipartite(self, node, cx, cy, width, height, depth, parent_id, component_path):
        """Layout bipartite graph with two columns."""
        all_infos = []
        all_vids = []

        group_id = self.vertex_id

        # Create component for this product node
        parent_comp_id = self.component_id
        self.component_id += 1

        left_child = node.children[0]
        right_child = node.children[1]

        # Left side (component 0)
        left_comp_path = component_path + [{'id': parent_comp_id, 'op': 'product', 'child': 0, 'level': depth}]
        left_infos, left_vids = self.layout_cotree(
            left_child,
            cx + width * 0.1, cy,
            width * 0.3, height,
            depth + 1, group_id, left_comp_path
        )
        for vid in left_vids:
            self.vertex_info[vid]['bipartite_side'] = 'left'
        all_infos.extend(left_infos)
        all_vids.extend(left_vids)

        # Right side (component 1)
        right_comp_path = component_path + [{'id': parent_comp_id, 'op': 'product', 'child': 1, 'level': depth}]
        right_infos, right_vids = self.layout_cotree(
            right_child,
            cx + width * 0.6, cy,
            width * 0.3, height,
            depth + 1, group_id, right_comp_path
        )
        for vid in right_vids:
            self.vertex_info[vid]['bipartite_side'] = 'right'
        all_infos.extend(right_infos)
        all_vids.extend(right_vids)

        return all_infos, all_vids

    def _layout_sum(self, node, cx, cy, width, height, depth, parent_id, component_path):
        """Layout sum node: arrange children in compact cluster (they form independent set)."""
        group_id = self.vertex_id

        all_infos = []
        all_vids = []

        # Create component for this sum node
        parent_comp_id = self.component_id
        self.component_id += 1

        # First pass: collect all children with component tracking
        child_results = []
        for i, child in enumerate(node.children):
            child_comp_path = component_path + [{'id': parent_comp_id, 'op': 'sum', 'child': i, 'level': depth}]
            infos, vids = self.layout_cotree(
                child, cx, cy, width, height, depth + 1, group_id, child_comp_path
            )
            child_results.append((infos, vids))
            all_infos.extend(infos)
            all_vids.extend(vids)

        # Arrange in compact grid or circle
        n_vertices = len(all_vids)
        if n_vertices == 0:
            return all_infos, all_vids

        center_x = cx + width / 2
        center_y = cy + height / 2

        if n_vertices <= 6:
            # Small: arrange in circle
            radius = min(width, height) * 0.25
            for i, vid in enumerate(all_vids):
                angle = 2 * math.pi * i / n_vertices - math.pi / 2
                self.vertex_info[vid]['x'] = center_x + radius * math.cos(angle)
                self.vertex_info[vid]['y'] = center_y + radius * math.sin(angle)
        else:
            # Larger: arrange in grid
            cols = math.ceil(math.sqrt(n_vertices))
            rows = math.ceil(n_vertices / cols)
            spacing = min(width, height) * 0.8 / max(cols, rows)

            start_x = center_x - (cols - 1) * spacing / 2
            start_y = center_y - (rows - 1) * spacing / 2

            for i, vid in enumerate(all_vids):
                row = i // cols
                col = i % cols
                self.vertex_info[vid]['x'] = start_x + col * spacing
                self.vertex_info[vid]['y'] = start_y + row * spacing

        for vid in all_vids:
            self.vertex_info[vid]['sum_group'] = group_id

        return all_infos, all_vids

    def _layout_product(self, node, cx, cy, width, height, depth, parent_id, component_path):
        """Layout product node: arrange children radially (they form complete multipartite)."""
        all_infos = []
        all_vids = []

        n_children = len(node.children)
        if n_children == 0:
            return all_infos, all_vids

        group_id = self.vertex_id
        center_x = cx + width / 2
        center_y = cy + height / 2

        # Create component for this product node
        parent_comp_id = self.component_id
        self.component_id += 1

        # Radial arrangement for products
        base_radius = min(width, height) * 0.3

        for i, child in enumerate(node.children):
            angle = 2 * math.pi * i / n_children - math.pi / 2

            child_cx = center_x + base_radius * math.cos(angle)
            child_cy = center_y + base_radius * math.sin(angle)

            child_size = min(width, height) * 0.35

            # Component path for this child
            child_comp_path = component_path + [{'id': parent_comp_id, 'op': 'product', 'child': i, 'level': depth}]

            infos, vids = self.layout_cotree(
                child,
                child_cx - child_size / 2,
                child_cy - child_size / 2,
                child_size, child_size,
                depth + 1, group_id, child_comp_path
            )

            all_infos.extend(infos)
            all_vids.extend(vids)

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
    parser = argparse.ArgumentParser(description="Generate D3.js visualization data from extremal graph exports")
    parser.add_argument("export_dir", nargs="?", default="exports",
                        help="Path to the exports directory (default: exports)")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    print(f"Generating D3.js visualization data from: {export_dir}")

    if not export_dir.exists():
        print(f"Error: exports directory not found: {export_dir}")
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

    # Write output to the exports folder as cache
    output_file = export_dir / "visualization_cache.json"
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
