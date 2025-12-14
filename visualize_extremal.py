#!/usr/bin/env python3
"""
Generate interactive browser visualization of extremal cographs.
Uses Cytoscape.js for graph rendering with structure-aware layout.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_conjectures import parse_structure, StructureNode


def cotree_to_positions(node: StructureNode, x_offset=0, y_offset=0, width=800, height=600, depth=0):
    """
    Compute positions for vertices based on cotree structure.
    - Sum nodes: arrange children in a circle (clique)
    - Product nodes: arrange children in two horizontal groups (bipartite)
    """
    positions = {}

    if node.op == "vertex":
        # Single vertex
        positions[node.vertex_id] = (x_offset + width/2, y_offset + height/2)
        return positions, 1

    if node.op == "sum":
        # Arrange children in a circle (since they form a clique)
        total_vertices = 0
        child_positions = []

        for child in node.children:
            child_pos, child_n = cotree_to_positions(
                child,
                x_offset=0,
                y_offset=0,
                width=100,
                height=100,
                depth=depth+1
            )
            child_positions.append((child_pos, child_n))
            total_vertices += child_n

        # Arrange in circle
        import math
        radius = min(width, height) * 0.4
        center_x = x_offset + width / 2
        center_y = y_offset + height / 2

        current_angle = 0
        for child_pos, child_n in child_positions:
            angle_span = 2 * math.pi * child_n / total_vertices

            for vid, (cx, cy) in child_pos.items():
                angle = current_angle + angle_span * (cx / 100)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions[vid] = (x, y)

            current_angle += angle_span

        return positions, total_vertices

    elif node.op == "product":
        # Arrange children in two groups (bipartite layout)
        if len(node.children) != 2:
            # Fall back to circular for non-binary products
            return cotree_to_positions(
                StructureNode(op="sum", children=node.children, n=node.n),
                x_offset, y_offset, width, height, depth
            )

        left_child, right_child = node.children

        # Recursively get positions for left and right
        left_pos, left_n = cotree_to_positions(
            left_child,
            x_offset=x_offset,
            y_offset=y_offset,
            width=width * 0.4,
            height=height,
            depth=depth+1
        )

        right_pos, right_n = cotree_to_positions(
            right_child,
            x_offset=x_offset + width * 0.6,
            y_offset=y_offset,
            width=width * 0.4,
            height=height,
            depth=depth+1
        )

        positions.update(left_pos)
        positions.update(right_pos)

        return positions, left_n + right_n

    return positions, 0


def structure_to_graph_data(node: StructureNode, vertex_counter=[0]):
    """
    Convert cotree structure to graph edges and vertex list.
    Returns (vertices, edges, cotree_info)
    """
    if node.op == "vertex":
        vid = vertex_counter[0]
        vertex_counter[0] += 1
        node.vertex_id = vid
        return [vid], [], {"id": vid, "type": "vertex"}

    all_vertices = []
    all_edges = []
    child_info = []

    for child in node.children:
        v, e, info = structure_to_graph_data(child, vertex_counter)
        all_vertices.extend(v)
        all_edges.extend(e)
        child_info.append(info)

    if node.op == "sum":
        # Sum: disjoint union (no edges between components, but within each)
        # Vertices in same sum are independent
        pass

    elif node.op == "product":
        # Product: complete bipartite between components
        for i, child_i_verts in enumerate([c["vertices"] for c in child_info]):
            for j, child_j_verts in enumerate([c["vertices"] for c in child_info]):
                if i < j:
                    for v1 in child_i_verts:
                        for v2 in child_j_verts:
                            all_edges.append((v1, v2))

    cotree_node = {
        "type": node.op,
        "vertices": all_vertices,
        "children": child_info
    }

    return all_vertices, all_edges, cotree_node


def generate_html_viewer(output_file="extremal_viewer.html"):
    """
    Generate interactive HTML viewer for all extremal graphs.
    """

    export_dir = Path("exports")
    all_files = sorted(export_dir.glob("extremal_K*.json"))

    # Build data structure for all graphs
    all_graphs = []

    for filepath in all_files[:10]:  # Limit to first 10 files for now
        try:
            with open(filepath) as f:
                data = json.load(f)

            s = data["s"]
            t = data["t"]

            for n_str, n_data in sorted(data.get("extremal_by_n", {}).items()):
                n = int(n_str)
                if n > 20:  # Limit graph size for visualization
                    continue

                for struct_idx, struct_data in enumerate(n_data["structures"][:5]):  # Max 5 per n
                    struct_str = struct_data["structure"]
                    edges_count = struct_data["edges"]

                    try:
                        # Reset vertex counter
                        vertex_counter = [0]
                        root = parse_structure(struct_str)
                        vertices, edges, cotree = structure_to_graph_data(root, vertex_counter)

                        # Compute layout positions
                        positions, _ = cotree_to_positions(root)

                        all_graphs.append({
                            "id": f"K{s}_{t}_n{n}_{struct_idx}",
                            "label": f"K_{{{s},{t}}}-free, n={n}, struct {struct_idx+1}/{len(n_data['structures'])}",
                            "s": s,
                            "t": t,
                            "n": n,
                            "edges_count": edges_count,
                            "structure": struct_str,
                            "vertices": vertices,
                            "edges": [[e[0], e[1]] for e in edges],
                            "positions": {str(vid): list(pos) for vid, pos in positions.items()},
                            "cotree": cotree
                        })
                    except Exception as e:
                        print(f"Error processing {filepath} n={n}: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)

    print(f"Generated data for {len(all_graphs)} graphs")

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Extremal Cograph Viewer</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #container {{
            display: flex;
            gap: 20px;
            height: calc(100vh - 40px);
        }}
        #controls {{
            width: 300px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
        }}
        #cy {{
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #info {{
            width: 300px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
        }}
        h2 {{
            margin-top: 0;
            color: #333;
        }}
        .graph-item {{
            padding: 10px;
            margin: 5px 0;
            background: #f9f9f9;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid transparent;
        }}
        .graph-item:hover {{
            background: #e9e9e9;
        }}
        .graph-item.active {{
            background: #e3f2fd;
            border-color: #2196F3;
        }}
        .graph-label {{
            font-weight: bold;
            color: #1976D2;
        }}
        .graph-details {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .filter {{
            margin: 10px 0;
        }}
        .filter label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .filter select, .filter input {{
            width: 100%;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        button {{
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px 5px 5px 0;
        }}
        button:hover {{
            background: #1976D2;
        }}
        pre {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h2>Extremal Graphs</h2>

            <div class="filter">
                <label>Filter by (s,t):</label>
                <select id="filter-st">
                    <option value="">All</option>
                </select>
            </div>

            <div class="filter">
                <label>Filter by n:</label>
                <select id="filter-n">
                    <option value="">All</option>
                </select>
            </div>

            <div>
                <button onclick="resetLayout()">Reset Layout</button>
                <button onclick="toggleLabels()">Toggle Labels</button>
            </div>

            <div id="graph-list"></div>
        </div>

        <div id="cy"></div>

        <div id="info">
            <h2>Graph Info</h2>
            <div id="info-content">
                <p>Select a graph to view details</p>
            </div>
        </div>
    </div>

    <script>
        const graphsData = {json.dumps(all_graphs, indent=2)};

        let cy;
        let currentGraph = null;
        let showLabels = true;

        // Initialize filters
        const stValues = [...new Set(graphsData.map(g => `${{g.s}},${{g.t}}`))].sort();
        const nValues = [...new Set(graphsData.map(g => g.n))].sort((a,b) => a-b);

        const stSelect = document.getElementById('filter-st');
        stValues.forEach(st => {{
            const opt = document.createElement('option');
            opt.value = st;
            opt.textContent = `K_${{st}}`;
            stSelect.appendChild(opt);
        }});

        const nSelect = document.getElementById('filter-n');
        nValues.forEach(n => {{
            const opt = document.createElement('option');
            opt.value = n;
            opt.textContent = `n=${{n}}`;
            nSelect.appendChild(opt);
        }});

        // Initialize Cytoscape
        function initCytoscape() {{
            cy = cytoscape({{
                container: document.getElementById('cy'),
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': '#2196F3',
                            'label': 'data(label)',
                            'width': 30,
                            'height': 30,
                            'font-size': 12,
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#fff',
                            'text-outline-width': 2,
                            'text-outline-color': '#2196F3'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                            'line-color': '#ccc',
                            'opacity': 0.6
                        }}
                    }},
                    {{
                        selector: 'node.sum-component',
                        style: {{
                            'background-color': '#4CAF50'
                        }}
                    }},
                    {{
                        selector: 'node.product-component',
                        style: {{
                            'background-color': '#FF9800'
                        }}
                    }}
                ]
            }});
        }}

        function loadGraph(graphData) {{
            currentGraph = graphData;

            // Clear existing graph
            cy.elements().remove();

            // Add nodes
            const nodes = graphData.vertices.map(v => ({{
                data: {{
                    id: String(v),
                    label: showLabels ? String(v) : ''
                }},
                position: {{
                    x: graphData.positions[String(v)][0],
                    y: graphData.positions[String(v)][1]
                }}
            }}));

            // Add edges
            const edges = graphData.edges.map((e, i) => ({{
                data: {{
                    id: `e${{i}}`,
                    source: String(e[0]),
                    target: String(e[1])
                }}
            }}));

            cy.add(nodes);
            cy.add(edges);

            cy.fit(50);

            // Update info panel
            updateInfo(graphData);
        }}

        function updateInfo(graphData) {{
            const info = document.getElementById('info-content');
            info.innerHTML = `
                <p><strong>Graph:</strong> K_${{graphData.s}},${{graphData.t}}}-free</p>
                <p><strong>Vertices:</strong> n = ${{graphData.n}}</p>
                <p><strong>Edges:</strong> ${{graphData.edges_count}}</p>
                <p><strong>Structure:</strong></p>
                <pre>${{graphData.structure}}</pre>
                <h3>Cotree Structure</h3>
                <pre>${{JSON.stringify(graphData.cotree, null, 2)}}</pre>
            `;
        }}

        function renderGraphList(graphs) {{
            const list = document.getElementById('graph-list');
            list.innerHTML = '';

            graphs.forEach((g, i) => {{
                const div = document.createElement('div');
                div.className = 'graph-item';
                div.innerHTML = `
                    <div class="graph-label">${{g.label}}</div>
                    <div class="graph-details">
                        Vertices: ${{g.n}} | Edges: ${{g.edges_count}}
                    </div>
                `;
                div.onclick = () => {{
                    document.querySelectorAll('.graph-item').forEach(el => el.classList.remove('active'));
                    div.classList.add('active');
                    loadGraph(g);
                }};
                list.appendChild(div);
            }});

            // Auto-load first graph
            if (graphs.length > 0) {{
                list.firstChild.classList.add('active');
                loadGraph(graphs[0]);
            }}
        }}

        function applyFilters() {{
            const stFilter = document.getElementById('filter-st').value;
            const nFilter = document.getElementById('filter-n').value;

            let filtered = graphsData;

            if (stFilter) {{
                const [s, t] = stFilter.split(',').map(Number);
                filtered = filtered.filter(g => g.s === s && g.t === t);
            }}

            if (nFilter) {{
                const n = Number(nFilter);
                filtered = filtered.filter(g => g.n === n);
            }}

            renderGraphList(filtered);
        }}

        function resetLayout() {{
            if (currentGraph) {{
                loadGraph(currentGraph);
            }}
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            if (currentGraph) {{
                cy.nodes().forEach(node => {{
                    node.data('label', showLabels ? node.id() : '');
                }});
            }}
        }}

        // Event listeners
        document.getElementById('filter-st').addEventListener('change', applyFilters);
        document.getElementById('filter-n').addEventListener('change', applyFilters);

        // Initialize
        initCytoscape();
        renderGraphList(graphsData);
    </script>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Generated interactive viewer: {output_file}")
    print(f"Open in browser: file://{Path(output_file).absolute()}")


if __name__ == "__main__":
    generate_html_viewer()
