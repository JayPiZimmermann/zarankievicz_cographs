# Extremal Cographs

Tools for computing and visualizing extremal K_{s,t}-free cographs.

## Overview

This project provides:
- **Dynamic programming algorithms** for enumerating extremal cographs
- **Interactive browser visualization** for exploring extremal graph structures
- **Theorem verification tools** for checking structural properties
- **Regularity analysis** for k-regular cographs

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cographs

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy
```

## Quick Start

### Browser Visualization

Launch the interactive viewer to explore precomputed extremal cographs:

```bash
python browser.py
```

This starts a local server at http://localhost:8765 and opens the visualization in your browser. The viewer allows you to:
- Browse extremal K_{s,t}-free cographs for various (s,t) pairs
- View cotree structures and graph properties
- Explore different export directories (e.g., `exports_star/`, `exports_lattice_12_2/`)

### Computing Extremal Cographs

The main CLI (`main.py`) provides several commands for building and analyzing extremal cographs:

```bash
# Build registry up to N=20 vertices, pruning K_{8,8}
python main.py run --name myrun --N 20 --T 8

# Check structural conjectures on a run
python main.py check --name myrun

# List all computed runs
python main.py runs

# Analyze extremal graphs for specific parameters
python main.py analyze --n 10 --s 2 --t 3

# Export extremal data
python main.py export --s 3 --t 5 --output exports/
```

#### Build Modes

The project supports multiple build strategies:

1. **Standard build** (`build`): Basic dynamic programming
2. **Fast build** (`fast`): Optimized with numpy and profile-level pruning
3. **Partition build** (`partition`): Partition-based approach with lattice reduction

```bash
# Fast build with lattice reduction
python main.py fast --N 25 --T 12 --lattice --export-dir exports_lattice/

# Partition-based build
python main.py partition --N 30 --T 12 --lattice
```

## Project Structure

```
cographs/
├── main.py                    # Main CLI for building/analyzing cographs
├── browser.py                 # HTTP server for visualization
├── extremal_viewer_d3.html    # Interactive D3.js visualization
├── generate_d3_data.py        # Convert registry to D3 JSON format
│
├── src/                       # Core library
│   ├── cotree.py              # Cotree data structure
│   ├── profile.py             # K_{i,j} profile computation
│   ├── profile_ops.py         # Optimized numpy profile operations
│   ├── registry.py            # Graph storage by profile
│   ├── builder.py             # Basic DP builder
│   ├── fast_builder.py        # Optimized builder with pruning
│   ├── partition_builder.py   # Partition-based builder with lattice
│   ├── compact_storage.py     # Memory-efficient FastRegistry
│   ├── cache.py               # Pickle-based caching
│   └── export.py              # Export to JSON/graph6
│
├── theorem_check/             # Theorem verification
│   └── check_small_cases.py   # Verify small_cases_classification theorem
│
├── regular_cographs/          # Regularity analysis
│   └── calculate_regularity_pairs.py  # Compute realizable regularities
│
├── exports_*/                 # Precomputed extremal data
│   ├── exports_star/          # K_{1,t} extremal (star-free)
│   ├── exports_lattice_12_2/  # K_{s,t} with lattice reduction
│   └── exports_12_complete/   # Complete enumeration up to n=12
│
└── tex/                       # LaTeX paper sources
```

## Theorem Verification

The `theorem_check/` directory contains tools for verifying the **small cases classification theorem**:

> For s ≤ t ≤ min(12, 2s-1) and n ≥ 2t-1, any (s,t)-extremal cograph has structure G_0 × G_1 where |G_0| = s-1 and G_1 is (1,t)-extremal.

```bash
# Verify theorem for s,t ≤ 6
python theorem_check/check_small_cases.py --max-bound 6

# Use custom cache directories
python theorem_check/check_small_cases.py --max-bound 6 \
    --exports-dir exports_lattice_12_2 \
    --exports-star-dir exports_star
```

The verification:
1. Pre-builds G_0 candidates (profile-extremal on s-1 vertices)
2. Pre-builds H_1 component candidates (connected, max_degree < t)
3. For each (s,t) pair, enumerates valid (H_0, G_0, H_1) triples
4. Checks that optimal configurations have H_0 empty and H_1 components are (1,t)-extremal
5. Saves results to `theorem_check/extremal_triples_cache.json`

## Regular Cographs

The `regular_cographs/` directory computes which regularities (degrees) are achievable for k-regular cographs:

```bash
cd regular_cographs
python calculate_regularity_pairs.py
```

This uses dynamic programming to find all pairs (n, k) where a k-regular cograph on n vertices exists:

- **Disjoint union (sum)**: Preserves degree (both parts must have same degree)
- **Complete join (product)**: Preserves deficiency n-k (new degree = n - deficiency)

Results are saved per vertex count in `cograph_data/n_*.json`.

## Key Concepts

### Cographs and Cotrees

A **cograph** (complement-reducible graph) can be recursively constructed by:
- Starting with single vertices
- Taking disjoint unions (sum: G + H)
- Taking complete joins (product: G × H)

Every cograph has a unique **cotree** representation encoding this construction.

### K_{s,t} Profile

For a graph G on n vertices, its **profile** is the tuple (p[0], p[1], ..., p[n]) where:
- p[i] = max{j : K_{i,j} is a subgraph of G}

A graph is **K_{s,t}-free** iff profile[s] < t and profile[t] < s.

### Extremal Graphs

An **(s,t)-extremal cograph** on n vertices is a K_{s,t}-free cograph with the maximum number of edges.

### Lattice Reduction

When only edge counts matter (not specific graph structures), we can use **lattice reduction** to keep only one representative per (profile, edges) pair, dramatically reducing memory usage.

## Precomputed Data

The `exports_*/` directories contain precomputed extremal data:

- `extremal_K{s}{t}.json`: Extremal numbers and structures for K_{s,t}
- Each file contains `extremal_by_n[n]` with edge count and cotree structures

Example JSON structure:
```json
{
  "s": 2, "t": 3,
  "max_n": 29,
  "extremal_by_n": {
    "5": {
      "ex": 4,
      "count": 1,
      "structures": [{"structure": "P(1,1,S(1,1,1))", "edges": 4, ...}]
    }
  }
}
```

## License

[Add license information]
