#!/usr/bin/env python3
"""
Analyze depth patterns in extremal graphs to find relationships between
depth and parameters s, t, n, and structure.
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

def analyze_depths_in_export(export_dir: Path):
    """Analyze depth patterns in a single export directory."""

    # Collect all depth data
    depth_data = []

    json_files = sorted(export_dir.glob("extremal_K*.json"))

    for json_path in json_files:
        # Parse s,t from filename
        filename = json_path.stem  # e.g., "extremal_K35"
        st_part = filename.replace("extremal_K", "")

        with open(json_path, 'r') as f:
            data = json.load(f)

        s = data['s']
        t = data['t']

        for n_str, n_data in data.get('extremal_by_n', {}).items():
            n = int(n_str)
            structures = n_data.get('structures', [])

            if not structures:
                continue

            # Collect depths for this (s,t,n)
            depths = [struct.get('depth') for struct in structures if 'depth' in struct]

            if depths:
                depth_data.append({
                    's': s,
                    't': t,
                    'n': n,
                    'min_depth': min(depths),
                    'max_depth': max(depths),
                    'num_structures': len(structures),
                    'structures': structures
                })

    return depth_data


def analyze_patterns(depth_data):
    """Analyze patterns in depth data."""

    print("=" * 80)
    print("DEPTH ANALYSIS")
    print("=" * 80)

    # Group by (s, t)
    by_st = defaultdict(list)
    for entry in depth_data:
        by_st[(entry['s'], entry['t'])].append(entry)

    print(f"\n1. OVERALL STATISTICS")
    print("-" * 80)

    all_min_depths = [e['min_depth'] for e in depth_data]
    all_max_depths = [e['max_depth'] for e in depth_data]

    print(f"Total (s,t,n) combinations: {len(depth_data)}")
    print(f"Global min depth: {min(all_min_depths)}")
    print(f"Global max depth: {max(all_max_depths)}")
    print(f"Average min depth: {sum(all_min_depths) / len(all_min_depths):.2f}")
    print(f"Average max depth: {sum(all_max_depths) / len(all_max_depths):.2f}")

    # Find maximum depth case
    max_depth_entry = max(depth_data, key=lambda e: e['max_depth'])
    print(f"\nMaximum depth observed:")
    print(f"  Depth: {max_depth_entry['max_depth']}")
    print(f"  K_{{{max_depth_entry['s']},{max_depth_entry['t']}}} at n={max_depth_entry['n']}")
    print(f"  Number of structures: {max_depth_entry['num_structures']}")

    # Show structure(s) with maximum depth
    for struct in max_depth_entry['structures']:
        if struct.get('depth') == max_depth_entry['max_depth']:
            print(f"  Structure: {struct.get('structure', 'N/A')[:100]}...")
            break

    print(f"\n2. DEPTH BY (s, t)")
    print("-" * 80)

    # Show some interesting cases
    for (s, t) in sorted(by_st.keys())[:20]:  # First 20 cases
        entries = by_st[(s, t)]
        max_n = max(e['n'] for e in entries)
        max_depth_at_max_n = [e for e in entries if e['n'] == max_n][0]['max_depth']

        print(f"K_{{{s},{t}}}: max_n={max_n}, max_depth={max_depth_at_max_n}")

    print(f"\n3. DEPTH GROWTH WITH n")
    print("-" * 80)

    # For a few fixed (s,t), see how depth grows with n
    sample_st = [(1, 1), (2, 2), (3, 3), (5, 5), (9, 9)]

    for (s, t) in sample_st:
        if (s, t) not in by_st:
            continue

        entries = sorted(by_st[(s, t)], key=lambda e: e['n'])
        print(f"\nK_{{{s},{t}}}:")

        # Show depth progression
        for entry in entries[:10]:  # First 10 n values
            print(f"  n={entry['n']:2d}: min_depth={entry['min_depth']}, max_depth={entry['max_depth']}, "
                  f"structures={entry['num_structures']}")

    print(f"\n4. DEPTH FORMULAS AND PATTERNS")
    print("-" * 80)

    # Check if min_depth follows patterns
    print("\nChecking pattern: min_depth = 1 for all n >= s+t:")
    pattern_holds = True
    exceptions = []

    for entry in depth_data:
        s, t, n = entry['s'], entry['t'], entry['n']
        min_depth = entry['min_depth']

        if n >= s + t - 1:  # We can form K_{s,t} with simple product
            if min_depth != 1:
                pattern_holds = False
                exceptions.append((s, t, n, min_depth))

    if pattern_holds:
        print("  ✓ Pattern holds: min_depth = 1 when n >= s+t-1")
    else:
        print(f"  ✗ Pattern violated in {len(exceptions)} cases:")
        for s, t, n, d in exceptions[:5]:
            print(f"    K_{{{s},{t}}} at n={n}: min_depth={d}")

    print("\nChecking pattern: depth = 0 only for n=1:")
    depth_0_cases = [e for e in depth_data if e['min_depth'] == 0]
    print(f"  Found {len(depth_0_cases)} cases with depth=0")
    if all(e['n'] == 1 for e in depth_0_cases):
        print("  ✓ All depth=0 cases have n=1")
    else:
        non_trivial = [e for e in depth_0_cases if e['n'] > 1]
        print(f"  ✗ Found {len(non_trivial)} depth=0 cases with n>1:")
        for e in non_trivial[:5]:
            print(f"    K_{{{e['s']},{e['t']}}} at n={e['n']}")

    print("\nAnalyzing max_depth growth:")
    # For K_{i,i}, see if max_depth grows with i
    symmetric_cases = [(s, t) for (s, t) in by_st.keys() if s == t]
    symmetric_cases.sort()

    print("  Max depth for K_{i,i} (symmetric cases) at largest n:")
    for (s, t) in symmetric_cases[:15]:
        entries = by_st[(s, t)]
        max_n_entry = max(entries, key=lambda e: e['n'])
        print(f"    K_{{{s},{s}}}: n={max_n_entry['n']}, max_depth={max_n_entry['max_depth']}")

    print(f"\n5. STRUCTURE ANALYSIS FOR HIGH DEPTH")
    print("-" * 80)

    # Look at structures with depth >= 3
    high_depth_entries = [e for e in depth_data if e['max_depth'] >= 3]
    print(f"\nFound {len(high_depth_entries)} (s,t,n) combinations with max_depth >= 3")

    # Sample some high-depth structures
    print("\nSample high-depth structures:")
    for entry in sorted(high_depth_entries, key=lambda e: e['max_depth'], reverse=True)[:10]:
        s, t, n = entry['s'], entry['t'], entry['n']
        max_d = entry['max_depth']

        # Find a structure with max depth
        for struct in entry['structures']:
            if struct.get('depth') == max_d:
                structure_str = struct.get('structure', 'N/A')
                print(f"\n  K_{{{s},{t}}} at n={n}, depth={max_d}:")
                print(f"    {structure_str[:120]}{'...' if len(structure_str) > 120 else ''}")
                break

    print(f"\n6. DEPTH AND OPERATION TYPE")
    print("-" * 80)

    # Count last operations by depth
    op_by_depth = defaultdict(lambda: defaultdict(int))

    for entry in depth_data:
        for struct in entry['structures']:
            depth = struct.get('depth')
            last_op = struct.get('last_op')
            if depth is not None and last_op:
                op_by_depth[depth][last_op] += 1

    print("\nLast operation distribution by depth:")
    for depth in sorted(op_by_depth.keys())[:8]:
        ops = op_by_depth[depth]
        total = sum(ops.values())
        print(f"  Depth {depth} (total {total}):")
        for op, count in sorted(ops.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            print(f"    {op}: {count} ({pct:.1f}%)")


def main():
    # Analyze exports_lattice_12_2 (most comprehensive)
    export_dirs = [
        Path("exports_lattice_12_2"),
        Path("exports_lattice_5_2"),
        Path("exports_star")
    ]

    for export_dir in export_dirs:
        if not export_dir.exists():
            print(f"Warning: {export_dir} not found, skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {export_dir}")
        print(f"{'=' * 80}")

        depth_data = analyze_depths_in_export(export_dir)

        if not depth_data:
            print("No depth data found")
            continue

        analyze_patterns(depth_data)
        print()


if __name__ == "__main__":
    main()
