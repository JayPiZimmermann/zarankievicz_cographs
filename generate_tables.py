#!/usr/bin/env python3
"""
Generate tables for extremal K_{s,t}-free cographs analysis.

Table 1: Presence of P(s-1, n-s+1) structure (0/1)
Table 2: Maximum cotree height with row/column maxima
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from analyze_conjectures import parse_structure, StructureNode


def get_tree_height(node: StructureNode) -> int:
    """Compute height of cotree (maximum depth)."""
    if node.op == "vertex":
        return 0

    if not node.children:
        return 0

    # Height is 1 + max height of children
    return 1 + max(get_tree_height(child) for child in node.children)


def has_partition(structures: list, n: int, s: int, expected_partition: tuple) -> bool:
    """Check if any structure has the expected partition."""
    for struct_data in structures:
        struct_str = struct_data["structure"]
        try:
            node = parse_structure(struct_str)
            if node.op == "product":
                actual = tuple(sorted([node.children[0].n, node.children[1].n]))
                if actual == expected_partition:
                    return True
        except Exception:
            pass
    return False


def get_max_height(structures: list) -> int:
    """Get maximum tree height among all structures."""
    max_h = 0
    for struct_data in structures:
        struct_str = struct_data["structure"]
        try:
            node = parse_structure(struct_str)
            h = get_tree_height(node)
            max_h = max(max_h, h)
        except Exception:
            pass
    return max_h


def get_min_height(structures: list) -> int:
    """Get minimum tree height among all structures."""
    min_h = float('inf')
    for struct_data in structures:
        struct_str = struct_data["structure"]
        try:
            node = parse_structure(struct_str)
            h = get_tree_height(node)
            min_h = min(min_h, h)
        except Exception:
            pass
    return min_h if min_h != float('inf') else 0


def main():
    export_dir = Path("exports")

    if not export_dir.exists():
        print("Error: exports directory not found")
        sys.exit(1)

    # Load all files
    all_files = sorted(export_dir.glob("extremal_K*.json"))
    print(f"DEBUG: Found {len(all_files)} JSON files", file=sys.stderr)

    # Parse (s, t) from filenames and load data
    data_by_st = {}

    for i, filepath in enumerate(all_files):
        try:
            print(f"DEBUG: Loading {i+1}/{len(all_files)}: {filepath.name}", file=sys.stderr)
            with open(filepath) as f:
                data = json.load(f)

            s = data["s"]
            t = data["t"]

            # Process all s,t >= 2
            if s >= 2 and t >= 2:
                data_by_st[(s, t)] = data
                print(f"DEBUG:   Loaded K_{{{s},{t}}} with {len(data.get('extremal_by_n', {}))} n values", file=sys.stderr)
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)

    # Determine range of s, t, n
    s_values = sorted(set(s for s, t in data_by_st.keys()))
    t_values = sorted(set(t for s, t in data_by_st.keys()))
    print(f"DEBUG: s values: {s_values}", file=sys.stderr)
    print(f"DEBUG: t values: {t_values}", file=sys.stderr)

    # Find max n across all files
    max_n = 0
    for data in data_by_st.values():
        for n_str in data.get("extremal_by_n", {}).keys():
            max_n = max(max_n, int(n_str))

    # Limit to reasonable n
    max_n = min(max_n, 25)
    n_values = list(range(2, max_n + 1))
    print(f"DEBUG: max_n = {max_n}, processing n from 2 to {max_n}", file=sys.stderr)

    print("=" * 120)
    print("TABLE 1: PRESENCE OF P(s-1, n-s+1) STRUCTURE")
    print("=" * 120)
    print()
    print("For each (s,t), shows which n have the P(s-1, n-s+1) partition among extremal graphs")
    print("1 = present, 0 = absent, - = no data")
    print()

    # Generate Table 1 for each (s, t)
    print(f"DEBUG: Generating Table 1...", file=sys.stderr)
    total_st = sum(1 for s in s_values for t in t_values if t >= s and (s,t) in data_by_st)
    current_st = 0

    for s in s_values:
        for t in t_values:
            if t < s:
                continue  # Skip since K_{s,t} = K_{t,s}

            if (s, t) not in data_by_st:
                continue

            current_st += 1
            print(f"DEBUG: Processing Table 1: {current_st}/{total_st} K_{{{s},{t}}}", file=sys.stderr)

            data = data_by_st[(s, t)]

            print(f"\nK_{{{s},{t}}}:")
            print(f"{'n':>4} | {'P(s-1,n-s+1)':>15} | Actual partition")
            print("-" * 60)

            for n in n_values:
                n_str = str(n)

                if n_str not in data.get("extremal_by_n", {}):
                    continue

                n_data = data["extremal_by_n"][n_str]
                structures = n_data["structures"]

                # Expected partition
                if n >= s:
                    expected = tuple(sorted([s - 1, n - s + 1]))
                    has_it = has_partition(structures, n, s, expected)
                    status = "1" if has_it else "0"
                    print(f"{n:4d} | {status:>15s} | P{expected}")
                else:
                    print(f"{n:4d} | {'-':>15s} | n < s")

    print("\n" + "=" * 120)
    print("TABLE 2: MAXIMUM COTREE HEIGHT")
    print("=" * 120)
    print()
    print("Maximum height of cotree structure among all extremal graphs")
    print()

    # Build height table data (both min and max)
    print(f"DEBUG: Building height tables...", file=sys.stderr)
    max_height_data = defaultdict(lambda: defaultdict(int))
    min_height_data = defaultdict(lambda: defaultdict(int))

    total_entries = sum(len(data.get("extremal_by_n", {})) for data in data_by_st.values())
    processed = 0

    for (s, t), data in data_by_st.items():
        print(f"DEBUG: Computing heights for K_{{{s},{t}}}", file=sys.stderr)
        for n_str, n_data in data.get("extremal_by_n", {}).items():
            n = int(n_str)
            if n > max_n:
                continue

            structures = n_data["structures"]
            max_h = get_max_height(structures)
            min_h = get_min_height(structures)
            max_height_data[(s, t)][n] = max_h
            min_height_data[(s, t)][n] = min_h

            processed += 1
            if processed % 100 == 0:
                print(f"DEBUG:   Processed {processed}/{total_entries} entries", file=sys.stderr)

    # For each (s, t), print a table
    for s in s_values[:5]:  # Limit for readability
        print(f"\n{'s/t':>4}", end="")
        for t in t_values:
            if t >= s:  # Only show t >= s
                print(f" {t:>3}", end="")
        print("  | Max")
        print("-" * (4 + 4 * len([t for t in t_values if t >= s]) + 8))

        for n in n_values[:15]:  # First 15 n values for readability
            print(f"{n:4d}", end="")
            row_max = 0
            for t in t_values:
                if t >= s:
                    if (s, t) in max_height_data and n in max_height_data[(s, t)]:
                        h = max_height_data[(s, t)][n]
                        row_max = max(row_max, h)
                        print(f" {h:>3}", end="")
                    else:
                        print(f" {'-':>3}", end="")
            print(f"  | {row_max:>3}")

    # Now create comprehensive table with row/column maxima
    print("\n" + "=" * 120)
    print("TABLE 3: MAXIMUM COTREE HEIGHT (COMPREHENSIVE WITH MAXIMA)")
    print("=" * 120)
    print()
    print("Each subtable shows heights for fixed s, rows are n, columns are t")
    print()

    for s in s_values[:5]:  # s = 2, 3, 4, 5, 6
        relevant_t = [t for t in t_values if t >= s]

        print(f"\nFor s = {s}:")
        print(f"{'n':>4}", end="")
        for t in relevant_t:
            print(f" {f't={t}':>5}", end="")
        print("  | RowMax")
        print("-" * (4 + 6 * len(relevant_t) + 12))

        # Column maxima
        col_maxima = [0] * len(relevant_t)

        for n in n_values[:20]:  # Show up to n=21
            print(f"{n:4d}", end="")
            row_max = 0

            for i, t in enumerate(relevant_t):
                if (s, t) in max_height_data and n in max_height_data[(s, t)]:
                    h = max_height_data[(s, t)][n]
                    row_max = max(row_max, h)
                    col_maxima[i] = max(col_maxima[i], h)
                    print(f" {h:>5}", end="")
                else:
                    print(f" {'-':>5}", end="")

            print(f"  | {row_max:>6}")

        # Print column maxima
        print("-" * (4 + 6 * len(relevant_t) + 12))
        print(f"{'Max':>4}", end="")
        total_max = 0
        for col_max in col_maxima:
            total_max = max(total_max, col_max)
            print(f" {col_max:>5}", end="")
        print(f"  | {total_max:>6}")

    # Overall maximum across everything
    print("\n" + "=" * 120)
    print("OVERALL MAXIMUM COTREE HEIGHT ACROSS ALL (s, t, n):")

    global_max = 0
    global_max_info = None

    for (s, t), n_dict in max_height_data.items():
        for n, h in n_dict.items():
            if h > global_max:
                global_max = h
                global_max_info = (s, t, n)

    if global_max_info:
        s, t, n = global_max_info
        print(f"  Maximum height: {global_max}")
        print(f"  Occurs at: K_{{{s},{t}}}, n = {n}")
    else:
        print(f"  Maximum height: {global_max}")

    print("=" * 120)

    # PIVOT TABLES: s,t in rows, n in columns
    print(f"DEBUG: Generating pivot tables...", file=sys.stderr)
    print("\n" + "=" * 120)
    print("PIVOT TABLE: MAXIMUM COTREE HEIGHTS")
    print("=" * 120)
    print("Format: Each row is (s,t), columns are n values")
    print()

    # Header
    print(f"{'s':>3} {'t':>3}", end="")
    for n in n_values:
        print(f" {n:>3}", end="")
    print()
    print("-" * (6 + 4 * len(n_values)))

    # Data rows
    total_rows = len(data_by_st)
    for row_idx, (s, t) in enumerate(sorted(data_by_st.keys()), 1):
        if row_idx % 10 == 0:
            print(f"DEBUG:   Printing row {row_idx}/{total_rows}", file=sys.stderr)
        print(f"{s:>3} {t:>3}", end="")
        for n in n_values:
            if n in max_height_data[(s, t)]:
                print(f" {max_height_data[(s, t)][n]:>3}", end="")
            else:
                print(f" {'':>3}", end="")
        print()

    print("\n" + "=" * 120)
    print("PIVOT TABLE: MINIMUM COTREE HEIGHTS")
    print("=" * 120)
    print("Format: Each row is (s,t), columns are n values")
    print()

    # Header
    print(f"{'s':>3} {'t':>3}", end="")
    for n in n_values:
        print(f" {n:>3}", end="")
    print()
    print("-" * (6 + 4 * len(n_values)))

    # Data rows
    for (s, t) in sorted(data_by_st.keys()):
        print(f"{s:>3} {t:>3}", end="")
        for n in n_values:
            if n in min_height_data[(s, t)]:
                print(f" {min_height_data[(s, t)][n]:>3}", end="")
            else:
                print(f" {'':>3}", end="")
        print()

    print("=" * 120)

    # Generate CSV files for easy import
    print(f"DEBUG: Generating CSV files...", file=sys.stderr)
    print("\n\nGenerating CSV files...")

    # CSV for partition presence
    print(f"DEBUG:   Writing partition_presence.csv", file=sys.stderr)
    with open("partition_presence.csv", "w") as f:
        f.write("s,t,n,has_partition\n")
        for (s, t), data in data_by_st.items():
            for n_str, n_data in data.get("extremal_by_n", {}).items():
                n = int(n_str)
                if n >= s:
                    expected = tuple(sorted([s - 1, n - s + 1]))
                    has_it = has_partition(n_data["structures"], n, s, expected)
                    f.write(f"{s},{t},{n},{1 if has_it else 0}\n")

    print("  Created: partition_presence.csv")

    # CSV for heights (both min and max)
    with open("cotree_heights.csv", "w") as f:
        f.write("s,t,n,min_height,max_height\n")
        for (s, t) in sorted(data_by_st.keys()):
            all_n = set(max_height_data[(s, t)].keys()) | set(min_height_data[(s, t)].keys())
            for n in sorted(all_n):
                min_h = min_height_data[(s, t)].get(n, 0)
                max_h = max_height_data[(s, t)].get(n, 0)
                f.write(f"{s},{t},{n},{min_h},{max_h}\n")

    print("  Created: cotree_heights.csv")

    # Pivot table format for MAX heights
    with open("cotree_max_heights_pivot.csv", "w") as f:
        # Header
        f.write("s,t")
        for n in n_values:
            f.write(f",n={n}")
        f.write("\n")

        # Data rows
        for (s, t) in sorted(data_by_st.keys()):
            f.write(f"{s},{t}")
            for n in n_values:
                if n in max_height_data[(s, t)]:
                    f.write(f",{max_height_data[(s, t)][n]}")
                else:
                    f.write(",")
            f.write("\n")

    print("  Created: cotree_max_heights_pivot.csv")

    # Pivot table format for MIN heights
    with open("cotree_min_heights_pivot.csv", "w") as f:
        # Header
        f.write("s,t")
        for n in n_values:
            f.write(f",n={n}")
        f.write("\n")

        # Data rows
        for (s, t) in sorted(data_by_st.keys()):
            f.write(f"{s},{t}")
            for n in n_values:
                if n in min_height_data[(s, t)]:
                    f.write(f",{min_height_data[(s, t)][n]}")
                else:
                    f.write(",")
            f.write("\n")

    print("  Created: cotree_min_heights_pivot.csv")
    print()


if __name__ == "__main__":
    main()
